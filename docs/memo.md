# Technical memo

A policy assistant is a retrieval problem pretending to be a generation problem.
The model is good at reading a question and finding the relevant paragraph, and
measurably bad at the one step that decides the answer: comparing a number
against a threshold. Most of what follows comes from taking that seriously.

Everything quantified here was measured against the live API, not estimated.

---

## 1. Why this model, and what the measurements said

I started on **Claude Haiku 4.5** for cost, and the first real call produced a
wrong answer to an easy question. Asked about a lost receipt for a 300 EUR
dinner, it said *"since your amount is under 250 EUR, you don't need manager
approval"*. Section 3.2 says approval is needed **above** 250 EUR. It cited the
right section, passed the schema, and reported `confidence: "high"`.

Rather than guess, I built a small eval: 12 policy questions with
known-correct answers (9 of them threshold comparisons), 5 repetitions, graded
on the structured fields deterministically and on the prose by a Claude Opus 5
judge against a required conclusion written from the policy. 240 answer calls.

| Config | Correct | q5 — needs arithmetic | Out of scope | p50 latency | $/call |
|---|---|---|---|---|---|
| Haiku 4.5 | 53/60 (88%) | **1/5** | 5/5 | 2.8 s | $0.0038 |
| Haiku + prompt rule | 55/60 (92%) | **0/5** | 5/5 | ~2.8 s | $0.0039 |
| Haiku + extended thinking | 56/60 (93%) | **5/5** | **1/5** | 5.0 s | $0.0059 |
| Sonnet 5 | 55/60 (92%) | 3/5 | 5/5 | 3.3 s | $0.0035 |

Four findings, in order of how much they changed the design.

**The weakness is arithmetic-before-comparison, not comprehension.** Nine of the
twelve questions scored 5/5 everywhere. The failures cluster on questions
needing a calculation *before* the comparison — q5 is "team dinner, 6 people,
300 EUR" against a 40 EUR per-person cap. Haiku got it right once in five. It
read the policy correctly and then fumbled the maths.

**Telling the model to be careful did nothing.** I added an explicit rule to
state the comparison before concluding. Overall moved 88% → 92%, inside the
noise, and q5 went **1/5 → 0/5**. The model dutifully computed "50 EUR per
person" and still concluded "yes, within policy". Reasoning a model can't do
isn't fixed by instructing it to do it. This is the result that pushed me to
solve the problem outside the model.

**Extended thinking fixes the arithmetic and breaks other things.** q5 went to
5/5, but out-of-scope handling collapsed (5/5 → 1/5), p50 latency nearly
doubled, and `tool_choice` cannot be forced while thinking is enabled — which
costs the schema guarantee the brief asks for. Not worth it here.

**Sonnet 5 costs the same as Haiku, and is still unusable.** Haiku's minimum
cacheable prefix is 4096 tokens and our prompt is around 3,000 (it has since
grown to ~3,400), so prompt caching is silently dead — no error,
`cache_creation_input_tokens: 0` on every call, ever. Sonnet's minimum
is 1024, so it caches: 237 fresh input tokens versus Haiku's 2,958, which makes
Sonnet *cheaper* ($0.0035 vs $0.0038) for 0.5 s more latency.

So I switched to Sonnet 5 — and the first call came back:

```
"answer": "...the per-person meal limit is the issue.</answer>
           <parameter name=\"sources\">[\"4.2\"]",
"sources": []
```

Sonnet wrote text-style tool-call syntax **into the answer string**. Measured:
7 of 16 calls on these questions. Worse, `strict: true` didn't fix it, it
*masked* it — constrained decoding forced the malformed output into a
schema-valid shape with `sources: []`, and an empty source list trivially passes
the citation check. Corrupted output that defeats validation is worse than a
known-wrong answer that doesn't, so I reverted.

**Decision: Claude Haiku 4.5**, with the arithmetic removed from the model's
hands (§2). Haiku never leaked tool-call syntax — 0 of 16 in the direct
comparison above — and returned valid structured output on all 120 of its eval
calls. Its one measured weakness is now handled in Python, and it
is the cheapest option. **If I were choosing for production I would re-run this
eval before trusting it** — n=5 per question means the 88%/92%/93% column is not
significant, and only the q5 column (1/5 vs 5/5) carries real signal.

Cost of the whole exercise, incidentally: about $2.60 of API spend.

---

## 2. Key design decisions

### The schema is enforced by the API, not requested politely

A single tool, forced via `tool_choice`, with **`strict: true`**. Without it the
API treats `required` as advice: Sonnet omitted the required `sources` field in
3 of 16 calls. With it, the same cells went to 8/8. `additionalProperties:
false` is set in the Anthropic client rather than on the shared model, because
it is a provider requirement and the model has to stay provider-agnostic.

Strict mode also rejects `minimum` on an integer schema, which is why
`headcount`'s lower bound is a validator rather than a `Field(ge=1)`.

### Three checks, because the schema only proves shape

1. **Re-validate** the tool input against the Pydantic model, which also enforces
   the cross-field rules JSON Schema can't express (out-of-scope answers must
   cite nothing and must escalate).
2. **Check every citation exists.** A model can cite section 9.9 as easily as
   3.2, and an invented citation is what a fabricated answer looks like.
3. **Recompute the arithmetic.**

Note what check 2 does *not* prove: the very first failure cited section 3.2
correctly and drew the opposite conclusion from it. Citation validation proves a
section exists, never that the answer follows from it.

### The decisive sentence is written by Python

The model returns its working — `amount_eur`, `headcount`, `per_person`,
`limit_applied`, `verdict` — and the service recomputes all of it. That alone
wasn't enough. On a later call every field was *correct* (`verdict:
above_threshold`, `per_person: 50`, `limit: 40`) while the prose read **"Yes,
that is within policy."** The structured output was right and the sentence the
employee reads was wrong.

So the service composes the opening sentence itself from the verified numbers
and the model writes only what follows. It states the comparison and never the
permission:

> 50 EUR per person is above the 40 EUR per person limit (section 4.2).

A yes/no opener would have introduced a new bug: a 30 EUR gift card is *within*
the 50 EUR gift limit and still forbidden outright.

### Caps and triggers are different numbers

Modelling every threshold as a cap produced a false rejection. Asked about the
lost 300 EUR receipt, the model declined to call 250 EUR a "limit" — correctly,
because crossing it means needing an approval, not breaking a rule. Each entry
in `app/limits.py` now carries that distinction, so triggers read as
"above the 250 EUR threshold" rather than implying a violation.

### Retries: only what a retry can fix

Retried with exponential backoff and jitter: `llm_unavailable` and `llm_timeout`.
Both mean the provider produced nothing, so **the retry is also free**.

Not retried: `llm_invalid_output`, because that call was billed and paying twice
should be deliberate rather than a default; and `llm_auth_error`, which fails
identically every time. The brief invites a justification here — mine is that
retrying a malformed answer is a cost decision, so it belongs to whoever pays
the bill, and the error is explicitly marked `retryable: true` so a caller can
make it. Four transient connection failures occurred during development, every
one of which a single retry absorbed.

`REQUEST_TIMEOUT_SECONDS` caps the whole request including retries: 4 attempts
at up to 20 s each could otherwise run for 80 s. The caller waits a bounded time
and gets `request_timeout`.

### Errors are separated by layer

Every failure carries `type`, `layer` (`http` / `llm` / `service`), `message`,
`retryable` and `request_id`. The layer is the operationally useful part: a
spike in `llm` means page the provider's status page, a spike in `http` means
someone's client is broken, and `service` means it's us. Provider 4xx other than
auth map to `internal_error`, because a 400 means *we* built a bad request and
no caller retry will help.

---

## 3. Streaming and structured output

These pull against each other. Structured output is only trustworthy once it is
complete; streaming means emitting before it is. Our composed sentence made it
sharper still: the sentence must come **first**, but it is computed from numbers
the model was generating **last**, after the prose.

**The resolution is to make field order load-bearing.** `PolicyAnswer` declares
the working first and the prose last, so the numbers are final while the answer
is still being written. The moment `answer` appears in a snapshot, every figure
the sentence needs is complete: compute it, send it as the first delta, stream
the prose behind it. Asking the model to state its arithmetic before committing
to an answer is also what extended thinking did for accuracy, so the ordering
may pay twice.

Two things this cost, both worth stating plainly:

**Deltas are provisional.** The checks need the whole answer, so text can reach
the client and then be contradicted by an `error` event. The contract is
explicit about it, and the demo page replaces the streamed text with the
validated answer when `result` arrives. A stricter alternative — buffer,
validate, then replay the text as fake deltas — gives up all the latency benefit
to preserve an appearance of streaming, which seemed the worse trade.

**`eager_input_streaming` weakens `strict: true`.** Without it the API buffers
tool input and nothing streams; with it, the API stops validating that input. Our
three checks already treat the output as untrusted, which is what makes this
survivable — but the guarantee genuinely is weaker on the streaming path than
the buffered one.

One implementation trap worth passing on: the SDK's parsed snapshot **omits an
unterminated string**, so `answer` stayed invisible until complete and the first
version delivered 287 characters in a 31 ms burst — an endpoint that satisfied
the requirement without streaming anything. Parsing the accumulated JSON with
jiter's `trailing-strings` mode fixed it: 39 deltas over a second. I only caught
it because I timed the deltas rather than eyeballing the output.

Retries stop at the first byte. Before it, a transient failure is retried and
the client sees nothing; after it, the stream is committed and the failure
becomes an `error` event. For the same reason the deadline is checked between
events rather than with `asyncio.timeout`, whose cancel scope cannot safely span
an async generator's yields.

---

## 4. Known limitations, and what changes at production scale

### Where it breaks today

**The retrieval is a stub.** The whole policy (~2,900 tokens) goes into every
prompt. That is right for one document and wrong immediately after: at 50
documents it stops fitting, gets expensive, and dilutes attention. Real version
chunks by section, embeds, retrieves the top handful, and — importantly — keeps
citations checkable against the chunk ids actually retrieved, which turns the
citation check from "does this section exist" into "was this section even in
context".

**Extraction is unverified.** Python recomputes `300 / 6`, but the model still
had to read "6 people" and "300 EUR" out of free text. A misread headcount
produces a confidently wrong answer with perfect internal arithmetic. This is
the largest remaining hole.

**The limits table is hand-maintained.** `app/limits.py` duplicates numbers that
also live in the markdown. A test asserts every limit cites a real section, but
nothing catches the policy changing 40 EUR to 45 EUR while the table doesn't.
Either parse the numbers out of the policy, or make the policy the derived
artefact.

**The model still oversteps.** It occasionally offers approval routes the policy
doesn't grant ("ask your manager") when the relevant rule only requires that
above 500 EUR. Rule 5 of the prompt forbids it; the prompt doesn't enforce it.

**No conversation.** Every request is independent. "What about for 8 people?"
doesn't work.

**Prompt injection is addressed but not solved.** The question is tag-wrapped
and the system prompt says to ignore instructions inside it. That is a
speed bump, not a defence. What actually limits the damage is that the output
schema is enforced and the citations are checked, so a successful injection
still has to produce a schema-valid answer citing real sections.

### Latency

p50 is ~2.8 s non-streaming; the streaming path puts the decisive sentence in
front of the user at roughly the same point but the perceived wait is shorter
because text then arrives continuously. Prompt caching would cut input
processing but is inert on Haiku at this prompt size (§1) — it becomes free
value the moment the policy grows past ~4K tokens, and the `cache_control`
marker is already in place for that. At scale I would also cache identical
questions outright; policy Q&A has a very heavy head.

### Cost

About $0.005 per answer at present — roughly 3,400 input tokens and 270 output,
dominated by the policy being resent in full every time. (The eval table above
reports $0.0038, measured before the working fields were added to the schema;
the extra ~450 tokens of prompt and output are what the arithmetic checking
costs.) Retrieval plus working caching is the big lever, worth roughly an order
of magnitude. Rejected answers are billed and currently discarded, which is the
argument for making the invalid-output retry decision explicitly rather than by
default.

### Observability

Today: structured logs with a request id, per-request latency and status, and
per-call token counts including cache reads. That is enough to debug one bad
answer, and not enough to run the thing. Missing, roughly in order of value:

1. **The rejection rate, by check.** How often validation, citations and
   arithmetic each fail is the single best early warning that a model or prompt
   change has regressed something — and it is already computed, just not counted.
2. Latency and cost percentiles per route, and retry/timeout rates.
3. Sampled full traces — prompt, raw tool input, verdict — behind a retention
   policy, since these contain employee questions.
4. The eval in CI, so a prompt or model change has to clear a known bar. This
   matters more than the metrics: the failure mode of an LLM feature is silent
   quality decay, which monitoring notices late and an eval notices immediately.

### Failure modes worth naming

- **Provider down or rate limited** — retried, bounded, surfaced as a retryable
  503/504. No circuit breaker: under sustained failure every request still burns
  its full retry budget before failing.
- **Malformed or incomplete output** — caught by the schema, and by the
  arithmetic check when the numbers disagree. Surfaced as a 502.
- **Confidently wrong prose** — partially handled, by writing the decisive
  sentence ourselves. Unhandled wherever an answer needs no arithmetic.
- **Routing errors** (404/405) currently return FastAPI's default body rather
  than our error shape. Small, known, worth fixing for contract consistency.
- **Single process, no backpressure.** One slow provider ties up a worker for up
  to 45 s; there is no queue, no shedding and no per-caller rate limit.

### If this were going to production

In priority order: retrieval with checkable chunk ids; the eval in CI with the
rejection-rate metrics above; verified extraction of amounts and headcounts;
then caching and a circuit breaker. The ordering is deliberate — the first two
are about knowing whether the thing works, and everything else is about making
a thing that works cheaper.
