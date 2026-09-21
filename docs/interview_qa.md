# Interview Q&A

Twenty-five questions a senior engineer would ask about this code, with the
answer to give. Grounded in what the code actually does — every claim here is
checkable against the repo.

The last six are the ones designed to expose weaknesses in the design. The answer
to those is never a defence. It is "yes, here is exactly where it breaks, here is
why I shipped it anyway, here is what I would do about it".

**One rule for the whole conversation:** if you do not know, say so and say what
you would measure. This repo's strongest asset is that its claims are measured.
Do not spend that credibility improvising.

---

## Architecture and the request path

### 1. Walk me through what happens when a request hits `POST /analyze`.

Middleware assigns a request id and starts a timer. FastAPI validates the body
into `AnalyzeRequest` — empty query is a 422, over-length is a 413. The route
opens a 45-second deadline with `asyncio.timeout`. Inside it, a retry loop calls
Anthropic with a forced tool call, four attempts maximum, retrying only transient
failures. The tool input comes back and goes through three checks: re-validate
against the Pydantic model including the cross-field rules, verify every cited
section exists in the policy, and recompute the arithmetic. If all three pass,
Python composes the opening sentence from the verified numbers and prepends it to
the model's prose. That goes out as JSON with the request id in both the body and
the `X-Request-ID` header.

The one-line version: the model reads and writes, Python decides.

### 2. Why tool use rather than a JSON-mode prompt and a parser?

Because "please return JSON" is a request and a forced tool call is a
constraint. `tool_choice: {"type": "tool", "name": ...}` removes the free-text
path entirely — there is no branch where the model returns prose and I have to
fish JSON out of a markdown fence.

I also set `strict: true`, and that one I measured. Without it the API treats
`required` as advice: Sonnet omitted the required `sources` field in 3 of 16
calls. With `strict` on, the same cells went 8 of 8. `additionalProperties:
false` goes with it, and it is set in the Anthropic client rather than on the
shared Pydantic model, because it is a provider requirement and the model has to
stay provider-neutral.

One consequence worth knowing: strict mode rejects `minimum` on an integer
schema, which is why `headcount`'s lower bound is a Pydantic validator rather
than `Field(ge=1)`.

### 3. Why Anthropic, and why Haiku 4.5?

Be honest about the two halves of that question, because the repo answers them
very unevenly.

**Why Haiku** is measured. I built a twelve-question eval with hand-written
ground truth, nine of them numeric comparisons, five repetitions, four configs —
240 answer calls. Haiku 4.5 returned valid structured output on all 120 of its
calls and never leaked tool-call syntax. Sonnet 5 was cheaper per call, because
Haiku's minimum cacheable prefix is 4096 tokens and my prompt is ~3,400, so
prompt caching is silently inert on Haiku — but Sonnet wrote text-style
tool-call syntax *into the answer string* in 7 of 16 calls, and `strict: true`
did not fix that, it masked it: constrained decoding forced the malformed output
into a schema-valid shape with `sources: []`, which trivially passes the citation
check. Corrupted output that defeats validation is worse than a wrong answer that
does not, so I reverted to Haiku and moved the arithmetic out of the model.

**Why Anthropic rather than Gemini** — the brief lists both, and I did not
measure it. My reasons are tool-use and strict-structured-output maturity, and
the partial-tool-input streaming behaviour the streaming design depends on, which
I could not assume elsewhere. But I want to be straight with you: that is a
judgement call, not a measurement. The eval harness is the thing that would
settle it, and running it against Gemini is about a day of work and a few dollars.

### 4. How did you reconcile streaming with structured output?

They pull against each other: structured output is only trustworthy once
complete, and streaming means emitting before it is. My design made it worse,
because the first thing the user must see is a sentence computed from numbers the
model would naturally generate *last*, after its prose.

The resolution is to make field order load-bearing. `PolicyAnswer` declares the
nine numeric and flag fields first and `answer` last. The model generates
properties in schema order, so **the moment the key `answer` appears in a
snapshot, every number the sentence needs is already final.** Compute the
sentence, send it as the first delta, stream the prose behind it.

There is a nice side effect: forcing the model to state its arithmetic before
writing its conclusion is the same thing extended thinking did for accuracy in my
eval. The ordering may pay twice.

### 5. What actually goes over the wire in the SSE stream?

Four event types. `meta` once, first, with the request id and model — sent before
the provider client is even constructed, so the caller gets a first byte
immediately. Then zero or more `delta` events carrying `{"text": ...}`. Then
either `result` with the complete validated `AnalyzeResponse`, or `error` with
the exact same error body the non-streaming path would have returned.

Deltas are explicitly provisional and the contract says so, because the checks
need the whole answer. The demo page renders deltas as they arrive and then
replaces that text with the validated `result.answer`.

One thing I would flag: a bad request body on `?stream=true` is still a normal
JSON 422, not an SSE error event — validation happens before the generator is
consumed. There is a test for that, because it is the kind of thing that quietly
becomes inconsistent.

---

## Failure handling

### 6. What happens if the model returns malformed or incomplete JSON?

It is caught in three different places and always becomes one error type:
`llm_invalid_output`, a 502 in the `llm` layer, marked `retryable: true`.

In the adapter: `stop_reason == "max_tokens"` means the answer was truncated
mid-generation; no tool-use block with my tool name; a tool input that is not an
object. In the service: Pydantic validation fails, including the cross-field
rules JSON Schema cannot express; a cited section does not exist in the policy;
the arithmetic does not check out.

On the streaming path the same failure becomes an `error` event, which can and
does arrive after visible text. That is the honest cost of streaming provisional
output, and it is in the contract rather than hidden.

### 7. The brief said "retry, fall back, or surface an error — your call". You surface. Why not retry?

Because that call already produced billable output. A transient failure produced
nothing, so retrying it is free; a malformed answer was paid for, and retrying it
pays twice. Paying twice should be a deliberate decision by whoever owns the
budget, not a library default I buried in a config file.

So I mark it `retryable: true` and let the caller decide. The information is in
the response; the policy is not mine to make.

If I owned the budget I would probably retry it exactly once, and I would want
the rejection rate by check — which is already computed and just not counted —
before choosing a number.

### 8. What is the retry policy, and why those numbers?

Four attempts — `1 + MAX_RETRIES`, with `MAX_RETRIES=3`. Exponential backoff with
jitter, 0.5 s initial, 8 s cap, via tenacity. Retried: `llm_unavailable` (which
covers connection failures, 429, and any 5xx including Anthropic's 529
overloaded) and `llm_timeout`. Not retried: `llm_invalid_output`, for the reason
above, and `llm_auth_error`, which fails identically every time and would only
burn the deadline.

The SDK's own retries are turned off with `max_retries=0`, deliberately — I want
exactly one retry policy in the system, in one place, so total latency against
the deadline is reasonable about.

**Now the honest part, and I would rather say it than have you find it:** those
numbers do not fit together. Four attempts at a 20-second per-call timeout plus
backoff is about 83 seconds against a 45-second deadline. Attempts 3 and 4 can
never complete. Nothing breaks — the deadline does its job and the caller gets a
bounded `request_timeout` — but the config claims a retry budget it cannot spend.
The right fix is to bring `LLM_TIMEOUT_SECONDS` down to 8 or 10, since p50 is
2.8 seconds and 20 is enormously generous, or to derive the attempt count from
the deadline rather than setting the two independently.

### 9. Describe the timeout layering.

Two levels, and they answer different questions.

`LLM_TIMEOUT_SECONDS` (20) is on the SDK client and bounds **one call to the
provider**. Blowing it raises `APITimeoutError`, which I map to `llm_timeout`, a
504 in the `llm` layer, and it is retryable.

`REQUEST_TIMEOUT_SECONDS` (45) bounds **the whole request including every retry
and every backoff**. On the buffered path that is `asyncio.timeout` around the
entire orchestration. Blowing it is `request_timeout`, a 504 in the `service`
layer.

The outer one is what makes "the service must not hang" true regardless of what
the retry loop does. The caller's worst case is 45 seconds, always.

### 10. Why is the streaming deadline a manual check instead of `asyncio.timeout`?

Because a timeout cancel scope cannot safely span the yields of an async
generator. The generator suspends at `yield` and its consumer resumes in a
different context, so the scope is entered and exited across contexts and
cancellation lands in the wrong place.

So the streaming path computes `time.monotonic() + REQUEST_TIMEOUT_SECONDS` once
and checks it at the top of each event. Slightly coarser — the deadline fires on
the next event rather than exactly on time — but correct, and each event is
milliseconds apart in practice. A provider that goes completely silent is caught
by the SDK's read timeout instead.

### 11. Why do you stop retrying once a delta has been sent?

Because a retry is only transparent while nothing has reached the client. Before
the first byte, a transient failure is retried and the client never knows it
happened. After the first delta, the stream is committed — I cannot un-send text,
and a fresh attempt would produce a different answer that does not continue the
one already on screen. So after the first delta, the same transient failure
becomes an `error` event instead.

Both halves are tested: `test_transient_failure_before_any_delta_is_retried` and
`test_transient_failure_after_a_delta_is_not_retried`.

### 12. How are HTTP-layer and LLM-layer errors separated?

Every error carries a `layer` field, which is a `Literal["http", "llm",
"service"]` — three, not two. `http` means the caller sent something I cannot
accept. `llm` means the provider failed me. `service` means my own orchestration
failed.

The layer is the operationally useful part: a spike in `llm` means check the
provider's status page, a spike in `http` means someone's client is broken, and
`service` means it is us. A bare 500 tells you none of that.

Two details I would point at. First, provider 4xx other than auth map to
`internal_error`, not to an `llm` error — a 400 from Anthropic means *I* built a
bad request, so it is my bug and no caller retry helps. Second, I intercept
Starlette's own `HTTPException` so a 404 answers in my error shape instead of
`{"detail": "Not Found"}`, while keeping headers the status depends on, like
`Allow` on a 405. There is exactly one error format in the whole surface.

---

## Design decisions

### 13. Why does Python write the first sentence instead of the model?

Because I measured the model getting it wrong in a way validation could not
catch. On one call every structured field was correct — `verdict:
above_threshold`, `per_person: 50`, `limit_applied: 40` — and the prose read
*"Yes, that is within policy."* The JSON was right and the sentence the employee
reads was wrong.

So the service composes the opening sentence from the verified numbers and
prompt rule 11 tells the model not to state the comparison at all. The decisive
numeric claim is computed, not generated.

The restraint matters as much as the sentence: it states **only the comparison,
never the permission**. A 30 EUR gift card is within the 50 EUR gift limit and
still forbidden outright, so a yes/no opener would have introduced a new bug in
the name of fixing one.

### 14. Before that, you tried prompting your way out of it. What happened?

I added an explicit rule telling the model to find the threshold, state whether
the amount was above or below it, and only then conclude. Overall accuracy moved
88% to 92%, which is inside the noise at n=5. And the hardest question — team
dinner, six people, 300 EUR against a 40 EUR per-person cap — went from 1 out of
5 to **0 out of 5**. The model dutifully computed "50 EUR per person" and still
concluded "yes, within policy".

That is the result that decided the architecture. Reasoning a model cannot do is
not fixed by instructing it to do it. I kept that config in the eval precisely
because it is the most useful negative result in the exercise.

### 15. What is the difference between a cap and a trigger, and why does the code care?

A cap is a number you must stay under — hotels at 180 EUR a night. A trigger is a
number that, once crossed, requires something extra — a missing receipt above
250 EUR also needs the manager's approval. Crossing a trigger is **not** a policy
breach.

I found this because modelling everything as a cap produced a false rejection.
Asked about a lost 300 EUR receipt, the model declined to call 250 EUR a "limit",
and it was right to. So every entry in `limits.py` carries `kind`, and the
generated sentence reads "above the 250 EUR *threshold*" for a trigger and "above
the 250 EUR *limit*" for a cap.

It is a small thing that matters a lot for trust: the sentence is the one part of
the answer I guarantee, so it cannot imply a violation that did not happen.

### 16. Your citation check proves a section exists. What does it not prove?

That the answer follows from it. My very first failure in development cited
section 3.2 perfectly correctly and drew the exact opposite conclusion from it,
with `confidence: "high"`.

Existence checking catches fabrication — a model will write "9.9" as readily as
"3.2", and an invented citation is what a hallucinated answer looks like. It
catches nothing about whether the cited text supports the claim. That is why the
arithmetic check exists on top of it, and why the one sentence that carries the
decision is written in Python rather than checked after the fact.

At scale, retrieval improves this: if you retrieve chunks and then only allow
citations to chunk ids that were actually in context, the check upgrades from
"does this section exist" to "was this section even in front of the model".

### 17. How do you handle prompt injection?

The question is wrapped in `<question>` tags and the system prompt tells the
model to treat anything inside as a question, never as instructions. I would call
that a speed bump, not a defence — I would not claim otherwise.

What actually limits the blast radius is the architecture. The output schema is
enforced, the citations are checked against a fixed policy document, and the
numeric claim is computed in Python. A successful injection still has to produce
a schema-valid answer citing sections that exist, with arithmetic that agrees
with itself. It cannot make the service say "approved" in the decisive sentence,
because the model does not write that sentence.

The remaining exposure is the free prose, where an injection could put arbitrary
text in front of an employee. In production I would add an output check for
anything resembling an instruction or a URL, and I would not put this in front of
external users at all.

---

## Production

### 18. What breaks first under load?

Worker starvation, before anything else. One slow provider call holds a worker
for up to 45 seconds, and there is no queue, no shedding, and no per-caller rate
limit. Concurrency is bounded by the worker count and by Anthropic's rate limit,
and when I hit the rate limit every request spends its full retry budget before
failing, which makes the outage worse rather than better — there is no circuit
breaker.

Second thing to break: the connection pool, because the streaming path does not
explicitly close the provider's async generator when it exits early. It gets
finalised by the event loop's async-gen hooks rather than deterministically. A
`contextlib.aclosing` fixes it and I would do that before anything else on this
list.

Third: nothing authenticates. Anyone who can reach the port can spend the API
budget. For an internal Finance tool that is the first thing I would put in front
of it.

The fix order is per-caller rate limiting plus a concurrency cap, then a circuit
breaker that fails fast when the provider is down, then auth.

### 19. What does this cost and what is the lever?

About half a cent per answer today — roughly 3,400 input tokens and 270 output —
and it is dominated by resending the entire policy in every prompt. The whole
exercise, eval included, cost about $2.60 of API spend.

The lever is retrieval, and it is worth roughly an order of magnitude. Today the
whole policy goes into every request, which is correct for one document and wrong
immediately after: at fifty documents it stops fitting, gets expensive, and
dilutes attention. Chunk by section, embed, retrieve the top handful.

Second lever is caching, and there is a wrinkle worth knowing: prompt caching is
silently inert on Haiku at this prompt size, because Haiku's minimum cacheable
prefix is 4096 tokens and my prompt is ~3,400. No error, `cache_creation_input_tokens: 0`
on every call, forever. The `cache_control` marker is already in place, so it
becomes free value the moment the policy grows past 4K. Beyond that I would cache
identical questions outright — policy Q&A has a very heavy head, and the top
twenty questions are probably half the traffic.

Third: rejected answers are billed and currently discarded. That is the concrete
argument for making the invalid-output retry decision explicitly rather than by
default.

### 20. What is the latency story?

p50 is about 2.8 seconds non-streaming. The streaming path puts the decisive
sentence in front of the user at roughly the same moment, but the perceived wait
is much shorter because text then arrives continuously — 39 deltas over about a
second in the measurement I took.

At scale, retrieval cuts input processing, caching cuts it further once the
prompt clears 4K tokens, and a question cache removes the call entirely for the
head of the distribution. I would not chase a faster model: I already measured
that the cheap fast model's weakness is arithmetic, and I moved arithmetic out of
the model rather than paying for a bigger one.

### 21. What is your observability story?

Today: structured logs with a request id on every line, per-request latency and
status, and per-call token counts including cache reads and stop reason — logged
whether or not the answer survives the checks, because a rejected answer is still
billed. The request id is in the response body and the `X-Request-ID` header, so
a user reporting a bad answer hands you one string that finds everything.

That is enough to debug one bad answer and not enough to run the thing. Missing,
in order of value:

1. **The rejection rate broken down by check** — how often validation, citations
   and arithmetic each fail. It is the single best early warning that a model or
   prompt change has regressed something, and it is already computed, just not
   counted.
2. Latency and cost percentiles per route, retry rate, timeout rate.
3. Sampled full traces — prompt, raw tool input, verdict — behind a retention
   policy, since these contain employee questions. Note that I deliberately do
   not log query text today; that is a privacy choice with a debugging cost, and
   sampling behind a policy is how I would buy the debugging back.
4. **The eval in CI.** This matters more than the metrics. The failure mode of an
   LLM feature is silent quality decay, which monitoring notices late and an eval
   notices immediately.

### 22. How would you swap the LLM provider?

Structurally it is a new module and one branch. `app/llm/base.py` defines the
interface — `get_structured_output` and `stream_structured_output`, plus an
`LLMResult` that carries data and usage — and `anthropic_client.py` is the only
file in the repo that imports a provider SDK. The factory picks the
implementation from `LLM_PROVIDER`. Nothing above that layer changes.

Two places I would not oversell it. First, I pass
`PolicyAnswer.model_json_schema()` through the interface, and each provider's
structured-output dialect supports a different subset of JSON Schema — Gemini's
`responseSchema` is not the same surface, and `strict` has no exact equivalent
everywhere. Second, and more important: the streaming design depends on the
provider emitting *partial tool input*, which is an Anthropic capability I
enabled explicitly. A provider that only hands over the complete tool call would
keep the interface intact and quietly give me a non-streaming endpoint.

So: the interface is provider-agnostic, the streaming guarantee is not. Swapping
providers is a day's work plus a re-run of the eval, and I would not trust the
swap without the eval.

### 23. What would you do first if this were going to production?

In this order: retrieval with checkable chunk ids; the eval in CI with the
rejection-rate metrics; verified extraction of amounts and headcounts; then
caching and a circuit breaker.

The ordering is the point. The first two are about *knowing whether the thing
works*. Everything after is about making a thing that works cheaper. Doing those
in the other order is how you end up with a fast, cheap, confidently wrong
service.

---

## The hard ones

### 24. All three of your checks pass and the answer is still wrong. Show me how.

Two ways, and both are live.

**The model picks the wrong limit.** A team meal compared against the 80 EUR
client-meal cap instead of the 40 EUR team-meal cap. `limit_applied: 80` is a
real number from a real section, the division is right, the verdict follows from
the comparison, the citation exists. Every check passes and the answer is
confidently, plausibly wrong — and the sentence I generate, the one part I
guarantee, states it.

**The model misreads the inputs.** Python recomputes 300 ÷ 6, but the model still
had to read "six people" and "300 EUR" out of free text. A misread headcount
produces perfect internal arithmetic over wrong numbers.

They are the same hole: **I verify the arithmetic, and the model still chooses
which numbers enter it.** That choice is unchecked, and it is the largest
remaining weakness in the design.

What I would do: extract amounts and headcounts with a deterministic pass and
diff it against the model's; have the model name the *rule* it is applying, not
just the number, and check the rule-to-limit mapping in code; and when they
disagree, escalate rather than answer. That turns an unverified choice into a
verified one, which is the same move I already made for the comparison.

### 25. You say the schema is enforced. But on the streaming path you turned the API's validation off. So is it enforced or not?

Weaker on the streaming path, and I would not claim otherwise.
`eager_input_streaming` is what makes partial tool input available at all —
without it the API buffers the whole tool call and there is literally nothing to
stream — and the cost is that the API stops validating that input as it goes. So
on the streaming path `strict: true` is doing less work than on the buffered one.

What makes it survivable is that my three checks already treat the final output
as untrusted: the same `_build_response` runs on both paths, re-validating
against Pydantic, checking citations, and recomputing the arithmetic. Nothing
reaches the `result` event that would not have reached a 200 on the buffered
path.

The genuinely weaker part is the *deltas*, not the result. Text can reach the
screen and then be contradicted by an `error`. I chose that over the alternative
— buffer, validate, then replay the text as fake deltas — because that gives up
the entire latency benefit to preserve the appearance of streaming, which seemed
the worse trade. But it is a trade, and a stricter product would take the other
side of it.

### 26. Your test suite never touches the file that talks to Anthropic. How do you know the error mapping is right?

I do not, to the standard I would want. That is the biggest coverage gap in the
repo and it is a fair hit.

Every test fakes the client one level above the adapter, so the suite proves the
*service* handles a timeout correctly and proves nothing about whether an
`APITimeoutError` actually becomes one. `_as_app_error`, `_map_status_error`,
`_to_result`, and the jiter partial parsing are all untested — and they contain
the subtlest code in the repo, including an ordering dependency where
`APITimeoutError` must be checked before `APIConnectionError` because it
subclasses it. The source is right. Nothing proves it stays right.

These are pure functions taking constructed SDK exceptions, so it is maybe an
hour of work. I would write it before I would write anything else, because that
mapping is the thing that decides whether a production incident retries or fails
fast.

What I did verify by hand: four transient connection failures happened during
development and a single retry absorbed every one of them. That is evidence, not
a test.

### 27. You stream text and then tell me it might be wrong. Isn't that worse than not streaming?

It is a genuine trade and I can argue either side.

The case for it: the first delta is the *computed* sentence, not model prose. It
comes from numbers Python has already read, and the arithmetic check at the end
recomputes from the same numbers — so the first thing the user sees is the part
least likely to be retracted. What follows is explanatory prose, where a
retraction means "the explanation was withdrawn", not "the verdict flipped".

The case against: a user who reads a sentence and looks away has read something I
later refused to stand behind, and no contract note fixes that. For a Finance
tool, where the whole product is trust, that is a real argument.

What I built: the contract states it, and the demo page replaces streamed text
with the validated answer on `result`. What I would build for production: stream
the verified numbers as *structured* events — a numbers panel that fills in — and
only stream prose once the checks have passed. You keep the perceived latency,
and nothing provisional is ever rendered as a statement. That is where I would
take it, and I would not defend the current version as the end state.

### 28. Your eval says 88% versus 92% versus 93%. Which model is best?

The table does not answer that and I would not let it. At five repetitions each
cell is n=5 and each config total is n=60 — enough to see a question fail four
times out of five, nowhere near enough to call 88% different from 92%. The
headline percentages are noise.

The column that carries signal is q5: 1 out of 5 for the shipped config versus 5
out of 5 with extended thinking. That is a real effect on a specific,
identifiable weakness, and it is what drove the design.

Two more caveats I would state before you ask. The judge is a Claude Opus 5 model
grading against hand-written ground truth, and **it has never been validated
against human labels** — I would spot-read its stated reasons before trusting a
close result. And extended thinking, which fixes the arithmetic, breaks
out-of-scope handling badly (5 of 5 down to 1 of 5), nearly doubles latency, and
cannot be combined with forced `tool_choice` — which would cost the schema
guarantee the brief explicitly asks for. So the config that scores best on the
headline number is the one I rejected.

If I were choosing a model for production rather than for an exercise, I would
re-run this at n=30 with confidence intervals before trusting any of it.

### 29. Finance asks you: can I trust this thing? What do you say?

For a question with a number in it, mostly yes, and I can tell you exactly why:
the comparison is computed in Python, the number it compares against must be one
that appears in the policy, and if the model's own figures disagree with each
other the answer is refused rather than shown. The sentence you read is generated
from verified numbers, not written by the model.

For everything else — the explanation, the procedure, what to do next — it is a
well-grounded draft, and it can be confidently wrong in ways nothing catches. It
cites its sources and every citation links to the paragraph, so it is checkable
in about five seconds. That checkability is the actual product.

What I would not do is let it answer without escalation on anything it flags as
low confidence, or on anything outside the policy — the schema already forces
out-of-scope answers to cite nothing and escalate. And I would not deploy it
without the rejection-rate metric, because the failure mode of this kind of
system is not an outage, it is quiet decay, and the day it starts being wrong
more often is a day I want to find out about from a dashboard rather than from
Finance.

The honest framing: this is a tool that makes the policy searchable and does the
arithmetic reliably. It is not a tool that makes decisions, and the design
deliberately prevents it from sounding like one.
