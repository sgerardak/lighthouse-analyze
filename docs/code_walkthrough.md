# Code walkthrough

A study guide for `lighthouse-analyze`. Written to be read once end to end before
the interview, and then dipped into per file.

Contents:

1. [What the service is](#1-what-the-service-is)
2. [The normal path: `POST /analyze`](#2-the-normal-path-post-analyze)
3. [The streaming path: `POST /analyze?stream=true`](#3-the-streaming-path-post-analyzestreamtrue)
4. [File by file](#4-file-by-file)
5. [Part B requirements, mapped to code](#5-part-b-requirements-mapped-to-code)
6. [Gaps, holes and things that are only half-built](#6-gaps-holes-and-things-that-are-only-half-built)
7. [Decisions the code does not explain — decide these before the interview](#7-decisions-the-code-does-not-explain)

---

## 1. What the service is

A FastAPI backend with one real endpoint. An employee asks a free-text question
about the company expense policy; the service answers with a JSON object whose
shape is guaranteed, which cites the policy sections it used, and whose numbers
have been recomputed in Python rather than trusted from the model.

The central idea, and the thing worth leading with in the demo: **the model is
used for reading and writing, never for deciding.** It reads the policy, extracts
the numbers, writes the prose. Python does the comparison, composes the sentence
that states the comparison, and refuses the answer if the model's own figures
disagree with each other.

Three things exist that the brief did not ask for, and each has a reason:

- `app/data/expense_policy.md` — a fictional policy, so there is something
  concrete to be right or wrong about. Without it "structured output" has no
  content to be structured.
- `app/static/index.html` — the demo page, so the live demo is a browser rather
  than a terminal, and so the SSE event timing is visible.
- `evals/model_eval.py` — the measurement behind the model choice. This is the
  thing that turns "I picked Haiku" into "I picked Haiku and here is the table".

---

## 2. The normal path: `POST /analyze`

Follow one request from the socket to the response body.

**1 — Middleware assigns an id.** `request_context` in `app/main.py` generates a
`uuid4().hex`, puts it on `request.state.request_id`, and starts a timer. On the
way out it stamps `X-Request-ID` on the response and logs one line with method,
path, status and duration. Every error body carries the same id, so a user
reporting a bad answer gives you one string that finds the request in the logs.

**2 — FastAPI parses and validates the body.** The body becomes an
`AnalyzeRequest` (`app/schemas.py`). Its field validator strips the query and
then splits into two outcomes:

- empty after stripping → plain `ValueError` → FastAPI raises
  `RequestValidationError` → our handler returns **422 `invalid_request`**.
- longer than `MAX_QUERY_LENGTH` (2000) → raises `QueryTooLongError`, which is
  *not* a `ValueError`. Pydantic carries it through in the error's `ctx`, and
  `_find_query_too_long` in `main.py` digs it back out so the caller gets
  **413 `query_too_long`** instead of a generic 422.

That second step is deliberate and slightly clever. Be ready to explain it: a
413 tells a client "your input was too big, do not retry as-is", where a 422 only
says "something about your body was wrong".

**3 — The route decides buffered or streamed.** `analyze()` reads the `stream`
query parameter. `stream=false` calls `analyze_query()`.

**4 — The request deadline opens.** `analyze_query` in `app/service.py` wraps
everything in `asyncio.timeout(REQUEST_TIMEOUT_SECONDS)` (45 s). If the whole
thing — retries and backoff included — outlasts it, the `TimeoutError` becomes
**504 `request_timeout`, layer `service`**. This is the guarantee that the
service never hangs: whatever the provider does, the caller waits at most 45 s.

**5 — Policy and client are fetched from cache.** `get_policy()` and
`get_llm_client()` are both `@lru_cache`. The policy was read and indexed during
`lifespan` startup, so a missing or malformed policy file stops the process from
booting rather than failing a request at 3am. The client is one shared
`AsyncAnthropic` with one connection pool.

**6 — `_call_llm` runs the retry loop.** Tenacity `AsyncRetrying` with:

- `stop_after_attempt(1 + MAX_RETRIES)` = 4 attempts,
- `wait_exponential_jitter(initial=0.5, max=8)`,
- `retry_if_exception_type((LLMUnavailableError, LLMTimeoutError))`,
- `reraise=True` so the caller sees the real error, not tenacity's `RetryError`.

Only those two error types are retried. The reasoning is in
[§5](#5-part-b-requirements-mapped-to-code); the short version is that both mean
the provider produced nothing, so the retry costs nothing.

**7 — The provider call.** `AnthropicClient.get_structured_output` builds the
request in `_request()` — shared with the streaming path so the two cannot drift:

```python
model:       MODEL_NAME
max_tokens:  MAX_OUTPUT_TOKENS (1024)
system:      [{type: text, text: <policy prompt>, cache_control: ephemeral}]
messages:    [{role: user, content: "<question>...</question>"}]
tools:       [{name: submit_policy_answer,
               input_schema: PolicyAnswer.model_json_schema()
                             + additionalProperties: false,
               strict: true}]
tool_choice: {type: tool, name: submit_policy_answer}
```

Four things to be able to defend here:

- `tool_choice: {type: "tool"}` forces the model to answer *through the tool*.
  There is no free-text path, so there is no "sometimes it returns prose" case.
- `strict: true` is what makes `required` binding. Without it the API treats
  required fields as advice; the memo measured Sonnet omitting `sources` in 3 of
  16 calls.
- `additionalProperties: false` is set in the Anthropic client, not on the
  Pydantic model, because it is a provider requirement and `PolicyAnswer` has to
  stay provider-neutral.
- `max_retries=0` on the SDK client. The SDK retries 429/5xx by itself by
  default; that is disabled so the service owns the entire retry policy. One
  place to reason about total latency against the 45 s deadline.

**8 — SDK exceptions are translated at the boundary.** `_as_app_error` maps:

| SDK exception | Our error | Retried? |
|---|---|---|
| `APITimeoutError` | `LLMTimeoutError` (504) | yes |
| `APIConnectionError` | `LLMUnavailableError` (503) | yes |
| `APIStatusError` 401/403 | `LLMAuthError` (500) | no |
| `APIStatusError` 429 | `LLMUnavailableError` (503) | yes |
| `APIStatusError` ≥500 | `LLMUnavailableError` (503) | yes |
| `APIStatusError` other 4xx | `InternalError` (500) | no |

`APITimeoutError` is checked before `APIConnectionError` because it subclasses
it. The last row is the interesting one: a provider 400 means *we* built a bad
request, so it is our bug and no caller retry helps.

**9 — `_to_result` pulls the tool call out.** Three rejections here, all
`LLMInvalidOutputError`: `stop_reason == "max_tokens"` (the answer was truncated
mid-generation), no `tool_use` block with our tool name, or a tool input that is
not an object. It also records usage — input tokens, output tokens, cache reads,
stop reason, wall-clock latency — into the frozen `LLMResult` dataclass.

**10 — `_log_usage` writes one line per call**, whether or not the answer
survives the checks below. That matters: rejected answers are still billed, so
they still have to appear in the cost log.

**11 — `_build_response` runs the three checks.** This is the heart of the
service and it is shared by both paths, so a streamed answer is held to exactly
the same standard as a buffered one.

1. **Re-validate** the tool input against `PolicyAnswer`. The JSON Schema
   constrained the shape; Pydantic adds the cross-field rules JSON Schema cannot
   express — `headcount >= 1`, and "out-of-scope answers must cite nothing and
   must escalate". Failure → `LLMInvalidOutputError`, summarised by `_summarise`
   into one readable line.
2. **Check every citation exists** via `validate_sources`. A model can write
   `"9.9"` as easily as `"3.2"`, and an invented citation is what a fabricated
   answer looks like. Note what this does *not* prove: the very first failure
   during development cited 3.2 correctly and drew the opposite conclusion from
   it. Existence is not support.
3. **Recompute the arithmetic** via `verify_arithmetic`. Details in
   [§4 `limits.py`](#applimitspy).

**12 — Python writes the decisive sentence.** `verdict_sentence(answer)` composes
something like

> 50 EUR per person is above the 40 EUR per person limit (section 4.2).

and it is prepended to the model's prose with `model_copy`. The model was
instructed (prompt rule 11) not to state the comparison itself. This exists
because during development every structured field was correct — `verdict:
above_threshold`, `per_person: 50`, `limit: 40` — while the prose read *"Yes,
that is within policy."* Correct JSON, wrong sentence, and the sentence is what
the employee reads.

**13 — Response out.** `AnalyzeResponse` → JSON → middleware stamps the header
and logs the line.

### What the caller sees when it goes wrong

Every failure, at every layer, has the same body:

```json
{"error": {"type": "llm_timeout", "layer": "llm",
           "message": "...", "retryable": true, "request_id": "..."}}
```

Including the ones the framework raises. `handle_http_exception` catches
Starlette's own `HTTPException` so a 404 answers in our shape instead of
`{"detail": "Not Found"}`, and it re-attaches headers the status depends on —
`Allow` on a 405 — with `setdefault`.

---

## 3. The streaming path: `POST /analyze?stream=true`

Same URL, same body, same validation. `stream=true` returns
`sse_response(stream_analyze_query(...))` — an `EventSourceResponse` wrapping an
async generator.

**Ordering point worth knowing:** the body is parsed and validated *before* the
generator is ever consumed. So a malformed body on the streaming path is still a
normal JSON **422**, not an SSE error event. There is a test for exactly this
(`test_endpoint_rejects_a_bad_body_without_streaming`).

### The event contract

| Event | Payload | When |
|---|---|---|
| `meta` | `{request_id, model}` | once, first, before the model is called |
| `delta` | `{text}` | zero or more, in order — **provisional** |
| `result` | the full `AnalyzeResponse` | once, on success |
| `error` | the same `ErrorResponse` as the buffered path | instead of `result`, on failure |

### The sequence

**1 — `meta` goes out immediately**, before the policy or the client are even
fetched, so the caller gets a first byte without waiting on a cold client.

**2 — A monotonic deadline is computed** (`time.monotonic() + 45`) rather than an
`asyncio.timeout` block. The comment in the code is the answer to the obvious
question: a timeout cancel scope cannot safely span the `yield`s of an async
generator, because the consumer runs in a different context and the scope would
be entered and exited across it. So the deadline is checked *between* events
instead, at the top of the event loop body.

**3 — The provider stream opens.** `AnthropicClient.stream_structured_output`
takes the same `_request()` dict and adds one thing:

```python
request["tools"][0]["eager_input_streaming"] = True
```

Without it the API buffers the tool input and hands it over in one piece — there
would be literally nothing to stream. With it, the API stops validating the tool
input as it goes. That is a real cost: **`strict: true` is weaker on the
streaming path than on the buffered path.** What makes it survivable is that
`_build_response` treats the final output as untrusted anyway.

**4 — Partial JSON is parsed on every chunk.** The adapter filters for
`event.type == "input_json"`, accumulates `partial_json` into a string, and
parses the accumulation with

```python
jiter.from_json(raw.encode(), partial_mode="trailing-strings")
```

This is the single most important implementation detail in the streaming path.
The SDK's own parsed snapshot **omits an unterminated string**, so `answer`
stayed invisible until it was complete and the first version of this endpoint
delivered 287 characters in one 31 ms burst — an SSE endpoint that satisfied the
requirement while streaming nothing. `trailing-strings` keeps the half-written
string, which turned that into 39 deltas over a second. Mid-token states that
parse as nothing at all return `None` and are skipped.

Note the abstraction boundary: the adapter yields **snapshots of the whole
accumulated object**, never "a delta of the `answer` field". Deciding which field
matters is the service's business, not the provider adapter's — which is what
makes the interface portable to a provider whose streaming shape is different.

**5 — `_stream` decides what the client is told.** For each snapshot:

- deadline check → `RequestTimeoutError` if blown;
- `event.result is not None` → log usage, run `_build_response` (the **same**
  three checks), emit `result`, return;
- otherwise, the interesting part:

```python
if "answer" in snapshot and not sentence_done:
    sentence_done = True
    sentence = verdict_sentence(_partial(snapshot))
    if sentence:
        prefix = f"{sentence} "; sent = prefix
        yield delta_event(prefix)
if not sentence_done:
    continue
composed = prefix + (snapshot.get("answer") or "")
if len(composed) > len(sent):
    yield delta_event(composed[len(sent):]); sent = composed
```

**This is the resolution of the streaming-vs-structured-output constraint, and
it is the thing to explain slowly in the interview.** The conflict is: our
decisive sentence must come *first*, but it is computed from numbers the model
generates *last*, after its prose — unless you change the order. So the field
order in `PolicyAnswer` is made load-bearing. The nine numeric/flag fields are
declared first and `answer` is declared last. The model generates properties in
schema order. Therefore **the moment the key `answer` appears in a snapshot,
every number the sentence needs is already final.** Compute the sentence, send it
as the first delta, then stream the prose behind it.

`_partial()` adapts a half-written snapshot dict into something
`verdict_sentence` can read (`SimpleNamespace` with the five fields it touches),
explicitly without validating it — the sentence is provisional until
`_build_response` confirms the same numbers at the end.

**6 — Retries stop at the first byte.**

```python
except RETRYABLE_ERRORS as exc:
    if sent or attempt == attempts:
        raise
    await asyncio.sleep(_backoff(attempt))
```

Before anything has reached the client, a transient failure is retried and the
client never knows. After the first delta, the stream is committed — you cannot
un-send text — so the failure becomes an `error` event instead. Two tests pin
both halves of this.

**7 — Failures become events, not statuses.** `stream_analyze_query` wraps
`_stream` and converts any `AppError` into an `error` event, and anything else
into `InternalError`. The HTTP status line went out with `meta`; it is far too
late to return a 502.

**8 — Deltas are provisional, and the contract says so.** The checks need the
whole answer, so text can reach the client and then be contradicted by an
`error`. The demo page handles this honestly: it renders deltas as they arrive
and then **replaces** the rendered text with the validated `result.answer` when
`result` lands.

---

## 4. File by file

### `app/main.py` — the edge

Everything HTTP-shaped lives here and nothing else does.

- `lifespan` — loads the policy at startup so a broken file is a boot failure.
- `request_context` middleware — request id, timing, access log, `X-Request-ID`.
- `handle_app_error` — the one handler that renders `AppError` → JSON.
- `handle_validation_error` — maps Pydantic failures to `invalid_request`, and
  rescues `QueryTooLongError` out of the `ctx` for its 413.
- `handle_http_exception` — gives framework-raised 404/405 our error shape.
- `handle_unexpected_error` — generic 500, traceback to logs not to the caller.
- Routes: `/` (demo page), `/policy` (rendered, anchored), `/policy.md` (raw, as
  the model sees it), `/health`, `/analyze`.

Worth knowing for a follow-up question: the `except` branch in the middleware
looks like dead code but is not. Starlette's stack is `ServerErrorMiddleware →
your middleware → ExceptionMiddleware → router`. Handlers for specific exception
classes (`AppError`, `RequestValidationError`, `HTTPException`) run *below* the
middleware, so the middleware sees an ordinary response with the real status. The
handler for bare `Exception` is installed in `ServerErrorMiddleware`, *above* the
middleware — so a genuinely unhandled exception is the one case that propagates
through, gets logged, and is re-raised.

### `app/config.py` — settings

`pydantic-settings` reading `.env`. `ANTHROPIC_API_KEY` has no default, so the
process refuses to start without one. `@lru_cache` on `get_settings()` so the
file is read once.

### `app/schemas.py` — the wire contract

`AnalyzeRequest`, `PolicyAnswer`, `AnalyzeResponse`, `ErrorDetail`,
`ErrorResponse`.

`PolicyAnswer` is the important one and every choice in it is deliberate:

- **Field order is load-bearing** (see §3). Do not reorder it. There is no test
  that catches a reordering — see [§6](#6-gaps-holes-and-things-that-are-only-half-built).
- The five numeric fields are **required but nullable**, because strict tool use
  expects every property to be present. `null` is how the model says "no amount
  in this question", which is different from "I forgot".
- `headcount` has no `Field(ge=1)` because **strict mode rejects `minimum` on an
  integer schema**. The bound is a `model_validator` instead.
- Every field's `description` is written *for the model to read* — the JSON
  schema is sent as the tool definition, so these descriptions are prompt.
- `verdict` is "purely the numeric comparison, never a judgement". The
  description labours this point because a cap and a trigger both use
  `above_threshold` and only one of them is a rule breach.
- The two `model_validator`s encode what JSON Schema cannot: headcount ≥ 1, and
  out-of-scope ⇒ no sources ∧ must escalate.

### `app/errors.py` — one error model, three layers

`AppError` base with class attributes (`error_type`, `layer`, `status_code`,
`retryable`, `default_message`) and a per-instance message. Subclasses grouped by
layer:

- **http** — `InvalidRequestError` (422), `QueryTooLongError` (413),
  `HTTPStatusError` (per-instance status, for router 404/405).
- **llm** — `LLMUnavailableError` (503, retryable), `LLMTimeoutError` (504,
  retryable), `LLMInvalidOutputError` (502, retryable), `LLMAuthError` (500, not
  retryable).
- **service** — `RequestTimeoutError` (504, retryable), `InternalError` (500),
  `NotImplementedYetError` (501).

The `layer` field is the part with operational value, and that is the argument to
make: a spike in `llm` means check the provider's status page, a spike in `http`
means a client is broken, `service` means it is us. A single `500` tells you
none of that.

`to_error_response` imports `app.schemas` inside the function, because
`app.schemas` imports `QueryTooLongError` from here — a module-level import would
be circular.

### `app/policy.py` — the grounding document

Loads the markdown, indexes `### N.N Title` headings into `{id: title}` with a
regex. Top-level `## N. Title` headings are deliberately *not* indexed, so a
citation is always a subsection. `PolicyLoadError` on missing file or zero
sections, raised at startup. `validate_sources` returns the ids that do not
exist. `@lru_cache` — the policy is read once and never reloaded, so editing the
markdown requires a restart.

### `app/limits.py` — the numbers, as data

The premise of the whole design: *the model reads the policy as prose and is
measurably bad at comparing a computed amount against a threshold*, so the limits
it is allowed to cite live here as numbers.

- `Limit(name, value, section, per_person, kind)` — ten entries.
- `kind` is `"cap"` or `"trigger"`, and this distinction is the subtlest thing in
  the repo. A **cap** is a number you must stay under (hotel 180/night). A
  **trigger** is a number that, once crossed, requires something extra
  (250 EUR missing receipt → also needs manager approval). Crossing a trigger is
  **not** a policy breach, and the sentence must not imply it is. Hence
  "above the 250 EUR *threshold*" for a trigger and "above the 250 EUR *limit*"
  for a cap.
- `find_limit(value, sources)` — several limits share a value (500 appears in
  both 4.1 and 4.2; 250 in both 3.2 and 5.2), so the cited sections disambiguate.
  Falls back to `matches[0]` — see [§6](#6-gaps-holes-and-things-that-are-only-half-built), this fallback can pick wrong.
- `verify_arithmetic(answer) -> list[str]` — three checks:
  1. if `amount_eur` and `headcount` are both present, redo the division and
     compare to `per_person` within `TOLERANCE = 0.01`;
  2. `limit_applied` must be a value that appears in `POLICY_LIMITS` — an
     invented limit is a fabrication;
  3. recompute the verdict from `compared vs limit_applied` and reject if the
     model's `verdict` disagrees.
- `verdict_sentence(answer) -> str | None` — composes the sentence, or `None`
  when there is nothing to compare. It states **only the comparison, never the
  permission**, and that restraint is load-bearing: a 30 EUR gift card is
  *within* the 50 EUR gift limit and still forbidden outright, so a yes/no opener
  would have been a new bug.

### `app/prompts.py` — the system prompt

Eleven numbered rules plus the policy inlined in `<policy>` tags. The rules worth
being able to recite:

- **1, 2, 3** — answer only from the policy, cite ids, out-of-scope ⇒ escalate.
- **8** — the question is wrapped in `<question>` tags and the model is told to
  treat anything inside as a question, not as instructions. A speed bump against
  prompt injection, not a defence.
- **9** — fill the numeric fields, and "these numbers are recomputed and the
  answer is rejected if they do not add up". Telling the model it is being
  checked.
- **10** — verdict is the comparison, never a judgement.
- **11** — do not state the comparison in prose, because Python prepends it.

### `app/streaming.py` — the SSE contract

Small and deliberately dumb. `ServiceEvent(name, data)`, four constructors
(`meta_event`, `delta_event`, `result_event`, `error_event`), `to_sse()` which
renders them as `ServerSentEvent` frames, and `sse_response()` which wraps them
in an `EventSourceResponse` with the `X-Request-ID` header. The module docstring
*is* the contract, including "deltas are provisional".

`error_event` reuses `to_error_response`, so a streamed failure and a buffered
failure are byte-identical in shape. That is worth pointing at.

### `app/policy_view.py` — the policy page

A deliberately non-general markdown renderer: headings, bullets, blockquotes,
paragraphs, and nothing else, because that is all the policy file uses. Its job
is to make `/policy#3.2` land on the text the citation came from, so a demo can
click a citation and land on the paragraph. Everything is `escape()`d, with a
test that proves a `<script>` in the policy renders as text.

### `app/llm/` — the provider seam

- `base.py` — `LLMResult` (data + usage + latency), `LLMStreamEvent` (either a
  `snapshot` or a `result`), and the `LLMClient` ABC with two methods.
- `__init__.py` — `get_llm_client()` factory, `@lru_cache`, imports the provider
  module lazily so the package stays free of SDKs.
- `anthropic_client.py` — the only file in the repo that imports `anthropic`.

The claim to make: adding Gemini is a new file plus one branch in the factory,
and nothing else changes. That claim is true *structurally*. See
[§7](#7-decisions-the-code-does-not-explain) for where it is optimistic.

### `app/static/index.html` — the demo page

Single file, no dependencies. Textarea, a "stream the answer" checkbox (on by
default), four example questions. Two collapsible panels that are the point of
it for an interview:

- **Response JSON** — what `POST /analyze` actually returns.
- **SSE events** — every frame with a millisecond timestamp and its character
  count. This is what makes "the first delta is the computed sentence, and it
  arrives before any model prose" *visible* rather than asserted.

`runStreaming` reads the response body manually with a `ReadableStream` reader
and splits on `\n\n` rather than using `EventSource` — because `EventSource`
cannot issue a POST. Comment lines starting with `:` (SSE keep-alives) are
skipped. On `result` it replaces the accumulated delta text with the validated
answer.

### `tests/` — six files, no network

Every test fakes the provider, so the suite is free and safe in CI.

- `test_errors.py` — the error contract: the five fields, the matching header,
  404/405 in our shape, the `Allow` header surviving, 422 and 413, unique ids.
- `test_limits.py` — the largest file, and rightly: reversed verdict, wrong
  division, missing `per_person`, invented limit, exactly-at-the-limit,
  disambiguation by citation, cap-vs-trigger wording in both directions.
- `test_policy.py` — loader, subsection indexing, unknown-id detection, and both
  failure modes of the loader.
- `test_policy_view.py` — every section gets an anchor; escaping.
- `test_retry.py` — 1 attempt on success, 3 on two transient failures, exactly
  `1+MAX_RETRIES` then reraise, **no** retry on malformed output, **no** retry on
  auth error, and the deadline cutting off slow retries.
- `test_streaming.py` — event order, the computed sentence as first delta, deltas
  reassembling into the final answer, failed checks becoming an `error`, invented
  citation, retry before the first delta, **no** retry after it, truncated
  stream, and the endpoint's content type and headers.

### `evals/model_eval.py` — the quality measurement

Not a unit test; it calls the API and costs money, and refuses to run without
`--yes` after printing an estimate. Twelve questions with hand-written
ground-truth conclusions, nine of them numeric comparisons, run N times across
four configs (`shipped`, `threshold-rule`, `thinking`, `sonnet5`).

The design choice to defend: **it grades the answer a user would actually
receive.** It runs `PolicyAnswer.model_validate`, `validate_sources`,
`verify_arithmetic` and `verdict_sentence` — the same functions the service uses
— and only then hands the composed text to a Claude Opus 5 judge. Grading the
model's raw prose would measure something the service never shows anyone.

Three columns are scored separately because they fail for different reasons:
*answers correct* (the judge), *citations ok* (deterministic), *usable output*
(did it survive the checks at all).

---

## 5. Part B requirements, mapped to code

### "The response schema must be enforced — use function calling, tool use, or a structured output API"

**Satisfied, in four layers.**

| Layer | Where | What it guarantees |
|---|---|---|
| Forced tool call | `anthropic_client._request` → `tool_choice: {type: "tool"}` | there is no free-text escape hatch |
| Strict tool use | same → `"strict": True` + `additionalProperties: False` | `required` is binding, not advice |
| Pydantic re-validation | `service._build_response` → `PolicyAnswer.model_validate` | cross-field rules JSON Schema cannot express |
| Domain checks | `policy.validate_sources`, `limits.verify_arithmetic` | citations exist; the numbers agree |

Be explicit that these are four different guarantees and only the first two come
from the API. The memo's measured evidence for `strict`: without it Sonnet
omitted the required `sources` field in 3 of 16 calls; with it, 8/8.

### "The same endpoint must support a `?stream=true` parameter, streaming via SSE"

**Satisfied.** Same route function, `stream: bool = False` query parameter,
`sse_response(...)` from `app/streaming.py` using `sse-starlette`. Four named
event types, documented in the module docstring and the README, tested end to end
including the response content type (`test_endpoint_returns_an_event_stream`).

### "Transient LLM API failures — retry with exponential backoff"

**Satisfied, in two places, and this is worth knowing precisely because it is a
likely question.**

- Buffered path: `service._call_llm`, tenacity, `wait_exponential_jitter(initial=0.5, max=8)`,
  4 attempts, only `LLMUnavailableError` and `LLMTimeoutError`.
- Streaming path: a hand-rolled loop in `service._stream` using
  `_backoff(attempt) = min(8.0, 0.5 * 2**(attempt-1))`, with the extra rule that
  a retry is only permitted while nothing has been sent.

Two implementations exist because tenacity cannot wrap an async generator that
has already yielded — retrying a generator mid-iteration is not a thing tenacity
does. **The divergence is real: the streaming backoff has no jitter.** See §6.

The SDK's own retries are turned off (`max_retries=0`) so there is exactly one
retry policy in the system.

### "Request timeouts — the service must not hang indefinitely"

**Satisfied, at two levels.**

| Level | Value | Mechanism | Error on expiry |
|---|---|---|---|
| One provider call | `LLM_TIMEOUT_SECONDS` = 20 | `AsyncAnthropic(timeout=...)` | `APITimeoutError` → `LLMTimeoutError` (504, retryable) |
| The whole request | `REQUEST_TIMEOUT_SECONDS` = 45 | `asyncio.timeout` (buffered) / monotonic deadline checked between events (streaming) | `RequestTimeoutError` (504, `service`) |

The outer one is what makes "must not hang" true regardless of what the retries
do. On the streaming path it is a deadline check rather than a timeout scope for
the async-generator reason given in §3.

### "Malformed or incomplete JSON from the LLM — retry, fall back, or surface a structured error (your call — justify it)"

**Satisfied: surfaced as a structured error, deliberately not retried.**

Caught in three distinct places:

1. `_to_result` — `stop_reason == "max_tokens"` (incomplete), no tool-use block,
   non-object tool input.
2. `_build_response` — Pydantic validation, including the cross-field rules.
3. `_build_response` — invented citations, and arithmetic that does not check
   out.

All become `LLMInvalidOutputError` → **502, layer `llm`, `retryable: true`**.

The justification the brief asks for: that call already produced billable output,
so retrying it pays twice. Paying twice should be a deliberate decision by
whoever owns the budget, not a library default — so the error is *marked*
retryable and the caller decides. `RETRYABLE_ERRORS` deliberately excludes it,
and `test_does_not_retry_a_malformed_answer` pins that.

On the streaming path the same failure arrives as an `error` event after visible
text, which is the honest cost of streaming provisional output.

### "HTTP-layer errors and LLM-layer errors handled separately"

**Satisfied, and slightly over-delivered** — there are three layers, not two.
`ErrorDetail.layer` is a `Literal["http", "llm", "service"]`, the exception
classes in `app/errors.py` are grouped by it, and the README tabulates every
type with its status, layer and retryability. Framework-raised 404/405 are pulled
into the same shape so there is no second error format anywhere in the surface.

### Deliverables

- **Working code** — yes, `uvicorn app.main:app`, plus a browser demo at `/`.
- **README with setup** — yes, including the two Windows traps (PowerShell's
  `curl` alias buffering SSE; the `jiter` Smart App Control block).
- **Technical memo** — `docs/memo.md`, which covers all three required topics
  and is the strongest artefact in the repo.
- **Ready to demo live** — yes; see §6 for what to check first.

---

## 6. Gaps, holes and things that are only half-built

Ordered by how likely they are to be found in a technical Q&A.

### A. The retry budget does not fit inside the request deadline

`MAX_RETRIES=3` means 4 attempts. At `LLM_TIMEOUT_SECONDS=20` plus backoff, the
worst case is roughly `20 + 0.5 + 20 + 1 + 20 + 2 + 20 ≈ 83 s` against a 45 s
deadline. **Attempts 3 and 4 can never complete.** In the worst case you get two
full attempts, a third that is cut mid-flight, and a `request_timeout`.

This is not a crash — the deadline does its job — but the configuration claims a
retry budget it cannot spend. Either `LLM_TIMEOUT_SECONDS` should be ~8–10 s (a
p50 of 2.8 s makes 20 s enormously generous), or `MAX_RETRIES` should be 1–2, or
the deadline should be derived rather than set independently. Pick one and have
the number ready.

### B. Two backoff implementations, one of them without jitter

`_backoff()` in `service.py` is deterministic: every streaming client that hits a
provider outage retries at exactly 0.5 s, 1 s, 2 s. That is a thundering herd,
and it is the exact failure mode jitter exists to prevent — and the buffered path
*does* use jitter. Fixing it is three lines. Know that it is there.

### C. The provider adapter has zero test coverage

`app/llm/anthropic_client.py` is the trickiest file in the repo —
`_as_app_error`, `_map_status_error`, `_parse_partial`, `_to_result`,
`eager_input_streaming` — and **every test fakes the client one level above it.**
Nothing exercises:

- the SDK-exception → `AppError` mapping (is `APITimeoutError` really checked
  before `APIConnectionError` at runtime? the ordering is right in the source,
  but no test proves it);
- `jiter` partial parsing, including the `None` path for unparseable states;
- `stop_reason == "max_tokens"` rejection;
- the tool-use block being absent.

These are all pure functions or trivially fakeable. This is the highest-value
missing test and the easiest to add. If there is time before the interview, add
it — it is more convincing to say "yes, and here is the test" than to say "good
catch".

### D. Field order is load-bearing and nothing enforces it

The entire streaming design rests on `answer` being the last property in
`PolicyAnswer`. A well-meaning refactor that moves it up breaks the first delta
silently — the sentence would be computed from half-written numbers, or not at
all — and **no test fails.** A three-line test asserting
`list(PolicyAnswer.model_json_schema()["properties"])[-1] == "answer"` closes it.

### E. Nothing verifies that the model picked the *right* limit

`verify_arithmetic` checks that `limit_applied` is *a* number from the policy and
that the verdict follows from it. It does not check that it is the *correct* limit
for the question. A team meal compared against the 80 EUR client-meal cap passes
all three checks, produces a confidently wrong answer with perfect internal
arithmetic, and reads exactly like a correct one.

The memo names unverified extraction ("a misread headcount produces a confidently
wrong answer") as the largest remaining hole. Limit *selection* is the same class
of hole and is not named. Treat them as one finding: **the model still chooses
which numbers enter the arithmetic, and that choice is unchecked.**

### F. `find_limit`'s fallback can mislabel a cap as a trigger

`find_limit` prefers a limit whose section the answer cited, and otherwise
returns `matches[0]`. The value 250 appears twice: 3.2 (`trigger`) and 5.2
(`cap`). If an answer sets `limit_applied: 250` and cites neither — or cites
something else entirely — the fallback picks 3.2, and the sentence reads
"above the 250 EUR **threshold** (section 3.2)" for what may be a hotel question.
Wrong noun, wrong section, on the one sentence the design exists to make
trustworthy.

Low probability, because the prompt asks for citations and the schema requires
the field — but `test_verdict_sentence_disambiguates_shared_values_by_citation`
only tests the case where the citation *is* present.

### G. `escalate_to_finance`, `in_scope` and `confidence` are unverified

- `escalate_to_finance` is checked only in one direction: out-of-scope must
  escalate. An in-scope answer that *should* escalate (prompt rules 4 and 5: a
  partially-answered question, or a request for an exception) and does not is
  accepted silently.
- `in_scope: true` with an empty `sources` list passes every check. The inverse
  is enforced; this direction is not.
- `confidence` is collected, displayed as a chip, and **acted on by nothing.**
  It does not gate escalation, does not appear in logs, does not affect the
  response. Either wire it to something (`low` ⇒ force escalation is the obvious
  one) or be ready to say it is decoration for the caller's benefit.

### H. `NotImplementedYetError` appears to be unused

Defined in `app/errors.py` with a 501, and I found no `raise` site. Grep before
the interview (`grep -rn NotImplementedYetError app/`) and delete it if it is
dead — a defined-and-unused error class invites the question "so what is not
implemented?" and you do not want to answer that with "nothing, that is leftover".

### I. No pinned dependencies

`requirements.txt` names nine packages and pins **none** of them. Two
consequences:

- A fresh `pip install -r requirements.txt` on the day of the interview can pull
  a new `anthropic` SDK. The streaming path depends on `messages.stream`, the
  `input_json` event type and the `eager_input_streaming` tool key — a minor SDK
  change there breaks the demo, and it breaks it *silently into a non-streaming
  endpoint* rather than into an error.
- The README says `jiter==0.16.0` fixed a Windows Smart App Control block, and
  `requirements.txt` does not pin it. The next clean install on that machine can
  hit the same wall.

**Do this before the interview: `pip freeze > requirements.lock.txt`, commit it,
and re-run the suite from a clean venv.** This is the single highest-risk item on
the list for the live demo, and the cheapest to fix.

### J. The demo path is untested off localhost

SSE dies behind a proxy that buffers. If the demo goes through anything other
than `127.0.0.1` — a tunnel, a screen-share of a deployed instance, a corporate
proxy — verify the deltas still arrive incrementally. `sse-starlette` sets
`Cache-Control: no-cache`; nginx additionally needs `X-Accel-Buffering: no`.
There is no such header set anywhere in the code.

### K. No auth, no rate limit, no backpressure

`/analyze` is open. Anyone who can reach the port can spend the API budget, and
one slow provider ties up a worker for up to 45 s with no queue and no shedding.
The memo names the backpressure half ("single process, no backpressure"); it does
not name that there is no authentication at all. For an internal Finance tool
this is the first thing a reviewer at Lighthouse would ask about.

### L. The provider stream is not explicitly closed on early exit

In `_stream`, `async for event in client.stream_structured_output(...)` is exited
early by `return` (on `result`) and by `raise` (on deadline). Neither explicitly
closes the inner async generator, so the underlying HTTP stream is finalised by
the event loop's async-generator hooks rather than deterministically.
`contextlib.aclosing` around it is the one-line fix. Under load, non-deterministic
connection release is how pools get exhausted.

### M. Smaller things

- **The limits table is hand-maintained.** `app/limits.py` duplicates numbers
  that also live in the markdown. A test asserts every limit cites a real
  section; nothing catches the policy changing 40 to 45 while the table does not.
  The memo names this.
- **`/health` is liveness, not readiness.** It reports the configured provider
  and model and counts policy sections; it never touches the provider. A bad API
  key yields a healthy `/health` and 100% failing requests.
- **The policy is cached forever.** Editing `expense_policy.md` requires a
  restart. Fine for the exercise, worth saying out loud.
- **Query text is never logged.** Good for privacy — and it means debugging a
  reported bad answer requires the user to resend the question. That is a
  trade-off, not an oversight; present it as one.
- **`AnalyzeResponse` carries no usage or latency.** The caller cannot see what
  the call cost. Cheap to add, useful for a client-side dashboard.
- **No `Retry-After`** on the 503 that wraps a provider 429, even though the
  provider usually tells us.
- **README claims "51 tests".** Count them before quoting the number.

---

## 7. Decisions the code does not explain

These are places where the code makes a choice and does not record the
alternative. Each needs an answer you can give in one sentence.

### 1. Why Anthropic at all, rather than Gemini?

The brief lists **"Anthropic / Google Gemini"** as the available providers. The
memo answers *which Claude model* in great measured detail, and never answers
*why Claude*. The eval framework only knows Anthropic configs.

This will be asked. Have a real answer: tool use / strict structured output
maturity, the `eager_input_streaming` behaviour the streaming design depends on,
prompt-caching semantics, existing team familiarity — and say honestly that a
provider comparison is exactly what the eval harness is for and that you did not
spend the budget on it for a take-home. Do not improvise this one live.

### 2. Why one tool call rather than two calls?

An alternative design: call 1 extracts the structured facts (fast, small, cheap,
no prose), call 2 streams the prose given the verified facts. That gets you a
*verified* stream — every delta is final, nothing is provisional, and the
contract stops needing the "deltas may be contradicted" caveat. The cost is two
round trips and roughly double the latency to first token.

The repo chose one call. The reasoning is implicit (latency, cost, simplicity)
and never stated. State it, and name the alternative — knowing the trade you did
*not* take is most of what "senior" means in this conversation.

### 3. Why text deltas rather than structured field events?

The SSE contract streams `{"text": ...}`. An alternative streams *fields* as they
finalise (`{"field": "per_person", "value": 50.0}`), which keeps the stream
structured end to end and lets a client render a numbers panel before any prose
exists. The current design chose text because the demo page renders prose.

Not wrong, but undefended in the code. One sentence: the consumer is a chat-style
UI, text deltas are what it needs, and field events would be the right call for a
dashboard consumer.

### 4. Why is `LLMAuthError` a 500 and not a 502?

It is in the `llm` layer but returns 500 rather than a 5xx that points upstream.
The reasoning is defensible — a bad key is *our* misconfiguration, not the
provider failing — but it makes `layer: "llm"` + `status: 500` the one
combination in the table that does not follow the pattern. Be ready to say it is
deliberate.

### 5. Is the provider seam really one file?

The claim is "adding a provider is a new module plus a branch". Structurally,
yes. But `PolicyAnswer.model_json_schema()` is passed to the adapter and each
provider's structured-output dialect differs (Gemini's `responseSchema` supports
a different subset of JSON Schema; `strict` has no exact equivalent), and
`eager_input_streaming` is Anthropic-specific with no guarantee another provider
streams partial tool input at all. **The interface is provider-agnostic; the
streaming guarantee is not.** Say that before someone else does — it is a better
answer than the clean version.

### 6. What is `eager_input_streaming` bound to?

The streaming design depends entirely on this one tool key. Before the demo,
confirm it is still accepted by the installed SDK and API version, and know what
happens if it is not: the API buffers, the whole tool input arrives at once, and
the endpoint quietly degrades to one giant delta — a passing test suite and a
non-streaming demo. The memo already tells the story of catching exactly this
class of bug by timing the deltas rather than eyeballing them. **Time the deltas
once more on the morning of the interview.**
