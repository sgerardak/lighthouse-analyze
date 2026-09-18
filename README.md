# Lighthouse Analyze Service

An LLM-powered endpoint that answers employee questions about a company expense
policy, returning a structured, schema-enforced JSON answer that cites the
policy sections it came from — with optional streaming over server-sent events.

Built for the Part B technical exercise ([docs/requirements.md](docs/requirements.md)).
The design reasoning, the measurements behind the model choice, and the known
failure modes are in **[docs/memo.md](docs/memo.md)**, which is the more
interesting read.

## Quickstart

Requires Python 3.11+ (developed on 3.12) and an Anthropic API key.
The request deadline uses `asyncio.timeout`, which is 3.11 and later.

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt   # Windows
# .venv/bin/python -m pip install -r requirements.txt     # macOS / Linux

cp .env.example .env          # then put a real key in ANTHROPIC_API_KEY

.venv/Scripts/python -m uvicorn app.main:app --reload
```

Then open **http://127.0.0.1:8000/**.

| URL | What it is |
|---|---|
| `/` | Demo page: ask a question, watch the answer stream in |
| `/policy` | The policy, anchored by section id (`/policy#3.2`) |
| `/docs` | Swagger UI |
| `/health` | Liveness, provider, model, number of policy sections |

Run it from the project root — `POLICY_PATH` is relative by default.

## The API

### `POST /analyze`

```bash
curl -s -X POST http://127.0.0.1:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"query":"Team dinner for 6 people costing 300 EUR total. Is that within policy?"}'
```

```json
{
  "request_id": "6c2b6a2d42c54d5ca33a724bb99d72d2",
  "model": "claude-haiku-4-5-20251001",
  "result": {
    "amount_eur": 300.0,
    "headcount": 6,
    "per_person": 50.0,
    "limit_applied": 40.0,
    "verdict": "above_threshold",
    "sources": ["4.2"],
    "in_scope": true,
    "escalate_to_finance": false,
    "confidence": "high",
    "answer": "50 EUR per person is above the 40 EUR per person limit (section 4.2). Team meals are limited to 40 EUR per person..."
  }
}
```

The numeric fields are the model's working. They are not decoration: the service
recomputes them and rejects the answer if they disagree, then writes the opening
sentence itself from the verified figures. See the memo for why.

### `POST /analyze?stream=true`

The same answer as server-sent events. Use `curl.exe -N` on Windows — in
PowerShell, `curl` is an alias for `Invoke-WebRequest`, which buffers.

```bash
curl -N -s -X POST "http://127.0.0.1:8000/analyze?stream=true" \
  -H "Content-Type: application/json" \
  -d '{"query":"Can I book a hotel at 220 EUR per night in a capital city?"}'
```

```
event: meta
data: {"request_id": "010608153b494160a1db65bde153eeba", "model": "claude-haiku-4-5-20251001"}

event: delta
data: {"text": "220 EUR is within the 250 EUR limit (section 5.2). "}

event: delta
data: {"text": "Yes, you can book this"}

event: delta
data: {"text": " hotel."}

event: result
data: {"request_id": "0106...", "model": "...", "result": { ... }}
```

The first delta is composed by the service from the model's verified numbers,
before any of its prose exists.

| Event | Meaning |
|---|---|
| `meta` | Sent first, before the model is called |
| `delta` | The next piece of the answer. **Provisional** |
| `result` | The complete, fully validated answer |
| `error` | A failure, in the same shape as the non-streaming path |

Deltas are provisional because the checks need the whole answer. A client should
render them but treat them as unconfirmed until `result` arrives, and be ready
for `error` to arrive after visible text.

### Errors

Every failure the service raises returns the same body:

```json
{"error": {"type": "llm_timeout", "layer": "llm", "message": "...",
           "retryable": true, "request_id": "..."}}
```

`layer` separates the caller's problem from the provider's from ours.

| Type | Status | Layer | Retryable |
|---|---|---|---|
| `invalid_request` | 422 | http | no |
| `query_too_long` | 413 | http | no |
| `llm_unavailable` | 503 | llm | yes |
| `llm_timeout` | 504 | llm | yes |
| `llm_invalid_output` | 502 | llm | yes |
| `llm_auth_error` | 500 | llm | no |
| `request_timeout` | 504 | service | yes |
| `internal_error` | 500 | service | no |

Every response carries an `X-Request-ID` header matching `request_id` in the
body, and each request is logged with its id, path, status and duration.

On the streaming path the status line is already sent by the time most failures
occur, so those arrive as an `error` event on a 200 response. Request validation
happens before the stream opens, so a malformed body is still a normal JSON 422.

## Configuration

All settings load from `.env` (see `.env.example`).

| Variable | Default | Purpose |
|---|---|---|
| `ANTHROPIC_API_KEY` | *required* | No default; the app will not start without it |
| `LLM_PROVIDER` | `anthropic` | Selects the client implementation |
| `MODEL_NAME` | `claude-haiku-4-5-20251001` | See the memo for why this one |
| `LLM_TIMEOUT_SECONDS` | `20` | Ceiling for one call to the provider |
| `REQUEST_TIMEOUT_SECONDS` | `45` | Ceiling for the whole request, retries included |
| `MAX_RETRIES` | `3` | Retries after the first attempt, so 4 attempts |
| `MAX_QUERY_LENGTH` | `2000` | Characters; longer gets 413 |
| `MAX_OUTPUT_TOKENS` | `1024` | Per answer |
| `POLICY_PATH` | `app/data/expense_policy.md` | Loaded once at startup |

The policy is read and indexed during startup, so a missing or malformed file
stops the service from starting rather than failing later during a request.

## Tests

```bash
.venv/Scripts/python -m pytest tests/ -q
```

51 tests, none of which call the API — the provider is faked throughout, so the
suite is free to run and safe in CI. They cover the policy loader, the
arithmetic checks and the sentence they produce, the retry policy and the
request deadline, the error contract including framework-raised 404s and 405s,
and the streaming event contract end to end.

Answer *quality* is a separate question, measured by the eval in
[evals/](evals/) — that one does call the API and costs money, so it prints an
estimate and refuses to run without `--yes`.

## How it fits together

```
app/
  main.py              routes, request-id middleware, exception handlers
  config.py            settings
  schemas.py           request and response models; field order matters
  errors.py            the error model, grouped by layer
  service.py           orchestration: retries, deadline, validation, streaming
  policy.py            loads and indexes the policy
  policy_view.py       renders the policy page
  limits.py            the policy's numbers as data, and the arithmetic checks
  prompts.py           the system prompt
  streaming.py         the SSE event contract
  llm/
    base.py            the provider interface. The app depends only on this
    anthropic_client.py the Anthropic implementation
    __init__.py        provider factory
  static/index.html    the demo page
```

Only `llm/anthropic_client.py` imports a provider SDK. Adding another provider
means adding a file and a branch in the factory; nothing else changes.

An answer passes three checks before anyone sees it: it must satisfy the schema
including cross-field rules, every section it cites must exist in the policy,
and its arithmetic must match what Python computes from the same inputs.

## Notes for Windows

- If `.\.venv\Scripts\Activate.ps1` is blocked, either run
  `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`, or
  skip activation and call `.\.venv\Scripts\python.exe` directly.
- With Smart App Control enforced, an unsigned native wheel can be blocked at
  import. It happened here with `jiter` 0.17.0 (`DLL load failed ... blocked by
  an application control policy`); installing `jiter==0.16.0` resolved it.
  Nothing in the code depends on the version.
