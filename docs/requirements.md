Part B - Technical exercise
Overview
Build a small backend service that exposes an LLM-powered analysis endpoint. This is not about building something polished - it is about seeing how you think through the engineering of an AI system under real constraints.
Endpoint: POST /analyze
Accept a free-text query and return a structured JSON response. The response schema must be enforced - use function calling, tool use, or a structured output API. The same endpoint must support a ?stream=true parameter, streaming the response via Server-Sent Events (SSE).
Error handling
Handle the following cleanly:
•	Transient LLM API failures - retry with exponential backoff
•	Request timeouts - the service must not hang indefinitely
•	Malformed or incomplete JSON from the LLM - retry, fall back, or surface a structured error (your call - justify it)
•	HTTP-layer errors and LLM-layer errors handled separately
Deliverables
•	Working code in any language or framework
•	A README with setup instructions
•	A short technical memo (see below)
•	Submit ahead of the interview and be ready to demo live
Technical memo
This is the most important part. Cover:
•	Known limitations and what you would change at production scale - latency, cost, observability, failure modes
•	Why you chose your LLM and the key design decisions you made
•	How you resolved the streaming + structured output constraint
We are not just evaluating whether the code runs. We want to know whether you understand where it breaks.
