# CLOSE-WAIT follow-up — 2026-08-24

## Observation

The Gemini 3.6 rollout worker itself accumulated proxy connections in
`CLOSE-WAIT`, all targeting `100.10.1.210:47899`. The count increased while
completed traces were being written, so this was a local transport-lifecycle
leak rather than only provider latency.

## Cause

Each isolated rollout created synchronous `requests.Session` objects for
Serper, Jina, reverse-image upload/search, reference-image download, and
Baidu OCR. Those sessions were stored in thread-local state but were not owned
by the per-case resource graph. The earlier cleanup covered Gemini async
clients and helper runtimes, but not these synchronous proxy pools.

## Fix

`src/integrations/http_sessions.py` tracks sessions created by each owner.
The owner is now closed from `Orchestrator.aclose()` through the nested tool
resource graph. Baidu OCR no longer uses one process-global session-bearing
client; its access-token cache remains process-global, while its HTTP session
is per rollout and closeable.

The fix does not alter prompts, model selection, sampling, tool policy, or
retry policy. It only makes transport ownership explicit and closes sessions
after a case reaches a terminal outcome.

## Validation before server rollout

- targeted lifecycle and transport tests: 51 passed;
- Python compilation of `src`: passed;
- the affected 3.6 run was stopped after preserving 30 terminal trace files;
- a fresh server smoke run must verify that `CLOSE-WAIT` does not grow with
  completed traces before resuming the larger rollout.
