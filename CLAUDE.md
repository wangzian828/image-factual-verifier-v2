# Claude Project Instructions

Follow `AGENTS.md` for contributor rules and `docs/architecture.md` for the active runtime contract.

The required production workflow is:

`Perception -> Planning -> iterative native ReAct -> Coverage audit/Replanning loop -> Judgment`

Use Gemini Interactions end to end for every active Gemini LLM and vision call. Tool-bearing stages use native `function_call` / `function_result` items linked by `previous_interaction_id`. Never switch wire protocols or providers after an error, parse native calls through prompt tags, or synthesize heuristic fallback plans or judgments. Invalid protocol responses, missing credentials, exhausted retries, and invalid required structured outputs are hard failures.

Verification evidence must come from successful recorded tool calls and map to explicit plan question IDs. Every tool result must have `status: "success"` or `status: "error"`; malformed/statusless results fail the contract. When the bounded audit/replanning budget ends with factual coverage gaps, continue to ledger-bound Judgment and return the exact typed `unverifiable` reasons. Engineering, provider, protocol, malformed-output, and all-tools-failed conditions still raise before Judgment.

Set `IMAGE_UPLOAD_PROVIDER`, `VISUAL_SEARCH_PROVIDER`, and `BROWSE_FETCH_PROVIDER` explicitly; a selected provider never falls through to another. The disk tool cache is disabled by default and, when enabled, is bounded by `TOOL_CACHE_TTL_SECONDS` and `TOOL_CACHE_NAMESPACE`.

Credentials belong only in environment variables or the untracked `.env`. Sanitize secret fields and signed URLs before any trace, HTML, or cache persistence. JSON and sibling HTML export are required; HTML rendering/write errors propagate. Dedicated face detection, embeddings, and biometric matching are absent; person-identity claims are investigated with public-source and non-biometric evidence.

Before finishing orchestration changes, run:

```powershell
python -m pytest -q test_unit.py test_native_interactions.py test_gemini_interactions_contract.py
python -m pytest -q test_full_native_agent_trace.py
python test_workflow_smoke.py
python scripts/probe_gemini_interactions.py --model gemini-3-flash-preview
```

Normal runs write canonical JSON traces and sibling standalone HTML diagnostics under `outputs/traces/`. Use `python -m src.render_trace_html outputs\traces` to regenerate HTML from saved JSON.
