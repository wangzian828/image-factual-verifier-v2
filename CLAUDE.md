# Claude Project Instructions

Follow `AGENTS.md` and `docs/architecture.md`.

This repository supports only v0.3 image-only input:

```text
case_id + image_path + image_sha256
```

The runtime is:

```text
Gemini perception + EasyOCR
-> deterministic VisualFact/task bootstrap
-> native Interactions ReAct
-> Reflection every four real actions
-> decisive-fact Coverage
-> reinspect-v2 Judgment
```

Do not restore claim modes, claim-ledger planning, fixed replanning, or
`reinspect-v1`. `reinspect-v2` is the current v3 verdict-policy identifier.

Evidence must be grounded in successful tool calls. Discovery is never Evidence.
Every verdict basis must trace through `VisualFact -> Finding -> Evidence -> successful
tool call`. Provider, protocol, malformed-output, all-tools-failed, and configuration
failures are engineering errors and must not become `unverifiable`.

Gold and source-access exclusions are evaluator-private. Gold is loaded only after all
rollouts; blocked sources are filtered before model-visible results.

Use Gemini Interactions end to end, one accepted tool call per action turn, and keep
credentials environment-only. Canonical JSON is the source trace; HTML is derived.

Before finishing runtime changes:

```powershell
python -m pytest -q
python -m compileall -q src scripts
git diff --check
```

Only a no-mock real canary plus strict trace audit proves live acceptance.
