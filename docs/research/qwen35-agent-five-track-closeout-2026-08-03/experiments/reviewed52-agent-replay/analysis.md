# H5 analysis: reviewed-52 frozen replay

Status: confirmed for Qwen and provider-independent failure semantics.

The scanner reviewed 52 cases and 975 historical snapshots, finding 976 candidate
Evidence records and nine strictly qualified source-to-pixel cases.

Source-hint fixes produce concise properties for:

- L'Hoest orange facial patch and white tail;
- sarcophagus clothing/rosary;
- mummy-mouth object;
- White-Faced Saki and six-finger attributes;
- white robe/red sash;
- Diana glove status;
- Sail4th sky colors.

The seed-11 Qwen batch at commit `0e45675` completed:

- 9/9 subprocesses;
- 9/9 focused visual requests and resolved pixel Evidence records;
- 9/9 second Decisions consuming their resolved pixel Evidence;
- zero missing visual consumption;
- zero engineering failures;
- zero deterministic fallback use;
- two terminal `fake` composites in this rollout.

Terminality varied relative to an earlier run, but the locked mechanism did not:
resolved pixel Evidence was never silently dropped and tool failure was never
converted into source support. Nonterminal cases retained explicit
insufficient/supporting/conflicted rationale.

Fresh Gemini online replay was unavailable because no Gemini credential is
configured. Gemini exhaustion semantics remain covered by provider-independent
runtime tests and historical failure replays.

Artifacts:

- manifest:
  `/gsdata/home/wza/image-factual-verifier-v4/runs/replays/five-track-0e45675-20260803/reviewed52-candidate-manifest.json`
- batch:
  `/gsdata/home/wza/image-factual-verifier-v4/runs/replays/five-track-0e45675-qwen-seed11-20260803/`

