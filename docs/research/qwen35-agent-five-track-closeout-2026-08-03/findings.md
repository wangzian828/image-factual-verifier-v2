# Five-track closeout findings

## Research question

Can the OMP1 production path, image preprocessing, scheduler/checkpoint behavior,
and reviewed-52 Agent mechanism be made production-safe without changing training
or verdict semantics?

## Current understanding

- OMP1 is mandatory server policy. Historical OMP8 speed results remain evidence,
  but the exact OMP1 workers0 production combination needs a matched gate.
- The ms-swift Arrow cache stores normalized rows, image references, and lengths;
  image open/decode/template processing remains lazy.
- Two-GPU ZeRO-3 resume is correct but exceeds 200 GiB process-tree RSS and writes
  approximately 132 GB per full checkpoint.
- Frozen replay already has deterministic guards for resolved pixel-Evidence
  consumption, but source-property extraction and visual-tool failure semantics
  still need broader regression evidence.

## Key results

Pending protocol execution.

## Patterns and insights

Pending.

## Lessons and constraints

- Do not use another user's GPU process or stop it.
- Do not mutate historical OMP8 profiles or artifacts.
- Do not weaken optimizer/scheduler/RNG recovery to claim checkpoint speed.
- Do not use free-running search to validate the source-to-pixel mechanism.
- Do not revive or stage the user's deleted research workspace.

## Open questions

- Does workers4 beat workers0 after OMP is restricted to one?
- Which exact ms-swift boundary can accept deterministic cached processor outputs?
- Is the scheduler warning semantic or cosmetic?
- Can model-weight gathering be separated from resumable state checkpoints safely?
- Which reviewed-52 cases still fail after deterministic source and visual guards?

## Optimization trajectory

Pending.
