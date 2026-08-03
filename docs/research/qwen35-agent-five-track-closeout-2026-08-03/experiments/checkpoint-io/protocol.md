# H4 protocol: checkpoint save/resume I/O

Status: locked before execution.

## Prediction

Most bytes are irreducible Adam state, but model-weight gathering, output
placement, and duplicate checkpoint classes may be separable. At minimum, timing
and storage preflight can make the cost explicit and prevent partial writes.

## Candidate interventions

- time model gather, optimizer-state write, finalization, and resume load
  separately;
- inspect local fast storage without moving authoritative state unsafely;
- evaluate state-only intermediate checkpoints plus explicit serving export;
- retain `save_total_limit=1` and atomic manifest completion.

## Acceptance

Any promoted path must resume optimizer, scheduler, RNG, and global step and still
support a serving-compatible model export. A weights-only checkpoint cannot replace
the recovery checkpoint.
