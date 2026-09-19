# Lightweight PSD production pipeline

This is the execution contract for runs after the current small-bank closure.
The objective is to spend wall time on Qwen inference and Gemini calls, not on
re-reading, hashing, or duplicating trajectory payloads.

## Storage contract

- Each rollout has one authoritative runtime trace and one compact terminal
  record.  Do not create duplicate `episode.json` or `continuation.json` copies.
- Candidate, attempt, review, and merge ledgers are append-only gzip JSONL.
- Large artifacts are not hashed by pipeline stages.  They use the upstream
  run identity plus an O(1) inode/size/mtime/ctime receipt.  Identity changes
  are hard failures; they are never repaired by scanning or re-hashing.
- Images in a runtime release are hardlinks to the frozen source pool.  Do not
  copy or hash image bytes.
- Preservation admission is one source-ordered verified success per case.
  Deduplication happens before target construction, not as a cleanup pass.
- Repair finalization consumes only terminal manifests and accepted compact
  records.  It must not replay every historical attempt or scan completed
  trajectories.

## Parallel schedule

While Qwen collection is running, sealed case groups may flow through Gemini
source review, CPU postprocessing/candidate extraction, and Gemini repair
proposal.  These stages use bounded queues and never wait for the full 4,000
rollouts.  Qwen repair rollouts do not compete with the main collection; they
start when collection releases Qwen capacity.  Gemini repair checking overlaps
Qwen repair of other cases.

The final bank is frozen only after the selected collection and repair budgets
are terminal.  Teacher top-k then owns all four GPUs, followed by streaming
datum packing, the DP4 training gate, five-epoch training, and fixed diagnostics.
Teacher scoring cannot safely start on mutable targets and is therefore the
only intentionally non-overlapped major stage.

No controller uses fixed sleep intervals or an outer polling-count stop.  It
wakes on ledger growth or child exit, processes only pending IDs, and records
an explicit terminal reason for every selected case.

## Dependency order

`collect → source_review → postprocess → candidate_extract → repair_propose →
repair_rollout → repair_check → freeze_bank → teacher_topk → datum_pack →
train_gate → train_5epoch → verify_training`

The first seven stages are per-case streaming stages.  The last six are frozen
whole-bank stages.  `training/ifv_training/psd_pipeline_plan.py` is the
machine-readable resource/dependency policy used by the production controller.
