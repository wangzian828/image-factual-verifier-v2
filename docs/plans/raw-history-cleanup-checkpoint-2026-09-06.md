# Raw-history cleanup checkpoint

Date: 2026-09-06
Branch: main
Base commit: 7aa74cb

## Active production path

The active image-only path is:

```text
pipeline.py -> react_runtime.py -> stage_runner.py
            -> provider-side cumulative interaction history
            -> raw observations in state.all_steps
            -> unified Judgment
```

The production path does not directly import any of the legacy reducer modules:

- `unified_react.py`
- `task_store.py`
- `progress_control.py`
- `discrepancy_coverage.py`
- `context_workspace.py`
- `unified_context.py`

The active runtime state is mechanical only. Observation truth is read from raw
action/result rows in `state.all_steps`; no graph state, reducer delta, or stage
handoff packet is added back.

## Legacy dependency boundary

The legacy modules still import one another:

- `unified_react.py` imports `progress_control.py` and `task_store.py`.
- `context_workspace.py` imports `task_store.py`.
- `discrepancy_coverage.py` imports `task_store.py`.
- `unified_context.py` imports `task_store.py`.
- `progress_control.py` imports `task_store.py`.

External consumers are historical only:

- Tests: old runtime, bootstrap/state machine, workspace, v4 replay/planning,
  reducer, and failure-contract tests.
- Scripts: `replay_snapshot_discrepancy.py`, `backfill_fact_check_reports.py`,
  and `build_reviewed52_replay_manifest.py`.
- Documentation and reports: historical graph/reducer design and experiment
  records.

Do not delete these modules in this checkpoint. First move or explicitly mark
historical replay/backfill tools as archive-only, then remove unreachable code
in a separate change. Do not add compatibility imports to the active runtime.

## Remaining stale code

- `scripts/audit_real_trace.py` still contains unreachable v3/v4 audit helpers;
  `audit_trace()` dispatches only to the current raw-history audit.
- `src/trajectory/exporter.py` still contains old v4 chain helpers; the active
  raw-history quality gate is separate.
- `src/trajectory/reference_chain.py` still uses generic matching helpers from
  scoring as an evaluation side path.
- `StageRunner` retains `evidence_so_far` text for a non-Interactions provider
  fallback; verify that it never becomes canonical state.
- Viewer/readable rendering retains compatibility display for old traces, while
  the raw-history branch does not read old graph state.

## Checkpoint changes

- Migrated `test_eval_artifacts.py` to a raw-history fixture with a successful
  `perceive_scene` action, `function_call_id`, judgment output, and aligned basis.
- Migrated `test_trajectory_media_projection.py` to raw-history state and raw
  tool fields.

## Validation

Passed:

```text
python -m pytest -q test_eval_artifacts.py test_trajectory_media_projection.py
13 passed

python -m pytest -q test_sft_eligibility.py test_unified_react_scoring.py test_workflow_lifecycle.py test_eval_artifacts.py test_trajectory_media_projection.py
40 passed

python -m compileall -q src scripts training/ifv_training
```

The retired v3/v4 test files and fixtures were removed after the raw-history
runtime became the only supported entry. Current strict-audit coverage is in
`test_raw_history_audit.py`; do not restore reducer compatibility merely to
make historical tests collect.

## Next work

1. Review remaining archive-only replay/backfill scripts before any source cleanup.
2. Remove unreachable v3/v4 audit and exporter branches only after dependency review.
3. Build the server one-click teacher rollout, audit, SFT, and dual-package path.
4. Run and audit ten real training traces on the server.
5. Package the local Direct QA comparison experiment separately.
