"""Pytest boundary for the current unified-react-v1 runtime.

These modules assert retired v3/v4 orchestration contracts (standalone
Planning, query/route Replan, core-fact Coverage, or the old trace schema).
They remain in the repository as historical reference material, but are not
part of the default regression gate for the active runtime.
"""

collect_ignore = [
    "test_audit_real_trace.py",
    "test_image_only_bootstrap.py",
    "test_image_only_state_machine.py",
    "test_image_only_trajectory.py",
    "test_reference_chain_scoring.py",
    "test_runtime_failure_contracts.py",
    "test_separate_vlm_mode.py",
    "test_trace_viewer.py",
    "test_trajectory_dataset.py",
    "test_trajectory_export.py",
    "test_frozen_v3_runtime.py",
    "test_prompt_boundaries.py",
    "test_tool_contract_repairs.py",
    "test_v4_historical_replay.py",
    "test_v4_planning.py",
    "test_v4_reducers.py",
    "test_react_runtime.py",
    "test_unified_react.py",
    "test_unified_react_accounting.py",
    "test_context_workspace.py",
    "test_progress_control.py",
    "test_agent_reference_candidate_recovery.py",
]
