"""Explicit grouped sampling and upstream failed-episode selection for PSD."""
from collections import defaultdict
from types import SimpleNamespace

from src.eval.rollout import rollout_specs


def require_token_capture_environment(environ):
    if str(environ.get("IFV_CAPTURE_POLICY_TOKENS", "")).strip().lower() not in {"1", "true", "yes", "on"}:
        raise ValueError("PSD source collection requires native policy token/logprob capture before dispatch")
    if str(environ.get("IFV_POLICY_TOPK", "20")).strip() != "20":
        raise ValueError("PSD source collection requires top-20 policy capture")


def verify_collection(rows, *, case_ids, manifest, expected_rollouts):
    """Count slots, not just case IDs; do not accept 400 rows as 400 x 8."""
    if type(expected_rollouts) is not int or expected_rollouts < 1:
        raise ValueError("PSD rollouts per case must be a positive integer")
    if not case_ids or len(set(case_ids)) != len(case_ids):
        raise ValueError("PSD collection case IDs must be nonempty and unique")
    agent = manifest.get("agent", {})
    recovery = agent.get("psd_sampling", {}).get("infrastructure_retry")
    if recovery is not None and (recovery.get("unresolved") or recovery.get("slots") != len(rows)
                                 or recovery.get("selection_by_answer") is not False):
        raise ValueError("PSD collection has unresolved or unbound infrastructure sampling slots")
    if agent.get("rollouts_per_case") != expected_rollouts:
        raise ValueError("PSD collection sampling budget differs from requested group size")
    if expected_rollouts == 8 and (agent.get("psd_sampling", {}).get("temperature") != .7
            or not agent.get("psd_sampling", {}).get("collector_sha256")):
        raise ValueError("PSD collection lacks explicit published sampling provenance")
    if type(agent.get("base_sampling_seed")) is not int or not manifest.get("git_commit") or not agent.get("model"):
        raise ValueError("PSD collection lacks sampling/policy identity")
    expected = rollout_specs([{} for _ in case_ids], [SimpleNamespace(case_id=c) for c in case_ids],
        rollouts_per_case=expected_rollouts, base_sampling_seed=agent["base_sampling_seed"],
        policy_revision=manifest["git_commit"], model=agent["model"],
        episode_namespace=agent.get("episode_namespace"))
    # The native run_result_record serializes slot identity but not group_size.
    # Group size remains bound by the manifest, hashed prompt_group_id, and the
    # exact complete slot set below; do not rewrite historical result rows.
    keys = ("case_id", "episode_id", "prompt_group_id", "rollout_index", "sampling_seed")
    expected_by_id = {r["episode_id"]: r for r in expected}
    if len(expected_by_id) != len(expected):
        raise ValueError("PSD generated episode ID collision")
    seen = set()
    for row in rows:
        episode = row.get("episode_id")
        if episode in seen or episode not in expected_by_id:
            raise ValueError("PSD collection contains duplicate or unexpected episodes")
        if any(row.get(k) != expected_by_id[episode][k] for k in keys):
            raise ValueError("PSD collection slot seed/index/policy binding mismatch")
        if "group_size" in row and row["group_size"] != expected_rollouts:
            raise ValueError("PSD collection explicit group size mismatch")
        seen.add(episode)
    if seen != set(expected_by_id):
        raise ValueError("PSD collection is incomplete; missing sampling slots")
    return {"passed": True, "cases": len(case_ids), "episodes": len(rows),
        "rollouts_per_case": expected_rollouts, "selection_by_success": False,
        "group_size_attestation": "manifest_and_hashed_group_and_complete_slot_set",
        "rows_without_redundant_group_size": sum("group_size" not in row for row in rows)}


def select_task_repair_sources(candidates, *, load_trace):
    """Upstream load_collection_rollout defaults to the longest failed episode.

    Keep every input in the source bank, but spend one task's repair search
    budget on one source episode. Preservation collection is not filtered here.
    """
    groups = defaultdict(list)
    for row in candidates:
        if row.get("class") != "repair_seed" or not row.get("case_id") or not row.get("episode_id"):
            raise ValueError("invalid PSD repair seed")
        groups[row["case_id"]].append(row)
    selected, records = [], []
    for case, group in sorted(groups.items()):
        if len({r["episode_id"] for r in group}) != len(group):
            raise ValueError("duplicate repair source episode")
        ranked = []
        for row in group:
            trace = load_trace(row)
            steps = trace.get("state", {}).get("all_steps")
            if not isinstance(steps, list) or not steps:
                raise ValueError("repair source lacks recorded steps")
            ranked.append((len(steps), row))
        ranked.sort(key=lambda pair: (-pair[0], pair[1]["episode_id"]))
        count, chosen = ranked[0]
        selected.append(chosen)
        records.append({"case_id": case, "selected_episode_id": chosen["episode_id"],
            "source_steps": count, "failed_episodes": len(group),
            "unselected_episode_ids": [r["episode_id"] for _, r in ranked[1:]]})
    return selected, {"policy": "longest_failed_episode_per_task; episode_id_tie_break",
        "original_candidates": len(candidates), "selected_tasks": len(selected), "tasks": records}
