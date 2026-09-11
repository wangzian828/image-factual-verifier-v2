"""Re-execute one real failed visual-tool call with PSD's bound service config."""
from __future__ import annotations
import argparse
import asyncio
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "training")]
from ifv_training.io import load_json, sha256_file, write_json
from scripts.run_psd_repair_driver import _policy_runtime_kwargs
from src.orchestrator.pipeline import Orchestrator
from src.orchestrator.source_access import SourceAccessPolicy
from src.orchestrator.react_runtime import RuntimeToolAdapter, UnifiedReactState
from src.orchestrator.runtime_events import CaseRuntimeStore, bind_case_runtime_store, reset_case_runtime_store


def bound_visual_tool(orchestrator, episode):
    digest = sha256_file(Path(episode["image_path"]))
    if digest != episode["state"]["runtime_case"]["image_sha256"]:
        raise ValueError("visual probe image differs from original runtime case")
    return RuntimeToolAdapter(delegate=orchestrator.all_tools["focused_visual_inspection"],
        state=UnifiedReactState(case_id=episode["case_id"], image_sha256=digest),
        image_path=episode["image_path"])


async def run(args):
    if args.output.exists():
        raise ValueError("visual diagnostic output must be new; do not silently resample")
    episode, profile = load_json(args.episode), load_json(args.serving_profile)
    rows = [r for r in episode["state"]["all_steps"] if r.get("tool_name") == "focused_visual_inspection"]
    if not rows:
        raise ValueError("episode has no real focused visual tool call")
    params = rows[-1]["tool_args"]
    role = SimpleNamespace(policy_provider="qwen_local", policy_model=profile["profile_id"],
                           policy_wire_api=profile["wire_api"])
    orchestrator = Orchestrator(**_policy_runtime_kwargs(role, profile["base_url"],
                                SourceAccessPolicy.load(args.source_access_policy)))
    store = CaseRuntimeStore(args.output, case_id=episode["case_id"], attempt_id="visual-endpoint-smoke")
    token = bind_case_runtime_store(store)
    try:
        write_json(args.output / "inputs.json", {"episode_sha256": sha256_file(args.episode),
            "serving_profile_sha256": sha256_file(args.serving_profile), "tool_args": params})
        # Canonical tool_args are the PUBLIC schema. Use the same existing
        # runtime adapter to bind image_input/internal fields, not a raw delegate.
        tool = bound_visual_tool(orchestrator, episode)
        result = await asyncio.to_thread(tool.call, params)
        write_json(args.output / "tool-result.json", result)
        summary = {"passed": result.get("status") == "success", "status": result.get("status"),
                   "answer_status": result.get("answer_status"), "runtime_archive": str(store.root),
                   "scope": "real_visual_tool_connection_and_schema_not_repair_quality"}
        write_json(args.output / "result.json", summary)
        return summary
    finally:
        reset_case_runtime_store(token)
        await orchestrator.aclose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--serving-profile", type=Path, required=True)
    parser.add_argument("--source-access-policy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    result = asyncio.run(run(parser.parse_args()))
    print(json.dumps(result))
    raise SystemExit(0 if result["passed"] else 1)
