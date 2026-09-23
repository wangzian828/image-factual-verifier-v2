"""Index verified smallbank preservation episodes without copying target payloads.

Only the existing assembled preservation file is read. Completion token IDs are
decoded with the frozen SFT3 tokenizer to audit native tool coverage, but no
model weights, image pixels, teacher scores, or private reviews are read.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any


def _stat(path: Path) -> dict[str, int]:
    value = path.stat()
    return {"device": value.st_dev, "inode": value.st_ino, "bytes": value.st_size,
            "mtime_ns": value.st_mtime_ns}


def _tools(step: dict[str, Any], tokenizer: Any) -> list[str]:
    raw = tokenizer.decode(step["completion_ids"], skip_special_tokens=False)
    if "</think>" not in raw:
        raise ValueError("react completion has no closed thinking span")
    value = raw.rsplit("</think>", 1)[1].split("<|im_end|>", 1)[0].strip()
    actions = json.loads(value)
    if not isinstance(actions, list) or not actions:
        raise ValueError("react completion lacks native actions")
    names = [action.get("name") for action in actions if isinstance(action, dict)]
    if len(names) != len(actions) or any(not isinstance(name, str) or not name for name in names):
        raise ValueError("react completion has invalid native action names")
    return names


def _roles(row: dict[str, Any], model: str, checkpoint_sha: str, checkpoint: str) -> None:
    roles = row.get("model_roles")
    if not isinstance(roles, dict):
        raise ValueError("preservation episode lacks model roles")
    for name, path_key in (("frozen_self_teacher", "round_start_checkpoint"),
                           ("trainable_student", "initial_checkpoint")):
        role = roles.get(name)
        if (not isinstance(role, dict) or role.get("provider") != "qwen_local"
                or role.get("model") != model
                or role.get("checkpoint_manifest_sha256") != checkpoint_sha
                or role.get(path_key) != checkpoint):
            raise ValueError("preservation episode has mixed Qwen/checkpoint identity")


def index_preservation(*, source: Path, output_dir: Path, tokenizer: Any,
                       model: str, checkpoint_sha: str, checkpoint: str,
                       expected_episodes: int, expected_steps: int) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    before = _stat(source)
    rows: list[dict[str, Any]] = []
    seen_cases: set[str] = set()
    tool_counts: Counter[str] = Counter()
    total_steps = 0
    with source.open("rb") as handle:
        while True:
            offset = handle.tell()
            line = handle.readline()
            if not line:
                break
            episode = json.loads(line)
            case_id, episode_id = episode.get("case_id"), episode.get("episode_id")
            if not isinstance(case_id, str) or not case_id or case_id in seen_cases:
                raise ValueError("duplicate or missing preservation case")
            if not isinstance(episode_id, str) or not episode_id:
                raise ValueError("missing preservation episode ID")
            seen_cases.add(case_id)
            if episode.get("verified_full_task") is not True or episode.get("strict_trace_audit_pass") is not True:
                raise ValueError("non-verified preservation episode")
            _roles(episode, model, checkpoint_sha, checkpoint)
            steps = episode.get("preservation_steps")
            if not isinstance(steps, list) or not steps:
                raise ValueError("empty preservation episode")
            ids: set[str] = set()
            tools: list[str] = []
            image_steps = 0
            judgments = 0
            max_prompt = 0
            completion_tokens = 0
            for step in steps:
                step_id = step.get("step_id")
                if not isinstance(step_id, str) or not step_id or step_id in ids:
                    raise ValueError("missing/duplicate preservation step")
                ids.add(step_id)
                prompt, completion = step.get("student_prompt_ids"), step.get("completion_ids")
                if not isinstance(prompt, list) or not isinstance(completion, list) or not completion:
                    raise ValueError("preservation step lacks exact token IDs")
                if any(type(value) is not int for value in prompt + completion):
                    raise ValueError("non-integer preservation token IDs")
                max_prompt = max(max_prompt, len(prompt))
                completion_tokens += len(completion)
                media = step.get("psd_media") or {}
                if not isinstance(media, dict):
                    raise ValueError("malformed preservation media")
                images = media.get("image_paths") or []
                if not isinstance(images, list):
                    raise ValueError("malformed preservation image paths")
                if 248056 in prompt and not images:
                    raise ValueError("image-bearing preservation step lacks pixel binding")
                image_steps += bool(images)
                stage = step.get("stage")
                if stage == "unified_react":
                    names = _tools(step, tokenizer)
                    tools.extend(names)
                    tool_counts.update(names)
                elif stage == "unified_judgment":
                    judgments += 1
                else:
                    raise ValueError("unknown preservation stage")
            if judgments != 1 or not tools:
                raise ValueError("preservation episode lacks actions or final judgment")
            total_steps += len(steps)
            rows.append({"case_id": case_id, "episode_id": episode_id,
                         "source_offset": offset, "source_length": len(line),
                         "step_count": len(steps), "native_action_count": len(tools),
                         "tool_sequence": tools, "image_steps": image_steps,
                         "max_prompt_tokens": max_prompt,
                         "completion_tokens": completion_tokens,
                         "verified_full_task": True, "strict_trace_audit_pass": True})
    if _stat(source) != before:
        raise ValueError("assembled preservation source changed during indexing")
    if len(rows) != expected_episodes or total_steps != expected_steps:
        raise ValueError("assembled preservation counts disagree with frozen manifest")
    output_dir.mkdir(parents=True, exist_ok=False)
    with (output_dir / "episodes.jsonl").open("x", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, sort_keys=True) + "\n")
    manifest = {"schema_version": "ifv-psd-smallbank-preservation-metadata-index-v1",
                "status": "metadata_only_for_whole_episode_selection",
                "source": str(source.resolve()), "source_identity": before,
                "episodes": len(rows), "steps": total_steps,
                "native_tool_counts": dict(sorted(tool_counts.items())),
                "qwen_model": model, "checkpoint_manifest_sha256": checkpoint_sha,
                "no_payload_copy": True, "no_image_hashing": True,
                "formal_training_started": False}
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    from transformers import AutoTokenizer
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-sha", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--expected-episodes", type=int, required=True)
    parser.add_argument("--expected-steps", type=int, required=True)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True, trust_remote_code=True)
    print(json.dumps(index_preservation(source=args.source, output_dir=args.output_dir,
        tokenizer=tokenizer, model=args.model, checkpoint_sha=args.checkpoint_sha,
        checkpoint=args.checkpoint, expected_episodes=args.expected_episodes,
        expected_steps=args.expected_steps), ensure_ascii=False))


if __name__ == "__main__":
    main()
