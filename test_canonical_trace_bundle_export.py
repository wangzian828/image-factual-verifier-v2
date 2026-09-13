from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from scripts.trajectory.export_canonical_trace_bundle import (
    _delivery_checksums,
    _verify_delivery_file,
    bind_reference_media,
)
from src.trajectory.exporter import _trajectory_candidate_steps


def _media(root: Path, content: bytes) -> str:
    digest = hashlib.sha256(content).hexdigest()
    path = root / "images" / f"{digest}.jpg"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return f"images/{path.name}"


def _messages() -> list[dict[str, str]]:
    return [
        {"role": "system", "content": "generated system"},
        {"role": "user", "content": "<image>\ngenerated user"},
        {"role": "assistant", "content": "<think>generated</think>"},
        {"role": "tool_call", "content": "generated call"},
        {"role": "tool_response", "content": "generated response"},
        {"role": "assistant", "content": "<answer>generated</answer>"},
    ]


def _reference(images: list[str]) -> dict:
    return {
        "roles": [message["role"] for message in _messages()],
        "marker_counts": [0, 1, 0, 0, 2, 0],
        "images": images,
    }


def test_media_binding_copies_only_verified_layout_and_paths(tmp_path: Path) -> None:
    images = [_media(tmp_path, value) for value in (b"one", b"two", b"three")]
    generated = _messages()

    messages, paths, digests = bind_reference_media(
        generated,
        _reference(images),
        delivery_root=tmp_path,
        digest_cache={},
    )

    assert messages[0]["content"] == "generated system"
    assert messages[3]["content"] == "generated call"
    assert messages[4]["content"].startswith("generated response")
    assert messages[4]["content"].count("<image>") == 2
    assert sum(message["content"].count("<image>") for message in messages) == 3
    assert all(Path(path).is_file() for path in paths)
    assert [Path(path).stem for path in paths] == digests
    assert generated[4]["content"] == "generated response"


def test_media_binding_rejects_role_or_marker_drift(tmp_path: Path) -> None:
    images = [_media(tmp_path, value) for value in (b"one", b"two", b"three")]
    wrong_role = _reference(images)
    wrong_role["roles"][4] = "assistant"
    with pytest.raises(ValueError, match="role sequence"):
        bind_reference_media(
            _messages(), wrong_role, delivery_root=tmp_path, digest_cache={}
        )

    wrong_marker = _reference(images)
    wrong_marker["marker_counts"] = [0, 1, 0, 1, 1, 0]
    with pytest.raises(ValueError, match="tool response"):
        bind_reference_media(
            _messages(), wrong_marker, delivery_root=tmp_path, digest_cache={}
        )


def test_media_binding_rejects_unverified_or_escaping_media(tmp_path: Path) -> None:
    valid = [_media(tmp_path, value) for value in (b"one", b"two")]
    outside = tmp_path / "outside.jpg"
    outside.write_bytes(b"three")
    reference = _reference([*valid, "../outside.jpg"])
    with pytest.raises(ValueError, match="escapes or is missing"):
        bind_reference_media(
            _messages(), reference, delivery_root=tmp_path, digest_cache={}
        )

    bad = tmp_path / "images" / ("0" * 64 + ".jpg")
    bad.write_bytes(b"not-zero-hash")
    reference = _reference([*valid, f"images/{bad.name}"])
    with pytest.raises(ValueError, match="filename/hash mismatch"):
        bind_reference_media(
            _messages(), reference, delivery_root=tmp_path, digest_cache={}
        )


def test_delivery_level_checksum_is_authoritative_for_published_bytes(
    tmp_path: Path,
) -> None:
    path = tmp_path / "ms-swift-policy" / "train.jsonl"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"published relocated bytes\n")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    (tmp_path / "SHA256SUMS").write_text(
        f"{digest}  ms-swift-policy/train.jsonl\n",
        encoding="utf-8",
    )

    checksums = _delivery_checksums(tmp_path)

    assert (
        _verify_delivery_file(
            tmp_path,
            "ms-swift-policy/train.jsonl",
            checksums,
        )
        == digest
    )


def test_delivery_level_checksum_rejects_modified_bytes(tmp_path: Path) -> None:
    path = tmp_path / "SOURCE_MANIFEST.json"
    path.write_bytes(b"original")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    (tmp_path / "SHA256SUMS").write_text(
        f"{digest}  SOURCE_MANIFEST.json\n",
        encoding="utf-8",
    )
    path.write_bytes(b"modified")

    with pytest.raises(ValueError, match="delivery-level checksum mismatch"):
        _verify_delivery_file(
            tmp_path,
            "SOURCE_MANIFEST.json",
            _delivery_checksums(tmp_path),
        )


@pytest.mark.parametrize("parent_action_type", ["output_rejected", "format_error"])
def test_forced_judgment_correction_recovers_parent_policy_input(
    parent_action_type: str,
) -> None:
    rejected_input = {
        "system_instruction": "Return the final judgment.",
        "input_payload": [{"role": "user", "content": "history"}],
        "response_format": {"type": "json_schema"},
    }
    corrected = {
        "verdict": "real",
        "confidence": 0.8,
        "verdict_observation_ids": ["call-1"],
        "overall_assessment": "The evidence supports the claim.",
        "fact_check_report": {"headline": "Supported"},
    }
    trace = {
        "judgment": {**corrected, "policy_rule_id": "unified-react-v1"},
        "state": {
            "all_steps": [
                {
                    "stage": "unified_judgment",
                    "action_type": parent_action_type,
                    "output": {"verdict": "maybe"},
                    "metadata": {
                        "context_request_id": "req-final-1",
                        "policy_input": rejected_input,
                        "policy_action": {"verdict": "maybe"},
                    },
                },
                {
                    "stage": "unified_judgment",
                    "action_type": "output",
                    "output": corrected,
                    "metadata": {
                        "forced_output": True,
                        "interaction_lifecycle_kind": "protocol_correction",
                        "parent_context_request_id": "req-final-1",
                    },
                },
            ]
        },
    }

    candidates = _trajectory_candidate_steps(trace)

    assert len(candidates) == 1
    _, step, example_type = candidates[0]
    assert example_type == "judgment"
    assert step["metadata"]["policy_input"] == rejected_input
    assert step["metadata"]["policy_action"] == corrected
    assert (
        step["metadata"]["sft_policy_input_provenance"]
        == "rejected_parent_request"
    )
    assert "policy_input" not in trace["state"]["all_steps"][1]["metadata"]


def test_forced_judgment_correction_rejects_unbound_output() -> None:
    corrected = {
        "verdict": "real",
        "confidence": 0.8,
        "verdict_observation_ids": ["call-1"],
        "overall_assessment": "The evidence supports the claim.",
        "fact_check_report": {"headline": "Supported"},
    }
    trace = {
        "judgment": {**corrected, "verdict": "fake"},
        "state": {
            "all_steps": [
                {
                    "stage": "unified_judgment",
                    "action_type": "output",
                    "output": corrected,
                    "metadata": {
                        "forced_output": True,
                        "interaction_lifecycle_kind": "protocol_correction",
                        "parent_context_request_id": "missing-parent",
                    },
                }
            ]
        },
    }

    assert _trajectory_candidate_steps(trace) == []
