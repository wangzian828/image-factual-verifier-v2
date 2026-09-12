import hashlib
import json
import pytest
from scripts.run_local_direct_qa import load_cases, payload, DEFAULT_PROMPT


def test_local_qa_sends_only_public_image_and_shared_prompt(tmp_path):
    image = tmp_path / "image.png"
    image.write_bytes(b"diagnostic")
    case = {"case_id": "a", "image_path": str(image), "gold": "never disclose"}
    request = payload(case, "local", 4096)
    assert "never disclose" not in json.dumps(request)
    assert request["messages"][0]["content"][1]["text"] == DEFAULT_PROMPT
    assert request["chat_template_kwargs"] == {"enable_thinking": False}
    assert request["model"] == "local"


def test_local_qa_requires_hash_bound_public_projection(tmp_path):
    image = tmp_path / "image.png"
    image.write_bytes(b"diagnostic")
    row = {"case_id": "a", "image_path": "image.png",
        "image_sha256": hashlib.sha256(image.read_bytes()).hexdigest()}
    manifest = tmp_path / "cases.jsonl"
    manifest.write_text(json.dumps(row))
    assert load_cases(manifest)[0]["image_path"] == str(image)
    manifest.write_text(json.dumps({**row, "label": "fake"}))
    with pytest.raises(ValueError, match="public"):
        load_cases(manifest)
