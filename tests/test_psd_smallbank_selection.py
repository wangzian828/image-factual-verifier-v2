import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/server/materialize_psd_smallbank_selection.py"
spec = importlib.util.spec_from_file_location("materialize_psd_smallbank_selection", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_review_source_model_is_read_from_its_actual_receipt(tmp_path):
    receipt = tmp_path / "review.json"
    receipt.write_text(json.dumps({"payload": {"verifier": {"model": "gemini-3.7-flash"}}}))
    assert module._review_model({"source_task_review": {"path": str(receipt)}}) == "gemini-3.7-flash"
    receipt.write_text(json.dumps({"payload": {"verifier": {"model": "qwen"}}}))
    with pytest.raises(ValueError, match="Gemini"):
        module._review_model({"source_task_review": {"path": str(receipt)}})


def test_stat_identity_changes_on_mutation(tmp_path):
    source = tmp_path / "rows.jsonl"
    source.write_text("one\n")
    before = module._stat(source)
    source.write_text("one\ntwo\n")
    assert module._stat(source) != before
