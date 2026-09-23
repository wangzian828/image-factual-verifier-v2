import gzip
import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/server/materialize_psd_old1000_selection.py"
spec = importlib.util.spec_from_file_location("materialize_psd_old1000_selection", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_serving_topk_is_never_accepted_as_raw_teacher():
    target = {"target_status": "complete", "teacher_topk_by_position": [[[1, 1.0]]],
              "teacher_logprob_semantics": "serving_masked"}
    assert module._pending(target) == {"target_status": "pending_topk",
        "teacher_topk_by_position": [], "teacher_logprob_semantics": ""}


def test_one_canonical_gzip_case_and_no_overwrite(tmp_path):
    path = tmp_path / "repair" / "main-00001.jsonl.gz"
    module._write_case(path, [{"target_id": "a"}, {"target_id": "b"}])
    with gzip.open(path, "rt") as source:
        assert [json.loads(line)["target_id"] for line in source] == ["a", "b"]
    with pytest.raises(FileExistsError):
        module._write_case(path, [{"target_id": "c"}])
