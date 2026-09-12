import pytest
from ifv_training.psd_repair_search import validate_live_model


def test_vllm_adapter_card_inherits_context_from_exact_parent(tmp_path):
    profile = {"profile_id": "policy", "model_path": str(tmp_path / "adapter"),
        "engine_model_path": str(tmp_path / "base"), "context_length": 131072}
    adapter = {"id": "policy", "root": profile["model_path"], "parent": "base-alias"}
    base = {"id": "base-alias", "root": profile["engine_model_path"], "max_model_len": 131072}
    assert validate_live_model(profile, {"data": [base, adapter]})["max_model_len"] == 131072
    for changed in ({**base, "root": str(tmp_path / "other-base")}, {**base, "max_model_len": 65536}):
        with pytest.raises(ValueError):
            validate_live_model(profile, {"data": [changed, adapter]})
    with pytest.raises(ValueError):
        validate_live_model(profile, {"data": [base, {**adapter, "root": base["root"]}]})
