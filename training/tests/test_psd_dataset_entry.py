from types import SimpleNamespace
import pytest
from ifv_training.psd_ms_swift import PsdDatasetPreprocessor


@pytest.mark.parametrize("fields", [("target_tokens", "weights"),
                                    ("sparse_target_tokens", "sparse_weights")])
def test_psd_native_loader_preserves_preencoded_arrays_without_messages(fields):
    dataset = SimpleNamespace(features={key: None for key in
        ("schema_version", "input_ids", "topk", "loss_positions", *fields)})
    assert PsdDatasetPreprocessor()(dataset, strict=True) is dataset
    assert "messages" not in dataset.features


def test_psd_native_loader_does_not_accept_normal_chat_dataset():
    with pytest.raises(ValueError, match="pre-tokenized"):
        PsdDatasetPreprocessor()(SimpleNamespace(features={"messages": None}))
