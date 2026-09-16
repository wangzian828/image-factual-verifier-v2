import json
from types import SimpleNamespace
import pytest


def test_observer_preserves_finite_values_and_records_first_bad_layer(tmp_path):
    torch = pytest.importorskip('torch')
    from scripts.server.psd_nan_boundary_capture import install_prefill_observer
    class Qwen3_5DecoderLayer(torch.nn.Module):
        def forward(self, x): return x + 1
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.layer = Qwen3_5DecoderLayer()
        def forward(self, x): return self.layer(x)
        def compute_logits(self, x): return x * 2
    model = Model()
    batch = SimpleNamespace(req_ids=['finite'], num_computed_tokens_cpu=torch.tensor([0]))
    worker = SimpleNamespace(model_runner=SimpleNamespace(input_batch=batch, get_model=lambda: model))
    install_prefill_observer(worker, tmp_path / 'capture', allowed_root=tmp_path)
    assert torch.equal(model.compute_logits(model(torch.tensor([[2., 3.]]))), torch.tensor([[6., 8.]]))
    batch.req_ids = ['invalid']
    with pytest.raises(RuntimeError, match='nonfinite raw model logits'):
        model.compute_logits(model(torch.tensor([[float('nan'), 3.]])))
    path = next((tmp_path / 'capture').glob('*.jsonl'))
    rows = [json.loads(l) for l in path.read_text().splitlines()]
    assert len(rows) == 2 and rows[0]['layers'][-1]['nan'] == 0
    assert rows[1]['layers'][0]['stage'] == 'layer' and rows[1]['layers'][0]['nan'] == 1
    assert rows[1]['layers'][-1]['stage'] == 'raw_model_logits'
