"""A raw teacher distribution must not inherit grammar/budget masks."""
from types import SimpleNamespace
import pytest

from ifv_training.psd_repairs import _captured_topk
from ifv_training.psd_capture_semantics import RAW_POLICY_LOGPROBS
from ifv_training.psd_datums import build_sparse_topk_datum


def test_legacy_masked_capture_becomes_pending_without_losing_tokens():
    capture = {'topk': 20, 'completion_token_ids': [2],
        'completion_topk_by_position': [[[i, float(i == 2)] for i in range(20)]]}
    assert _captured_topk(capture, completion_ids=[2]) is None
    assert capture['completion_token_ids'] == [2]
    capture['teacher_logprob_semantics'] = RAW_POLICY_LOGPROBS
    assert _captured_topk(capture, completion_ids=[2]) is not None


def test_old_complete_target_cannot_skip_teacher_semantics_gate():
    with pytest.raises(ValueError, match='teacher_logprob_semantics_unattested'):
        build_sparse_topk_datum({'target_status': 'complete'}, topk=20)


def test_capture_semantics_requires_explicit_response_marker(monkeypatch):
    import sys
    from src.orchestrator import llm_backend
    from ifv_training.psd_capture_semantics import install_capture_semantics
    original = lambda payload, **kwargs: {'status': 'complete'}
    monkeypatch.setattr(llm_backend, 'extract_policy_token_capture', original)
    stage = SimpleNamespace(extract_policy_token_capture=original)
    monkeypatch.setitem(sys.modules, 'src.orchestrator.stage_runner', stage)
    install_capture_semantics()
    assert 'teacher_logprob_semantics' not in llm_backend.extract_policy_token_capture({})
    assert llm_backend.extract_policy_token_capture({'ifv_policy_logprobs': RAW_POLICY_LOGPROBS})[
        'teacher_logprob_semantics'] == RAW_POLICY_LOGPROBS
    assert stage.extract_policy_token_capture is llm_backend.extract_policy_token_capture
    assert stage.extract_policy_token_capture({'ifv_policy_logprobs': RAW_POLICY_LOGPROBS})[
        'teacher_logprob_semantics'] == RAW_POLICY_LOGPROBS
    install_capture_semantics()
    assert stage.extract_policy_token_capture is llm_backend.extract_policy_token_capture


def test_capture_installed_before_stage_import_is_idempotent(monkeypatch):
    import sys
    from src.orchestrator import llm_backend
    from ifv_training.psd_capture_semantics import install_capture_semantics
    monkeypatch.delitem(sys.modules, 'src.orchestrator.stage_runner', raising=False)
    monkeypatch.setattr(llm_backend, 'extract_policy_token_capture', lambda payload, **kwargs: {})
    install_capture_semantics()
    captured = llm_backend.extract_policy_token_capture
    monkeypatch.setitem(sys.modules, 'src.orchestrator.stage_runner', SimpleNamespace(extract_policy_token_capture=captured))
    install_capture_semantics()
    assert llm_backend.extract_policy_token_capture is captured


def test_candidate_projection_preserves_probability_provenance():
    from ifv_training.psd_candidates import _policy_token_capture
    projected = _policy_token_capture({'policy_token_capture': {'status': 'complete',
        'prompt_token_ids': [1, 2], 'completion_token_ids': [3], 'completion_logprobs': [-0.5],
        'teacher_logprob_semantics': RAW_POLICY_LOGPROBS, 'topk': 20}})
    assert projected['teacher_logprob_semantics'] == RAW_POLICY_LOGPROBS


def test_gateway_cannot_label_unknown_workers_as_raw():
    from scripts.server.psd_raw_logprobs import validate_raw_worker_receipts
    with pytest.raises(ValueError, match='all live worker receipts'):
        validate_raw_worker_receipts([], ['http://127.0.0.1:19002'], {})


def test_raw_capture_preserves_sampling_and_original_teacher_logits():
    torch = pytest.importorskip('torch')
    from scripts.server.psd_raw_logprobs import install_raw_policy_logprobs
    class Sampler:
        logprobs_mode = 'raw_logprobs'
        compute_logprobs = staticmethod(lambda x: x.log_softmax(-1))
        @staticmethod
        def gather_logprobs(x, count, token_ids):
            return x.clone()
        def forward(self, logits, metadata):
            before_budget = logits.log_softmax(-1)
            # Stand-in for a forcing processor; sampling must remain unchanged.
            logits[:, 1] = -float('inf')
            return SimpleNamespace(sampled_token_ids=logits.argmax(-1)[:, None], logprobs_tensors=before_budget)
    def mask(logits):
        logits[:, 0] = -float('inf')
    module = SimpleNamespace(apply_grammar_bitmask=mask)
    sampler = Sampler()
    original = torch.tensor([[9., 8., 2., 1.]])
    reference = original.clone(); mask(reference)
    stock = sampler.forward(reference, SimpleNamespace(max_num_logprobs=4))
    install_raw_policy_logprobs(module, sampler)
    candidate = original.clone(); module.apply_grammar_bitmask(candidate)
    result = sampler.forward(candidate, SimpleNamespace(max_num_logprobs=4))
    assert torch.equal(stock.sampled_token_ids, result.sampled_token_ids)
    assert torch.equal(result.logprobs_tensors, original.log_softmax(-1))
    assert torch.isfinite(result.logprobs_tensors).all()
    # The next unconstrained batch cannot reuse the previous logits snapshot.
    next_logits = torch.tensor([[1., 3., 5., 8.]])
    next_result = sampler.forward(next_logits.clone(), SimpleNamespace(max_num_logprobs=4))
    assert torch.equal(next_result.logprobs_tensors, next_logits.log_softmax(-1))
    with pytest.raises(ValueError, match='already installed'):
        install_raw_policy_logprobs(module, sampler)


def test_grammar_exception_does_not_leave_a_stale_raw_batch():
    torch = pytest.importorskip('torch')
    from scripts.server.psd_raw_logprobs import install_raw_policy_logprobs
    def mask(logits):
        raise ValueError('grammar failure')
    sampler = SimpleNamespace(forward=lambda logits, metadata: logits, logprobs_mode='raw_logprobs')
    module = SimpleNamespace(apply_grammar_bitmask=mask)
    install_raw_policy_logprobs(module, sampler)
    for _ in range(2):
        with pytest.raises(ValueError, match='grammar failure'):
            module.apply_grammar_bitmask(torch.ones(1, 4))
