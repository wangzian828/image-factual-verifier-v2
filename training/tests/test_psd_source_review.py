from __future__ import annotations

import asyncio
import json

import pytest

from ifv_training import psd_source_review as source
from ifv_training.psd_repair import _sha
from ifv_training.psd_repair_storage import save_bound
from ifv_training.io import load_jsonl, sha256_file


def source_trace():
    return {"case_id": "a", "verdict": "real", "state": {
        "runtime_case": {"case_id": "a", "image_sha256": "a" * 64},
        "all_steps": [{"stage": "unified_react", "tool_result": "search returned no results"},
                      {"stage": "unified_judgment", "output": "the source confirms the event"}]}}


def artifact(trace, *, status="pass", gold=None):
    gold = gold or {"case_id": "a", "factual_status": "supported"}
    media = {"task_image_sha256": trace["state"]["runtime_case"]["image_sha256"], "policy_images": []}
    steps = source.trace_steps(trace)
    quote = next(value for value in source._packet(trace, gold, media)["source_steps"][0].values()
                 if isinstance(value, str))
    packet = source._packet(trace, gold, media)
    return {"schema_version": source.VERSION, "source_trace_canonical_sha256": _sha(trace),
        "private_reference_sha256": _sha(gold), "media": media,
        "decision": {"status": status, "explanation": "Synthetic control, not a quality result.",
            "evidence": [{"trace": "source", "step_index": steps[0]["index"], "quote": quote}]},
        "verifier": {"kind": "task", "name": "gemini-psd-source", "version": source.VERSION, "model": "test"},
        "provenance": {"response_sha256": "b" * 64, "request_binding": {
            "model": "test", "prompt_sha256": _sha(source.PROMPT), "schema_sha256": _sha(source.SCHEMA),
            "packet_sha256": _sha(packet)}}}


class Client:
    def __init__(self, decision):
        self.decision, self.calls = decision, 0

    async def create(self, **kwargs):
        self.calls += 1
        return {"status": "completed", "outputs": [{"type": "text", "text": json.dumps(self.decision)}]}


def test_corrective_review_accepts_first_valid_fail_and_is_cached(tmp_path, monkeypatch):
    trace, gold = source_trace(), {"case_id": "a", "factual_status": "supported"}
    expected = artifact(trace, status="fail", gold=gold)
    monkeypatch.setattr(source, "review_images", lambda *a, **kw: ([], expected["media"]))
    invalid = {**expected["decision"], "status": "pass", "evidence": [
        {"trace": "source", "step_index": 0, "quote": "fabricated quotation"}]}
    class SequenceClient(Client):
        async def create(self, **kwargs):
            self.decision = invalid if self.calls == 0 else expected["decision"]
            return await super().create(**kwargs)
    client = SequenceClient(None)
    args = dict(gold=gold, image_path=tmp_path / "unused", model="test", cache_dir=tmp_path / "cache")
    value = asyncio.run(source.judge_source(client, trace, **args))
    assert source.validate_source_review(value, trace=trace, gold=gold) == "fail"
    assert len(value["source_review_attempts"]) == 2 and client.calls == 2
    assert asyncio.run(source.judge_source(client, trace, **args)) == value
    assert client.calls == 2 and len(list((tmp_path / "cache").glob('*.json'))) == 2
    value['source_review_attempts'][0]['decision'] = expected['decision']
    with pytest.raises(ValueError, match='never be resampled'):
        source.validate_source_review(value, trace=trace, gold=gold)


def test_quote_repair_rejects_malformed_response_without_resampling_decision(tmp_path, monkeypatch):
    trace, gold = source_trace(), {"case_id": "a", "factual_status": "supported"}
    expected = artifact(trace, gold=gold)
    monkeypatch.setattr(source, "review_images", lambda *a, **kw: ([], expected["media"]))
    invalid = {**expected['decision'], 'evidence': [{'trace': 'source', 'step_index': 0, 'quote': 'not observed'}]}
    client = Client(invalid)
    args = dict(gold=gold, image_path=tmp_path / 'unused', model='test', cache_dir=tmp_path / 'cache')
    for _ in range(2):
        with pytest.raises(ValueError, match='invalid PSD source evidence quote repair'):
            asyncio.run(source.judge_source(client, trace, **args))
    assert client.calls == 3


def test_quote_only_repair_cannot_change_decision_or_location(tmp_path, monkeypatch):
    trace, gold = source_trace(), {"case_id": "a", "factual_status": "supported"}
    expected = artifact(trace, status="fail", gold=gold)
    monkeypatch.setattr(source, "review_images", lambda *a, **kw: ([], expected["media"]))
    invalid = {**expected["decision"], "evidence": [
        {"trace": "source", "step_index": 0, "quote": "the search failed"}]}
    literal = expected["decision"]["evidence"][0]["quote"]
    repair = {"repairable": True, "quotes": [{"evidence_index": 0, "quote": literal}]}
    class SequenceClient(Client):
        async def create(self, **kwargs):
            self.decision = invalid if self.calls < 2 else repair
            return await super().create(**kwargs)
    client = SequenceClient(None)
    args = dict(gold=gold, image_path=tmp_path / "unused", model="test", cache_dir=tmp_path / "cache")
    value = asyncio.run(source.judge_source(client, trace, **args))
    assert value["decision"] == expected["decision"]
    assert value["source_review_attempts"][-1]["decision"] == invalid
    assert value["evidence_quote_repair"]["response"] == repair
    assert source.validate_source_review(value, trace=trace, gold=gold) == "fail"
    assert client.calls == 3
    assert asyncio.run(source.judge_source(client, trace, **args)) == value
    assert client.calls == 3
    value["decision"]["status"] = "pass"
    with pytest.raises(ValueError, match="invalid bound source evidence quote repair"):
        source.validate_source_review(value, trace=trace, gold=gold)


@pytest.mark.parametrize('encoding', ['json_field', 'json_quoted_excerpt'])
def test_json_literal_interpretation_recovers_cached_final_without_resampling(tmp_path, monkeypatch, encoding):
    trace, gold = source_trace(), {'case_id':'a','factual_status':'supported'}
    if encoding == 'json_field':
        text = 'The record says "2019", not "2018".'
        trace['state']['all_steps'][0]['tool_result'] = json.dumps({'evidence':text})
        quote = text
    else:
        trace['state']['all_steps'][0]['tool_result'] = 'Prefix: the record states 2019. More follows.'
        quote = json.dumps('the record states 2019.')
    expected = artifact(trace, status='fail', gold=gold)
    monkeypatch.setattr(source,'review_images',lambda *a,**kw:([],expected['media']))
    decision = {**expected['decision'],'evidence':[{'trace':'source','step_index':0,'quote':quote}]}
    client = Client(decision)
    args = dict(gold=gold,image_path=tmp_path/'unused',model='test',cache_dir=tmp_path/'cache')
    value = asyncio.run(source.judge_source(client,trace,**args))
    assert client.calls == 2 and len(value['source_review_attempts']) == 2
    assert value['decision'] == decision and value['evidence_encoding'] == source.JSON_EVIDENCE_ENCODING
    assert source.validate_source_review(value,trace=trace,gold=gold) == 'fail'
    assert asyncio.run(source.judge_source(client,trace,**args)) == value and client.calls == 2
    value.pop('evidence_encoding')
    with pytest.raises(ValueError,match='literal observed quote'):
        source.validate_source_review(value,trace=trace,gold=gold)


def test_json_interpretation_does_not_replace_previously_valid_corrective_decision(tmp_path,monkeypatch):
    trace,gold=source_trace(),{'case_id':'a','factual_status':'supported'}
    trace['state']['all_steps'][0]['tool_result']=json.dumps({'evidence':'He said "2019".'})
    expected=artifact(trace,status='fail',gold=gold)
    monkeypatch.setattr(source,'review_images',lambda *a,**kw:([],expected['media']))
    first={**expected['decision'],'status':'pass','evidence':[{'trace':'source','step_index':0,'quote':'He said "2019".'}]}
    class SequenceClient(Client):
        async def create(self,**kwargs):
            self.decision=first if self.calls==0 else expected['decision']
            return await super().create(**kwargs)
    client=SequenceClient(None)
    value=asyncio.run(source.judge_source(client,trace,gold=gold,image_path=tmp_path/'unused',model='test',cache_dir=tmp_path))
    assert value['decision']==expected['decision'] and 'evidence_encoding' not in value
    assert client.calls==2 and source.validate_source_review(value,trace=trace,gold=gold)=='fail'


def test_unknown_json_evidence_encoding_rejected():
    trace=source_trace();value=artifact(trace);value['evidence_encoding']='allow-paraphrase'
    with pytest.raises(ValueError,match='unknown PSD source evidence encoding'):
        source.validate_source_review(value,trace=trace)


@pytest.mark.parametrize("status", ["pass", "fail", "unresolved"])
def test_bound_source_review_and_cached_decision(tmp_path, monkeypatch, status):
    trace, gold = source_trace(), {"case_id": "a", "factual_status": "supported"}
    expected = artifact(trace, status=status, gold=gold)
    monkeypatch.setattr(source, "review_images", lambda *a, **kw: ([], expected["media"]))
    client = Client(expected["decision"])
    args = dict(gold=gold, image_path=tmp_path / "unused.png", model="test", cache_dir=tmp_path / "cache")
    first = asyncio.run(source.judge_source(client, trace, **args))
    assert source.validate_source_review(first, trace=trace, gold=gold) == status
    client.decision = {**expected["decision"], "status": "pass"}
    assert asyncio.run(source.judge_source(client, trace, **args)) == first
    assert client.calls == 1


@pytest.mark.parametrize("changed", ["trace", "gold", "image", "reviewer", "prompt", "packet", "evidence", "response"])
def test_source_review_rejects_stale_or_missing_binding(changed):
    trace, gold = source_trace(), {"case_id": "a", "factual_status": "supported"}
    value = artifact(trace, gold=gold)
    if changed == "trace":
        trace["verdict"] = "fake"
    elif changed == "gold":
        gold["factual_status"] = "refuted"
    elif changed == "image":
        value["media"]["task_image_sha256"] = "wrong"
    elif changed == "reviewer":
        value["verifier"]["name"] = "final-answer-judge"
    elif changed in {"prompt", "packet"}:
        value["provenance"]["request_binding"][changed + "_sha256"] = "wrong"
    elif changed == "evidence":
        value["decision"]["evidence"] = []
    else:
        value["provenance"].pop("response_sha256")
    with pytest.raises(ValueError):
        source.validate_source_review(value, trace=trace, gold=gold)


@pytest.mark.parametrize("status,expected", [("pass", False), ("fail", True), ("unresolved", False)])
def test_correct_label_can_fail_semantically(monkeypatch, status, expected):
    from ifv_training import psd_repair_verifier as verifier
    trace, gold = source_trace(), {"case_id": "a", "factual_status": "supported"}
    monkeypatch.setattr(verifier, "score_process_trace", lambda *a, **kw: ({"result_correct": True}, {}))
    result = verifier.verify_source_rollout_failure(trace, gold=gold,
        source_task_review=artifact(trace, status=status, gold=gold))
    assert result["passed"] is expected
    assert result["result_correct"] is True
    assert result["semantic_failure"] is expected
    assert not verifier.verify_source_rollout_failure(trace, gold=gold)["passed"]


def test_artifact_file_tamper_rejected(tmp_path):
    path = tmp_path / "review.json"
    save_bound(path, identity={"test": 1}, payload=artifact(source_trace()))
    ref = {"source_task_review": {"path": str(path), "sha256": sha256_file(path)}}
    assert source.source_review_reference(ref)
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="artifact changed"):
        source.source_review_reference(ref)


def test_correct_label_structural_failure_requires_bound_audit(monkeypatch):
    from ifv_training import psd_repair_verifier as verifier
    trace = source_trace()
    monkeypatch.setattr(verifier, "score_process_trace", lambda *a, **kw: ({"result_correct": True}, {}))
    audit = {"source_trace_canonical_sha256": _sha(trace), "passed": False,
             "failures": [{"code": "invalid_observation_id"}]}
    assert verifier.verify_source_rollout_failure(trace, gold={}, source_audit=audit)["structural_failure"]
    audit["source_trace_canonical_sha256"] = "stale"
    assert not verifier.verify_source_rollout_failure(trace, gold={}, source_audit=audit)["passed"]


def test_correct_label_semantic_repair_reaches_complete_causal_gate(monkeypatch):
    from ifv_training import psd_repair_verifier as verifier
    from test_psd_repair import _patch_episode_verifiers, _teacher_step, _complete_episode, _local_verification
    _patch_episode_verifiers(monkeypatch)
    monkeypatch.setattr(verifier, "score_process_trace", lambda *a, **kw: (
        {"result_correct": True, "expected_verdict": "real", "engineering_error": False}, {}))
    trace, gold = source_trace(), {"case_id": "a", "factual_status": "supported"}
    args = dict(base_trace=trace, source_trace_sha256="a" * 64, teacher_steps=[_teacher_step()],
        student_steps=[], hinted_teacher_episode_trace=_complete_episode(image_id="teacher", verdict="real"),
        unhinted_student_episode_trace=None, gold=gold, repair_step_id="a-step-0", hint_sha256="b" * 64,
        local_verification=_local_verification("a-step-0", source_trace_sha256="a" * 64))
    accepted, _ = verifier.verify_continuation_pair(**args, source_task_review=artifact(trace, status="fail", gold=gold))
    assert accepted.accepted_for_primary_psd
    refused, _ = verifier.verify_continuation_pair(**args, source_task_review=artifact(trace, status="pass", gold=gold))
    assert not refused.accepted_for_primary_psd


@pytest.mark.parametrize("status,queue", [("pass", "preservation_candidates"),
    ("fail", "repair_candidates"), ("unresolved", "source_review_abstained"), (None, "source_review_pending")])
def test_candidate_routing_requires_semantics(tmp_path, status, queue):
    from test_psd_candidates import _trace, _write_json, _write_jsonl, _rollout_gate
    from ifv_training.psd_candidates import build_psd_candidate_package
    run = tmp_path / "run"
    trace = _trace(case_id="a")
    _write_json(run / "traces/episode-a.json", trace)
    _write_jsonl(run / "rollout_groups.jsonl", [{"episode_id": "episode-a", "trace_path": "traces/episode-a.json"}])
    reward = {"case_id": "a", "episode_id": "episode-a", "classification_correct": True, "strict_trace_audit_pass": True}
    if status is not None:
        path = tmp_path / "review.json"
        save_bound(path, identity={"test": True}, payload=artifact(trace, status=status))
        reward.update(source_task_status=status, source_task_review={"path": str(path), "sha256": sha256_file(path)})
    _write_jsonl(run / "post_rollout_rewards.jsonl", [reward])
    cases = tmp_path / "train.jsonl"
    _write_jsonl(cases, [{"case_id": "a", "split": "train"}])
    result = build_psd_candidate_package(run_dir=run, train_cases_path=cases,
        rollout_gate_path=_rollout_gate(run, cases, add_review=False), output_dir=tmp_path / "out")
    assert result["counts"][queue] == 1
    if status == "unresolved":
        row = load_jsonl(tmp_path / "out/source_review_abstained.jsonl")[0]
        assert row["queue_reason"] == "source_semantic_review_abstained"
    if status == "fail":
        row = load_jsonl(tmp_path / "out/repair_candidates.jsonl")[0]
        assert row["repair_signal"] == "source_semantic_failure"
        assert "private_reference" not in json.dumps(row["repair_site"]["model_visible"])


def test_review_run_resume_keeps_decisions_and_does_not_label_transport_failure(tmp_path, monkeypatch):
    from scripts import review_psd_sources as script
    from ifv_training.io import write_json, write_jsonl
    import hashlib
    run = tmp_path / "run"
    image = tmp_path / "a.png"
    image.write_bytes(b"synthetic image fixture; native media helper is mocked")
    trace = source_trace()
    trace["state"]["runtime_case"]["image_sha256"] = hashlib.sha256(image.read_bytes()).hexdigest()
    write_json(run / "run_manifest.json", {"status": "completed"})
    write_json(run / "traces/a.json", trace)
    write_jsonl(run / "run_results.jsonl", [{"case_id": "a"}])
    write_jsonl(tmp_path / "public.jsonl", [{"case_id": "a", "image_path": "a.png"}])
    write_jsonl(tmp_path / "split.jsonl", [{"case_id": "a", "split": "train"}])
    gold = {"case_id": "a", "factual_status": "supported"}
    write_jsonl(tmp_path / "gold.jsonl", [gold])
    calls = []
    async def provider(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise TimeoutError("secret-free synthetic transport error")
        result = artifact(trace, status="fail", gold=gold)
        result['trace_projection'] = source.TRACE_PROJECTION
        result['provenance']['request_binding']['packet_sha256'] = _sha(
            source._packet(trace, gold, result['media'], include_transport_ids=True))
        return result
    monkeypatch.setattr(script, "judge_source", provider)
    args = dict(run_dir=run, benchmark=tmp_path / "public.jsonl", train_cases=tmp_path / "split.jsonl",
        private_gold=tmp_path / "gold.jsonl", output=tmp_path / "review", model="test", client=object())
    first = asyncio.run(script.review_sources(**args))
    assert first["counts"]["pending_error"] == 1 and first["counts"]["fail"] == 0
    second = asyncio.run(script.review_sources(**args))
    assert second["counts"]["fail"] == 1 and second["pending"] == 0
    offloads = []
    original_to_thread = script.asyncio.to_thread
    async def tracked_to_thread(function, *values, **options):
        offloads.append(function)
        return await original_to_thread(function, *values, **options)
    monkeypatch.setattr(script.asyncio, "to_thread", tracked_to_thread)
    assert asyncio.run(script.review_sources(**args)) == second
    assert len(offloads) == 1
    assert len(calls) == 2
    # A valid old review remains readable as history but must not be silently
    # reused by the new driver under an unchanged prompt/schema hash.
    from ifv_training.io import load_json
    marker = args['output'] / 'inputs.json'
    saved = load_json(marker)
    assert saved['identity']['trace_projection'] == source.TRACE_PROJECTION
    legacy_identity = {k: v for k, v in saved['identity'].items() if k != 'trace_projection'}
    save_bound(marker, identity=legacy_identity, payload=saved['payload'])
    with pytest.raises(ValueError, match='binding changed'):
        asyncio.run(script.review_sources(**args))
    assert len(calls) == 2
    write_jsonl(tmp_path / "split.jsonl", [{"case_id": "a", "split": "test"}])
    with pytest.raises(ValueError, match="membership"):
        asyncio.run(script.review_sources(**args))
    assert len(calls) == 2


def test_review_run_treats_bound_unresolved_as_terminal_abstention(tmp_path, monkeypatch):
    from scripts import review_psd_sources as script
    from ifv_training.io import write_json, write_jsonl
    import hashlib
    run = tmp_path / "run"
    image = tmp_path / "a.png"
    image.write_bytes(b"synthetic image fixture; native media helper is mocked")
    trace = source_trace()
    trace["state"]["runtime_case"]["image_sha256"] = hashlib.sha256(image.read_bytes()).hexdigest()
    write_json(run / "run_manifest.json", {"status": "completed"})
    write_json(run / "traces/a.json", trace)
    write_jsonl(run / "run_results.jsonl", [{"case_id": "a"}])
    write_jsonl(tmp_path / "public.jsonl", [{"case_id": "a", "image_path": "a.png"}])
    write_jsonl(tmp_path / "split.jsonl", [{"case_id": "a", "split": "train"}])
    gold = {"case_id": "a", "factual_status": "supported"}
    write_jsonl(tmp_path / "gold.jsonl", [gold])
    async def provider(*args, **kwargs):
        result = artifact(trace, status="unresolved", gold=gold)
        result["trace_projection"] = source.TRACE_PROJECTION
        result["provenance"]["request_binding"]["packet_sha256"] = _sha(
            source._packet(trace, gold, result["media"], include_transport_ids=True))
        return result
    monkeypatch.setattr(script, "judge_source", provider)
    result = asyncio.run(script.review_sources(run_dir=run, benchmark=tmp_path / "public.jsonl",
        train_cases=tmp_path / "split.jsonl", private_gold=tmp_path / "gold.jsonl",
        output=tmp_path / "review", model="test", client=object()))
    assert result["pending"] == 0 and result["abstained"] == 1
    assert result["status"] == "source_reviews_complete_with_abstentions"
