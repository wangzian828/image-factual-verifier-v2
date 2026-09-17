import asyncio
import copy
from pathlib import Path
from types import SimpleNamespace

import pytest

from ifv_training.psd_repair import _sha, build_hint_proposal, verify_repair
from ifv_training.psd_slate import (validate_slate_review, assemble_slate_attempts,
    validate_prefix_lineage, propose_slate)
from ifv_training.psd_repairs import _validate_attempt
from test_psd_repairs import _repair_candidate, _attempt


def test_slate_review_requires_observed_support_for_every_position():
    packet = {"decision_map": {0: 0, 1: 2, 24: 3}, "hinted_positions": [0, 1],
        "episode_complete": True, "repaired_steps": [
            {"index": 0, "thought": "earlier observed fact"},
            {"index": 2, "thought": "later observed fact"}]}
    review = {"status": "pass", "failed_position": -1, "passing_positions": [0, 1],
        "explanation": "supported", "evidence": [
            {"position": 0, "trace": "repaired", "step_index": 0, "quote": "earlier observed fact"},
            {"position": 1, "trace": "repaired", "step_index": 2, "quote": "later observed fact"}]}
    assert validate_slate_review(review, packet=packet) == review
    for mutation in ("missing", "invented", "future", "incomplete", "nonexistent"):
        bad, data = copy.deepcopy(review), copy.deepcopy(packet)
        if mutation == "missing":
            bad["evidence"].pop()
        elif mutation == "invented":
            bad["evidence"][0]["quote"] = "unobserved fact"
        elif mutation == "future":
            bad["evidence"][1]["position"] = 0
        elif mutation == "nonexistent":
            bad["passing_positions"].append(23)
        else:
            data["episode_complete"] = False
        with pytest.raises(ValueError):
            validate_slate_review(bad, packet=data)


def make_targets():
    result = []
    for pos in [0, 1]:
        h = build_hint_proposal(text="Check the unresolved relation." if not pos else "Revisit the unsupported inference.",
            level=1, provider="gemini", model="judge", candidate_id=f"case-{pos}", public_failure_context={})
        result.append({"position": pos, "hint": h.text, "hint_audit": dict(h.audit), "hint_level": 1,
            "hint_record": vars(h), "student_prompt_ids": [1, 2 + pos], "teacher_prompt_ids": [1, 2 + pos, 4],
            "completion_ids": [5, 6], "teacher_token_capture": {}, "context_request_id": f"r{pos}",
            "runtime_store_path": "archive", "student_prefix_kind": "corrected_hint_free_history"})
    return result


@pytest.mark.parametrize('review_version', ['v2', 'v3'])
def test_two_verified_local_targets_assemble_with_explicit_corrected_lineage(monkeypatch, review_version):
    import ifv_training.psd_repair_verifier as verifier
    verified = verify_repair(source_rollout_failed=True, hinted_local_pass=True,
        hinted_recorded_verdict="real", expected_verdict="real", hinted_strict_trace_audit_pass=True,
        hinted_episode_pass=True)
    monkeypatch.setattr(verifier, "verify_causal_episode", lambda *a, **kw: verified)
    episode = {"psd_repair": {"slate": {"used_positions": [0, 1], "unused_positions": []}}}
    review = {"episode_sha256": _sha(episode), "private_reference_sha256": _sha({}),
        "schema_version": "ifv-psd-slate-review-" + review_version,
        "hints_sha256": _sha({str(t["position"]): t["hint"] for t in make_targets()}),
        "decision": {"status": "pass", "passing_positions": [0, 1], "evidence": [{"quote": "observed"}]},
        "provenance": {"request_binding": {"model": "judge"}}}
    role_record = _attempt("x", hint="Check the unresolved relation.", hint_level=1)["model_roles"]
    candidates, records = assemble_slate_attempts(seed=_repair_candidate(), source={}, source_hash="trace-sha",
        episode=episode, targets=make_targets(), review=review, gold={}, source_task_review={}, source_audit=None,
        source_policy=None, roles=SimpleNamespace(record=lambda: role_record))
    assert len(candidates) == len(records) == 2
    for candidate, record in zip(candidates, records):
        assert record['model_roles']['hint_constructor']['model'] == record['hint_record']['model']
        checked = _validate_attempt(candidate=candidate, attempt=record)
        assert checked["row_weight"] == 1.0
        assert "rollout_token_capture" not in candidate["repair_site"]
        assert checked["student_prefix_capture"]["local_target"]["student_prefix_kind"] == "corrected_hint_free_history"
        bad = copy.deepcopy(record)
        bad["student_prompt_ids"] = [999]
        with pytest.raises(ValueError, match="local target"):
            validate_prefix_lineage(candidate, bad)
    with pytest.raises(ValueError, match="coverage"):
        assemble_slate_attempts(seed=_repair_candidate(), source={}, source_hash="trace-sha",
            episode=episode, targets=make_targets()[:1], review=review, gold={}, source_task_review={},
            source_audit=None, source_policy=None, roles=None)


def test_proposer_receives_public_feedback_only_and_audits_actual_hint(monkeypatch):
    import ifv_training.psd_gemini_judge as judge
    packets = []
    async def request(client, packet, **kwargs):
        packets.append(packet)
        return {"hints": [{"position": 0, "hint": "Check the unresolved relation."}]}, {}
    monkeypatch.setattr(judge, "_request", request)
    hints, _ = asyncio.run(propose_slate(None, public_context={"observed": "tool error", "decision_map": {0: 0}}, previous={},
        passing_positions=[], failed_position=0, model="judge", cache_dir=None, private_context={"label": "fake"}))
    assert hints[0].audit["passed"] is True
    assert "fake" not in str(packets) and "label" not in str(packets)


def test_invalid_hint_budget_is_bounded_and_pending_proposal_is_reused(monkeypatch):
    import ifv_training.psd_slate_search as search
    from ifv_training.psd_slate import SlateProposalRejected
    from ifv_training.psd_repair import HintProposal
    calls, saved, state = [], [], {}
    async def proposal(*a,**kw):
        calls.append(kw)
        if len(calls)==1:raise SlateProposalRejected({'response_sha256':'rejected'})
        target=make_targets()[0]
        return {0:HintProposal(**target['hint_record'])},{'response_sha256':'accepted'}
    monkeypatch.setattr(search,'propose_slate',proposal)
    options=dict(state=state,round_index=0,budget=2,persist=lambda:saved.append(copy.deepcopy(state)),
                 judge=None,kwargs={})
    first=asyncio.run(search.propose_with_budget(**options))
    assert len(calls)==2 and len(saved)==2
    assert calls[1]['proposal_feedback']=={'rejected_proposals':1,'reason':'invalid_or_nonprocedural_slate'}
    assert asyncio.run(search.propose_with_budget(**options))==first and len(calls)==2
    state.pop('pending_proposal')
    assert asyncio.run(search.propose_with_budget(**{**options,'round_index':1}))==(None,None)
    assert len(calls)==2


def test_proposal_transport_errors_do_not_spend_hint_budget(monkeypatch):
    import ifv_training.psd_slate_search as search
    state={}
    async def proposal(*a,**kw):raise TimeoutError('provider unavailable')
    monkeypatch.setattr(search,'propose_slate',proposal)
    with pytest.raises(TimeoutError):
        asyncio.run(search.propose_with_budget(state=state,round_index=0,budget=12,
            persist=lambda:None,judge=None,kwargs={}))
    assert state['proposals']==[]


def test_rejected_hint_feedback_never_reveals_private_match(monkeypatch):
    import ifv_training.psd_gemini_judge as judge
    from ifv_training.psd_slate import SlateProposalRejected
    packets=[]
    async def request(client,packet,**kw):
        packets.append(packet)
        return {'hints':[{'position':0,'hint':'Use the exact query {secret}.'}]},{'response_sha256':'x'}
    monkeypatch.setattr(judge,'_request',request)
    with pytest.raises(SlateProposalRejected) as e:
        asyncio.run(propose_slate(None,public_context={'observed':'tool error','decision_map':{0:0}},previous={},
            passing_positions=[],failed_position=0,model='judge',cache_dir=None,
            private_context={'private_reference':'hidden_private_gold'},
            proposal_feedback={'rejected_proposals':1,'reason':'invalid_or_nonprocedural_slate'}))
    assert str(e.value)=='invalid_or_nonprocedural_slate'
    assert 'hidden_private_gold' not in str(packets)


def test_psd_child_sampling_is_explicit_and_does_not_change_native_workflow(monkeypatch):
    from scripts.collect_psd_rollouts import PSDWorkflow
    from src.workflow import VerificationWorkflow
    native = SimpleNamespace(_stage_generation_config=lambda stage: {"temperature": 1.0, "top_p": .95},
                             llm=SimpleNamespace(get_response=lambda: None, max_retries=0))
    monkeypatch.setattr(VerificationWorkflow, "_get_orchestrator", lambda *a, **kw: native)
    workflow = PSDWorkflow()
    assert isinstance(workflow._new_batch_child(workflow.config), PSDWorkflow)
    policy = workflow._get_orchestrator()
    assert policy._stage_generation_config("UNIFIED_REACT") == {"temperature": .7, "top_p": .95}
    assert policy._stage_generation_config("OTHER")["temperature"] == 1.0
    # Repeated initialization must not stack wrappers / cause recursion.
    assert workflow._get_orchestrator()._stage_generation_config("UNIFIED_JUDGMENT")["temperature"] == .7


@pytest.mark.parametrize('infrastructure_fault', [False, True])
def test_slate_search_resume_does_not_repeat_completed_full_reruns(monkeypatch, tmp_path, infrastructure_fault):
    import httpx
    import src.integrations.gemini as gemini
    import src.orchestrator.runtime_events as runtime
    import ifv_training.psd_gemini_judge as judge
    import ifv_training.psd_slate_search as search
    from ifv_training.psd_repair_runtime import ContinuationResult
    from ifv_training.psd_repair import HintProposal
    import ifv_training.psd_infrastructure_retry as recovery
    native_retry = recovery.retry_episode
    async def no_sleep(_):
        pass
    async def fast_retry(**kwargs):
        return await native_retry(**kwargs, sleep=no_sleep)
    monkeypatch.setattr(recovery, 'retry_episode', fast_retry)
    class Client:
        async def __aenter__(self):
            return self
        async def __aexit__(self, *a):
            pass
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: Client())
    monkeypatch.setattr(gemini, "GeminiInteractionsClient", lambda **kw: Client())
    monkeypatch.setattr(runtime, "CaseRuntimeStore", lambda *a, **kw: SimpleNamespace(root=tmp_path))
    monkeypatch.setattr(judge, "review_images", lambda *a, **kw: ([], {}))
    trace = {"state": {"all_steps": [{"stage": "unified_react", "action_type": "tool_call",
        "metadata": {"policy_input": {}, "policy_action": {}}} for _ in range(22)]}}
    source_path = tmp_path / "source.json"
    source_path.write_text("{}")
    args = SimpleNamespace(max_suffix_actions=None, run_student_diagnostic=False, repair_attempts=6,
        output_dir=tmp_path, trace=source_path, search_seconds=100000, policy_model="frozen",
        hint_constructor_model="judge", judge_model="judge", image=tmp_path / "image",
        round_start_checkpoint="frozen")
    calls, assemblies, infrastructure_attempts = [], [], []
    class Adapter:
        policy_llm = SimpleNamespace(api_key="", get_response=lambda: None, max_retries=0)
        def _initial_runtime_state(self, **kw):
            return SimpleNamespace(action_count=21), []
        async def run_hinted_episode(self, **kw):
            infrastructure_attempts.append(kw)
            if infrastructure_fault and len(infrastructure_attempts) == 1:
                raise recovery.PolicyInfrastructureFailure('model_nonfinite_selected_logprob')
            calls.append(kw)
            targets = make_targets()[:len(kw["hints_by_action"])]
            return ContinuationResult(teacher_steps=[], student_steps=[], teacher_history=[], student_history=[],
                hint="", teacher_complete=True, teacher_episode_trace={**trace, "rerun": len(calls)}, local_targets=targets)
    proposal_calls = []
    async def request(client, packet, **kw):
        proposal_calls.append(packet)
        n = 2 if packet["previous_hints"] else 1
        return {"hints": [{"position": t["position"], "hint": t["hint"]}
                          for t in make_targets()[:n]]}, {}
    async def review(*a, **kw):
        n = kw["episode"]["rerun"]
        return {"decision": {"status": "pass" if n == 2 else "fail", "passing_positions": [0, 1] if n == 2 else [0],
            "failed_position": -1 if n == 2 else 1}}
    def assemble(**kw):
        assemblies.append(kw)
        if len(assemblies) == 1:
            raise RuntimeError("simulated CPU materialization crash")
        return [{"candidate_id": "a"}, {"candidate_id": "b"}], [{"accepted": True}, {"accepted": True}]
    monkeypatch.setattr(judge, "_request", request)
    monkeypatch.setattr(search, "review_slate", review)
    monkeypatch.setattr(search, "assemble_slate_attempts", assemble)
    options = dict(args=args, adapter=Adapter(), site=SimpleNamespace(stage="unified_react"),
        candidate={"case_id": "train"}, trace=trace, gold={}, private_context={}, source_task_review={},
        source_audit=None, source_policy=None, roles=None, profile={"base_url": "http://127.0.0.1:1/v1"}, config={})
    with pytest.raises(RuntimeError, match="materialization crash"):
        asyncio.run(search.run_slate_search(**options))
    assert len(calls) == 2
    result = asyncio.run(search.run_slate_search(**options))
    assert len(calls) == 2 and len(assemblies) == 2
    assert len(proposal_calls) == 2 and proposal_calls[0]['failed_position'] == 21
    assert len(infrastructure_attempts) == (3 if infrastructure_fault else 2)
    if infrastructure_fault:
        assert infrastructure_attempts[0]['hints_by_action'] == infrastructure_attempts[1]['hints_by_action']
    assert set(calls[0]['hints_by_action']) == {0}
    assert result["accepted_count"] == 2 and result["complete_reruns"] == 2
    assert all(c["hint"] is None and c["failure_site"].step_index == 0 for c in calls)
    assert search.audit_slate_search(tmp_path)["passed"]
    assert asyncio.run(search.run_slate_search(**options)) == result
