from __future__ import annotations

import json

from src.orchestrator.ledger import (
    VerificationLedger,
    _update_claim_statuses,
    build_verification_case,
    compile_runtime_ledgers,
    derive_unverifiable_reasons,
    _direction_is_decisive,
)
from src.orchestrator.investigation_state import InvestigationState, VisualQuestion
from src.orchestrator.pipeline import Orchestrator
from src.orchestrator.stage_runner import StageStep
from src.orchestrator.state import (
    ClaimRecord,
    EvidenceItem,
    EvidenceRecord,
    InvestigationQuestion,
    SourceRecord,
    UnverifiableReason,
    VerificationPlan,
    VerificationLedgers,
    VerificationResult,
    VisualAnomaly,
)


def _orchestrator() -> Orchestrator:
    return object.__new__(Orchestrator)


def _plan() -> VerificationPlan:
    return VerificationPlan(
        questions=[
            InvestigationQuestion(
                question_id="q0",
                question="Did Reuters publish the flood image?",
                claim_text="Reuters published the flood image.",
                related_entities=["Reuters", "flood"],
                priority=1,
            )
        ]
    )


def _successful_step() -> StageStep:
    return StageStep(
        round=1,
        stage_name="verification",
        action_type="tool_call",
        tool_name="visit",
        tool_args={"__question_id": "q0"},
        tool_result=(
            '{"status":"success","selected_url":"https://reuters.example/flood",'
            '"goal":"Reuters published the flood image.",'
            '"evidence":"Reuters published the flood image on 10 July.",'
            '"summary":"Reuters published the flood image on 10 July.",'
            '"relevance":"high","stance":"support",'
            '"artifact_sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",'
            '"evidence_span":{"start":0,"end":45},'
            '"retrieved_at":"2026-07-11T00:00:00+00:00",'
            '"injection_flags":[],"directness":"direct","evidence_eligible":true}'
        ),
        metadata={"tool_success": True, "function_call_id": "call-visit-1"},
    )


def _evidence(**updates) -> EvidenceItem:
    values = {
        "function_call_id": "call-visit-1",
        "source": "https://reuters.example/flood",
        "summary": "Reuters published the flood image on 10 July.",
        "raw_excerpt": "Reuters published the flood image on 10 July.",
        "direction": "supports",
        "quality": "strong",
        "tool_used": "visit",
        "related_question": "q0",
    }
    values.update(updates)
    return EvidenceItem(**values)


def test_evidence_requires_exact_successful_function_call_id() -> None:
    orchestrator = _orchestrator()
    step = _successful_step()
    evidence = _evidence(function_call_id="call-other")

    assert orchestrator._evidence_item_is_grounded(evidence, [step]) is False


def test_evidence_rejects_fabricated_excerpt_even_with_real_call_id() -> None:
    orchestrator = _orchestrator()
    step = _successful_step()
    evidence = _evidence(raw_excerpt="Reuters confirmed this was an AI-generated hoax.")

    assert orchestrator._evidence_item_is_grounded(evidence, [step]) is False


def test_invalid_model_citation_is_filtered_before_merge_and_ledger(tmp_path) -> None:
    orchestrator = _orchestrator()
    step = _successful_step()
    invalid = _evidence(
        raw_excerpt="Reuters confirmed an excerpt that the tool never returned.",
        summary="Reuters confirmed an excerpt that the tool never returned.",
    )

    merged = orchestrator._merge_verification_results(
        VerificationResult(),
        [VerificationResult(evidence=[invalid], authenticity_assessment="uncertain")],
        [step],
        _plan(),
    )

    image_path = tmp_path / "invalid-citation.jpg"
    image_path.write_bytes(b"invalid-citation-filter")
    ledgers = compile_runtime_ledgers(
        build_verification_case(str(image_path)),
        _plan(),
        merged,
        [step],
    )
    assert merged.evidence == []
    assert ledgers.evidence == []


def test_coverage_rejects_real_but_question_irrelevant_excerpt() -> None:
    orchestrator = _orchestrator()
    step = _successful_step()
    step.tool_result = (
        '{"status":"success","selected_url":"https://reuters.example/weather",'
        '"evidence":"Temperatures reached 30 degrees on 10 July.",'
        '"summary":"Temperatures reached 30 degrees on 10 July."}'
    )
    evidence = _evidence(
        source="https://reuters.example/weather",
        summary="Temperatures reached 30 degrees on 10 July.",
        raw_excerpt="Temperatures reached 30 degrees on 10 July.",
    )
    parsed = VerificationResult(evidence=[evidence])

    accepted, reason = orchestrator._validate_verification_output(parsed, [step], _plan())

    assert accepted is False
    assert "not an eligible exact passage" in reason


def test_one_generic_token_does_not_make_evidence_answer_a_question() -> None:
    question = InvestigationQuestion(
        question_id="q0",
        question="Did Sajith Premadasa join the United National Party?",
        claim_text="Sajith Premadasa joined the United National Party.",
        priority=1,
    )
    evidence = EvidenceItem(
        function_call_id="call-1",
        source="https://example.test",
        summary="A different politician joined another party.",
        raw_excerpt="A different politician joined another party.",
        direction="refutes",
        quality="moderate",
        tool_used="visit",
        related_question="q0",
    )

    assert Orchestrator._evidence_answers_question(evidence, question) is False


def test_model_cannot_insert_anomaly_without_anomaly_tool_result() -> None:
    orchestrator = _orchestrator()
    step = _successful_step()
    anomaly = VisualAnomaly(
        function_call_id="call-visit-1",
        related_question="q0",
        name="Fabricated anomaly",
        region="center",
        phenomenon="An invented visual defect.",
        reasoning="This text was not returned by a visual tool.",
        severity=90,
        type="manipulation",
        entities_involved=["image"],
    )
    parsed = VerificationResult(evidence=[_evidence()], visual_anomalies=[anomaly])

    accepted, reason = orchestrator._validate_verification_output(parsed, [step], _plan())

    assert accepted is False
    assert "visual anomalies must copy" in reason


def test_canonicalization_does_not_trust_model_summary_or_direction() -> None:
    orchestrator = _orchestrator()
    step = _successful_step()
    model_item = _evidence(
        summary="The image is definitely fake.",
        direction="refutes",
    )

    canonical = orchestrator._canonicalize_model_evidence(model_item, [step], _plan())

    assert canonical is not None
    assert canonical.summary == canonical.raw_excerpt
    assert canonical.direction == "supports"
    assert canonical.function_call_id == "call-visit-1"


def test_canonicalization_uses_refuting_tool_stance_not_model_direction() -> None:
    orchestrator = _orchestrator()
    step = _successful_step()
    step.tool_result = step.tool_result.replace('"stance":"support"', '"stance":"refute"')
    model_item = _evidence(direction="supports")

    canonical = orchestrator._canonicalize_model_evidence(model_item, [step], _plan())

    assert canonical is not None
    assert canonical.direction == "refutes"


def test_nested_text_search_stance_is_bound_to_exact_excerpt() -> None:
    orchestrator = _orchestrator()
    step = _successful_step()
    step.tool_name = "text_search"
    step.tool_result = (
        '{"status":"success","queries":['
        '{"query":"flood","selected_url":"https://reuters.example/flood",'
        '"goal":"Reuters published the flood image.",'
        '"evidence":"Reuters published the flood image on 10 July.",'
        '"summary":"Reuters published the flood image on 10 July.",'
        '"relevance":"high","stance":"support",'
        '"artifact_sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",'
        '"evidence_span":{"start":0,"end":45},'
        '"retrieved_at":"2026-07-11T00:00:00+00:00",'
        '"injection_flags":[],"directness":"direct","evidence_eligible":true},'
        '{"query":"hoax","selected_url":"https://example.test/hoax",'
        '"goal":"Reuters published the flood image.",'
        '"evidence":"An unrelated page calls another image a hoax.",'
        '"summary":"Unrelated claim.","relevance":"low","stance":"refute",'
        '"artifact_sha256":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",'
        '"evidence_span":{"start":0,"end":45},'
        '"retrieved_at":"2026-07-11T00:00:00+00:00",'
        '"injection_flags":[],"directness":"direct","evidence_eligible":true}]}'
    )
    model_item = _evidence(tool_used="text_search", direction="refutes")

    canonical = orchestrator._canonicalize_model_evidence(model_item, [step], _plan())

    assert canonical is not None
    assert canonical.direction == "supports"


def test_image_region_evidence_is_not_decisive_for_image_provenance() -> None:
    evidence = [
        EvidenceRecord(
            evidence_id="e1",
            claim_id="claim-q0",
            source_id="source-image",
            function_call_id="call-1",
            tool_name="crop_and_inspect",
            evidence_kind="image_region",
            exact_text="This region contains only a text caption.",
            image_region=[0.0, 0.0, 1.0, 1.0],
            artifact_sha256="a" * 64,
            retrieved_at="2026-07-11T00:00:00+00:00",
            stance="refute",
            quality="moderate",
        )
    ]
    sources = {
        "source-image": SourceRecord(
            source_id="source-image",
            source_family="image:sha",
            source_class="visual",
            artifact_sha256="a" * 64,
            retrieved_at="2026-07-11T00:00:00+00:00",
        )
    }

    assert _direction_is_decisive(evidence, sources, claim_scope="image_provenance") is False


def test_multi_query_search_promotes_each_distinct_grounded_page(tmp_path) -> None:
    claim = "Reuters and NASA published reports about the flood image."

    def record(url: str, excerpt: str, artifact: str, stance: str = "support") -> dict:
        return {
            "status": "success",
            "url": url,
            "goal": claim,
            "evidence": excerpt,
            "summary": excerpt,
            "relevance": "high",
            "stance": stance,
            "directness": "direct",
            "artifact_sha256": artifact * 64,
            "evidence_span": {"start": 0, "end": len(excerpt)},
            "retrieved_at": "2026-07-12T00:00:00+00:00",
            "injection_flags": [],
            "evidence_eligible": True,
        }

    reuters = record(
        "https://www.reuters.com/world/flood-report",
        "Reuters published a report containing the flood image.",
        "a",
    )
    nasa = record(
        "https://www.nasa.gov/earth/flood-report",
        "NASA published satellite context for the same flood image.",
        "b",
    )
    archive = record(
        "https://archive.example.org/flood-image",
        "The independent archive dates the flood image to 10 July.",
        "c",
    )
    first_query = {
        **reuters,
        "query": "Reuters flood image",
        "selected_url": reuters["url"],
        "visited_pages": [reuters, archive],
    }
    second_query = {
        **nasa,
        "query": "NASA flood satellite image",
        "selected_url": nasa["url"],
        "visited_pages": [nasa],
    }
    step = StageStep(
        round=1,
        stage_name="verification",
        action_type="tool_call",
        tool_name="text_search",
        tool_args={"__question_id": "q0", "queries": ["Reuters", "NASA"], "goal": claim},
        tool_result=json.dumps(
            {"status": "success", "queries": [first_query, second_query]}
        ),
        metadata={"tool_success": True, "function_call_id": "call-search-many"},
    )
    plan = VerificationPlan(
        questions=[
            InvestigationQuestion(
                question_id="q0",
                question="Did Reuters and NASA publish reports about the flood image?",
                claim_text=claim,
                related_entities=["Reuters", "NASA", "flood"],
                claim_scope="external_fact",
                priority=1,
            )
        ]
    )
    state = type("State", (), {"plan": plan, "perception": None})()

    result = _orchestrator()._build_verification_result_from_steps([step], state)

    assert len(result.evidence) == 3
    assert {item.source for item in result.evidence} == {
        reuters["url"],
        nasa["url"],
        archive["url"],
    }
    assert {item.function_call_id for item in result.evidence} == {"call-search-many"}
    assert len(result.source_findings) == 3
    assert {
        (
            item["source"],
            item["artifact_sha256"],
            item["evidence_span"]["start"],
            item["evidence_span"]["end"],
            item["retrieved_at"],
            item["stance"],
            item["directness"],
        )
        for item in result.source_findings
    } == {
        (
            page["url"],
            page["artifact_sha256"],
            page["evidence_span"]["start"],
            page["evidence_span"]["end"],
            page["retrieved_at"],
            page["stance"],
            page["directness"],
        )
        for page in (reuters, nasa, archive)
    }

    image_path = tmp_path / "input.jpg"
    image_path.write_bytes(b"multi-page-search-ledger")
    ledgers = compile_runtime_ledgers(
        build_verification_case(str(image_path), user_claim=claim),
        plan,
        result,
        [step],
    )
    assert len(ledgers.sources) == 3
    assert len(ledgers.evidence) == 3
    assert {item.exact_text for item in ledgers.evidence} == {
        reuters["evidence"],
        nasa["evidence"],
        archive["evidence"],
    }


def test_browse_evidence_without_explicit_stance_is_rejected() -> None:
    orchestrator = _orchestrator()
    step = _successful_step()
    step.tool_result = step.tool_result.replace(',"relevance":"high","stance":"support"', "")

    assert orchestrator._canonicalize_model_evidence(_evidence(), [step], _plan()) is None


def test_visual_and_semantic_search_candidates_keep_distinct_types(tmp_path) -> None:
    image_path = tmp_path / "input.jpg"
    image_path.write_bytes(b"bounded-ledger-fixture")
    step = StageStep(
        round=1,
        stage_name="verification",
        action_type="tool_call",
        tool_name="reverse_image_search",
        tool_args={"__question_id": "q0"},
        tool_result=json.dumps(
            {
                "status": "success",
                "lens_results": [
                    {
                        "url": "https://example.test/reference",
                        "title": "Reference image",
                        "snippet": "Short snippet.",
                    }
                ],
                "semantic_results": [
                    {
                        "url": "https://example.test/reference",
                        "title": "Reference image",
                        "snippet": "A longer and more informative reference snippet.",
                    }
                ],
            }
        ),
        metadata={"tool_success": True, "function_call_id": "call-reverse-1"},
    )

    ledgers = compile_runtime_ledgers(
        build_verification_case(
            str(image_path),
            user_claim="Does the image match the reference?",
        ),
        _plan(),
        VerificationResult(),
        [step],
    )

    assert len(ledgers.discoveries) == 2
    by_type = {item.candidate_type: item for item in ledgers.discoveries}
    assert by_type["reverse_image"].snippet == "Short snippet."
    assert by_type["serp"].snippet == "A longer and more informative reference snippet."


def test_visual_ledger_compilation_is_idempotent(tmp_path) -> None:
    image_path = tmp_path / "input.jpg"
    image_path.write_bytes(b"idempotent-visual-fixture")
    case = build_verification_case(
        str(image_path),
        user_claim="Does the image contain a reference object?",
    )
    steps = [
        StageStep(
            round=index,
            stage_name="verification",
            action_type="tool_call",
            tool_name="crop_and_inspect",
            tool_args={"__question_id": "q0", "bbox": bbox},
            tool_result=json.dumps({"status": "success", "summary": excerpt}),
            metadata={
                "tool_success": True,
                "function_call_id": call_id,
                "observed_at": observed_at,
            },
        )
        for index, (call_id, bbox, excerpt, observed_at) in enumerate(
            [
                (
                    "call-visual-1",
                    [0.1, 0.1, 0.5, 0.5],
                    "The first inspected region contains a reference object.",
                    "2026-07-11T08:00:00+00:00",
                ),
                (
                    "call-visual-2",
                    [0.5, 0.5, 0.9, 0.9],
                    "The second inspected region contains a matching object.",
                    "2026-07-11T08:01:00+00:00",
                ),
            ],
            start=1,
        )
    ]
    result = VerificationResult(
        evidence=[
            EvidenceItem(
                function_call_id=step.metadata["function_call_id"],
                source="",
                summary=json.loads(step.tool_result)["summary"],
                raw_excerpt=json.loads(step.tool_result)["summary"],
                direction="supports",
                quality="moderate",
                tool_used="crop_and_inspect",
                related_question="q0",
            )
            for step in steps
        ]
    )

    first = compile_runtime_ledgers(case, _plan(), result, steps)
    second = compile_runtime_ledgers(case, _plan(), result, steps)

    assert first == second
    assert len(first.sources) == 1
    assert first.sources[0].retrieved_at == case.created_at
    assert [item.retrieved_at for item in first.evidence] == [
        "2026-07-11T08:00:00+00:00",
        "2026-07-11T08:01:00+00:00",
    ]


def test_current_time_is_recorded_as_runtime_anchor(tmp_path) -> None:
    image_path = tmp_path / "input.jpg"
    image_path.write_bytes(b"runtime-anchor-fixture")
    case = build_verification_case(
        str(image_path),
        user_claim="Was this claim posted before today?",
    )
    excerpt = "2026-07-11T16:00:00+08:00"
    step = StageStep(
        round=1,
        stage_name="verification",
        action_type="tool_call",
        tool_name="current_time",
        tool_args={"__question_id": "q0"},
        tool_result=json.dumps({"status": "success", "datetime": excerpt}),
        metadata={
            "tool_success": True,
            "function_call_id": "call-time-1",
            "observed_at": "2026-07-11T08:00:01+00:00",
        },
    )
    result = VerificationResult(
        evidence=[
            EvidenceItem(
                function_call_id="call-time-1",
                source="current_time",
                summary=excerpt,
                raw_excerpt=excerpt,
                direction="neutral",
                quality="strong",
                tool_used="current_time",
                related_question="q0",
            )
        ]
    )

    ledgers = compile_runtime_ledgers(case, _plan(), result, [step])

    assert len(ledgers.sources) == 1
    assert ledgers.sources[0].source_class == "runtime"
    assert ledgers.evidence[0].evidence_kind == "runtime_anchor"
    assert ledgers.evidence[0].image_region is None


def test_same_source_artifact_merges_retrieval_metadata() -> None:
    ledger = VerificationLedger()
    base = {
        "source_id": "source-stable",
        "canonical_url": "https://example.test/report",
        "hostname": "example.test",
        "registered_domain": "example.test",
        "source_family": "content:" + "a" * 64,
        "source_class": "unknown",
        "artifact_sha256": "a" * 64,
    }

    ledger.add_source(
        SourceRecord(
            **base,
            retrieved_at="2026-07-11T08:01:00+00:00",
            risk_flags=["second_flag"],
        )
    )
    merged = ledger.add_source(
        SourceRecord(
            **base,
            retrieved_at="2026-07-11T08:00:00+00:00",
            risk_flags=["first_flag"],
        )
    )

    assert len(ledger.data.sources) == 1
    assert merged.retrieved_at == "2026-07-11T08:00:00+00:00"
    assert merged.risk_flags == ["first_flag", "second_flag"]


def test_same_source_id_rejects_immutable_conflict() -> None:
    ledger = VerificationLedger()
    ledger.add_source(
        SourceRecord(
            source_id="source-conflict",
            canonical_url="https://example.test/a",
            source_family="domain:example.test",
            artifact_sha256="a" * 64,
        )
    )

    try:
        ledger.add_source(
            SourceRecord(
                source_id="source-conflict",
                canonical_url="https://example.test/b",
                source_family="domain:example.test",
                artifact_sha256="a" * 64,
            )
        )
    except ValueError as exc:
        assert "canonical_url" in str(exc)
    else:
        raise AssertionError("immutable source conflict was merged")


def test_same_domain_or_artifact_does_not_count_as_independent_corroboration() -> None:
    source_a = SourceRecord(
        source_id="source-a",
        canonical_url="https://example.test/a",
        registered_domain="example.test",
        source_family="domain:example.test",
        artifact_sha256="a" * 64,
    )
    source_b = SourceRecord(
        source_id="source-b",
        canonical_url="https://example.test/b",
        registered_domain="example.test",
        source_family="domain:example.test",
        artifact_sha256="b" * 64,
    )
    source_c = SourceRecord(
        source_id="source-c",
        canonical_url="https://other.test/c",
        registered_domain="other.test",
        source_family="domain:other.test",
        artifact_sha256="a" * 64,
    )
    records = [
        _supporting_record(source_a, "evidence-a"),
        _supporting_record(source_b, "evidence-b"),
    ]
    assert _direction_is_decisive(records, {source_a.source_id: source_a, source_b.source_id: source_b}) is False

    records[1] = _supporting_record(source_c, "evidence-c")
    assert _direction_is_decisive(records, {source_a.source_id: source_a, source_c.source_id: source_c}) is False


def _supporting_record(source: SourceRecord, evidence_id: str) -> EvidenceRecord:
    return EvidenceRecord(
        evidence_id=evidence_id,
        claim_id="claim-q0",
        source_id=source.source_id,
        function_call_id=f"call-{evidence_id}",
        tool_name="visit",
        evidence_kind="web_span",
        exact_text="support",
        span_start=0,
        span_end=7,
        artifact_sha256=source.artifact_sha256,
        retrieved_at="2026-07-11T00:00:00+00:00",
        stance="support",
        quality="moderate",
    )


def test_browse_stance_for_search_query_instead_of_claim_is_rejected() -> None:
    orchestrator = _orchestrator()
    step = _successful_step()
    payload = json.loads(step.tool_result)
    payload["goal"] = "Reuters flood image keywords"
    step.tool_result = json.dumps(payload)

    canonical = orchestrator._canonicalize_model_evidence(
        _evidence(),
        [step],
        _plan(),
    )

    assert canonical is None


def test_visual_anomaly_cannot_refute_external_event_claim(tmp_path) -> None:
    image_path = tmp_path / "input.jpg"
    image_path.write_bytes(b"external-event-visual-evidence")
    plan = VerificationPlan(
        questions=[
            InvestigationQuestion(
                question_id="q0",
                question="Did the politician join the party?",
                claim_text="The politician joined the party.",
                claim_scope="external_fact",
                priority=1,
            )
        ]
    )
    excerpt = "Commentary text was overlaid on a news screenshot."
    step = StageStep(
        round=1,
        stage_name="verification",
        action_type="tool_call",
        tool_name="analyze_visual_anomalies",
        tool_args={"__question_id": "q0"},
        tool_result=json.dumps(
            {
                "status": "success",
                "anomalies": [{"phenomenon": excerpt}],
                "overall_authenticity": "likely_manipulated",
                "notes": excerpt,
            }
        ),
        metadata={
            "tool_success": True,
            "function_call_id": "call-anomaly-q0",
            "observed_at": "2026-07-12T00:00:00+00:00",
        },
    )
    model_item = EvidenceItem(
        function_call_id="call-anomaly-q0",
        source="analyze_visual_anomalies",
        summary=excerpt,
        raw_excerpt=excerpt,
        direction="refutes",
        quality="moderate",
        tool_used="analyze_visual_anomalies",
        related_question="q0",
    )

    canonical = _orchestrator()._canonicalize_model_evidence(model_item, [step], plan)
    ledgers = compile_runtime_ledgers(
        build_verification_case(str(image_path), user_claim=plan.questions[0].claim_text),
        plan,
        VerificationResult(evidence=[model_item]),
        [step],
    )

    assert canonical is not None
    assert canonical.direction == "neutral"
    assert ledgers.evidence[0].stance == "neutral"
    assert ledgers.claims[0].status == "open"


def test_weak_opposing_ugc_signals_do_not_create_claim_conflict() -> None:
    source = SourceRecord(
        source_id="source-ugc",
        canonical_url="https://facebook.com/post",
        registered_domain="facebook.com",
        source_family="domain:facebook.com",
        source_class="ugc",
        artifact_sha256="a" * 64,
    )
    support = _supporting_record(source, "ugc-support")
    refute = support.model_copy(
        update={"evidence_id": "ugc-refute", "function_call_id": "call-ugc-refute", "stance": "refute"}
    )
    ledgers = VerificationLedgers(
        claims=[
            ClaimRecord(
                claim_id="claim-q0",
                question_id="q0",
                text="The politician joined the party.",
                claim_scope="external_fact",
            )
        ],
        sources=[source],
        evidence=[support, refute],
    )

    _update_claim_statuses(ledgers)

    assert ledgers.claims[0].status == "open"
    assert "source-independence" in ledgers.claims[0].unresolved_distinction


def test_exhausted_required_visual_question_reopens_web_supported_claim() -> None:
    ledgers = VerificationLedgers(
        claims=[
            ClaimRecord(
                claim_id="claim-q0",
                question_id="q0",
                text="The visible banner reads X.",
                claim_scope="visible_content",
                status="supported",
            )
        ]
    )
    investigation = InvestigationState(
        visual_questions=[
            VisualQuestion(
                visual_question_id="vq-q0",
                claim_id="claim-q0",
                source_evidence_id="e-web",
                target_bbox=[0.1, 0.1, 0.9, 0.4],
                expected_property="Whether the banner reads X",
                recommended_tools=["ocr_with_position"],
                status="exhausted",
                failed_attempts=2,
            )
        ]
    )

    Orchestrator._gate_claims_on_pending_visual_questions(ledgers, investigation)

    assert ledgers.claims[0].status == "open"
    assert "could not be observed after two real attempts" in (
        ledgers.claims[0].unresolved_distinction
    )

    reasons = derive_unverifiable_reasons(ledgers, "hard_budget_exhausted")
    assert UnverifiableReason.UNREADABLE_REGION in reasons
    assert UnverifiableReason.BUDGET_EXHAUSTED in reasons
