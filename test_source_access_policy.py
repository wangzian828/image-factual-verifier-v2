from __future__ import annotations

import asyncio
import json
from typing import Any

from src.orchestrator.source_access import (
    benchmark_source_access_policy,
    url_variants,
)
from src.orchestrator.source_provenance import classify_source
from src.orchestrator.stage_runner import StageRunner
from src.integrations.search.serper import (
    SerperImageSearchClient,
    SerperTextSearchClient,
)
from pydantic import ValidationError
from test_support_models import ToolStageOutput, TruthAptQuestion
from src.tools.reverse_image_search import ReverseImageSearchTool
from src.tools.text_search import TextSearchTool
from src.tools.visit import VisitTool


FACT_CHECK_URL = (
    "https://web.archive.org/web/20230424080500/"
    "https://srilanka.factcrescendo.com/english/benchmark-answer/"
)


def test_compound_public_suffix_is_classified_as_official_metadata() -> None:
    identity = classify_source(
        "https://www.antarctica.gov.au/about-antarctica/animals/penguins/"
    )

    assert identity.source_class == "official"


class SearchClient:
    def search(self, query: str, **_kwargs):
        return {
            "query": query,
            "results": [
                {
                    "title": "Fact-check answer: claim is false",
                    "url": "https://srilanka.factcrescendo.com/english/benchmark-answer/",
                    "snippet": "This snippet contains the benchmark verdict.",
                },
                {
                    "title": "Independent primary report",
                    "url": "https://independent.example/report",
                    "snippet": "A non-benchmark lead.",
                },
            ],
            "answer_box": {"answer": "claim is false"},
            "knowledge_graph": {"description": "benchmark answer"},
        }


class BrowseClient:
    def __init__(self) -> None:
        self.urls: list[str] = []
        self.policy = None

    def set_source_access_policy(self, policy) -> None:
        self.policy = policy

    def visit_many(self, urls, goal):
        self.urls.extend(urls)
        return {
            "status": "success",
            "visits": [],
            "selected_url": "",
            "summary": "",
            "evidence": "",
            "rationale": "",
            "relevance": "low",
            "stance": "unclear",
            "directness": "none",
            "artifact_sha256": "",
            "evidence_span": {},
            "retrieved_at": "",
            "injection_flags": [],
            "evidence_eligible": False,
        }


class VlmClient:
    def create_image_json(self, **_kwargs):
        return {"query": "semantic image query", "keywords": []}


class ImageSearchClient:
    def search(self, **_kwargs):
        return [
            {
                "title": "Leaking semantic match",
                "url": "https://srilanka.factcrescendo.com/english/benchmark-answer",
                "image_url": "https://srilanka.factcrescendo.com/images/gold.jpg",
            },
            {
                "title": "Independent match",
                "url": "https://independent.example/photo",
                "image_url": "https://independent.example/photo.jpg",
            },
        ]


class VisualSearchClient:
    def search(self, *_args, **_kwargs):
        return {
            "status": "success",
            "provider": "fixture",
            "results": [
                {
                    "title": "Leaking Lens match",
                    "url": FACT_CHECK_URL,
                    "image_url": "https://srilanka.factcrescendo.com/images/gold.jpg",
                    "snippet": "The hidden verdict appears here.",
                }
            ],
            "timings": {},
        }


def _policy():
    return benchmark_source_access_policy([FACT_CHECK_URL], policy_id="fixture-eval")


def test_wayback_policy_blocks_outer_and_embedded_fact_check_urls() -> None:
    variants = url_variants(FACT_CHECK_URL)
    assert any("web.archive.org" in value for value in variants)
    assert any("factcrescendo.com" in value for value in variants)
    policy = _policy()
    assert not policy.allows(FACT_CHECK_URL)
    assert not policy.allows("https://sub.srilanka.factcrescendo.com/answer")
    assert policy.allows("https://independent.example/report")


def test_wordpress_proxy_for_excluded_origin_is_blocked() -> None:
    policy = _policy()
    proxied = (
        "https://i0.wp.com/srilanka.factcrescendo.com/wp-content/uploads/"
        "2023/04/image.png?resize=444,447&ssl=1"
    )

    assert not policy.allows(proxied)


def test_provider_row_naming_excluded_fact_checker_is_filtered() -> None:
    policy = _policy()
    rows, blocked = policy.filter_rows(
        [
            {
                "title": "Fact Crescendo explains the claim",
                "url": "https://independent.example/repost",
                "image_url": "https://cdn.example/image.jpg",
            },
            {
                "title": "Independent official statement",
                "url": "https://independent.example/primary",
                "image_url": "https://cdn.example/primary.jpg",
            },
        ]
    )

    assert blocked == 1
    assert [row["url"] for row in rows] == ["https://independent.example/primary"]


def test_query_naming_excluded_fact_checker_is_rejected_without_domain() -> None:
    policy = benchmark_source_access_policy(
        [FACT_CHECK_URL, "https://factcheck.afp.com/doc.afp.com.fixture"],
        policy_id="fixture-eval",
    )

    assert policy.blocked_query_reference("Fact Crescendo Sajith Ceylon Newsline")
    assert policy.blocked_query_reference("AFP fact check Sajith claim")
    assert policy.blocked_query_reference("independent official Sajith statement") == ""


def test_mainstream_fact_check_origin_is_exact_url_only() -> None:
    policy = benchmark_source_access_policy(
        ["https://www.newsweek.com/china-us-life-expectancy-birth-2021-fact-check-1740991"]
    )
    assert not policy.allows(
        "https://www.newsweek.com/china-us-life-expectancy-birth-2021-fact-check-1740991"
    )
    assert policy.allows("https://www.newsweek.com/unrelated-primary-report")


def test_fact_check_subdomain_scope_does_not_block_parent_news_domain() -> None:
    policy = benchmark_source_access_policy(
        ["https://factcheck.afp.com/doc.afp.com.example"]
    )
    assert not policy.allows("https://factcheck.afp.com/another-answer")
    assert policy.allows("https://www.afp.com/primary-report")


def test_policy_does_not_block_unlisted_fact_check_domains() -> None:
    policy = benchmark_source_access_policy(
        ["https://seed.example/hidden-benchmark-answer"]
    )

    assert policy.allows("https://www.politifact.com/factchecks/example/")
    assert policy.blocked_query_reference("PolitiFact article about the event") == ""
    assert policy.blocked_content_reference("PolitiFact reviewed this claim") == ""
    assert policy.allows("https://independent.example/primary-report")


def test_text_search_keeps_result_from_unlisted_fact_check_domain() -> None:
    class MixedSearch:
        def search(self, query: str, **_kwargs):
            return {
                "query": query,
                "results": [
                    {
                        "title": "PolitiFact review",
                        "url": "https://www.politifact.com/factchecks/example/",
                        "snippet": "A ready-made verdict.",
                    },
                    {
                        "title": "Independent primary report",
                        "url": "https://independent.example/primary",
                        "snippet": "The underlying event record.",
                    },
                ],
                "answer_box": {"answer": "ready-made verdict"},
                "knowledge_graph": None,
            }

    policy = benchmark_source_access_policy(
        ["https://seed.example/hidden-benchmark-answer"]
    )
    tool = TextSearchTool(client=MixedSearch())
    tool.set_source_access_policy(policy)

    result = tool.search("underlying event record")

    response = result["queries"][0]
    assert [item["url"] for item in response["results"]] == [
        "https://www.politifact.com/factchecks/example/",
        "https://independent.example/primary"
    ]
    assert "policy_filtered_count" not in response
    assert response["answer_box"] is None


def test_serper_adds_policy_exclusions_before_provider_call() -> None:
    policy = _policy()
    client = SerperTextSearchClient(api_key="test-key")
    payloads: list[dict[str, Any]] = []

    def fake_post(payload, _headers):
        payloads.append(payload)
        return {
            "organic": [
                {
                    "title": "Independent report",
                    "link": "https://independent.example/report",
                    "snippet": "Underlying event record.",
                }
            ]
        }

    client._post_json = fake_post  # type: ignore[method-assign]
    client.set_source_access_policy(policy)
    try:
        result = client.search("underlying event record", top_k=1)
    finally:
        client.close()

    assert result["query"] == "underlying event record"
    assert payloads
    assert payloads[0]["q"].startswith("underlying event record ")
    assert "-site:factcrescendo.com" in payloads[0]["q"]
    assert "factcrescendo.com" not in result["results"][0]["url"]


def test_serper_image_search_adds_policy_exclusions_before_provider_call() -> None:
    policy = _policy()
    client = SerperImageSearchClient(api_key="test-key")
    payloads: list[dict[str, Any]] = []

    def fake_post(payload, _headers):
        payloads.append(payload)
        return {
            "images": [
                {
                    "title": "Independent image",
                    "link": "https://independent.example/photo",
                    "imageUrl": "https://cdn.example/photo.jpg",
                }
            ]
        }

    client._post_json = fake_post  # type: ignore[method-assign]
    client.set_source_access_policy(policy)
    try:
        result = client.search("bridge event", top_k=1)
    finally:
        client.close()

    assert payloads
    assert payloads[0]["q"].startswith("bridge event ")
    assert "-site:factcrescendo.com" in payloads[0]["q"]
    assert result[0]["url"] == "https://independent.example/photo"


def test_text_search_filters_results_and_removes_aggregates() -> None:
    tool = TextSearchTool(client=SearchClient())
    tool.set_source_access_policy(_policy())

    result = tool.search("claim keywords")

    assert result["status"] == "success"
    response = result["queries"][0]
    serialized = json.dumps(result)
    assert "benchmark verdict" not in serialized
    assert "claim is false" not in serialized
    assert [item["url"] for item in response["results"]] == [
        "https://independent.example/report"
    ]
    assert response["answer_box"] is None
    assert response["knowledge_graph"] is None
    assert "visited_pages" not in response
    assert "evidence" not in response


def test_text_search_rejects_query_that_targets_excluded_domain_before_provider_call() -> None:
    class RecordingSearch:
        called = False

        def search(self, *_args, **_kwargs):
            self.called = True
            raise AssertionError("policy-blocked query must not reach search provider")

    search = RecordingSearch()
    tool = TextSearchTool(client=search)
    tool.set_source_access_policy(_policy())

    result = tool.search("site:factcrescendo.com Sajith Premadasa UNP")

    assert result["status"] == "error"
    assert "independent open-web sources" in result["error"]
    assert search.called is False


def test_direct_visit_refuses_blocked_url_without_calling_provider() -> None:
    class Provider:
        called = False

        def visit(self, _url, **_kwargs):
            self.called = True
            raise AssertionError("blocked URL must not reach fetch provider")

    provider = Provider()
    tool = VisitTool(client=provider)
    tool.set_source_access_policy(_policy())
    result = tool.visit(
        FACT_CHECK_URL,
        image_claim="claim",
        retrieval_goal="claim",
    )
    assert result["status"] == "error"
    assert provider.called is False


def test_reverse_search_removes_blocked_pages_and_reference_images() -> None:
    tool = ReverseImageSearchTool(
        vlm_client=VlmClient(),
        image_search_client=ImageSearchClient(),
        visual_search_client=VisualSearchClient(),
        lens_client=object(),
    )
    tool.set_source_access_policy(_policy())

    result = tool.search("image.jpg", branch="semantic")
    serialized = json.dumps(result)
    assert "factcrescendo" not in serialized
    assert "hidden verdict" not in serialized
    assert result["candidate_page_urls"] == ["https://independent.example/photo"]
    assert result["reference_image_candidates"] == [
        "https://independent.example/photo.jpg"
    ]


def test_react_tool_selection_is_not_forced_into_priority_rotation() -> None:
    runner = StageRunner(
        llm=object(),
        system_prompt="",
        tools=[],
        output_schema=ToolStageOutput,
        stage_name="verification",
        priority_question_ids=["q0", "q1"],
    )
    first = runner._priority_coverage_error(
        {"__question_id": "q0"},
        [],
    )
    assert first == ""
    from src.orchestrator.stage_runner import StageStep

    repeated = runner._priority_coverage_error(
        {"__question_id": "q0"},
        [StageStep(action_type="tool_call", tool_args={"__question_id": "q0"})],
    )
    assert repeated == ""

    pending_reinspection = runner._priority_coverage_error(
        {"__question_id": "q0", "visual_question_id": "vq-fabricated"},
        [
            StageStep(action_type="tool_call", tool_args={"__question_id": "q0"}),
            StageStep(action_type="tool_call", tool_args={"__question_id": "q0"}),
        ],
    )
    assert pending_reinspection == ""


def test_iteration_output_requires_every_unresolved_required_question_attempt() -> None:
    from src.orchestrator.stage_runner import StageStep

    runner = StageRunner(
        llm=object(),
        system_prompt="",
        tools=[],
        output_schema=ToolStageOutput,
        stage_name="verification",
        priority_question_ids=["q0"],
        supporting_question_ids=["q1"],
    )
    touched_p1 = [
        StageStep(action_type="tool_call", tool_args={"__question_id": "q0"})
    ]

    assert "q1" in runner._required_question_output_error(touched_p1)
    assert runner._required_question_output_error(
        [
            *touched_p1,
            StageStep(action_type="tool_call", tool_args={"__question_id": "q1"}),
        ]
    ) == ""


def test_scheduler_does_not_count_rejected_call_as_question_attempt() -> None:
    from src.orchestrator.stage_runner import StageStep

    runner = StageRunner(
        llm=object(),
        system_prompt="",
        tools=[],
        output_schema=ToolStageOutput,
        stage_name="verification",
        priority_question_ids=["q0", "q1"],
    )
    rejected = [
        StageStep(action_type="format_error", tool_args={"__question_id": "q0"})
    ]

    assert "q0" in runner._required_question_output_error(rejected)


def test_resolved_required_question_does_not_need_another_attempt_before_output() -> None:
    from src.orchestrator.stage_runner import StageStep

    resolved = StageStep(
        action_type="tool_call",
        tool_args={"__question_id": "q0"},
        metadata={
            "investigation_state_update": {
                "belief_delta": {
                    "claim_id": "claim-q0",
                    "new_status": "supported",
                }
            }
        },
    )
    runner = StageRunner(
        llm=object(),
        system_prompt="",
        tools=[],
        output_schema=ToolStageOutput,
        stage_name="verification",
        priority_question_ids=["q0", "q1"],
    )
    assert "q1" in runner._required_question_output_error([resolved])
    assert runner._required_question_output_error(
        [
            resolved,
            StageStep(action_type="tool_call", tool_args={"__question_id": "q1"}),
        ]
    ) == ""


def test_agent_control_state_exposes_full_pending_reinspect_spec() -> None:
    from src.orchestrator.stage_runner import StageStep

    spec = {
        "visual_question_id": "vq-required",
        "claim_id": "claim-q0",
        "source_discovery_id": "discovery-1",
        "source_evidence_id": None,
        "reference_image_url": "https://example.test/reference.jpg",
        "target_bbox": [0.0, 0.0, 1.0, 1.0],
        "expected_property": "Whether the images match.",
        "recommended_tools": ["compare_with_reference"],
        "status": "pending",
        "resolution_call_id": None,
        "failed_attempts": 0,
    }
    prior = StageStep(
        action_type="tool_call",
        metadata={
            "investigation_state_update": {
                "created_visual_questions": [spec],
                "resolved_visual_questions": [],
            }
        },
    )
    runner = StageRunner(
        llm=object(),
        system_prompt="",
        tools=[],
        output_schema=ToolStageOutput,
        stage_name="verification",
        prior_steps=[prior],
    )

    control = runner._agent_control_state()
    assert control["pending_visual_question_ids"] == ["vq-required"]
    assert control["pending_visual_questions"] == [spec]


def test_claim_and_retrieval_goal_are_runtime_bound_before_tool_execution() -> None:
    class Tool:
        name = "visit"
        parameters = {
            "type": "object",
            "properties": {
                "url": {"type": "string"},
                "image_claim": {"type": "string"},
                "retrieval_goal": {"type": "string"},
            },
            "required": ["url", "image_claim", "retrieval_goal"],
        }

        def __init__(self):
            self.params = None

        def call(self, params):
            self.params = params
            return {"status": "success", "visits": []}

    tool = Tool()
    runner = StageRunner(
        llm=object(),
        system_prompt="",
        tools=[tool],
        output_schema=ToolStageOutput,
        stage_name="verification",
        question_claims={"q0": "The immutable declarative claim."},
        question_evidence_goals={"q0": "Find the actual value."},
        attach_image=False,
    )
    args = runner._prepare_tool_args(
        "visit",
        {"question_id": "q0", "url": "https://example.test", "goal": "model query"},
        "",
    )
    asyncio.run(runner._execute_tool("visit", args))
    assert "goal" not in tool.params
    assert tool.params["image_claim"] == "The immutable declarative claim."
    assert tool.params["retrieval_goal"] == "Find the actual value."


def test_stage_runner_sanitizes_missed_blocked_rows_before_context_or_ledger() -> None:
    class LeakyTool:
        name = "reverse_image_search"
        parameters = {"type": "object", "properties": {}, "required": []}

        def call(self, _params):
            return {
                "status": "success",
                "lens_results": [
                    {
                        "url": FACT_CHECK_URL,
                        "title": "Gold verdict",
                        "snippet": "The exact benchmark answer.",
                    },
                    {
                        "url": "https://independent.example/lead",
                        "title": "Independent lead",
                    },
                ],
            }

    runner = StageRunner(
        llm=object(),
        system_prompt="",
        tools=[LeakyTool()],
        output_schema=ToolStageOutput,
        stage_name="verification",
        source_access_policy=_policy(),
        attach_image=False,
    )
    serialized, metadata = asyncio.run(
        runner._execute_tool("reverse_image_search", {})
    )
    assert metadata["tool_success"] is True
    assert "benchmark answer" not in serialized
    assert "Gold verdict" not in serialized
    assert "independent.example" in serialized


def test_investigation_question_requires_truth_apt_claim_text() -> None:
    try:
        TruthAptQuestion(question_id="q0", question="Where did this come from?")
    except ValidationError:
        pass
    else:
        raise AssertionError("claim_text must be schema-required")
