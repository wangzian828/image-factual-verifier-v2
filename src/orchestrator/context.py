# -*- coding: utf-8 -*-
"""Context rendering for each pipeline stage."""
from __future__ import annotations

from src.orchestrator.state import (
    CoverageAudit,
    PerceptionReport,
    VerificationPlan,
    VerificationResult,
    VerificationCase,
    VerificationLedgers,
)


class ContextRenderer:
    """Render compact structured context for LLM stages."""

    @staticmethod
    def _time_semantics(verification_case: VerificationCase | None) -> str:
        if verification_case is not None and verification_case.claim_observed_at:
            return (
                f"Claim time semantics: evaluate as of "
                f"{verification_case.claim_observed_at}; an event first occurring later "
                "cannot decide the claim as it stood then, although a later source may "
                "report evidence about the earlier state."
            )
        return "Claim time semantics: current time."

    @staticmethod
    def render_for_planning(
        perception: PerceptionReport,
        verification_case: VerificationCase | None = None,
    ) -> str:
        parts = ["## Perception", ""]
        if verification_case is not None:
            parts.append(f"Claim mode: {verification_case.claim_mode.value}")
            if verification_case.user_claim:
                parts.append(f"External claim (trusted runtime input): {verification_case.user_claim}")
            elif verification_case.claim_surface:
                parts.append(f"Embedded claim surface recovered from pixels: {verification_case.claim_surface}")
        parts.append(ContextRenderer._time_semantics(verification_case))
        parts.append(f"Scene: {perception.scene_description or '(empty)'}")
        parts.append(f"Image type: {perception.image_type or 'photo'}")

        if perception.entities:
            parts.append("")
            parts.append("Entities:")
            for entity in perception.entities[:10]:
                attrs = ", ".join(f"{k}={v}" for k, v in entity.attributes.items())
                suffix = f" ({attrs})" if attrs else ""
                parts.append(f"- [{entity.entity_type}] {entity.name}{suffix}")

        if perception.text_regions:
            parts.append("")
            parts.append("Visible text:")
            for region in perception.text_regions[:10]:
                text = region.text.replace("\n", " ").strip()
                if not text:
                    continue
                parts.append(f'- "{text[:120]}" [{region.language}]')

        return "\n".join(parts)

    @staticmethod
    def render_for_verification(
        perception: PerceptionReport,
        plan: VerificationPlan,
        verification_case: VerificationCase | None = None,
    ) -> str:
        parts = ["## Verification Context", ""]
        parts.append(f"Scene: {perception.scene_description or '(empty)'}")
        parts.append(f"Image type: {perception.image_type or 'photo'}")
        parts.append(f"Intent: {plan.image_intent or '(empty)'}")
        parts.append(f"Risk: {plan.risk_assessment or '(empty)'}")
        parts.append(f"Plan revision: {plan.revision}")
        parts.append(ContextRenderer._time_semantics(verification_case))
        if plan.revision_reason:
            parts.append(f"Revision reason: {plan.revision_reason}")
        parts.append(f"Trying to look real: {plan.is_trying_to_be_real}")

        if perception.entities:
            names = [entity.name for entity in perception.entities[:8] if entity.name]
            if names:
                parts.append(f"Entities: {', '.join(names)}")

        if perception.text_regions:
            texts = []
            for region in perception.text_regions[:5]:
                text = region.text.replace("\n", " ").strip()
                if text:
                    texts.append(text[:80])
            if texts:
                parts.append(f"Text: {'; '.join(texts)}")

        if plan.questions:
            parts.append("")
            parts.append("Investigation questions:")
            for question in plan.questions[:6]:
                row = f"- [{question.question_id}] P{question.priority}: {question.question}"
                row += f" | immutable claim: {question.claim_text}"
                row += f" | scope: {question.claim_scope}"
                if question.why:
                    row += f" | why: {question.why}"
                if question.suggested_tools:
                    row += f" | tools: {', '.join(question.suggested_tools[:5])}"
                if question.suggested_queries:
                    row += f" | queries: {' ; '.join(question.suggested_queries[:3])}"
                parts.append(row)

        return "\n".join(parts)

    @staticmethod
    def render_for_judgment(
        perception: PerceptionReport,
        plan: VerificationPlan,
        verification: VerificationResult,
        ledgers: VerificationLedgers | None = None,
    ) -> str:
        if ledgers is not None:
            parts = [
                "## Ledger Judgment Context",
                "",
                "Select only exact claim_id and evidence_id values listed below.",
                "The orchestrator will compile all final prose from these records.",
                "",
                "Decisive claims:",
            ]
            for claim in ledgers.claims:
                if claim.criticality != "decisive":
                    continue
                parts.append(
                    f"- {claim.claim_id}: status={claim.status}; text={claim.text[:500]}"
                )
            parts.append("")
            parts.append("Eligible non-neutral evidence:")
            for evidence in ledgers.evidence:
                if evidence.stance == "neutral":
                    continue
                parts.append(
                    f"- {evidence.evidence_id}: claim={evidence.claim_id}; "
                    f"stance={evidence.stance}; quality={evidence.quality}; "
                    f"text={evidence.exact_text[:280]}"
                )
            return "\n".join(parts)

        parts = ["## Judgment Context", ""]
        parts.append(f"Scene: {perception.scene_description or '(empty)'}")
        parts.append(f"Image type: {perception.image_type or 'photo'}")
        parts.append(f"Intent: {plan.image_intent or '(empty)'}")
        parts.append(f"Risk: {plan.risk_assessment or '(empty)'}")
        parts.append(f"Verification assessment: {verification.authenticity_assessment}")

        if verification.key_findings:
            parts.append("")
            parts.append("Key findings:")
            for finding in verification.key_findings[:10]:
                parts.append(f"- {finding}")

        if verification.evidence:
            parts.append("")
            parts.append("Evidence:")
            for evidence in verification.evidence[:12]:
                row = (
                    f"- [{evidence.direction}/{evidence.quality}] "
                    f"{evidence.summary} (via {evidence.tool_used}, "
                    f"call={evidence.function_call_id}, q={evidence.related_question})"
                )
                parts.append(row)
                if evidence.raw_excerpt:
                    parts.append(f'  excerpt: "{evidence.raw_excerpt[:280]}"')

        if verification.visual_anomalies:
            parts.append("")
            parts.append("Visual anomalies:")
            for anomaly in verification.visual_anomalies[:8]:
                label = anomaly.name or anomaly.phenomenon or "anomaly"
                parts.append(
                    f"- {label} (call={anomaly.function_call_id}, q={anomaly.related_question})"
                )

        return "\n".join(parts)

    @staticmethod
    def render_for_replanning(
        perception: PerceptionReport,
        plan: VerificationPlan,
        audit: CoverageAudit,
        verification: VerificationResult,
        available_tools: list[str] | None = None,
        investigation_state: Any = None,
        ledgers: VerificationLedgers | None = None,
        verification_case: VerificationCase | None = None,
    ) -> str:
        """Render only unresolved gaps and compact evidence for plan revision."""

        unresolved = set(audit.unresolved_priority_questions)
        unresolved.update(audit.unattempted_supporting_questions)
        pending_visual = []
        if investigation_state is not None:
            pending_visual = [
                item
                for item in investigation_state.visual_questions
                if item.status == "pending"
            ]
            unresolved.update(
                item.claim_id.removeprefix("claim-")
                for item in pending_visual
                if item.claim_id.startswith("claim-")
            )
        resolutions = {
            item.question_id: item for item in audit.question_resolutions
        }
        parts = ["## Replanning Context", ""]
        parts.append(f"Scene: {perception.scene_description[:500] or '(empty)'}")
        parts.append(f"Intent: {plan.image_intent[:500] or '(empty)'}")
        parts.append(ContextRenderer._time_semantics(verification_case))
        if available_tools:
            parts.append("Available verification tools: " + ", ".join(available_tools))
        parts.append("Unresolved questions to update:")
        for question in plan.questions:
            if question.question_id not in unresolved:
                continue
            resolution = resolutions.get(question.question_id)
            gap = resolution.remaining_gap if resolution else "No grounded answer."
            parts.append(
                f"- [{question.question_id}] P{question.priority} | {question.question[:500]} | "
                f"immutable claim: {question.claim_text[:500]} | "
                f"scope: {question.claim_scope} | "
                f"gap: {gap[:500]}"
            )
            if ledgers is not None:
                ledger_claim = next(
                    (
                        item
                        for item in ledgers.claims
                        if item.question_id == question.question_id
                        or item.claim_id == f"claim-{question.question_id}"
                    ),
                    None,
                )
                if ledger_claim is not None:
                    distinction = ledger_claim.unresolved_distinction or "(none)"
                    parts.append(
                        f"  ledger: status={ledger_claim.status}; "
                        f"unresolved_distinction={distinction[:500]}"
                    )
                    sources = {item.source_id: item for item in ledgers.sources}
                    claim_evidence = [
                        item
                        for item in ledgers.evidence
                        if item.claim_id == ledger_claim.claim_id
                    ]
                    for evidence in claim_evidence[:8]:
                        source = sources.get(evidence.source_id)
                        source_class = source.source_class if source else "unknown"
                        source_family = source.source_family if source else "unknown"
                        parts.append(
                            f"  evidence: stance={evidence.stance}; "
                            f"source_class={source_class}; source_family={source_family}"
                        )
            if question.suggested_queries:
                parts.append(
                    "  previous queries: "
                    + " ; ".join(query[:160] for query in question.suggested_queries[:4])
                )

        if pending_visual:
            parts.append("Pending ReInspect specifications:")
            for item in pending_visual:
                parts.append(
                    f"- [{item.visual_question_id}] claim={item.claim_id} | "
                    f"target_bbox={item.target_bbox} | expected={item.expected_property[:500]} | "
                    f"recommended_tools={', '.join(item.recommended_tools)}"
                )

        if ledgers is None:
            relevant = [
                item
                for item in verification.evidence
                if item.related_question in unresolved
            ]
            if relevant:
                parts.append("Evidence already collected for unresolved questions:")
                for item in relevant[:8]:
                    parts.append(
                        f"- [{item.related_question}] {item.summary[:400]} "
                        f"(via {item.tool_used})"
                    )
            else:
                parts.append("Evidence already collected for unresolved questions: none")

        parts.append(
            "Return one update for every listed unresolved or ReInspect-blocked id "
            "and no updates for any other id."
        )
        return "\n".join(parts)
