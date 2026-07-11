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
    def render_for_verification(perception: PerceptionReport, plan: VerificationPlan) -> str:
        parts = ["## Verification Context", ""]
        parts.append(f"Scene: {perception.scene_description or '(empty)'}")
        parts.append(f"Image type: {perception.image_type or 'photo'}")
        parts.append(f"Intent: {plan.image_intent or '(empty)'}")
        parts.append(f"Risk: {plan.risk_assessment or '(empty)'}")
        parts.append(f"Plan revision: {plan.revision}")
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
        parts = ["## Judgment Context", ""]
        parts.append(f"Scene: {perception.scene_description or '(empty)'}")
        parts.append(f"Image type: {perception.image_type or 'photo'}")
        parts.append(f"Intent: {plan.image_intent or '(empty)'}")
        parts.append(f"Risk: {plan.risk_assessment or '(empty)'}")
        parts.append(f"Verification assessment: {verification.authenticity_assessment}")

        if ledgers is not None:
            parts.append("")
            parts.append("Claim ledger (use these exact claim_id values):")
            for claim in ledgers.claims:
                parts.append(
                    f"- {claim.claim_id}: status={claim.status}, criticality={claim.criticality}, "
                    f"text={claim.text}"
                )
            parts.append("")
            parts.append("Eligible evidence ledger (use these exact evidence_id values):")
            for evidence in ledgers.evidence:
                parts.append(
                    f"- {evidence.evidence_id}: claim={evidence.claim_id}, "
                    f"stance={evidence.stance}, quality={evidence.quality}, "
                    f"source={evidence.source_id}, exact_text={evidence.exact_text[:500]}"
                )

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
    ) -> str:
        """Render only unresolved gaps and compact evidence for plan revision."""

        unresolved = set(audit.unresolved_priority_questions)
        resolutions = {
            item.question_id: item for item in audit.question_resolutions
        }
        parts = ["## Replanning Context", ""]
        parts.append(f"Scene: {perception.scene_description[:500] or '(empty)'}")
        parts.append(f"Intent: {plan.image_intent[:500] or '(empty)'}")
        parts.append("Unresolved questions to update:")
        for question in plan.questions:
            if question.question_id not in unresolved:
                continue
            resolution = resolutions.get(question.question_id)
            gap = resolution.remaining_gap if resolution else "No grounded answer."
            parts.append(
                f"- [{question.question_id}] {question.question[:500]} | "
                f"gap: {gap[:500]}"
            )
            if question.suggested_queries:
                parts.append(
                    "  previous queries: "
                    + " ; ".join(query[:160] for query in question.suggested_queries[:4])
                )

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
            "Return one update for every unresolved id and no updates for any other id."
        )
        return "\n".join(parts)
