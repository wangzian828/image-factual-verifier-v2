"""Dependency-free standalone HTML renderer for unified-ReAct traces."""
from __future__ import annotations

import base64
import html
import json
import os
from typing import Any, Dict, Iterable, List

from src.redaction import sanitize_for_persistence


def save_trace_html(trace_data: Dict[str, Any], output_path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write(_render_trace_html(sanitize_for_persistence(trace_data)))


def render_trace_file(json_path: str, output_path: str) -> None:
    with open(json_path, encoding="utf-8") as handle:
        save_trace_html(json.load(handle), output_path)


def _render_trace_html(trace_data: Dict[str, Any]) -> str:
    state = trace_data.get("state") if isinstance(trace_data.get("state"), dict) else {}
    image_path = str(trace_data.get("image_path") or state.get("image_path") or "")
    judgment = trace_data.get("judgment") if isinstance(trace_data.get("judgment"), dict) else {}
    if not judgment and isinstance(state.get("judgment"), dict):
        judgment = state["judgment"]
    steps = state.get("all_steps", []) if isinstance(state.get("all_steps"), list) else []
    investigation = state.get("investigation_state", {}) if isinstance(state.get("investigation_state"), dict) else {}
    audits = investigation.get("discrepancy_coverage_audits", []) if isinstance(investigation.get("discrepancy_coverage_audits"), list) else []
    timings = state.get("stage_timings", {}) if isinstance(state.get("stage_timings"), dict) else {}
    token_usage = trace_data.get("token_usage") or state.get("token_usage") or {}
    embedded = _embed_local_image(image_path)
    stage_flow = (
        '<div class="stage">Unified ReAct</div><span class="arrow">&#8596;</span>'
        '<div class="stage">Reflection / Decision</div><span class="arrow">&#8594;</span>'
        '<div class="stage">unified-react Judgment</div>'
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Image Verification Trace</title>
<style>
:root{{--ink:#17202a;--muted:#5e6b75;--line:#d8dee4;--paper:#fff;--bg:#f4f6f8;--blue:#1769aa;--green:#26734d;--red:#b42318;--amber:#9a6700}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font-family:Segoe UI,Arial,sans-serif;font-size:14px;letter-spacing:0}}
.wrap{{max-width:1280px;margin:auto;padding:20px}}h1{{font-size:26px;margin:0 0 14px}}h2{{font-size:18px;margin:0 0 12px}}h3{{font-size:14px;margin:14px 0 7px}}
.band{{background:var(--paper);border:1px solid var(--line);border-radius:7px;padding:16px;margin-bottom:14px}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:10px}}
.kv{{min-width:0}}.kv b{{display:block;color:var(--muted);font-size:11px;text-transform:uppercase;margin-bottom:3px}}.pipeline{{display:flex;gap:8px;align-items:center;overflow:auto;padding:4px 0}}
.stage{{border-left:4px solid var(--blue);background:#f7fafc;padding:8px 10px;min-width:135px}}.arrow{{color:var(--muted)}}.two{{display:grid;grid-template-columns:minmax(260px,360px) 1fr;gap:14px}}
.image img{{display:block;max-width:100%;max-height:440px;border:1px solid var(--line)}}.revision,.audit,.step{{border-top:1px solid var(--line);padding-top:12px;margin-top:12px}}
.revision:first-child,.audit:first-child,.step:first-child{{border-top:0;padding-top:0;margin-top:0}}.badge{{display:inline-block;background:#e8f1f8;color:#174f78;padding:3px 7px;border-radius:4px;margin:0 5px 4px 0;font-size:11px}}
.ok{{background:#e7f6ec;color:var(--green)}}.bad{{background:#feeceb;color:var(--red)}}.warn{{background:#fff3cd;color:var(--amber)}}code{{font-family:Consolas,monospace}}pre{{white-space:pre-wrap;word-break:break-word;overflow:auto;background:#f6f8fa;border:1px solid #e7eaee;padding:10px;margin:7px 0 0;font-size:12px}}
.interaction-chain{{display:grid;grid-template-columns:minmax(0,1fr) auto minmax(0,1fr);gap:8px;align-items:stretch;margin:8px 0 12px;padding:9px;background:#f6f8fa;border:1px solid #e7eaee}}.interaction-node{{min-width:0;padding:6px 8px;border-left:3px solid var(--line);background:var(--paper)}}.interaction-node.current{{border-left-color:var(--blue)}}.interaction-key{{display:block;color:var(--muted);font-size:11px;margin-bottom:3px}}.interaction-node code{{display:block;white-space:normal;overflow-wrap:anywhere;word-break:break-word}}.chain-arrow{{align-self:center;color:var(--muted);font-size:18px;line-height:1}}
table{{border-collapse:collapse;width:100%;font-size:12px}}th,td{{border:1px solid var(--line);padding:7px;text-align:left;vertical-align:top}}th{{background:#f6f8fa}}ul{{margin:6px 0;padding-left:20px}}
@media(max-width:760px){{.wrap{{padding:10px}}.two{{grid-template-columns:1fr}}.pipeline{{align-items:stretch}}.stage{{min-width:125px}}.interaction-chain{{grid-template-columns:minmax(0,1fr)}}.chain-arrow{{justify-self:center;transform:rotate(90deg)}}}}
</style>
</head>
<body><main class="wrap">
<section class="band"><h1>Image Verification Trace</h1>{_render_meta(trace_data, state, judgment, image_path, token_usage)}</section>
<section class="band"><h2>Stage Flow</h2><div class="pipeline">{stage_flow}</div></section>
<div class="two"><div>{_render_image(embedded)}{_render_judgment(judgment, trace_data)}{_render_timings(timings)}</div><div>{_render_audits(audits)}</div></div>
{_render_visual_fact_investigation(investigation)}
<section class="band"><h2>Agent Trajectory</h2>{''.join(_render_step(step) for step in steps if isinstance(step, dict)) or '<p>No recorded steps.</p>'}</section>
</main></body></html>"""


def _render_meta(trace: Dict[str, Any], state: Dict[str, Any], judgment: Dict[str, Any], image_path: str, tokens: Dict[str, Any]) -> str:
    values = [
        ("Image", os.path.basename(image_path) or "unknown"),
        ("Verdict", judgment.get("verdict", trace.get("verdict", "unverifiable"))),
        ("Confidence", _number(judgment.get("confidence", trace.get("confidence", 0.0)), 2)),
        ("Termination", trace.get("termination", state.get("termination", ""))),
        ("Tool Calls", trace.get("total_tool_calls", state.get("total_tool_calls", 0))),
        ("LLM Calls", trace.get("llm_api_calls", state.get("llm_api_calls", 0))),
        ("Tokens", f"{tokens.get('prompt', 0)} in / {tokens.get('completion', 0)} out"),
        ("Time", f"{_number(trace.get('time_taken', 0.0), 1)}s"),
    ]
    return '<div class="grid">' + ''.join(f'<div class="kv"><b>{_e(key)}</b>{_e(value)}</div>' for key, value in values) + '</div>'


def _render_image(data_url: str) -> str:
    if not data_url:
        return '<section class="band"><h2>Input Image</h2><p>Image is not available locally.</p></section>'
    return f'<section class="band image"><h2>Input Image</h2><img src="{data_url}" alt="Input image"></section>'


def _render_judgment(judgment: Dict[str, Any], trace: Dict[str, Any]) -> str:
    assessment = judgment.get("overall_assessment", trace.get("overall_assessment", ""))
    basis = trace.get("verdict_basis")
    return (
        '<section class="band"><h2>Final Judgment</h2>'
        + f'<p>{_e(assessment)}</p>'
        + _pre(judgment.get("reasoning_chain"))
        + _list("Key Evidence", judgment.get("key_evidence"))
        + _list("Anomalies", judgment.get("anomalies"))
        + _labeled_json("Verdict Basis", basis)
        + '</section>'
    )


def _render_audits(audits: List[Any]) -> str:
    blocks: List[str] = []
    for audit in audits:
        if not isinstance(audit, dict):
            continue
        complete = bool(audit.get("complete"))
        stop_reason = str(
            audit.get("stop_reason") or ("complete" if complete else "continue")
        )
        resolutions = audit.get("question_resolutions", []) if isinstance(audit.get("question_resolutions"), list) else []
        fact_resolutions = audit.get("facts", []) if isinstance(audit.get("facts"), list) else []
        rows = ''.join(
            f'<tr><td>{_e(item.get("question_id", ""))}</td><td>{_status(item.get("status", ""))}</td><td>{_e(item.get("tool_attempts", 0))}</td><td>{_e(item.get("evidence_count", 0))}</td><td>{_e(item.get("remaining_gap", ""))}</td></tr>'
            for item in resolutions if isinstance(item, dict)
        )
        if fact_resolutions:
            rows = ''.join(
                f'<tr><td><code>{_e(item.get("fact_id", ""))}</code></td>'
                f'<td>{_status(item.get("status", ""))}</td>'
                f'<td>{_e(len(item.get("finding_ids", []) or []))}</td>'
                f'<td>{_e(len(item.get("evidence_ids", []) or []))}</td>'
                f'<td>{_e(item.get("reason", ""))}</td></tr>'
                for item in fact_resolutions if isinstance(item, dict)
            )
        labels = {
            "coverage_complete": "coverage complete",
            "verdict_determined": "verdict determined",
            "complete": "complete",
            "information_saturated": "information saturated",
            "hard_budget_exhausted": "hard budget exhausted",
            "continue": "continue investigation",
        }
        badge = 'ok' if complete else 'warn'
        iteration = audit.get("iteration", audit.get("action_count", 0))
        gain = audit.get("information_gain", audit.get("substantive_gain", False))
        streak = audit.get("low_information_gain_streak", audit.get("low_gain_intervals", 0))
        first_column = "Fact" if fact_resolutions else "Question"
        third_column = "Findings" if fact_resolutions else "Attempts"
        fifth_column = "Reason" if fact_resolutions else "Gap"
        blocks.append(f'<div class="audit"><span class="badge {badge}">Iteration {_e(iteration)}: {_e(labels.get(stop_reason, stop_reason))}</span><span class="badge">information gain: {_e(gain)}</span><span class="badge">low-gain streak: {_e(streak)}</span><p>{_e(audit.get("reason", ""))}</p><table><thead><tr><th>{first_column}</th><th>Status</th><th>{third_column}</th><th>Evidence</th><th>{fifth_column}</th></tr></thead><tbody>{rows}</tbody></table></div>')
    return f'<section class="band"><h2>Coverage Audits</h2>{"".join(blocks) or "<p>No coverage audits.</p>"}</section>'


def _render_step(step: Dict[str, Any]) -> str:
    metadata = step.get("metadata", {}) if isinstance(step.get("metadata"), dict) else {}
    action = str(step.get("action_type", ""))
    badge_class = "bad" if action in {"format_error", "output_rejected"} else ("ok" if action in {"tool_call", "output"} else "")
    labels = [
        f'<span class="badge">{_e(step.get("stage", "unknown"))}</span>',
        f'<span class="badge">Round {_e(step.get("round", "?"))}</span>',
        f'<span class="badge {badge_class}">{_e(action)}</span>',
    ]
    if metadata.get("verification_iteration") is not None:
        labels.append(f'<span class="badge">Verification iteration {_e(metadata["verification_iteration"])}</span>')
    if metadata.get("native_interactions"):
        labels.append('<span class="badge">Gemini Interactions</span>')
    if metadata.get("forced_output"):
        labels.append('<span class="badge warn">Forced output</span>')
    if metadata.get("rejection_reason"):
        labels.append(f'<span class="badge bad">{_e(metadata["rejection_reason"])}</span>')
    tool_name = step.get("tool_name", "")
    body = f'<h3>{_e(tool_name or "Structured output")}</h3>'
    body += _render_interaction_chain(metadata)
    body += _labeled_pre("Thought", step.get("thought"))
    body += _labeled_json("Arguments", step.get("tool_args"))
    body += _labeled_pre("Result", step.get("tool_result"))
    body += _labeled_json("Investigation state update", metadata.get("investigation_state_update"))
    body += _labeled_json("Output", step.get("output"))
    return '<article class="step"><div>' + ''.join(labels) + '</div>' + body + '</article>'


def _render_visual_fact_investigation(investigation: Dict[str, Any]) -> str:
    if not investigation:
        return '<section class="band"><h2>Visual Facts</h2><p>No investigation state.</p></section>'
    facts = investigation.get("facts", []) if isinstance(investigation.get("facts"), list) else []
    tasks = investigation.get("tasks", []) if isinstance(investigation.get("tasks"), list) else []
    findings = investigation.get("findings", []) if isinstance(investigation.get("findings"), list) else []
    evidence = investigation.get("evidence", []) if isinstance(investigation.get("evidence"), list) else []
    discoveries = investigation.get("discoveries", []) if isinstance(investigation.get("discoveries"), list) else []
    reflections = investigation.get("reflections", []) if isinstance(investigation.get("reflections"), list) else []
    decisive = set(investigation.get("decisive_fact_ids", []) or [])
    fact_rows = ''.join(
        f'<tr><td><code>{_e(item.get("fact_id", ""))}</code></td>'
        f'<td>{_e(item.get("kind", ""))}</td><td>{_status(item.get("status", ""))}</td>'
        f'<td>{_e("decisive" if item.get("fact_id") in decisive else item.get("decision_relevance", ""))}</td>'
        f'<td>{_e(item.get("statement", ""))}</td></tr>'
        for item in facts if isinstance(item, dict)
    )
    task_rows = ''.join(
        f'<tr><td><code>{_e(item.get("task_id", ""))}</code></td>'
        f'<td>P{_e(item.get("priority", ""))}</td><td>{_status(item.get("status", ""))}</td>'
        f'<td>{_e(item.get("attempt_count", 0))}</td><td>{_e(item.get("question", ""))}</td></tr>'
        for item in tasks if isinstance(item, dict)
    )
    finding_rows = ''.join(
        f'<tr><td><code>{_e(item.get("finding_id", ""))}</code></td>'
        f'<td><code>{_e(item.get("task_id", ""))}</code></td><td>{_e(item.get("stance", ""))}</td>'
        f'<td>{_e(", ".join(item.get("evidence_ids", []) or []))}</td>'
        f'<td>{_e(item.get("statement", ""))}</td></tr>'
        for item in findings if isinstance(item, dict)
    )
    return (
        '<section class="band"><h2>Visual Facts</h2>'
        '<table><thead><tr><th>ID</th><th>Kind</th><th>Status</th><th>Role</th><th>Statement</th></tr></thead><tbody>'
        + fact_rows + '</tbody></table></section>'
        '<section class="band"><h2>Research Tasks</h2>'
        '<table><thead><tr><th>ID</th><th>Priority</th><th>Status</th><th>Attempts</th><th>Question</th></tr></thead><tbody>'
        + task_rows + '</tbody></table></section>'
        '<section class="band"><h2>Findings &amp; Evidence</h2>'
        '<table><thead><tr><th>Finding</th><th>Task</th><th>Stance</th><th>Evidence IDs</th><th>Statement</th></tr></thead><tbody>'
        + finding_rows + '</tbody></table>'
        + _labeled_json("Evidence", evidence)
        + _labeled_json("Discoveries (not Evidence)", discoveries)
        + '</section>'
        '<section class="band"><h2>Reflection Checkpoints</h2>'
        + (_labeled_json("Reflections", reflections) or '<p>No Reflection checkpoints.</p>')
        + '</section>'
        '<section class="band"><h2>Verdict Basis</h2>'
        + (_pre(json.dumps(investigation.get("verdict_basis"), ensure_ascii=False, indent=2))
           if investigation.get("verdict_basis") else '<p>No verdict basis.</p>')
        + '</section>'
    )


def _render_interaction_chain(metadata: Dict[str, Any]) -> str:
    if not metadata.get("native_interactions"):
        return ""
    parent_recorded = "previous_interaction_id" in metadata
    parent = metadata.get("previous_interaction_id")
    interaction_id = metadata.get("interaction_id")
    if not parent_recorded:
        parent_text = "not recorded"
    else:
        parent_text = "null (root)" if parent in (None, "") else str(parent)
    child_text = str(interaction_id) if interaction_id not in (None, "") else "missing"
    return (
        '<div class="interaction-chain" aria-label="Interaction request chain">'
        '<div class="interaction-node parent"><span class="interaction-key">'
        'previous_interaction_id</span><code>'
        + _e(parent_text)
        + '</code></div><span class="chain-arrow" aria-hidden="true">&#8594;</span>'
        '<div class="interaction-node current"><span class="interaction-key">'
        'interaction_id</span><code>'
        + _e(child_text)
        + "</code></div></div>"
    )


def _render_timings(timings: Dict[str, Any]) -> str:
    if not timings:
        return ""
    return '<section class="band"><h2>Stage Timings</h2>' + _list("", [f"{key}: {value}s" for key, value in timings.items()]) + '</section>'


def _status(value: Any) -> str:
    text = str(value)
    css = "ok" if text in {"resolved", "supported"} else (
        "bad" if text in {"exhausted", "refuted", "conflicted", "failed"} else "warn"
    )
    return f'<span class="badge {css}">{_e(text)}</span>'


def _list(title: str, items: Any) -> str:
    if not isinstance(items, list) or not items:
        return ""
    heading = f'<h3>{_e(title)}</h3>' if title else ""
    return heading + '<ul>' + ''.join(f'<li>{_e(item)}</li>' for item in items) + '</ul>'


def _labeled_json(label: str, value: Any) -> str:
    if value in (None, {}, []):
        return ""
    return f'<b>{_e(label)}</b>' + _pre(json.dumps(value, ensure_ascii=False, indent=2, default=str))


def _labeled_pre(label: str, value: Any) -> str:
    if value in (None, ""):
        return ""
    return f'<b>{_e(label)}</b>' + _pre(value)


def _pre(value: Any) -> str:
    if value in (None, ""):
        return ""
    return f'<pre>{_e(value)}</pre>'


def _e(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _number(value: Any, digits: int) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "0"


def _embed_local_image(image_path: str) -> str:
    if not image_path or not os.path.isfile(image_path):
        return ""
    with open(image_path, "rb") as handle:
        encoded = base64.b64encode(handle.read()).decode("ascii")
    mime = {".png": "image/png", ".webp": "image/webp", ".gif": "image/gif"}.get(os.path.splitext(image_path)[1].lower(), "image/jpeg")
    return f"data:{mime};base64,{encoded}"
