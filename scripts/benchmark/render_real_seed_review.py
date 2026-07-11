#!/usr/bin/env python3
"""Render a self-contained browser review queue for real-seed candidates."""

from __future__ import annotations

import argparse
import html
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping
from urllib.parse import quote

from src.storage import data_path


DECISIONS = (
    "core",
    "external_claim_transfer",
    "excluded",
    "needs_review",
)


def _default_candidates() -> Path:
    return data_path(
        "benchmarks/candidates/real_seed_v0/averimatec/candidates.jsonl",
        "data/benchmark_candidates/real_seed_v0/averimatec/candidates.jsonl",
    )


def _default_prefilter() -> Path:
    return data_path(
        "benchmarks/candidates/real_seed_v0/prefilter/prefilter.jsonl",
        "data/benchmark_candidates/real_seed_v0/prefilter/prefilter.jsonl",
    )


def _default_output_dir() -> Path:
    return data_path(
        "benchmarks/candidates/real_seed_v0/review",
        "data/benchmark_candidates/real_seed_v0/review",
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", default=str(_default_candidates()))
    parser.add_argument("--prefilter", default=str(_default_prefilter()))
    parser.add_argument("--output-dir", default=str(_default_output_dir()))
    return parser.parse_args()


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise TypeError(f"Expected JSON object at {path}:{line_number}")
        rows.append(value)
    return rows


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def _e(value: Any) -> str:
    return html.escape(str(value if value is not None else ""), quote=True)


def _render_links(urls: Iterable[str]) -> str:
    items = []
    for url in urls:
        rendered = str(url or "").strip()
        if not rendered:
            continue
        items.append(
            f'<li><a href="{_e(rendered)}" target="_blank" rel="noreferrer">'
            f"{_e(rendered)}</a></li>"
        )
    return "<ul>" + "".join(items) + "</ul>" if items else "<p>None</p>"


def _render_images(assets: Iterable[Mapping[str, Any]]) -> str:
    blocks = []
    for asset in assets:
        image_path = str(asset.get("image_path", ""))
        if not image_path:
            continue
        dimensions = f'{asset.get("width", "?")} x {asset.get("height", "?")}'
        blocks.append(
            '<figure class="asset">'
            f'<img src="{_e(asset.get("review_src") or image_path)}" loading="lazy" '
            f'alt="{_e(asset.get("filename") or image_path)}">'
            f'<figcaption>{_e(asset.get("filename") or image_path)} · '
            f'{_e(dimensions)} · <code>{_e(str(asset.get("sha256", ""))[:12])}</code>'
            "</figcaption></figure>"
        )
    return '<div class="assets">' + "".join(blocks) + "</div>"


def _render_questions(questions: Iterable[Mapping[str, Any]]) -> str:
    rows = []
    for question in questions:
        rows.append(
            "<tr>"
            f'<td>{_e(question.get("question"))}</td>'
            f'<td>{_e(", ".join(question.get("question_type") or []))}</td>'
            f'<td>{_e(question.get("answer_method"))}</td>'
            "</tr>"
        )
    if not rows:
        return "<p>None</p>"
    return (
        '<table><thead><tr><th>Question</th><th>Type</th><th>Method</th></tr></thead>'
        f'<tbody>{"".join(rows)}</tbody></table>'
    )


def _render_ocr(rows: Iterable[Mapping[str, Any]]) -> str:
    blocks = []
    for row in rows:
        status = row.get("status", "")
        text = row.get("full_text", "")
        blocks.append(
            f'<p><strong>{_e(status)}</strong> · regions '
            f'{_e(row.get("region_count", 0))}<br>{_e(text)}</p>'
        )
    return "".join(blocks) or "<p>Not extracted</p>"


def _review_row(
    candidate: Mapping[str, Any], prefilter: Mapping[str, Any]
) -> Dict[str, Any]:
    return {
        "sample_id": candidate.get("sample_id"),
        "ground_truth": candidate.get("ground_truth"),
        "original_label": candidate.get("original_label"),
        "gold_claim_text": candidate.get("gold_claim_text"),
        "primary_image_path": candidate.get("primary_image_path"),
        "source_article_url": candidate.get("source_article_url"),
        "evidence_urls": candidate.get("evidence_urls") or [],
        "image_question_count": candidate.get("image_question_count", 0),
        "suggested_route": prefilter.get("suggested_route", "needs_review"),
        "review_priority": int(prefilter.get("review_priority", 0) or 0),
        "reasons": prefilter.get("reasons") or [],
        "duplicate_group": prefilter.get("duplicate_group"),
        "duplicate_group_size": prefilter.get("duplicate_group_size", 1),
        "claim_assets": candidate.get("claim_assets") or [],
        "questions": candidate.get("questions") or [],
        "ocr": prefilter.get("ocr") or [],
        "url_checks": prefilter.get("url_checks") or [],
        "license": candidate.get("license") or {},
    }


def _render_card(row: Mapping[str, Any]) -> str:
    sample_id = str(row.get("sample_id", ""))
    reasons = "".join(f"<li>{_e(reason)}</li>" for reason in row.get("reasons") or [])
    decisions = "".join(
        '<label class="decision-option">'
        f'<input type="radio" name="decision-{_e(sample_id)}" value="{decision}"> '
        f"{_e(decision)}</label>"
        for decision in DECISIONS
    )
    duplicate = row.get("duplicate_group") or "none"
    license_payload = json.dumps(row.get("license") or {}, ensure_ascii=False)
    return (
        f'<article class="case" data-id="{_e(sample_id)}" '
        f'data-label="{_e(row.get("ground_truth"))}" '
        f'data-route="{_e(row.get("suggested_route"))}" '
        f'data-decision="">'
        '<header class="case-header">'
        f'<h2>{_e(sample_id)}</h2><span class="badge">{_e(row.get("ground_truth"))}</span>'
        f'<span class="badge route">{_e(row.get("suggested_route"))}</span>'
        f'<span>priority {_e(row.get("review_priority"))}</span>'
        "</header>"
        f'{_render_images(row.get("claim_assets") or [])}'
        '<section><h3>Gold Claim (Review Only)</h3>'
        f'<p class="claim">{_e(row.get("gold_claim_text"))}</p></section>'
        '<section class="grid">'
        f'<div><h3>Rule Reasons</h3><ul>{reasons}</ul>'
        f'<p>Duplicate group: <code>{_e(duplicate)}</code> '
        f'({_e(row.get("duplicate_group_size", 1))})</p></div>'
        f'<div><h3>OCR</h3>{_render_ocr(row.get("ocr") or [])}</div>'
        "</section>"
        f'<section><h3>Investigation Questions</h3>{_render_questions(row.get("questions") or [])}</section>'
        f'<section><h3>Evidence URLs</h3>{_render_links(row.get("evidence_urls") or [])}</section>'
        f'<section><h3>License State</h3><code>{_e(license_payload)}</code></section>'
        '<section class="decision"><h3>Human Decision</h3>'
        f'<div class="decision-options">{decisions}</div>'
        f'<textarea data-notes-for="{_e(sample_id)}" placeholder="Review notes"></textarea>'
        "</section></article>"
    )


def _document(rows: List[Mapping[str, Any]]) -> str:
    cards = "".join(_render_card(row) for row in rows)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Real Seed Review</title>
<style>
*{{box-sizing:border-box}} body{{margin:0;font-family:Arial,sans-serif;color:#1e2329;background:#f4f6f8}}
.toolbar{{position:sticky;top:0;z-index:10;display:flex;gap:12px;align-items:center;flex-wrap:wrap;padding:12px 18px;background:#fff;border-bottom:1px solid #ccd3da}}
.toolbar select,.toolbar button{{height:34px}} main{{max-width:1400px;margin:auto;padding:16px}}
.case{{background:#fff;border:1px solid #ccd3da;border-radius:6px;margin:0 0 16px;padding:16px}}
.case[hidden]{{display:none}} .case-header{{display:flex;gap:10px;align-items:center;flex-wrap:wrap}}
.case-header h2{{font-size:18px;margin:0 auto 0 0}} .badge{{padding:3px 7px;border:1px solid #8b98a5;border-radius:4px}}
.route{{background:#eef4fa}} .assets{{display:flex;gap:12px;overflow:auto;margin:14px 0}}
.asset{{margin:0;min-width:260px;max-width:520px}} .asset img{{display:block;max-width:100%;max-height:420px;object-fit:contain;background:#e9edf1}}
.asset figcaption{{font-size:12px;overflow-wrap:anywhere}} h3{{font-size:14px;margin:14px 0 6px}}
.claim{{font-size:17px;font-weight:600}} .grid{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}
table{{border-collapse:collapse;width:100%}} th,td{{text-align:left;vertical-align:top;border:1px solid #d7dde3;padding:6px}}
a,code{{overflow-wrap:anywhere}} .decision-options{{display:flex;gap:12px;flex-wrap:wrap}}
.decision-option{{padding:6px;border:1px solid #ccd3da}} textarea{{width:100%;min-height:70px;margin-top:8px}}
@media(max-width:760px){{.grid{{grid-template-columns:1fr}}}}
</style></head><body>
<div class="toolbar">
<strong>Real Seed Review</strong><span id="count"></span>
<label>Gold label <select id="label-filter"><option value="">all</option><option>real</option><option>fake</option><option>unverifiable</option></select></label>
<label>Suggested route <select id="route-filter"><option value="">all</option>{''.join(f'<option>{d}</option>' for d in DECISIONS)}</select></label>
<label>Decision <select id="decision-filter"><option value="">all</option><option value="unreviewed">unreviewed</option>{''.join(f'<option>{d}</option>' for d in DECISIONS)}</select></label>
<button id="export" type="button">Export decisions.json</button>
</div><main>{cards}</main>
<script>
const STORAGE_KEY='ifv-real-seed-review-v1';
const cases=[...document.querySelectorAll('.case')];
let decisions={{}}; try{{decisions=JSON.parse(localStorage.getItem(STORAGE_KEY)||'{{}}')}}catch(_e){{decisions={{}}}}
function save(){{localStorage.setItem(STORAGE_KEY,JSON.stringify(decisions)); applyFilters();}}
function hydrate(){{cases.forEach(card=>{{const id=card.dataset.id;const item=decisions[id]||{{}};card.dataset.decision=item.decision||'';const radio=card.querySelector(`input[value="${{item.decision||''}}"]`);if(radio)radio.checked=true;const notes=card.querySelector('textarea');notes.value=item.notes||'';card.querySelectorAll('input[type=radio]').forEach(input=>input.addEventListener('change',()=>{{decisions[id]={{decision:input.value,notes:notes.value}};card.dataset.decision=input.value;save();}}));notes.addEventListener('change',()=>{{decisions[id]={{decision:(decisions[id]||{{}}).decision||'',notes:notes.value}};save();}});}});}}
function applyFilters(){{const label=document.getElementById('label-filter').value;const route=document.getElementById('route-filter').value;const decision=document.getElementById('decision-filter').value;let shown=0;cases.forEach(card=>{{const actual=card.dataset.decision||'';const visible=(!label||card.dataset.label===label)&&(!route||card.dataset.route===route)&&(!decision||(decision==='unreviewed'?!actual:actual===decision));card.hidden=!visible;if(visible)shown++;}});document.getElementById('count').textContent=`${{shown}} / ${{cases.length}} visible`;}}
['label-filter','route-filter','decision-filter'].forEach(id=>document.getElementById(id).addEventListener('change',applyFilters));
document.getElementById('export').addEventListener('click',()=>{{const payload=cases.map(card=>({{sample_id:card.dataset.id,decision:(decisions[card.dataset.id]||{{}}).decision||'needs_review',notes:(decisions[card.dataset.id]||{{}}).notes||''}}));const blob=new Blob([JSON.stringify(payload,null,2)],{{type:'application/json'}});const link=document.createElement('a');link.href=URL.createObjectURL(blob);link.download='decisions.json';link.click();URL.revokeObjectURL(link.href);}});
hydrate();applyFilters();
</script></body></html>"""


def render_review(
    *, candidates_path: Path, prefilter_path: Path, output_dir: Path
) -> Dict[str, Any]:
    candidates = _load_jsonl(candidates_path)
    prefilter = {
        str(row.get("sample_id")): row for row in _load_jsonl(prefilter_path)
    }
    rows = [
        _review_row(candidate, prefilter.get(str(candidate.get("sample_id")), {}))
        for candidate in candidates
    ]
    for row in rows:
        normalized_assets = []
        for asset in row.get("claim_assets") or []:
            normalized = dict(asset)
            image_path = str(normalized.get("image_path", ""))
            if image_path:
                relative = os.path.relpath(Path(image_path).resolve(), output_dir.resolve())
                normalized["review_src"] = quote(Path(relative).as_posix(), safe="/:~")
            normalized_assets.append(normalized)
        row["claim_assets"] = normalized_assets
    rows.sort(key=lambda row: (-int(row["review_priority"]), str(row["sample_id"])))
    output_dir.mkdir(parents=True, exist_ok=True)
    queue_path = output_dir / "review_queue.jsonl"
    html_path = output_dir / "review.html"
    _write_jsonl(queue_path, rows)
    html_path.write_text(_document(rows), encoding="utf-8")
    return {
        "num_cases": len(rows),
        "review_queue": str(queue_path.resolve()),
        "review_html": str(html_path.resolve()),
    }


def main() -> None:
    args = _parse_args()
    result = render_review(
        candidates_path=Path(args.candidates),
        prefilter_path=Path(args.prefilter),
        output_dir=Path(args.output_dir),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
