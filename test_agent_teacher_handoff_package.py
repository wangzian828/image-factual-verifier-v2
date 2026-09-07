from __future__ import annotations

import json
from pathlib import Path

from scripts.build_agent_teacher_handoff_package import build_package


def test_build_agent_teacher_handoff_package_is_code_only(tmp_path: Path) -> None:
    output = tmp_path / "handoff"
    manifest = build_package(Path(__file__).resolve().parent, output)

    assert manifest["schema_version"] == "ifv-agent-teacher-handoff-v1"
    assert manifest["contains_credentials"] is False
    assert manifest["contains_dataset"] is False
    assert manifest["training_data"]["manifest_rows"] == 8490
    assert (output / "CODEX_HANDOFF.md").is_file()
    assert (output / "runtime.env.example").is_file()
    assert "--limit 10" in (output / "run_smoke.sh").read_text(encoding="utf-8")
    assert "--full" in (output / "run_full.sh").read_text(encoding="utf-8")
    bootstrap = (output / "bootstrap.sh").read_text(encoding="utf-8")
    assert "git clone" in bootstrap
    assert "Refusing to update a dirty checkout" in bootstrap
    assert "status --porcelain" in bootstrap
    rendered = "\n".join(
        path.read_text(encoding="utf-8", errors="ignore")
        for path in output.iterdir()
        if path.is_file()
    )
    assert "本流程不需要 Gemini API" in rendered
    assert "OCR_BACKEND=easyocr" in rendered
    assert "OCR 自动 fallback" in rendered
    assert "SERPER_API_KEY" in rendered
    assert "JINA_API_KEY" in rendered
    assert "OSS_ACCESS_KEY_ID" in rendered
    assert "ms-4647" not in rendered
    assert "000000" not in rendered
    parsed = json.loads((output / "MANIFEST.json").read_text(encoding="utf-8"))
    assert parsed["source"]["repository"].endswith(
        "wangzian828/image-factual-verifier-v2.git"
    )
