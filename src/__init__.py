from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv


def _load_project_dotenv() -> None:
    root_env = Path(__file__).resolve().parent.parent / ".env"
    load_dotenv(dotenv_path=root_env, override=False)


_load_project_dotenv()
