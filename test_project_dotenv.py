from __future__ import annotations

from pathlib import Path

from src import project_dotenv_candidates


def test_linked_worktree_discovers_main_repository_dotenv(tmp_path: Path) -> None:
    main = tmp_path / "main"
    worktree = tmp_path / "worktrees" / "v3"
    git_dir = main / ".git" / "worktrees" / "v3"
    git_dir.mkdir(parents=True)
    worktree.mkdir(parents=True)
    (worktree / ".git").write_text(
        f"gitdir: {git_dir.as_posix()}\n",
        encoding="utf-8",
    )
    (git_dir / "commondir").write_text("../..\n", encoding="utf-8")

    candidates = list(project_dotenv_candidates(worktree))

    assert candidates == [
        worktree / ".env",
        main / ".env",
    ]
