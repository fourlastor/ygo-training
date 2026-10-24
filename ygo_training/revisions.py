"""The revisions a run is made with.

A duel is what the pilots, the engine, the card scripts and the card database
of the ``native/ygo-policies`` checkout make of a seed: at other revisions
the same seed can be another duel.  A run records them, and is not continued
across a change of them.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from .environment import POLICIES_DIRECTORY


def _git(directory: Path, *arguments: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(directory), *arguments],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout if result.returncode == 0 else None


def _checkout(directory: Path) -> dict[str, Any]:
    commit = _git(directory, "rev-parse", "HEAD")
    changes = _git(directory, "status", "--porcelain", "--untracked-files=no")
    return {
        "commit": commit.strip() if commit else None,
        # Nothing tracked is modified, and every submodule is at its pin.
        "clean": changes is not None and not changes.strip(),
    }


def trainer_revision(root: str | Path) -> dict[str, Any]:
    """The commit of the trainer itself, for the record."""
    return _checkout(Path(root).resolve())


def duel_revisions(root: str | Path) -> dict[str, Any]:
    """What the duels of this checkout run on.

    The pilots' commit and, as checked out below it, the engine, the card
    scripts and the card database.  Everything is ``None`` where git cannot
    say (an export without history).
    """
    policies = Path(root).resolve() / POLICIES_DIRECTORY
    if not (policies / ".git").exists():
        # Not a checkout of its own: git would answer for the trainer's.
        return {"commit": None, "clean": False}
    revisions = _checkout(policies)
    listed = _git(policies, "submodule", "status", "--recursive") or ""
    for line in listed.splitlines():
        # "<state><commit> <path> (<description>)"
        fields = line[1:].split()
        if len(fields) >= 2:
            revisions[fields[1]] = fields[0]
    return revisions
