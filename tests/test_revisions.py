"""The revisions a run records, and what a continuation makes of them."""

from __future__ import annotations

from pathlib import Path

import pytest

from ygo_training.environment import POLICIES_DIRECTORY
from ygo_training.revisions import duel_revisions, trainer_revision
from ygo_training.run_repository import RunRepository

ROOT = Path(__file__).resolve().parents[1]


def test_a_directory_git_knows_nothing_of_has_no_revision(tmp_path):
    assert duel_revisions(tmp_path) == {"commit": None, "clean": False}
    assert trainer_revision(tmp_path)["commit"] is None


@pytest.mark.skipif(
    not (ROOT / POLICIES_DIRECTORY / ".git").exists(), reason="not a checkout with its submodule"
)
def test_the_duels_revisions_are_the_pilots_and_what_they_pin():
    revisions = duel_revisions(ROOT)

    assert len(revisions["commit"]) == 40 and isinstance(revisions["clean"], bool)
    for pinned in ("vendor/ocgcore", "vendor/CardScripts", "vendor/BabelCdb"):
        assert len(revisions[pinned]) == 40 and revisions[pinned] != revisions["commit"]
    assert len(trainer_revision(ROOT)["commit"]) == 40


def test_a_run_is_not_continued_on_other_pilots_or_another_engine(tmp_path):
    run = RunRepository(tmp_path)
    first = {
        "duels": 4,
        "duel_revisions": {"commit": "a" * 40, "clean": True, "vendor/ocgcore": "b" * 40},
        "trainer_revision": {"commit": "c" * 40, "clean": True},
    }
    run.initialize(first)
    # The trainer's own commit may move between invocations.
    moved = {**first, "trainer_revision": {"commit": "d" * 40, "clean": False}}
    run.initialize(moved, continue_run=True, mutable_keys={"trainer_revision"})

    other_engine = {
        **first,
        "duel_revisions": {**first["duel_revisions"], "vendor/ocgcore": "e" * 40},
    }
    with pytest.raises(ValueError, match="duel_revisions.vendor/ocgcore"):
        run.initialize(other_engine, continue_run=True, mutable_keys={"trainer_revision"})
    modified = {**first, "duel_revisions": {**first["duel_revisions"], "clean": False}}
    with pytest.raises(ValueError, match="duel_revisions.clean"):
        run.initialize(modified, continue_run=True, mutable_keys={"trainer_revision"})
