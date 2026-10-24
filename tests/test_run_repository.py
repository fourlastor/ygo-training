import json

import pytest

from ygo_training.run_repository import RunRepository, checkpoint_version


def test_repository_finds_latest_numbered_checkpoint(tmp_path):
    repository = RunRepository(tmp_path / "run")
    repository.checkpoint_dir.mkdir(parents=True)
    repository.checkpoint(2).touch()
    repository.checkpoint(11).touch()
    (repository.checkpoint_dir / "notes.pt").touch()

    assert repository.latest_checkpoint() == repository.checkpoint(11)


def test_repository_rejects_existing_metric_versions(tmp_path):
    repository = RunRepository(tmp_path / "run")
    repository.metrics_path.parent.mkdir(parents=True)
    repository.metrics_path.write_text('{"version": 3}\n{"version": 7}\n', encoding="utf8")

    with pytest.raises(FileExistsError, match=r"version\(s\) 7\.\.7"):
        repository.ensure_versions_available(range(5, 9))


def test_schedule_origin_is_where_the_run_began(tmp_path):
    scratch = RunRepository(tmp_path / "scratch")
    assert scratch.schedule_origin() == 0

    # A branch resumed from version 9 records the checkpoint's step.
    branch = RunRepository(tmp_path / "branch")
    branch.initialize({"schedule_origin": 10})
    assert branch.schedule_origin() == 10

    # A run recorded before the field began at its first version, and a
    # continuation that adds the field keeps it.
    older = RunRepository(tmp_path / "older")
    older.initialize({"learner": "blackwing"})
    older.metrics_path.write_text('{"version": 10}\n{"version": 11}\n', encoding="utf8")
    assert older.schedule_origin() == 10
    older.initialize(
        {"learner": "blackwing", "schedule_origin": older.schedule_origin()}, continue_run=True
    )
    assert json.loads(older.config_path.read_text())["schedule_origin"] == 10


def test_repository_persists_config_without_implicit_best_checkpoint_policy(tmp_path):
    repository = RunRepository(tmp_path / "run")
    repository.initialize({"learner": "Normal Combat Training"})
    assert json.loads(repository.config_path.read_text()) == {"learner": "Normal Combat Training"}

    assert not hasattr(repository, "best_path")
    assert not hasattr(repository, "update_best")


def test_a_run_that_has_only_imitated_may_imitate_for_longer(tmp_path):
    repository = RunRepository(tmp_path / "run")
    assert not repository.may_imitate_longer(0, 8)
    repository.initialize({"seed": 2011, "imitation_only_updates": 64})

    # After its 64 imitation-only updates, or before the last of them.
    assert repository.may_imitate_longer(64, 256)
    assert repository.may_imitate_longer(40, 64)
    # Not shorter, and not once an update has come after the phase.
    assert not repository.may_imitate_longer(64, 32)
    assert not repository.may_imitate_longer(65, 256)

    older = RunRepository(tmp_path / "older")
    older.initialize({"seed": 2011})
    assert not older.may_imitate_longer(0, 8)


def test_repository_continuation_rejects_protocol_drift(tmp_path):
    repository = RunRepository(tmp_path / "run")
    repository.initialize({"seed": 2011, "board_representation": "mean-pool"})

    with pytest.raises(ValueError, match="board_representation"):
        repository.initialize(
            {"seed": 2011, "board_representation": "individual-tokens"},
            continue_run=True,
        )


def test_repository_continuation_augments_old_config_and_allows_explicit_mutation(
    tmp_path,
):
    repository = RunRepository(tmp_path / "run")
    repository.initialize({"seed": 2011, "learning_rate": 3e-4})

    repository.initialize(
        {
            "seed": 2011,
            "learning_rate": 1e-4,
            "board_representation": "mean-pool",
        },
        continue_run=True,
        mutable_keys={"learning_rate"},
    )

    assert json.loads(repository.config_path.read_text()) == {
        "seed": 2011,
        "learning_rate": 1e-4,
        "board_representation": "mean-pool",
    }


def test_repository_migrates_legacy_per_worker_duel_count(tmp_path):
    repository = RunRepository(tmp_path / "run")
    repository.initialize({"workers": 16, "duels_per_worker": 32})

    repository.initialize(
        {"workers": 16, "duels": 512},
        continue_run=True,
    )
    assert json.loads(repository.config_path.read_text()) == {
        "workers": 16,
        "duels": 512,
    }

    with pytest.raises(ValueError, match="duels"):
        repository.initialize(
            {"workers": 16, "duels": 640},
            continue_run=True,
        )


def test_repository_continuation_accepts_new_nested_metadata_but_not_nested_drift(
    tmp_path,
):
    repository = RunRepository(tmp_path / "run")
    repository.initialize({"evaluation": {"games": 1024, "seed": 2011}})

    repository.initialize(
        {
            "evaluation": {
                "games": 1024,
                "seed": 2011,
                "every_updates": 5,
            }
        },
        continue_run=True,
    )
    with pytest.raises(ValueError, match="evaluation.games"):
        repository.initialize(
            {
                "evaluation": {
                    "games": 512,
                    "seed": 2011,
                    "every_updates": 5,
                }
            },
            continue_run=True,
        )


def _kept(tmp_path, versions, **thinning):
    repository = RunRepository(tmp_path / "run")
    repository.checkpoint_dir.mkdir(parents=True)
    for version in versions:
        repository.checkpoint(version).touch()
        repository.thin(version, **thinning)
    return sorted(checkpoint_version(path) for path in repository.checkpoint_dir.glob("*.pt"))


def test_a_long_run_keeps_one_checkpoint_in_so_many_and_its_latest(tmp_path):
    # Version 3 was there before this invocation began at 4.
    assert _kept(tmp_path / "a", range(3, 12), every=4, first=4) == [3, 7, 11]
    assert _kept(tmp_path / "b", range(3, 10), every=4, first=4) == [3, 7, 9]
    assert _kept(tmp_path / "c", range(5), every=1, first=0) == [0, 1, 2, 3, 4]


def test_the_checkpoint_a_duel_the_engine_stopped_was_played_with_is_kept(tmp_path):
    repository = RunRepository(tmp_path / "run")
    repository.checkpoint_dir.mkdir(parents=True)
    for version in range(4, 12):
        repository.checkpoint(version).touch()
        # Update 6, played with checkpoint 5, had a duel the engine stopped.
        repository.thin(version, every=4, first=4, keep=version == 6)
    kept = sorted(checkpoint_version(path) for path in repository.checkpoint_dir.glob("*.pt"))
    assert kept == [5, 7, 11]


def test_checkpoint_version_requires_numbered_name(tmp_path):
    assert checkpoint_version(tmp_path / "model-00123.pt") == 123
    with pytest.raises(ValueError, match="numbered"):
        checkpoint_version(tmp_path / "champion.pt")
