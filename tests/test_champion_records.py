"""A copy's games played by the teachers, then played again from the
answers they gave (`pool.taught_games` with ``played``): the same games."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from ygo_training.environment import library_path
from ygo_training.model import PolicyValueNet
from ygo_training.pool import DuelError, DuelPool, taught_games
from ygo_training.schedules import rollout_games

ROOT = Path(__file__).resolve().parents[1]


def _built() -> bool:
    try:
        library_path(ROOT)
    except Exception:  # noqa: BLE001 - not built
        return False
    return True


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not _built(), reason="the environment is not built: cargo build --release"),
]


def drawn(seed: int) -> PolicyValueNet:
    torch.manual_seed(seed)
    return PolicyValueNet(board_representation="mean-pool").eval()


def copy_games() -> list[dict]:
    """Two decks' games as a copy deals them: mirrors, both seats the
    teacher's, and games against pilots."""
    learners = ["blackwing", "burn", "blackwing", "burn", "blackwing", "burn"]
    opponents = ["blackwing", "burn", "heroes", "monarch", "lightsworn", "gishki"]
    others = ["self", "self", "pilot", "pilot", "pilot", "pilot"]
    return rollout_games(11 * 1024, 3, opponents, learners, others)


def test_a_copys_games_played_again_from_their_answers_are_the_games_played():
    torch.set_num_threads(2)
    teachers = {"blackwing": drawn(1), "burn": drawn(2)}
    games = copy_games()
    with DuelPool(ROOT, threads=1, slots=1) as pool:
        live, _ = taught_games(
            pool, teachers, "cpu", learner="blackwing", games=games, limit=4096, grouped=False
        )
        answers = [[step["chosen"] for step in trajectory["steps"]] for trajectory in live]
        again, asked = taught_games(
            pool,
            {},
            "cpu",
            learner="blackwing",
            games=games,
            limit=4096,
            grouped=False,
            played=answers,
        )
        assert asked["inference_seconds"] < 5
        for first, second in zip(live, again, strict=True):
            assert first["digest"] == second["digest"]
            kept = {key: value for key, value in first.items() if key != "steps"}
            assert kept == {key: value for key, value in second.items() if key != "steps"}
            assert len(first["steps"]) == len(second["steps"]) > 0
            for one, other in zip(first["steps"], second["steps"], strict=True):
                assert {k: v for k, v in one.items() if k != "row"} == {
                    k: v for k, v in other.items() if k != "row"
                }
            ones = pool.batch([step["row"] for step in first["steps"]])
            others = pool.batch([step["row"] for step in second["steps"]])
            for name, array in ones.items():
                assert np.array_equal(array, others[name]), name
        # Both seats of a mirror are the teacher's, and imitated.
        assert {step["player"] for step in live[0]["steps"]} == {0, 1}
        short = [given[:-1] for given in answers]
        with pytest.raises(DuelError, match="end before its duel does"):
            taught_games(
                pool,
                {},
                "cpu",
                learner="blackwing",
                games=games,
                limit=4096,
                grouped=False,
                played=short,
            )


def test_no_two_chunks_share_a_deal_and_none_is_a_suites():
    from ygo_training import champion_records as records
    from ygo_training.schedules import TrainingSchedule

    decks = ["blackwing", "burn", "cyber", "heroes"]
    indices = {
        records.rollout_index(decks, deck, kind, chunk)
        for deck in decks
        for kind in [records.MIRROR, *(f"against-{other}" for other in decks)]
        for chunk in range(records.KIND_STRIDE)
    }
    assert len(indices) == len(decks) * (len(decks) + 1) * records.KIND_STRIDE
    schedule = TrainingSchedule(seed=records.SEED)
    assert len({schedule.rollout_seed(index) for index in indices}) == len(indices)
    assert records.held_out(2012) and records.held_out(2012 + 3 * 10_000_000 + 2047)
    assert not records.held_out(2011) and not records.held_out(2012 + 2048)
    with pytest.raises(ValueError, match="at most 64 chunks"):
        records.rollout_index(decks, "burn", records.MIRROR, records.KIND_STRIDE)
    assert records.kind_of({"other": "self", "opponent": "burn"}) == "mirror"
    assert records.kind_of({"other": "pilot", "opponent": "heroes"}) == "against-heroes"


def saved_network(folder: Path, seed: int) -> Path:
    from ygo_training.checkpoints import save

    model = drawn(seed)
    path = folder / f"champion-{seed}" / "model-00959.pt"
    save(path, model, torch.optim.Adam(model.parameters()), 960, {})
    return path


def test_a_recording_is_made_and_a_copy_deals_its_games_from_it(tmp_path, monkeypatch):
    import importlib.util
    import tarfile

    from ygo_training import champion_records as records
    from ygo_training.config import TrainerConfig
    from ygo_training.ppo_trainer import PpoTrainer

    torch.set_num_threads(2)
    monkeypatch.setattr(records, "CHUNK", 3)
    spec = importlib.util.spec_from_file_location(
        "record_champion", ROOT / "tools/record_champion.py"
    )
    tool = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(tool)
    out = tmp_path / "records"
    pilots = ["heroes", "monarch"]
    for deck, seed in (("blackwing", 1), ("burn", 2)):
        champion = saved_network(tmp_path, seed)
        made = tool.record(
            deck, champion, out, mirrors=2, against=2, opponents=pilots, threads=1, slots=4
        )
        assert made.is_file()
        with tarfile.open(made) as archive:
            names = archive.getnames()
        assert f"{deck}/mirror/00001.jsonl.gz" in names and f"{deck}/about.json" in names
        lines = records.read_chunk(records.chunk_path(out, deck, "against-heroes", 1))
        assert len(lines) == 3 and all(line["learner"] == deck for line in lines)
    # Made again, a recording plays nothing it holds.
    started = records.chunk_path(out, "burn", "mirror", 0).stat().st_mtime_ns
    tool.record("burn", saved_network(tmp_path, 2), out, mirrors=2, against=2, opponents=pilots)
    assert records.chunk_path(out, "burn", "mirror", 0).stat().st_mtime_ns == started
    with pytest.raises(SystemExit, match="another champion"):
        tool.record("burn", saved_network(tmp_path, 3), out, mirrors=2, against=2, opponents=pilots)

    def copy(**more) -> PpoTrainer:
        return PpoTrainer(
            ROOT,
            TrainerConfig(
                workers=3,
                duels=4,
                device="cpu",
                seed=71,
                learner_deck="blackwing",
                learner_decks=["blackwing", "burn"],
                opponent_decks=pilots,
                imitation_only_updates=3,
                imitation_value_weight=0.5,
                imitation_value_positions=4,
                self_play_duels=2,
                self_play_pairs="mirrors",
                copy_records=str(out),
                pool_threads=1,
                pool_slots=4,
                **more,
            ),
        )

    with copy() as trainer:
        dealt = []
        for _ in range(2):
            trajectories, _ = trainer.collect()
            dealt.append(trajectories)
    mirrors = [t for t in dealt[0] if t.get("other") == "self"]
    assert len(mirrors) == 2 and all({s["player"] for s in t["steps"]} == {0, 1} for t in mirrors)
    recorded = {
        line["digest"]: line
        for deck in ("blackwing", "burn")
        for kind in ("mirror", "against-heroes", "against-monarch")
        for chunk in range(2)
        for line in records.read_chunk(records.chunk_path(out, deck, kind, chunk))
    }
    for trajectory in dealt[0] + dealt[1]:
        line = recorded[trajectory["digest"]]
        assert [step["chosen"] for step in trajectory["steps"]] == line["answers"]
        assert trajectory["learner_deck"] == line["learner"]
    # Gone on from its first update, a copy deals its second as the one before.
    with copy() as again:
        again.set_rollout_index(1)
        again.set_training_schedule_index(1)
        trajectories, _ = again.collect()
    assert [t["digest"] for t in trajectories] == [t["digest"] for t in dealt[1]]
