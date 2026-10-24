"""The model's small calls at sizes fixed in advance (`graphs.Graphs`): a
call filled out is the library's own batch of it in a wider one, its answers
are the model's, and an update's games through it are the games."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from ygo_training.checkpoints import value_fingerprint
from ygo_training.config import GROUPS_OVER, Calls, TrainerConfig
from ygo_training.environment import library_path
from ygo_training.graphs import WIDTHS, Graphs, axes, counts, fixed
from ygo_training.ppo_trainer import PpoTrainer

ROOT = Path(__file__).resolve().parents[1]
OPPONENTS = ["monarch", "heroes", "burn", "gladiator", "lightsworn"]


def _built() -> bool:
    try:
        library_path(ROOT)
    except FileNotFoundError:
        return False
    return True


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not _built(), reason="the environment is not built: cargo build --release"),
]


def config(**more) -> TrainerConfig:
    return TrainerConfig(
        **{
            "workers": 1,
            "duels": 8,
            "device": "cpu",
            "seed": 33,
            "opponent_decks": OPPONENTS,
            "pool_threads": 2,
            "pool_slots": 4,
            "self_play_duels": 4,
            "ppo_epochs": 1,
            "minibatch": 48,
            # Updates the same however their games' calls were made, from
            # graphs or not: so with the estimates worked out again.
            "rollout_estimates": "recomputed",
            **more,
        }
    )


def widths_of(batch: dict[str, np.ndarray]) -> dict[str, int]:
    """The widths a batch of the library's is padded to."""
    return {
        "state": batch["state_card"].shape[1],
        "history": batch["history_card"].shape[1],
        "action": batch["action_kind"].shape[1],
        "members": batch["action_member_card"].shape[2],
    }


@pytest.fixture(scope="module")
def played():
    """A trainer, the rows of the decisions of one update's games in the
    order they were played, and the batches the model was given."""
    torch.set_num_threads(2)
    with PpoTrainer(ROOT, config()) as trainer:
        given: list[dict[str, np.ndarray]] = []
        watching = trainer.model.register_forward_pre_hook(
            lambda _module, asked: given.append({name: value.numpy().copy() for name, value in asked[0].items()})
        )
        trajectories, _ = trainer.collect()
        watching.remove()
        rows = [int(step["row"]) for trajectory in trajectories for step in trajectory["steps"]]
        yield trainer, rows, given


def test_the_counts_and_the_widths_a_call_is_filled_out_to():
    assert counts(1) == [1]
    assert counts(8) == [1, 2, 4, 8]
    assert counts(160) == [1, 2, 4, 8, 16, 32, 64, 128, 160]
    assert axes("state_card") == ("state",) and axes("state_lengths") == ("state",)
    # No deck list since format 15: no array of one follows a width.
    assert set(WIDTHS) == {"state", "history", "action", "members"}
    assert axes("action_member_value") == ("action", "members") and axes("action_mask") == ("action",)
    assert axes("history_scalars") == ("history",)
    assert axes("context_card") == () and axes("scalars") == () and axes("phase") == ()
    # An axis that is not there is not filled out: one number a decision stays one.
    assert fixed("state_lengths", (5,), WIDTHS) == ()
    assert fixed("state_numeric", (5, 7, 6), WIDTHS) == (WIDTHS["state"], 6)
    assert fixed("action_member_categories", (5, 3, 1, 8), WIDTHS) == (WIDTHS["action"], WIDTHS["members"], 8)
    assert fixed("context_numeric", (5, 4, 6), WIDTHS) == (4, 6)
    with pytest.raises(ValueError, match="one decision at least"):
        Graphs(torch.nn.Identity(), "cpu", largest=0)


def test_a_call_filled_out_is_the_librarys_batch_of_it_in_a_wider_one(played):
    """The library pads a batch to its longest rows.  A few decisions filled
    out here to the widths of a batch that holds them among wider ones are
    what the library made of them there, every array of the input."""
    trainer, rows, _ = played
    few, more = rows[:3], rows[3:60]
    alone = trainer._pool().batch(few)
    among = trainer._pool().batch(few + more)
    widths = widths_of(among)
    assert any(widths[part] > mine for part, mine in widths_of(alone).items()), "nothing was wider"
    graphs = Graphs(trainer.model, "cpu", largest=4, widths=widths)
    assert graphs.fits(alone)
    sized = graphs._sized(4, alone)
    graphs._fill(sized, alone)
    assert sorted(sized.arrays) == sorted(among)
    for name, wide in among.items():
        assert sized.arrays[name].shape == (4, *wide.shape[1:]), name
        assert sized.arrays[name].dtype == wide.dtype, name
        np.testing.assert_array_equal(sized.arrays[name][:3], wide[:3], err_msg=name)
        # The decision that is none is zeros, and so on the device.
        assert not sized.arrays[name][3:].any(), name
        np.testing.assert_array_equal(sized.tensors[name].numpy(), sized.arrays[name], err_msg=name)


def test_calls_are_answered_at_the_fixed_sizes_with_the_models_numbers(played):
    trainer, _, given = played
    model = trainer.model
    graphs = Graphs(model, "cpu", largest=4)
    graphs.begin_update()
    answered = 0
    for batch in given[:40]:
        made = graphs.answer(batch)
        with torch.no_grad():
            logits, values = model({name: torch.from_numpy(array) for name, array in batch.items()})
        decisions, answers = logits.shape
        if decisions > 4:
            assert made is None
            continue
        assert made is not None
        answered += 1
        # The answers the call has, to the last digits; the others are none.
        assert made[0].shape == (decisions, WIDTHS["action"]) and made[1].shape == (decisions,)
        there = torch.isfinite(logits)
        assert torch.equal(torch.isfinite(made[0][:, :answers]), there)
        assert torch.allclose(made[0][:, :answers][there], logits[there], atol=1e-4, rtol=0)
        assert torch.isinf(made[0][:, answers:]).all()
        assert torch.allclose(made[1], values, atol=1e-4, rtol=0)
    assert answered > 5 and graphs.told == {"answered": answered, "recorded": 0, "checked": 0}
    assert sorted(graphs.sized) == sorted({count for count in (1, 2, 4) if count in graphs.sized})
    # What a wider call left in a batch does not reach the next one: a call
    # is answered as by graphs that never held another.
    small = min((batch for batch in given if len(batch["phase"]) == 1), key=lambda batch: batch["state_card"].shape[1])
    fresh = Graphs(model, "cpu", largest=4)
    first = [made.clone() for made in fresh.answer(small)]
    again = graphs.answer(small)
    assert again is not None and all(torch.equal(one, other) for one, other in zip(again, first, strict=True))
    # A call wider than the sizes fixed, or of more decisions, is not answered here.
    narrow = Graphs(model, "cpu", largest=4, widths={**WIDTHS, "state": small["state_card"].shape[1] - 1})
    assert not narrow.fits(small) and narrow.answer(small) is None and narrow.failed is None
    assert graphs.answer({name: np.concatenate([array] * 5) for name, array in small.items()}) is None


class Standing(Graphs):
    """Graphs whose recording is a stand-in: a replay answers what is in the
    batch now, as a graph does, unless it is told to go ``stale``."""

    stale = False

    def _record(self, sized) -> None:
        with torch.no_grad():
            logits, values = self.model(sized.tensors)
        outer = self

        class Graph:
            def replay(self) -> None:
                if outer.stale:
                    return
                with torch.no_grad():
                    fresh = outer.model(sized.tensors)
                logits.copy_(fresh[0])
                values.copy_(fresh[1])

        sized.graph, sized.answer = Graph(), (logits, values)
        sized.checked = -1
        self.told["recorded"] += 1


def test_a_replay_is_held_against_the_model_once_an_update(played):
    trainer, _, given = played
    model = trainer.model
    calls = [batch for batch in given if len(batch["phase"]) <= 4][:12]
    plain = Graphs(model, "cpu", largest=4)
    graphs = Standing(model, "cpu", largest=4, records=True)
    graphs.begin_update()
    for batch in calls:
        ours, theirs = graphs.answer(batch), plain.answer(batch)
        assert ours is not None and theirs is not None
        assert all(torch.equal(one, other) for one, other in zip(ours, theirs, strict=True))
    used = len(graphs.sized)
    assert graphs.told == {"answered": len(calls), "recorded": used, "checked": used}
    # The next update's first call of each count is held against the model again.
    graphs.begin_update()
    for batch in calls:
        assert graphs.answer(batch) is not None
    assert graphs.told["checked"] == 2 * used and graphs.told["recorded"] == used
    # Weights that are no longer where they were: every graph is recorded again.
    moved = next(model.parameters())
    moved.data = moved.data.clone()
    graphs.begin_update()
    for batch in calls:
        assert graphs.answer(batch) is not None
    assert graphs.told["recorded"] == 2 * used
    # A replay that is not the model's answer ends it: the calls are
    # answered as they are from there, and the reason is kept.
    graphs.stale = True
    graphs.begin_update()
    other = next(batch for batch in calls if not np.array_equal(batch["scalars"], calls[-1]["scalars"]))
    assert graphs.answer(other) is None
    assert graphs.failed is not None and "was not the model's answer" in graphs.failed
    assert graphs.answer(calls[0]) is None


def test_how_the_model_is_asked_follows_the_device_where_it_is_not_named():
    """On a card the games of a pool are asked the way measured fastest
    there: calls of up to half the pool's duels from graphs, the duels in two
    groups that go on together once their rounds are small.  On a processor
    nothing is changed; and what a run names is what it takes."""
    pooled = {"pool_threads": 20, "pool_slots": 16}
    fastest = Calls(groups=2, groups_over=GROUPS_OVER, graphs=160)
    assert GROUPS_OVER == 64
    assert TrainerConfig(**pooled).calls("cuda") == fastest == TrainerConfig(**pooled).calls("cuda:0")
    assert TrainerConfig(**pooled).calls("cpu") == Calls() == Calls(groups=1, groups_over=0, graphs=0)
    assert TrainerConfig().calls("cuda") == Calls(2, GROUPS_OVER, 8)
    # A thread's duels that do not share out in two, or one queue for all the
    # threads: one group, and the graphs all the same.
    assert TrainerConfig(pool_threads=20, pool_slots=5).calls("cuda") == Calls(graphs=50)
    assert TrainerConfig(**pooled, pool_shared=True).calls("cuda") == Calls(graphs=160)
    assert TrainerConfig(pool_threads=1, pool_slots=1).calls("cuda") == Calls(graphs=1)
    # Named: no graphs is one group, since a round split in two is slower
    # with calls as they are.
    assert TrainerConfig(**pooled, inference_graphs=0).calls("cuda") == Calls()
    assert TrainerConfig(**pooled, inference_graphs=64).calls("cuda") == Calls(2, GROUPS_OVER, 64)
    assert TrainerConfig(**pooled, pool_groups=1).calls("cuda") == Calls(graphs=160)
    assert TrainerConfig(**pooled, pool_groups=4).calls("cuda") == Calls(4, GROUPS_OVER, 160)
    assert TrainerConfig(**pooled, pool_groups_over=0).calls("cuda") == Calls(2, 0, 160)
    assert TrainerConfig(**pooled, pool_groups=2).calls("cpu") == Calls(2, GROUPS_OVER, 0)
    assert TrainerConfig(**pooled, pool_groups=2, pool_groups_over=0).calls("cpu") == Calls(2, 0, 0)
    assert TrainerConfig(**pooled, inference_graphs=8).calls("cpu") == Calls(graphs=8)
    with PpoTrainer(ROOT, config()) as trainer:
        assert trainer.calls == Calls() and trainer.graphs is None
        assert trainer._pool().groups == 1


def test_an_updates_games_through_the_fixed_sizes_are_the_games():
    """The games, their figures and the update after them are those of the
    calls answered as they are; and with the duels in two groups too."""
    torch.set_num_threads(2)
    calls = ("inference_batches", "mean_inference_batch", "max_inference_batch", "pool_grouped_batches")

    def played_(metrics: dict) -> dict:
        return {
            name: value
            for name, value in metrics.items()
            if "second" not in name and name not in calls and not name.startswith("pool_graph")
        }

    def course(trajectories: list[dict]) -> list[dict]:
        dropped = ("packed", "row", "log_prob", "value")
        return [
            {**game, "steps": [{k: v for k, v in step.items() if k not in dropped} for step in game["steps"]]}
            for game in trajectories
        ]

    with (
        PpoTrainer(ROOT, config()) as plain,
        PpoTrainer(ROOT, config(inference_graphs=4)) as graphed,
        PpoTrainer(
            ROOT, config(inference_graphs=4, pool_groups=2, pool_groups_over=5)
        ) as both,
    ):
        assert plain.graphs is None and graphed.graphs is not None and not graphed.graphs.records
        whole, asked = plain.collect()
        fixed_, through = graphed.collect()
        split, grouped = both.collect()
        assert course(fixed_) == course(whole) == course(split)
        estimates = [
            [step[name] for game in games for step in game["steps"] for name in ("log_prob", "value")]
            for games in (fixed_, split, whole)
        ]
        assert estimates[0] == pytest.approx(estimates[2], abs=1e-4)
        assert estimates[1] == pytest.approx(estimates[2], abs=1e-4)
        # The calls of up to four decisions went through the fixed sizes, the others not.
        assert "pool_graph_batches" not in asked
        assert 0 < through["pool_graph_batches"] < through["inference_batches"] == asked["inference_batches"]
        assert through["pool_graph_failed"] == 0.0 and graphed.graphs.failed is None
        assert graphed.graphs.told["answered"] == through["pool_graph_batches"]
        # In two groups a group's round is four decisions at most.
        assert grouped["pool_graph_batches"] > through["pool_graph_batches"]
        for trainer in (plain, graphed, both):
            trainer._pool().forget()
        first, second, third = plain.train_update(), graphed.train_update(), both.train_update()
        assert played_(second) == played_(first) == played_(third)
        for other in (graphed, both):
            assert value_fingerprint(other.model.state_dict()) == value_fingerprint(plain.model.state_dict())
            assert value_fingerprint(other.optimizer.state_dict()) == value_fingerprint(
                plain.optimizer.state_dict()
            )
    with pytest.raises(ValueError, match="inference_graphs must be non-negative"):
        config(inference_graphs=-1)


def test_the_way_of_a_card_with_a_checkpoint_held_and_duels_against_itself(tmp_path):
    """Two groups of duels and calls at fixed sizes, with every kind of other
    seat in an update: the games and the update of one group asked as it is.
    What the held checkpoint answers is asked as it is either way."""
    from ygo_training.checkpoints import save
    from ygo_training.model import PolicyValueNet

    torch.set_num_threads(2)
    torch.manual_seed(5)
    held = PolicyValueNet()
    checkpoint = tmp_path / "held" / "model-00006.pt"
    save(checkpoint, held, torch.optim.Adam(held.parameters()), 7, {})
    shares = {
        "duels": 12,
        "opponent_decks": OPPONENTS[:2],
        "self_play_duels": 4,
        "opponent_checkpoint": str(checkpoint),
        "opponent_checkpoint_duels": 4,
    }
    calls = ("inference_batches", "mean_inference_batch", "max_inference_batch", "pool_grouped_batches")

    def played_(metrics: dict) -> dict:
        return {
            name: value
            for name, value in metrics.items()
            if "second" not in name and name not in calls and not name.startswith("pool_graph")
        }

    def course(trajectories: list[dict]) -> list[dict]:
        dropped = ("packed", "row", "log_prob", "value")
        return [
            {**game, "steps": [{k: v for k, v in step.items() if k not in dropped} for step in game["steps"]]}
            for game in trajectories
        ]

    with (
        PpoTrainer(ROOT, config(**shares)) as plain,
        PpoTrainer(
            ROOT, config(**shares, inference_graphs=4, pool_groups=2, pool_groups_over=3)
        ) as fast,
    ):
        assert plain.calls == Calls() and fast.calls == Calls(groups=2, groups_over=3, graphs=4)
        whole, asked = plain.collect()
        split, grouped = fast.collect()
        assert sorted(game["other"] for game in whole) == ["checkpoint"] * 4 + ["pilot"] * 4 + ["self"] * 4
        assert course(split) == course(whole)
        assert grouped["held_decisions"] == asked["held_decisions"] > 10
        assert grouped["decisions"] == asked["decisions"] > 200
        assert grouped["pool_graph_batches"] > 0 and grouped["pool_graph_failed"] == 0.0
        assert 0 < grouped["pool_grouped_batches"] < grouped["inference_batches"]
        for trainer in (plain, fast):
            trainer._pool().forget()
        first, second = plain.train_update(), fast.train_update()
        assert played_(second) == played_(first)
        assert value_fingerprint(fast.model.state_dict()) == value_fingerprint(plain.model.state_dict())
        assert value_fingerprint(fast.optimizer.state_dict()) == value_fingerprint(
            plain.optimizer.state_dict()
        )
