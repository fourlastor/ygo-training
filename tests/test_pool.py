"""Games played through the library's pool are the duels played one at a time."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from ygo_training.checkpoints import value_fingerprint
from ygo_training.config import TrainerConfig
from ygo_training.environment import DuelError, Environment, library_path
from ygo_training.inference import _sample_with_seeds
from ygo_training.orchestration import evaluation_assignments
from ygo_training.pool import (
    DuelPool,
    _setups,
    evaluate_games,
    imitation_games,
    sampled_games,
    taught_games,
)
from ygo_training.ppo_trainer import PpoTrainer
from ygo_training.schedules import duel_sample_seed, keeps_label, rollout_games
from ygo_training.searched import Reading

ROOT = Path(__file__).resolve().parents[1]


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

OPPONENTS = ["monarch", "heroes", "burn", "gladiator", "lightsworn", "blackwing"]


def assignments(count: int) -> list[dict]:
    return [
        {
            "index": game,
            "opponent": OPPONENTS[game % len(OPPONENTS)],
            "learner_player": game % 2,
            "seed": 52_000 + game * 3,
        }
        for game in range(count)
    ]


@pytest.fixture(scope="module")
def trainer():
    import torch

    # A few threads are enough here, and leave the machine to others.
    torch.set_num_threads(2)
    config = TrainerConfig(workers=2, duels=2, device="cpu", seed=45, opponent_decks=OPPONENTS)
    with PpoTrainer(ROOT, config) as built:
        yield built


@pytest.fixture(scope="module")
def environment():
    with Environment(ROOT) as opened:
        yield opened


def given(environment: Environment, view: dict) -> dict:
    """One decision as the batch of it alone the model is given."""
    import torch

    return {name: torch.from_numpy(array) for name, array in environment.batch([view]).items()}


def alone(environment: Environment, model, games: list[dict], *, pilot: bool = False) -> list[dict]:
    """The games of a Blackwing learner played one duel at a time, each
    decision of its seat scored alone by the model's first choice, or with
    ``pilot`` played by the pilot of its deck: what an evaluation through
    the pool is held to, by their index."""
    import torch

    reading = Reading(environment.vocabulary)
    results = []
    for game in games:
        seat, opponent = int(game["learner_player"]), str(game["opponent"])
        seats = [{"control": "pilot"}, {"control": "pilot"}]
        seats[seat] = {"control": "pilot" if pilot else "caller"}
        setup = {
            "seed": int(game["seed"]),
            "decks": ["blackwing", opponent] if seat == 0 else [opponent, "blackwing"],
            "seats": seats,
            "limit": 4096,
        }
        decisions = labelled = agreed = 0
        answers: Counter[str] = Counter()
        with environment.start(setup) as episode:
            while True:
                event = episode.advance()
                view = event["decision"]
                if view is not None:
                    with torch.no_grad():
                        chosen = int(model(given(environment, view))[0].argmax())
                    episode.choose(chosen)
                    decisions += 1
                    answers[":".join(reading.kinds(view, chosen))] += 1
                    if view["label"] is not None:
                        labelled += 1
                        agreed += int(chosen == int(view["label"]))
                over = event["over"]
                if over is not None:
                    break
        winner = over["winner"]
        results.append(
            {
                "index": int(game["index"]),
                "seed": int(game["seed"]),
                "opponent": opponent,
                "learner_player": seat,
                "outcome": "win" if winner == seat else "loss" if winner == 1 - seat else "draw",
                "terminal": not over["limit"],
                "win_reason": over["reason"],
                "turns": over["turns"],
                "decisions": decisions,
                "pilot_labels": labelled,
                "pilot_agreements": agreed,
                "led": 0,
                "learner_answers": dict(answers),
            }
        )
    return sorted(results, key=lambda result: int(result["index"]))


def pooled(trainer, games, threads: int, slots: int, **more):
    with DuelPool(ROOT, threads=threads, slots=slots) as pool:
        return evaluate_games(
            pool,
            trainer.model,
            "cpu",
            learner="blackwing",
            assignments=games,
            limit=trainer.config.limit,
            **more,
        )


def test_the_models_games_are_the_duels_played_one_at_a_time(trainer, environment):
    """Deal for deal: the result, the turns, the decisions the model took,
    how often it took its pilot's answer, and what kinds of answer it gave."""
    games = assignments(18)
    reference = alone(environment, trainer.model, games)
    assert sum(game["decisions"] for game in reference) > 150
    assert sum(game["pilot_agreements"] for game in reference) > 20
    # One duel at a time on one thread: each decision is scored alone there.
    assert pooled(trainer, games, 1, 1) == reference


def test_many_duels_at_once_play_the_same_games_but_for_a_tie(trainer):
    """A batch changes the last digit of what the network answers, so where
    it rates two answers alike a game can go another way.  A pool's own
    rounds do not depend on the threads' pace: it plays the same games every
    time."""
    games = assignments(18)
    alone = pooled(trainer, games, 1, 1)
    together = pooled(trainer, games, 3, 4)
    assert sum(a != b for a, b in zip(alone, together, strict=True)) <= 2
    assert pooled(trainer, games, 3, 4) == together
    assert sum(a != b for a, b in zip(pooled(trainer, games, 6, 3), together, strict=True)) <= 2


def test_the_pilots_games_are_the_duels_played_one_at_a_time(trainer, environment):
    games = assignments(12)
    played = pooled(trainer, games, 4, 2, pilot=True)
    assert played == alone(environment, trainer.model, games, pilot=True)
    assert all(game["decisions"] == 0 for game in played)


def test_a_trainer_plays_its_evaluations_through_one_pool(trainer, environment):
    """The threads and slots of the configuration, and one pool for all of a
    trainer's evaluations."""
    opponents = ["monarch", "heroes", "burn"]
    games = evaluation_assignments(opponents, 4, 91)
    config = TrainerConfig(
        workers=1,
        duels=1,
        device="cpu",
        seed=45,
        opponent_decks=OPPONENTS,
        pool_threads=1,
        pool_slots=1,
    )
    with PpoTrainer(ROOT, config) as other:
        other.model.load_state_dict(trainer.model.state_dict())
        played = other.evaluate_pool("blackwing", opponents, 4, seed=91)
        assert played == alone(environment, trainer.model, games)
        opened = other._duel_pool
        assert opened is not None
        reference = other.evaluate_pool("blackwing", opponents, 4, seed=91, pilot=True)
        assert reference == alone(environment, trainer.model, games, pilot=True)
        assert other._duel_pool is opened


def test_a_round_is_the_decisions_of_the_duels_in_flight():
    setup = {
        "decks": ["blackwing", "monarch"],
        "seats": [{"control": "caller"}, {"control": "pilot"}],
    }
    with DuelPool(ROOT, threads=2, slots=2) as pool:
        assert pool.capacity == 4
        assert pool.submit([{"seed": seed, **setup} for seed in range(60, 66)]) == 0
        first = pool.step()
        # Four of the six duels have a slot, each waits at its first decision.
        assert (len(first), first.in_flight, first.finished) == (4, 6, [])
        assert first.duel.tolist() == [0, 1, 2, 3]
        assert first.player.tolist() == [0, 0, 0, 0]
        assert (
            first.arrays["action_mask"].shape[0] == 4
            and first.arrays["scalars"].dtype == np.float32
        )
        assert not any(name.startswith("meta_") for name in first.arrays)
        offered = first.arrays["action_mask"].sum(axis=1)
        assert ((first.label >= 0) & (first.label < offered)).all()
        with pytest.raises(DuelError, match="3 answers for the 4 decisions"):
            pool.step([0, 0, 0])
        second = pool.step(first.label)
        assert len(second) == 4 and (second.answer > first.answer).all()
        assert pool.submit([{"seed": 70, **setup}]) == 6


def test_a_decision_is_kept_with_nothing_hidden_beside_the_view_the_network_is_given():
    """What a critic that sees the hidden cards reads in training: beside each
    of the network's decisions, the same decision with every card the duel
    has shown named, but the Decks'.  The network's own input is the same
    with it or without, and never names a card its seat has not seen."""
    games = [
        {"seed": 1000 + index, "opponent": opponent, "learner_player": index % 2}
        for index, opponent in enumerate(["burn", "monarch", "gishki", "heroes"] * 2)
    ]
    caller = {"control": "caller", "alone": True}

    def played(whole: bool) -> tuple[dict, dict | None]:
        with DuelPool(ROOT, threads=2, slots=4) as pool:
            pool.submit(_setups("blackwing", games, caller, 4096, whole=whole), keep=True)
            rows: list[int] = []
            answers: Any = ()
            while True:
                round_ = pool.step(answers)
                rows += round_.row.tolist()
                answers = (
                    round_.arrays["action_mask"].argmax(axis=1).astype(np.uint32)
                    if len(round_)
                    else ()
                )
                if round_.in_flight == 0 and not len(round_):
                    break
            seen = pool.batch(rows)
            if not whole:
                with pytest.raises(DuelError, match="kept without its whole view"):
                    pool.batch(rows, whole=True)
                return seen, None
            return seen, pool.batch(rows, whole=True)

    without, _ = played(False)
    seen, whole = played(True)
    assert seen.keys() == without.keys()
    assert all(np.array_equal(seen[name], without[name]) for name in seen)
    width = seen["state_card"].shape[1]

    def unknown(arrays: dict) -> np.ndarray:
        present = np.arange(width)[None, :] < arrays["state_lengths"][:, None]
        return present & (arrays["state_card"] == 0)

    # The seat does not see the other player's hand or face-down cards; with
    # nothing hidden every card on hand and field is named, the other
    # player's Extra Deck as well.
    assert unknown(seen).sum() > 0
    assert unknown(whole).sum() == 0
    assert (whole["state_lengths"] >= seen["state_lengths"]).all()
    assert (whole["state_lengths"] > seen["state_lengths"]).any()
    # What the seat does see is the same in both.
    assert np.array_equal(whole["phase"], seen["phase"])
    assert np.array_equal(whole["action_mask"], seen["action_mask"])


def test_the_cards_of_a_decision_are_followed_to_its_horizon_as_the_seat_sees_them():
    """What a head that learns how cards work is taught: beside each
    transaction's last decision, where each card of its state is at the
    horizon and what changed.  The games and the network's input are the
    same with it or without, and every card the seat knew at both ends is
    found where it was followed to."""
    games = [
        {"seed": 1100 + index, "opponent": opponent, "learner_player": index % 2}
        for index, opponent in enumerate(["burn", "monarch", "gishki", "heroes", "lightsworn"] * 2)
    ]
    caller = {"control": "caller", "alone": True}

    def played(follow: bool) -> tuple[dict, dict, list[tuple[int, int, int]]]:
        with DuelPool(ROOT, threads=2, slots=4) as pool:
            pool.submit(_setups("blackwing", games, caller, 4096, follow=follow), keep=True)
            rows: list[int] = []
            transactions: list[tuple[int, int, int]] = []
            answers: Any = ()
            while True:
                round_ = pool.step(answers)
                rows += round_.row.tolist()
                transactions += zip(
                    round_.duel.tolist(), round_.player.tolist(), round_.transaction.tolist()
                )
                answers = (
                    round_.arrays["action_mask"].argmax(axis=1).astype(np.uint32)
                    if len(round_)
                    else ()
                )
                if round_.in_flight == 0 and not len(round_):
                    break
            return pool.batch(rows), pool.labels(rows), transactions

    without, nothing, _ = played(False)
    seen, labels, transactions = played(True)
    assert seen.keys() == without.keys()
    assert all(np.array_equal(seen[name], without[name]) for name in seen)
    assert not nothing["labelled"].any() and (nothing["card_place"] == -1).all()
    labelled = labels["labelled"].astype(bool)
    # A label at each transaction's last decision: most transactions are one
    # decision long.
    assert 0.5 < labelled.mean() < 1.0
    assert labelled.sum() <= len(set(transactions))
    assert labels["label_checked"].sum() > 1000
    assert labels["label_agreed"].sum() == labels["label_checked"].sum()
    places = labels["card_place"][labelled]
    present = np.arange(places.shape[1])[None, :] < seen["state_lengths"][labelled][:, None]
    assert places.shape[1] == seen["state_card"].shape[1]
    assert ((places >= 0) | ~present).sum() / present.size > 0.99
    assert (places[~present] == -1).all() and (places[present] < 23).all()
    # Cards go to the Graveyard, to the field and to the hand.
    for place in (6, 1, 0, 11 + 6):
        assert (places == place).sum() > 0
    changes = labels["card_change"][labelled][present]
    assert np.isfinite(changes[:, 0]).sum() > 0
    assert (labels["life_change"][labelled] != 0).any()
    assert (labels["count_change"][labelled] != 0).any()


def imitating(**more) -> TrainerConfig:
    """Updates that imitate the pilot of the learner's deck in games it plays
    now, the value head taught the result at four positions of a game."""
    return TrainerConfig(
        workers=3,
        duels=8,
        device="cpu",
        seed=71,
        opponent_decks=OPPONENTS[:5],
        imitation_only_updates=3,
        imitation_value_weight=0.5,
        imitation_value_positions=4,
        ppo_epochs=2,
        minibatch=48,
        **more,
    )


def told(trajectories: list[dict]) -> list[dict]:
    """Trajectories without where the input of their steps is held."""
    return [
        {
            **trajectory,
            "steps": [
                {key: value for key, value in step.items() if key not in ("packed", "row")}
                for step in trajectory["steps"]
            ],
        }
        for trajectory in trajectories
    ]


def measured(metrics: dict) -> dict:
    """What an update reports, but for how long it took."""
    return {name: value for name, value in metrics.items() if "second" not in name}


def test_the_pilots_games_to_imitate_are_the_duels_played_one_at_a_time(environment):
    """The pilot of the learner's deck in its seat, watched: each of its
    decisions a step, with its answer and in the order of play, kept with the
    input the duel played alone shows there."""
    games = rollout_games(7 * 1024, 3, [OPPONENTS[game % 5] for game in range(8)])
    with DuelPool(ROOT, threads=3, slots=2) as pool:
        trajectories = imitation_games(pool, learner="blackwing", games=games, limit=4096)
        steps = [step for trajectory in trajectories for step in trajectory["steps"]]
        rows = [step["row"] for step in steps]
        assert len(steps) > 400 and len(set(rows)) == len(rows) == pool.kept
        assert {trajectory["learner_player"] for trajectory in trajectories} == {0, 1}
        views = []
        for game, trajectory in zip(games, trajectories, strict=True):
            seat, opponent = int(game["learner_player"]), str(game["opponent"])
            seats = [{"control": "pilot"}, {"control": "pilot"}]
            seats[seat] = {"control": "pilot", "observe": True}
            setup = {
                "seed": int(game["seed"]),
                "decks": ["blackwing", opponent] if seat == 0 else [opponent, "blackwing"],
                "seats": seats,
                "limit": 4096,
            }
            with environment.start(setup) as episode:
                # No seat is the caller's: the duel goes to its end at once.
                event = episode.advance()
            over = event["over"]
            assert (trajectory["winner"], trajectory["turns"]) == (over["winner"], over["turns"])
            assert [
                (int(view["label"]), int(view["chosen"]), view["player"], view["transaction"])
                for view in event["samples"]
            ] == [
                (step["chosen"], step["executed_chosen"], step["player"], step["transaction"])
                for step in trajectory["steps"]
            ]
            views.extend(event["samples"])
        for start in (0, 7, len(steps) - 60):
            picked = list(range(start, len(steps), 9))[:40][::-1]
            kept = pool.batch([rows[index] for index in picked])
            shown = environment.batch([views[index] for index in picked])
            assert list(kept) == list(shown)
            for name, array in shown.items():
                assert kept[name].dtype == array.dtype and np.array_equal(kept[name], array), name


def test_an_update_that_imitates_through_the_pool_is_the_same_every_time():
    """The same games, the same minibatches, and so the same model after it,
    to the last digit; the decisions of an update kept until the next."""
    import torch

    torch.set_num_threads(2)
    with (
        PpoTrainer(ROOT, imitating(pool_threads=3, pool_slots=2)) as one,
        PpoTrainer(ROOT, imitating(pool_threads=3, pool_slots=2)) as other,
    ):
        ours, _ = one.collect()
        theirs, _ = other.collect()
        assert ours == theirs
        steps = [step for trajectory in ours for step in trajectory["steps"]]
        first = one.imitation_update(ours)
        assert measured(other.imitation_update(theirs)) == measured(first)
        assert first["imitation_value_targets"] > 0 and first["samples"] == len(steps) > 400
        assert value_fingerprint(one.model.state_dict()) == value_fingerprint(
            other.model.state_dict()
        )
        # A whole update: other games, the same on both sides again.
        second = measured(one.train_update())
        assert measured(other.train_update()) == second
        assert second["rollout_training_fingerprint"] != value_fingerprint(told(ours))
        assert value_fingerprint(one.model.state_dict()) == value_fingerprint(
            other.model.state_dict()
        )
        # The first update's decisions are gone, and are not taken for others.
        with pytest.raises(DuelError, match="no decision is kept under row"):
            one.imitation_update(ours)


def test_threads_that_draw_their_duels_from_one_queue_imitate_the_same_update():
    """One queue for all the threads, where each is otherwise dealt its
    duels: the pilot's games are the same games, kept under the same rows,
    and the model after two updates is the same to the last digit."""
    import torch

    torch.set_num_threads(2)
    with (
        PpoTrainer(ROOT, imitating(pool_threads=3, pool_slots=2)) as dealt,
        PpoTrainer(
            ROOT, imitating(pool_threads=3, pool_slots=2, pool_shared=True)
        ) as shared,
    ):
        theirs, _ = dealt.collect()
        ours, _ = shared.collect()
        assert ours == theirs and len(ours) > 4
        first = measured(dealt.imitation_update(theirs))
        assert measured(shared.imitation_update(ours)) == first
        assert measured(shared.train_update()) == measured(dealt.train_update())
        assert value_fingerprint(shared.model.state_dict()) == value_fingerprint(
            dealt.model.state_dict()
        )


def test_a_networks_games_to_imitate_are_the_games_it_plays_when_it_is_measured(trainer):
    """A network in the pilot's place takes its first choice at every
    decision: deal for deal the game an evaluation has it play, each decision
    kept with the answer it gave and with its pilot's own beside it."""
    import torch

    games = rollout_games(11 * 1024, 3, [OPPONENTS[game % 5] for game in range(18)])
    with DuelPool(ROOT, threads=1, slots=1) as pool:
        trajectories, asked = taught_games(
            pool, trainer.model, "cpu", learner="blackwing", games=games, limit=4096
        )
        steps = [step for trajectory in trajectories for step in trajectory["steps"]]
        assert pool.kept == len(steps) == asked["prompts"] > 150
        assert (asked["batches"], asked["max_batch"]) == (len(steps), 1)
        for step in steps[::5]:
            kept = pool.batch([step["row"]])
            given = {name: torch.from_numpy(array) for name, array in kept.items()}
            with torch.no_grad():
                assert int(trainer.model(given)[0].argmax()) == step["chosen"]
    assert all(step["teacher"] and step["executed_chosen"] == step["chosen"] for step in steps)
    assert {trajectory["learner_player"] for trajectory in trajectories} == {0, 1}
    measured_ = pooled(
        trainer,
        [
            {
                "index": place,
                "opponent": game["opponent"],
                "learner_player": game["learner_player"],
                "seed": game["seed"],
            }
            for place, game in enumerate(games)
        ],
        1,
        1,
    )
    for trajectory, game in zip(trajectories, measured_, strict=True):
        seat, winner, taken = (
            trajectory["learner_player"],
            trajectory["winner"],
            trajectory["steps"],
        )
        outcome = "win" if winner == seat else "loss" if winner == 1 - seat else "draw"
        assert (outcome, trajectory["turns"], trajectory["win_reason"], len(taken)) == (
            game["outcome"],
            game["turns"],
            game["win_reason"],
            game["decisions"],
        )
        assert sum("pilot_chosen" in step for step in taken) == game["pilot_labels"]
        assert (
            sum(step.get("pilot_chosen") == step["chosen"] for step in taken)
            == game["pilot_agreements"]
        )
    # A network as it is first drawn leaves its pilot at most decisions, and
    # its games are short.
    assert sum(step.get("pilot_chosen", step["chosen"]) != step["chosen"] for step in steps) > 100


def saved(folder: Path, seed: int, **shape) -> Path:
    """A checkpoint of a network as it is first drawn: a teacher that plays,
    however badly."""
    import torch

    from ygo_training.checkpoints import save
    from ygo_training.model import PolicyValueNet

    torch.manual_seed(seed)
    model = PolicyValueNet(**shape)
    path = folder / f"teacher-{seed}" / "model-00006.pt"
    save(path, model, torch.optim.Adam(model.parameters()), 7, {})
    return path


def deals(trajectories: list[dict]) -> list[tuple]:
    return [
        (game["episode_index"], game["worker"], game["learner_player"], game["opponent_deck"])
        for game in trajectories
    ]


def test_updates_that_imitate_a_checkpoint_play_its_games_through_the_pool(tmp_path):
    """The deals are the ones a run that imitates the pilot plays, with the
    checkpoint in the pilot's place; the learner starts as it does without a
    teacher; and the update is the same from one run to the next."""
    import torch

    torch.set_num_threads(2)
    teacher = saved(tmp_path, 5)

    def taught(**more) -> TrainerConfig:
        return imitating(pool_threads=3, pool_slots=2, imitation_teacher=str(teacher), **more)

    with PpoTrainer(ROOT, imitating(pool_threads=3, pool_slots=2)) as pilot:
        started = pilot.initial_model_fingerprint
        drawn = torch.get_rng_state()
        assert pilot.teacher is None and pilot.teacher_description is None
        theirs, _ = pilot.collect()
    with (
        PpoTrainer(ROOT, taught()) as one,
        PpoTrainer(ROOT, taught()) as other,
        PpoTrainer(ROOT, taught(imitation_teacher_targets="distribution")) as soft,
    ):
        # Building the teacher drew nothing the run would have drawn.
        assert torch.equal(torch.get_rng_state(), drawn)
        assert one.initial_model_fingerprint == started == soft.initial_model_fingerprint
        assert one.teacher_description == {
            "path": str(teacher.resolve()),
            "step": 7,
            "fingerprint": value_fingerprint(one.teacher.state_dict()),
            "board_representation": "mean-pool",
            "targets": "choice",
        }
        assert soft.teacher_description["targets"] == "distribution"
        assert one.teacher_description["fingerprint"] != started
        assert not one.teacher.training
        assert not any(parameter.requires_grad for parameter in one.teacher.parameters())

        ours, rolled = one.collect()
        assert deals(ours) == deals(theirs)
        steps = [step for trajectory in ours for step in trajectory["steps"]]
        assert all(step["teacher"] and "row" in step for step in steps)
        assert rolled["decisions"] == len(steps) > 50 and rolled["max_inference_batch"] > 3
        assert rolled["inference_seconds"] > 0 and rolled["pool_threads_seconds"] > 0
        # The games are the teacher's, not the pilot's.
        assert told(ours) != told(theirs)

        other.collect()
        first = measured(one.train_update())
        assert measured(other.train_update()) == first
        assert first["samples"] > 50 and first["warmup"] == 1.0
        assert first["imitation_value_targets"] > 0
        assert first["imitation_teacher_agreement"] == 1.0
        assert first["imitation_departures"] > 20
        assert 0 <= first["imitation_model_departure_agreement"] <= 1
        assert first["imitation_teacher_divergence"] > 0 and first["imitation_teacher_entropy"] > 0
        assert value_fingerprint(one.model.state_dict()) == value_fingerprint(
            other.model.state_dict()
        )
        assert value_fingerprint(one.model.state_dict()) != started
        assert value_fingerprint(one.teacher.state_dict()) == one.teacher_description["fingerprint"]

        # What is imitated of the teacher changes the lesson and not the
        # games, which are the teacher's whatever the learner has become.
        soft.collect()
        softly = measured(soft.train_update())
        assert softly["rollout_decision_fingerprint"] == first["rollout_decision_fingerprint"]
        assert softly["imitation_loss"] != first["imitation_loss"]
        assert softly["imitation_teacher_entropy"] == first["imitation_teacher_entropy"]
        assert value_fingerprint(soft.model.state_dict()) != value_fingerprint(
            one.model.state_dict()
        )


def test_a_teacher_may_read_the_board_another_way_than_its_learner(tmp_path):
    import torch

    torch.set_num_threads(2)
    teacher = saved(tmp_path, 6, board_representation="individual-tokens")
    config = imitating(pool_threads=2, pool_slots=2, imitation_teacher=str(teacher))
    with PpoTrainer(ROOT, config) as trainer:
        assert trainer.model.board_representation == "mean-pool"
        assert trainer.teacher_description["board_representation"] == "individual-tokens"
        metrics = trainer.train_update()
        assert metrics["samples"] > 50 and metrics["imitation_teacher_agreement"] == 1.0


def test_a_learner_of_several_decks_plays_each_in_its_share_of_an_updates_games():
    """Each deck against every opponent as often, each game the one its own
    deck's pilot plays, and the agreement told deck by deck."""
    import torch

    torch.set_num_threads(2)
    decks = ["blackwing", "monarch"]

    def several(**more) -> TrainerConfig:
        return TrainerConfig(
            workers=3,
            duels=12,
            device="cpu",
            seed=71,
            learner_decks=decks,
            opponent_decks=["heroes", "burn", "gladiator"],
            pool_threads=3,
            pool_slots=2,
            ppo_epochs=1,
            minibatch=64,
            **more,
        )

    imitating_ = {
        "imitation_only_updates": 2,
        "imitation_value_weight": 0.5,
        "imitation_value_positions": 4,
    }
    with (
        PpoTrainer(ROOT, several(**imitating_)) as one,
        PpoTrainer(ROOT, several(**imitating_)) as other,
    ):
        assert one.coordinator.learners == decks
        ours, rolled = one.collect()
        # Two games of each deck against each opponent.
        assert sorted((game["learner_deck"], game["opponent_deck"]) for game in ours) == sorted(
            (deck, opponent) for deck in decks for opponent in one.opponents * 2
        )
        assert [game["learner_deck"] for game in ours] != sorted(
            game["learner_deck"] for game in ours
        )
        assert rolled["learner_deck_duels"] == {"blackwing": 6.0, "monarch": 6.0}
        assert set(rolled["learner_deck_win_rate"]) == set(decks)
        # A deck's games are the games a learner of that deck alone has of
        # the same deals: its own pilot in its seat.
        for deck in decks:
            mine = [game for game in ours if game["learner_deck"] == deck]
            games = [
                {
                    "episode_index": game["episode_index"],
                    "worker": game["worker"],
                    "seed": rolled_seed(one, game),
                    "opponent": game["opponent_deck"],
                    "learner_player": game["learner_player"],
                }
                for game in mine
            ]
            with DuelPool(ROOT, threads=2, slots=2) as pool:
                alone = imitation_games(pool, learner=deck, games=games, limit=4096)
            assert told(alone) == told(mine)
        assert told(ours) != told([{**game, "learner_deck": "blackwing"} for game in ours])

        theirs, _ = other.collect()
        first = measured(one.imitation_update(ours))
        assert measured(other.imitation_update(theirs)) == first
        agreed = first["imitation_model_teacher_agreement_by_deck"]
        counted = first["imitation_samples_by_deck"]
        assert set(agreed) == set(counted) == set(decks)
        assert sum(counted.values()) == first["samples"]
        assert first["imitation_model_teacher_agreement"] == pytest.approx(
            sum(agreed[deck] * counted[deck] for deck in decks) / first["samples"]
        )
        # A whole update, the same from one run to the next.
        second = measured(one.train_update())
        assert measured(other.train_update()) == second
        assert set(second["imitation_model_teacher_agreement_by_deck"]) == set(decks)
        assert value_fingerprint(one.model.state_dict()) == value_fingerprint(
            other.model.state_dict()
        )
        # Past the imitating updates the model plays each deck itself.
        third = one.train_update()
        assert third["learner_deck_duels"] == {"blackwing": 6.0, "monarch": 6.0}
        assert third["ppo_optimizer_steps"] > 0 and "warmup" not in third
        # Each deck is measured with its own list as the model's own.
        played = one.evaluate_pool("monarch", ["heroes"], 2, seed=91)
        assert [game["opponent"] for game in played] == ["heroes", "heroes"]


def test_a_learner_of_several_decks_plays_itself_with_any_two_of_them():
    """Of an update's duels, those against itself have one of the learner's
    decks in each seat and its network in both: every decision of them is
    learned from, with a result for each seat.  Deck by deck an update tells
    the duels against the pilots, and the seats against itself.  The update
    is the same from one run to the next, and asked the way of a card."""
    from collections import Counter

    import torch

    torch.set_num_threads(2)
    decks = ["blackwing", "monarch"]
    pilots = ["heroes", "burn"]

    def several(**more) -> TrainerConfig:
        return TrainerConfig(
            workers=3,
            duels=12,
            device="cpu",
            seed=71,
            learner_decks=decks,
            opponent_decks=pilots,
            pool_threads=2,
            pool_slots=4,
            ppo_epochs=1,
            minibatch=64,
            self_play_duels=8,
            # An update the same however its games' calls were batched: so
            # with the estimates worked out again, not with each game's own.
            rollout_estimates="recomputed",
            **more,
        )

    def course(trajectories: list[dict]) -> list[dict]:
        """The games without what the model estimated in its own batches."""
        dropped = ("packed", "row", "log_prob", "value")
        return [
            {**game, "steps": [{k: v for k, v in step.items() if k not in dropped} for step in game["steps"]]}
            for game in trajectories
        ]

    def played(metrics: dict) -> dict:
        asked = ("inference_batches", "mean_inference_batch", "max_inference_batch")
        return {
            name: value
            for name, value in measured(metrics).items()
            if name not in asked and not name.startswith(("pool_graph", "pool_grouped"))
        }

    with (
        PpoTrainer(ROOT, several()) as one,
        PpoTrainer(ROOT, several()) as other,
        PpoTrainer(
            ROOT, several(inference_graphs=4, pool_groups=2, pool_groups_over=3)
        ) as fast,
    ):
        ours, rolled = one.collect()
        assert Counter(game["other"] for game in ours) == {"self": 8, "pilot": 4}
        assert [game["other"] for game in ours] != sorted(game["other"] for game in ours)
        # Against itself each pair of its decks twice, a deck in each seat;
        # against the pilots each deck once against each.
        itself = [game for game in ours if game["other"] == "self"]
        assert Counter((game["learner_deck"], game["opponent_deck"]) for game in itself) == {
            (own, other_): 2 for own in decks for other_ in decks
        }
        assert sorted(
            (game["learner_deck"], game["opponent_deck"]) for game in ours if game["other"] == "pilot"
        ) == sorted((deck, pilot) for deck in decks for pilot in pilots)
        assert rolled["other_seat_duels"] == {"pilot": 4.0, "self": 8.0}
        # Deck by deck: the duels against the pilots, and the seats against
        # itself, of which one wins in a duel that has a winner.
        assert rolled["learner_deck_duels"] == {"blackwing": 2.0, "monarch": 2.0}
        assert set(rolled["learner_deck_win_rate"]) == set(decks)
        assert rolled["self_play_deck_seats"] == {"blackwing": 8.0, "monarch": 8.0}
        rates = rolled["self_play_deck_win_rate"]
        assert sum(rates[deck] * 8 for deck in decks) == pytest.approx(
            sum(game["winner"] in (0, 1) for game in itself)
        )
        assert rolled["self_play_mean_turns"] > 1

        # A duel of two decks against itself has the decisions of both seats.
        crossed = next(game for game in itself if game["learner_deck"] != game["opponent_deck"])
        assert {step["player"] for step in crossed["steps"]} == {0, 1}

        # Every step is learned from, and a duel against itself has a result
        # for each of its seats: one wins what the other loses.
        samples, _, prepared = one.policy_optimizer.prepare_samples(ours)
        assert prepared["samples"] == len(samples) == sum(len(game["steps"]) for game in ours)
        for place, game in enumerate(ours):
            ends = {
                (sample.step["player"], sample.terminal_return)
                for sample in samples
                if sample.step["_transaction_key"][0] == place
            }
            if game["other"] != "self":
                assert {player for player, _ in ends} == {game["learner_player"]}
            elif game["winner"] in (0, 1):
                assert ends == {(game["winner"], 1.0), (1 - game["winner"], -1.0)}
            else:
                assert ends == {(0, -0.5), (1, -0.5)}

        # The same from one run to the next, and with the duels in two
        # groups and the calls at fixed sizes.
        theirs, _ = other.collect()
        split, grouped = fast.collect()
        assert told(theirs) == told(ours)
        assert course(split) == course(ours)
        assert grouped["pool_graph_batches"] > 0 and grouped["pool_graph_failed"] == 0.0
        for trainer in (one, other, fast):
            trainer._pool().forget()
        first = one.train_update()
        assert measured(other.train_update()) == measured(first)
        assert played(fast.train_update()) == played(first)
        assert first["ppo_optimizer_steps"] > 0 and first["samples"] > 200
        for again in (other, fast):
            assert value_fingerprint(again.model.state_dict()) == value_fingerprint(
                one.model.state_dict()
            )


def rolled_seed(trainer, game: dict) -> int:
    """The seed of a game of the trainer's first update."""
    from ygo_training.schedules import ROLLOUT_SEED_SLOTS, ROLLOUT_SEED_STRIDE, RUN_SEED_MULTIPLIER

    slot = (trainer.config.seed * RUN_SEED_MULTIPLIER) % ROLLOUT_SEED_SLOTS
    return slot * ROLLOUT_SEED_STRIDE + int(game["episode_index"])


def one_at_a_time(
    environment: Environment,
    model,
    games: list[dict],
    *,
    rollout_seed: int,
    fraction: float,
    held=None,
) -> list[dict]:
    """The games ``sampled_games`` plays, each duel alone and each decision
    scored alone, with what the model was given at every step.  In a game
    against ``checkpoint`` the network ``held`` answers the other seat by its
    first choice, and how often it did is told (``held``)."""
    import torch

    reading = Reading(environment.vocabulary)
    trajectories = []
    for game in games:
        seat, opponent = int(game["learner_player"]), str(game["opponent"])
        names = ["blackwing", opponent] if seat == 0 else [opponent, "blackwing"]
        other = str(game.get("other", "pilot"))
        seats = [{"control": "pilot" if other == "pilot" else "caller"} for _ in range(2)]
        seats[seat] = {"control": "caller"}
        setup = {"seed": int(game["seed"]), "decks": names, "seats": seats, "limit": 4096}
        steps = []
        theirs = 0
        with environment.start(setup) as episode:
            while True:
                event = episode.advance()
                view = event["decision"]
                if view is not None:
                    shown = given(environment, view)
                    if other == "checkpoint" and int(view["player"]) != seat:
                        with torch.no_grad():
                            episode.choose(int(held(shown)[0].argmax()))
                        theirs += 1
                        continue
                    with torch.no_grad():
                        logits, values = model(shown)
                    seed = duel_sample_seed(rollout_seed, int(game["episode_index"]), len(steps))
                    drawn = _sample_with_seeds(logits, [seed], 1.0)
                    chosen = int(drawn[0])
                    question, answer = reading.kinds(view, chosen)
                    step = {
                        "given": shown,
                        "chosen": chosen,
                        "log_prob": float(
                            torch.distributions.Categorical(logits=logits).log_prob(drawn)[0]
                        ),
                        "value": float(values[0]),
                        "player": int(view["player"]),
                        "transaction": int(view["transaction"]),
                        "prompt_type": question,
                        "action_kind": answer,
                    }
                    if view["label"] is not None and keeps_label(
                        int(game["seed"]), len(steps), fraction
                    ):
                        step["teacher_chosen"] = int(view["label"])
                    steps.append(step)
                    episode.choose(chosen)
                if event["over"] is not None:
                    over = event["over"]
                    break
        trajectory = {
            "kind": "trajectory",
            "worker": int(game["worker"]),
            "winner": over["winner"],
            "steps": steps,
            "terminal": not over["limit"],
            "learner_player": seat,
            "win_reason": over["reason"],
            "turns": over["turns"],
            "learner_deck": "blackwing",
            "opponent_deck": opponent,
            "episode_index": int(game["episode_index"]),
        }
        if "other" in game:
            trajectory.update(other=other, held=theirs)
        trajectories.append(trajectory)
    return trajectories


def without(trajectories: list[dict], *keys: str) -> list[dict]:
    return [
        {
            **trajectory,
            "steps": [
                {key: value for key, value in step.items() if key not in keys}
                for step in trajectory["steps"]
            ],
        }
        for trajectory in trajectories
    ]


def test_the_games_the_model_draws_its_answers_in_follow_from_each_duel_alone(trainer, environment):
    """Duel for duel what is played one duel at a time: the answers drawn,
    their probabilities, the values, the pilot's answers kept beside them,
    and the input each decision is kept with."""
    import torch

    seed = 9 * 1024
    games = rollout_games(seed, 3, [OPPONENTS[game % 5] for game in range(10)])
    alone = one_at_a_time(environment, trainer.model, games, rollout_seed=seed, fraction=0.5)

    def played(threads: int, slots: int):
        with DuelPool(ROOT, threads=threads, slots=slots) as pool:
            trajectories, asked = sampled_games(
                pool,
                trainer.model,
                "cpu",
                learner="blackwing",
                games=games,
                limit=4096,
                rollout_seed=seed,
                temperature=1.0,
                labels=True,
                label_fraction=0.5,
            )
            steps = [step for trajectory in trajectories for step in trajectory["steps"]]
            assert pool.kept == len(steps) == asked["prompts"]
            given = [pool.batch([step["row"]]) for step in steps]
            return trajectories, asked, given

    single, asked, given = played(1, 1)
    assert without(single, "row") == without(alone, "given")
    steps = [step for trajectory in alone for step in trajectory["steps"]]
    assert len(steps) > 200 and (asked["batches"], asked["max_batch"]) == (len(steps), 1)
    labelled = sum("teacher_chosen" in step for step in steps)
    assert 0.3 * len(steps) < labelled < 0.7 * len(steps)
    # Drawn, not the first choice: the pilot's answer is often another.
    assert sum(step["chosen"] != step.get("teacher_chosen", step["chosen"]) for step in steps) > 20
    for step, kept in zip(steps, given, strict=True):
        assert list(kept) == list(step["given"])
        for name, tensor in step["given"].items():
            assert torch.equal(torch.from_numpy(kept[name]), tensor), name

    # Many duels at once: a batch changes the last digit of a probability and
    # of a value, so the same answers but where a draw falls on that digit;
    # and the same every time, to that digit.
    together, asked, _ = played(3, 4)
    assert asked["max_batch"] > 6 and asked["batches"] < len(steps) / 2
    scored = ("row", "log_prob", "value")
    differing = sum(
        a != b for a, b in zip(without(together, *scored), without(single, *scored), strict=True)
    )
    assert differing <= 2
    close = [
        abs(a["log_prob"] - b["log_prob"]) < 1e-4 and abs(a["value"] - b["value"]) < 1e-4
        for ours, theirs in zip(together, single, strict=True)
        if len(ours["steps"]) == len(theirs["steps"])
        for a, b in zip(ours["steps"], theirs["steps"], strict=True)
    ]
    assert len(close) > 200 and sum(close) > 0.9 * len(close)
    assert without(played(3, 4)[0], "row") == without(together, "row")


def test_labels_are_kept_for_the_asked_share_and_leave_the_duels_alone(trainer):
    """The pilot's answers kept beside the model's, at every decision or at a
    share of them, or at none: the same games every time."""
    seed = 10 * 1024
    games = rollout_games(seed, 3, [OPPONENTS[game % 5] for game in range(6)])

    def played(labels: bool, fraction: float = 1.0) -> list[dict]:
        with DuelPool(ROOT, threads=2, slots=2) as pool:
            return sampled_games(
                pool,
                trainer.model,
                "cpu",
                learner="blackwing",
                games=games,
                limit=4096,
                rollout_seed=seed,
                temperature=1.0,
                labels=labels,
                label_fraction=fraction,
            )[0]

    def labelled(trajectories: list[dict]) -> list[list[bool]]:
        return [["teacher_chosen" in step for step in t["steps"]] for t in trajectories]

    full, thinned, none = played(True), played(True, 0.3), played(False)
    steps = sum(len(flags) for flags in labelled(full))
    assert steps > 100 and all(all(flags) for flags in labelled(full))
    assert not any(any(flags) for flags in labelled(none))
    assert 0 < sum(sum(flags) for flags in labelled(thinned)) < steps
    # The labels do not change what is played.
    assert without(thinned, "teacher_chosen") == without(full, "teacher_chosen") == none
    assert keeps_label(7, 3, 1.0) and keeps_label(7, 3, 0.3) == keeps_label(7, 3, 0.3)


def test_an_update_whose_games_the_model_plays_goes_through_the_pool():
    """The deals are the same from one run to the next, and so is the
    update."""
    import torch

    torch.set_num_threads(2)

    def config(**more) -> TrainerConfig:
        return TrainerConfig(
            workers=2,
            duels=6,
            device="cpu",
            seed=33,
            opponent_decks=OPPONENTS[:5],
            teacher_imitation_weight=0.5,
            ppo_epochs=2,
            minibatch=48,
            # The estimates worked out again, as the trainer did by default
            # before 2026-10-20: the way through the pool that does more.
            rollout_estimates="recomputed",
            **more,
        )

    with (
        PpoTrainer(ROOT, config(pool_threads=2, pool_slots=3)) as one,
        PpoTrainer(ROOT, config(pool_threads=2, pool_slots=3)) as other,
    ):
        ours, rolled = one.collect()
        steps = [step for trajectory in ours for step in trajectory["steps"]]
        assert all("row" in step and "packed" not in step for step in steps)
        assert rolled["decisions"] == len(steps) > 100 and rolled["max_inference_batch"] > 3
        assert rolled["mean_inference_batch"] > 2 and rolled["inference_seconds"] > 0
        assert {trajectory["learner_player"] for trajectory in ours} == {0, 1}

        theirs, _ = other.collect()
        assert deals(theirs) == deals(ours)
        first = measured(one.train_update())
        assert measured(other.train_update()) == first
        assert first["recomputed_rollout_estimates"] == first["samples"] > 100
        assert 0 < first["teacher_label_coverage"] <= 1 and first["teacher_imitation_samples"] > 0
        assert first["ppo_optimizer_steps"] > 0
        assert value_fingerprint(one.model.state_dict()) == value_fingerprint(
            other.model.state_dict()
        )
        assert value_fingerprint(one.model.state_dict()) != one.initial_model_fingerprint


def test_a_pass_builds_its_next_minibatch_while_a_step_is_made(monkeypatch):
    """Another thread builds the pass's minibatches, each once and one ahead
    of the step being made; and the update is the one that building each
    when it is asked for gives."""
    import threading

    import torch

    from ygo_training import optimization

    torch.set_num_threads(2)

    def config() -> TrainerConfig:
        return TrainerConfig(
            workers=1,
            duels=6,
            device="cpu",
            seed=33,
            opponent_decks=OPPONENTS[:5],
            pool_threads=2,
            pool_slots=3,
            ppo_epochs=2,
            minibatch=48,
        )

    class WhenAsked:
        """In the builder's place: a minibatch is built when it is read."""

        def __init__(self, **_) -> None:
            pass

        def submit(self, build, rows):
            class Asked:
                def result(self):
                    return build(rows)

            return Asked()

        def shutdown(self, wait: bool = True) -> None:
            pass

    with (
        PpoTrainer(ROOT, config()) as ahead,
        PpoTrainer(ROOT, config()) as asked,
    ):
        optimizer = ahead.policy_optimizer
        build, step = optimizer.kept, optimizer._step
        built: list[tuple[str, tuple[int, ...]]] = []
        waiting: list[list[tuple[int, ...]]] = []

        def kept(rows):
            built.append((threading.current_thread().name, tuple(rows)))
            return build(rows)

        def stepped(loss):
            waiting.append(list(optimizer._ahead))
            return step(loss)

        optimizer.kept, optimizer._step = kept, stepped
        first = measured(ahead.train_update())
        steps = int(first["ppo_optimizer_steps"])
        # In the pass the pool is asked by the builder alone, a minibatch once.
        threaded = [rows for name, rows in built if name.startswith("minibatch")]
        assert steps > 4 and len(threaded) == steps and len(set(threaded)) == steps
        passes = steps // 2
        for epoch in (threaded[:passes], threaded[passes:]):
            rows = sorted(row for minibatch in epoch for row in minibatch)
            assert len(rows) == first["samples"] and len(set(rows)) == len(rows)
        # While a step is made the one minibatch waiting is the next of its pass.
        assert waiting == [
            [] if position % passes == passes - 1 else [threaded[position + 1]]
            for position in range(steps)
        ]
        assert optimizer._ahead == {} and optimizer._builder is None

        monkeypatch.setattr(optimization, "ThreadPoolExecutor", WhenAsked)
        assert measured(asked.train_update()) == first
        assert value_fingerprint(asked.model.state_dict()) == value_fingerprint(
            ahead.model.state_dict()
        )
        assert value_fingerprint(asked.optimizer.state_dict()) == value_fingerprint(
            ahead.optimizer.state_dict()
        )


def test_groups_of_duels_taken_forward_in_turn_play_the_games_of_one_group():
    """While the model answers one group of a thread's duels the engines of
    the other run: the same duels with the same draws, answered in other
    batches, and the same update after them."""
    import torch

    torch.set_num_threads(2)

    def config(groups: int, over: int = 0) -> TrainerConfig:
        return TrainerConfig(
            workers=1,
            duels=8,
            device="cpu",
            seed=33,
            opponent_decks=OPPONENTS[:5],
            pool_threads=2,
            pool_slots=4,
            pool_groups=groups,
            pool_groups_over=over,
            self_play_duels=4,
            ppo_epochs=1,
            minibatch=48,
            # The same update however the duels were grouped: so with the
            # estimates worked out again, not with each game's own.
            rollout_estimates="recomputed",
        )

    # How the model was asked is another matter than what it was asked.
    calls = (
        "inference_batches",
        "mean_inference_batch",
        "max_inference_batch",
        "pool_grouped_batches",
    )

    def played(metrics: dict) -> dict:
        return {name: value for name, value in measured(metrics).items() if name not in calls}

    with (
        PpoTrainer(ROOT, config(1)) as one,
        PpoTrainer(ROOT, config(2)) as two,
        PpoTrainer(ROOT, config(2, over=5)) as mixed,
    ):
        assert one._pool().groups == 1 and two._pool().groups == 2
        whole, asked = one.collect()
        halves, in_turn = two.collect()
        # By groups while their rounds hold more than five decisions between
        # them, and all together from there: the same games again, in fewer
        # calls than the groups make to the last duel.
        largest, then_whole = mixed.collect()

        def course(trajectories: list[dict]) -> list[dict]:
            """The games without what the model estimated in their own
            batches, whose last digits change with what shares a batch: an
            update works those out again."""
            return [
                {
                    **game,
                    "steps": [
                        {key: value for key, value in step.items() if key not in ("log_prob", "value")}
                        for step in game["steps"]
                    ],
                }
                for game in told(trajectories)
            ]

        assert course(halves) == course(whole) == course(largest)
        estimates = [
            [step[name] for game in games for step in game["steps"] for name in ("log_prob", "value")]
            for games in (halves, largest, whole)
        ]
        assert estimates[0] == pytest.approx(estimates[2], abs=1e-5)
        assert estimates[1] == pytest.approx(estimates[2], abs=1e-5)
        assert len({step["row"] for game in halves for step in game["steps"]}) == asked["decisions"]
        # More calls of fewer decisions, and time the engines ran beside them.
        assert in_turn["decisions"] == asked["decisions"] > 300
        assert in_turn["inference_batches"] > asked["inference_batches"]
        assert in_turn["max_inference_batch"] <= 4 < asked["max_inference_batch"]
        assert in_turn["pool_wait_seconds"] >= 0 and "pool_wait_seconds" not in asked
        assert in_turn["pool_grouped_batches"] == in_turn["inference_batches"]
        assert "pool_grouped_batches" not in asked
        assert then_whole["decisions"] == asked["decisions"]
        assert (
            asked["inference_batches"]
            < then_whole["inference_batches"]
            < in_turn["inference_batches"]
        )
        assert 2 <= then_whole["pool_grouped_batches"] < then_whole["inference_batches"]
        assert then_whole["max_inference_batch"] > 4
        for trainer in (one, two, mixed):
            trainer._pool().forget()
        first = one.train_update()
        second = two.train_update()
        third = mixed.train_update()
        assert played(second) == played(first) == played(third)
        assert second["inference_batches"] > third["inference_batches"] > first["inference_batches"]
        for other in (two, mixed):
            assert value_fingerprint(other.model.state_dict()) == value_fingerprint(
                one.model.state_dict()
            )
            assert value_fingerprint(other.optimizer.state_dict()) == value_fingerprint(
                one.optimizer.state_dict()
            )
        # A measurement takes all the groups forward together, as one group.
        games = assignments(6)
        assert evaluate_games(
            two._pool(), two.model, "cpu", learner="blackwing", assignments=games, limit=4096
        ) == evaluate_games(
            one._pool(), one.model, "cpu", learner="blackwing", assignments=games, limit=4096
        )


def test_a_pool_of_groups_is_refused_what_it_cannot_do():
    with pytest.raises(ValueError, match="pool_groups must share pool_slots out evenly"):
        TrainerConfig(pool_threads=2, pool_slots=5, pool_groups=2)
    with pytest.raises(ValueError, match="pool_groups must share pool_slots out evenly"):
        TrainerConfig(pool_threads=2, pool_groups=0)
    with pytest.raises(ValueError, match="do not go together"):
        TrainerConfig(pool_threads=2, pool_slots=4, pool_groups=2, pool_shared=True)
    with pytest.raises(ValueError, match="pool_groups_over must be non-negative"):
        TrainerConfig(pool_threads=2, pool_slots=4, pool_groups=2, pool_groups_over=-1)
    with pytest.raises(DuelError, match="not shared out evenly"):
        DuelPool(ROOT, threads=1, slots=3, groups=2)
    with DuelPool(ROOT, threads=1, slots=2, groups=2) as pool:
        with pytest.raises(DuelError, match="group 0 was not begun"):
            pool.finish(0)
        pool.begin(0)
        with pytest.raises(DuelError, match="on its way already"):
            pool.begin(0)
        assert len(pool.finish(0)) == 0


def drawn(seed: int, **shape):
    """A network as it is first drawn, which plays however badly."""
    import torch

    from ygo_training.model import PolicyValueNet

    torch.manual_seed(seed)
    return PolicyValueNet(**shape).eval()


def mirrors(count: int, **more) -> list[dict]:
    """Games of the mirror, the learner's seat in turn."""
    return [
        {
            "index": game,
            "opponent": "blackwing",
            "learner_player": game % 2,
            "seed": 53_000 + game * 7,
            **more,
        }
        for game in range(count)
    ]


def from_the_other_seat(games: list[dict]) -> list[dict]:
    """The same duels, told for the other seat."""
    return [{**game, "learner_player": 1 - int(game["learner_player"])} for game in games]


OPPOSITE = {"win": "loss", "loss": "win", "draw": "draw"}


def test_a_game_against_a_held_checkpoint_is_one_duel_of_two_networks(trainer, environment):
    """Each network takes its first choice at the decisions of its own seat.
    The duel is the same whichever of the two is the one measured: told for
    the other seat it has the other outcome, and the decisions told are those
    of the seat that is measured."""
    import torch

    other = drawn(5)
    games = mirrors(6, other="checkpoint")
    with DuelPool(ROOT, threads=1, slots=1) as pool:
        ours = evaluate_games(
            pool,
            trainer.model,
            "cpu",
            learner="blackwing",
            assignments=games,
            limit=4096,
            held=other,
        )
        theirs = evaluate_games(
            pool,
            other,
            "cpu",
            learner="blackwing",
            assignments=from_the_other_seat(games),
            limit=4096,
            held=trainer.model,
        )
    assert [game["learner_player"] for game in ours] == [0, 1] * 3
    for game, mine, yours in zip(games, ours, theirs, strict=True):
        assert (mine["outcome"], mine["turns"], mine["win_reason"], mine["terminal"]) == (
            OPPOSITE[yours["outcome"]],
            yours["turns"],
            yours["win_reason"],
            yours["terminal"],
        )
        # The duel one decision at a time, each seat's network asked alone.
        names = ["blackwing", "blackwing"]
        seat = int(game["learner_player"])
        asked = [0, 0]
        agreed = 0
        setup = {
            "seed": int(game["seed"]),
            "decks": names,
            "seats": [{"control": "caller"}, {"control": "caller"}],
            "limit": 4096,
        }
        with environment.start(setup) as episode:
            while True:
                event = episode.advance()
                view = event["decision"]
                if view is not None:
                    player = int(view["player"])
                    network = trainer.model if player == seat else other
                    with torch.no_grad():
                        chosen = int(network(given(environment, view))[0].argmax())
                    asked[player] += 1
                    agreed += int(player == seat and view["label"] == chosen)
                    episode.choose(chosen)
                if event["over"] is not None:
                    over = event["over"]
                    break
        winner = over["winner"]
        outcome = "win" if winner == seat else "loss" if winner == 1 - seat else "draw"
        assert (mine["outcome"], mine["turns"], mine["win_reason"]) == (
            outcome,
            over["turns"],
            over["reason"],
        )
        assert (mine["decisions"], yours["decisions"]) == (asked[seat], asked[1 - seat])
        assert mine["pilot_agreements"] == agreed and min(asked) > 20
    # Two networks are not one: neither side's games are the other's.
    assert [game["decisions"] for game in ours] != [game["decisions"] for game in theirs]


def test_the_pilot_against_a_held_checkpoint_is_the_checkpoint_against_the_pilot(trainer):
    """The reference of a case against a checkpoint: the deck's pilot in the
    learner's seat.  It is the duel in which the checkpoint is measured
    against that pilot, told for the other seat."""
    other = drawn(5)
    games = mirrors(6, other="checkpoint")
    with DuelPool(ROOT, threads=1, slots=1) as pool:
        reference = evaluate_games(
            pool,
            trainer.model,
            "cpu",
            learner="blackwing",
            assignments=games,
            limit=4096,
            pilot=True,
            held=other,
        )
        plain = [
            {key: value for key, value in game.items() if key != "other"}
            for game in from_the_other_seat(games)
        ]
        measured_ = evaluate_games(
            pool, other, "cpu", learner="blackwing", assignments=plain, limit=4096
        )
    assert all(game["decisions"] == 0 for game in reference)
    assert [(game["outcome"], game["turns"]) for game in reference] == [
        (OPPOSITE[game["outcome"]], game["turns"]) for game in measured_
    ]
    assert {game["outcome"] for game in reference} == {"win"}
    # A checkpoint is named where a game asks for one, and a game is measured
    # against a pilot or a checkpoint and nothing else.
    with (
        DuelPool(ROOT, threads=1, slots=1) as pool,
        pytest.raises(DuelError, match="none is held"),
    ):
        evaluate_games(
            pool, trainer.model, "cpu", learner="blackwing", assignments=games[:1], limit=4096
        )
    with (
        DuelPool(ROOT, threads=1, slots=1) as pool,
        pytest.raises(ValueError, match="a pilot or a checkpoint"),
    ):
        evaluate_games(
            pool,
            trainer.model,
            "cpu",
            learner="blackwing",
            assignments=mirrors(1, other="self"),
            limit=4096,
        )


def test_games_against_itself_and_against_a_held_checkpoint_follow_from_each_duel_alone(
    trainer, environment
):
    """Duel for duel what is played one duel at a time.  Against itself the
    model draws at both seats and every decision is a step; against a held
    checkpoint the other seat's are the checkpoint's first choices, and no
    steps; against a pilot nothing has changed."""
    import torch

    seed = 13 * 1024
    held = drawn(5)
    others = ["pilot", "self", "checkpoint", "self", "checkpoint", "pilot", "self", "checkpoint"]
    decks = [
        OPPONENTS[place] if other == "pilot" else "blackwing" for place, other in enumerate(others)
    ]
    games = rollout_games(seed, 3, decks, None, others)
    alone = one_at_a_time(
        environment, trainer.model, games, rollout_seed=seed, fraction=0.5, held=held
    )

    def played(threads: int, slots: int):
        with DuelPool(ROOT, threads=threads, slots=slots) as pool:
            trajectories, asked = sampled_games(
                pool,
                trainer.model,
                "cpu",
                learner="blackwing",
                games=games,
                limit=4096,
                rollout_seed=seed,
                temperature=1.0,
                labels=True,
                label_fraction=0.5,
                held=held,
            )
            steps = [step for trajectory in trajectories for step in trajectory["steps"]]
            given = [pool.batch([step["row"]]) for step in steps]
            return trajectories, asked, given, pool.kept

    single, asked, given, kept = played(1, 1)
    assert without(single, "row") == [
        {key: value for key, value in trajectory.items() if key != "held"}
        for trajectory in without(alone, "given")
    ]
    assert [trajectory["other"] for trajectory in single] == others
    steps = [step for trajectory in alone for step in trajectory["steps"]]
    theirs = sum(trajectory["held"] for trajectory in alone)
    assert asked["prompts"] == len(steps) > 300 and asked["held_decisions"] == theirs > 30
    # The pool keeps every decision that waited, the checkpoint's among them.
    assert kept == len(steps) + theirs
    for step, kept_ in zip(steps, given, strict=True):
        for name, tensor in step["given"].items():
            assert torch.equal(torch.from_numpy(kept_[name]), tensor), name
    for trajectory in single:
        seats = {step["player"] for step in trajectory["steps"]}
        if trajectory["other"] == "self":
            # Both seats' decisions, in the order they were played.
            assert seats == {0, 1}
        else:
            assert seats == {trajectory["learner_player"]}

    # Many duels at once: the same games but where a draw falls on the last
    # digit of a probability, and the same every time.
    together, asked, _, _ = played(3, 3)
    assert asked["max_batch"] > 3 and asked["batches"] < len(steps)
    scored = ("row", "log_prob", "value")
    differing = sum(
        a != b for a, b in zip(without(together, *scored), without(single, *scored), strict=True)
    )
    assert differing <= 2
    assert without(played(3, 3)[0], "row") == without(together, "row")


def test_an_update_with_duels_against_itself_and_a_held_checkpoint(tmp_path):
    """An update's duels in three shares, by who sits in the other seat.  The
    learner starts as it does with pilots throughout, both seats of a duel
    against itself are learned from, and the update is the same from one run
    to the next."""
    import torch

    torch.set_num_threads(2)
    checkpoint = saved(tmp_path, 5)

    def config(**more) -> TrainerConfig:
        return TrainerConfig(
            workers=3,
            duels=12,
            device="cpu",
            seed=33,
            opponent_decks=OPPONENTS[:2],
            ppo_epochs=1,
            minibatch=64,
            pool_threads=2,
            pool_slots=3,
            **more,
        )

    shares = {
        "self_play_duels": 4,
        "opponent_checkpoint": str(checkpoint),
        "opponent_checkpoint_duels": 4,
    }
    with PpoTrainer(ROOT, config()) as plain:
        started = plain.initial_model_fingerprint
        drawn_ = torch.get_rng_state()
        assert plain.held is None and plain.held_description is None
        theirs, rolled = plain.collect()
        assert all("other" not in game for game in theirs) and "other_seat_duels" not in rolled
    with (
        PpoTrainer(ROOT, config(**shares)) as one,
        PpoTrainer(ROOT, config(**shares)) as other,
    ):
        # Holding the checkpoint drew nothing the run would have drawn.
        assert torch.equal(torch.get_rng_state(), drawn_)
        assert one.initial_model_fingerprint == started
        assert one.held_description == {
            "path": str(checkpoint.resolve()),
            "step": 7,
            "fingerprint": value_fingerprint(one.held.state_dict()),
            "board_representation": "mean-pool",
        }
        assert not one.held.training
        assert not any(parameter.requires_grad for parameter in one.held.parameters())

        ours, rolled = one.collect()
        kinds = [game["other"] for game in ours]
        assert sorted(kinds) == ["checkpoint"] * 4 + ["pilot"] * 4 + ["self"] * 4
        assert kinds != sorted(kinds)
        # The mirror against a network, the named decks against their pilots.
        assert all(
            (game["opponent_deck"] == "blackwing") == (game["other"] != "pilot") for game in ours
        )
        assert sorted(game["opponent_deck"] for game in ours if game["other"] == "pilot") == sorted(
            OPPONENTS[:2] * 2
        )
        assert rolled["other_seat_duels"] == {"checkpoint": 4.0, "pilot": 4.0, "self": 4.0}
        assert set(rolled["other_seat_win_rate"]) == {"checkpoint", "pilot", "self"}
        assert 0 <= rolled["self_play_first_seat_win_rate"] <= 1
        assert rolled["self_play_mean_turns"] > 1 and rolled["held_decisions"] > 10
        steps = [step for game in ours for step in game["steps"]]
        assert rolled["decisions"] == len(steps) > 200

        # Every step is learned from, and a duel against itself has a result
        # for each of its seats: one wins what the other loses.
        samples, _, prepared = one.policy_optimizer.prepare_samples(ours)
        assert prepared["samples"] == len(samples) == len(steps)
        for place, game in enumerate(ours):
            ends = {
                (sample.step["player"], sample.terminal_return)
                for sample in samples
                if sample.step["_transaction_key"][0] == place
            }
            if game["other"] != "self":
                assert {player for player, _ in ends} == {game["learner_player"]}
            elif game["winner"] in (0, 1):
                assert ends == {(game["winner"], 1.0), (1 - game["winner"], -1.0)}
            else:
                assert ends == {(0, -0.5), (1, -0.5)}

        other.collect()
        first = measured(one.train_update())
        assert measured(other.train_update()) == first
        assert first["samples"] > 200 and first["ppo_optimizer_steps"] > 0
        assert first["other_seat_duels"] == {"checkpoint": 4.0, "pilot": 4.0, "self": 4.0}
        assert value_fingerprint(one.model.state_dict()) == value_fingerprint(
            other.model.state_dict()
        )
        assert value_fingerprint(one.model.state_dict()) != started
        assert value_fingerprint(one.held.state_dict()) == one.held_description["fingerprint"]

        # Measured against the checkpoint it holds: the games of the pool's own
        # function, and of no other opponent.
        against = one.evaluate_pool("blackwing", ["blackwing"], 4, seed=91, checkpoint=True)
        with DuelPool(ROOT, threads=2, slots=3) as pool:
            direct = evaluate_games(
                pool,
                one.model,
                "cpu",
                learner="blackwing",
                assignments=[
                    {**game, "other": "checkpoint"}
                    for game in evaluation_assignments(["blackwing"], 4, 91)
                ],
                limit=4096,
                held=one.held,
            )
        assert against == direct and len(against) == 4
        assert against != one.evaluate_pool("blackwing", ["blackwing"], 4, seed=91)
        with pytest.raises(ValueError, match="No checkpoint is held"):
            plain_again = PpoTrainer(ROOT, config())
            try:
                plain_again.evaluate_pool("blackwing", ["blackwing"], 2, seed=91, checkpoint=True)
            finally:
                plain_again.close()


def test_updates_that_imitate_play_the_pilots_whatever_sits_in_the_other_seat_later(tmp_path):
    """The duels against itself are of the updates whose games the model
    plays: an imitating update before them is the one a run without has."""
    import torch

    torch.set_num_threads(2)

    def config(**more) -> TrainerConfig:
        return TrainerConfig(
            workers=3,
            duels=8,
            device="cpu",
            seed=71,
            opponent_decks=OPPONENTS[:2],
            imitation_only_updates=1,
            ppo_epochs=1,
            minibatch=64,
            pool_threads=2,
            pool_slots=3,
            **more,
        )

    with (
        PpoTrainer(ROOT, config()) as plain,
        PpoTrainer(ROOT, config(self_play_duels=6)) as itself,
    ):
        theirs, _ = plain.collect()
        ours, _ = itself.collect()
        assert told(ours) == told(theirs) and all("other" not in game for game in ours)
        assert measured(itself.imitation_update(ours)) == measured(plain.imitation_update(theirs))
        later, rolled = itself.collect()
        assert rolled["other_seat_duels"] == {"pilot": 2.0, "self": 6.0}
        assert [game["other"] for game in later].count("self") == 6


# What makes Blackwing - Sirocco the Dawn's script fail from the 20th call of
# any of its functions on, as `crates/ygo-env/tests/duels.rs` makes it fail.
FAILING_AT_SIROCCO = """
local calls = 0
for name, f in pairs(s) do
	if type(f) == "function" and name ~= "initial_effect" then
		s[name] = function(...)
			calls = calls + 1
			if calls >= 20 then error("a script that fails, for a test") end
			return f(...)
		end
	end
end
"""


def _root_whose_script_fails(tmp_path: Path) -> Path:
    """A checkout as the pool reads one: the policies' cards and decks, and
    their scripts with Sirocco's failing, at the root of a folder of links
    that the engine reads before ``official/``."""
    policies = ROOT / "native/ygo-policies"
    root = tmp_path / "root"
    scripts = root / "native/ygo-policies/vendor/CardScripts"
    scripts.mkdir(parents=True)
    (root / "native/ygo-policies/vendor/BabelCdb").symlink_to(policies / "vendor/BabelCdb")
    (root / "native/ygo-policies/decks").symlink_to(policies / "decks")
    for path in (policies / "vendor/CardScripts").glob("*.lua"):
        (scripts / path.name).symlink_to(path)
    for kept in ("official", "pre-errata"):
        (scripts / kept).symlink_to(policies / "vendor/CardScripts" / kept)
    sirocco = (policies / "vendor/CardScripts/official/c75498415.lua").read_text()
    (scripts / "c75498415.lua").write_text(sirocco + FAILING_AT_SIROCCO)
    return root


def test_a_duel_the_engine_stops_ends_as_the_limit_ends_one_and_is_reported(
    trainer, tmp_path, monkeypatch
):
    monkeypatch.setenv("YGO_ENV_LIBRARY", str(library_path(ROOT)))
    root = _root_whose_script_fails(tmp_path)
    game = {"index": 0, "opponent": "monarch", "learner_player": 0, "seed": 7}
    with DuelPool(root, threads=1, slots=2) as pool:
        [played] = evaluate_games(
            pool,
            trainer.model,
            "cpu",
            learner="blackwing",
            assignments=[game],
            limit=4096,
            pilot=True,
        )
        failed = pool.failed()
        assert pool.failed() == []
    assert (played["terminal"], played["outcome"]) == (False, "draw")
    [report] = failed
    assert "c75498415.lua" in report["failure"]
    assert "a script that fails, for a test" in report["failure"]
    # The game it was, what was submitted for it, and its record so far.
    assert {key: report[key] for key in game} == game
    assert report["learner"] == "blackwing"
    assert (report["setup"]["seed"], report["setup"]["decks"]) == (7, ["blackwing", "monarch"])
    assert report["decisions"] > 0
    assert len(report["record"]["responses"]) == report["decisions"]
