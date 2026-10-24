"""Recorded duels played again through the pool, and what a search's log
says of their decisions, read from each decision as the seat is shown it.

The model is scripted: it answers as each test tells it.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from tests.recorded import recorded_duel
from tests.views import (
    ACTIVATE,
    FACE_DOWN_DEFENSE,
    PASS,
    SELECT_CARDS,
    SPELL,
    SPELL_TRAP_ZONE,
    TRAP,
    card,
    choice,
    hidden,
    named,
    view,
)
from ygo_training.environment import LIBRARY, Environment
from ygo_training.pool import DuelPool, evaluate_games, replayed_games, watched_games
from ygo_training.records import recorded_games
from ygo_training.searched import Reading, recorded_setup

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "native/ygo-policies/target/release"

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not (ROOT / LIBRARY).exists(), reason="the environment is not built: cargo build --release"
    ),
]


@pytest.fixture(scope="module")
def environment():
    with Environment(ROOT) as opened:
        yield opened


@pytest.fixture(scope="module")
def reading(environment):
    return Reading(environment.vocabulary)


class Scripted:
    """A network that answers as it is told, in the order it is asked: at
    each decision the answer ``answer`` names, given the decision's place
    among those asked and how many answers it offers, scored 2 where the
    others score 0; a value of 0.25; and with ``leads``, the lead its head
    predicts for each answer, given how many there are."""

    def __init__(self, answer, leads=None):
        self.answer = answer
        self.leads = leads
        self.asked = 0

    def evaluate(self, batch):
        import torch

        mask = batch["action_mask"]
        logits = torch.zeros(mask.shape).masked_fill(~mask, float("-inf"))
        predicted = torch.zeros(mask.shape)
        for row, offered in enumerate(mask.sum(-1).tolist()):
            logits[row, self.answer(self.asked, offered)] = 2.0
            if self.leads is not None:
                predicted[row, :offered] = torch.tensor(self.leads(offered))
            self.asked += 1
        values = torch.full((mask.shape[0],), 0.25)
        return logits, values, None if self.leads is None else predicted

    def __call__(self, batch):
        logits, values, _ = self.evaluate(batch)
        return logits, values


def first(place, offered):
    return 0


def recorded(environment, **duel):
    return recorded_duel(environment, **duel)


def replay(games, label="played"):
    """Recorded duels played again for an update that imitates them, from
    the sixth game of the update on."""
    with DuelPool(ROOT, threads=2, slots=2) as pool:
        return replayed_games(
            pool,
            learner="blackwing",
            games=[
                {**game, "episode_index": 5 + place, "worker": 3}
                for place, game in enumerate(games)
            ],
            limit=4096,
            label=label,
        )


def watch(game, answer, leads=None):
    """A recorded duel watched by a scripted network: its reading, and the
    network."""
    network = Scripted(answer, leads)
    with DuelPool(ROOT, threads=2, slots=2) as pool:
        (result,) = watched_games(
            pool, network, "cpu", learner="blackwing", assignments=[game], limit=4096
        )
    return result, network


def total(result, name):
    return sum(counts.get(name, 0) for counts in result["questions"].values())


def test_a_recorded_duel_is_played_again_and_its_answers_are_the_steps(environment):
    searched = recorded(environment)
    piloted = recorded(environment, seed=8, opponent="heroes", learner_player=1, leave_at=0)
    trajectories = replay([searched, piloted])

    assert [t["opponent_deck"] for t in trajectories] == ["monarch", "heroes"]
    assert [t["learner_player"] for t in trajectories] == [0, 1]
    assert [t["episode_index"] for t in trajectories] == [5, 6]
    assert [t["worker"] for t in trajectories] == [3, 3]
    assert [t["winner"] for t in trajectories] == [searched["winner"], piloted["winner"]]
    assert [t["departures"] for t in trajectories] == [1, 0]
    rows = [step["row"] for t in trajectories for step in t["steps"]]
    assert rows == list(range(len(rows))) and len(rows) > 100
    for trajectory in trajectories:
        for step in trajectory["steps"]:
            assert step["teacher"] and step["chosen"] == step["executed_chosen"]
            assert step["player"] == trajectory["learner_player"]
            assert "log_prob" not in step
    # The record left its pilot once, and there the two answers are told apart.
    left = [
        [step for step in t["steps"] if step.get("pilot_chosen", step["chosen"]) != step["chosen"]]
        for t in trajectories
    ]
    assert [len(steps) for steps in left] == [1, 0]
    assert all("pilot_chosen" in step for step in trajectories[1]["steps"])


def test_a_record_that_plays_as_another_duel_is_refused(environment):
    game = recorded(environment)
    with pytest.raises(ValueError, match="it was made with another engine"):
        replay([{**game, "digest": "0" * 16}])
    with pytest.raises(ValueError, match="leaves its pilot 1 times, not the 0 it lists"):
        replay([{**game, "deviations": []}])
    with pytest.raises(ValueError, match="it was made with another engine"):
        watch({**game, "digest": "0" * 16}, first)


def test_a_model_is_read_against_the_answers_of_a_recorded_duel(environment):
    game = recorded(environment)
    (trajectory,) = replay([game])
    played = [step["chosen"] for step in trajectory["steps"]]
    pilots = [step.get("pilot_chosen") for step in trajectory["steps"]]

    # A model that answers as the record does: the search's answer where it
    # left the pilot, the pilot's everywhere else.
    result, network = watch(game, lambda place, offered: played[place])
    assert network.asked == result["decisions"] == len(played)
    assert (result["recorded_deviations"], result["examined_known"]) == (1, False)
    assert total(result, "left_search") == 1 and total(result, "kept_other") == 0
    assert total(result, "kept_pilot") + total(result, "unread") == len(played) - 1
    (deviation,) = result["deviations"]
    assert deviation["answer"] == game["deviations"][0] and deviation["reading"] == "search"
    assert deviation["search_probability"] > deviation["pilot_probability"] > 0.0
    assert deviation["question"] in ("Idle", "Battle", "Chain")

    # A model that answers as the pilot does is read the other way there.
    answers = [played[index] if pilot is None else pilot for index, pilot in enumerate(pilots)]
    result, _ = watch(game, lambda place, offered: answers[place])
    assert total(result, "left_pilot") == 1 and total(result, "kept_other") == 0
    assert result["deviations"][0]["reading"] == "pilot"

    # What the search examined splits the decisions it kept.
    result, _ = watch({**game, "examined": [*game["deviations"], 0, 1, 2, 3, 4, 5]}, first)
    assert result["examined_known"]
    examined = total(result, "examined_pilot") + total(result, "examined_other")
    assert 0 < examined < total(result, "kept_pilot") + total(result, "kept_other")


def offered(environment, kind, code=0, location=None):
    """A choice as a decision offers it."""
    vocabulary = environment.vocabulary
    card = None
    if location is not None:
        card = {"code": code, "controller": 0, "location": vocabulary["location"][location]}
    return {"kind": vocabulary["choice"][kind], "card": card, "description": 0}


def test_like_cards_in_one_pile_are_one_alternative_as_a_search_tries_them(environment, reading):
    decision = {
        "pilot": 2,
        "choices": [
            offered(environment, "NormalSummon", 5, "Hand"),
            offered(environment, "NormalSummon", 5, "Hand"),
            offered(environment, "NormalSummon", 6, "Hand"),
            offered(environment, "NormalSummon", 6, "Hand"),
            # The same card twice on the field is two cards.
            offered(environment, "ChangePosition", 7, "MonsterZone"),
            offered(environment, "ChangePosition", 7, "MonsterZone"),
            # Another thing done with a card is another alternative.
            offered(environment, "SetMonster", 5, "Hand"),
            offered(environment, "EndTurn"),
        ],
    }

    assert reading.alternatives(decision) == [[2, 3], [0, 1], [4], [5], [6], [7]]
    assert reading.alternative(decision, 2) == ["NormalSummon", 6]
    assert reading.alternative(decision, 7) == ["EndTurn", 0]

    # A search tries twelve alternatives at most, the pilot's among them.
    # What comes after is in none, but for a like card of one of the twelve.
    many = {
        "pilot": 0,
        "choices": [
            offered(environment, "NormalSummon", 5, "Hand"),
            *(offered(environment, "ChangePosition", 7, "MonsterZone") for _ in range(13)),
            offered(environment, "NormalSummon", 5, "Hand"),
        ],
    }
    listed = reading.alternatives(many)
    assert len(listed) == 12 and listed[0] == [0, 14]
    assert [group for group in listed[1:]] == [[index] for index in range(1, 12)]


def decisions(environment, game):
    """What the watched seat of a recorded duel is asked, decision by decision."""
    names = ["blackwing", game["opponent"]]
    if game["learner_player"] == 1:
        names.reverse()
    with environment.start(recorded_setup(game, names, 4096)) as episode:
        event = episode.advance()
        assert event["over"] is not None
        return event["samples"]


def commands(environment, reading, game):
    """The decisions of a recorded duel a search examines: the commands."""
    return [
        shown
        for shown in decisions(environment, game)
        if shown["pilot"] is not None
        and len(shown["choices"]) > 1
        and reading.kinds(shown, 0)[0] in ("Idle", "Battle", "Chain")
    ]


def logged(reading, shown, leads=None, **fields):
    """A decision as a search's log would hold it, read from the decision."""
    listed = reading.alternatives(shown)
    return {
        "answer": shown["answer"],
        "labels": [reading.alternative(shown, group[0]) for group in listed],
        "leads": leads or [0.0, *(0.125 for _ in listed[1:])],
        "taken": 0,
        "outright": False,
        **fields,
    }


def test_a_models_answer_is_set_among_what_the_search_tried(environment, reading):
    game = recorded(environment)
    views = commands(environment, reading, game)
    left = next(shown for shown in views if shown["answer"] == game["deviations"][0])
    others = [shown for shown in views if shown is not left][:4]
    log = [logged(reading, shown, rate=0.5, worlds=8) for shown in others]
    log.append(logged(reading, left, taken=1, outright=False))
    log.sort(key=lambda entry: entry["answer"])

    # A model that always gives the first answer offered.
    result, _ = watch({**game, "tried": json.dumps(log)}, first)

    assert [entry["answer"] for entry in result["tried"]] == [entry["answer"] for entry in log]
    assert result["tried_unread"] == 0
    for entry, asked in zip(result["tried"], sorted([*others, left], key=lambda v: v["answer"])):
        listed = reading.alternatives(asked)
        assert entry["model"] == next(place for place, group in enumerate(listed) if 0 in group)
        assert len(entry["weights"]) == len(entry["leads"]) == len(listed)
        assert 0.0 < sum(entry["weights"]) <= 1.0 + 1e-9
        # The scripted model gives its answer most of the weight.
        assert max(range(len(listed)), key=entry["weights"].__getitem__) == entry["model"]
        assert entry["question"] in ("Idle", "Battle", "Chain")
        # The model's value of the decision, and what the pilot's answer is
        # worth by the worlds where the log holds them.
        assert (entry["value"], entry["turn"]) == (0.25, asked["turn"])
        assert (entry["rate"], entry["worlds"]) == ((None, None) if asked is left else (0.5, 8))
    assert [entry["taken"] for entry in result["tried"] if entry["answer"] == left["answer"]] == [1]
    # A command opens its transaction: that is where the value head is trained.
    first_of = {}
    for asked in decisions(environment, game):
        first_of.setdefault(asked["transaction"], asked["answer"])
    assert [entry["start"] for entry in result["tried"]] == [
        first_of[asked["transaction"]] == asked["answer"]
        for asked in sorted([*others, left], key=lambda v: v["answer"])
    ]
    assert any(entry["start"] for entry in result["tried"])

    # An answer of the log the seat is not asked is counted, not read.
    stray = {**log[0], "answer": 10**6}
    result, _ = watch({**game, "tried": json.dumps([*log, stray])}, first)
    assert (len(result["tried"]), result["tried_unread"]) == (len(log), 1)
    # A duel without a log is read as before.
    result, _ = watch(game, first)
    assert (result["tried"], result["tried_unread"]) == ([], 0)


def test_a_log_that_tries_other_alternatives_than_the_decision_offers_is_refused(
    environment, reading
):
    game = recorded(environment)
    entry = logged(reading, commands(environment, reading, game)[0])
    entry["labels"][-1] = ["Activate", 12345]

    with pytest.raises(ValueError, match="the search's log tries") as failure:
        watch({**game, "tried": json.dumps([entry])}, first)
    assert "where the decision offers" in str(failure.value)
    with pytest.raises(ValueError, match="the search's log tries"):
        replay([{**game, "tried": json.dumps([entry])}])


def test_the_leads_a_search_logged_are_the_targets_of_its_decisions(environment, reading):
    game = recorded(environment)
    views = commands(environment, reading, game)
    left = next(shown for shown in views if shown["answer"] == game["deviations"][0])
    others = [shown for shown in views if shown is not left and len(shown["choices"]) > 2][:3]
    log = [logged(reading, shown) for shown in others]
    # The last alternative of the first was not played out: it has no lead.
    log[0]["leads"][-1] = None
    leads = [0.0, 0.25, *(-0.5 for _ in reading.alternatives(left)[2:])]
    log.append(logged(reading, left, leads=leads))
    log.sort(key=lambda entry: entry["answer"])
    examined = sorted([*others, left], key=lambda shown: shown["answer"])

    trajectories = replay([{**game, "tried": json.dumps(log)}])

    targeted = [step for step in trajectories[0]["steps"] if "lead_targets" in step]
    assert len(targeted) == len(log)
    for step, entry, asked in zip(targeted, log, examined, strict=True):
        listed = reading.alternatives(asked)
        wanted = [
            [group[0], lead]
            for group, lead in zip(listed, entry["leads"], strict=True)
            if lead is not None
        ]
        assert step["lead_targets"] == wanted
        # The pilot's own answer comes first, with no lead over itself.
        assert step["lead_targets"][0] == [step["pilot_chosen"], 0.0]
    assert len(targeted[0]["lead_targets"]) == len(reading.alternatives(examined[0])) - 1

    # A duel without a log has no targets, and a log is not read past the duel.
    trajectories = replay([game])
    assert not any("lead_targets" in step for step in trajectories[0]["steps"])
    stray = {**log[0], "answer": 10**6}
    with pytest.raises(ValueError, match="the seat was not asked"):
        replay([{**game, "tried": json.dumps([*log, stray])}])


def test_the_policy_can_imitate_the_pilot_where_a_search_played_another_answer(
    environment, reading
):
    game = recorded(environment)
    views = commands(environment, reading, game)
    left = next(shown for shown in views if shown["answer"] == game["deviations"][0])
    log = [logged(reading, left, taken=1)]
    played = replay([{**game, "tried": json.dumps(log)}])
    piloted = replay([{**game, "tried": json.dumps(log)}], label="pilot")

    steps = piloted[0]["steps"]
    # The same duel, with the steps its pilot was never asked left out.
    assert piloted[0]["winner"] == played[0]["winner"]
    assert len(steps) == sum("pilot_chosen" in step for step in played[0]["steps"])
    assert all(step["chosen"] == step["pilot_chosen"] for step in steps)
    # Where the search left the pilot, the label is the pilot's and the
    # targets are the search's measurements all the same.
    departed = [step for step in steps if step["chosen"] != step["executed_chosen"]]
    assert len(departed) == 1
    assert departed[0]["lead_targets"][0] == [departed[0]["chosen"], 0.0]


def test_a_search_examines_a_choice_with_no_chain_and_no_face_down_monster_before_it(reading):
    assert reading.examined(view(pilot=0))
    # A model has no pilot's answer: its first choice stands in its place.
    assert not reading.examined(view())
    assert reading.examined(view(), first=1)
    # A chain under way, or a monster of the other player's it cannot deal again.
    assert not reading.examined(view(pilot=0, chain=[{"code": TRAP}]))
    assert not reading.examined(view(pilot=0, cards=[hidden()]))
    assert reading.examined(view(pilot=0, cards=[card(position=FACE_DOWN_DEFENSE)]))
    assert reading.examined(view(pilot=0, cards=[hidden(location=SPELL_TRAP_ZONE)]))
    # A chain window that cannot be passed, a selection, and nothing to choose between.
    forced = view(pilot=0)
    forced["question"]["forced"] = True
    assert not reading.examined(forced)
    selection = view(pilot=0)
    selection["question"]["kind"] = SELECT_CARDS
    assert not reading.examined(selection)
    assert not reading.examined(view(pilot=0, choices=[choice(PASS)]))


def test_a_player_leaves_its_first_choice_for_a_lead_of_the_margin(reading):
    asked = view(
        choices=[
            choice(ACTIVATE, named(code=TRAP, location=SPELL_TRAP_ZONE)),
            choice(PASS),
            choice(ACTIVATE, named(code=SPELL, location=SPELL_TRAP_ZONE)),
        ]
    )
    leads = [0.0, 0.05, 0.3]

    assert reading.led(asked, 0, leads, 0.2) == 2
    assert reading.led(asked, 0, leads, 0.4) == 0
    # Nothing leads the answer it predicts the most for.
    assert reading.led(asked, 2, leads, 0.05) == 2
    # Where a search examines nothing the head is not asked.
    assert reading.led({**asked, "cards": [hidden()]}, 0, leads, 0.2) == 0


def evaluate(leads=None, **more):
    """Two games against Monarch's pilot, the scripted network's first
    choice the first answer offered."""
    games = [
        {"index": index, "opponent": "monarch", "learner_player": index % 2, "seed": 700 + index}
        for index in range(2)
    ]
    with DuelPool(ROOT, threads=1, slots=2) as pool:
        return evaluate_games(
            pool,
            Scripted(first, leads),
            "cpu",
            learner="blackwing",
            assignments=games,
            limit=4096,
            **more,
        )


def test_an_evaluation_can_be_played_by_following_the_lead_head():
    # A head that always predicts a lead for the last answer offered.
    def last(offered):
        return [1.0 if answer == offered - 1 else 0.0 for answer in range(offered)]

    plain = evaluate()
    assert all(result["led"] == 0 for result in plain)

    results = evaluate(last, lead_margin=0.5)
    assert all(result["led"] > 0 for result in results)
    # Out of reach of any lead, it plays its first choices.
    results = evaluate(last, lead_margin=2.0)
    assert [result["led"] for result in results] == [0, 0]
    assert results == plain


def test_a_watched_duel_holds_what_the_lead_head_predicts_and_what_is_examined(
    environment, reading
):
    game = recorded(environment)
    examined = [shown for shown in decisions(environment, game) if reading.examined(shown)]
    log = [logged(reading, shown) for shown in examined]

    def rising(offered):
        return [0.01 * answer for answer in range(offered)]

    result, _ = watch({**game, "tried": json.dumps(log)}, first, rising)
    assert result["examined_otherwise"] == 0 and result["tried_unread"] == 0
    for entry, asked in zip(result["tried"], examined, strict=True):
        listed = reading.alternatives(asked)
        assert entry["predicted"] == pytest.approx([0.01 * group[0] for group in listed])
    # A log that leaves out a decision a search examines is told apart.
    result, _ = watch({**game, "tried": json.dumps(log[1:])}, first, rising)
    assert result["examined_otherwise"] == 1
    # A model without a lead head, or a duel without a log.
    result, _ = watch({**game, "tried": json.dumps(log)}, first)
    assert all(entry["predicted"] is None for entry in result["tried"])
    assert watch(game, first)[0]["examined_otherwise"] == 0


@pytest.mark.skipif(
    not ((BENCH / "policy-bench").exists() and (BENCH / "libygo_policies.so").exists()),
    reason="policy-bench is not built: cargo build --release in native/ygo-policies",
)
def test_the_alternatives_are_the_ones_policy_bench_tries(tmp_path):
    """A small search run, logged: at every answer it examined, the
    alternatives listed here from the decision are the ones its log names."""
    rows = tmp_path / "rows.jsonl"
    subprocess.run(
        [
            str(BENCH / "policy-bench"),
            "search",
            *("--policies", "blackwing", "--opponents", "monarch", "--games", "2"),
            *("--seed", "4242", "--workers", "2"),
            *("--worlds", "2", "--confirm", "4", "--final", "8"),
            *("--log", "full", "--record", "true", "--output", str(rows)),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        timeout=300,
    )
    games = recorded_games(rows, "search")

    assert len(games) == 2 and all(game.examined for game in games)
    outright = 0
    for index, game in enumerate(games):
        result, _ = watch(game.replay(index), first)
        assert [entry["answer"] for entry in result["tried"]] == list(game.examined)
        assert result["tried_unread"] == 0
        # The decisions the search examined are the ones told from the
        # decision alone, and with the full log every alternative has a lead,
        # none of them above 0 where the pilot's answer won every world.
        assert result["examined_otherwise"] == 0
        assert all(lead is not None for entry in result["tried"] for lead in entry["leads"])
        won = [entry for entry in result["tried"] if entry["outright"]]
        assert all(max(entry["leads"]) == 0.0 for entry in won)
        outright += len(won)
        # And every one of them is a target of the decision it belongs to.
        (trajectory,) = replay([game.replay(index)])
        targeted = [step["lead_targets"] for step in trajectory["steps"] if "lead_targets" in step]
        assert [len(targets) for targets in targeted] == [
            len(entry["leads"]) for entry in result["tried"]
        ]
    assert outright
