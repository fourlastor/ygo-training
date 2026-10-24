"""The rows of a search run as recorded games, and their order."""

from __future__ import annotations

import json

import pytest

from ygo_training.records import fingerprint, recorded_games, shared_deals, update_games


def row(opponent, seed, seat, *, deviations=(), examined=(), winners=(0, 1), record=True):
    """One row as ``policy-bench search --record true`` writes it."""

    def game(name, winner, **more):
        played = {"winner": winner, "digest": f"{name}-{opponent}-{seed}", "decisions": 9, **more}
        if record:
            played["record"] = {"seed": [seed, 0, 0, 0], "responses": [[0, "00"]], "game": name}
        return played

    return {
        "a": "blackwing",
        "b": opponent,
        "seed": seed,
        "seat": seat,
        "baseline": game("baseline", winners[0]),
        "search": game(
            "search",
            winners[1],
            deviations=[{"decision": place, "kind": "Idle"} for place in deviations],
            examined=[{"decision": place, "kind": "Idle"} for place in examined],
        ),
    }


def write(tmp_path, rows):
    path = tmp_path / "rows.jsonl"
    path.write_text("".join(json.dumps(item) + "\n" for item in rows), encoding="utf8")
    return path


def run(opponents=("monarch", "heroes", "burn"), games=4):
    """A run's rows in the order they might finish: not the order of play."""
    rows = [
        row(opponent, 100 * (index + 1) + game, game % 2)
        for index, opponent in enumerate(opponents)
        for game in range(games)
    ]
    return rows[::-1]


def test_games_are_ordered_by_their_place_among_their_opponents(tmp_path):
    games = recorded_games(write(tmp_path, run()))

    assert [(game.opponent, game.seed) for game in games[:6]] == [
        ("burn", 300),
        ("heroes", 200),
        ("monarch", 100),
        ("burn", 301),
        ("heroes", 201),
        ("monarch", 101),
    ]
    # Whatever order the rows were written in.
    assert recorded_games(write(tmp_path, run()[::-1])) == games


def test_a_smaller_set_is_an_equal_share_and_part_of_a_larger(tmp_path):
    path = write(tmp_path, run())
    six, all_of_them = recorded_games(path, games=6), recorded_games(path)

    assert six == all_of_them[:6]
    assert sorted(game.opponent for game in six) == ["burn"] * 2 + ["heroes"] * 2 + ["monarch"] * 2
    assert sorted(game.learner_player for game in six) == [0, 0, 0, 1, 1, 1]
    assert fingerprint(six) != fingerprint(all_of_them) and len(fingerprint(six)) == 16
    with pytest.raises(ValueError, match="holds 12 games; 13 were asked for"):
        recorded_games(path, games=13)


def test_a_row_gives_both_games_of_its_deal(tmp_path):
    path = write(tmp_path, [row("monarch", 7, 1, deviations=(4, 12), winners=(0, 1))])
    (searched,) = recorded_games(path, "search")
    (piloted,) = recorded_games(path, "baseline")

    assert (searched.learner, searched.opponent, searched.learner_player) == (
        "blackwing",
        "monarch",
        1,
    )
    assert searched.deviations == (4, 12) and piloted.deviations == ()
    # The search wins from the second seat what the pilot loses.
    assert (searched.outcome, piloted.outcome) == ("win", "loss")
    assert json.loads(searched.record)["game"] == "search"
    assert json.loads(piloted.record)["game"] == "baseline"
    assert searched.deal(3) == {"index": 3, "opponent": "monarch", "learner_player": 1, "seed": 7}
    replay = searched.replay(3)
    assert replay["digest"] == "search-monarch-7" and replay["deviations"] == [4, 12]
    assert replay["record"] == searched.record and replay["examined"] is None
    assert fingerprint([searched]) != fingerprint([piloted])


def test_a_game_no_one_won_is_a_draw(tmp_path):
    path = write(tmp_path, [row("monarch", 7, 0, winners=(None, 2))])
    assert recorded_games(path, "baseline")[0].outcome == "draw"
    assert recorded_games(path, "search")[0].outcome == "draw"


def test_what_the_search_examined_is_known_only_from_a_logged_run(tmp_path):
    plain = recorded_games(write(tmp_path, [row("monarch", 7, 0), row("monarch", 8, 1)]))
    assert [game.examined for game in plain] == [None, None]

    logged = recorded_games(
        write(tmp_path, [row("monarch", 7, 0, examined=(3, 9)), row("monarch", 8, 1)])
    )
    # In a logged run a game without any examined none.
    assert [game.examined for game in logged] == [(3, 9), ()]


def tried(decision, labels, played, *, chosen=0, skipped=False):
    """One answer as a logged search holds it: the pilot's answer first, and
    how each alternative did in every world it was played in."""
    return {
        "decision": decision,
        "kind": "Idle",
        "labels": labels,
        "played": played,
        "chosen": chosen,
        "skipped": skipped,
    }


def test_a_logged_run_gives_each_alternatives_lead_in_the_first_worlds(tmp_path):
    summon = {"kind": "NormalSummon", "code": 7}
    end = {"kind": "EndTurn", "code": None}
    defend = {"kind": {"Position": {"face_up": True, "attack": False}}, "code": None}
    rows = [row("monarch", 100, 0), row("monarch", 101, 1)]
    rows[0]["search"]["examined"] = [
        # The first alternative went on to a second stage: only the first
        # four worlds, which every alternative is played in, are read.
        tried(4, [summon, end, defend], ["++--" + "----", "+++-" + "++++", "+---"], chosen=1),
        # The pilot's answer wins every first world: nothing else is tried.
        tried(9, [summon, end], ["++++", ""], skipped=True),
        # A world that could not be played out counts for neither answer,
        # and a draw as half a win.
        tried(12, [summon, end], ["+x0-", "-+++"]),
    ]

    logged, other = recorded_games(write(tmp_path, rows))
    first, outright, failed = json.loads(logged.tried)

    assert first == {
        "answer": 4,
        "labels": [["NormalSummon", 7], ["EndTurn", 0], ["PositionFaceUpDefense", 0]],
        "leads": [0.0, 0.25, -0.25],
        # What the pilot's answer is worth by those four worlds: two won, two lost.
        "rate": 0.0,
        "worlds": 4,
        "taken": 1,
        "outright": False,
    }
    assert (outright["leads"], outright["outright"]) == ([0.0, None], True)
    assert (outright["rate"], outright["worlds"]) == (1.0, 4)
    assert failed["leads"] == [0.0, pytest.approx((-1 + 0.5 + 1) / 3)]
    # The world that could not be played out is none of the pilot's answer's.
    assert (failed["rate"], failed["worlds"]) == (0.0, 3)
    # A game of a logged run in which nothing was examined tried nothing.
    assert (other.examined, other.tried) == ((), "[]")
    assert logged.replay(0)["tried"] == logged.tried
    # The pilot's own game has no log, nor has a run that kept none.
    assert recorded_games(write(tmp_path, rows), "baseline")[0].tried is None
    plain = recorded_games(write(tmp_path, [row("monarch", 100, 0, examined=(3,))]))
    assert plain[0].examined == (3,) and plain[0].tried is None


def test_rows_that_cannot_be_imitated_are_refused(tmp_path):
    with pytest.raises(ValueError, match="--record true"):
        recorded_games(write(tmp_path, [row("monarch", 7, 0, record=False)]))
    with pytest.raises(ValueError, match="not the learner's deck 'monarch'"):
        recorded_games(write(tmp_path, [row("heroes", 7, 0)]), learner="monarch")
    with pytest.raises(ValueError, match="repeats the game"):
        recorded_games(write(tmp_path, [row("monarch", 7, 0), row("monarch", 7, 0)]))
    with pytest.raises(ValueError, match="Unknown game"):
        recorded_games(write(tmp_path, [row("monarch", 7, 0)]), "pilot")
    with pytest.raises(ValueError, match="holds no recorded game"):
        recorded_games(write(tmp_path, []))
    mixed = [row("monarch", 7, 0), {**row("heroes", 8, 0), "a": "monarch"}]
    with pytest.raises(ValueError, match="Row 2 records 'monarch'"):
        recorded_games(write(tmp_path, mixed))


def test_two_runs_share_the_deals_their_seeds_have_in_common(tmp_path):
    def games(first_seed):
        rows = [row("monarch", first_seed + game, game % 2) for game in range(4)]
        return recorded_games(write(tmp_path, [*rows, row("heroes", first_seed, 0)]))

    assert shared_deals(games(100), games(200)) == 0
    # A seed two further on deals two of a matchup's four games again,
    # whichever seat the learner then has.
    assert shared_deals(games(100), games(102)) == 2
    assert shared_deals(games(100), games(100)) == 5


def test_updates_go_through_the_games_pass_after_pass():
    count, duels = 10, 4
    updates = [update_games(count, duels, update, seed=5) for update in range(5)]
    stream = [game for update in updates for game in update]

    assert all(len(update) == duels for update in updates)
    # Every pass holds each game once, in an order of its own.
    assert sorted(stream[:10]) == sorted(stream[10:20]) == list(range(10))
    assert stream[:10] != stream[10:20]
    # A continued run goes on where it stood; another seed sees another order.
    assert update_games(count, duels, 3, seed=5) == updates[3]
    assert update_games(count, duels, 0, seed=6) != updates[0]
    with pytest.raises(ValueError):
        update_games(0, duels, 0, seed=5)
