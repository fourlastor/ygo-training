"""A duel EDOPro kept, read again: the replay's bytes, and each player's moves
beside the pilot's and a checkpoint's."""

from __future__ import annotations

import json
import lzma
import struct
from argparse import Namespace
from pathlib import Path
from typing import Any

import pytest

from tests.views import ACTIVATE, PASS, SPELL, SPELL_TRAP_ZONE, choice, named, view
from ygo_training.edopro_review import (
    _fidelity,
    alike,
    beside,
    checkpoint_asked,
    decisions,
    matched,
    read_replay,
    reread,
    review,
)
from ygo_training.environment import Environment, library_path
from ygo_training.model import PolicyValueNet

ROOT = Path(__file__).resolve().parents[1]
PLAYERS = ["Alice", "[AI] blackwing"]
SEED = [11, 22, 33, 44]


def _built() -> bool:
    try:
        library_path(ROOT)
    except FileNotFoundError:
        return False
    return True


needs_the_environment = pytest.mark.skipif(
    not _built(), reason="the environment is not built: cargo build --release"
)


def kept(**changed: Any) -> dict[str, Any]:
    """A small duel as a replay keeps one."""
    return {
        "seed": SEED,
        "flags": 0xD0700,
        "start_lp": 8000,
        "start_hand": 5,
        "draw_count": 1,
        "decks": [{"main": [1, 2, 3], "extra": [9]}, {"main": [4, 5], "extra": []}],
        "responses": ["ffffffff", "04000100", "000801", "0300000004"],
        **changed,
    }


def as_yrp(
    duel: dict[str, Any], *, packed: bool = False, flags: int = 0x330, time: int = 1791651189
) -> bytes:
    """The bytes of a `.yrp` of EDOPro: its header, the players, how the duel
    began, both players' cards and every answer."""
    name = lambda text: text.encode("utf-16le").ljust(40, b"\0")
    body = b"".join(struct.pack("<I", 1) + name(player) for player in PLAYERS)
    body += struct.pack(
        "<3IQ", duel["start_lp"], duel["start_hand"], duel["draw_count"], duel["flags"]
    )
    for deck in duel["decks"]:
        for pile in (deck["main"], deck["extra"]):
            body += struct.pack(f"<I{len(pile)}I", len(pile), *pile)
    body += struct.pack("<I", 0)
    for answer in duel["responses"]:
        body += bytes([len(answer) // 2]) + bytes.fromhex(answer)
    properties, size = bytes(8), 0
    if packed:
        # EDOPro writes LZMA's properties and the size in the header.
        alone = lzma.compress(body, format=lzma.FORMAT_ALONE)
        properties, size, body = alone[:5] + bytes(3), len(body), alone[13:]
        flags |= 0x1
    header = b"yrp1" + struct.pack("<5I", 0x000B0029, flags, time, size, 0) + properties
    if flags & 0x200:
        header += struct.pack("<Q4Q", 1, *duel["seed"])
    return header + body


def as_yrpx(yrp: bytes) -> bytes:
    """A replay as EDOPro saves one under its time: what the duel showed,
    and at the end the duel itself."""
    name = lambda text: text.encode("utf-16le").ljust(40, b"\0")
    body = b"".join(struct.pack("<I", 1) + name(player) for player in PLAYERS) + struct.pack(
        "<Q", 0xD0700
    )
    for packet, data in ((4, b"\x00" * 17), (40, b"\x00"), (231, yrp)):
        body += struct.pack("<BI", packet, len(data)) + data
    alone = lzma.compress(body, format=lzma.FORMAT_ALONE)
    header = b"yrpX" + struct.pack("<5I", 0x000B0029, 0x331, 1791651189, len(body), 0)
    return header + alone[:5] + bytes(3) + struct.pack("<Q4Q", 1, *SEED) + alone[13:]


def test_a_replay_is_read_plain_packed_or_inside_a_saved_one():
    duel = kept()
    read = {"time": 1791651189, "players": PLAYERS, **duel}

    assert read_replay(as_yrp(duel)) == read
    assert read_replay(as_yrp(duel, packed=True)) == read
    assert read_replay(as_yrpx(as_yrp(duel, packed=True))) == read


def test_what_is_no_duel_of_two_players_is_said():
    with pytest.raises(ValueError, match="not a replay of EDOPro"):
        read_replay(b"something else entirely, long enough for a header")
    with pytest.raises(ValueError, match="teams or a puzzle"):
        read_replay(as_yrp(kept(), flags=0x330 | 0x2))
    with pytest.raises(ValueError, match="kept its seed otherwise"):
        read_replay(as_yrp(kept(), flags=0x130))
    with pytest.raises(ValueError, match="cut short"):
        read_replay(as_yrp(kept())[:-2])


def test_copies_of_a_card_that_lie_alike_are_one_answer():
    card = lambda code, controller: {"code": code, "controller": controller, "location": 2}
    choice = lambda *cards: {"kind": 3, "description": 0, "card": None, "members": list(cards)}
    view = {
        "choices": [choice(card(7, 0)), choice(card(7, 0)), choice(card(7, 1)), choice(card(8, 0))]
    }

    assert alike(view, 0, 1) and alike(view, 2, 2)
    # The other player's copy is another card, and so is another name.
    assert not alike(view, 0, 2) and not alike(view, 0, 3)
    assert not alike(view, 0, None)


def test_a_rooms_decisions_are_set_beside_the_trainers_field_by_field():
    """A room that plays other card scripts calls an answer otherwise: the
    decisions are the same ones, and what differs in them is told."""

    def window(description: int, hint: int = 0, **fields: Any) -> dict[str, Any]:
        activation = choice(
            ACTIVATE, named(code=SPELL, location=SPELL_TRAP_ZONE), description=description
        )
        question = {**view()["question"], "hint": hint}
        return view(choices=[activation, choice(PASS)], question=question, **fields)

    described = SPELL << 20
    here = [view(), window(described, 23), window(described, turn=3)]
    # What the two count their own way is not a difference; nor is a field
    # of that name further in.
    room = [
        view(answer=7, label=1, pilot=1, chosen=0),
        window(0, 16, answer=9),
        window(0, turn=3),
    ]

    def describe(shown: dict[str, Any], index: int) -> str:
        return "activate" if shown["choices"][index]["kind"] == ACTIVATE else "pass"

    found = beside(describe, room, here)
    assert found == {
        "room": 3,
        "here": 3,
        "same": 1,
        "fields": {"description": 2, "hint": 1},
        "answers": {"activate": 2},
    }

    # A checkpoint asked both ways: its first choice where the room named
    # no effect, and where the trainer names one.
    def asked(shown: dict[str, Any]) -> dict[str, Any]:
        if shown["turn"] == 3:
            return {"model": 1, "probabilities": [0.2, 0.8]}
        sure = shown["choices"][0]["description"] != 0
        return {"model": 0 if sure else 1, "probabilities": [0.52, 0.48] if sure else [0.48, 0.52]}

    found = beside(describe, room, here, asked)
    assert (found["first_choice_changed"], found["probabilities_moved"]) == (1, 1)
    said = _fidelity("[AI] bot", Path("logs/duel.trace"), found)
    assert said[0].startswith("[AI] bot's 3 decisions as the room showed them (duel.trace)")
    assert "  the same in every field: 1" in said
    assert "  what differs: description (2), hint (1)" in said
    assert "  at the answers: activate (2)" in said
    assert any("first choice is another at 1 of them" in line for line in said)
    assert "EDOPro's card scripts" in said[-1]

    alike_ = beside(describe, here, here, asked)
    assert (alike_["same"], alike_["fields"], alike_["first_choice_changed"]) == (3, {}, 0)
    assert "EDOPro's card scripts" not in _fidelity("bot", Path("duel.trace"), alike_)[-1]
    # A trace of another duel has other decisions.
    other = beside(describe, room[:2], here)
    assert other == {"room": 2, "here": 3}
    assert "it is not the trace of this duel" in _fidelity("bot", Path("duel.trace"), other)[0]


@pytest.fixture(scope="module")
def environment():
    with Environment(ROOT) as opened:
        yield opened


def played(environment: Environment, seed: int = 7) -> dict[str, Any]:
    """A duel two pilots play, as the policies record it."""
    setup = {
        "seed": seed,
        "decks": ["blackwing", "monarch"],
        "seats": [{"control": "pilot"}, {"control": "pilot"}],
        "keep_record": True,
    }
    with environment.start(setup) as episode:
        while True:
            over = episode.advance()["over"]
            if over is not None:
                return over


def as_edopro_keeps_it(record: dict[str, Any]) -> dict[str, Any]:
    """A record as a replay would hold the duel: the cards in the order the
    engine was given them, and the answers without who gave them."""
    return {
        **{key: record[key] for key in ("seed", "flags", "start_lp", "start_hand", "draw_count")},
        "decks": [
            {pile: deck[pile][::-1] for pile in ("main", "extra")} for deck in record["decks"]
        ],
        "responses": [answer for _, answer in record["responses"]],
    }


@pytest.mark.integration
@needs_the_environment
def test_a_kept_duel_tells_who_answered_and_is_set_beside_the_pilots(environment):
    over = played(environment)
    duel = read_replay(as_yrp(as_edopro_keeps_it(over["record"]), packed=True))

    again = reread(environment, duel)
    assert again["record"] == over["record"]
    assert (again["winner"], again["turns"]) == (over["winner"], over["turns"])

    model = PolicyValueNet(embedding_dim=8, hidden_dim=16).eval()
    rows = decisions(
        environment,
        again["record"],
        ["blackwing", "monarch"],
        asked=checkpoint_asked(model, environment, [True, False]),
    )
    first = [row for row in rows if row["player"] == 0]
    second = [row for row in rows if row["player"] == 1]
    assert len(first) > 10 and len(second) > 10
    # Pilots that are asked again, seeded as they were, give their own answers.
    assert all(row["with_pilot"] for row in rows)
    # The checkpoint is asked where its deck is played, and nowhere else.
    assert all("model" not in row for row in second)
    for row in first:
        assert 0 <= row["model"] < len(row["said"])
        assert sum(row["probabilities"]) == pytest.approx(1.0)
        assert 0.0 <= row["chances"] <= 1.0
    assert {row["kind"] for row in rows} >= {"Idle", "Chain"}


@pytest.mark.integration
@needs_the_environment
def test_a_duel_given_up_is_read_as_far_as_it_went(environment):
    over = played(environment)
    duel = as_edopro_keeps_it(over["record"])
    duel["responses"] = duel["responses"][:60]

    again = reread(environment, duel)
    assert again["winner"] is None
    assert again["record"]["responses"] == over["record"]["responses"][:60]
    rows = decisions(environment, again["record"], ["blackwing", "monarch"], stop=60)
    assert rows and all(row["with_pilot"] for row in rows)


@pytest.mark.integration
@needs_the_environment
def test_a_deck_is_told_from_its_cards(environment):
    blackwing = environment.decks["blackwing"]["main"]
    assert matched(environment, blackwing) == ("blackwing", 1.0)
    # A player's own list of the deck is that deck still.
    theirs = [*blackwing[:-4], 1, 2, 3, 4]
    name, shared = matched(environment, theirs)
    assert name == "blackwing" and shared == pytest.approx(1 - 4 / len(blackwing))
    assert matched(environment, list(range(1, 41)))[0] is None


@pytest.mark.integration
@needs_the_environment
def test_the_command_writes_the_review(environment, tmp_path, capsys):
    over = played(environment)
    replay = tmp_path / "_LastReplay.yrp"
    replay.write_bytes(as_yrp(as_edopro_keeps_it(over["record"])))
    args = Namespace(
        replay=replay,
        root=ROOT,
        device="cpu",
        checkpoint=None,
        learner="blackwing",
        player=None,
        decks=None,
        trace=None,
        output=tmp_path / "reviews",
    )

    review(args)

    printed = capsys.readouterr().out
    winner = PLAYERS[over["winner"]]
    assert (
        f"_LastReplay.yrp: Alice went first against [AI] blackwing, {over['turns']} turns, won by {winner}"
        in printed
    )
    assert "Alice plays blackwing: 100% of the Main Deck is that list's" in printed
    assert "[AI] blackwing plays monarch: 100% of the Main Deck is that list's" in printed
    # The pilots' own duel: the player who is no bot is the one reviewed.
    assert "Alice answered as the pilot and the checkpoint would" in printed
    stem = tmp_path / "reviews" / "duel-1791651189"
    assert json.loads(stem.with_suffix(".record.json").read_text())["players"] == PLAYERS
    rows = json.loads(stem.with_suffix(".decisions.json").read_text())
    assert rows and {row["player"] for row in rows} == {0, 1}
    assert (
        stem.with_suffix(".review.txt").read_text().startswith("_LastReplay.yrp: Alice went first")
    )


def test_a_file_that_is_not_there_is_said(tmp_path):
    replay = tmp_path / "_LastReplay.yrp"
    args = Namespace(replay=replay, root=ROOT, checkpoint=None, trace=tmp_path / "DUEL.trace")

    with pytest.raises(SystemExit, match="No replay at .*_LastReplay.yrp: EDOPro keeps its duels"):
        review(args)
    replay.write_bytes(b"")
    with pytest.raises(SystemExit, match="No trace at .*DUEL.trace$"):
        review(args)


@pytest.mark.integration
@needs_the_environment
def test_a_duel_of_another_engine_is_said_to_be_another(environment, tmp_path):
    over = played(environment)
    duel = as_edopro_keeps_it(over["record"])
    duel["seed"] = [seed + 1 for seed in duel["seed"]]
    replay = tmp_path / "other.yrp"
    replay.write_bytes(as_yrp(duel))
    args = Namespace(
        replay=replay,
        root=ROOT,
        device="cpu",
        checkpoint=None,
        learner="blackwing",
        player=None,
        decks=None,
        trace=None,
        output=tmp_path,
    )

    with pytest.raises(SystemExit, match="EDOPro's may be others than this checkout's"):
        review(args)
