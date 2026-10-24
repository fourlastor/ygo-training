"""A checkpoint's seat in a room of EDOPro, against a scripted room."""

from __future__ import annotations

import os
import signal
import socket
import struct
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from ygo_training.edopro import FirstChoice, sit
from ygo_training.environment import Environment, library_path
from ygo_training.model import PolicyValueNet

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

# EDOPro's packets (`ygo-policies-edopro`'s protocol): the client's, then the
# server's, and the engine's messages the duel below is made of.
RESPONSE, UPDATE_DECK, HAND_RESULT, TP_RESULT = 0x01, 0x02, 0x03, 0x04
PLAYER_INFO, JOIN_GAME, HS_READY = 0x10, 0x12, 0x22
GAME_MSG, SELECT_HAND, SELECT_TP, TYPE_CHANGE, DUEL_START = 0x01, 0x03, 0x04, 0x13, 0x15
MSG_WAITING, MSG_START, MSG_WIN, MSG_SELECT_IDLECMD, MSG_NEW_TURN, MSG_NEW_PHASE = (
    3,
    4,
    5,
    11,
    40,
    41,
)
MAIN_PHASE = 0x04
# The engine's answers to an idle command that offers no card.
TO_BATTLE, TO_END = 6, 7


@pytest.fixture(scope="module")
def environment():
    with Environment(ROOT) as opened:
        yield opened


class Room:
    """A scripted room: the lobby, then a duel in which the client, seated
    second, is asked one command with two answers and wins whatever it says."""

    def __init__(self) -> None:
        self.listener = socket.create_server(("127.0.0.1", 0))
        self.port = self.listener.getsockname()[1]
        self.responses: list[bytes] = []
        self.left_early = False
        self.thread = threading.Thread(target=self._serve, daemon=True)
        self.thread.start()

    def _read(self, connection: socket.socket) -> tuple[int, bytes] | None:
        def exactly(count: int) -> bytes | None:
            data = b""
            while len(data) < count:
                chunk = connection.recv(count - len(data))
                if not chunk:
                    return None
                data += chunk
            return data

        head = exactly(2)
        if head is None:
            return None
        body = exactly(struct.unpack("<H", head)[0])
        return None if body is None else (body[0], body[1:])

    def _write(self, connection: socket.socket, opcode: int, payload: bytes = b"") -> None:
        connection.sendall(struct.pack("<HB", len(payload) + 1, opcode) + payload)

    def _serve(self) -> None:
        connection, _ = self.listener.accept()
        with connection:
            connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

            def expect(opcode: int) -> bytes:
                packet = self._read(connection)
                assert packet is not None and packet[0] == opcode, packet
                return packet[1]

            def game(message: bytes) -> None:
                self._write(connection, GAME_MSG, message)

            expect(PLAYER_INFO)
            expect(JOIN_GAME)
            self._write(connection, JOIN_GAME, bytes(20))
            expect(UPDATE_DECK)
            self._write(connection, TYPE_CHANGE, b"\x01")
            expect(HS_READY)
            self._write(connection, DUEL_START)
            self._write(connection, SELECT_HAND)
            expect(HAND_RESULT)
            self._write(connection, SELECT_TP)
            expect(TP_RESULT)
            # The client is seat 1: 8000 Life Points each, 40 cards a Deck.
            game(struct.pack("<BBIIHHHH", MSG_START, 1, 8000, 8000, 40, 0, 40, 0))
            game(struct.pack("<BB", MSG_NEW_TURN, 0))
            game(struct.pack("<BH", MSG_NEW_PHASE, MAIN_PHASE))
            game(bytes([MSG_WAITING]))
            game(struct.pack("<BB", MSG_NEW_TURN, 1))
            game(struct.pack("<BH", MSG_NEW_PHASE, MAIN_PHASE))
            # No card to use: the Battle Phase or the end of the turn.
            game(struct.pack("<BB6IBBB", MSG_SELECT_IDLECMD, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0))
            answered = self._read(connection)
            if answered is None:
                self.left_early = True
                return
            assert answered[0] == RESPONSE
            self.responses.append(answered[1])
            game(struct.pack("<BBB", MSG_WIN, 1, 0))
            # The client hangs up after the duel.
            assert self._read(connection) is None

    def settings(self) -> dict[str, Any]:
        return {"port": self.port, "wait": 5, "seed": 1}

    def close(self) -> None:
        self.thread.join(timeout=10)
        self.listener.close()
        assert not self.thread.is_alive()


def named(environment: Environment, view: dict[str, Any]) -> list[str]:
    names = {number: name for name, number in environment.vocabulary["choice"].items()}
    return [names[int(choice["kind"])] for choice in view["choices"]]


def test_the_seat_asks_the_caller_what_an_episode_asks_it(environment):
    room = Room()
    views, notes = [], []

    def answer(view: dict[str, Any]) -> int:
        views.append(view)
        return named(environment, view).index("EndTurn")

    done = sit(environment, "blackwing", answer, room=room.settings(), told=notes.append)
    room.close()

    assert done == {
        "played": [{"result": "won", "seat": 1, "reason": 0, "decisions": 1, "log": None}]
    }
    # One decision, a view as an episode gives one, and its answer the engine's.
    assert len(views) == 1
    assert sorted(named(environment, views[0])) == ["EndTurn", "EnterBattle"]
    assert (views[0]["player"], views[0]["turn"], views[0]["step"]) == (1, 2, 0)
    assert views[0]["history"] == [] and views[0]["acting"] is None
    assert views[0]["label"] is not None
    assert room.responses == [struct.pack("<i", TO_END)]
    assert notes[0].startswith("duel 1 of 1") and notes[-1] == "duel 1: won"
    assert "duel started" in notes


def test_a_models_first_choice_is_played(environment):
    room = Room()
    model = PolicyValueNet(embedding_dim=8, hidden_dim=16).eval()
    said = []
    answer = FirstChoice(
        model,
        environment,
        said=lambda view, chosen, probability: said.append((chosen, probability)),
    )

    done = sit(environment, "blackwing", answer, room=room.settings(), told=lambda note: None)
    room.close()

    assert done["played"][0]["decisions"] == 1
    assert (answer.decisions, answer.labelled) == (1, 1)
    ((chosen, probability),) = said
    assert chosen in (0, 1) and 0.5 <= probability <= 1.0
    assert room.responses in ([struct.pack("<i", TO_BATTLE)], [struct.pack("<i", TO_END)])


def test_what_the_caller_raises_ends_the_session(environment):
    room = Room()

    def answer(view: dict[str, Any]) -> int:
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        sit(environment, "blackwing", answer, room=room.settings(), told=lambda note: None)
    room.close()

    assert room.left_early and room.responses == []


def test_an_answer_that_is_none_of_the_choices_ends_the_session(environment):
    room = Room()

    with pytest.raises(Exception, match="choice 5 of a decision that offers 2"):
        sit(environment, "blackwing", lambda view: 5, room=room.settings(), told=lambda note: None)
    room.close()

    assert room.left_early


def test_a_room_that_is_not_there_is_said(environment):
    listener = socket.create_server(("127.0.0.1", 0))
    port = listener.getsockname()[1]
    listener.close()
    notes = []

    with pytest.raises(Exception, match="cannot reach EDOPro"):
        sit(
            environment,
            "blackwing",
            lambda view: 0,
            room={"port": port, "wait": 1},
            told=notes.append,
        )

    assert notes and notes[0].startswith("waiting for EDOPro")


def test_an_interruption_from_the_keyboard_ends_the_wait(environment):
    listener = socket.create_server(("127.0.0.1", 0))
    port = listener.getsockname()[1]
    listener.close()
    # Nobody answers there: the seat would wait a minute for a room.
    interrupt = threading.Timer(0.6, lambda: os.kill(os.getpid(), signal.SIGINT))
    handler = signal.getsignal(signal.SIGINT)
    started = time.monotonic()
    interrupt.start()

    with pytest.raises(KeyboardInterrupt):
        sit(
            environment,
            "blackwing",
            lambda view: 0,
            room={"port": port, "wait": 60},
            told=lambda note: None,
        )

    interrupt.join()
    assert time.monotonic() - started < 5
    # The handler that was there is there again.
    assert signal.getsignal(signal.SIGINT) is handler


def test_a_table_is_a_room_or_a_trace(environment, tmp_path):
    with pytest.raises(ValueError, match="one of the two"):
        sit(environment, "blackwing", lambda view: 0)
    with pytest.raises(Exception, match="no deck"):
        sit(environment, "nothing", lambda view: 0, room={"port": 1})
    with pytest.raises(Exception, match="missing.trace"):
        sit(environment, "blackwing", lambda view: 0, trace=tmp_path / "missing.trace")
