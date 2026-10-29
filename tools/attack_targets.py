"""What a network attacks into: every attack the learner's seat declares in
its games against pilots, sorted by what the seat could see of the target
when it chose it, and how the attack came out; with ``--bar`` the same deals
again, the network barred from attacking into a monster it could see would
win the battle (a row that offers nothing else is left whole).

    uv run python tools/attack_targets.py play CHECKPOINT --opponents trained \\
        --games 64 --output analysis/attack-targets/fourteen-decks-trained.json
    uv run python tools/attack_targets.py report analysis/attack-targets/*.json

The learners are the fourteen decks of ``fourteen-decks`` unless others are
named; the opponents the pilots of the thirteen decks it trained against
(``trained``), of the pool's decks it never met (``unmet``), or those named.
Each learner deck meets the opponents on seeds of its own, the same in every
run: two networks, or a network with and without the bar, play the same
deals.

What the seat sees is its view: an opponent's face-down monster has no code
and no ATK or DEF, a face-up one the ATK and DEF the engine gives it now,
with what equips, fields and continuous effects make of them.  To the
attacker's ATK is added the boost its own text gives it when it attacks (Jain,
Lightsworn Paladin's 300), which comes only in the Damage Step.  A target is
then *stronger* when it is face-up in Attack Position with more ATK, or in
Defense Position with more DEF: the attacker is destroyed, or its player takes
the difference.  Such an attack is *planned* when the attacker's text gains
from the battle (its battle decided at the start of the Damage Step, an effect
when it is destroyed by battle or leaves the field, a Crystal Beast placed as
a Spell, a Gladiator Beast that tags out after a battle it survives), and
*armed* when the seat holds a trick of its deck that raises that attacker in
the battle (`TRICKS`); the bar is on the stronger ones alone.  How each attack
came out is read at the seat's next decision of the turn once the battle is
over, whether the attacker is still on the field and whether its player has
lost LP, so that a stronger target that did not win shows.  Where the engine
picks the target itself (a single one may be attacked), it is read from the
board, or from the decision that follows the attack; an attacker that may
attack directly is asked whether it does, and its answer says which it did.
The bar leaves such an attacker free to attack directly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sqlite3
import sys
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from card_use import THIRTEEN, Names
from ppo_head_report import difference
from search_leads_report import percent, points

from ygo_training.environment import CARDS_DATABASE, Environment, pool
from ygo_training.model import MONSTER_ZONE, SPELL_TRAP_ZONE
from ygo_training.orchestration import evaluation_assignments
from ygo_training.pool import DuelPool, Round, evaluate_games
from ygo_training.schema import ACTION_SCALARS, CONTEXT_SLOTS, SCALARS

FOURTEEN = f"{THIRTEEN},cyber"
DIRECT, FACE_DOWN, WEAKER, EVEN, PLANNED, ARMED, STRONGER, UNKNOWN = (
    "direct",
    "face-down",
    "weaker",
    "even",
    "planned",
    "armed",
    "stronger",
    "unknown",
)
KINDS = (DIRECT, FACE_DOWN, WEAKER, EVEN, PLANNED, ARMED, STRONGER, UNKNOWN)
# How an attack came out, at the seat's next decision of the turn once its
# battle is over: the attacker left the field; it stayed and its player lost
# LP; neither; or the seat had no decision of that turn left.
LOST, HURT, FINE, UNSEEN = "lost", "hurt", "fine", "unseen"
OUTCOMES = (LOST, HURT, FINE, UNSEEN)
FACE_UP_ATTACK, FACE_UP_DEFENSE = 0x1, 0x4
# Players are relative to the seat.
OWN, OPPONENT = 0, 1
# A battle's cards, in a decision of its; and the card a selection is
# for, the attacker where the target of its attack is asked.
ATTACKER = CONTEXT_SLOTS.index("attacker")
TARGET = CONTEXT_SLOTS.index("attack_target")
ACTING = CONTEXT_SLOTS.index("acting")
LIFE_POINTS = SCALARS.index("own_life_points")
TURN = SCALARS.index("turn")
# An Attack's value is 1 where its monster may attack directly: with
# monsters on the other side, the seat is then asked whether it does.
MAY_ATTACK_DIRECTLY = ACTION_SCALARS.index("value_low")
# ygo-model-input's scales: ATK and DEF over 4,000, LP over 8,000, the turn
# over 100.
COMBAT_SCALE, LIFE_SCALE, TURN_SCALE = 4000, 8000, 100
# A learner deck meets its opponents on seeds of its own, a group of
# opponents on seeds of the group's.
DECK_SEEDS = 10_000_000
GROUP_SEEDS = {"trained": 0, "unmet": 5_000_000}
# What an attacker's text makes of a battle it would lose by the numbers: a
# boost of its own when it attacks, or a gain from the battle itself; and a
# Gladiator Beast's tag out, after a battle it survives.
OWN_BOOST = re.compile(
    r"(?:if|when) this card attacks[^.]*?gains? (\d+) ATK during the Damage Step", re.IGNORECASE
)
GAINS = (
    re.compile(r"At the start of the Damage Step, if this card battles", re.IGNORECASE),
    re.compile(r"(?:When|If) this card is destroyed by battle", re.IGNORECASE),
    re.compile(r"If this face-up card leaves the field", re.IGNORECASE),
    re.compile(
        r"destroyed in a Monster Zone, you can place it face-up in your Spell & Trap Zone",
        re.IGNORECASE,
    ),
)
TAG_OUT = re.compile(
    r"At the end of the Battle Phase, if this card (?:attacked|battled)[^.]*?: You can shuffle it into the Deck",
    re.IGNORECASE,
)
# OCGCore's LIGHT and Machine, and the Blackwings' set code.
LIGHT, MACHINE, BLACKWING = 0x10, 0x20, 0x33


def combat(value: float) -> int:
    return round(float(value) * COMBAT_SCALE)


def target_kind(attack: int, code: int, position: int, atk: int, defense: int) -> str:
    """What a monster is to an attacker of ``attack`` ATK, as the seat sees it."""
    if code == 0 or not position & (FACE_UP_ATTACK | FACE_UP_DEFENSE):
        return FACE_DOWN
    held = atk if position & FACE_UP_ATTACK else defense
    return WEAKER if held < attack else EVEN if held == attack else STRONGER


def monsters(
    arrays: dict[str, np.ndarray], row: int, controller: int
) -> list[tuple[int, int, int, int]]:
    """The monsters of a player in a row's view: code, position, ATK and DEF."""
    length = int(np.asarray(arrays["state_lengths"]).reshape(-1)[row])
    places = np.flatnonzero(
        (arrays["state_location"][row, :length] == MONSTER_ZONE)
        & (arrays["state_controller"][row, :length] == controller)
    )
    return [
        (
            int(arrays["state_card"][row, place]),
            int(arrays["state_position"][row, place]),
            combat(arrays["state_numeric"][row, place, 0]),
            combat(arrays["state_numeric"][row, place, 1]),
        )
        for place in places.tolist()
    ]


def answered(arrays: dict[str, np.ndarray], row: int, answer: int) -> tuple[int, int, int, int]:
    """The card of an answer: code, position, ATK and DEF."""
    numeric = arrays["action_numeric"][row, answer]
    return (
        int(arrays["action_card"][row, answer]),
        int(arrays["action_position"][row, answer]),
        combat(numeric[0]),
        combat(numeric[1]),
    )


def in_context(arrays: dict[str, np.ndarray], row: int, slot: int) -> tuple[int, int, int, int]:
    """The card of a slot of the decision's context: code, position, ATK and DEF."""
    numeric = arrays["context_numeric"][row, slot]
    return (
        int(arrays["context_card"][row, slot]),
        int(arrays["context_position"][row, slot]),
        combat(numeric[0]),
        combat(numeric[1]),
    )


@dataclass(frozen=True)
class Printed:
    """What a card's row of the database says of its battles."""

    attribute: int = 0
    race: int = 0
    setcodes: tuple[int, ...] = ()
    boost: int = 0
    gains: bool = False
    tags_out: bool = False


# The battle tricks of the fourteen decks' lists (a scan of 2026-10-29): held
# in the hand, or also set, each raises an attacker of its kind in the battle.
TRICKS: dict[str, tuple[bool, Callable[[Printed], bool]]] = {
    "Blackwing - Kalut the Moon Shadow": (False, lambda card: BLACKWING in card.setcodes),
    "Honest": (False, lambda card: card.attribute == LIGHT),
    "Limiter Removal": (True, lambda card: card.race == MACHINE),
}


class Battles:
    """What the cards' texts make of an attack the numbers say is lost."""

    def __init__(
        self,
        printed: dict[int, Printed],
        tricks: dict[int, tuple[bool, Callable[[Printed], bool]]],
        hand: int,
    ) -> None:
        self.printed = printed
        self.tricks = tricks
        self.hand = hand

    @classmethod
    def read(cls, database: Path, hand: int) -> Battles:
        connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
        printed, tricks = {}, {}
        for code, attribute, race, setcode, name, told in connection.execute(
            "select d.id, d.attribute, d.race, d.setcode, t.name, t.desc"
            " from datas d join texts t on t.id = d.id"
        ):
            told = " ".join(str(told).split())
            boost = OWN_BOOST.search(told)
            printed[int(code)] = Printed(
                attribute=int(attribute),
                race=int(race),
                setcodes=tuple(
                    part for shift in (0, 16, 32, 48) if (part := (int(setcode) >> shift) & 0xFFFF)
                ),
                boost=int(boost.group(1)) if boost else 0,
                gains=any(pattern.search(told) for pattern in GAINS),
                tags_out=bool(TAG_OUT.search(told)),
            )
            if name in TRICKS:
                tricks[int(code)] = TRICKS[name]
        return cls(printed, tricks, hand)

    def held(self, arrays: dict[str, np.ndarray], row: int) -> list[tuple[int, bool]]:
        """The seat's tricks: each one's code and whether it is in the hand."""
        length = int(np.asarray(arrays["state_lengths"]).reshape(-1)[row])
        cards = arrays["state_card"][row, :length]
        places = arrays["state_location"][row, :length]
        mine = arrays["state_controller"][row, :length] == OWN
        return [
            (int(code), int(place) == self.hand)
            for code, place, own in zip(cards.tolist(), places.tolist(), mine.tolist(), strict=True)
            if own and int(code) in self.tricks and int(place) in (self.hand, SPELL_TRAP_ZONE)
        ]

    def kind(
        self,
        attacker: int,
        attack: int,
        target: tuple[int, int, int, int],
        held: list[tuple[int, bool]],
    ) -> str:
        """What a target is to an attacker of ``attack`` ATK, by the numbers
        the seat sees and by the cards' texts."""
        card = self.printed.get(attacker, Printed())
        found = target_kind(attack + card.boost, *target)
        if found != STRONGER:
            return found
        if card.gains or (card.tags_out and target[1] & FACE_UP_DEFENSE):
            return PLANNED
        for code, in_hand in held:
            set_too, raises = self.tricks[code]
            if (in_hand or set_too) and raises(card):
                return ARMED
        return STRONGER


def placed_answers(arrays: dict[str, np.ndarray], row: int) -> np.ndarray:
    """The answers of a row that are cards where they lie."""
    return np.flatnonzero((arrays["action_kind"][row] > 0) & (arrays["action_location"][row] > 0))


class Questions:
    """The numbers of the questions and answers an attack goes through."""

    def __init__(self, duels: DuelPool) -> None:
        self.attack = _kind(duels.choice_names, "Attack")
        self.yes = _kind(duels.choice_names, "Yes")
        self.battle = _kind(duels.decision_names, "Battle")
        self.select = _kind(duels.decision_names, "SelectCards")
        self.yes_no = _kind(duels.decision_names, "YesNo")
        with Environment(ROOT) as environment:
            vocabulary = environment.vocabulary
        self.attack_target = int(vocabulary["hint"]["AttackTarget"])
        self.hand = int(vocabulary["location"]["Hand"])
        self.damage_step = {
            int(vocabulary["phase"][name]) for name in ("Damage", "DamageCalculation")
        }

    def aiming(self, arrays: dict[str, np.ndarray], row: int) -> bool:
        """A selection of the target of the seat's own attack."""
        return bool(
            arrays["decision_kind"].reshape(-1)[row] == self.select
            and arrays["decision_hint"].reshape(-1)[row] == self.attack_target
        )

    def asking_direct(self, arrays: dict[str, np.ndarray], row: int) -> bool:
        """Whether the seat's attacker attacks directly, asked as it attacks."""
        return bool(
            arrays["decision_kind"].reshape(-1)[row] == self.yes_no
            and arrays["context_mask"][row, ACTING]
        )

    def in_battle(self, arrays: dict[str, np.ndarray], row: int) -> bool:
        """A decision within an attack: its target asked for, a window of its
        battle, or the damage step."""
        return bool(
            self.aiming(arrays, row)
            or arrays["context_mask"][row, ATTACKER]
            or arrays["context_mask"][row, TARGET]
            or int(arrays["phase"].reshape(-1)[row]) in self.damage_step
        )


def _kind(names: dict[int, str], name: str) -> int:
    return next(kind for kind, told in names.items() if told == name)


@dataclass
class Watch:
    """An attack of the seat's, from its declaration until it is told."""

    attacker: int
    attack: int
    turn: int
    life_points: int
    copies: int
    held: list[tuple[int, bool]]
    kind: str | None = None
    # Whether its next decision may ask if it attacks directly, and whether
    # its target is still to be chosen among several.
    asked: bool = False
    aiming: bool = False


class Attacks:
    """The attacks of the learner's seat, game by game: at each Attack it
    takes, what its target was to it, and how the attack came out."""

    def __init__(self, duels: DuelPool, questions: Questions, battles: Battles) -> None:
        self.duels = duels
        self.questions = questions
        self.battles = battles
        # By the game's index: each kind, and each kind with its outcome.
        self.games: dict[int, Counter[str]] = defaultdict(Counter)
        # By the attacker's code, the kind and the outcome.
        self.cards: Counter[tuple[int, str, str]] = Counter()
        self._game: dict[int, int] = {}
        self._watching: dict[int, Watch] = {}

    def __call__(self, round_: Round, mine: np.ndarray, chosen: np.ndarray) -> None:
        arrays = round_.arrays
        questions = arrays["decision_kind"].reshape(-1)
        for row in mine.tolist():
            duel = int(round_.duel[row])
            if duel not in self._game:
                self._game[duel] = int(self.duels._about[duel]["index"])
            turn = round(float(arrays["scalars"][row, TURN]) * TURN_SCALE)
            answer = int(chosen[row])
            watch = self._watching.get(duel)
            if watch is not None:
                if watch.asked:
                    watch.asked = False
                    if self.questions.asking_direct(arrays, row):
                        if arrays["action_kind"][row, answer] == self.questions.yes:
                            watch.kind, watch.aiming = DIRECT, False
                        continue
                if watch.aiming:
                    watch.aiming = False
                    watch.kind = self._aimed(arrays, row, answer, watch)
                if turn != watch.turn:
                    self._tell(duel, watch, UNSEEN)
                elif not self.questions.in_battle(arrays, row):
                    self._tell(duel, watch, self._outcome(arrays, row, watch))
            if (
                questions[row] == self.questions.battle
                and arrays["action_kind"][row, answer] == self.questions.attack
            ):
                attacker = int(arrays["action_card"][row, answer])
                watch = Watch(
                    attacker=attacker,
                    attack=combat(arrays["action_numeric"][row, answer, 0]),
                    turn=turn,
                    life_points=round(float(arrays["scalars"][row, LIFE_POINTS]) * LIFE_SCALE),
                    copies=sum(card[0] == attacker for card in monsters(arrays, row, OWN)),
                    held=self.battles.held(arrays, row),
                )
                theirs = monsters(arrays, row, OPPONENT)
                watch.asked = bool(theirs) and bool(
                    arrays["action_scalars"][row, answer, MAY_ATTACK_DIRECTLY] > 0
                )
                watch.aiming = len(theirs) > 1
                if len(theirs) <= 1:
                    watch.kind = (
                        self.battles.kind(attacker, watch.attack, theirs[0], watch.held)
                        if theirs
                        else DIRECT
                    )
                self._watching[duel] = watch

    def _aimed(self, arrays: dict[str, np.ndarray], row: int, answer: int, watch: Watch) -> str:
        """The target of an attack among several, from the decision that follows it."""
        if self.questions.aiming(arrays, row):
            return self.battles.kind(
                watch.attacker, watch.attack, answered(arrays, row, answer), watch.held
            )
        if (
            arrays["context_mask"][row, TARGET]
            and int(arrays["context_controller"][row, TARGET]) == OPPONENT
        ):
            return self.battles.kind(
                watch.attacker, watch.attack, in_context(arrays, row, TARGET), watch.held
            )
        return UNKNOWN

    def _outcome(self, arrays: dict[str, np.ndarray], row: int, watch: Watch) -> str:
        copies = sum(card[0] == watch.attacker for card in monsters(arrays, row, OWN))
        if copies < watch.copies:
            return LOST
        life_points = round(float(arrays["scalars"][row, LIFE_POINTS]) * LIFE_SCALE)
        return HURT if life_points < watch.life_points else FINE

    def _tell(self, duel: int, watch: Watch, outcome: str) -> None:
        del self._watching[duel]
        kind = watch.kind or UNKNOWN
        game = self.games[self._game[duel]]
        game[kind] += 1
        game[f"{kind}:{outcome}"] += 1
        self.cards[(watch.attacker, kind, outcome)] += 1

    def flush(self) -> None:
        """The attacks of the games that ended before they were told."""
        for duel, watch in list(self._watching.items()):
            self._tell(duel, watch, UNSEEN)


class Bar:
    """What bars the learner's seat from attacking into a monster it can see
    would win the battle, with no gain its attacker's text or a trick it holds
    makes of it: an Attack whose every possible target is one, unless its
    monster may attack directly, and such a target where the attack's target
    is asked for."""

    def __init__(self, questions: Questions, battles: Battles) -> None:
        self.questions = questions
        self.battles = battles

    def __call__(self, round_: Round, mine: np.ndarray) -> np.ndarray:
        arrays = round_.arrays
        questions = arrays["decision_kind"].reshape(-1)
        barred = np.zeros((len(mine), arrays["action_kind"].shape[1]), dtype=bool)
        for place, row in enumerate(mine.tolist()):
            if questions[row] == self.questions.battle:
                theirs = monsters(arrays, row, OPPONENT)
                if not theirs:
                    continue
                held = self.battles.held(arrays, row)
                for answer in np.flatnonzero(
                    arrays["action_kind"][row] == self.questions.attack
                ).tolist():
                    if arrays["action_scalars"][row, answer, MAY_ATTACK_DIRECTLY] > 0:
                        continue
                    attacker = int(arrays["action_card"][row, answer])
                    attack = combat(arrays["action_numeric"][row, answer, 0])
                    barred[place, answer] = all(
                        self.battles.kind(attacker, attack, monster, held) == STRONGER
                        for monster in theirs
                    )
            elif self.questions.aiming(arrays, row):
                attacker, _, attack, _ = in_context(arrays, row, ACTING)
                held = self.battles.held(arrays, row)
                for answer in placed_answers(arrays, row).tolist():
                    barred[place, answer] = (
                        self.battles.kind(attacker, attack, answered(arrays, row, answer), held)
                        == STRONGER
                    )
        return barred


def opponents_of(group: str, learners: list[str]) -> list[str]:
    """The opponents a group names: the thirteen decks the network trained
    against, the pool's decks it never met, or those listed."""
    if group == "trained":
        return THIRTEEN.split(",")
    if group == "unmet":
        met = set(FOURTEEN.split(",")) | set(learners)
        return [deck for deck in sorted(pool(ROOT)) if deck not in met]
    return [deck.strip() for deck in group.split(",") if deck.strip()]


def play(
    model: Any,
    *,
    learners: list[str],
    group: str,
    games: int,
    seed: int,
    bar: bool,
    threads: int,
    slots: int,
    limit: int = 4096,
) -> dict[str, Any]:
    opponents = opponents_of(group, learners)
    results: list[dict[str, Any]] = []
    cards: Counter[tuple[int, str, str]] = Counter()
    with DuelPool(ROOT, threads=threads, slots=slots) as duels:
        questions = Questions(duels)
        battles = Battles.read(_database(), questions.hand)
        for index, learner in enumerate(learners):
            attacks = Attacks(duels, questions, battles)
            played = evaluate_games(
                duels,
                model,
                "cpu",
                learner=learner,
                assignments=evaluation_assignments(
                    opponents, games, seed + index * DECK_SEEDS + GROUP_SEEDS.get(group, 0)
                ),
                limit=limit,
                observe=attacks,
                forbid=Bar(questions, battles) if bar else None,
            )
            attacks.flush()
            for result in played:
                result.pop("learner_answers", None)
                result["learner"] = learner
                result["attacks"] = dict(attacks.games.get(int(result["index"]), {}))
            results.extend(played)
            cards.update(attacks.cards)
            won = sum(result["outcome"] == "win" for result in played)
            print(f"{learner}: {won}/{len(played)} won", file=sys.stderr, flush=True)
    return {
        "group": group,
        "opponents": opponents,
        "learners": learners,
        "games_per_opponent": games,
        "seed": seed,
        "bar": bar,
        "results": results,
        "cards": [
            [code, kind, outcome, count] for (code, kind, outcome), count in sorted(cards.items())
        ],
    }


def _database() -> Path:
    return Path(CARDS_DATABASE) if Path(CARDS_DATABASE).is_absolute() else ROOT / CARDS_DATABASE


def _games(run: dict[str, Any]) -> dict[tuple[str, str, int, int], dict[str, Any]]:
    return {
        (str(r["learner"]), str(r["opponent"]), int(r["learner_player"]), int(r["seed"])): r
        for r in run["results"]
    }


def _won(result: dict[str, Any]) -> float:
    return float(result["outcome"] == "win")


def share(results: list[dict[str, Any]], part: str, whole: tuple[str, ...]) -> tuple[float, float]:
    """The share of the attacks of ``whole`` kinds that are ``part``, and its
    standard error, the games taken as clusters."""
    parts = np.array([r["attacks"].get(part, 0) for r in results], dtype=float)
    wholes = np.array(
        [sum(r["attacks"].get(kind, 0) for kind in whole) for r in results], dtype=float
    )
    total = wholes.sum()
    if not total:
        return math.nan, math.nan
    value = parts.sum() / total
    return value, math.sqrt(((parts - value * wholes) ** 2).sum()) / total


def _name(run: dict[str, Any]) -> str:
    return f"{run['network']}{' barred' if run['bar'] else ''} vs {run['group']}"


def report(runs: list[dict[str, Any]], names: Names) -> str:
    lines = []
    for run in runs:
        results = run["results"]
        attacks = sum(sum(r["attacks"].get(kind, 0) for kind in KINDS) for r in results)
        won = sum(map(_won, results)) / len(results)
        lines.append(
            f"{_name(run)}: {len(results):,} games, won {percent(won)}, "
            f"{attacks / len(results):.2f} attacks a game"
            + (
                f", {sum(r.get('barred', 0) for r in results) / len(results):.2f} answers barred a game"
                if run["bar"]
                else ""
            )
        )
        for kind in KINDS:
            value, error = share(results, kind, KINDS)
            outcomes = ", ".join(
                f"{outcome} {percent(share(results, f'{kind}:{outcome}', (kind,))[0])}"
                for outcome in OUTCOMES
            )
            lines.append(f"    {kind:10} {percent(value):>6} ± {100 * error:.1f}  ({outcomes})")
    lines.append("")
    lines.append("Won, paired on the same deals, the first less the second:")
    gaps: dict[tuple[str, bool, str, bool], dict[str, tuple[float, float]]] = defaultdict(dict)
    for first in runs:
        for second in runs:
            if (
                first is second
                or first["group"] != second["group"]
                or first["seed"] != second["seed"]
            ):
                continue
            same_network = first["network"] == second["network"]
            if same_network and not (first["bar"] and not second["bar"]):
                continue
            if not same_network and (
                first["bar"] or second["bar"] or first["network"] < second["network"]
            ):
                continue
            a, b = _games(first), _games(second)
            keys = sorted(set(a) & set(b))
            gap, error = difference([_won(a[k]) for k in keys], [_won(b[k]) for k in keys])
            gaps[(first["network"], first["bar"], second["network"], second["bar"])][
                first["group"]
            ] = (gap, error)
            lines.append(
                f"  {_name(first)} less {_name(second)}: {points(gap, error)} over {len(keys):,} deals"
            )
    for (network, barred, other, other_barred), by_group in gaps.items():
        if "trained" in by_group and "unmet" in by_group:
            (unmet, unmet_error), (trained, trained_error) = by_group["unmet"], by_group["trained"]
            lines.append(
                f"  {network}{' barred' if barred else ''} less {other}{' barred' if other_barred else ''}, "
                f"unmet less trained: {points(unmet - trained, math.hypot(unmet_error, trained_error))}"
            )
    lines.append("")
    lines.append(
        "Stronger targets, planned or armed or not, by attacker, as-is runs, most attacks first"
        " (lost / hurt / fine / unseen):"
    )
    for run in runs:
        if run["bar"]:
            continue
        by_card: dict[int, Counter[str]] = defaultdict(Counter)
        for code, kind, outcome, count in run["cards"]:
            if kind in (PLANNED, ARMED, STRONGER):
                by_card[int(code)][kind] += int(count)
                by_card[int(code)][outcome if kind == STRONGER else "other"] += int(count)
        lines.append(f"  {_name(run)}:  planned, armed, stronger (lost / hurt / fine / unseen)")
        for code, counted in sorted(by_card.items(), key=lambda item: -item[1][STRONGER])[:15]:
            told = " / ".join(str(counted[outcome]) for outcome in OUTCOMES)
            lines.append(
                f"    {names(code)[:40]:40}{counted[PLANNED]:>6}{counted[ARMED]:>6}"
                f"{counted[STRONGER]:>6}   ({told})"
            )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    playing = commands.add_parser("play")
    playing.add_argument("checkpoint", type=Path)
    playing.add_argument("--learners", default=FOURTEEN)
    playing.add_argument("--opponents", default="trained", help="trained, unmet, or policy ids")
    playing.add_argument("--games", type=int, default=16, help="games against each opponent")
    playing.add_argument("--seed", type=int, default=2026)
    playing.add_argument("--bar", action="store_true")
    playing.add_argument("--pool-threads", type=int, default=4)
    playing.add_argument("--pool-slots", type=int, default=64)
    playing.add_argument("--output", type=Path, required=True)
    reporting = commands.add_parser("report")
    reporting.add_argument("runs", type=Path, nargs="+")
    args = parser.parse_args()

    names = Names(_database())
    if args.command == "report":
        print(report([json.loads(path.read_text(encoding="utf8")) for path in args.runs], names))
        return

    from ygo_training.checkpoints import load_inference_model

    model, payload = load_inference_model(args.checkpoint, device="cpu")
    run = play(
        model,
        learners=[deck.strip() for deck in args.learners.split(",") if deck.strip()],
        group=args.opponents,
        games=args.games,
        seed=args.seed,
        bar=args.bar,
        threads=args.pool_threads,
        slots=args.pool_slots,
    )
    run.update(
        {
            "network": args.checkpoint.stem,
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
            "checkpoint_step": payload.get("step"),
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(run, sort_keys=True), encoding="utf8")
    print(report([run], names))


if __name__ == "__main__":
    main()
