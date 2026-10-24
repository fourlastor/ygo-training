"""Whether vectors made from the cards' texts put together the cards that do
the same things: a test of card-effect embeddings before any network reads
them.

    uv run python tools/card_embedding_check.py texts analysis/card-embeddings/texts.json
    uv run python tools/card_embedding_check.py check analysis/card-embeddings/texts.json \\
        --embeddings voyage-4-nano=analysis/card-embeddings/voyage-4-nano.npy \\
        --output analysis/card-embeddings/report.txt

``texts`` writes the cards of the policies' pool (``data/wc2011.lflist.conf``)
that have an effect, one text each: the card's kind and its text from the
card database, its own name put as "this card" (a name would stand for the
identity, and the test is of what a card does), the database's note that a
text is unofficial taken off, an alternate artwork counted once and a Normal
Monster, whose text is a story, left out.  Beside each, the
answer key: what the card's script declares to the engine, which is the rules
as the engine plays them and not the words.  From every ``SetCategory``,
``SetType``, ``SetCode`` and ``SetProperty`` of the script the names it
gives (``CATEGORY_SPECIAL_SUMMON``, ``EFFECT_TYPE_QUICK_O``,
``EVENT_SUMMON_SUCCESS``, ``EFFECT_FLAG_CARD_TARGET``...), and every
operation of the engine it calls that acts (``Duel.Draw``, ``Duel.Destroy``...;
not those that ask or tell, ``Duel.GetMatchingGroup`` or ``Duel.Hint``).

``check`` measures vectors given in the order of those texts (a ``.npy`` file
of one row a text, made by whatever made them), beside two of its own: random
vectors, which say what nothing is worth, and vectors of the words alone (each
word and pair of words weighed by how rare it is among the texts, and the
whole brought down to 128 numbers), which need no model.  Vectors given wider
than that are measured brought down to 128 numbers too, and either way beside
the words' (the two set end to end).  With ``--scripts`` vectors made from the
cards' scripts are measured too: the scripts' words, their key as flags, and
any given with ``--from-scripts``; each kind is told by the judge it is not
made from (a vector made from a script is judged by how the texts read, as
the key is made from the scripts).  Each set of vectors is told by:

- its neighbours: of the ten nearest cards to each card, how much of its key
  they share (the Jaccard index of the two sets of names), against ten cards
  taken at random;
- a readout: brought down to 128 numbers (its main directions), a linear
  readout of each name of the key fitted on half the cards and tried on the
  other half, by the area under its curve (0.5 is a guess, 1 every card in
  its place); the names that at least 20 cards hold and no more than half;
- the same script: cards whose scripts are the same but for numbers (500
  damage or 800) found among each other's nearest;
- for some cards, by name, their five nearest, to be read.

Nothing is trained and nothing in the trainer changes; numpy alone.
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from itertools import pairwise
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
POLICIES = ROOT / "native" / "ygo-policies"
POOL = POLICIES / "data" / "wc2011.lflist.conf"
DATABASE = POLICIES / "vendor" / "BabelCdb" / "cards.cdb"
SCRIPTS = POLICIES / "vendor" / "CardScripts"

# The kinds of a card the database's type field holds, in the words of a card.
KINDS = (
    (0x10000, "Quick-Play"),
    (0x20000, "Continuous"),
    (0x40000, "Equip"),
    (0x80000, "Field"),
    (0x100000, "Counter"),
    (0x80, "Ritual"),
    (0x40, "Fusion"),
    (0x2000, "Synchro"),
    (0x800000, "Xyz"),
    (0x1000, "Tuner"),
    (0x200000, "Flip"),
    (0x200, "Spirit"),
    (0x400, "Union"),
    (0x800, "Gemini"),
    (0x400000, "Toon"),
    (0x100, "Trap Monster"),
)
MONSTER, SPELL, TRAP, EFFECT = 0x1, 0x2, 0x4, 0x20
# What the database adds below some texts, which the card does not say.
NOTE = "* The above text is unofficial and describes the card's functionality in the OCG."
DECLARED = re.compile(r":Set(Category|Type|Code|Property)\(([^)]*)\)")
NAMED = re.compile(r"\b(CATEGORY|EFFECT_TYPE|EVENT|EFFECT_FLAG|EFFECT)_[A-Z0-9_]+")
OPERATION = re.compile(r"\bDuel\.([A-Z][A-Za-z]+)\(")
# Operations that ask or tell rather than act: no part of what a card does.
ASKING = ("Get", "Is", "Select", "Check", "Hint", "Confirm", "Shuffle", "Register", "Readjust")
ASKING += ("Adjust", "Raise", "Break")
TELLING = {"SetOperationInfo", "SetPossibleOperationInfo", "SetTargetCard", "SetTargetPlayer"}
TELLING |= {"SetTargetParam", "SetChainLimit", "SetChainLimitTillChainEnd", "SetSelectedCard"}
NEIGHBOURS = 10
READ_WIDTH = 128
# Cards whose nearest are told by name.
READ = (
    "Monster Reborn",
    "Premature Burial",
    "Dark Hole",
    "Mirror Force",
    "Mystical Space Typhoon",
    "Pot of Greed",
    "Sangan",
    "Black Whirlwind",
    "Solemn Judgment",
    "Gladiator Beast Bestiari",
    "Card Trooper",
    "Brionac, Dragon of the Ice Barrier",
)


def kind_of(kind: int) -> str:
    """A card's kind as its frame says it: "Quick-Play Spell", "Flip Effect
    Monster", "Tuner Synchro Monster"."""
    words = [word for bit, word in KINDS if kind & bit]
    if kind & SPELL:
        return " ".join([*words, "Spell"])
    if kind & TRAP:
        return " ".join([*words, "Trap"])
    return " ".join([*words, "Effect Monster" if kind & EFFECT else "Monster"])


def script_of(code: int, scripts: dict[int, Path]) -> str:
    path = scripts.get(code)
    return path.read_text(errors="replace") if path else ""


def key_of(script: str) -> list[str]:
    """What a script declares to the engine and the engine's operations it
    calls: the answer key of a card."""
    names = set()
    for _, given in DECLARED.findall(script):
        names.update(match.group(0) for match in NAMED.finditer(given))
    names.update(
        f"Duel.{name}"
        for name in OPERATION.findall(script)
        if not name.startswith(ASKING) and name not in TELLING
    )
    return sorted(names)


def readable(script: str) -> str:
    """A script as a model is given it: without its comments, which name the
    card, and without blank lines."""
    lines = (re.sub(r"--.*", "", line).rstrip() for line in script.splitlines())
    return "\n".join(line for line in lines if line.strip())


def same_script(script: str, code: int) -> str:
    """A script with its comments, numbers and blanks taken out: two cards
    whose scripts are the same but for numbers come to the same string."""
    text = re.sub(r"--[^\n]*", "", script).replace(f"c{code}", "c")
    return re.sub(r"\s+", "", re.sub(r"\d+", "#", text))


def pool_codes() -> list[int]:
    """The codes of the policies' pool, in the order of its list."""
    return [
        int(match.group(1))
        for line in POOL.read_text().splitlines()
        if (match := re.match(r"^(\d+)\s+\d", line))
    ]


def database_cards() -> dict[int, tuple[int, int, int, str, str]]:
    """Every card of the database by its code: its code, alias, kind, name
    and text."""
    database = sqlite3.connect(DATABASE)
    return {
        row[0]: row
        for row in database.execute(
            "select d.id, d.alias, d.type, t.name, t.desc from datas d join texts t on t.id = d.id"
        )
    }


def alternate_of(code: int, cards: dict[int, tuple], pool: set[int]) -> int | None:
    """The card of the pool an alternate artwork is of, where ``code`` is one:
    its alias, of the same name. A card whose name is only treated as
    another's (Harpie Lady 1, as "Harpie Lady") is a card of its own."""
    alias, name = cards[code][1], cards[code][3]
    if alias and alias in pool and cards[alias][3] == name:
        return alias
    return None


def said_of(card: tuple[int, int, int, str, str]) -> str | None:
    """A card's text as vectors are made of it: its kind and its text, its
    own name put as "this card" and the database's note taken off; none for
    a Normal Monster, whose text is a story."""
    _, _, kind, name, text = card
    if kind & MONSTER and not kind & EFFECT:
        return None
    said = f"{kind_of(kind)}. {text.replace(name, 'this card')}".replace("\r", "")
    # A note of the database's own, not of the card.
    return said.replace(NOTE, "").strip()


def made_texts(out: Path) -> None:
    codes = pool_codes()
    cards = database_cards()
    scripts = {
        int(path.stem[1:]): path for path in SCRIPTS.rglob("c*.lua") if path.stem[1:].isdigit()
    }
    pool = set(codes)
    told: dict[str, dict] = {}
    for code in codes:
        if alternate_of(code, cards, pool) is not None:
            continue
        _, _, _, name, _ = cards[code]
        said = said_of(cards[code])
        if said is None or said in told:
            continue
        script = script_of(code, scripts)
        told[said] = {
            "code": code,
            "name": name,
            "text": said,
            "key": key_of(script),
            "script": same_script(script, code) if script else "",
            "lua": readable(script),
        }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(list(told.values()), indent=1))
    keyed = sum(1 for card in told.values() if card["key"])
    print(f"{len(told)} texts of the pool's {len(codes)} cards, {keyed} with a key: {out}")


def normalized(vectors: np.ndarray) -> np.ndarray:
    vectors = vectors.astype(np.float64)
    return vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)


def main_directions(vectors: np.ndarray, width: int) -> np.ndarray:
    """The vectors brought down to their ``width`` main directions."""
    centred = vectors - vectors.mean(axis=0)
    gram = centred @ centred.T
    values, places = np.linalg.eigh(gram)
    order = np.argsort(values)[::-1][:width]
    return places[:, order] * np.sqrt(np.maximum(values[order], 0.0))


def words(texts: list[str]) -> np.ndarray:
    """Each text by its words and pairs of words, weighed by how rare they
    are, brought down to its main directions."""
    return main_directions(weighed(texts), READ_WIDTH)


def weighed(texts: list[str]) -> np.ndarray:
    """Each text by its words and pairs of words, weighed by how rare they
    are, of length one: the cosine of two is how alike their words are."""
    tokens = []
    for text in texts:
        found = re.findall(r"[a-z0-9]+", text.lower())
        tokens.append(found + [f"{a} {b}" for a, b in pairwise(found)])
    counted: dict[str, int] = {}
    for found in tokens:
        for token in set(found):
            counted[token] = counted.get(token, 0) + 1
    kept = sorted(token for token, count in counted.items() if count >= 2)
    place = {token: index for index, token in enumerate(kept)}
    weights = np.zeros((len(texts), len(kept)), dtype=np.float32)
    for row, found in enumerate(tokens):
        for token in found:
            if token in place:
                weights[row, place[token]] += 1.0
    rarity = np.log((1 + len(texts)) / (1 + np.array([counted[token] for token in kept]))) + 1
    return normalized(weights * rarity)


def auc(scores: np.ndarray, truth: np.ndarray) -> float:
    order = np.argsort(scores)
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    positives = truth.sum()
    negatives = len(truth) - positives
    return float((ranks[truth].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def measured(
    vectors: np.ndarray, cards: list[dict], read_alike: np.ndarray, seed: int = 2012
) -> dict:
    names = sorted({name for card in cards for name in card["key"]})
    held = np.array([[name in card["key"] for name in names] for card in cards])
    keyed = held.any(axis=1)
    unit = normalized(vectors)
    alike = unit @ unit.T
    np.fill_diagonal(alike, -np.inf)
    nearest = np.argsort(-alike, axis=1)[:, :NEIGHBOURS]
    generator = np.random.default_rng(seed)

    def jaccard(one: int, other: np.ndarray) -> np.ndarray:
        both = (held[one] & held[other]).sum(axis=1)
        either = (held[one] | held[other]).sum(axis=1)
        return both / np.maximum(either, 1)

    # The texts' judge: how alike the words of a card's nearest are to its own
    # (``read_alike``, the cosines of the texts' weighed words).
    every = np.arange(len(cards))
    read_near = np.take_along_axis(read_alike, nearest, axis=1).mean(axis=1)
    drawn = np.array([generator.choice(every[every != card], NEIGHBOURS) for card in every])
    read_chance = np.take_along_axis(read_alike, drawn, axis=1).mean(axis=1)
    near = []
    chance = []
    for card in np.flatnonzero(keyed):
        others = nearest[card][keyed[nearest[card]]]
        if len(others):
            near.append(jaccard(card, others).mean())
        others = generator.choice(
            np.flatnonzero(keyed & (np.arange(len(cards)) != card)), NEIGHBOURS
        )
        chance.append(jaccard(card, others).mean())

    # The readout, on the vectors' main directions, the same halves for all.
    reduced = main_directions(vectors.astype(np.float64), min(READ_WIDTH, vectors.shape[1]))
    reduced = (reduced - reduced.mean(axis=0)) / np.maximum(reduced.std(axis=0), 1e-12)
    halves = np.random.default_rng(seed).permutation(len(cards))
    fit, test = halves[: len(cards) // 2], halves[len(cards) // 2 :]
    fit, test = fit[keyed[fit]], test[keyed[test]]
    design = np.hstack([reduced, np.ones((len(cards), 1))])
    penalty = 10.0 * np.eye(design.shape[1])
    solved = np.linalg.solve(design[fit].T @ design[fit] + penalty, design[fit].T)
    areas = {}
    for column, name in enumerate(names):
        count = held[keyed, column].sum()
        if count < 20 or count > keyed.sum() / 2:
            continue
        truth = held[:, column].astype(np.float64)
        weights = solved @ (truth[fit] - truth[fit].mean())
        if held[test, column].any() and not held[test, column].all():
            areas[name] = auc(design[test] @ weights, held[test, column])

    # The same script but for numbers: where a card's group-mates come.
    groups: dict[str, list[int]] = {}
    for index, card in enumerate(cards):
        if card["script"]:
            groups.setdefault(card["script"], []).append(index)
    ranks = []
    for members in groups.values():
        if len(members) < 2:
            continue
        for card in members:
            order = np.argsort(-alike[card])
            mates = np.isin(order, [mate for mate in members if mate != card])
            ranks.append(int(np.argmax(mates)) + 1)
    ranks_array = np.array(ranks)
    told = {}
    by_name = {card["name"]: index for index, card in enumerate(cards)}
    for name in READ:
        if name in by_name:
            told[name] = [cards[other]["name"] for other in nearest[by_name[name]][:5]]
    return {
        "neighbours": float(np.mean(near)),
        "chance": float(np.mean(chance)),
        "read_alike": float(read_near.mean()),
        "read_chance": float(read_chance.mean()),
        "readout": float(np.mean(list(areas.values()))),
        "readouts": areas,
        "same_script": {
            "cards": len(ranks),
            "first": float((ranks_array == 1).mean()) if len(ranks) else float("nan"),
            "five": float((ranks_array <= 5).mean()) if len(ranks) else float("nan"),
            "median": float(np.median(ranks_array)) if len(ranks) else float("nan"),
        },
        "nearest": told,
    }


def checked(
    texts_file: Path,
    given: list[str],
    from_scripts: list[str],
    output: Path | None,
    scripts: bool = False,
) -> None:
    cards = json.loads(texts_file.read_text())
    texts = [card["text"] for card in cards]
    said = weighed(texts).astype(np.float32)
    read_alike = (said @ said.T).astype(np.float64)
    del said
    names = sorted({name for card in cards for name in card["key"]})
    # Each set of vectors, and what it was made from: a judge made from the
    # same is no test of it.
    sets: dict[str, tuple[np.ndarray, str]] = {
        "random": (np.random.default_rng(2012).standard_normal((len(cards), READ_WIDTH)), ""),
        "words": (words(texts), "texts"),
    }
    if scripts:
        sets["script words"] = (words([card["lua"] for card in cards]), "scripts")
        sets["script key"] = (
            np.array([[name in card["key"] for name in names] for card in cards], dtype=float),
            "scripts",
        )
        sets["words + script words"] = (
            np.hstack([normalized(sets["words"][0]), normalized(sets["script words"][0])]),
            "both",
        )
    for items, made in ((given, "texts"), (from_scripts, "scripts")):
        for item in items:
            name, _, path = item.partition("=")
            vectors = np.load(path)
            if len(vectors) != len(cards):
                raise SystemExit(f"{path} has {len(vectors)} rows for {len(cards)} texts")
            sets[name] = (vectors, made)
            # As a network would take them: brought down to as many numbers
            # as the words, and beside the words.
            if vectors.shape[1] > READ_WIDTH:
                vectors = main_directions(vectors.astype(np.float64), READ_WIDTH)
                sets[f"{name} at {READ_WIDTH}"] = (vectors, made)
            if made == "texts":
                sets[f"words + {name}"] = (
                    np.hstack([normalized(sets["words"][0]), normalized(vectors)]),
                    "texts",
                )
    results = {name: measured(vectors, cards, read_alike) for name, (vectors, _) in sets.items()}
    keyed = sum(1 for card in cards if card["key"])
    out = [
        (
            f"Card-effect vectors against what the cards' scripts declare and how their texts "
            f"read: {len(cards)} texts of the pool, {keyed} with a key"
        ),
        "",
        (f"{'':<34}{'':>6}{'by the scripts':^32}{'by the texts':^22}{'same script':^18}"),
        (
            f"{'vectors':<26}{'from':<8}{'width':>6}{'nearest':>11}{'at random':>11}{'readout':>10}"
            f"{'nearest':>11}{'at random':>11}{'first':>9}{'in five':>9}"
        ),
    ]

    def cell(value: float, width: int, form: str, own: bool) -> str:
        told = format(value, form)
        return f"({told})".rjust(width) if own else told.rjust(width)

    for name, found in results.items():
        vectors, made = sets[name]
        same = found["same_script"]
        scripts_own = made in ("scripts", "both")
        texts_own = made in ("texts", "both")
        out.append(
            f"{name:<26}{made or '-':<8}{vectors.shape[1]:>6}"
            + cell(found["neighbours"], 11, ".3f", scripts_own)
            + cell(found["chance"], 11, ".3f", scripts_own)
            + cell(found["readout"], 10, ".3f", scripts_own)
            + cell(found["read_alike"], 11, ".3f", texts_own)
            + cell(found["read_chance"], 11, ".3f", texts_own)
            + cell(same["first"], 9, ".1%", scripts_own)
            + cell(same["five"], 9, ".1%", scripts_own)
        )
    out += [
        "",
        "In brackets, a judge made from what the vectors were made from: no test of them.",
        f"By the scripts: how much of a card's key its {NEIGHBOURS} nearest share on average",
        f"(Jaccard), and {NEIGHBOURS} cards drawn at random; the readout, the mean area under the",
        f"curve of a linear readout of each name of the key from {READ_WIDTH} main directions,",
        "fitted on half the cards and tried on the others. By the texts: how alike the words",
        f"of a card's {NEIGHBOURS} nearest are to its own (the cosine of their weighed words),",
        "and of cards drawn at random. Same script: of the cards whose scripts are the same but",
        (
            f"for numbers ({next(iter(results.values()))['same_script']['cards']} cards), how "
            "often one of them is the nearest, and among the five."
        ),
    ]
    readouts = sorted(next(iter(results.values()))["readouts"])
    worst = {
        name: sorted(found["readouts"].items(), key=lambda pair: pair[1])[:6]
        for name, found in results.items()
        if name != "random"
    }
    out += ["", f"The readout by name ({len(readouts)} names), the hardest for each:"]
    for name, pairs in worst.items():
        out.append(f"  {name}: " + ", ".join(f"{key} {area:.2f}" for key, area in pairs))
    out += ["", "Nearest five, by name:"]
    for name, found in results.items():
        if name == "random":
            continue
        out.append(f"  {name}:")
        for card, near in found["nearest"].items():
            out.append(f"    {card}: {'; '.join(near)}")
    print("\n".join(out))
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("\n".join(out) + "\n")
        output.with_suffix(".json").write_text(json.dumps(results, indent=1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parts = parser.add_subparsers(dest="part", required=True)
    texts = parts.add_parser("texts", help="the pool's cards, their texts and keys")
    texts.add_argument("output", type=Path)
    check = parts.add_parser("check", help="vectors against the key")
    check.add_argument("texts", type=Path)
    check.add_argument(
        "--embeddings", action="append", default=[], help="NAME=FILE.npy, made from the texts"
    )
    check.add_argument(
        "--from-scripts", action="append", default=[], help="NAME=FILE.npy, made from the scripts"
    )
    check.add_argument(
        "--scripts",
        action="store_true",
        help="the scripts' words and key measured as vectors too, beside those made from texts",
    )
    check.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.part == "texts":
        made_texts(args.output)
    else:
        checked(args.texts, args.embeddings, args.from_scripts, args.output, args.scripts)


if __name__ == "__main__":
    sys.exit(main())
