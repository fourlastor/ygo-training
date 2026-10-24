"""How often each kind of alternative is offered, and how often the search takes it.

`policy-bench search` rows say where the search left the pilot's answer, not
how often the same alternative came up and was left alone. Plain traced games
give that: `policy-bench compare --trace true` with one library as baseline
and candidate plays the pilot alone on the search's seeds and seats.

Usage, from the repository root::

    uv run python tools/policy_search_offers.py reduce traces.jsonl decisions.jsonl
    uv run python tools/policy_search_offers.py rates decisions.jsonl \\
        analysis/policy-search/blackwing-old-strict.jsonl --cards native/ygo-policies/vendor/BabelCdb/cards.cdb

`reduce` keeps, of each traced game, the pilot's decisions a strict search
would examine. `rates` sets the search's overrides against them: the rate at
which each kind of alternative is taken, and what a gate that only tries the
kinds above a rate keeps and costs. The kinds are chosen on the first half of
each opponent's games and judged on the second.
"""

from __future__ import annotations

import argparse
import collections
import json
import sqlite3
from collections.abc import Iterator
from pathlib import Path
from typing import Any

SEARCHED = {"Idle", "Battle", "YesNo", "Position"}
PILES = {"Hand", "Deck", "Graveyard", "Banished", "Extra"}
MAX_CANDIDATES = 12
FIRST_STAGE_WORLDS = 8


def kind_text(kind: Any) -> str:
    return json.dumps(kind, sort_keys=True, separators=(",", ":"))


def key(decision: str, kind: Any, code: Any) -> str:
    return f"{decision}|{kind_text(kind)}|{json.dumps(code)}"


def same_card(choice: dict[str, Any]) -> tuple[Any, ...] | None:
    """Two copies of a card in one pile are one alternative, as in the search."""
    card = choice.get("card") or {}
    at = card.get("at") or {}
    if at.get("location") not in PILES:
        return None
    return (
        kind_text(choice["kind"]),
        card.get("code"),
        at.get("controller"),
        at.get("location"),
        choice.get("description"),
    )


def examined(seat: int, trace: list[dict[str, Any]]) -> Iterator[dict[str, Any]]:
    for answer in trace:
        observation = answer["observation"]
        if observation["me"] != seat:
            continue
        decision = answer["decision"]
        kind = decision["kind"]
        if isinstance(kind, dict):
            if "Chain" not in kind or kind["Chain"].get("forced") is not False:
                continue
            kind = "Chain"
        elif kind not in SEARCHED:
            continue
        choices = decision["choices"]
        if len(choices) != len(answer["responses"]):
            continue
        chosen = answer["choice"]
        tried = {same_card(choices[chosen])} - {None}
        alternatives = []
        for index, choice in enumerate(choices):
            if index == chosen or choice["kind"] in ("ShuffleHand", "Cancel"):
                continue
            duplicate = same_card(choice)
            if duplicate is not None:
                if duplicate in tried:
                    continue
                tried.add(duplicate)
            if len(alternatives) + 1 == MAX_CANDIDATES:
                break
            alternatives.append(key(kind, choice["kind"], (choice.get("card") or {}).get("code")))
        if not alternatives:
            continue
        yield {
            "kind": kind,
            "alternatives": alternatives,
            "in_chain": bool(observation["chain"]),
            "facing_set": any(
                card["at"]["controller"] != seat
                and card["at"]["location"] == "MonsterZone"
                and not card["position"]["face_up"]
                for card in observation["cards"]
            ),
        }


def reduce(traces: Path, output: Path) -> None:
    games = decisions = 0
    with traces.open(encoding="utf8") as rows, output.open("w", encoding="utf8") as out:
        for line in rows:
            row = json.loads(line)
            kept = list(examined(row["seat"], row["baseline"]["trace"]))
            out.write(json.dumps({"b": row["b"], "seed": row["seed"], "decisions": kept}) + "\n")
            games += 1
            decisions += len(kept)
    print(f"{games} games, {decisions} decisions a search would list")


def pretty(text: str, names: dict[int, str]) -> str:
    decision, kind, code = text.split("|", 2)
    kind = json.loads(kind) if kind.startswith('"') else kind
    code = json.loads(code)
    return f"{decision}: {kind}" + ("" if code is None else f" {names.get(code, code)}")


def rates(decisions: Path, search: Path, cards: Path) -> None:
    with sqlite3.connect(cards) as connection:
        names = dict(connection.execute("select id, name from texts"))
    games = [json.loads(line) for line in decisions.read_text().splitlines()]
    strict = [d for g in games for d in g["decisions"] if not d["in_chain"] and not d["facing_set"]]
    offers = collections.Counter(a for d in strict for a in d["alternatives"])
    full = sum(FIRST_STAGE_WORLDS * (1 + len(d["alternatives"])) for d in strict)
    rows = [json.loads(line) for line in search.read_text().splitlines()]
    first = {}
    for row in rows:
        first[row["b"]] = min(first.get(row["b"], row["seed"]), row["seed"])
    half = max(row["seed"] - first[row["b"]] for row in rows) // 2 + 1
    chosen: list[collections.Counter[str]] = [collections.Counter(), collections.Counter()]
    counts = [0, 0]
    for row in rows:
        held_out = row["seed"] - first[row["b"]] >= half
        counts[held_out] += 1
        for deviation in row["search"]["deviations"]:
            to = deviation["to"]
            chosen[held_out][key(deviation["kind"], to["kind"], to.get("code"))] += 1
    rate = {
        kind: (chosen[0][kind] / counts[0]) / (offers[kind] / len(games))
        for kind in offers
    }
    print(
        f"{len(games)} traced games, {len(strict) / len(games):.1f} decisions a game the strict search "
        f"examines, {len(offers)} kinds of alternative, {full / len(games):.0f} first-stage playouts a game"
    )
    print("most offered:   offers a game   taken")
    for kind, count in offers.most_common(10):
        print(f"   {count / len(games):6.2f}   {100 * rate[kind]:5.1f}%   {pretty(kind, names)}")
    print("most often taken (offered at least once in five games):")
    regular = [kind for kind in offers if offers[kind] / len(games) >= 0.2]
    for kind in sorted(regular, key=lambda k: -rate[k])[:10]:
        print(f"   {offers[kind] / len(games):6.2f}   {100 * rate[kind]:5.1f}%   {pretty(kind, names)}")
    print("a gate on the kinds taken at least this often (three or more times in the first half):")
    print("   rate   kinds   decisions a game   first-stage playouts   overrides kept: first half, second half")
    for threshold in (0.0, 0.01, 0.02, 0.03, 0.05):
        gate = {kind for kind in offers if chosen[0][kind] >= 3 and rate[kind] >= threshold}
        cost = searched = 0
        for d in strict:
            kept = sum(a in gate for a in d["alternatives"])
            if kept:
                searched += 1
                cost += FIRST_STAGE_WORLDS * (1 + kept)
        shares = [
            100 * sum(v for k, v in chosen[part].items() if k in gate) / sum(chosen[part].values())
            for part in (0, 1)
        ]
        print(
            f"   {100 * threshold:4.1f}%  {len(gate):4d}   {searched / len(games):8.1f}   "
            f"{100 * cost / full:12.1f}%   {shares[0]:14.1f}%  {shares[1]:5.1f}%"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    reducer = commands.add_parser("reduce", help="keep the decisions a strict search would examine")
    reducer.add_argument("traces", type=Path)
    reducer.add_argument("output", type=Path)
    rater = commands.add_parser("rates", help="override rates and what a gate on them keeps")
    rater.add_argument("decisions", type=Path)
    rater.add_argument("search", type=Path)
    rater.add_argument("--cards", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "reduce":
        reduce(args.traces, args.output)
    else:
        rates(args.decisions, args.search, args.cards)


if __name__ == "__main__":
    main()
