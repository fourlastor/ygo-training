"""What tells a decision worth a search from one that is not.

Reads `policy-bench search --log true` rows, where every decision the search
examined is recorded with its situation, how each alternative did in the
first-stage worlds, the playouts it took and what was chosen. Reports where
the overrides are, and how many of them a cheaper search would keep.

Usage, from the repository root::

    uv run python tools/policy_search_decisions.py analysis/policy-search/blackwing-logged.jsonl
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Any

import numpy as np

PHASES = ("Draw", "Standby", "Main1", "BattleStart", "BattleStep", "Damage", "DamageCalculation", "Battle", "Main2", "End")


def kind_text(kind: Any) -> str:
    return kind if isinstance(kind, str) else json.dumps(kind, sort_keys=True, separators=(",", ":"))


def key(decision: str, label: dict[str, Any]) -> str:
    return f"{decision}|{kind_text(label['kind'])}|{json.dumps(label.get('code'))}"


def load(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    rows = [r for r in rows if "error" not in r["search"] and "error" not in r["baseline"]]
    records = []
    for game, row in enumerate(rows):
        for entry in row["search"]["examined"]:
            situation = entry["situation"]
            wins, played = entry["first"][0]
            records.append(
                {
                    "game": game,
                    "kind": entry["kind"],
                    "turn": entry["turn"],
                    "phase": situation["phase"],
                    "own_turn": bool(situation["own_turn"]),
                    "life": situation["life"],
                    "monsters": situation["monsters"],
                    "backrow": situation["backrow"],
                    "set_backrow": situation["set_backrow"],
                    "hand": situation["hand"],
                    "attack": bool(situation["attack"]),
                    "pilot": key(entry["kind"], entry["labels"][0]),
                    "alternatives": [key(entry["kind"], label) for label in entry["labels"][1:]],
                    "pilot_wins": wins if played else None,
                    "playouts": entry["playouts"],
                    "override": entry["chosen"] != 0,
                    "skipped": bool(entry["skipped"]),
                }
            )
    return rows, records


def table(title: str, groups: dict[Any, list[dict[str, Any]]], total_overrides: int, total_playouts: int) -> None:
    print(f"   {title}")
    print("      decisions  share  override rate  share of overrides  share of playouts")
    count = sum(len(v) for v in groups.values())
    for name, members in groups.items():
        overrides = sum(m["override"] for m in members)
        playouts = sum(m["playouts"] for m in members)
        print(
            f"      {name!s:22} {len(members):6d} {100 * len(members) / count:5.1f}%  "
            f"{100 * overrides / max(1, len(members)):5.1f}%  {100 * overrides / max(1, total_overrides):6.1f}%  "
            f"{100 * playouts / max(1, total_playouts):6.1f}%"
        )


def features(records: list[dict[str, Any]], *, probe: bool) -> tuple[np.ndarray, list[str]]:
    kinds = sorted({r["kind"] for r in records})
    phases = sorted({r["phase"] for r in records})
    alternative_counts = collections.Counter(a for r in records for a in r["alternatives"])
    pilot_counts = collections.Counter(r["pilot"] for r in records)
    alternatives = [k for k, v in alternative_counts.most_common() if v >= 30]
    pilots = [k for k, v in pilot_counts.most_common() if v >= 30]
    names = (
        [f"kind={k}" for k in kinds]
        + [f"phase={p}" for p in phases]
        + ["own_turn", "attack", "turn", "life_me", "life_opp", "life_diff", "monsters_me", "monsters_opp",
           "backrow_me", "backrow_opp", "set_me", "set_opp", "hand_me", "hand_opp", "alternatives"]
        + [f"alt={a}" for a in alternatives]
        + [f"pilot={p}" for p in pilots]
        + (["pilot_wins"] if probe else [])
    )
    matrix = np.zeros((len(records), len(names)), dtype=np.float32)
    column = {name: index for index, name in enumerate(names)}
    for row, r in enumerate(records):
        matrix[row, column[f"kind={r['kind']}"]] = 1
        matrix[row, column[f"phase={r['phase']}"]] = 1
        values = {
            "own_turn": r["own_turn"], "attack": r["attack"], "turn": min(r["turn"], 30),
            "life_me": r["life"][0] / 8000, "life_opp": r["life"][1] / 8000,
            "life_diff": (r["life"][0] - r["life"][1]) / 8000,
            "monsters_me": r["monsters"][0], "monsters_opp": r["monsters"][1],
            "backrow_me": r["backrow"][0], "backrow_opp": r["backrow"][1],
            "set_me": r["set_backrow"][0], "set_opp": r["set_backrow"][1],
            "hand_me": r["hand"][0], "hand_opp": r["hand"][1], "alternatives": len(r["alternatives"]),
        }
        for name, value in values.items():
            matrix[row, column[name]] = float(value)
        for a in r["alternatives"]:
            if f"alt={a}" in column:
                matrix[row, column[f"alt={a}"]] = 1
        if f"pilot={r['pilot']}" in column:
            matrix[row, column[f"pilot={r['pilot']}"]] = 1
        if probe:
            matrix[row, column["pilot_wins"]] = -1 if r["pilot_wins"] is None else r["pilot_wins"]
    return matrix, names


def frontier(records: list[dict[str, Any]], *, probe: bool) -> None:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold

    matrix, _ = features(records, probe=probe)
    target = np.array([r["override"] for r in records])
    groups = np.array([r["game"] for r in records])
    scores = np.zeros(len(records))
    for train, test in GroupKFold(n_splits=5).split(matrix, target, groups):
        model = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.08, max_iter=200, l2_regularization=1.0)
        model.fit(matrix[train], target[train])
        scores[test] = model.predict_proba(matrix[test])[:, 1]
    # What searching every alternative costs in the first stage; with the
    # probe, the pilot's own first-stage worlds are spent on every decision.
    full = np.array([8 * (1 + len(r["alternatives"])) for r in records], dtype=float)
    order = np.argsort(-scores)
    overrides = np.cumsum(target[order]) / max(1, target.sum())
    decisions = np.arange(1, len(records) + 1) / len(records)
    cost = np.cumsum(full[order]) / full.sum()
    if probe:
        cost = (8 * len(records) + np.cumsum((full - 8)[order])) / full.sum()
    label = "the pilot's first-stage worlds and the situation" if probe else "the situation alone"
    print(f"   ranking decisions by {label} (5-fold by game, AUC {roc_auc_score(target, scores):.3f}):")
    print("      overrides kept   decisions searched   first-stage playouts")
    for share in (0.5, 0.7, 0.8, 0.9, 0.95, 0.99):
        index = int(np.searchsorted(overrides, share))
        index = min(index, len(records) - 1)
        print(f"      {100 * share:5.0f}%           {100 * decisions[index]:5.1f}%               {100 * cost[index]:5.1f}%")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("rows", type=Path, nargs="+")
    parser.add_argument("--compare", type=Path, help="rows of the same games from a search without the shortcut")
    args = parser.parse_args()
    for path in args.rows:
        rows, records = load(path)
        games = len(rows)
        overrides = sum(r["override"] for r in records)
        playouts = sum(r["playouts"] for r in records)
        print(f"== {path.name}: {games} games, {len(records)} decisions examined ({len(records) / games:.1f} a game), "
              f"{overrides} overrides ({100 * overrides / len(records):.1f}%), {playouts / games:.0f} playouts a game")
        if args.compare is not None:
            before = {(r["b"], r["seed"]): r for r in map(json.loads, args.compare.read_text().splitlines())}
            shared = [r for r in rows if (r["b"], r["seed"]) in before]
            same = sum(r["search"]["digest"] == before[(r["b"], r["seed"])]["search"]["digest"] for r in shared)
            old = sum(before[(r["b"], r["seed"])]["search"]["playouts"] for r in shared)
            new = sum(r["search"]["playouts"] for r in shared)
            print(f"   against {args.compare.name}: {same} of {len(shared)} games identical; "
                  f"playouts {old / len(shared):.0f} -> {new / len(shared):.0f} a game ({100 * new / old:.1f}%)")
        skipped = [r for r in records if r["skipped"]]
        print(f"   pilot's answer won every first-stage world: {100 * len(skipped) / len(records):.1f}% of decisions "
              f"(no alternative tried, none could be ahead)")
        by_wins: dict[Any, list[dict[str, Any]]] = collections.OrderedDict()
        for wins in range(9):
            members = [r for r in records if r["pilot_wins"] == wins]
            if members:
                by_wins[f"pilot wins {wins} of 8"] = members
        table("by how the pilot's answer did in the first-stage worlds", by_wins, overrides, playouts)
        by_kind: dict[Any, list[dict[str, Any]]] = collections.OrderedDict()
        for kind in sorted({r["kind"] for r in records}):
            for own in (True, False):
                members = [r for r in records if r["kind"] == kind and r["own_turn"] == own]
                if members:
                    by_kind[f"{kind}, {'own' if own else 'their'} turn"] = members
        table("by decision and whose turn it is", by_kind, overrides, playouts)
        frontier(records, probe=False)
        frontier(records, probe=True)


if __name__ == "__main__":
    main()
