"""What the experiment on self-play says (experiments/2026-10-11-self-play.md).

    uv run python tools/self_play_report.py analysis/self-play --arm 960 --step 120

The folder `experiments/commands/self-play.sh` leaves.

The base has no reading of its own.  Under `head-to-head`, what players win
of the mirror against the start held in the other seat (`start.json`:
checkpoints of the start's run and the start itself, each told by the
updates its network has had in all; `pilot.json`: the deck's pilot), and
beside it the figures of each exploiter of the start
(`exploiter-SEED.metrics.jsonl`), a learner that begins as the start and
plays every duel against it.  An exploiter is told at each evaluation: what
its first choices win of the same deals against its victim, what they win
of the suite against the pilots, and what it won of the duels it trained on
since the evaluation before.

The arms are `with-itself` and `pilots-alone`, each `--arm` updates on from
the start with a checkpoint every `--step` of them: their figures
(`NAME.metrics.jsonl`), each checkpoint on the held-out games
(`heldout/NAME.NNNN.json`, beside `heldout/start.json`), the two head to
head (`head-to-head/with-itself.json`: the last three checkpoints of
with-itself against the last of pilots-alone, held in the other seat;
`head-to-head/pilots-alone.NNNN.json`: that checkpoint against each of the
three, held) and the decks outside training (`unseen.json`).

The exploiters of an arm's last checkpoint are `exploiter-ARM-SEED`, each
`--exploit` updates with a checkpoint every `--every` of them.  Two
checkpoints with each in turn the one held are a pair of files under
`head-to-head` (`NAME.json` and `NAME.back.json`), every deal from either
seat: an exploiter's last three checkpoints with its victim
(`exploiter-...NNNN`), and the start and the last checkpoint of each of its
exploiters with an arm's last (`start.at-ARM`, `exploiter-SEED.at-ARM`).

The readings were fixed on 2026-10-11, the third when the base was read.
They are taken by an arm's last three checkpoints, only once it has had the
updates of `--arm`:

1. Self-play keeps what it has against the pilots.  With-itself against the
   start on the held-out deals, deal by deal: lost when it wins at least 2
   points and two standard errors fewer.
2. Self-play makes the better player of the mirror.  With-itself against
   pilots-alone, every deal from either seat: better when it wins at least
   52% of the games and two standard errors over half, worse when at most
   48% and two under.
3. Self-play leaves less for an exploiter.  Two exploiters of each arm's
   last checkpoint, each by its last three checkpoints against its victim,
   every deal from either seat: less when with-itself's win at least 3
   points and two standard errors fewer of their games than pilots-alone's
   of theirs, and each of its two fewer than each of the other two; more
   when the same the other way.

Beside the first, fixed on 2026-10-13 for a second pair of arms: with-itself
is under the control against the pilots when it wins at least 1 point and two
standard errors fewer of the held-out deals than pilots-alone, deal by deal
over each arm's last three checkpoints, and over it when the same the other
way.

An arm that goes on from a shorter one is read the same way at its new
length (`--arm`), in a folder of its own.  With `--since N` it is also told
whether with-itself has gained on itself: its last three checkpoints with
its own checkpoint after N updates, each in turn the one held
(`head-to-head/with-itself.NNNN.since.json` and `.back.json`), every deal
from either seat: gained when they win at least 52% of the games and two
standard errors over half, lost when at most 48% and two under (fixed on
2026-10-12).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ppo_head_report import difference
from search_leads_report import percent, points

AGAINST = "blackwing-vs-checkpoint"
ITSELF, PILOTS = "with-itself", "pilots-alone"
ARMS = (ITSELF, PILOTS)
TITLES = {ITSELF: "With itself", PILOTS: "Pilots alone"}
# The checkpoints an arm is read by: its last ones.
END = 3
# The marks of the readings: so far under its start, so far from half, so
# far between the arms' exploiters, so far under the control against the
# pilots, and in standard errors.
LOST = 0.02
BETTER = 0.02
LEFT = 0.03
UNDER = 0.01
ERRORS = 2.0
# The exploiters of an arm's last checkpoint the third reading is taken by.
EXPLOITERS = 2
# The start, where an arm is named.
START = "start"
VICTIMS = {START: "the start", ITSELF: "With itself", PILOTS: "Pilots alone"}


def error(wins: float, games: int) -> float:
    """The standard error of a rate of ``games`` deals."""
    rate = wins / games
    return math.sqrt(rate * (1.0 - rate) / games)


def rate(wins: float, games: int) -> str:
    return f"{wins / games:.1%} ± {100 * error(wins, games):.1f}"


def within(share: float, wide: float) -> str:
    """A share of games with its standard error."""
    return f"{percent(share)} ± {100 * wide:.1f}"


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def spread(deals: list[float]) -> float:
    """The standard error of the mean of the deals."""
    middle = mean(deals)
    return math.sqrt(sum((deal - middle) ** 2 for deal in deals) / (len(deals) - 1) / len(deals))


def shares(games: list[list[float]]) -> list[float]:
    """For each deal the share won of its games, a list of the deals a game."""
    return [mean(list(deal)) for deal in zip(*games, strict=True)]


def kept(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf8"))


def head_to_head(folder: Path) -> list[dict[str, Any]]:
    """Each player measured against the start held in the other seat, the
    pilot first and the checkpoints in the order of their updates."""
    found = []
    held = None
    for name in ("pilot", "start"):
        path = folder / "head-to-head" / f"{name}.json"
        if not path.is_file():
            continue
        measured = kept(path)
        held = measured.get("opponent_checkpoint") or held
        for record in measured["records"]:
            case = record["evaluation_suite"][AGAINST]
            if "error" in case:
                raise SystemExit(
                    f"{path}: {record.get('checkpoint_label')} failed: {case['error']}"
                )
            found.append(
                {
                    "player": str(record["checkpoint_label"]),
                    "step": record.get("checkpoint_step"),
                    "wins": int(case["wins"]),
                    "games": int(case["games"]),
                    "itself": held is not None and record.get("checkpoint") == held.get("path"),
                }
            )
    return sorted(found, key=lambda player: (player["step"] is not None, player["step"] or 0))


def figures(metrics: Path) -> list[dict[str, Any]]:
    """A run's evaluations, each with the duels trained on since the one
    before: what its first choices win of the mirror against the checkpoint
    held, and of the suite against the pilots."""
    rows = [json.loads(line) for line in metrics.read_text(encoding="utf8").splitlines() if line]
    first = int(rows[0]["version"])
    found = []
    since: list[dict[str, Any]] = []
    played = 0.0
    for row in rows:
        since.append(row)
        played += float(row["duels"])
        if "evaluation_suite" not in row:
            continue
        cases = row["evaluation_suite"]
        pilots = [case for name, case in cases.items() if name != AGAINST]
        found.append(
            {
                "updates": int(row["version"]) - first + 1,
                "games": int(played),
                "against": [int(cases[AGAINST]["wins"]), int(cases[AGAINST]["games"])],
                "pilots": [
                    sum(int(case["wins"]) for case in pilots),
                    sum(int(case["games"]) for case in pilots),
                ],
                "since": since,
            }
        )
        since = []
    return found


def exploiter(metrics: Path) -> dict[str, Any]:
    """One exploiter: its evaluations, and at each what it won of the duels
    it trained on since the one before, drawing its answers."""
    described = kept(metrics.with_name(metrics.name.replace(".metrics.jsonl", ".run.json")))
    found = []
    for point in figures(metrics):
        since = point.pop("since")
        duels = sum(float(row["other_seat_duels"]["checkpoint"]) for row in since)
        point["sampled"] = (
            sum(
                float(row["other_seat_win_rate"]["checkpoint"])
                * float(row["other_seat_duels"]["checkpoint"])
                for row in since
            )
            / duels
        )
        found.append(point)
    held = described.get("opponent_checkpoint") or {}
    return {
        "seed": described.get("training_seed"),
        "victim": held.get("path"),
        "victim_step": held.get("step"),
        "of": arm_of(held.get("path")) or START,
        "learning_rate": described.get("learning_rate"),
        "points": found,
    }


def whose(victim: str) -> str:
    """A victim in words: the start, or an arm's last checkpoint."""
    return VICTIMS[victim] if victim == START else f"{VICTIMS[victim]}'s last checkpoint"


def base(folder: Path, result: dict[str, Any]) -> None:
    """The start measured before any arm, head to head, and the exploiters
    as they went: the start's, and those of each arm's last checkpoint."""
    result["head_to_head"] = head_to_head(folder)
    result["exploiters"] = {}
    if result["head_to_head"]:
        games = result["head_to_head"][0]["games"]
        print(f"Against the start held in the other seat, {games:,} deals of the mirror:")
        for player in result["head_to_head"]:
            name = (
                "the pilot" if player["step"] is None else f"after {player['step']} updates in all"
            )
            if player["itself"]:
                name += " (the start itself)"
            print(f"  {name:<46} {rate(player['wins'], player['games'])}")
        print()
    for metrics in sorted(folder.glob("exploiter-*.metrics.jsonl")):
        name = metrics.name.removesuffix(".metrics.jsonl")
        result["exploiters"][name] = exploiter(metrics)
    if not result["exploiters"]:
        return
    print("Exploiters: from their victim itself, every duel against it, a deal from one seat.")
    result["apart"] = {}
    for victim in VICTIMS:
        ends = []
        for name, found in result["exploiters"].items():
            if found["of"] != victim:
                continue
            print(
                f"\n  {name}, of {whose(victim)} (training seed {found['seed']}, "
                f"learning rate {found['learning_rate']})"
            )
            print(
                f"  {'Updates':>8} {'Games':>9} {'Against it':>19} {'The pilots':>14} "
                f"{'Duels trained on':>18}"
            )
            for point in found["points"]:
                print(
                    f"  {point['updates']:>8} {point['games']:>9,} {rate(*point['against']):>19} "
                    f"{rate(*point['pilots']):>14} {point['sampled']:>18.1%}"
                )
            last = found["points"][-END:]
            wins = sum(point["against"][0] for point in last)
            games = sum(point["against"][1] for point in last)
            found["last"] = {"evaluations": len(last), "rate": wins / games}
            ends.append(wins / games)
            # The evaluations are of the same deals: their error is that of
            # one of them at most.
            print(
                f"  over its last {len(last)} evaluations {wins / games:.1%} of the deals against "
                f"it (± {100 * error(wins / len(last), last[0]['against'][1]):.1f} at most)"
            )
        if len(ends) > 1:
            result["apart"][victim] = max(ends) - min(ends)
            print(
                f"\n  The exploiters of {whose(victim)} end "
                f"{100 * result['apart'][victim]:.1f} points apart."
            )
    print()


def heldout(folder: Path) -> tuple[list[float] | None, dict[str, dict[int, list[float]]]]:
    """The start and each arm's checkpoints on the held-out games: for every
    deal whether it was won, an arm's checkpoints by the updates of its own
    they had.  All of them are of the same deals."""
    start = None
    arms: dict[str, dict[int, list[float]]] = {name: {} for name in ARMS}
    deals = set()

    def won(path: Path) -> list[float]:
        measured = kept(path)
        deals.add((measured.get("fingerprint"), measured.get("games")))
        return [
            float(outcome == "win") for outcome in measured["checkpoints"][0]["deals"]["outcomes"]
        ]

    if (folder / "heldout" / "start.json").is_file():
        start = won(folder / "heldout" / "start.json")
    for name in ARMS:
        for path in sorted((folder / "heldout").glob(f"{name}.*.json")):
            arms[name][int(path.name.split(".")[1])] = won(path)
    if len(deals) > 1:
        raise SystemExit(f"{folder}/heldout: the measurements are not of the same held-out games")
    return start, arms


def over_last(found: dict[int, list[float]], last: list[int]) -> list[float] | None:
    """For each deal, the share of an arm's last checkpoints that win it."""
    if any(point not in found for point in last):
        return None
    return [mean(list(deal)) for deal in zip(*(found[point] for point in last), strict=True)]


def arm_of(checkpoint: str | None) -> str | None:
    """The arm a checkpoint is of, by the folder of its run."""
    if not checkpoint:
        return None
    name = Path(checkpoint).parent.parent.name
    return name if name in ARMS else None


def score(outcome: str) -> float:
    """A game for the seat it is told of: a draw is half a win."""
    return 1.0 if outcome == "win" else 0.0 if outcome == "loss" else 0.5


def games_of(path: Path, record: dict[str, Any]) -> list[float]:
    """A measured checkpoint's games of the mirror, each told for it."""
    case = record["evaluation_suite"][AGAINST]
    if "error" in case:
        raise SystemExit(f"{path}: {record.get('checkpoint_label')} failed: {case['error']}")
    return [score(outcome) for outcome in case["outcomes"]]


def pair(folder: Path, name: str) -> dict[str, Any] | None:
    """Two checkpoints on the mirror's deals, each in turn the one held
    (`NAME.json` and `NAME.back.json`): for the first of them its two games
    of every deal, the one it was measured in and the one it was held in."""
    paths = [folder / "head-to-head" / f"{name}{suffix}.json" for suffix in ("", ".back")]
    if not all(path.is_file() for path in paths):
        return None
    forward, back = (kept(path) for path in paths)
    one, other = forward["records"], back["records"]
    if (
        len(one) != 1
        or len(other) != 1
        or forward["opponent_checkpoint"]["path"] != other[0].get("checkpoint")
        or back["opponent_checkpoint"]["path"] != one[0].get("checkpoint")
        or (forward.get("seed"), forward.get("suite")) != (back.get("seed"), back.get("suite"))
    ):
        raise SystemExit(f"{paths[1]} is not {paths[0]} with the other checkpoint held")
    return {
        "one": str(one[0]["checkpoint"]),
        "step": one[0].get("checkpoint_step"),
        "other": str(other[0]["checkpoint"]),
        "other_step": other[0].get("checkpoint_step"),
        "deals": (forward.get("seed"), forward.get("suite")),
        "games": [
            games_of(paths[0], one[0]),
            [1.0 - game for game in games_of(paths[1], other[0])],
        ],
    }


def of_exploiter(played: dict[str, Any], name: str, told: dict[str, Any], updates: int) -> bool:
    """Whether a pair's first checkpoint is that exploiter after so many
    updates of its own: by the folder of its run and the updates it has had.
    The folder has the exploiter's name, with the length of its victim's arm
    after it where the arm is a longer one (`exploiter-ARM-SEED.LENGTH`)."""
    run = Path(played["one"]).parent.parent.name
    named = run == name or (run.startswith(f"{name}.") and run[len(name) + 1 :].isdigit())
    return (
        named
        and told["victim_step"] is not None
        and played["step"] == told["victim_step"] + updates
    )


def last_of(folder: Path, arm: str, updates: int) -> int | None:
    """The updates an arm's checkpoint has had in all after so many of its
    own, by its figures."""
    metrics = folder / f"{arm}.metrics.jsonl"
    if not metrics.is_file():
        return None
    with metrics.open(encoding="utf8") as lines:
        return int(json.loads(next(lines))["version"]) + updates


def exploited(folder: Path, result: dict[str, Any], updates: int, exploit: int, every: int) -> None:
    """The exploiters by their last checkpoints against their victims, every
    deal from either seat, and the third reading once each arm's are whole."""
    last = [exploit - (END - 1 - place) * every for place in range(END)]
    found: dict[str, dict[str, list[float]]] = {}
    deals = set()
    for name, told in result["exploiters"].items():
        games: list[list[float]] = []
        for point in last:
            played = pair(folder, f"{name}.{point:04d}")
            if played is None:
                break
            if played["other"] != told["victim"] or not of_exploiter(played, name, told, point):
                raise SystemExit(
                    f"{folder}/head-to-head/{name}.{point:04d}.json is not {name} after {point} "
                    "updates with its victim"
                )
            deals.add(played["deals"])
            games += played["games"]
        else:
            found.setdefault(told["of"], {})[name] = shares(games)
    if len(deals) > 1:
        raise SystemExit(
            f"{folder}/head-to-head: the exploiters are not measured on the same deals"
        )
    if not found and "arms" not in result:
        return
    readings = result.setdefault("readings", {})
    summary = result["exploited"] = {"last": last, "victims": {}}
    print(
        "Exploiters, every deal from either seat: their checkpoints after "
        f"{', '.join(map(str, last))} updates and their victim, each in turn the one held."
    )
    for victim in VICTIMS:
        if victim not in found:
            continue
        together = shares(list(found[victim].values()))
        summary["victims"][victim] = {
            "exploiters": {
                name: {"rate": mean(won), "error": spread(won)}
                for name, won in found[victim].items()
            },
            "rate": mean(together),
            "error": spread(together),
        }
        print(f"  of {whose(victim)}:")
        for name, won in found[victim].items():
            print(f"    {name:<34} {within(mean(won), spread(won))}")
        if len(found[victim]) > 1:
            print(f"    {'together':<34} {within(mean(together), spread(together))}")
    mine, theirs = found.get(ITSELF, {}), found.get(PILOTS, {})
    seeds = {
        arm: sorted(str(result["exploiters"][name]["seed"]) for name in found.get(arm, {}))
        for arm in ARMS
    }
    ends = {arm: last_of(folder, arm, updates) for arm in ARMS}
    whole = len(mine) == EXPLOITERS and seeds[ITSELF] == seeds[PILOTS] and None not in ends.values()
    for arm in ARMS if whole else ():
        for name in found[arm]:
            if result["exploiters"][name]["victim_step"] != ends[arm]:
                raise SystemExit(
                    f"{name} is not of {whose(arm)} after {updates} updates "
                    f"({result['exploiters'][name]['victim']})"
                )
    if not whole:
        print("  3. Self-play leaves less for an exploiter: not measured")
        print()
        return
    ours, others = shares(list(mine.values())), shares(list(theirs.values()))
    gap, wide = difference(ours, others)
    low = {arm: min(mean(won) for won in found[arm].values()) for arm in ARMS}
    high = {arm: max(mean(won) for won in found[arm].values()) for arm in ARMS}
    under, over = high[ITSELF] < low[PILOTS], low[ITSELF] > high[PILOTS]
    less = gap <= -LEFT and gap <= -ERRORS * wide and under
    more = gap >= LEFT and gap >= ERRORS * wide and over
    readings["exploited"] = {
        ITSELF: mean(ours),
        PILOTS: mean(others),
        "gap": gap,
        "error": wide,
        "each_under": under,
        "each_over": over,
        "less": less,
        "more": more,
    }
    sides = (
        "each of its own under each of the other's"
        if under
        else "each of its own over each of the other's"
        if over
        else "its own not all on one side of the other's"
    )
    print(
        f"  3. Self-play leaves less for an exploiter: "
        f"{'yes' if less else 'no: more' if more else 'not shown'} "
        f"({TITLES[ITSELF]}'s exploiters win {percent(mean(ours))} of their games, "
        f"{TITLES[PILOTS]}'s {percent(mean(others))}: {points(gap, wide)}; {sides})"
    )
    print()


def found_again(folder: Path, result: dict[str, Any], exploit: int) -> None:
    """What the start's exploiters found, against each arm's last
    checkpoint: the start and the last checkpoint of each of its exploiters
    with that one, every deal from either seat, and beside them the same
    exploiters against the start itself.  It has no reading."""
    own = {name: told for name, told in result["exploiters"].items() if told["of"] == START}
    rows: dict[str, dict[str, Any]] = {}
    for arm in ARMS:
        basis = pair(folder, f"start.at-{arm}")
        theirs = {name: pair(folder, f"{name}.at-{arm}") for name in own}
        if basis is None or not theirs or any(played is None for played in theirs.values()):
            continue
        for name, played in theirs.items():
            assert played is not None
            if (
                arm_of(basis["other"]) != arm
                or played["other"] != basis["other"]
                or basis["one"] != own[name]["victim"]
                or played["deals"] != basis["deals"]
                or not of_exploiter(played, name, own[name], exploit)
            ):
                raise SystemExit(
                    f"{folder}/head-to-head/{name}.at-{arm}.json is not {name} after {exploit} "
                    f"updates with the checkpoint the start is measured with"
                )
        start = shares(basis["games"])
        won = shares([shares(played["games"]) for played in theirs.values() if played])
        rows[arm] = {
            "start": [mean(start), spread(start)],
            "exploiters": [mean(won), spread(won)],
            "over": list(difference(won, start)),
        }
    if not rows:
        return
    result["found_again"] = rows
    print(
        f"What the start's exploiters found, against each arm's last checkpoint: the start, and "
        f"its exploiters after {exploit} updates, every deal from either seat."
    )
    print(f"  {'Against':<32} {'The start':>14} {'Its exploiters':>16} {'Over the start':>16}")
    itself = [pair(folder, f"{name}.{exploit:04d}") for name in own]
    if all(played is not None for played in itself):
        won = shares([shares(played["games"]) for played in itself if played])
        rows[START] = {"exploiters": [mean(won), spread(won)]}
        print(
            f"  {'the start itself':<32} {'50.0%':>14} {within(mean(won), spread(won)):>16} "
            f"{points(mean(won) - 0.5, spread(won)):>16}"
        )
    for arm in ARMS:
        if arm not in rows:
            continue
        row = rows[arm]
        print(
            f"  {whose(arm):<32} {within(*row['start']):>14} {within(*row['exploiters']):>16} "
            f"{points(*row['over']):>16}"
        )
    print()


def mirror(folder: Path) -> dict[str, Any] | None:
    """With-itself against pilots-alone, every deal from either seat: its
    last checkpoints measured against the other's last held, and the other's
    last measured against each of them held.  For each deal the share of its
    games with-itself wins, and how it does in the seat that opens and in the
    other."""
    path = folder / "head-to-head" / f"{ITSELF}.json"
    if not path.is_file():
        return None
    forward = kept(path)
    held = forward["opponent_checkpoint"]["path"]
    mine = [str(record.get("checkpoint")) for record in forward["records"]]
    if arm_of(held) != PILOTS or any(arm_of(checkpoint) != ITSELF for checkpoint in mine):
        raise SystemExit(f"{path} is not {ITSELF}'s checkpoints against {PILOTS}'s")
    back = {}
    for other in sorted((folder / "head-to-head").glob(f"{PILOTS}.*.json")):
        measured = kept(other)
        told = [str(record.get("checkpoint")) for record in measured["records"]]
        if told != [held] or (measured.get("seed"), measured.get("suite")) != (
            forward.get("seed"),
            forward.get("suite"),
        ):
            raise SystemExit(f"{other} is not {held} on the deals of {path}")
        back[measured["opponent_checkpoint"]["path"]] = measured["records"][0]
    if len(mine) != END or set(back) != set(mine):
        return None
    opening: list[float] = []
    second: list[float] = []
    games: list[list[float]] = []
    for record in forward["records"]:
        ours = [score(outcome) for outcome in record["evaluation_suite"][AGAINST]["outcomes"]]
        theirs = [
            1.0 - score(outcome)
            for outcome in back[str(record["checkpoint"])]["evaluation_suite"][AGAINST]["outcomes"]
        ]
        # The measured seat opens every other deal, from the first.
        opening += ours[0::2] + theirs[1::2]
        second += ours[1::2] + theirs[0::2]
        games += [ours, theirs]
    deals = shares(games)
    share, wide = mean(deals), spread(deals)
    return {
        "deals": len(deals),
        "games": len(deals) * len(games),
        "share": share,
        "error": wide,
        "opening": mean(opening),
        "second": mean(second),
        "better": share >= 0.5 + BETTER and share - 0.5 >= ERRORS * wide,
        "worse": share <= 0.5 - BETTER and 0.5 - share >= ERRORS * wide,
    }


def gained(folder: Path, last: list[int], since: int) -> dict[str, Any] | None:
    """With-itself's last checkpoints with its own checkpoint after `since`
    updates, each in turn the one held: for each deal the share of its games
    the later checkpoints win."""
    games: list[list[float]] = []
    for point in last:
        name = f"{ITSELF}.{point:04d}.since"
        played = pair(folder, name)
        if played is None:
            return None
        if (
            arm_of(played["one"]) != ITSELF
            or arm_of(played["other"]) != ITSELF
            or played["step"] != last_of(folder, ITSELF, point)
            or played["other_step"] != last_of(folder, ITSELF, since)
        ):
            raise SystemExit(
                f"{folder / 'head-to-head' / name}.json is not {ITSELF} after {point} updates "
                f"with its checkpoint after {since}"
            )
        games += played["games"]
    deals = shares(games)
    share, wide = mean(deals), spread(deals)
    return {
        "since": since,
        "deals": len(deals),
        "games": len(deals) * len(games),
        "share": share,
        "error": wide,
        "better": share >= 0.5 + BETTER and share - 0.5 >= ERRORS * wide,
        "worse": share <= 0.5 - BETTER and 0.5 - share >= ERRORS * wide,
    }


def unseen(folder: Path) -> dict[str, Any] | None:
    """The start and each arm's last checkpoints against the decks outside
    training: for each game the share of an arm's checkpoints that win it."""
    path = folder / "unseen.json"
    if not path.is_file():
        return None
    won: dict[str | None, list[list[float]]] = {}
    for record in kept(path)["records"]:
        cases = record["evaluation_suite"]
        games = [
            float(outcome == "win") for case in sorted(cases) for outcome in cases[case]["outcomes"]
        ]
        won.setdefault(arm_of(record.get("checkpoint")), []).append(games)
    if set(won) != {None, *ARMS} or len(won[None]) != 1:
        return None
    start = won[None][0]
    shares = {name: [mean(list(game)) for game in zip(*won[name], strict=True)] for name in ARMS}
    return {
        "games": len(start),
        "start": mean(start),
        **{name: mean(shares[name]) for name in ARMS},
        "over_start": {name: difference(shares[name], start) for name in ARMS},
        "between": difference(shares[ITSELF], shares[PILOTS]),
    }


def arms(
    folder: Path, result: dict[str, Any], updates: int, step: int, since: int | None = None
) -> None:
    """The two arms: their figures, and the readings once they are whole."""
    start, measured = heldout(folder)
    rows = {
        name: {point["updates"]: point for point in figures(folder / f"{name}.metrics.jsonl")}
        for name in ARMS
        if (folder / f"{name}.metrics.jsonl").is_file()
    }
    if start is None and not rows and not any(measured.values()):
        return
    last = [updates - (END - 1 - place) * step for place in range(END)]
    result["arms"] = {
        "updates": updates,
        "last": last,
        "start": None if start is None else mean(start),
    }
    print(
        "The arms, on from the start"
        + (
            ""
            if start is None
            else f" ({percent(mean(start))} of the {len(start):,} held-out deals)"
        )
        + ": held-out deals, the suite against the pilots, the mirror against the start."
    )
    print(f"  {'Updates':>8}" + "".join(f"{TITLES[name]:>34}" for name in ARMS))
    for point in sorted(
        {
            *(update for name in ARMS for update in measured[name]),
            *(u for r in rows.values() for u in r),
        }
    ):
        line = f"  {point:>8}"
        for name in ARMS:
            deals = measured[name].get(point)
            told = rows.get(name, {}).get(point)
            line += (
                f"{'-' if deals is None else percent(mean(deals)):>12}"
                f"{'-' if told is None else percent(told['pilots'][0] / told['pilots'][1]):>11}"
                f"{'-' if told is None else percent(told['against'][0] / told['against'][1]):>11}"
            )
        print(line)
    print()

    readings: dict[str, Any] = {}
    shares = {name: over_last(measured[name], last) for name in ARMS}
    print(f"The readings, by the checkpoints after {', '.join(map(str, last))} updates:")
    mine = shares[ITSELF]
    if start is None or mine is None:
        print("  1. Self-play keeps what it has against the pilots: not measured")
    else:
        gap, spread = difference(mine, start)
        lost = gap <= -LOST and gap <= -ERRORS * spread
        readings["kept"] = {"rate": mean(mine), "gap": gap, "error": spread, "lost": lost}
        print(
            f"  1. Self-play keeps what it has against the pilots: {'no' if lost else 'yes'} "
            f"({percent(mean(mine))} of the held-out deals, {points(gap, spread)} on the start)"
        )
        theirs = shares[PILOTS]
        if theirs is not None:
            readings["kept"]["pilots_alone"] = difference(theirs, start)
            readings["kept"]["between"] = difference(mine, theirs)
            print(
                f"     beside it: {TITLES[PILOTS]} {percent(mean(theirs))}, "
                f"{points(*readings['kept']['pilots_alone'])} on the start; "
                f"{TITLES[ITSELF]} {points(*readings['kept']['between'])} on it"
            )
            between, wide = readings["kept"]["between"]
            under = between <= -UNDER and between <= -ERRORS * wide
            over = between >= UNDER and between >= ERRORS * wide
            readings["under"] = {"gap": between, "error": wide, "under": under, "over": over}
            verdict = "yes" if under else "no: over it" if over else "not shown"
            print(
                f"  Against the pilots, {TITLES[ITSELF]} is under {TITLES[PILOTS]}: {verdict} "
                f"({points(between, wide)} on it, the held-out deals deal by deal)"
            )
    played = mirror(folder)
    if played is None:
        print("  2. Self-play makes the better player of the mirror: not measured")
    else:
        readings["mirror"] = played
        verdict = "yes" if played["better"] else "no: worse" if played["worse"] else "not shown"
        print(
            f"  2. Self-play makes the better player of the mirror: {verdict} "
            f"({percent(played['share'])} ± {100 * played['error']:.1f} of {played['games']:,} games "
            f"against {TITLES[PILOTS]}, every deal from either seat; "
            f"{percent(played['opening'])} opening, {percent(played['second'])} second)"
        )
    if since is not None:
        further = gained(folder, last, since)
        told = f"  Since its checkpoint after {since} updates, {TITLES[ITSELF]} has gained on itself: "
        if further is None:
            print(told + "not measured")
        else:
            readings["gained"] = further
            verdict = (
                "yes" if further["better"] else "no: it loses to it" if further["worse"] else "not shown"
            )
            print(
                told + f"{verdict} ({percent(further['share'])} ± {100 * further['error']:.1f} of "
                f"{further['games']:,} games against it, every deal from either seat)"
            )
    result["readings"] = readings
    outside = unseen(folder)
    if outside is not None:
        result["unseen"] = outside
        print(
            f"\nThe decks outside training, {outside['games']:,} games: the start "
            f"{percent(outside['start'])}; "
            + "; ".join(
                f"{TITLES[name]} {percent(outside[name])} ({points(*outside['over_start'][name])})"
                for name in ARMS
            )
            + f"; {TITLES[ITSELF]} {points(*outside['between'])} on {TITLES[PILOTS]}."
        )
    print()


def report(
    folder: Path,
    updates: int = 960,
    step: int = 120,
    exploit: int = 240,
    every: int = 30,
    since: int | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    arms(folder, result, updates, step, since)
    base(folder, result)
    exploited(folder, result, updates, exploit, every)
    found_again(folder, result, exploit)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("folder", type=Path)
    parser.add_argument("--arm", type=int, default=960, help="the updates of an arm")
    parser.add_argument("--step", type=int, default=120, help="the updates between its checkpoints")
    parser.add_argument("--exploit", type=int, default=240, help="the updates of an exploiter")
    parser.add_argument(
        "--every", type=int, default=30, help="the updates between an exploiter's checkpoints"
    )
    parser.add_argument(
        "--since",
        type=int,
        help="the updates of with-itself's earlier checkpoint its last three are measured with",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    found = report(args.folder, args.arm, args.step, args.exploit, args.every, args.since)
    if not found["head_to_head"] and not found["exploiters"] and "arms" not in found:
        raise SystemExit(f"{args.folder} holds no measurement of the experiment")
    if args.output:
        args.output.write_text(json.dumps(found, indent=2), encoding="utf8")


if __name__ == "__main__":
    main()
