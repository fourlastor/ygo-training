"""Append-only training metrics and human-readable per-checkpoint status."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def append(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf8") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")


def read(path: Path) -> list[dict[str, Any]]:
    """Read append-only records, merging later patches for the same version.

    Standalone checkpoint evaluation adds a small record containing only the
    new evaluation.  Keeping the JSONL append-only avoids rewriting a live
    training log while still letting ``status`` show one coherent row.
    """
    merged: dict[int, dict[str, Any]] = {}
    order: list[int] = []
    for line in path.read_text(encoding="utf8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        version = int(record["version"])
        if version not in merged:
            merged[version] = record
            order.append(version)
        else:
            merged[version].update(record)
    return [merged[version] for version in order]


def evaluation_for(record: dict[str, Any]) -> dict[str, Any]:
    """The evaluation attached to a record: its suite's primary case."""
    return record.get("evaluation") or {}


def evaluation_summary_for(record: dict[str, Any]) -> dict[str, Any]:
    """Return the pooled suite result when available, otherwise the primary evaluation."""
    suite = record.get("evaluation_suite")
    if not isinstance(suite, dict):
        return evaluation_for(record)

    completed = []
    for result in suite.values():
        if not isinstance(result, dict):
            continue
        total = sum(int(result.get(key, 0)) for key in ("wins", "losses", "draws"))
        if total:
            completed.append(result)
    if not completed:
        return evaluation_for(record)

    summary = {
        key: sum(int(result.get(key, 0)) for result in completed)
        for key in (
            "wins",
            "losses",
            "draws",
            "cutoff_draws",
            "deckout_losses",
            "deckout_wins",
        )
    }
    summary["games"] = summary["wins"] + summary["losses"] + summary["draws"]
    summary["win_rate"] = summary["wins"] / summary["games"]
    return summary


def format_status(records: list[dict[str, Any]], verbose: bool = False) -> str:
    if not records:
        return "No completed training updates found."
    header = "version  samples  train-win  reward  eval-win  eval W-L-D  cut E %  cut train T/S  roll   opt   eval  total   eval DO L/W"
    lines = [header]
    for record in records:
        evaluation = evaluation_summary_for(record)
        has_evaluation = bool(evaluation)
        evaluation_deckout_losses = evaluation.get("deckout_losses", 0)
        evaluation_deckout_wins = evaluation.get("deckout_wins", 0)

        def duration(name: str) -> str:
            value = record.get(name)
            return "  - " if value is None else f"{value:>4.0f}s"

        def reward() -> str:
            value = record.get("learner_total_reward")
            return "   -  " if value is None else f"{value:>7.1f}"

        def evaluation_win_rate() -> str:
            return f"{evaluation.get('win_rate', 0.0):>7.1%}" if has_evaluation else "   -  "

        def evaluation_record() -> str:
            if not has_evaluation:
                return "   -     "
            return f"{evaluation.get('wins', 0)}-{evaluation.get('losses', 0)}-{evaluation.get('draws', 0):<3}"

        def evaluation_cutoffs() -> str:
            if not has_evaluation:
                return "      -"
            games = int(evaluation.get("games", 0))
            rate = evaluation.get("cutoff_draws", 0) / games if games else 0.0
            return f"{rate:>7.1%}"

        def training_cutoffs() -> str:
            trajectories = int(record.get("cutoff_loss_trajectories", 0))
            steps = int(record.get("cutoff_loss_steps", 0))
            return f"{trajectories}/{steps:<11}"

        def evaluation_deckouts() -> str:
            if not has_evaluation:
                return "-"
            return f"{int(evaluation_deckout_losses)}/{int(evaluation_deckout_wins)}"

        lines.append(
            f"{record['version']:>7}  {int(record.get('samples', 0)):>7}  "
            f"{record.get('learner_win_rate', 0.0):>8.1%}  "
            f"{reward()}  "
            f"{evaluation_win_rate()}  "
            f"{evaluation_record()}  "
            f"{evaluation_cutoffs()}  "
            f"{training_cutoffs()}  "
            f"{duration('seconds')}  {duration('ppo_update_seconds')}  {duration('evaluation_seconds')}  {duration('wall_seconds')}  "
            f"{evaluation_deckouts():<12}"
        )
    return "\n".join(lines)


def print_rich_status(records: list[dict[str, Any]], verbose: bool = False) -> None:
    """Render an adaptive terminal table for interactive ``status`` calls."""
    from rich.console import Console
    from rich.table import Table

    console = Console()
    if not records:
        console.print("No completed training updates found.")
        return
    columns = (
        ("version", "right", "crop"),
        ("samples", "right", "crop"),
        ("train-win", "right", "crop"),
        ("reward", "right", "crop"),
        ("eval-win", "right", "crop"),
        ("eval W-L-D", "right", "crop"),
        ("cut E %", "right", "crop"),
        ("cut train T/S", "right", "crop"),
        ("roll", "right", "crop"),
        ("opt", "right", "crop"),
        ("eval", "right", "crop"),
        ("total", "right", "crop"),
        ("eval DO L/W", "right", "crop"),
    )
    # A full status row needs roughly 220 columns.  Below that, preserve the
    # learning and anti-stall signals rather than letting every heading shrink
    # into unreadable fragments.
    compact = console.width < 150
    medium = not compact and console.width < 220
    if compact:
        visible = (0, 2, 3, 5, 6, 7)
        displayed_columns = [
            ("ver", "right", "crop"),
            ("train", "right", "crop"),
            ("reward", "right", "crop"),
            ("eval W-L-D", "right", "crop"),
            ("cut E %", "right", "crop"),
            ("cut T/S", "right", "crop"),
        ]
    elif medium:
        visible = tuple(range(12))
        displayed_columns = [
            ("ver", "right", "crop"),
            ("samples", "right", "crop"),
            ("train", "right", "crop"),
            ("reward", "right", "crop"),
            ("eval", "right", "crop"),
            ("eval W-L-D", "right", "crop"),
            ("cut E %", "right", "crop"),
            ("cut T/S", "right", "crop"),
            ("roll", "right", "crop"),
            ("opt", "right", "crop"),
            ("eval s", "right", "crop"),
            ("total", "right", "crop"),
        ]
    else:
        visible = tuple(range(len(columns)))
        displayed_columns = [columns[index] for index in visible]
    table = Table(box=None, pad_edge=False, show_edge=False, expand=not compact)
    for title, justify, overflow in displayed_columns:
        table.add_column(title, justify=justify, no_wrap=True, overflow=overflow)

    for record in records:
        evaluation = evaluation_summary_for(record)
        has_evaluation = bool(evaluation)

        def duration(name: str) -> str:
            value = record.get(name)
            return "-" if value is None else f"{value:.0f}s"

        evaluation_seconds = duration("evaluation_seconds")
        cells = (
            str(record["version"]),
            str(int(record.get("samples", 0))),
            f"{record.get('learner_win_rate', 0.0):.1%}",
            "-"
            if record.get("learner_total_reward") is None
            else f"{record['learner_total_reward']:.1f}",
            f"{evaluation.get('win_rate', 0.0):.1%}" if has_evaluation else "-",
            (
                f"{evaluation.get('wins', 0)}-{evaluation.get('losses', 0)}-{evaluation.get('draws', 0)}"
                if has_evaluation
                else "-"
            ),
            (
                f"{evaluation.get('cutoff_draws', 0) / evaluation.get('games', 1):.1%}"
                if has_evaluation and evaluation.get("games", 0)
                else "-"
            ),
            f"{int(record.get('cutoff_loss_trajectories', 0))}/{int(record.get('cutoff_loss_steps', 0))}",
            duration("seconds"),
            duration("ppo_update_seconds"),
            evaluation_seconds,
            duration("wall_seconds"),
            (
                f"{int(evaluation.get('deckout_losses', 0))}/{int(evaluation.get('deckout_wins', 0))}"
                if has_evaluation
                else "-"
            ),
        )
        table.add_row(*(cells[index] for index in visible))
    console.print(table)


def _evaluation_suite_rows(
    record: dict[str, Any],
) -> list[tuple[str, int, int, int]]:
    suite = record.get("evaluation_suite")
    if not isinstance(suite, dict):
        return []
    rows = []
    for matchup, result in suite.items():
        if not isinstance(result, dict):
            continue
        wins = int(result.get("wins", 0))
        losses = int(result.get("losses", 0))
        draws = int(result.get("draws", 0))
        if wins + losses + draws:
            rows.append((str(matchup), wins, losses, draws))
    return rows


def format_evaluation_suite_status(records: list[dict[str, Any]]) -> str:
    """Render pooled and per-matchup evaluation-suite results."""
    rows: list[tuple[str, ...]] = []
    for record in records:
        matchups = _evaluation_suite_rows(record)
        if not matchups:
            continue
        wins = sum(row[1] for row in matchups)
        losses = sum(row[2] for row in matchups)
        draws = sum(row[3] for row in matchups)
        version = str(record["version"])
        for matchup, row_wins, row_losses, row_draws in (
            ("aggregate", wins, losses, draws),
            *matchups,
        ):
            total = row_wins + row_losses + row_draws
            rows.append(
                (
                    version,
                    matchup,
                    str(row_wins),
                    str(row_losses),
                    str(row_draws),
                    f"{row_wins / total:.1%}",
                )
            )
    if not rows:
        return "No completed evaluation-suite records found."

    headings = ("version", "matchup", "wins", "losses", "draws", "win-rate")
    widths = tuple(
        max(len(heading), *(len(row[index]) for row in rows))
        for index, heading in enumerate(headings)
    )

    def render(row: tuple[str, ...]) -> str:
        return "  ".join(
            cell.ljust(width) if index == 1 else cell.rjust(width)
            for index, (cell, width) in enumerate(zip(row, widths))
        )

    return "\n".join((render(headings), *(render(row) for row in rows)))


def print_evaluation_suite_status(records: list[dict[str, Any]]) -> None:
    """Render pooled and per-matchup evaluation-suite results with Rich."""
    from rich.console import Console
    from rich.table import Table

    console = Console()
    completed = [
        (record, rows)
        for record in records
        if (rows := _evaluation_suite_rows(record))
    ]
    if not completed:
        console.print("No completed evaluation-suite records found.")
        return

    table = Table(box=None, pad_edge=False, show_edge=False)
    for heading in ("version", "matchup", "wins", "losses", "draws", "win-rate"):
        table.add_column(heading, justify="left" if heading == "matchup" else "right")
    for record, matchups in completed:
        wins = sum(row[1] for row in matchups)
        losses = sum(row[2] for row in matchups)
        draws = sum(row[3] for row in matchups)
        version = str(record["version"])
        for matchup, row_wins, row_losses, row_draws in (
            ("aggregate", wins, losses, draws),
            *matchups,
        ):
            total = row_wins + row_losses + row_draws
            table.add_row(
                version,
                matchup,
                str(row_wins),
                str(row_losses),
                str(row_draws),
                f"{row_wins / total:.1%}",
            )
    console.print(table)


OPTIMIZATION_COLUMNS = (
    "version",
    "phase",
    "samples",
    "train-win",
    "eval-win",
    "value EV",
    "approx KL",
    "final KL",
    "cum KL",
    "epochs",
    "clip",
    "entropy",
    "value loss",
    "imit loss",
    "teacher agree",
    "lead loss",
    "lead EV",
)


def _optional(record: dict[str, Any], key: str, spec: str) -> str:
    value = record.get(key)
    return "-" if value is None else format(float(value), spec)


def _optimization_cells(
    record: dict[str, Any], cumulative_kl: float | None
) -> tuple[str, ...]:
    """PPO health per update; imitation-only updates show their imitation metrics."""
    evaluation = evaluation_summary_for(record)
    imitation = bool(record.get("warmup"))
    return (
        str(record["version"]),
        "imitation" if imitation else "ppo",
        str(int(record.get("samples", 0))),
        _optional(record, "learner_win_rate", ".1%"),
        f"{evaluation.get('win_rate', 0.0):.1%}" if evaluation else "-",
        _optional(
            record,
            "imitation_value_explained_variance" if imitation else "duel_explained_variance",
            ".3f",
        ),
        "-" if imitation else _optional(record, "duel_approximate_kl", ".4f"),
        "-" if imitation else _optional(record, "duel_final_approximate_kl", ".4f"),
        "-" if imitation or cumulative_kl is None else f"{cumulative_kl:.3f}",
        "-" if imitation else _optional(record, "ppo_epochs_completed", ".2f"),
        "-" if imitation else _optional(record, "duel_clip_fraction", ".3f"),
        "-" if imitation else _optional(record, "duel_entropy", ".3f"),
        _optional(record, "imitation_value_loss" if imitation else "duel_value", ".3f"),
        _optional(record, "imitation_loss", ".3f") if imitation else "-",
        _optional(
            record,
            "imitation_model_teacher_agreement" if imitation else "teacher_argmax_agreement",
            ".1%",
        ),
        # The lead head against a search's measurements, where the update's
        # duels carried them; like the value head's, measured before it.
        _optional(record, "imitation_lead_loss", ".4f"),
        _optional(record, "imitation_lead_explained_variance", ".3f"),
    )


def _optimization_rows(records: list[dict[str, Any]]) -> list[tuple[str, ...]]:
    """Cells per record, with end-of-update KL summed over PPO updates.

    The running total is the distance-travelled axis for comparing runs whose
    step sizes differ. Runs recorded before end-of-update KL existed show
    ``-`` rather than a total built from the epoch-averaged KL.
    """
    rows = []
    cumulative: float | None = 0.0
    for record in records:
        if not record.get("warmup"):
            final = record.get("duel_final_approximate_kl")
            cumulative = None if final is None or cumulative is None else cumulative + final
        rows.append(_optimization_cells(record, cumulative))
    return rows


def format_optimization_status(records: list[dict[str, Any]]) -> str:
    """Render per-update optimization health for redirected output."""
    if not records:
        return "No completed training updates found."
    rows = [OPTIMIZATION_COLUMNS, *_optimization_rows(records)]
    widths = [max(len(row[column]) for row in rows) for column in range(len(rows[0]))]
    return "\n".join(
        "  ".join(f"{cell:>{width}}" for cell, width in zip(row, widths)) for row in rows
    )


def print_optimization_status(records: list[dict[str, Any]]) -> None:
    """Render an adaptive Rich table of per-update optimization health."""
    from rich.console import Console
    from rich.table import Table

    console = Console()
    if not records:
        console.print("No completed training updates found.")
        return
    table = Table(box=None, pad_edge=False, show_edge=False, expand=True)
    for heading in OPTIMIZATION_COLUMNS:
        table.add_column(heading, justify="right", no_wrap=True, overflow="crop")
    for row in _optimization_rows(records):
        table.add_row(*row)
    console.print(table)


def _percent(value: float | None) -> str:
    return "-" if value is None else f"{100.0 * value:.1f}%"


def _points(value: float | None, error: float | None = None) -> str:
    """A difference of win rates, in points."""
    if value is None:
        return "-"
    return f"{100 * value:+.1f}" + ("" if error is None else f" ± {100 * error:.1f}")


def _tried_lines(tried: dict[str, Any]) -> list[str]:
    """A model's answers by what the search measured for them."""

    def row(name: str, part: dict[str, Any]) -> list[str]:
        others = part["decisions"] - part["pilot"]
        return [
            name,
            str(others),
            f"{others / tried['games']:.2f}",
            str(part["outright"]),
            str(part["untried"]),
            str(part["tried"]),
            str(part["ahead"]),
            str(part["level"]),
            str(part["behind"]),
            _points(part["lead"], part["lead_error"]),
            _points(part["lead_per_game"], part["lead_per_game_error"]),
        ]

    rows = [
        [
            f"{tried['decisions']} examined",
            "others",
            "a game",
            "outright",
            "untried",
            "tried",
            "ahead",
            "level",
            "behind",
            "lead",
            "sum a game",
        ],
        row("first choice", tried),
        *(row(f"at {bar['ratio']:g} of the pilot's", bar) for bar in tried["bars"][1:]),
        row("the search", tried["search"]),
    ]
    widths = [max(len(line[column]) for line in rows) for column in range(len(rows[0]))]
    lines = [
        "  "
        + line[0].ljust(widths[0])
        + "".join("  " + cell.rjust(width) for cell, width in zip(line[1:], widths[1:]))
        for line in rows
    ]
    other = tried["any_other"]
    lines.append(
        f"  any other answer the search tried ({other['tried']}): "
        f"{_points(other['lead'], other['lead_error'])}"
    )
    if tried.get("tried_outright"):
        lines.append(
            f"  of the first choices tried, {tried['tried_outright']} where the pilot's "
            f"answer wins outright: {_points(tried['lead_outright_per_game'])} a game"
        )
    if tried.get("led"):
        lines.extend(_led_lines(tried))
    if tried.get("value"):
        lines.extend(_value_lines(tried["value"]))
    return lines


def _share(value: float | None) -> str:
    return "-" if value is None else f"{100.0 * value:.1f}%"


def _value_lines(value: dict[str, Any]) -> list[str]:
    """What a model's value head tells of the decisions a search examined."""
    correlation = "-" if value["correlation"] is None else f"{value['correlation']:+.3f}"
    told, result = value["told"], value["result"]
    of_a_result = None if told is None or result is None else told * result
    lines = [
        (
            f"  the value head at {value['decisions']} examined decisions that open a "
            f"transaction ({value['elsewhere']} others left out):"
        ),
        (
            "    its values and what the pilot's answer wins of its worlds correlate at "
            f"{correlation}; of what can be told there it tells {_share(told)} "
            f"({_share(value['explained'])} as its values stand)"
        ),
        (
            f"    of one result a perfect predictor explains {_share(result)} there, "
            f"this head {_share(of_a_result)}"
        ),
    ]
    if value.get("by_turn"):
        lines.append(
            "    by turn, what it tells of what can be told: "
            + ", ".join(
                f"{part['turns'][0]}"
                + ("+" if part["turns"][1] is None else f"-{part['turns'][1]}")
                + f" {_share(part['told'])} of {_share(part['result'])}"
                for part in value["by_turn"]
            )
        )
    return lines


def _led_lines(tried: dict[str, Any]) -> list[str]:
    """A model's answers when it follows its lead head, and what the head knows."""
    rows = [["following the lead head", "changed", "a game", "others", "sum a game", "gain a game"]]
    for part in tried["led"]:
        rows.append(
            [
                f"by {100 * part['margin']:g} points",
                str(part["changed"]),
                f"{part['changed'] / tried['games']:.2f}",
                f"{(part['decisions'] - part['pilot']) / tried['games']:.2f}",
                _points(part["lead_per_game"], part["lead_per_game_error"]),
                _points(part["gain_per_game"], part["gain_per_game_error"]),
            ]
        )
    widths = [max(len(line[column]) for line in rows) for column in range(len(rows[0]))]
    lines = [
        "  "
        + line[0].ljust(widths[0])
        + "".join("  " + cell.rjust(width) for cell, width in zip(line[1:], widths[1:]))
        for line in rows
    ]
    if all("split" in part for part in tried["led"]):
        # Where the changes start: how many, and what they gain a game.
        rows = [["the changes apart", "from the pilot's answer", "back to it", "between others"]]
        for part in tried["led"]:
            row = [f"by {100 * part['margin']:g} points"]
            for start in ("from_pilot", "to_pilot", "between_others"):
                group = part["split"][start]
                gain = _points(group["gain_per_game"], group["gain_per_game_error"])
                row.append(f"{group['changed']}: {gain}")
            rows.append(row)
        widths = [max(len(line[column]) for line in rows) for column in range(len(rows[0]))]
        lines.extend(
            "  "
            + line[0].ljust(widths[0])
            + "".join("  " + cell.rjust(width) for cell, width in zip(line[1:], widths[1:]))
            for line in rows
        )
    head = tried.get("head")
    if head:
        correlation = "-" if head["correlation"] is None else f"{head['correlation']:+.3f}"
        lines.append(
            f"  the head over {head['alternatives']} alternatives: predicted and measured "
            f"leads correlate at {correlation}; those it ranks highest lead by "
            + ", ".join(
                f"{_points(part['lead'], part['lead_error'])} ({part['share']:.1%})"
                for part in head["highest"]
            )
        )
    return lines


def format_search_evaluation(record: dict[str, Any]) -> str:
    """One checkpoint's `evaluate-search` result, for reading at a terminal."""
    deals = record["deals"]
    key = record["answer_key"]
    references = ", ".join(
        f"{name} {_percent(reference['win_rate'])}"
        for name, reference in deals["references"].items()
    )
    examined = "examined_kept" in key
    header = ["decisions", "deviations", "search's", "pilot's", "other", "kept", "another"]
    if examined:
        header += ["examined", "another"]

    def cells(name: str, part: dict[str, Any]) -> list[str]:
        row = [
            name,
            str(part["deviations"]),
            _percent(part["search_rate"]),
            _percent(part["pilot_rate"]),
            _percent(part["other"] / part["deviations"] if part["deviations"] else None),
            str(part["kept"]),
            _percent(part["kept_left_rate"]),
        ]
        if examined:
            row += [str(part["examined_kept"]), _percent(part["examined_kept_left_rate"])]
        return row

    rows = [header, cells("all", key)]
    rows.extend(cells(question, part) for question, part in key["by_question"].items())
    widths = [max(len(row[column]) for row in rows) for column in range(len(header))]
    lines = [
        (
            f"{record['checkpoint_label']}: {deals['wins']}/{deals['games']} "
            f"({_percent(deals['win_rate'])}) on the run's deals; {references}"
        ),
    ]
    if record.get("seen_deals"):
        lines.append(f"  {record['seen_deals']} of these deals were imitated in training")
    led = record.get("led_deals")
    if led:
        lines.append(
            f"  following its lead head by {100 * led['margin']:g} points: {led['wins']}/"
            f"{led['games']} ({_percent(led['win_rate'])}), {led['changed']} first choices left"
        )
    lines.extend(
        "  "
        + row[0].ljust(widths[0])
        + "".join("  " + cell.rjust(width) for cell, width in zip(row[1:], widths[1:]))
        for row in rows
    )
    if key.get("tried"):
        lines.extend(_tried_lines(key["tried"]))
    return "\n".join(lines)
