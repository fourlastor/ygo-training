"""The teachers of the network of the seventy-one decks
(experiments/2026-10-28-seventy-one-champions.md), held to the record before
its copy is made: each deck's recording made with the run its line names, by
the fingerprint of the run's weights, and no run under `all71` in its mirror
by two standard errors, by the figure of its making
(`CHAMPIONS/DECK/report.json`, tools/champions_report.py), for the record
has such a deck taught by `all71` instead. Only the decks of the champions'
release made before that report (RELEASE) have no figure there.

    uv run python tools/seventy_one_teachers.py experiments/commands/champion-recordings.list \\
        --records ../records --champions analysis/champions

Each line of the list names a deck and its run's checkpoint. Each deck is
told with its figure; the first that does not hold ends it in error.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ygo_training.checkpoints import load_inference_model, value_fingerprint

ERRORS = 2.0
EVEN = 0.50
# The decks of the champions' release made in experiments of their own.
RELEASE = ("blackwing", "cyber", "lightsworn", "infernity")


def listed(path: Path) -> list[tuple[str, Path]]:
    """The decks of a list and their runs' checkpoints."""
    found = []
    for line in path.read_text(encoding="utf8").splitlines():
        if line.strip() and not line.startswith("#"):
            deck, checkpoint = line.split()[:2]
            found.append((deck, Path(checkpoint)))
    return found


def under(figure: dict[str, Any]) -> bool:
    """A run at most even against `all71` in its mirror by two standard
    errors."""
    return figure["share"] + ERRORS * figure["error"] <= EVEN


def checked(
    decks: list[tuple[str, Path]], records: Path, champions: Path, release: tuple[str, ...]
) -> list[str]:
    """Each deck told with its run's figure; the first that does not hold
    ends it in error."""
    told = []
    for deck, checkpoint in decks:
        about = records / deck / "about.json"
        if not about.is_file():
            raise SystemExit(f"No recording of {deck} in {records}")
        if not checkpoint.is_file():
            raise SystemExit(f"Missing {checkpoint}, {deck}'s run")
        recorded = json.loads(about.read_text(encoding="utf8"))["teacher"]["fingerprint"]
        model, _ = load_inference_model(checkpoint, device="cpu")
        if value_fingerprint(model.state_dict()) != recorded:
            raise SystemExit(f"{deck} was recorded with another network than {checkpoint}")
        report = champions / deck / "report.json"
        if not report.is_file():
            if deck not in release:
                raise SystemExit(f"Missing {report}: {deck}'s run is told by its figure")
            told.append(f"{deck}: of the champions' release")
            continue
        figure = json.loads(report.read_text(encoding="utf8"))[deck]
        if under(figure):
            raise SystemExit(
                f"{deck}'s run is under all71 in its mirror, {figure['share']:.1%} ± "
                f"{figure['error']:.1%}: the record has all71 teach it, its games recorded the "
                "same way"
            )
        told.append(
            f"{deck}: {figure['share']:.1%} ± {figure['error']:.1%} against all71 in its "
            f"mirror, {figure['verdict']}"
        )
    return told


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("list", type=Path, help="a deck and its run's checkpoint a line")
    parser.add_argument("--records", type=Path, required=True, help="a folder each deck")
    parser.add_argument("--champions", type=Path, required=True, help="CHAMPIONS/DECK/report.json")
    parser.add_argument(
        "--release", default=",".join(RELEASE), help="the decks that have no figure there"
    )
    args = parser.parse_args()
    decks = listed(args.list)
    for line in checked(decks, args.records, args.champions, tuple(args.release.split(","))):
        print(line)
    print(f"The {len(decks)} decks were recorded with their runs, none of them under all71.")


if __name__ == "__main__":
    main()
