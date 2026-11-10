"""The checks with no training of the screen's second session
(experiments/2026-11-02-expanded-pool.md, its second session).

    uv run python tools/network_screen_checks.py analysis/network-screen/checks

The directory is what `experiments/commands/network-screen-checks.sh` leaves,
in the files of `tools/two_decks_report.py`: `copies/` the screen's two
copies' last checkpoints head to head, the board as its cards in the
learner's seat first; `cyber-tokens-against-cyber-old/` and
`cyber-against-cyber-old/` each run at 720 updates of PPO against
`cyber-old` at 720.

The reading, fixed on 2026-11-10 before anything was played: the edge
**carries to a third network** when `cyber-tokens` wins more against
`cyber-old` than the control does, by two standard errors of the deals
paired, otherwise it is **not shown to carry**. The copies are told without
a mark.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from two_decks_report import deals, paired, told

ERRORS = 2.0


def report(directory: Path) -> dict[str, Any]:
    tokens = deals(directory / "cyber-tokens-against-cyber-old")
    control = deals(directory / "cyber-against-cyber-old")
    gap = paired(tokens, control)
    carries = gap["gap"] - ERRORS * gap["error"] > 0
    return {
        "copies": told(deals(directory / "copies")),
        "cyber-tokens against cyber-old": told(tokens),
        "cyber against cyber-old": told(control),
        "cyber-tokens over cyber against cyber-old": gap,
        "reading": "carries to a third network" if carries else "not shown to carry",
    }


def shown(result: dict[str, Any]) -> str:
    copies = result["copies"]
    tokens = result["cyber-tokens against cyber-old"]
    control = result["cyber against cyber-old"]
    gap = result["cyber-tokens over cyber against cyber-old"]
    return "\n".join(
        [
            f"The edge against a third network: {result['reading']}.",
            f"  cyber-tokens against cyber-old: {tokens['share']:.1%} ± {tokens['error']:.1%}",
            f"  cyber against cyber-old: {control['share']:.1%} ± {control['error']:.1%}",
            (
                f"  the difference, the deals paired: {gap['gap']:+.1%} ± {gap['error']:.1%}"
                f" ({tokens['deals']:,} deals)"
            ),
            (
                "Told without a mark, the copies head to head (240 updates): cyber-tokens-copy "
                f"{copies['share']:.1%} ± {copies['error']:.1%} against cyber-copy"
            ),
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = report(args.directory)
    if args.output is not None:
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf8")
    print(shown(result))


if __name__ == "__main__":
    main()
