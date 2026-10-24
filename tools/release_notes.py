"""A network's card as its release's notes (`.github/workflows/model-release.yml`):
the card without its title, its links into the repository pinned to the
tagged commit, and each paragraph and list item on one line, as release
notes break a line wherever their text does.

    python3 tools/release_notes.py models/all71.md \\
        --repository fourlastor/ygo-training --commit COMMIT > notes.md

The standard library alone: the workflow runs it without the trainer's
environment.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

# A line that begins a block of its own rather than going on with the line
# before it.
BLOCK = re.compile(r"(#{1,6} |[-*] |\d+\. |\||```)")


def pinned(text: str, repository: str, commit: str) -> str:
    """The card's links into the repository, from `models/`, as links to the
    commit's files."""
    base = f"https://github.com/{repository}/blob/{commit}"
    text = re.sub(r"\]\(\.\./", f"]({base}/", text)
    return re.sub(r"\]\(([a-z0-9-]+\.md)\)", rf"]({base}/models/\1)", text)


def unwrapped(text: str) -> str:
    """Each paragraph and list item on one line."""
    lines: list[str] = []
    for line in text.split("\n"):
        told = line.strip()
        if lines and lines[-1].strip() and told and not BLOCK.match(told):
            lines[-1] = f"{lines[-1].rstrip()} {told}"
        else:
            lines.append(line)
    return "\n".join(lines)


def notes(card: str, repository: str, commit: str) -> str:
    _, _, body = card.partition("\n")
    return unwrapped(pinned(body.strip("\n"), repository, commit)) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("card", type=Path)
    parser.add_argument("--repository", required=True, help="OWNER/NAME")
    parser.add_argument("--commit", required=True)
    args = parser.parse_args()
    print(notes(args.card.read_text(encoding="utf8"), args.repository, args.commit), end="")


if __name__ == "__main__":
    main()
