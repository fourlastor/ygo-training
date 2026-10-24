"""`tools/release_notes.py`: a network's card as its release's notes."""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("release_notes", ROOT / "tools/release_notes.py")
assert SPEC is not None and SPEC.loader is not None
tool = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(tool)

CARD = """# tiny: a network

A network of one deck, made
by the recipe.

- **The checkpoint:** format 15,
  after 960 updates.
- **Measured** ([its record](../experiments/record.md)) beside
  [another](other.md).

## The files

- `tiny.pt`: the network.
"""


def test_a_card_becomes_notes_without_its_title_its_lines_joined_its_links_pinned():
    found = tool.notes(CARD, "owner/repo", "abc123")
    base = "https://github.com/owner/repo/blob/abc123"
    assert found == (
        "A network of one deck, made by the recipe.\n"
        "\n"
        "- **The checkpoint:** format 15, after 960 updates.\n"
        f"- **Measured** ([its record]({base}/experiments/record.md)) beside "
        f"[another]({base}/models/other.md).\n"
        "\n"
        "## The files\n"
        "\n"
        "- `tiny.pt`: the network.\n"
    )


def test_every_card_has_a_title_and_notes_without_a_line_broken_mid_paragraph():
    for card in sorted((ROOT / "models").glob("*.md")):
        if card.name == "README.md":
            continue
        text = card.read_text(encoding="utf8")
        assert text.startswith("# "), card.name
        found = tool.notes(text, "owner/repo", "abc123")
        for line in found.splitlines():
            assert not line.startswith("  "), (card.name, line)
        assert "](../" not in found, card.name
