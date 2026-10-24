"""The cards' descriptions the model reads from format 16: a fixed vector of
what each card's text says, made once and kept in the repository.

    uv run python tools/card_descriptions.py \\
        analysis/card-descriptions/first-look/texts.json \\
        analysis/card-descriptions/first-look/voyage-4-nano-text.npy \\
        data/card-descriptions.npz

The texts and voyage-4-nano's vectors of them are those of
``experiments/commands/card-embeddings.sh``, in the order of the texts. A
text's description is what that check measured as "words + voyage-4-nano":
the words of the texts brought down to their 128 main directions beside
voyage's 128, each of length one; then each of the 256 numbers is set over
the texts to a mean of 0 and a spread of 1, as the network's other inputs
are near.

Written to the file:

- ``codes``: every code of the card database, in order;
- ``rows``: the row of each code's description, 0 where it has none: a
  Normal Monster, whose text is a story, a card outside the policies' pool,
  a token. An alternate artwork has its original's row, in the pool or not
  (decks hold some the pool does not list), and cards whose texts read the
  same share one;
- ``descriptions``: the rows, as 16-bit floats, the first all zeros;
- ``mean`` and ``spread``: what each number was set from, so that the
  vectors as the check measured them are a row times the spread plus the
  mean;
- ``about``: what they were made from, as JSON.

The model looks a card's description up by the code the input already
carries (``ygo_training.card_table``): no array of the input changes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from card_embedding_check import (
    DATABASE,
    POOL,
    READ_WIDTH,
    alternate_of,
    database_cards,
    main_directions,
    normalized,
    pool_codes,
    said_of,
    words,
)
from card_text_embeddings import MODEL, REVISION

ROOT = Path(__file__).resolve().parents[1]


def revision(folder: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(folder), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()


def made(texts_file: Path, voyage_file: Path) -> dict[str, np.ndarray]:
    cards = json.loads(texts_file.read_text())
    texts = [card["text"] for card in cards]
    voyage = np.load(voyage_file)
    if len(voyage) != len(texts):
        raise SystemExit(f"{voyage_file} has {len(voyage)} rows for {len(texts)} texts")
    described = np.hstack(
        [
            normalized(words(texts)),
            normalized(main_directions(voyage.astype(np.float64), READ_WIDTH)),
        ]
    )
    centre, spread = described.mean(axis=0), described.std(axis=0)
    described = (described - centre) / spread
    database = database_cards()
    codes = np.array(sorted(database), dtype=np.int64)
    pool = pool_codes()
    member = set(pool)
    place = {text: index + 1 for index, text in enumerate(texts)}
    of_code: dict[int, int] = {}
    for code in pool:
        original = alternate_of(code, database, member)
        said = said_of(database[code if original is None else original])
        if said is not None:
            if said not in place:
                raise SystemExit(
                    f"{code} {database[code][3]}: its text is not among {texts_file}'s"
                )
            of_code[code] = place[said]
    # An alternate artwork a deck holds that the pool does not list.
    for code, (_, alias, _, name, _) in database.items():
        if code not in of_code and alias in of_code and database[alias][3] == name:
            of_code[code] = of_code[alias]
    rows = np.array([of_code.get(int(code), 0) for code in codes], dtype=np.int32)
    descriptions = np.vstack([np.zeros((1, described.shape[1])), described]).astype(np.float16)
    about = {
        "made_by": "tools/card_descriptions.py",
        "texts": {
            "file": str(texts_file),
            "count": len(texts),
            "sha256": hashlib.sha256(texts_file.read_bytes()).hexdigest(),
        },
        "voyage": {
            "model": MODEL,
            "revision": REVISION,
            "sha256": hashlib.sha256(voyage_file.read_bytes()).hexdigest(),
        },
        "pool": str(POOL.relative_to(ROOT)),
        "database": {
            "file": str(DATABASE.relative_to(ROOT)),
            "revision": revision(DATABASE.parent),
        },
        "width": int(described.shape[1]),
        "rows": "words of the texts at 128 main directions and voyage's 128, each of length "
        "one, every number then set to a mean of 0 and a spread of 1 over the texts",
    }
    return {
        "codes": codes,
        "rows": rows,
        "descriptions": descriptions,
        "mean": centre.astype(np.float32),
        "spread": spread.astype(np.float32),
        "about": np.array(json.dumps(about, indent=1)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("texts", type=Path, help="the texts of card_embedding_check.py texts")
    parser.add_argument("voyage", type=Path, help="voyage-4-nano's vectors of those texts")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    table = made(args.texts, args.voyage)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **table)
    described = int((table["rows"] > 0).sum())
    print(
        f"{len(table['codes'])} codes, {described} with one of {len(table['descriptions']) - 1} "
        f"descriptions of {table['descriptions'].shape[1]} numbers: {args.output} "
        f"({args.output.stat().st_size / 1e6:.1f} MB)"
    )


if __name__ == "__main__":
    sys.exit(main())
