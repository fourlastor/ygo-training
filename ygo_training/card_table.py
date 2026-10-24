"""What a model is told of a card beside its input, from checkpoint format 16.

A card of the input is its code (``schema``, the ``*card`` arrays), which the
model reads as an identity learnt from the games the card was in
(``model.CardIdentityEmbedding``). A card table is the codes of the card
database in order, and with them the row of each code's description: a
fixed vector of what the card's text says, the same for a card trained on
and one never seen (``tools/card_descriptions.py`` makes them,
``data/card-descriptions.npz`` keeps them). A model told the descriptions
looks each card's up by its code; a model that hides identities hides them
by the same codes and keeps which it was trained on (``PolicyValueNet``).
No array of the input changes, and a checkpoint carries its table, so that
a card means to the network what it meant when it was trained.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

# The repository's table (`tools/card_descriptions.py`).
DESCRIPTIONS = Path(__file__).resolve().parents[1] / "data" / "card-descriptions.npz"
# What a model keeps of its table, by the names of its state.
CODES = "card_codes"
ROWS = "card_description_rows"
TABLE = "card_descriptions"


@dataclass(frozen=True)
class CardTable:
    """Codes in order, each once; for each the row of its description, 0
    where it has none; and the rows, the first all zeros. The rows are none
    for the table of a model that is not told them."""

    codes: torch.Tensor
    rows: torch.Tensor | None = None
    descriptions: torch.Tensor | None = None

    def __post_init__(self) -> None:
        codes = self.codes
        if codes.dtype != torch.int64 or codes.ndim != 1 or not len(codes):
            raise ValueError("A card table's codes are a row of 64-bit integers")
        if len(codes) > 1 and not bool((codes[1:] > codes[:-1]).all()):
            raise ValueError("A card table's codes are in order, each once")
        if (self.rows is None) != (self.descriptions is None):
            raise ValueError("A card table has both rows and descriptions, or neither")
        if self.rows is not None and self.descriptions is not None:
            if self.rows.dtype != torch.int64 or self.rows.shape != codes.shape:
                raise ValueError("A card table has a row of 64-bit integers a code")
            if self.descriptions.ndim != 2 or self.descriptions.dtype != torch.float32:
                raise ValueError("A card table's descriptions are rows of 32-bit floats")
            if bool(self.descriptions[0].any()):
                raise ValueError("A card table's first description is all zeros")
            if int(self.rows.min()) < 0 or int(self.rows.max()) >= len(self.descriptions):
                raise ValueError("A card table's rows are rows of its descriptions")

    @property
    def width(self) -> int:
        """The numbers of a description."""
        if self.descriptions is None:
            raise ValueError("This card table has no descriptions")
        return int(self.descriptions.shape[1])

    @classmethod
    def read(cls, path: Path | str) -> CardTable:
        """The table of a file the descriptions tool wrote (``.npz``), or the
        one a checkpoint (``.pt``) carries."""
        path = Path(path)
        if path.suffix == ".npz":
            with np.load(path) as found:
                return cls(
                    codes=torch.from_numpy(found["codes"].astype(np.int64)),
                    rows=torch.from_numpy(found["rows"].astype(np.int64)),
                    descriptions=torch.from_numpy(found["descriptions"].astype(np.float32)),
                )
        state = torch.load(path, map_location="cpu", weights_only=False)["model"]
        table = cls.of_state(state)
        if table is None:
            raise ValueError(f"{path} carries no card table")
        return table

    @classmethod
    def of_state(cls, state: dict[str, Any]) -> CardTable | None:
        """The table a model's state holds, if it holds one."""
        if CODES not in state:
            return None
        return cls(
            codes=state[CODES].detach().cpu(),
            rows=None if ROWS not in state else state[ROWS].detach().cpu(),
            descriptions=None if TABLE not in state else state[TABLE].detach().cpu(),
        )
