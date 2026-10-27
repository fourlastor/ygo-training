"""Networks of one shape answering their decisions in one call: their
weights stacked (`torch.func`), each network's decisions laid out beside the
others' and filled out with zeros to as many as the most of them.

A copy's teachers, one network for each deck, answer a round of their decks'
decisions so (`pool.taught_games`) in place of a call for each deck.

`torch.func.vmap` does not take `torch.nn.LSTM`: the history's LSTM is
written out from its equations here, and its sums are not worked out in the
order PyTorch's own kernels work them out, so the answers are the networks'
own only to their last digits.
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.func import functional_call, stack_module_state, vmap


def lstm(steps: torch.Tensor, cells: torch.nn.LSTM) -> torch.Tensor:
    """An LSTM of one layer over every step of every row, from zero states,
    as `torch.nn.LSTM` works it out: its output at each step."""
    inputs = F.linear(steps, cells.weight_ih_l0, cells.bias_ih_l0)
    hidden = steps.new_zeros((steps.shape[0], cells.hidden_size))
    cell = steps.new_zeros((steps.shape[0], cells.hidden_size))
    outputs = []
    for step in range(steps.shape[1]):
        gates = inputs[:, step] + F.linear(hidden, cells.weight_hh_l0, cells.bias_hh_l0)
        enter, forget, cell_gate, out = gates.chunk(4, -1)
        cell = torch.sigmoid(forget) * cell + torch.sigmoid(enter) * torch.tanh(cell_gate)
        hidden = torch.sigmoid(out) * torch.tanh(cell)
        outputs.append(hidden)
    return torch.stack(outputs, 1)


class Stacked:
    """The networks ``models``, by deck, answering together on ``device``."""

    def __init__(self, models: dict[str, torch.nn.Module], device: str) -> None:
        if not models:
            raise ValueError("Networks stacked are one at least")
        self.decks = sorted(models)
        self.place = {deck: place for place, deck in enumerate(self.decks)}
        self.device = str(device)
        params, buffers = stack_module_state([models[deck] for deck in self.decks])
        self.params = {name: value.detach() for name, value in params.items()}
        self.buffers = buffers
        # The shape the stacked weights are put in, holding none of its own.
        base = copy.deepcopy(models[self.decks[0]]).to("meta")
        if base.history.num_layers != 1 or base.history.bidirectional:
            raise ValueError("The history's LSTM written out is of one layer and one way")

        def history(tensor: dict[str, Any], described: Any = None) -> Any:
            steps = base._history_steps(tensor, described)
            return base._history_read(lstm(steps, base.history), tensor["history_lengths"])

        base._history = history

        def one(params: dict[str, Any], buffers: dict[str, Any], tensors: dict[str, Any]) -> Any:
            return functional_call(base, (params, buffers), (tensors,))

        self.call = vmap(one)

    def answer(self, arrays: dict[str, np.ndarray], decks: np.ndarray) -> torch.Tensor:
        """The logits of each decision of ``arrays`` by the network of its
        deck (``decks``, one for each), in their order."""
        rows = {deck: np.flatnonzero(decks == deck) for deck in sorted(set(decks.tolist()))}
        missing = [deck for deck in rows if deck not in self.place]
        if missing:
            raise ValueError(f"No network stacked plays {', '.join(missing)}")
        most = max(len(found) for found in rows.values())
        laid = {}
        for name, array in arrays.items():
            whole = np.zeros((len(self.decks), most, *array.shape[1:]), dtype=array.dtype)
            for deck, found in rows.items():
                whole[self.place[deck], : len(found)] = array[found]
            laid[name] = torch.from_numpy(whole).to(self.device)
        with torch.no_grad():
            logits, _ = self.call(self.params, self.buffers, laid)
        answered = torch.empty(
            (len(decks), logits.shape[-1]), dtype=logits.dtype, device=logits.device
        )
        for deck, found in rows.items():
            places = torch.from_numpy(found).to(logits.device)
            answered[places] = logits[self.place[deck], : len(found)]
        return answered
