"""Networks of one shape answering their decisions in one call: their
weights stacked (`torch.func`), each network's decisions laid out beside the
others' and filled out with zeros to as many as the most of them.

A copy's teachers, one network for each deck, answer a round of their decks'
decisions so (`pool.taught_games`) in place of a call for each deck.

`torch.func.vmap` does not take `torch.nn.LSTM`: the history's LSTM is
written out from its equations here, and its sums are not worked out in the
order PyTorch's own kernels work them out, so the answers are the networks'
own only to their last digits.

With ``largest`` the call is answered from graphs of fixed sizes
(`graphs.Graphs`, CUDA graphs on a card), the networks' decisions laid end to
end, each network's filled out to the next of the counts of decisions a graph
holds, as a single network's call is.
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.func import functional_call, stack_module_state, vmap

from .graphs import Graphs, counts


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


class Laid(torch.nn.Module):
    """Stacked networks as one model of their decisions laid end to end, as
    many of each network's: what a graph of them records."""

    def __init__(self, stacked: Stacked) -> None:
        super().__init__()
        self.stacked = stacked

    def forward(self, tensors: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        networks = len(self.stacked.decks)
        count = next(iter(tensors.values())).shape[0] // networks
        laid = {
            name: tensor.view(networks, count, *tensor.shape[1:])
            for name, tensor in tensors.items()
        }
        logits, values = self.stacked.call(self.stacked.params, self.stacked.buffers, laid)
        return logits.reshape(networks * count, -1), values.reshape(networks * count)


class Stacked:
    """The networks ``models``, by deck, answering together on ``device``,
    from graphs of calls of up to ``largest`` decisions of each where it is
    named."""

    def __init__(
        self, models: dict[str, torch.nn.Module], device: str, *, largest: int = 0
    ) -> None:
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
        # The counts of decisions of each network a graph holds.
        self.counts = counts(largest) if largest else []
        self.graphs: Graphs | None = None
        if largest:
            self.graphs = Graphs(Laid(self), self.device, largest=largest)
            self.graphs.counts = [len(self.decks) * count for count in self.counts]

    def begin_update(self) -> None:
        """An update's games begin (`graphs.Graphs.begin_update`)."""
        if self.graphs is not None:
            self.graphs.begin_update()

    def answer(self, arrays: dict[str, np.ndarray], decks: np.ndarray) -> torch.Tensor:
        """The logits of each decision of ``arrays`` by the network of its
        deck (``decks``, one for each), in their order."""
        rows = {deck: np.flatnonzero(decks == deck) for deck in sorted(set(decks.tolist()))}
        missing = [deck for deck in rows if deck not in self.place]
        if missing:
            raise ValueError(f"No network stacked plays {', '.join(missing)}")
        most = max(len(found) for found in rows.values())
        # From a graph, each network's decisions are as many as a graph holds.
        each = next((count for count in self.counts if count >= most), most)
        networks = len(self.decks)
        laid = {}
        for name, array in arrays.items():
            whole = np.zeros((networks, each, *array.shape[1:]), dtype=array.dtype)
            for deck, found in rows.items():
                whole[self.place[deck], : len(found)] = array[found]
            laid[name] = whole
        made = None
        if self.graphs is not None:
            made = self.graphs.answer(
                {
                    name: whole.reshape(networks * each, *whole.shape[2:])
                    for name, whole in laid.items()
                }
            )
        if made is None:
            tensors = {
                name: torch.from_numpy(whole).to(self.device) for name, whole in laid.items()
            }
            with torch.no_grad():
                logits, _ = self.call(self.params, self.buffers, tensors)
        else:
            # A graph's answers are filled out to its widths: the call's are
            # the first of them.
            width = arrays["action_mask"].shape[-1]
            logits = made[0].reshape(networks, each, -1)[..., :width]
        answered = torch.empty(
            (len(decks), logits.shape[-1]), dtype=logits.dtype, device=logits.device
        )
        for deck, found in rows.items():
            places = torch.from_numpy(found).to(logits.device)
            answered[places] = logits[self.place[deck], : len(found)]
        return answered
