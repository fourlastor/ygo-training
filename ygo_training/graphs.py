"""The model's small calls answered at sizes fixed in advance, from CUDA
graphs on a card.

A call of the model costs about the same whatever it holds: its answer is
some five hundred kernels launched one by one, and most of the calls of an
update's games hold a few decisions (docs/commands.md, "Left to try for
speed").  A CUDA graph is the card's work for one answer, recorded once and
launched again in one call.

A graph is recorded for one set of sizes.  A call answered from one is
therefore filled out with zeros to sizes fixed in advance, as the library
fills a batch out to its longest row: to the next of a few counts of
decisions, and to widths a call is expected to fit (``WIDTHS``).  A call of
more decisions, or wider, is answered as it is.

* A replay gives the model's own numbers at the sizes it was recorded at:
  once an update, each graph's answer is held against the model's to the
  same batch, and a graph that gives another is the end of all of them (the
  calls are answered as they are from then on, and the run goes on).
* Filled out, a decision's scores differ in their last digits from those it
  has in the batch it came in, since sums run over more zeros.  A draw
  within that of its boundary goes the other way: the games answered here
  are, now and then, other games of the same model.
* A graph reads the weights where they were when it was recorded.  An
  optimizer's step leaves them there; were they moved, the graphs are
  recorded again.

On a processor there is no graph: the calls are filled out and answered by
the model, which is what the tests hold to.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

# What a call's arrays are filled out to, by the part of the input an axis
# follows: the state's cards, the seat's earlier answers, the answers and the
# members of an answer.  Wide enough for nearly every decision of the games
# measured (the widest of an update: 89 cards, 32 earlier answers, 27
# answers of 2 members); the input's own limits are far wider
# (schema.MAX_CARDS and the others), and what a batch is filled out to is
# worked out for all of it.  Until the input dropped the deck lists
# (format 15) they were parts too, and a list of more than 48 cards kept a
# call off the graphs.  The widths are Blackwing's: in the games of the
# thirteen decks of training, answers of 3 to 5 members keep a fifth of the
# calls off (docs/commands.md, its section on calls from CUDA graphs).
WIDTHS = {
    "state": 96,
    "history": 32,
    "action": 32,
    "members": 2,
}


def axes(name: str) -> tuple[str, ...]:
    """The widths the axes of the input's array ``name`` follow, after the
    first, which is its decisions: none for an array of one row a decision.
    As the library pads them."""
    if name.startswith("action_member_"):
        return ("action", "members")
    if name.startswith("action_"):
        return ("action",)
    for part in ("state", "history"):
        if name == part or name.startswith(f"{part}_"):
            return (part,)
    return ()


def fixed(name: str, shape: tuple[int, ...], widths: dict[str, int]) -> tuple[int, ...]:
    """What the array ``name`` of ``shape`` is filled out to after its first
    axis: the widths its axes follow, and its own sizes for the others."""
    after = list(shape[1:])
    for axis, part in enumerate(axes(name)[: len(after)]):
        after[axis] = widths[part]
    return tuple(after)


def counts(largest: int) -> list[int]:
    """The counts of decisions a call is filled out to: the powers of two up
    to ``largest``, and ``largest``."""
    found = []
    count = 1
    while count < largest:
        found.append(count)
        count *= 2
    return [*found, largest]


@dataclass
class Sized:
    """A batch of one count of decisions at the fixed widths: the bytes of
    its arrays laid end to end here and on the device, the arrays as views
    of both, and on a card the graph of the model's answer to them."""

    count: int
    here: np.ndarray
    there: torch.Tensor
    arrays: dict[str, np.ndarray]
    tensors: dict[str, torch.Tensor]
    # What the last call put in each array: more than that is zeros.
    held: dict[str, tuple[int, ...]] = field(default_factory=dict)
    graph: Any = None
    answer: tuple[torch.Tensor, torch.Tensor] | None = None
    # The update a replay was last held against the model's answer in.
    checked: int = -1


class Graphs:
    """The model's answers to calls of up to ``largest`` decisions, at sizes
    fixed in advance: replayed from CUDA graphs on a card, the model's own on
    a processor."""

    def __init__(
        self,
        model: torch.nn.Module,
        device: str,
        *,
        largest: int,
        widths: dict[str, int] | None = None,
        records: bool | None = None,
    ) -> None:
        if largest < 1:
            raise ValueError("A graph answers one decision at least")
        self.model = model
        self.device = str(device)
        # Whether answers are replayed from graphs: on a card.
        self.records = self.device.startswith("cuda") if records is None else records
        self.widths = dict(WIDTHS if widths is None else widths)
        self.counts = counts(largest)
        self.sized: dict[int, Sized] = {}
        # Where the weights were when the graphs were recorded.
        self.weights: tuple[int, ...] | None = None
        self.update = 0
        # Why no call is answered here any more, once that is so.
        self.failed: str | None = None
        self.told = {"answered": 0, "recorded": 0, "checked": 0}

    def begin_update(self) -> None:
        """An update's games begin: each graph is held against the model
        again at its first call, and all are recorded again if the weights
        are no longer where they were."""
        self.update += 1
        weights = tuple(parameter.data_ptr() for parameter in self.model.parameters())
        if self.weights is not None and weights != self.weights:
            for sized in self.sized.values():
                sized.graph = sized.answer = None
        self.weights = weights

    def fits(self, arrays: dict[str, np.ndarray]) -> bool:
        """Whether a call's arrays are no more decisions and no wider than
        the sizes fixed."""
        decisions = len(next(iter(arrays.values())))
        if not 1 <= decisions <= self.counts[-1]:
            return False
        return all(
            size <= most
            for name, array in arrays.items()
            for size, most in zip(array.shape[1:], fixed(name, array.shape, self.widths), strict=True)
        )

    def _sized(self, count: int, arrays: dict[str, np.ndarray]) -> Sized:
        """The batch of ``count`` decisions at the fixed widths, made the
        first time it is asked for: the arrays of a call say what each holds
        a decision."""
        known = self.sized.get(count)
        if known is not None:
            return known
        places: dict[str, tuple[int, int, tuple[int, ...]]] = {}
        length = 0
        for name, array in arrays.items():
            shape = (count, *fixed(name, array.shape, self.widths))
            start = -(-length // 8) * 8
            length = start + int(np.prod(shape, dtype=np.int64)) * array.dtype.itemsize
            places[name] = (start, length, shape)
        here = np.zeros(length, dtype=np.uint8)
        there = torch.zeros(length, dtype=torch.uint8, device=self.device)
        sized = Sized(
            count=count,
            here=here,
            there=there,
            arrays={
                name: here[start:end].view(arrays[name].dtype).reshape(shape)
                for name, (start, end, shape) in places.items()
            },
            tensors={
                name: there[start:end].view(torch.from_numpy(arrays[name]).dtype).view(shape)
                for name, (start, end, shape) in places.items()
            },
        )
        self.sized[count] = sized
        return sized

    def _fill(self, sized: Sized, arrays: dict[str, np.ndarray]) -> None:
        """A call's arrays into the batch, zeros around them, and the batch
        to the device in one copy."""
        for name, array in arrays.items():
            whole = sized.arrays[name]
            before = sized.held.get(name)
            if before is not None and any(was > now for was, now in zip(before, array.shape, strict=True)):
                whole[tuple(slice(0, size) for size in before)] = 0
            whole[tuple(slice(0, size) for size in array.shape)] = array
            sized.held[name] = array.shape
        sized.there.copy_(torch.from_numpy(sized.here))

    def _record(self, sized: Sized) -> None:
        """The model's answer to the batch recorded as a graph: after it,
        the answer is where the recording left it once the graph is
        replayed."""
        with torch.no_grad():
            for _ in range(3):
                self.model(sized.tensors)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                logits, values = self.model(sized.tensors)
        sized.graph, sized.answer = graph, (logits, values)
        sized.checked = -1
        self.told["recorded"] += 1

    def _replayed(self, sized: Sized) -> tuple[torch.Tensor, torch.Tensor]:
        if sized.graph is None:
            self._record(sized)
        sized.graph.replay()
        assert sized.answer is not None
        if sized.checked != self.update:
            # Once an update: the replay against the model's own answer to
            # the same batch, which it is to the last digit or not at all.
            sized.checked = self.update
            self.told["checked"] += 1
            with torch.no_grad():
                plain = self.model(sized.tensors)
            for ours, theirs in zip(sized.answer, plain, strict=True):
                if not torch.equal(ours, theirs):
                    raise RuntimeError(
                        f"a replay of {sized.count} decisions was not the model's answer"
                    )
        return sized.answer

    def answer(self, arrays: dict[str, np.ndarray]) -> tuple[torch.Tensor, torch.Tensor] | None:
        """The logits and the values of a call's decisions, answered at the
        fixed sizes; none where the call is not answered here (it does not
        fit, or a graph failed).  They are read before the next call: a
        graph writes its answer to the same memory every time."""
        if self.failed is not None or not self.fits(arrays):
            return None
        decisions = len(next(iter(arrays.values())))
        count = next(count for count in self.counts if count >= decisions)
        try:
            sized = self._sized(count, arrays)
            self._fill(sized, arrays)
            if self.records:
                logits, values = self._replayed(sized)
            else:
                with torch.no_grad():
                    logits, values = self.model(sized.tensors)
        except Exception as error:  # noqa: BLE001 - the games go on, answered as they are
            self.failed = f"{type(error).__name__}: {' '.join(str(error).split())[:300]}"
            return None
        self.told["answered"] += 1
        return logits[:decisions], values[:decisions]
