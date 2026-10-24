"""What a call of the model in an update's games spends its time on.

    uv run python tools/call_timing.py CHECKPOINT --device cuda --itself 260

An update's games take as many calls of the model as their longest duel has
decisions, and most of those calls answer a few decisions: a call costs
about the same whatever it holds.  This takes that cost apart.  One update's
games are played with the checkpoint, through the pool as a run plays them,
and a batch of every size the model was given is kept.  Each is then
answered again many times, the parts of a call timed one by one with the
card waited for after each:

- the seeds of the draws, which Python works out a decision at a time;
- the batch taken to the card, array by array;
- the model's answer, and inside it the encoders it is made of;
- the draw, the probability of what was drawn, and the three read back.

It also counts what an answer launches (PyTorch's operations, and the
kernels and copies on the card), and tells how an update's calls are spread
over the sizes, so that the parts can be weighed.

On a card each kept batch's answer is also recorded as a CUDA graph and
replayed.  A graph is the card's work for one answer, recorded once at the
sizes of its batch and launched again in one call, where the model launches
each of its kernels by itself.  The report sets a replay beside the model's
own answer, says whether the two are the same numbers, and what recording
cost.  The batch is also taken to the card in one copy, as one run of bytes
its arrays are views of, where it goes array by array.  Both are tries: the
trainer uses neither.

A graph serves one set of sizes only, so that the calls of a run would have
to be filled out to sizes fixed in advance.  Each kept batch is therefore
also filled out with zeros to the widest any call of the update is, as the
library fills a batch out to its longest row, and answered again, plainly
and from a graph of those sizes: what the filling out costs, and whether a
decision keeps its numbers in a wider batch.

With ``--compile`` the batches filled out are also answered by the model
compiled by ``torch.compile`` for their sizes, none left free: once with
the LSTM compiled with the rest, once with it left to cuDNN.  For each, what
an answer takes, alone and replayed from a graph, how far its numbers are
from the plain model's, and what compiling took; and a step of a pass at
the sizes of a minibatch, the answer with its gradient kept and the
gradient, plain and compiled.  The compiler wants a C compiler and Python's
own headers, which a Python of uv's has (``tools/update_timing.py`` says
how to borrow them for another).  Minutes more, most of them compiling.

On a card a step of a pass is tried as a graph too, last of all: the answer
with its gradient kept, a loss, the gradient, its clipping and the
optimizer's step, recorded whole at the sizes of a minibatch filled out.  A
step is first made with every wait for the card an error, since a graph
cannot hold one; it is then recorded and replayed, held against the same
steps made one by one, on the batch it was recorded on and on another of
its sizes.  And again with the tables looked up by ``index_select``, whose
gradient is another kernel's.

Where the trainer has it (``ygo_training.graphs``), its own way of answering
calls at fixed sizes is run on the kept batches too: what a call takes
through it, and that it answers.  With ``--members-width`` its graphs are
fixed at that many members of an answer, where the trainer's have two.

A measure of speed: nothing is trained, and nothing written but the report.
On a processor it runs to be tried, and its times say nothing of a card.
Two minutes on one.
"""

from __future__ import annotations

import argparse
import copy
import os
import statistics
import sys
import sysconfig
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ygo_training.checkpoints import load
from ygo_training.config import TrainerConfig
from ygo_training.inference import _sample_with_seeds
from ygo_training.ppo_trainer import PpoTrainer
from ygo_training.schedules import duel_sample_seed

OPPONENTS = (
    "blackwing,burn,crystal,dragunity,gishki,gladiator,heroes,infernity,lightsworn,"
    "monarch,morphtronic,rock-block,spellcaster"
)
# The sizes of a call are told in these classes, each named by its largest.
CLASSES = (1, 8, 32, 64, 128, 256, 512, 1024, 2048, 100_000)
PARTS = (
    ("seeds", "The seeds"),
    ("card", "To the card"),
    ("answer", "The answer"),
    ("draw", "The draw"),
    ("probability", "Its chance"),
    ("back", "Read back"),
)
# The encoders an answer is made of, as they are called: their times lie
# inside each other as the calls do.
ENCODERS = ("_card_representation", "_placed_cards", "_history", "encode_state")


def of_class(size: int) -> int:
    return next(largest for largest in CLASSES if size <= largest)


def named(largest: int) -> str:
    place = CLASSES.index(largest)
    if not place:
        return "1"
    least = CLASSES[place - 1] + 1
    return f"{least} or more" if largest == CLASSES[-1] else f"{least} to {largest}"


class Kept:
    """The sizes of the batches the model is given, and the first of each
    class as the arrays the pool handed over."""

    def __init__(self) -> None:
        self.sizes: list[int] = []
        self.batches: dict[int, dict[str, np.ndarray]] = {}
        # The widest every array was in any batch, its decisions aside.
        self.widest: dict[str, tuple[int, ...]] = {}

    def __call__(self, _module: torch.nn.Module, given: tuple[Any, ...]) -> None:
        tensor = given[0]
        size = int(tensor["history_lengths"].shape[0])
        self.sizes.append(size)
        for name, value in tensor.items():
            known = self.widest.get(name, value.shape[1:])
            self.widest[name] = tuple(max(pair) for pair in zip(known, value.shape[1:], strict=True))
        if of_class(size) not in self.batches:
            self.batches[of_class(size)] = {
                name: value.detach().cpu().numpy().copy() for name, value in tensor.items()
            }


def waited(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def timed(device: str, repeats: int, work) -> float:
    """The median of ``repeats`` runs of ``work`` in milliseconds, after three
    that warm up, each waited for."""
    taken = []
    for repeat in range(repeats + 3):
        waited(device)
        started = time.perf_counter()
        work()
        waited(device)
        if repeat >= 3:
            taken.append(1000 * (time.perf_counter() - started))
    return statistics.median(taken)


class Call:
    """One call of the model as the games make it, a part at a time."""

    def __init__(self, model: torch.nn.Module, device: str, arrays: dict[str, np.ndarray]) -> None:
        self.model, self.device, self.arrays = model, device, arrays
        self.size = int(arrays["history_lengths"].shape[0])
        self.seeds_()
        self.card()
        self.answer()
        self.draw()
        self.probability()

    def seeds_(self) -> None:
        self.seeds = [duel_sample_seed(2012, row, 7) for row in range(self.size)]

    def card(self) -> None:
        self.given = {
            name: torch.from_numpy(array).to(self.device) for name, array in self.arrays.items()
        }

    def answer(self) -> None:
        with torch.no_grad():
            self.logits, self.values = self.model(self.given)

    def draw(self) -> None:
        with torch.no_grad():
            self.drawn = _sample_with_seeds(self.logits, self.seeds, 1.0)

    def probability(self) -> None:
        with torch.no_grad():
            self.log_probs = torch.distributions.Categorical(logits=self.logits).log_prob(self.drawn)

    def back(self) -> None:
        self.drawn.cpu().numpy()
        self.log_probs.tolist()
        self.values.tolist()

    def whole(self) -> None:
        """All of it, with nothing waited for but what reading back waits for."""
        self.seeds_()
        self.card()
        self.answer()
        self.draw()
        self.probability()
        self.back()

    def measured(self, repeats: int) -> dict[str, float]:
        works = {
            "seeds": self.seeds_, "card": self.card, "answer": self.answer, "draw": self.draw,
            "probability": self.probability, "back": self.back,
        }  # fmt: skip
        told = {name: timed(self.device, repeats, work) for name, work in works.items()}
        told["whole"] = timed(self.device, repeats, self.whole)
        return told


def encoders(call: Call, repeats: int) -> list[tuple[str, int, float]]:
    """The encoders inside an answer: how often each is called in one, and
    the milliseconds of all its calls, each waited for.  A time holds those
    of the encoders called inside it."""
    model, device = call.model, call.device
    taken: dict[str, list[float]] = {}

    def watched(name: str, encoder):
        def run(*given, **more):
            label = name
            if name == "_placed_cards":
                label = f"{name} {given[1]}"
            waited(device)
            started = time.perf_counter()
            made = encoder(*given, **more)
            waited(device)
            taken.setdefault(label, []).append(1000 * (time.perf_counter() - started))
            return made

        return run

    for name in ENCODERS:
        setattr(model, name, watched(name, getattr(model, name)))
    try:
        for _ in range(3):
            call.answer()
        taken.clear()
        for _ in range(repeats):
            call.answer()
    finally:
        for name in ENCODERS:
            delattr(model, name)
    return [(label, len(times) // repeats, sum(times) / repeats) for label, times in taken.items()]


def launched(call: Call, answers: int = 20) -> dict[str, Any]:
    """What one answer launches: PyTorch's operations, and on a card its
    kernels and copies; with the operations launched most."""
    from torch.profiler import ProfilerActivity, profile

    activities = [ProfilerActivity.CPU]
    if call.device == "cuda":
        activities.append(ProfilerActivity.CUDA)
    call.answer()
    with profile(activities=activities) as watched:
        for _ in range(answers):
            call.answer()
        waited(call.device)
    operations: Counter[str] = Counter()
    called: Counter[str] = Counter()
    on_card = 0
    for event in watched.events():
        if str(event.device_type).endswith("CUDA"):
            on_card += 1
        elif event.name.startswith("aten::"):
            operations[event.name] += 1
            # One the model's own code calls, and not another operation.
            parent = getattr(event, "cpu_parent", None)
            if parent is None or not parent.name.startswith("aten::"):
                called[event.name] += 1
    return {
        "operations": sum(operations.values()) / answers,
        "called": sum(called.values()) / answers,
        "on_card": on_card / answers,
        "most": [(name, count / answers) for name, count in called.most_common(10)],
    }


def same_numbers(one: tuple[torch.Tensor, ...], other: tuple[torch.Tensor, ...]) -> tuple[bool, float]:
    """Whether two answers are the same numbers, and how far apart they are
    at most where both are finite (an answer that is none has no score)."""
    same, apart = True, 0.0
    for ours, theirs in zip(one, other, strict=True):
        same = same and bool(torch.equal(ours, theirs))
        finite = torch.isfinite(ours) & torch.isfinite(theirs)
        if bool(finite.any()):
            apart = max(apart, float((ours[finite] - theirs[finite]).abs().max()))
    return same, apart


def replayed(call: Call, repeats: int) -> dict[str, Any]:
    """The answer to ``call``'s batch recorded as a CUDA graph and replayed:
    what a replay takes, the batch copied into the graph's own tensors array
    by array, a call whole that way, whether a replay gives the model's own
    numbers, and what recording took and keeps.

    A graph reads its input from the same memory every time and writes its
    answer to the same: the batch is copied into tensors that stay, and the
    answer read from those the recording left.  While it is recorded the
    work is not done, so that the answer is there after a replay only."""
    model, device = call.model, call.device
    with torch.no_grad():
        plain = tuple(made.clone() for made in model(call.given))
        static = {name: tensor.clone() for name, tensor in call.given.items()}
        for _ in range(3):
            model(static)
        torch.cuda.synchronize()
        before = torch.cuda.memory_allocated()
        graph = torch.cuda.CUDAGraph()
        started = time.perf_counter()
        with torch.cuda.graph(graph):
            answer = tuple(model(static))
        torch.cuda.synchronize()
        told: dict[str, Any] = {
            "recording": 1000 * (time.perf_counter() - started),
            "keeps": torch.cuda.memory_allocated() - before,
        }
        graph.replay()
        torch.cuda.synchronize()
        told["same"], told["apart"] = same_numbers(answer, plain)
        # Another batch of the same sizes, the rows of this one turned round:
        # a replay has to answer what is in its tensors now.
        told["turned"] = None
        if call.size > 1:
            turned = {name: tensor.flip(0) for name, tensor in call.given.items()}
            wanted = tuple(made.clone() for made in model(turned))
            for name, tensor in static.items():
                tensor.copy_(turned[name])
            graph.replay()
            torch.cuda.synchronize()
            told["turned"] = same_numbers(answer, wanted)[0]

    def place() -> None:
        for name, tensor in static.items():
            tensor.copy_(torch.from_numpy(call.arrays[name]))

    def whole() -> None:
        call.seeds_()
        place()
        graph.replay()
        with torch.no_grad():
            drawn = _sample_with_seeds(answer[0], call.seeds, 1.0)
            log_probs = torch.distributions.Categorical(logits=answer[0]).log_prob(drawn)
        drawn.cpu().numpy()
        log_probs.tolist()
        answer[1].tolist()

    told["replay"] = timed(device, repeats, graph.replay)
    told["place"] = timed(device, repeats, place)
    told["whole"] = timed(device, repeats, whole)
    return told


def in_one_copy(call: Call, repeats: int) -> dict[str, Any]:
    """The batch of ``call`` taken to the device in one copy: its arrays laid
    end to end in one run of bytes, each starting at a multiple of eight,
    and on the device a tensor of those bytes the model's arrays are views
    of.  What laying them out and the copy take, and whether the views are
    the arrays."""
    device = call.device
    places: dict[str, tuple[int, int]] = {}
    length = 0
    for name, array in call.arrays.items():
        start = -(-length // 8) * 8
        places[name] = (start, start + array.nbytes)
        length = start + array.nbytes
    here = np.zeros(length, dtype=np.uint8)
    there = torch.zeros(length, dtype=torch.uint8, device=device)
    views = {
        name: there[start:end].view(torch.from_numpy(call.arrays[name]).dtype).view(call.arrays[name].shape)
        for name, (start, end) in places.items()
    }

    def work() -> None:
        for name, (start, end) in places.items():
            here[start:end] = np.ascontiguousarray(call.arrays[name]).reshape(-1).view(np.uint8)
        there.copy_(torch.from_numpy(here))

    took = timed(device, repeats, work)
    waited(device)
    same = all(
        bool(torch.equal(view.cpu(), torch.from_numpy(call.arrays[name]))) for name, view in views.items()
    )
    return {"took": took, "same": same, "bytes": length, "arrays": len(places)}


def widest(batches: dict[int, dict[str, np.ndarray]]) -> dict[str, tuple[int, ...]]:
    """For every array of the model's input, the widest it is in any of
    ``batches``, the decisions of a batch aside."""
    shapes: dict[str, tuple[int, ...]] = {}
    for arrays in batches.values():
        for name, array in arrays.items():
            known = shapes.get(name, array.shape[1:])
            shapes[name] = tuple(max(pair) for pair in zip(known, array.shape[1:], strict=True))
    return shapes


def filled(arrays: dict[str, np.ndarray], shapes: dict[str, tuple[int, ...]]) -> dict[str, np.ndarray]:
    """A batch with every array filled out with zeros to ``shapes``: what the
    library makes of its decisions in a batch whose longest rows are that
    long (the library fills with zeros, and an answer that is none is
    masked)."""
    out = {}
    for name, array in arrays.items():
        wide = np.zeros((array.shape[0], *shapes[name]), dtype=array.dtype)
        wide[tuple(slice(0, size) for size in array.shape)] = array
        out[name] = wide
    return out


def filled_out(own: Call, wide: Call, repeats: int) -> dict[str, Any]:
    """What the model's answer takes to the batch filled out, and whether it
    is the answer to the batch as it was: the same numbers where that batch
    has answers, and none where it has none."""
    with torch.no_grad():
        logits, values = own.model(own.given)
        more, same_values = wide.model(wide.given)
    answers = logits.shape[1]
    same, apart = same_numbers((logits, values), (more[:, :answers], same_values))
    none = bool(torch.isinf(more[:, answers:]).all())
    return {"answer": timed(wide.device, repeats, wide.answer), "same": same and none, "apart": apart}


def headers() -> bool:
    """Whether Python's own headers are where a C compiler will look."""
    folders = [sysconfig.get_paths()["include"], *os.environ.get("CPATH", "").split(os.pathsep)]
    return any(folder and (Path(folder) / "Python.h").exists() for folder in folders)


def compiling(model: torch.nn.Module, lstm_too: bool) -> Any:
    """The model as ``torch.compile`` makes it for the sizes it is given,
    none left free: a version for each set of sizes, and more of them than
    the eight PyTorch stops at.  With ``lstm_too`` the LSTM is compiled with
    the rest, taken apart into its steps; without, PyTorch steps around it
    and it stays cuDNN's.  What was compiled before is forgotten: the
    setting is read when a version is made."""
    import torch._dynamo

    torch._dynamo.reset()
    torch._dynamo.config.allow_rnn = lstm_too
    torch._dynamo.config.recompile_limit = 64
    return torch.compile(model, dynamic=False)


def compiled_call(plain: Call, answerer: Any, repeats: int) -> dict[str, Any]:
    """The batch of ``plain`` answered by ``answerer``, the model compiled for
    its sizes: what compiling took (the first answer), an answer and a call
    whole, how far its numbers are from the plain model's, and on a card the
    compiled answer replayed from a graph."""
    device = plain.device
    waited(device)
    started = time.perf_counter()
    call = Call(answerer, device, plain.arrays)
    waited(device)
    told: dict[str, Any] = {"compiling": time.perf_counter() - started}
    with torch.no_grad():
        ours, theirs = tuple(answerer(call.given)), tuple(plain.model(plain.given))
    told["same"], told["apart"] = same_numbers(ours, theirs)
    told["answer"] = timed(device, repeats, call.answer)
    told["whole"] = timed(device, repeats, call.whole)
    told["graph"] = tried(replayed, call, repeats) if device == "cuda" else None
    return told


def stepped(call: Call, answerer: Any, repeats: int) -> dict[str, float]:
    """A step of a pass at the batch of ``call``: the milliseconds of the
    model's answer with its gradient kept and of the gradient, the medians of
    ``repeats`` steps, and the seconds the first of them took, which is when
    a compiled model is compiled.  ``answerer`` is the model of ``call`` or
    that model compiled.

    The loss is a plain one of the shape of an update's (the first answer
    taken, the value against nothing): what it costs is the model's."""
    device = call.device
    parameters = list(call.model.parameters())
    taken_ = torch.zeros(call.size, dtype=torch.long, device=device)

    def once() -> tuple[float, float]:
        waited(device)
        started = time.perf_counter()
        logits, values = answerer(call.given)
        loss = torch.nn.functional.cross_entropy(logits, taken_) + values.pow(2).mean()
        waited(device)
        middle = time.perf_counter()
        loss.backward()
        waited(device)
        ended = time.perf_counter()
        for parameter in parameters:
            parameter.grad = None
        return 1000 * (middle - started), 1000 * (ended - middle)

    started = time.perf_counter()
    once()
    first = time.perf_counter() - started
    once()
    taken = [once() for _ in range(repeats)]
    return {
        "answer": statistics.median(answer for answer, _ in taken),
        "gradient": statistics.median(gradient for _, gradient in taken),
        "first": first,
    }


class Lookup(torch.nn.Module):
    """The rows of a table by ``index_select``: what ``nn.Embedding`` looks
    up, with the gradient of another kernel (an ``index_add`` into a table
    of zeros, where an embedding has one of its own)."""

    def __init__(self, table: torch.nn.Embedding) -> None:
        super().__init__()
        self.weight = table.weight

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        rows = self.weight.index_select(0, indices.reshape(-1))
        return rows.view(*indices.shape, self.weight.shape[1])


def with_lookups(model: torch.nn.Module) -> torch.nn.Module:
    """``model`` with each of its embedding tables looked up by a ``Lookup``
    of the same weight."""
    for parent in list(model.modules()):
        for name, child in list(parent.named_children()):
            if isinstance(child, torch.nn.Embedding):
                setattr(parent, name, Lookup(child))
    return model


def weights_beside(one: torch.nn.Module, other: torch.nn.Module) -> tuple[bool, float]:
    """Whether two models have the same weights, and how far apart at most."""
    same, apart = True, 0.0
    for ours, theirs in zip(one.parameters(), other.parameters(), strict=True):
        same = same and bool(torch.equal(ours, theirs))
        apart = max(apart, float((ours.detach() - theirs.detach()).abs().max()))
    return same, apart


def recorded_step(
    call: Call, repeats: int, lookups: bool = False, on_card: bool | None = None
) -> dict[str, Any]:
    """A step of a pass at the batch of ``call``, recorded whole as a CUDA
    graph: the answer with its gradient kept, a loss, the gradient, its
    clipping and the optimizer's step.

    Three models of their own, each the weights of the call's, take the same
    steps: one as training makes them, one with an optimizer that can be
    recorded, and one whose step is recorded and replayed.  Three steps on
    the batch come first for all; then two on it and two on another batch of
    its sizes (its second half twice over), after each of which the replayed
    weights are held against those of the steps made one by one.  A graph
    that kept something of the batch it was recorded on would give itself
    away on the other.

    Before anything is recorded a step is made with every wait for the card
    an error: a graph cannot hold one, and what waits is told instead.  With
    ``lookups`` the tables are looked up by ``index_select``.

    The loss is a plain one of the shape of an update's: what it costs is
    the model's."""
    device = call.device
    on_card = device == "cuda" if on_card is None else on_card
    half = call.size // 2
    batch = {name: tensor.clone() for name, tensor in call.given.items()}
    # The second half of the batch, as often as fills it.
    rows = half + torch.arange(call.size, device=device) % max(half, 1)
    other = {name: tensor[rows] for name, tensor in batch.items()}
    static = {name: tensor.clone() for name, tensor in batch.items()}
    taken_ = torch.zeros(call.size, dtype=torch.long, device=device)

    def made(capturable: bool) -> tuple[torch.nn.Module, torch.optim.Optimizer]:
        model = copy.deepcopy(call.model)
        # A copied LSTM's weights are no longer one run of memory, and cuDNN
        # then packs them again at every call: put them back in one.
        for module in model.modules():
            if isinstance(module, torch.nn.RNNBase):
                module.flatten_parameters()
        if lookups:
            with_lookups(model)
        # An optimizer that can be recorded is a card's: a processor has the other.
        capturable = capturable and device == "cuda"
        return model, torch.optim.Adam(model.parameters(), lr=1e-4, capturable=capturable)

    def loss_of(model: torch.nn.Module) -> torch.Tensor:
        logits, values = model(static)
        return torch.nn.functional.cross_entropy(logits, taken_) + values.pow(2).mean()

    def work(model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> None:
        loss_of(model).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    def step(model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> None:
        optimizer.zero_grad()
        work(model, optimizer)

    def given(tensors: dict[str, torch.Tensor]) -> None:
        for name, tensor in static.items():
            tensor.copy_(tensors[name])

    plain, able, replayed_ = made(False), made(True), made(True)
    for way in (plain, able, replayed_):
        for _ in range(3):
            step(*way)
    told: dict[str, Any] = {"waits": None, "replay": None}
    # Whether the optimizer made to be recorded takes the same steps.
    told["able_same"], told["able_apart"] = weights_beside(plain[0], able[0])
    if on_card:
        # A step with every wait for the card an error, a part at a time:
        # the first part that waits is told.
        model, optimizer = made(True)
        step(model, optimizer)
        optimizer.zero_grad()
        held: list[torch.Tensor] = []
        parts = (
            ("the answer and the loss", lambda: held.append(loss_of(model))),
            ("the gradient", lambda: held[0].backward()),
            ("the clipping", lambda: torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)),
            ("the optimizer's step", optimizer.step),
        )
        torch.cuda.synchronize()
        torch.cuda.set_sync_debug_mode("error")
        try:
            for part, made_ in parts:
                try:
                    made_()
                except Exception as error:  # noqa: BLE001 - told in the report
                    told["waits"] = f"{part}: {' '.join(str(error).split())[:200]}"
                    break
        finally:
            torch.cuda.set_sync_debug_mode("default")
        torch.cuda.synchronize()
    if on_card and told["waits"] is None:
        model, optimizer = replayed_
        try:
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            before = torch.cuda.memory_allocated()
            graph = torch.cuda.CUDAGraph()
            started = time.perf_counter()
            with torch.cuda.graph(graph):
                work(model, optimizer)
            torch.cuda.synchronize()
            told["recording"] = 1000 * (time.perf_counter() - started)
            told["keeps"] = torch.cuda.memory_allocated() - before
            for name, tensors in (("same", batch), ("another", other)):
                given(tensors)
                for _ in range(2):
                    step(*able)
                    graph.replay()
                torch.cuda.synchronize()
                told[name], told[f"{name}_apart"] = weights_beside(able[0], model)
            given(batch)
            told["replay"] = timed(device, repeats, graph.replay)
        except Exception as error:  # noqa: BLE001 - told in the report
            told["replay"] = None
            told["waits"] = f"while it was recorded: {type(error).__name__}: {' '.join(str(error).split())[:200]}"
    told["plain"] = timed(device, repeats, lambda: step(*plain))
    told["able"] = timed(device, repeats, lambda: step(*able))
    return told


def trainers_way(
    model: torch.nn.Module,
    device: str,
    calls: dict[int, Call],
    repeats: int,
    members: int | None = None,
) -> Any:
    """The trainer's own way of answering calls at fixed sizes
    (``ygo_training.graphs``), where this checkout has it: for each kept
    batch it takes, a call whole through it, as ``Call.whole`` times the
    plain one; and what it says of itself after.  ``members``: the answers'
    members it is fixed at, where not the trainer's."""
    try:
        from ygo_training.graphs import WIDTHS, Graphs
    except ImportError:
        return None
    widths = dict(WIDTHS) | ({"members": members} if members else {})
    graphs = Graphs(model, device, largest=max(call.size for call in calls.values()), widths=widths)
    graphs.begin_update()
    found: dict[str, Any] = {"widths": widths, "calls": {}}
    for largest, call in calls.items():
        if not graphs.fits(call.arrays):
            found["calls"][largest] = "wider than the sizes fixed"
            continue

        def whole(call: Call = call) -> None:
            call.seeds_()
            made = graphs.answer(call.arrays)
            if made is None:
                raise RuntimeError(graphs.failed or "not answered")
            with torch.no_grad():
                drawn = _sample_with_seeds(made[0], call.seeds, 1.0)
                log_probs = torch.distributions.Categorical(logits=made[0]).log_prob(drawn)
            drawn.cpu().numpy()
            log_probs.tolist()
            made[1].tolist()

        found["calls"][largest] = tried(timed, device, repeats, whole)
    found["told"], found["failed"] = dict(graphs.told), graphs.failed
    return found


def last_report(found: dict[str, Any]) -> list[str]:
    """What is tried last of all: the trainer's own way, and a step of a
    pass as a graph."""
    out: list[str] = []
    way = found.get("trainer")
    if isinstance(way, str):
        out += ["", f"The trainer's own way at fixed sizes was not run: {way}"]
    elif way:
        widths = way["widths"]
        out += [
            "",
            (
                "The trainer's own way (--inference-graphs), a call whole (ms): filled out to "
                f"{widths['state']} cards of the state, {widths['history']} earlier answers, "
                f"{widths['action']} answers "
                f"of {widths['members']} members, and to the next count of decisions it keeps."
            ),
            "",
        ]
        for largest, took in way["calls"].items():
            plain = found["calls"][largest]
            line = f"{named(largest):>12} {plain['size']:>6} {plain['whole']:>8.2f} "
            out.append(line + (f"{took:>8.2f}" if isinstance(took, float) else f"       -   {took}"))
        told = way["told"]
        out += [
            "",
            (
                f"It answered {told['answered']:,} calls from {told['recorded']} graphs, each held against "
                f"the model's own answer when first used ({told['checked']} times)."
                if told["recorded"]
                else f"It answered {told['answered']:,} calls, the model's own at fixed sizes: no graph here."
            ),
        ]
        if way["failed"]:
            out.append(f"IT FAILED, and answered the calls as they are from there: {way['failed']}")
    steps = found.get("step_graphs")
    if steps:
        out += [
            "",
            (
                f"A step of a pass at the batch of {steps['size']} decisions filled out, whole: the answer, a "
                "loss, the gradient, its clipping and the optimizer's step (ms)."
            ),
        ]
        for label, step in steps["ways"].items():
            out += ["", f"  {label}"]
            if isinstance(step, str):
                out.append(f"    not made: {step}")
                continue
            able = "the same weights" if step["able_same"] else f"weights {step['able_apart']:.1e} apart at most"
            out += [
                f"    step by step, as training makes it          {step['plain']:>8.2f}",
                f"    with an optimizer that can be recorded       {step['able']:>8.2f}   {able} after three steps",
            ]
            if step["waits"] is not None:
                out.append(f"    not recorded, a step waits for the card: {step['waits']}")
            elif step["replay"] is not None:
                numbers = []
                for name, where in (("same", "on its batch"), ("another", "on another of its sizes")):
                    numbers.append(
                        f"the weights of the steps one by one {where}"
                        if step[name]
                        else f"NOT those weights {where}, {step[f'{name}_apart']:.1e} apart at most"
                    )
                out += [
                    f"    recorded as a graph and replayed             {step['replay']:>8.2f}   "
                    + "; ".join(numbers),
                    (
                        f"    recording took {step['recording']:.0f} ms, and the graph keeps "
                        f"{step['keeps'] / 2**20:.0f} MiB of the card's memory"
                    ),
                ]
    return out


def tried(work, *given: Any) -> Any:
    """What ``work`` finds, or why it found nothing: a try that fails leaves
    the rest of the report standing."""
    try:
        return work(*given)
    except Exception as error:  # noqa: BLE001 - told in the report
        return f"{type(error).__name__}: {' '.join(str(error).split())[:300]}"


def graphs_report(played: dict[str, Any], found: dict[str, Any]) -> list[str]:
    """The table of the answers replayed from graphs, beside the model's own."""
    graphs, copies = found.get("graphs", {}), found.get("copies", {})
    if not graphs and not copies and not found.get("filled") and found.get("compiled") is None:
        return []
    sizes = played["sizes"]
    out = [
        "",
        "Each call's answer recorded as a CUDA graph at the sizes of its batch, and replayed (ms).",
        "A graph serves those sizes only: a run's calls would be padded to sizes fixed in advance.",
        "",
        (
            f"{'Decisions':>12} {'Timed':>6} {'The answer':>11} {'replayed':>9} {'To the card':>12} "
            f"{'in place':>9} {'in one':>8} {'A call':>8} {'by a':>8}   The same numbers"
        ),
        (
            f"{'in a call':>12} {'at':>6} {'':>11} {'':>9} {'':>12} {'':>9} {'copy':>8} {'whole':>8} "
            f"{'graph':>8}"
        ),
    ]
    plainly = by_graphs = 0.0
    failed: list[str] = []
    recordings: list[float] = []
    kept: list[float] = []
    for largest in CLASSES:
        inside = [size for size in sizes if of_class(size) == largest]
        told = found["calls"].get(largest)
        if not inside or told is None:
            continue
        graph, copy = graphs.get(largest), copies.get(largest)
        one = "-"
        if isinstance(copy, dict):
            one = f"{copy['took']:.2f}" if copy["same"] else "other"
        elif isinstance(copy, str):
            failed.append(f"  {named(largest)}, in one copy: {copy}")
        plainly += len(inside) * told["whole"] / 1000
        if not isinstance(graph, dict):
            if isinstance(graph, str):
                failed.append(f"  {named(largest)}, the graph: {graph}")
            by_graphs += len(inside) * told["whole"] / 1000
            out.append(
                f"{named(largest):>12} {told['size']:>6} {told['answer']:>11.2f} {'-':>9} "
                f"{told['card']:>12.2f} {'-':>9} {one:>8} {told['whole']:>8.2f} {'-':>8}   not recorded"
            )
            continue
        # A graph is worth its calls where it is the shorter way.
        by_graphs += len(inside) * min(told["whole"], graph["whole"]) / 1000
        recordings.append(graph["recording"])
        kept.append(graph["keeps"] / 2**20)
        numbers = "yes" if graph["same"] else f"NO, {graph['apart']:.1e} apart at most"
        if graph["turned"] is not None:
            numbers += ", and for its rows turned round" if graph["turned"] else ", NOT for its rows turned round"
        out.append(
            f"{named(largest):>12} {told['size']:>6} {told['answer']:>11.2f} {graph['replay']:>9.2f} "
            f"{told['card']:>12.2f} {graph['place']:>9.2f} {one:>8} {told['whole']:>8.2f} "
            f"{graph['whole']:>8.2f}   {numbers}"
        )
    out.append("")
    if recordings:
        out.append(
            f"Recording a graph took {min(recordings):.0f} to {max(recordings):.0f} ms, and one keeps "
            f"{min(kept):.1f} to {max(kept):.1f} MiB of the card's memory."
        )
        out.append(
            f"By those, the calls of an update take {by_graphs:.2f} s with a graph wherever it is the "
            f"shorter way, where they take {plainly:.2f} s: the copy in one not counted."
        )
    out += wide_report(played, found, failed)
    out += compiled_report(played, found, failed)
    out += ["", f"Not found ({len(failed)}):", *failed] if failed else []
    return out


def compiled_report(played: dict[str, Any], found: dict[str, Any], failed: list[str]) -> list[str]:
    """The tables of the model compiled for fixed sizes, and of a step of a
    pass; what was not found is added to ``failed``."""
    compiled = found.get("compiled")
    if compiled is None:
        return []
    if isinstance(compiled, str):
        return ["", f"Not compiled: {compiled}"]
    sizes = played["sizes"]
    out: list[str] = []
    for way, made in compiled.items():
        out += [
            "",
            f"The batches filled out, answered by the model compiled for their sizes, {way} (ms).",
            "",
            (
                f"{'Decisions':>12} {'Timed':>6} {'The answer':>11} {'compiled':>9} {'replayed':>9} "
                f"{'A call':>8} {'compiled':>9} {'by its':>8} {'Compiling':>10}   Its numbers beside the plain model's"
            ),
            (
                f"{'in a call':>12} {'at':>6} {'':>11} {'':>9} {'':>9} {'whole':>8} {'':>9} {'graph':>8} "
                f"{'s':>10}"
            ),
        ]
        if isinstance(made, str):
            failed.append(f"  compiled, {way}: {made}")
            continue
        plainly = shortest = 0.0
        for largest in CLASSES:
            inside = [size for size in sizes if of_class(size) == largest]
            told = found["calls"].get(largest)
            if not inside or told is None:
                continue
            plain, call = found.get("filled", {}).get(largest), made["calls"].get(largest)
            plainly += len(inside) * told["whole"] / 1000
            shorter = told["whole"]
            if not isinstance(call, dict):
                if isinstance(call, str):
                    failed.append(f"  {named(largest)}, compiled, {way}: {call}")
                shortest += len(inside) * shorter / 1000
                continue
            answer = f"{plain['answer']:.2f}" if isinstance(plain, dict) else "-"
            numbers = "the same" if call["same"] else f"{call['apart']:.1e} apart at most"
            replay = whole = "-"
            graph = call["graph"]
            shorter = min(shorter, call["whole"])
            if isinstance(graph, dict):
                replay, whole = f"{graph['replay']:.2f}", f"{graph['whole']:.2f}"
                shorter = min(shorter, graph["whole"])
                if not graph["same"] or graph["turned"] is False:
                    numbers += "; its replay NOT its own answer"
            elif isinstance(graph, str):
                failed.append(f"  {named(largest)}, the graph of the model compiled, {way}: {graph}")
            shortest += len(inside) * shorter / 1000
            out.append(
                f"{named(largest):>12} {told['size']:>6} {answer:>11} {call['answer']:>9.2f} {replay:>9} "
                f"{told['whole']:>8.2f} {call['whole']:>9.2f} {whole:>8} {call['compiling']:>10.1f}   {numbers}"
            )
        out += [
            "",
            (
                f"By those, the calls of an update take {shortest:.2f} s with the compiled model wherever "
                f"it is the shorter way, where they take {plainly:.2f} s."
            ),
        ]
    steps = found.get("steps")
    if steps:
        out += [
            "",
            (
                f"A step of a pass at the batch of {steps['size']} decisions: the model's answer with its "
                "gradient kept, and the gradient (ms)."
            ),
            "",
        ]
        for label, step in steps["ways"].items():
            if isinstance(step, str):
                failed.append(f"  a step, {label}: {step}")
                out.append(f"  {label:<44} not made")
                continue
            first = f"   the first took {step['first']:.1f} s" if step["first"] >= 1 else ""
            out.append(f"  {label:<44} {step['answer']:>7.2f} {step['gradient']:>8.2f}{first}")
    return out


def wide_report(played: dict[str, Any], found: dict[str, Any], failed: list[str]) -> list[str]:
    """The table of the batches filled out to sizes fixed in advance; what
    was not found is added to ``failed``."""
    wide, graphs = found.get("filled", {}), found.get("wide_graphs", {})
    if not wide:
        return []
    shapes = found.get("widest", {})
    sizes = played["sizes"]

    def width(name: str, axis: int = 0) -> str:
        return str(shapes[name][axis]) if name in shapes and len(shapes[name]) > axis else "?"

    out = [
        "",
        (
            f"The same with each batch filled out to the widest any call of the update is: {width('state_card')} "
            f"cards of the state, {width('history_card')} earlier answers, {width('action_kind')} answers of "
            f"{width('action_member_card', 1)} members (ms)."
        ),
        "",
        (
            f"{'Decisions':>12} {'Timed':>6} {'The answer':>11} {'filled':>8} {'replayed':>9} {'A call':>8} "
            f"{'by a':>8}   A decision's numbers in the wider batch"
        ),
        f"{'in a call':>12} {'at':>6} {'':>11} {'out':>8} {'':>9} {'whole':>8} {'graph':>8}",
    ]
    plainly = by_graphs = 0.0
    recorded = False
    for largest in CLASSES:
        inside = [size for size in sizes if of_class(size) == largest]
        told = found["calls"].get(largest)
        if not inside or told is None:
            continue
        plain, graph = wide.get(largest), graphs.get(largest)
        plainly += len(inside) * told["whole"] / 1000
        shorter = told["whole"]
        if not isinstance(plain, dict):
            if isinstance(plain, str):
                failed.append(f"  {named(largest)}, filled out: {plain}")
            out.append(
                f"{named(largest):>12} {told['size']:>6} {told['answer']:>11.2f} {'-':>8} {'-':>9} "
                f"{told['whole']:>8.2f} {'-':>8}   not answered"
            )
            by_graphs += len(inside) * shorter / 1000
            continue
        numbers = "the same" if plain["same"] else f"OTHERS, {plain['apart']:.1e} apart at most"
        replay = whole = "-"
        if isinstance(graph, dict):
            recorded = True
            replay, whole = f"{graph['replay']:.2f}", f"{graph['whole']:.2f}"
            shorter = min(shorter, graph["whole"])
            if not graph["same"]:
                numbers += f"; the replay NOT the plain answer, {graph['apart']:.1e} apart at most"
            if graph["turned"] is False:
                numbers += "; the replay NOT for its rows turned round"
        elif isinstance(graph, str):
            failed.append(f"  {named(largest)}, the graph of the batch filled out: {graph}")
        by_graphs += len(inside) * shorter / 1000
        out.append(
            f"{named(largest):>12} {told['size']:>6} {told['answer']:>11.2f} {plain['answer']:>8.2f} "
            f"{replay:>9} {told['whole']:>8.2f} {whole:>8}   {numbers}"
        )
    if recorded:
        out += [
            "",
            (
                f"By those, the calls of an update take {by_graphs:.2f} s with a graph of fixed sizes "
                f"wherever it is the shorter way, where they take {plainly:.2f} s."
            ),
        ]
    return out


def report(args: argparse.Namespace, played: dict[str, Any], found: dict[str, Any]) -> list[str]:
    against = f"{args.itself} of them against itself" if args.itself else "against the pilots"
    library = "the processor"
    if args.device == "cuda":
        library = torch.cuda.get_device_name(0)
    sizes = played["sizes"]
    out = [
        f"A call of the model in an update's games: {args.checkpoint}",
        (
            f"  device {args.device} ({library}), torch {torch.__version__}, {args.duels} duels, "
            f"{against}, a pool of {args.pool_threads} threads with {args.pool_slots} duels each"
        ),
        (
            f"  {sum(sizes):,} decisions in {len(sizes):,} calls, the model's turns "
            f"{played['model_seconds']:.2f} s of the {played['games_seconds']:.2f} s the games took"
        ),
        f"  each part the median of {args.repeats} runs, in ms, the card waited for after it",
        "",
        (
            f"{'Decisions':>12} {'Calls':>6} {'Their':>10} {'Timed':>6} {'A call':>8} "
            + " ".join(f"{title:>11}" for _, title in PARTS)
        ),
        f"{'in a call':>12} {'':>6} {'decisions':>10} {'at':>6} {'whole':>8}",
    ]
    by_calls = 0.0
    for largest in CLASSES:
        inside = [size for size in sizes if of_class(size) == largest]
        told = found["calls"].get(largest)
        if not inside or told is None:
            continue
        by_calls += len(inside) * told["whole"] / 1000
        out.append(
            f"{named(largest):>12} {len(inside):>6,} {sum(inside):>10,} {told['size']:>6} "
            f"{told['whole']:>8.2f} " + " ".join(f"{told[part]:>11.2f}" for part, _ in PARTS)
        )
    out += [
        "",
        (
            f"By those, the calls of an update take {by_calls:.2f} s: a call's whole time by the "
            "calls of its class."
        ),
    ]
    for size, apart in found["apart"].items():
        out += ["", f"The answer taken apart, a call of {size} decisions (how often, ms in all):"]
        for label, count, took in sorted(apart["encoders"], key=lambda told: -told[2]):
            out.append(f"  {label:<34} {count:>3} {took:>8.2f}")
        counted = apart["launched"]
        if isinstance(counted, str):
            out.append(f"  what an answer launches was not counted: {counted}")
            continue
        out.append(
            f"  an answer calls {counted['called']:,.0f} operations of PyTorch, which make "
            f"{counted['operations']:,.0f} with those they call"
            + (f", and launches {counted['on_card']:,.0f} kernels and copies on the card" if counted["on_card"] else "")
        )
        out.append(
            "  the most called: "
            + ", ".join(f"{name.removeprefix('aten::')} {count:.0f}" for name, count in counted["most"])
        )
    out += graphs_report(played, found)
    if args.device != "cuda":
        out += ["", "On a processor the times say nothing of a card, and no graph is recorded."]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--itself", type=int, default=0, help="duels of the update against itself")
    parser.add_argument("--duels", type=int, default=520)
    parser.add_argument("--pool-threads", type=int, default=20)
    parser.add_argument("--pool-slots", type=int, default=16)
    parser.add_argument("--learner", default="blackwing")
    parser.add_argument("--opponents", default=OPPONENTS)
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument(
        "--members-width", type=int,
        help="the answers' members the trainer's graphs are fixed at, where not its own",
    )  # fmt: skip
    parser.add_argument(
        "--compile", action="store_true",
        help="the batches filled out answered by the model compiled for their sizes too",
    )  # fmt: skip
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("No card PyTorch can use here: --device cpu tries the tool")
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    board = saved.get("metadata", {}).get("board_representation", "mean-pool")
    del saved
    config = TrainerConfig(
        workers=2, duels=args.duels, device=args.device, seed=2012, learner_deck=args.learner,
        opponent_decks=args.opponents.split(","), pool_threads=args.pool_threads,
        pool_slots=args.pool_slots, ppo_epochs=1, board_representation=board,
        self_play_duels=args.itself,
        # Every call as it is, and every duel together: the calls kept here
        # are those the model answers itself, whatever a run takes on a card.
        pool_groups=1, pool_groups_over=0, inference_graphs=0,
    )  # fmt: skip
    with PpoTrainer(ROOT, config) as trainer:
        device = trainer.device
        load(args.checkpoint, trainer.model, None, map_location=device, expected_board_representation=board)
        model = trainer.model
        kept = Kept()
        watching = model.register_forward_pre_hook(kept)
        started = time.perf_counter()
        _, rolled = trainer.collect()
        played = {
            "sizes": kept.sizes,
            "games_seconds": time.perf_counter() - started,
            "model_seconds": float(rolled["inference_seconds"]),
        }
        watching.remove()
        found: dict[str, Any] = {"calls": {}, "apart": {}}
        calls = {largest: Call(model, device, arrays) for largest, arrays in sorted(kept.batches.items())}
        for largest, call in calls.items():
            found["calls"][largest] = {"size": call.size, **call.measured(args.repeats)}
        # The batch in one copy, and on a card the answer replayed from a graph.
        found["copies"] = {
            largest: tried(in_one_copy, call, args.repeats) for largest, call in calls.items()
        }
        # Each batch filled out to the widest of them, as sizes fixed in
        # advance would have it.
        found["widest"] = kept.widest
        wide: dict[int, Call | str] = {
            largest: tried(Call, model, device, filled(arrays, found["widest"]))
            for largest, arrays in sorted(kept.batches.items())
        }
        found["filled"] = {
            largest: tried(filled_out, calls[largest], call, args.repeats) if isinstance(call, Call) else call
            for largest, call in wide.items()
        }
        found["graphs"], found["wide_graphs"] = {}, {}
        if device == "cuda":
            found["graphs"] = {
                largest: tried(replayed, call, args.repeats) for largest, call in calls.items()
            }
            found["wide_graphs"] = {
                largest: tried(replayed, call, args.repeats)
                for largest, call in wide.items()
                if isinstance(call, Call)
            }
        if args.compile and not headers():
            found["compiled"] = (
                "the compiler wants Python's own headers (Python.h), which this Python has not: "
                "tools/update_timing.py says how to borrow them"
            )
        elif args.compile:
            # A step of a pass is timed at the batch nearest a minibatch's 256 decisions.
            sized = {largest: call for largest, call in wide.items() if isinstance(call, Call)}
            nearest = min(sized, key=lambda largest: abs(sized[largest].size - 256)) if sized else None
            few = max(10, args.repeats // 10)
            steps: dict[str, Any] = {}
            if nearest is not None:
                steps = {
                    "as a pass has it, the batch at its own sizes": tried(stepped, calls[nearest], model, few),
                    "the batch filled out": tried(stepped, sized[nearest], model, few),
                }
            found["compiled"] = {}
            for way, lstm_too in (("the LSTM with the rest", True), ("the LSTM left to cuDNN", False)):
                answerer = tried(compiling, model, lstm_too)
                if isinstance(answerer, str):
                    found["compiled"][way] = answerer
                    continue
                print(f"compiling, {way}: a version for each batch, without a word", flush=True)
                found["compiled"][way] = {
                    "calls": {
                        largest: tried(compiled_call, call, answerer, args.repeats)
                        for largest, call in sized.items()
                    }
                }
                if nearest is not None:
                    steps[f"the batch filled out, compiled, {way}"] = tried(
                        stepped, sized[nearest], answerer, few
                    )
            if nearest is not None:
                found["steps"] = {"size": sized[nearest].size, "ways": steps}
            tried(torch._dynamo.reset)
        # The smallest and the largest call are taken apart.
        for largest in sorted({min(calls), max(calls)}):
            call = calls[largest]
            try:
                counted: dict[str, Any] | str = launched(call)
            except Exception as error:  # noqa: BLE001 - the times stand without the count
                counted = f"{type(error).__name__}: {error}"
            found["apart"][call.size] = {
                "encoders": encoders(call, max(10, args.repeats // 10)),
                "launched": counted,
            }
        out = report(args, played, found)
        print("\n".join(out), flush=True)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text("\n".join(out) + "\n", encoding="utf8")
        # Last, with the report written: what a card may not take well.
        found["trainer"] = tried(
            trainers_way, model, device, calls, args.repeats, args.members_width
        )
        sized = {largest: call for largest, call in wide.items() if isinstance(call, Call)}
        if sized:
            nearest = min(sized, key=lambda largest: abs(sized[largest].size - 256))
            few = max(10, args.repeats // 10)
            found["step_graphs"] = {"size": sized[nearest].size, "ways": {}}
            for label, lookups in (
                ("The tables looked up as the model looks them up", False),
                ("The tables looked up by index_select", True),
            ):
                found["step_graphs"]["ways"][label] = tried(recorded_step, sized[nearest], few, lookups)
        more = last_report(found)
        print("\n".join(more), flush=True)
        if args.output:
            with args.output.open("a", encoding="utf8") as told:
                told.write("\n".join(more) + "\n")


if __name__ == "__main__":
    main()
