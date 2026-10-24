"""How long a pass over an update's decisions takes, by the size of its
minibatches.  Timing only: nothing here says how a model learns.

    uv run python tools/update_timing.py CHECKPOINT --device cuda \\
        --pool-threads 20 --output analysis/update-timing/report.txt

Plays one update's games with the checkpoint's model in the learner's seat,
through the library's pool, and makes one pass of the PPO update over their
decisions for each size, from the checkpoint's weights every time.  It
reports what comes before a pass (the games' estimates worked out again,
and the advantages), then a pass's steps, its seconds, the milliseconds of
a step and the card's memory at its peak, and where a step's time goes at
every size.

What it is for: a step has a cost that does not depend on how many
decisions it holds, and one that grows with them.  If a step of four times
the decisions costs little more, a pass with larger minibatches is that much
shorter, and arithmetic is not what the pass waits for.

With ``--compile`` all of it is measured a second time with the model
compiled by ``torch.compile``, which training does not use: what compiling
would save, what it costs at the start of a process, and whether the model
still answers with the same numbers.  The compiler builds its kernels with a
C compiler and Python's own headers (``Python.h``).  A Python of uv's has
them; for another one they can be borrowed:

    uv python install 3.13.7 --no-bin
    CPATH=$(uv python dir)/cpython-3.13.7-linux-x86_64-gnu/include/python3.13 \\
        uv run python tools/update_timing.py CHECKPOINT --compile ...
"""

from __future__ import annotations

import argparse
import copy
import os
import platform
import sys
import sysconfig
import time
import traceback
from collections.abc import Callable
from pathlib import Path

import torch
import torch._dynamo
from torch._dynamo.utils import counters

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ygo_training.checkpoints import load
from ygo_training.config import TrainerConfig
from ygo_training.ppo_trainer import PpoTrainer

# A step taken apart is timed over this many minibatches.
SPLIT_STEPS = 12


def synchronize(device: str) -> None:
    """The card answers when it likes: wait for it before reading a clock."""
    if device == "cuda":
        torch.cuda.synchronize()


def one_pass(trainer: PpoTrainer, trajectories: list[dict]) -> tuple[float, int]:
    """The seconds and the steps of one pass of the update, as training makes it."""
    device = trainer.device
    synchronize(device)
    started = time.perf_counter()
    trainer.policy_optimizer.begin_update(1)
    metrics = trainer.update(trajectories, 0.0)
    synchronize(device)
    return time.perf_counter() - started, int(metrics["ppo_optimizer_steps"])


def step_apart(trainer: PpoTrainer, trajectories: list[dict]) -> dict[str, float]:
    """Milliseconds of a step's parts, over a few minibatches of the update:
    the batch built where the decisions are kept, its way to the card, the
    model's answer, the gradient, and the optimizer's step.

    The loss here is a plain one of the same shape (the answer taken, the
    value against the result): what it costs is the model's, not the loss's.
    """
    device = trainer.device
    optimizer = trainer.policy_optimizer
    samples, _, _ = optimizer.prepare_samples(trajectories)
    chunks = optimizer._chunks(samples, list(range(len(samples))), trainer.config.minibatch)
    spent = dict.fromkeys(("batch", "to the card", "model", "gradient", "optimizer"), 0.0)
    counted = 0
    trainer.model.train()
    for indices in chunks[:SPLIT_STEPS]:
        steps = [samples[index].step for index in indices]
        clock = time.perf_counter()
        arrays = trainer._pool().batch([int(step["row"]) for step in steps])
        tensors = {name: torch.from_numpy(array) for name, array in arrays.items()}
        spent["batch"] += time.perf_counter() - clock
        clock = time.perf_counter()
        if device != "cpu":
            tensors = {name: value.to(device) for name, value in tensors.items()}
        synchronize(device)
        spent["to the card"] += time.perf_counter() - clock
        clock = time.perf_counter()
        logits, values = trainer.model(tensors)
        chosen = torch.tensor([int(step["chosen"]) for step in steps], device=device)
        returns = torch.tensor([samples[index].return_ for index in indices], device=device)
        loss = torch.nn.functional.cross_entropy(logits, chosen) + (values - returns).pow(2).mean()
        synchronize(device)
        spent["model"] += time.perf_counter() - clock
        clock = time.perf_counter()
        trainer.optimizer.zero_grad()
        loss.backward()
        synchronize(device)
        spent["gradient"] += time.perf_counter() - clock
        clock = time.perf_counter()
        torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), 1.0)
        trainer.optimizer.step()
        synchronize(device)
        spent["optimizer"] += time.perf_counter() - clock
        counted += 1
    return {part: 1000.0 * seconds / max(1, counted) for part, seconds in spent.items()}


def headers() -> bool:
    """Whether Python's own headers are where a C compiler will look."""
    folders = [sysconfig.get_paths()["include"], *os.environ.get("CPATH", "").split(os.pathsep)]
    return any(folder and (Path(folder) / "Python.h").exists() for folder in folders)


def compile_model(model: torch.nn.Module) -> None:
    """Compile what every caller of the model goes through.

    Three settings, each for what this model is.  Every size of a batch is
    left free from the start, since no two batches have the same.  The LSTM
    is compiled with the rest: PyTorch steps around one unless told, which
    leaves the model in pieces.  And a version is compiled for every kind of
    batch that turns up: PyTorch stops at eight and from then on runs the
    plain model for the others without a word.
    """
    torch._dynamo.config.allow_rnn = True
    torch._dynamo.config.recompile_limit = 64
    model.evaluate = torch.compile(model.evaluate, dynamic=True)


def furthest(one: torch.Tensor, other: torch.Tensor) -> float:
    """The largest difference between two answers, where both are numbers."""
    both = torch.isfinite(one) & torch.isfinite(other)
    return float((one[both] - other[both]).abs().max()) if bool(both.any()) else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--workers", type=int, default=2, help="the trainer's own, idle here")
    parser.add_argument("--pool-threads", type=int, default=20)
    parser.add_argument("--pool-slots", type=int, default=16)
    parser.add_argument("--duels", type=int, default=520)
    parser.add_argument("--learner", default="blackwing")
    parser.add_argument(
        "--opponents", help="decks the learner meets, by commas; all others if left out"
    )
    parser.add_argument("--sizes", default="256,512,1024,2048,4096")
    parser.add_argument("--seed", type=int, default=2012)
    parser.add_argument(
        "--compile", action="store_true", help="all of it again with the model compiled"
    )
    parser.add_argument(
        "--matmul-precision",
        choices=("highest", "high", "medium"),
        default="highest",
        help="torch.set_float32_matmul_precision: training leaves it at highest",
    )
    parser.add_argument(
        "--fused-adam",
        action="store_true",
        help="Adam's step as one kernel (fused=True), which training does not ask for",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.set_float32_matmul_precision(args.matmul_precision)
    sizes = [int(size) for size in args.sizes.split(",")]
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    board = saved.get("metadata", {}).get("board_representation", "mean-pool")
    del saved
    config = TrainerConfig(
        workers=args.workers,
        duels=args.duels,
        device=args.device,
        seed=args.seed,
        learner_deck=args.learner,
        opponent_decks=args.opponents.split(",") if args.opponents else None,
        pool_threads=args.pool_threads,
        pool_slots=args.pool_slots,
        # The games are only what the passes are timed on: asked as they
        # were when the figures in docs/commands.md were taken.
        pool_groups=1,
        pool_groups_over=0,
        inference_graphs=0,
        ppo_epochs=1,
        minibatch=sizes[0],
        board_representation=board,
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text("", encoding="utf8")

    def say(line: str = "") -> None:
        """To the terminal and to the report at once: a size too large for
        the card's memory ends the run, and what was measured stays."""
        print(line, flush=True)
        if args.output:
            with args.output.open("a", encoding="utf8") as report:
                report.write(line + "\n")

    with PpoTrainer(ROOT, config) as trainer:
        device = trainer.device
        if args.fused_adam:
            trainer.optimizer = torch.optim.Adam(
                trainer.model.parameters(), lr=config.learning_rate, fused=True
            )
            trainer.policy_optimizer.optimizer = trainer.optimizer
        load(
            args.checkpoint,
            trainer.model,
            None,
            map_location=device,
            expected_board_representation=board,
        )
        weights = copy.deepcopy(trainer.model.state_dict())
        fresh = copy.deepcopy(trainer.optimizer.state_dict())

        def again() -> None:
            """The checkpoint's weights and an optimizer that has taken no step."""
            trainer.model.load_state_dict(weights)
            trainer.optimizer.load_state_dict(fresh)

        name = torch.cuda.get_device_name(0) if device == "cuda" else "the processor"

        def measure(say: Callable[[str], None]) -> list[dict]:
            """All the report says, of the model as it is called now."""
            trainer.config.minibatch = sizes[0]
            again()
            # The same deals and the same draws every time it is asked.
            trainer.set_rollout_index(0)
            trainer.set_training_schedule_index(0)
            started = time.perf_counter()
            trajectories, rolled = trainer.collect()
            played = time.perf_counter() - started
            # What an update does before its first pass, each timed the
            # second time: the first of anything in a process pays for the
            # others.
            before = {}
            for part, work in (
                ("estimates", trainer.policy_optimizer.recompute_rollout_estimates),
                ("advantages", trainer.policy_optimizer.prepare_samples),
            ):
                work(trajectories)
                synchronize(device)
                started = time.perf_counter()
                work(trajectories)
                synchronize(device)
                before[part] = time.perf_counter() - started
            decisions = sum(len(trajectory["steps"]) for trajectory in trajectories)
            say(
                f"{len(trajectories)} games and {decisions} decisions (played in {played:.1f} s, "
                f"the model's turns {rolled['inference_seconds']:.1f} s of them, "
                f"{rolled['mean_inference_batch']:.0f} decisions a batch) on {name}"
            )
            say(
                f"before a pass: the games' estimates worked out again "
                f"{before['estimates']:.2f} s, the advantages {before['advantages']:.2f} s "
                "(a pass below includes the second)"
            )
            say("")
            say(
                f"  {'minibatch':>9}  {'steps':>5}  {'a pass':>8}  {'a step':>9}  {'memory':>8}   "
                "of a step: batch, to the card, model, gradient, optimizer"
            )
            # Once without a clock: the first pass of a process pays for
            # what the later ones find ready.
            one_pass(trainer, trajectories)
            first = None
            for size in sizes:
                trainer.config.minibatch = size
                again()
                if device == "cuda":
                    torch.cuda.reset_peak_memory_stats()
                try:
                    seconds, steps = one_pass(trainer, trajectories)
                    peak = torch.cuda.max_memory_allocated() / 2**30 if device == "cuda" else 0.0
                    again()
                    parts = step_apart(trainer, trajectories)
                except torch.OutOfMemoryError:
                    say(f"  {size:>9}  does not fit in the card's memory")
                    break
                first = first or seconds
                say(
                    f"  {size:>9}  {steps:>5}  {seconds:6.2f} s  {1000 * seconds / steps:6.1f} ms  "
                    f"{peak:5.2f} GB   "
                    + ", ".join(f"{value:.1f}" for value in parts.values())
                    + f" ms   ({first / seconds:.2f} times the first)"
                )
            return trajectories

        say(f"A pass over an update's decisions, by the size of its minibatches: {args.checkpoint}")
        if args.matmul_precision != "highest":
            say(
                f"float32 matrix products at '{args.matmul_precision}' precision, which a card "
                "with tensor cores multiplies with fewer digits; training uses 'highest'"
            )
        if args.fused_adam:
            say("Adam's step fused into one kernel; training's takes its parameters by lists")
        trajectories = measure(say)
        if args.matmul_precision != "highest":
            # The same minibatch at this precision and at training's.
            first = [step for trajectory in trajectories for step in trajectory["steps"]]
            first = first[: sizes[0]]
            again()
            trainer.model.eval()
            with torch.no_grad():
                lower = trainer.policy_optimizer.forward_steps(trainer.model, first)
                torch.set_float32_matmul_precision("highest")
                full = trainer.policy_optimizer.forward_steps(trainer.model, first)
                torch.set_float32_matmul_precision(args.matmul_precision)
            trainer.model.train()
            apart = [furthest(one, other) for one, other in zip(lower, full, strict=True)]
            say("")
            say(
                f"one minibatch's answers at this precision and at 'highest' differ by at most "
                f"{apart[0]:.1e} in an answer's score and {apart[1]:.1e} in the value"
            )
        if not args.compile:
            return
        say()
        say("The same with the model compiled (every size left free, the LSTM with the rest)")
        if not headers():
            version = platform.python_version()
            folder = f"cpython-{version}-linux-x86_64-gnu/include/python{version.rsplit('.', 1)[0]}"
            say("not compiled: the compiler wants Python's own headers (Python.h), which this")
            say("Python has not.  To borrow them from a Python of uv's:")
            say(f"    uv python install {version} --no-bin")
            say(f"    CPATH=$(uv python dir)/{folder} uv run python {' '.join(sys.argv)}")
            return

        def compiled_too() -> None:
            # One minibatch's answers both ways, while the pool still keeps its
            # decisions: the games played next take their place.
            optimizer = trainer.policy_optimizer
            chunk = [step for trajectory in trajectories for step in trajectory["steps"]][
                : sizes[0]
            ]
            again()
            trainer.model.eval()
            with torch.no_grad():
                plain = optimizer.forward_steps(trainer.model, chunk)
            started = time.perf_counter()
            compile_model(trainer.model)
            with torch.no_grad():
                compiled = optimizer.forward_steps(trainer.model, chunk)
                repeated = optimizer.forward_steps(trainer.model, chunk)
            differences = [furthest(one, other) for one, other in zip(plain, compiled, strict=True)]
            same = all(torch.equal(one, other) for one, other in zip(plain, compiled, strict=True))
            itself = all(
                torch.equal(one, other) for one, other in zip(compiled, repeated, strict=True)
            )
            # As training leaves it between updates: a version is compiled for
            # each of the two modes, and the games are played in this one.
            trainer.model.train()
            # All of it once without a clock: every version gets compiled there.
            print("compiling: all of it is run once first, without a word", flush=True)
            measure(lambda line="": None)
            synchronize(device)
            compiling = time.perf_counter() - started
            versions = int(counters["frames"]["ok"])
            started = time.perf_counter()
            say(
                "its answers to one minibatch: "
                + (
                    "the same numbers as the plain model's"
                    if same
                    else f"not the plain model's, by at most {differences[0]:.1e} in an answer's "
                    f"score and {differences[1]:.1e} in the value"
                )
                + ("; the same twice" if itself else "; NOT the same twice")
            )
            measure(say)
            synchronize(device)
            timed = time.perf_counter() - started
            say("")
            say(
                f"compiling: {max(0.0, compiling - timed):.0f} s more the first time all of this "
                f"ran than the second, for {versions} versions; {int(counters['frames']['ok']) - versions} "
                "more were compiled while the clock ran"
            )

        try:
            compiled_too()
        except Exception as error:
            # Never run on this machine before, perhaps: what went wrong
            # belongs in the report, which is what gets read afterwards.
            say(f"the compiled model failed: {type(error).__name__}")
            for line in str(error).splitlines()[:8]:
                say("    " + line[:240])
            if args.output:
                whole = args.output.with_suffix(".error.txt")
                whole.write_text(traceback.format_exc(), encoding="utf8")
                say(f"all of the error is in {whole}")
            raise


if __name__ == "__main__":
    main()
