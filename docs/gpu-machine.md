# GPU machine runbook

## Prerequisites

- NVIDIA driver visible through `nvidia-smi`.
- A Rust toolchain (`cargo`) and a C++ compiler: the engine is compiled from
  source with the duel environment.
- Python 3.11 to 3.13 and `uv`.

The .NET SDK is no longer needed.

## Setup, and after every pull that moves a submodule or the Rust code

From the repository root:

```sh
git submodule update --init --recursive
cargo build --release
uv sync
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

`git submodule update` fetches the pilots and, through them, the engine, the
card scripts and the card database. If the last command reports `False`,
install the CUDA wheel recommended by the current PyTorch installation
instructions, then rerun `uv sync`.

`runs/` and `analysis/` are not in the repository. They live at the
repository root on this machine and are never touched by a pull.

The engine and the pilots are compiled into the environment's library, while
the card scripts and the card database are read when a duel starts. A library
left unbuilt after a pull that moved the pin would therefore run the old
engine on the new scripts. `cargo build --release` does nothing when the library is up
to date, and the run scripts under `experiments/commands/` written for this
setup begin with it and with a check that every submodule is at its pin.

## Smoke validation

```sh
cargo test --release
uv run python -m pytest -q
uv run python -m ygo_training train --device cuda \
  --run /tmp/ygo-smoke --learner blackwing --opponents monarch \
  --workers 2 --duels 4 --updates 1 --imitation-only-updates 1 --eval-every 0
```

## Training

```sh
uv run python -m ygo_training train --device cuda \
  --run runs/example --updates 100
```

Metrics include duels and decisions per second, inference timing and observed
batch sizes. Resume the latest checkpoint of a run with `--run runs/example
--continue`: the model and the optimizer are restored before the next
numbered checkpoint is written. `--updates` is the number of additional
updates of that invocation, not a target version.

A checkpoint takes its name once it is written whole and its update has
been evaluated ([resume and run integrity](commands.md#resume-and-run-integrity)).
A run killed before that goes on, with `--continue`, from the checkpoint
before, and writes over the `model-NNNNN.pt-tmp` the kill left.

Checkpoints written before the duel environment replaced the .NET worker
(formats up to 12) do not load, and the code that ran them is not part of
this repository.
