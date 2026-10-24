# Training neural Yu-Gi-Oh! players

This repository trains neural players of Yu-Gi-Oh! decks, by imitating the
rule-based pilots of [ygo-policies](native/ygo-policies) and then by PPO
against those pilots and against themselves, on OCGCore duels. One network
can play one deck or many. The engine is the rules authority: the network
only scores the answers the engine offers.

- `crates/ygo-env`: the duel environment, a Rust library on top of the
  ygo-policies crates. It runs the duels, seats the pilots, and hands the
  learner's decisions to the trainer.
- `ygo_training`: the trainer. The environment's pool plays an update's
  duels on threads, the network's calls are batched on the GPU, and the
  trainer updates it (imitation and PPO).
- `native/ygo-policies`: the pilots, their decks, and through its own
  submodules the engine, the card scripts and the card database.
- `native/ygo-model-input`: a seat's question as the model's input, in Rust.
  A repository of its own, shared with other projects that seat a
  checkpoint.
- `eval_suites`: held-out evaluation suites.
- `experiments`, `EXPERIMENT_LOG.md`: what was tried and found.
- `tools`: the experiments' reports and the tools that measure and inspect
  runs.

## Project names

The training project uses the following informal names:

- **The Seventh Samurai**: the learned model;
- **Shien**: the scripted teacher;
- **Shien's Dojo**: the training infrastructure as a whole.

These are flavour names, not CLI identifiers or experiment terminology.

## Where to read

- [docs/commands.md](docs/commands.md): the commands and what their options
  mean.
- [docs/architecture.md](docs/architecture.md): how the pieces fit, and what
  the network is shown.
- [docs/gpu-machine.md](docs/gpu-machine.md): setting up the training machine.
- [models/](models/): the networks released, a card each, and how a release
  is made.
- [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md): the research state. Read it, and the
  single active record under [experiments/](experiments/), before starting or
  resuming a run.

The records cite commits of the development history this repository was
started from, which is not published; earlier experiments, against an
earlier set of pilots, are not included.

Training checkpoints, metrics and run artifacts stay on the CUDA training
machine and are not expected in a development checkout. Findings go in the
relevant file under `experiments/`.

## Setup

```sh
git submodule update --init --recursive
cargo build --release
uv sync
uv run python -m pytest -q
```

`cargo build` compiles the engine and the environment
(`target/release/libygo_env.so`); it needs a Rust toolchain and a C++
compiler. The tests marked `integration` are skipped until it is built.

`bash tools/benchmark.sh` does this setup by itself on a machine that has
none of it (Rust by rustup, a Python environment by pip, in a folder of its
own) and measures how many duels an hour the machine trains in every mode:
see the section on a machine's speed in [docs/commands.md](docs/commands.md).

## A first run

A small run on the CPU, to check a machine:

```sh
uv run python -m ygo_training train --device cpu \
  --run /tmp/ygo-smoke --learner blackwing --opponents monarch \
  --workers 2 --duels 4 --updates 1 --ppo-epochs 1 --minibatch 64 \
  --imitation-only-updates 1 --eval-every 0
```

The two ways a run usually goes:

```sh
# The deck's pilot plays, the learner imitates it.
uv run python -m ygo_training train --device cuda --run runs/seed \
  --learner blackwing --updates 30 \
  --imitation-only-updates 30 --imitation-value-weight 0.5

# The learner plays; PPO, with the pilot's answers as an anchor.
uv run python -m ygo_training train --device cuda --run runs/anchored \
  --learner blackwing --updates 50 \
  --resume runs/seed/checkpoints/model-00029.pt --teacher-imitation-weight 0.1
```

Progress and evaluation:

```sh
uv run python -m ygo_training status runs/anchored/metrics.jsonl --ppo
uv run python -m ygo_training evaluate-suite runs/anchored/checkpoints/model-00079.pt \
  --device cuda --suite eval_suites/ygo-policies-blackwing-screen.json
# The reference: the same deals with the deck's pilot in the learner's seat.
uv run python -m ygo_training evaluate-suite --pilot \
  --suite eval_suites/ygo-policies-blackwing-screen.json
```

Held-out evaluation by the model's first choice is the evidence a model is
judged on. Sampled training rollouts are diagnostics.

## License

The code of this repository is licensed under the
[Apache License 2.0](LICENSE).

The duel engine, a fork of [ygopro-core](https://github.com/fourlastor/ygopro-core),
and ProjectIgnis' [card scripts](https://github.com/ProjectIgnis/CardScripts),
both brought in by the ygo-policies submodule, are licensed under the GNU
Affero General Public License v3. The environment library
(`target/release/libygo_env.so`) links the engine and runs the scripts, so
the library, and anything distributed with it built in, falls under the
AGPL-3.0. The card database comes from ProjectIgnis'
[BabelCdb](https://github.com/ProjectIgnis/BabelCdb), under its own terms.
