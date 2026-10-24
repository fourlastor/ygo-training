# Training architecture

```text
Python: trainer, schedule, the model, optimizer
        |  ctypes: answers in, a round's batch out, arrays not JSON
        v
crates/ygo-env (Rust): a pool of duels on threads; the duel, its seats,
the pilots, recorded duels
        |  crates of native/ygo-policies          |  crates of native/ygo-model-input
        v                                         v
ygo-policies-duel -> OCGCore, card scripts,    a seat's question as a view,
card database                                  a view as the model's arrays
```

The duel is the one `policy-bench` plays: the same deal from a seed, the same
messages to the same seats, pilots seeded the same way. Two pilots seated in
the environment play the bench's duel digest for digest, which the crate's
tests check on every build.

## The duel environment (`crates/ygo-env`)

Each seat of a duel is one of three things:

- a **pilot**: the policy written for the seat's deck plays it;
- the **caller**: the trainer decides. `advance` stops at each decision and
  `choose` answers it. The pilot of the seat's deck is asked alongside and its
  answer comes with the decision, as the label to imitate;
- a **record**: a recorded duel (the form `policy-bench --record true`
  writes) is played again, answer by answer.

`seat.rs` holds the seat. It is built from the pieces the policies' own
`Seat` is made of (redaction, projection, `decision::prompt`, `Sequential`),
used in the same order, and returns what it is asked instead of asking a
policy. A decision therefore shows the network what a pilot is given at the
same point, and nothing else.

The view turns that into what the trainer receives: players relative to the
seat, cards named by their place in the view, and no zone numbers. It is
made in `native/ygo-model-input` (`seat/src/view.rs`), see below.

Three rules keep decisions that are not a network's to take away from it:

- the zone a card goes to is the pilot's choice;
- a decision with a single answer takes it;
- hand shuffling and the cancelling of a selection are never offered. Passing
  a chain window, answering no and finishing a selection are.

Two keep it from answers that change nothing (`Trail::offered`, see
[commands.md](commands.md)): a command, or an activation at a chain window,
given at the same position earlier in the turn, and the activation of a
monster's effect while it is face-up on the field with its effects negated.

The engine still supplies every legal answer; nothing is filtered to give the
network a tactical answer.

### Many duels at once (`pool.rs`)

Every game is played through the pool: threads, each with its engine and
its duels in flight (`--pool-threads`, `--pool-slots`).

- A **round** answers every waiting duel, takes each to its next decision or
  its end, and returns the decisions as one batch, in the order of their
  duels. It does not depend on the threads' pace.
- `features.rs` and `pack.rs` of `native/ygo-model-input` (`input/src`)
  build that batch, as they build every array the model reads.
- Decisions of duels submitted to be kept stay in the pool, each under a
  row. An update's minibatches are built from them by row; Python never
  holds those decisions.
- A duel submitted to show its decisions hands over the view of each with
  it, for what is read of a decision beside the model's input: a search's
  log of a recorded duel, the lead head followed.

Until 2026-10-16 worker processes, one duel each, were another route, and
recorded duels and the lead head stayed with them; they were dropped
that day.

### The model's input, shared (`native/ygo-model-input`)

What turns a seat's question into the arrays the model reads is a
repository of its own, a submodule here, so that another project that seats
a checkpoint (a web client, say) shows it what its training showed it. Two
crates, with no workspace of their own: they are members of this one and
take the policies from its `[workspace.dependencies]`.

- `input` (`ygo-model-input`): the view as plain data, printed card data
  from a card's row of the database, and the arrays (`pack`, `batch`). It
  depends on `serde` alone and builds on its own and for WebAssembly.
- `seat` (`ygo-model-seat`): the view made from a seat's `Observation` and
  `Decision`, the seat's own trail, and `Caller`, the policy of a seat whose
  decisions go to a closure. The seat at an EDOPro table (`edopro.rs`) is
  that policy with a closure that asks the trainer.

`crates/ygo-env` keeps what reads the card database file, and modules named
`view`, `trail`, `features` and `pack` that pass the shared ones on.

That repository is the input's one definition (decided on 2026-10-16): the
trainer has no packing of its own, and asks the library for the arrays of
any decision outside its pool too (`Environment.pack`, `Environment.batch`).
`ygo_training/schema.py` describes the arrays for the model and for
checkpoints, and `tests/test_packed.py` holds it, and what the input says of
a card or a decision, to what the library builds.
`native/ygo-model-input/input/vectors/decisions.json` holds what the
library makes of some 130 decisions, which that repository tests itself
against: `tools/model_input_vectors.py` writes the file, and
`tests/test_model_input_vectors.py` fails when it is not what the library
gives now. Until 2026-10-16 a Python reference (`packed.py`,
`tensorize.py`, `card_features.py`) built the same arrays and the two were
held together byte for byte.

A **transaction** is a command (a Main or Battle Phase action, a chain
response) and the selections it leads to, until the other seat has something
to answer. An attack and its target are one transaction, as are the steps of
a card selection. The optimizer treats a transaction as one action: its
steps share one advantage and one term of the imitation loss, and the value
head has one target, at its first step. The head is measured there too; what
it says at a later step is used nowhere.

## Python module boundaries

| Module | Responsibility |
|---|---|
| `environment.py` | Load the environment's library; start duels, advance them, answer; the model's input of any decision |
| `schema.py` | The arrays the model reads: names, shapes, limits, vocabularies; the input contract a checkpoint records and is checked against |
| `model.py` | The policy/value network |
| `pool.py` | The library's pool of duels, and every game played through it: an update's, an evaluation's, recorded duels |
| `searched.py` | What a search's log says of a recorded duel's decisions, read from their views; the lead head's rule |
| `orchestration.py` | The games of an update: deals, decks, who sits where; the rollout metrics |
| `inference.py` | How an answer is drawn from its seed; a network asked outside the pool |
| `optimization.py`, `ppo.py` | Returns, advantages, the PPO and imitation objectives |
| `schedules.py`, `rewards.py` | Seeds and schedules; the outcome reward |
| `records.py` | The recorded duels of a search run, their order, which an update imitates, and what the search's log measured |
| `ppo_trainer.py`, `trainer.py` | The trainer and its composition |
| `checkpoints.py`, `run_repository.py` | Checkpoints and run metadata |
| `evaluation.py`, `evaluation_suite.py` | Evaluation summaries and suite files |
| `metrics.py`, `training_logging.py` | The metrics stream and the status tables |
| `config.py`, `cli_parser.py`, `__main__.py` | Configuration and commands |

## What the network is shown

The library builds, from one decision (`input/src/pack.rs` of
`native/ygo-model-input`):

- **every card the seat sees**, where it lies and as it is now: identity and
  printed data, location, battle position, controller, current ATK and DEF,
  level, counters, whether it can attack, how often it battled this turn, and
  what is happening to it (selected, attacking, attacked, the event's card,
  in the chain, targeted). A card the seat cannot see has its place and those
  events, and no identity;
- **the cards the decision is about**, each in a slot of its own: the last
  two links of the chain, the attacker and its target, the card of the event
  being responded to, the question's subject, and the card of the command the
  decision follows from (the attacker whose target is being chosen);
- **the question**: its kind, the engine's hint, bounds, and where the duel
  stands (Life Points, turn, phase, pile sizes), all from the seat's side;
- **the answers**, each with its kind, its card and its member cards;
- **the seat's own earlier answers**, the last 32.

Of the decks it holds only how many cards each pile has: the deck lists
were dropped with format 15 (`experiments/2026-10-15-without-deck-lists.md`).

One encoder of a card where it lies serves the state, the context and the
answers, so a target is seen with its current statistics wherever it is
named. Card identities are two 16-bit components, without collisions. Battle
position is kept; zone numbers never reach the trainer.

With `--board-representation individual-tokens` each answer also attends to
every card on the field; `mean-pool` pools them. Both own the same
parameters.

The network gives every answer a score, the state a value, and every answer
a second number: the *lead* it predicts for that answer over the pilot's,
how much more often it wins the duel, in units of win rate, as a search
measures it by playing both out in the same worlds (`PolicyValueNet.evaluate`).
The head is taught at the decisions a one-step search examines and asked
only there (`searched.Reading.examined`); a model that follows it leaves its
policy's first choice for the alternative it predicts the largest lead for,
when that lead is at least a margin (`searched.Reading.led`).

The current checkpoint format is 15: format 14 (format 13 with the lead
head) without the deck lists in the input. Checkpoints of earlier formats
do not load; the revision before format 15, `05d27141`, runs them. A
checkpoint records the input schema it was trained on and is refused when
the runtime's differs.

## Training

Rewards are duel outcomes only: `+1` for a win, `-1` for a loss or a duel
stopped at the decision limit, `-0.5` for a draw. Returns use GAE over
transactions.

An **imitation-only update** has the pilot play the learner's seat; the
learner is trained on its decisions (and, with `--imitation-value-weight`,
its value head on the duels' outcomes). In every other update the learner
plays, and `--teacher-imitation-weight` adds the pilot's answers to its own
decisions as an auxiliary loss. A zero `--policy-loss-weight` with that
leaves only the labels (DAgger).

With `--imitation-records` the imitation-only updates play recorded duels
again instead: the rows of a `policy-bench search --record true` run, each
of which holds one deal as the pilot played it and as the pilot with the
search did. The answers of the record are the labels, so a learner can
imitate the search without running it. A record is imitated only if it plays
again with the digest it was recorded with: at other revisions of the engine,
the scripts, the card database or the pilots its answers may all still be
accepted while the questions are no longer the ones that were answered.

A run made with `--log true` also holds, for every answer the search
examined, what it tried and how each alternative did in each world.
`records.py` keeps of that the lead of every alternative over the pilot's
answer in the worlds of the first stage, which all of them go through.
`evaluate-search` reads a model's answers by those leads. For that the
alternatives of the decision are listed again from what the network is
offered, as the bench lists them (`Reading.alternatives`: the pilot's
answer first, like cards in one pile as one, twelve at most), and a log
that names others is refused.

Rollout seeds rotate through the signed 31-bit range: the run seed picks a
starting slot and each rollout owns 1,024 duel seeds. The learner's seat in
each duel is drawn by share (`--workers`, as the worker processes once drew
it), and every draw of an answer is seeded by its duel and its place in
it, so a rollout does not depend on how its decisions were batched.

Held-out evaluation plays fixed seeds with the model's first choice. It is
the progress signal; sampled rollout win rates and losses are diagnostics.
Every checkpoint is kept.

Run `uv run python -m pytest -q` and `cargo test --release` before relying on
a change.
