# Project context

This repository trains neural players for Yu-Gi-Oh! duels. `ocgcore` is the
rules authority: the duel environment presents the engine's requests, submits
the chosen response, and reflects the engine's messages. Do not duplicate
legality checks outside the engine.

The duels are run by the crates of the `native/ygo-policies` submodule: its
engine build, card data, scripts, decks and pilots. `crates/ygo-env` adds what
training needs on top (a seat that stops at a decision, a pilot asked
alongside, recorded duels played again, the same seat in a room of EDOPro)
and `ygo_training` is the trainer.
That submodule knows nothing about training, and should not: what the trainer
needs from a duel belongs in `crates/ygo-env`.

What turns a seat's question into the model's input is a second submodule,
`native/ygo-model-input`: a public repository of its own, shared with other
projects that seat a checkpoint (a web client, for one). `crates/ygo-env`
uses its two crates, and its modules of the same names only pass them on.
That repository is the input's one definition: the trainer has no packing
of its own. A change of the input is a commit in that repository with its
vectors written again (`tools/model_input_vectors.py`), and the new revision
pinned here.

## Training work

For PPO, imitation, or checkpoint work, read
[`EXPERIMENT_LOG.md`](EXPERIMENT_LOG.md) before changing the
model, tensorization, observations, or experiment protocol. It is a
newest-first index; detailed records live under
[`experiments/`](experiments/). Read the single `ACTIVE`
experiment before acting, and inspect linked historical or invalidated records
when they affect the work.

There must be exactly one active experiment. Each experiment belongs in its
own Markdown file under `experiments/`, containing its status,
hypothesis, protocol, evidence, and conclusion or current next step. When
starting a new experiment, first move the previous active record to
`HISTORICAL`, `COMPLETED`, or `INVALIDATED`, create the new `ACTIVE` record,
and add it at the top of the newest-first index. Do not append experiment
details directly to `EXPERIMENT_LOG.md`.

Keep the relevant experiment record and index current when training work
produces a material finding, changes the trusted protocol, invalidates earlier
evidence, promotes or withdraws a checkpoint, or changes a representation or
evaluation invariant. Do not record transient debug output, unverified
suspicions, routine run progress, or every intermediate checkpoint; record
them only after they affect an experimental decision or become reproducible.

Current representation invariants:

- OCGCore still supplies the complete set of legal answers. Do not filter
  losing battle targets, or anything else, to give the network the tactical
  answer. Five things only are kept from it: hand shuffling and the
  cancelling of a selection are never offered, which prevents no-progress
  loops; nor, for the same reason, is a command of the Main or the Battle
  Phase, or an activation at a chain window, that the seat gave at the same
  position earlier in the turn, which brought it back there (Lyla,
  Lightsworn Sorceress under Skill Drain; `Trail::offered` in
  `native/ygo-model-input/seat/src/trail.rs`, decided on
  2026-10-15 and, for chain windows, 2026-10-16); nor, where the seat may
  decline it, the activation of a monster's effect while it is face-up on
  the field with its effects negated, as the engine's status of the card
  tells (`STATUS_DISABLED`; the same `Trail::offered`, decided
  on 2026-10-22: Destructotron under Fiendish Chain, its cost paid each
  time by a Psychic Counter, looped where no position came back); the zone
  a card goes to is its pilot's choice; a decision with a single answer
  takes it. Passing a chain window, answering no and finishing a selection
  remain answers.
- The network's input is built only from what a pilot is given at the same
  decision: the `Observation` and `Decision` of a ygo-policies seat, through
  `native/ygo-model-input/seat/src/view.rs`. Visibility is decided before
  that, by the policies' own redaction and projection. The trainer adds printed card data and scales
  numbers; it must not add anything a seat does not see. The identity of a
  card the seat cannot see never enters the input. One exception, by the
  decision of 2026-10-22, never the policy's: a critic used in
  training only (`ygo_training/critic.py`) reads beside the seat's view
  the same decision with nothing hidden but the Decks' order, and with the
  next cards of each Deck where asked (a setup's `whole`, the `Sight` of
  `crates/ygo-env/src/seat.rs`, `DuelPool.batch(whole=True)`). The
  policy's input is the same with it or without
  (`tests/test_pool.py`), and nothing that plays holds the critic.
- The environment plays the duel `policy-bench` plays. Two pilots seated in it
  reproduce the bench's digest of every question and answer; keep that test.
- An attack and its target are two decisions of one transaction. A
  transaction is a command and the selections it leads to, until the other
  seat has something to answer, and the optimizer treats it as one action.
  Targets are only ever those the engine offers.
- Zone numbers do not reach the trainer. Battle position does: face-up
  Attack, face-up Defense and face-down determine the relevant statistic.
- A card is shown with its current ATK and DEF, each with its own scaling,
  beside its printed data. One encoder of a card where it lies serves the
  state, the decision's context and the answers.
- From checkpoint format 16 a model may be told more of a card than its
  input gives, by the code the input carries, from a table its checkpoint
  carries (`ygo_training/card_table.py`, `data/card-descriptions.npz`):
  its description, a fixed vector of its printed text, added to its
  identity; and its identity hidden, in training for a share of the cards
  drawn each update, at play for every card the model was not trained on.
  Neither tells the network anything a seat does not see: a card the seat
  cannot see has the code 0, which has no description and is never hidden.
  No array of the input changes.
- The model's input has one definition, in `native/ygo-model-input`
  (`input/src/pack.rs`; decided on 2026-10-16). The library
  builds every array the model reads, for the duels of its pool and for any
  decision it is handed (`Environment.pack`, `Environment.batch`), as does
  whatever else seats a checkpoint. Python describes the arrays
  (`ygo_training/schema.py`, held to the library's by
  `tests/test_packed.py`) and builds none of them. That repository's vectors
  are the library's own output, and an input changed by mistake fails its
  tests and `tests/test_model_input_vectors.py`. Nothing else constructs
  model input.
- In a room of EDOPro the model is shown a decision as an episode shows
  it, and what an episode answers without it is answered there the same
  way. The seat is the policies' own, with their client around it, and the
  bookkeeping of a seat is one code for both
  (`native/ygo-model-input/seat/src/trail.rs`), as is the policy of a seat
  at such a table (`caller.rs` beside it), which other projects seat a
  checkpoint with. `crates/ygo-env/tests/edopro.rs` holds the two together:
  keep that test.
- A recorded duel is imitated or measured on only when it plays again with
  the digest it was recorded with. At other revisions of the engine, the
  scripts, the card database or the pilots its answers can all still be
  accepted while the questions differ; keep that check
  (`searched.check_replay`).
- What a search's log says of an alternative is read for an answer only when
  the alternatives listed from the decision, as `policy-bench` lists them,
  are the ones the log names; keep that check too, and the test that runs
  the bench.
- The lead head is taught at the decisions a search examined and asked at
  the decisions `searched.Reading.examined` takes a search to examine, told
  from the decision alone. The two must be the same decisions: `evaluate-search`
  refuses a logged run where they are not; keep that, and the test that
  runs the bench. Its targets are the leads of the first stage only, which
  every alternative goes through whatever comes of it.

The current checkpoint format, production representation, permitted training
seeds, and active experiment are canonical in
[`EXPERIMENT_LOG.md`](EXPERIMENT_LOG.md) and its single
linked `ACTIVE` experiment record. Do not duplicate those changing facts in
this file or infer them from an older experiment. Read both records before
starting or resuming training or loading a checkpoint as a training seed.
Training checkpoints and full run artifacts normally reside on the CUDA
training machine and are not expected in the development checkout. Verify
completion and evaluation evidence from that machine before treating a run as
a result.

Held-out argmax evaluation is the model-selection evidence. Sampled training
rollouts are diagnostics and are not directly comparable to held-out rates.
