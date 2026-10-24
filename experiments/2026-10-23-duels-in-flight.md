# Duels in flight and larger minibatches: a faster update

**Status:** COMPLETED (closed on 2026-10-23, the day it was read)

**Proposed:** 2026-10-23

**Activated:** 2026-10-23, when no experiment was active. **The design and
the readings were fixed before anything was built.**

**Last updated:** 2026-10-23 (run and read: both readings are met; closed,
its settings made the default)

Of what was built for speed and is not in use (`docs/commands.md`, its
section on what is left to try for speed), most of what is left is the duels
kept in flight from one update to the next, built on 2026-10-18 as options
and never run on the card, with larger minibatches beside them: an update of
`half` might go from 13 s to 7 or 8 by the measurements of 2026-10-17. They
make another training, which is why they wait for an experiment: a duel is
trained on once, whole, when it has ended, with what the weights that
answered it gave, and its first answers may be an update older than its
last; and a pass of minibatches of 512 takes half the steps (of 1,024, as
first designed, a quarter).

## Research questions

1. Does PPO with 1,040 duels kept in flight and minibatches of 512 learn
   as well per duel as `all71` did?
2. How much faster is an update?

## Design (fixed on 2026-10-23)

- **One arm**, `fast`: PPO on from `all71`'s copy
  (`all71-copy/checkpoints/model-17039.pt`, 17,040 updates of imitation)
  for 1,920 updates as `all71` went on from it, with its settings (seed
  2012, the 71 decks, 260 of an update's 520 duels against itself, a
  learning rate of 1e-4, one pass), and with two changes:
  - **1,040 duels kept in flight** (`--duels-in-flight 1040`, two updates'
    duels; the pool's 20 threads with 52 slots each): an update is the next
    520 duels to end, each trained on once, whole, with what its game was
    answered with;
  - **minibatches of 512** (`--minibatch 512`), where `all71`'s were of
    256: of 1,024 as first designed, changed on 2026-10-23 when the card ran
    out of memory with them (the section below on the arm running out of
    memory).
  Read as `all71`'s R1, so that both have trained on as many duels.
- **A trial first**, on the CUDA machine: 20 updates of `all71`'s own
  settings at today's code from the copy, the speed it is set beside (`all71`
  ran at the code of before the field was written straight into its
  buffer); and a few updates of `fast` for each way of answering the
  model's calls (from CUDA graphs or as they are, the duels in one group or
  two), the fastest kept for the run. The way of the calls moves the last
  digits only.
- **What else differs from `all71`**: since 2026-10-23 a monster's effect is
  not offered while its effects are negated ([Seventy-one
  decks](2026-10-19-seventy-one-decks.md), its section on the duels it cut),
  and `all71` was trained without that rule. It is accepted as the rule
  against repeated commands was between the arms of thirteen decks; where
  the first reading is not decided or worse, a control (`all71`'s own
  settings at today's code) tells the two apart. As the rule moves some
  measured games (21 of 1,664 on the Rock-Block screen), `all71` is measured
  again at today's code: both by the same rules.
- **Measured** as `all71`'s R1 was: the checkpoints after 1,680, 1,800 and
  1,920 updates on the thirteen screens, the eighteen and the forty, game
  by game. About 20 minutes for the trial, some four hours for the arm and
  an hour and a half of measuring, by reckoning. Before it, the script is
  tried small on the development machine.
- **Not in it**: the engine built from a profile of its own games, which
  is exact, is left off the CUDA machine until after the run, or `fast`
  would look 3 to 5% faster than it is beside the old timings.
  Where the first reading is not equivalent, a follow-up would part the two
  changes (duels in flight alone, or the larger minibatches alone).

## Readings (fixed on 2026-10-23, before anything was built)

1. *It learns as well*: `fast` against `all71` at R1, both by their
   checkpoints after 1,680, 1,800 and 1,920 updates of PPO, game by game
   over the three suites together (33,696 games): **equivalent** when the
   gap less two standard errors is above −2 points, **worse** when the gap
   plus two standard errors is below −2 points, otherwise **not decided**
   (the rule of seventy-one decks).
2. *It is faster*: `fast`'s first 20 updates against the trial's 20 updates
   of `all71`'s own settings, the same code and the same start, by duels an
   hour: **faster** when it plays at least 1.3 times as many.

Told beside them, without a mark: each suite and each screen; the whole
run against `all71`'s 209,000 duels an hour; the training win rate along
the run; how an update's time parts between its games, the model's calls
and the pass; how old a duel's first answers are when it is trained on;
the duels cut off.

## The build (2026-10-23)

Nothing of the trainer changes: duels in flight (`pool.Stream`, `train
--duels-in-flight`) and the minibatch are options since 2026-10-18 and
before. Built for the run:

- **`experiments/commands/duels-in-flight.sh`** (`PARTS="trial arm
  measure"`). The trial makes 20 updates of `all71`'s own settings into
  `RUNS/plain`, then 8 updates of the arm's settings for each of four ways
  of answering the model's calls (`WAYS`): from CUDA graphs up to half the
  pool's duels with two groups, as a card does unless told
  (`graphs-two`); graphs only for calls of 64 decisions or fewer, two
  groups (`graphs64-two`); no graphs, one group (`eager-one`) or two
  (`eager-two`). The way whose updates after their first two take the
  least time is the arm's (`OUT/trial.json`). The arm and `all71` are then
  measured at this code, both, as the critic's run measured them.
- **`tools/duels_in_flight_report.py`** takes the two readings: the first by
  the rule of seventy-one decks (`card_descriptions_report.kept_rule`) on
  the comparison the critic's run made (`privileged_critic_run_report`),
  the second from the runs' own figures; and what is told beside them,
  among it how many updates before a duel trained on was begun
  (`stream_age`, `stream_oldest`). Its tests:
  `tests/test_duels_in_flight_report.py`.
- **Tried small on the development machine**, against a stand-in of
  `all71` made by its own script (three decks, a copy of six updates and
  three of PPO, small suites): every part ran, the trial chose a way, the
  arm kept 8 duels in flight, both were measured and the report read.

## The run (2026-10-23)

Started on the CUDA machine at 09:57, at `f07eabee` with the submodules at
their pins (the policies' `c6d7cc4`, `native/ygo-model-input`'s `ef1ba77`;
the checkout taken from the development machine's commits by the branch
`critic-build` of each repository, as before), nothing else running there,
without the engine's profile build, in tmux `dif-run`, from the repository
root:

```sh
PYTHON=.venv/bin/python bash experiments/commands/duels-in-flight.sh
```

**The trial** (09:58 to 10:05; `analysis/duels-in-flight/run/trial.json`
there). An update after the first two of each, in seconds:

| | An update | Its games | The model's calls in them | The pass | Calls an update, of decisions |
| --- | --- | --- | --- | --- | --- |
| `all71`'s own settings at this code | 9.75 | 5.36 | 2.62 | 3.92 | 1,149 of 53 |
| The arm, graphs and two groups (`graphs-two`) | 6.56 | 2.85 | 1.35 | 2.77 | 158 of 385 |
| Graphs to 64 decisions, two groups (`graphs64-two`) | 6.47 | 2.78 | 1.24 | 2.77 | 158 of 385 |
| No graphs, one group (`eager-one`) | 7.59 | 3.85 | 1.07 | 2.75 | 78 of 772 |
| No graphs, two groups (`eager-two`) | 6.47 | 2.76 | 1.22 | 2.75 | 158 of 385 |

The arm's calls are answered as they are, the duels in two groups
(`eager-two`), the shortest by a hair over graphs to 64 decisions. The
duels trained on were begun 0.92 updates before on average. `all71`'s own
updates 3 to 20 took 10.0 s at its code.

## The duels it cut (2026-10-23)

Two duels ran to the cap of 4,096 answers in the first 203 updates of the
arm's first attempt (minibatches of 1,024; the section below on the arm
running out of memory): one in update 17,059, against a pilot, with 388 of
the network's decisions, and one in 17,126, against itself, with 1,641. They
are not loops: both are duels against Cloudian that no battle can end. Its
monsters cannot be destroyed by battle, and Spirit Barrier and The Sanctuary
in the Sky keep battle damage from its player, so such a duel goes on at
some 65 answers a turn until a Deck runs out, after 60 turns or more: at the
cap, or past it.

An update's duels are no longer its rollout's, but these were found again.
A duel is given at most one of the network's decisions at each call of its
group, so its decisions say how early it began at the latest, and the
oldest duel of its update how early at most: rollouts 17,049 to 17,056 for
the first, 17,106 and 17,107 for the second. The per-deck figures tell a
duel against a pilot from one against itself, and the first's deck (a win
rate that fits one duel fewer). Each duel of those rollouts that could be
it was played again on the development machine with its seed, decks and
seats, the network drawing its answers, 20 times
(`analysis/dif-cut-duels/`):

- Of the 541 against a pilot, with `all71`'s copy: Dark World against the
  Cloudian pilot (rollout 17,056) averaged 2,550 answers, six times a
  median duel's, and its longest draw ended 22 answers short of the cap,
  with 386 of the network's decisions: Dark World's Deck ran out in turn
  62, at 6,000 LP against 900, its five monsters attacking Cloudian's
  every turn for nothing.
- Of the 520 against itself, with the run's checkpoint after 120 updates:
  only Crystal against Cloudian (rollout 17,106) reached the cap, in turn
  61 with both players at 8,000 LP, with 1,685 of the network's decisions;
  its other draws mostly ended in 37 to 41 turns.

The duel `all71` and the critic's arm cut in update 17,074 is the same kind
([Seventy-one decks](2026-10-19-seventy-one-decks.md), its section on the
duels it cut): Crystal against the Cloudian pilot of that rollout reached
the cap in 17 of 20 draws, with some 340 of the network's decisions, where
they cut it at 339 and 373.

## The arm ran out of memory (2026-10-23)

The arm's first attempt stopped at 10:47, in the pass of its 370th update:
one minibatch needed more of the card's memory than it has (12 GiB), in its
backward pass. A minibatch's arrays are as wide as its widest decision (its
cards, its answers, and the cards each answer takes), and with 1,024
decisions to a minibatch the pass's peak was 1.9 GiB in the median update,
over 6 GiB in 27 of the 369 and 8.5 at most, until this one went past what
the card holds. `all71`'s minibatches were a quarter as large. A third duel
cut at the cap, in update 17,408 (838 of the network's decisions), was not
looked into.

There were three ways on: to go on from update 369 with the memory laid out
otherwise (nothing computed changed), to pass a minibatch that would not fit
in parts whose gradients add up, or to begin again with minibatches of 512.
The last was taken: the arm is made again from `all71`'s copy with
`--minibatch 512`, all else as designed, the readings unchanged. The first
attempt's 369 updates are kept aside on the CUDA machine
(`runs/duels-in-flight/fast-1024`, its log
`~/ygo-training/dif-run-1024.log`). The trial stands: its ways of answering
the model's calls differ in the games, which a minibatch does not touch.

Begun again at 11:08 at `4284c9d9`, the submodules at the same pins, as
before (`PYTHON=.venv/bin/python bash experiments/commands/duels-in-flight.sh`,
the trial read again): some 6.7 s an update, the pass's peak about 1 GiB.

## Results (2026-10-23)

The arm's 1,920 updates of PPO ran from 11:08 to about 14:45, and the
command measured them by 16:08, at `4284c9d9` (the report and the runs'
metrics copied to `analysis/duels-in-flight/run/`).

1. **It learns as well: equivalent.** `fast` against `all71` at R1, by
   their checkpoints after 1,680, 1,800 and 1,920 updates of PPO, game by
   game over the three suites: 55.2% against 54.8%, +0.3 ± 0.2 over 33,696
   games. With two standard errors taken away the gap is −0.1, above −2.
2. **It is faster: faster.** Its first 20 updates played 278,767 duels an
   hour, the trial's 20 of `all71`'s own settings at the same code 191,048:
   1.46 times.

Beside them, without a mark:

- **The suites:** the thirteen screens +0.6 ± 0.2 (55.4% against 54.8%,
  21,632 games), the eighteen other decks of thirty-one −1.4 ± 0.5 (50.0%
  against 51.4%, 3,744 games), the forty new decks +0.4 ± 0.4 (56.9%
  against 56.4%, 8,320 games). Screen by screen from −2.3 (Gishki) to +2.7
  (Rock Block).
- **The whole run:** 275,662 duels an hour against `all71`'s 209,056 over
  the same updates (1.32 times): 6.79 s an update (its games 2.73, the
  model's calls in them 1.21, the pass 3.01) against 8.95 (4.93, 2.36,
  3.58).
- **How old a duel is when it is trained on:** begun 1.00 update before on
  average, 30 at most.
- **Against the pilots in training:** level with `all71` or a little over
  it along the run, from 51.0% against 51.3% over the first 240 updates to
  57.8% against 57.1% over the last 240.
- **Cut off:** 9 duels (`all71` 6 over the same updates); the two looked
  into are the Cloudian stalls above.

What it says: PPO with 1,040 duels kept in flight and minibatches of 512
learns per duel as `all71` did, by the rule, and its update takes 6.8 s
where `all71`'s took 9.0 over the same updates. The pass is now the
largest part of an update (3.0 of 6.8 s), and the games wait on it.

## Closed (2026-10-23)

Both readings met, the record was closed and `fast`'s settings were made the
default. Since then a run on a card keeps twice its duels in flight in
minibatches of 512 where its command line does not say, its model's calls
answered as they are in two groups and the pool's slots raised to hold them,
for a run whose updates can keep them (the index's standing notes). The lead
for after it is PufferLib's, the pass beside the next update's games: [The
pass beside the next update's games](2026-10-23-overlap.md).
