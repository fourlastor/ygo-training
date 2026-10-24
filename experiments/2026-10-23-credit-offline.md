# The credit offline: a value head with more results, and pairing at the decision

**Status:** ACTIVE (made beside [Duels in flight](2026-10-23-duels-in-flight.md),
as it trains nothing and touches nothing of that one; beside
[Overlap](2026-10-23-overlap.md) since that one was closed)

**Proposed:** 2026-10-23

**Activated:** 2026-10-23

**Last updated:** 2026-10-23 (design and readings fixed before anything was
built; built, run and read on the development machine that day: neither
remedy's reading met)

On 2026-10-23, while the run of duels in flight went, other ideas for the
noise in the learning signal were sought. What was known of it:
[The signal of an update](2026-10-22-update-signal.md) found an update's
step at λ = 0.95 about 2.5% signal (a noise scale of some 20,000 duels,
against the update's 520), pairing deals cancelling 0.037 of the
advantages' variance and λ = 1 not told apart from 0.95: the advantages
are the value head's estimates from one transaction to the next, and the
noise is in that credit. [A privileged critic](2026-10-22-privileged-critic.md),
begun as the head, seeing the hidden cards as well and taught five times as
long, explained 0.400 of a result on held games where the head explains
0.398, and its advantages gave no stronger signal (4.7e-5 a duel against
the head's 5.2e-5).

Two remedies were proposed: a value phase that gives the head more results
(Phasic Policy Gradient, Cobbe et al. 2020), which was expected, from the
critic's figures, to come out neither; and pairing at the decision, two
answers played on from one position with the same draws to come, so that
their difference holds the answers' worth without the luck of what
follows, the noise no estimate made at a position can take out. Both are
made offline, beside duels in flight, on the development machine: they train
nothing and touch nothing of the CUDA machine's run.

## Design (fixed on 2026-10-23)

Both of `all71`'s last checkpoint (`model-27599.pt`, after 10,560 updates of
PPO) in its run's own games, as the measurements of the update's signal and
of the critic were, so that their figures can be set beside these.

### 1. A value head with more results

- The policy fixed, playing the games of 48 updates (24,960 duels). Two
  copies of its network are taught on them, their value alone:
  - **control**: as `all71`'s PPO taught its head: each update's games
    once, the λ-returns at every transaction (λ = 0.95, from the copy's own
    estimates), one pass of minibatches of 256, a learning rate of 1e-4;
  - **variant**: the same, and after every 16th update a value phase, as
    Phasic Policy Gradient has one: two passes over the last 16 updates'
    duels, each seat's result at four of its transactions a duel (some
    50,000 positions), the way the head was found to learn best
    ([The learner itself](2026-10-08-value-head-and-board.md);
    more positions, or more passes, and it learnt results by heart).
- Both read on the games of 8 updates neither was taught (4,160 duels), at
  every decision.

### 2. Pairing at the decision

- 2,000 decisions, one of each of 2,000 duels from another stretch of the
  schedule: the first decision of a transaction of a seat the network
  plays, with at least two answers, drawn uniformly. *a* is the answer the
  game took, *b* another drawn from the policy's chances without *a*.
- Each duel is played again from its start with its seed and draws up to
  that decision, which it reaches as it was (checked: plays that do not are
  counted and left out); then four times answering *a* and four times *b*.
  The k-th play of *a* and the k-th of *b* share their future: the engine's
  random stream from there, the network's draws and the pilots'. Plays of
  different k are independent.
- A play's result: +1, −1 or 0, for the seat that decided.

## Readings (fixed on 2026-10-23, before anything was built)

Of the value head with more results:

1. *It predicts better*: the variant explains at least 0.02 more of a result
   than the control, over every decision of the held games, by two standard
   errors.
2. *A stronger signal*, made only where the first is met: with the
   variant's advantages the update's signal per duel is at least 1.5 times
   the control's, the gap more than two standard errors, on the games of 32
   updates in 256 groups of 65 duels, as the critic's second phase.

Of pairing at the decision:

1. *A shared future cancels the luck*: pairing pays when the paired
   difference of the two answers' results has at most half the variance of
   the same difference with independent futures, by two standard errors; it
   does not pay at a quarter or less cancelled, by two standard errors;
   otherwise it is not told apart. Pairing deals cancelled 0.037.
2. *A paired play judges better than the head*: one paired play's
   difference strays less from what the two answers are worth (the mean of
   the four pairs, its own noise allowed for) than the head's difference
   does (its estimate at the seat's next decision after *a*, less the same
   after *b*), by two standard errors.

Told beside them, without a mark: by the chance of the answer taken (under
0.9, 0.9 to 0.99, over 0.99), against the pilots and against itself, and
what a decision played on costs in play.

## The build (2026-10-23)

- **`tools/decision_pairing.py`** (`play`, `report`; tests in
  `tests/test_decision_pairing.py`): the duels on the pool, the network's
  answers drawn as the run's updates draw them (a draw seeded by the duel and
  the decision's place in it), so that a duel played again from its start
  with the same draws comes to the same position; at the decision the answer
  given, and after it the draws of the play's pair. Tried small: two plays of
  the same answer with the same pair's draws came out the same in all 30
  decisions tried, and another pair's draws changed the play in 27.
- **`tools/value_phase.py`** (`fit`, `report`; tests in
  `tests/test_value_phase.py`): the trainer's own minibatches and value term
  teach the control (the returns from its own estimates); the variant's
  phase teaches the results at the transactions kept for it in the pool.
  Tried small: before any teaching both copies explain what the head explains
  (0.418 of the first update's results).

## Results of pairing at the decision (read on 2026-10-23)

On the development machine from 12:00 to 12:16 (`tools/decision_pairing.py
play` and `report`, `analysis/credit-offline/pairing.json` and
`pairing-report.txt`): `all71`'s last checkpoint, 2,000 duels of its run's
games 20,000 updates on, one decision drawn in each and played on four times
with each answer (16,000 plays); every play came to its decision as it was.
**Neither reading is met.**

1. **Pairing does not pay.** A shared future cancels 0.136 ± 0.025 of the
   variance of the two answers' difference (0.400 with independent futures),
   under a quarter by more than two standard errors.
2. **Not shown: the head strays less.** One paired play's difference strays
   from what the two answers are worth by 0.346 ± 0.017 (a variance, in a
   result's units), the head's difference by 0.183 ± 0.015: the head by
   0.163 ± 0.024 less.

Told beside them (variances in a result's units; the spread of what the
answers are worth apart is the variance of the pairs' means less their own
noise, its square root here; the head's correlation with it from the pairs
that share no play with its difference):

| Decisions | Count | Cancelled | A pair strays | The head strays | Worth apart spreads by | The head's correlation with it |
| --- | --- | --- | --- | --- | --- | --- |
| All | 2,000 | 0.136 ± 0.025 | 0.346 | 0.183 | 0.371 | 0.16 ± 0.03 |
| The answer taken at a chance under 0.9 | 658 | 0.327 ± 0.047 | 0.229 | 0.134 | 0.324 | 0.01 ± 0.06 |
| 0.9 to 0.99 | 268 | 0.129 ± 0.062 | 0.401 | 0.098 | 0.257 | 0.17 ± 0.14 |
| Over 0.99 | 1,074 | 0.043 ± 0.031 | 0.403 | 0.234 | 0.415 | 0.19 ± 0.04 |
| Against itself | 995 | 0.146 ± 0.032 | 0.424 | 0.173 | 0.369 | 0.20 ± 0.05 |
| Against the pilots | 1,005 | 0.120 ± 0.040 | 0.268 | 0.192 | 0.374 | 0.13 ± 0.05 |

A play on from a decision takes 66 of the network's decisions, 57% of its
duel's: the four pairs of a decision are some four and a half duels of play.

What it says:

- **Two answers' futures part as soon as they differ.** What a shared future
  cancels is a third of the variance where the policy hesitates (an answer
  taken at a chance under 0.9) and next to nothing where it does not (0.043
  over 0.99): the difference of two answers' results carries almost all the
  luck of each play. Played on from a decision, a credit costs four and a
  half duels for a figure noisier than the head's: not a cheaper credit.
- **The head strays less than a play, but not because it tells the answers
  apart.** Its difference follows what they are worth at a correlation of
  0.16 ± 0.03, and not at all where the policy hesitates (0.01 ± 0.06). It
  strays less because its differences are small (a variance of 0.078 where
  what the answers are worth apart spreads by 0.138): a head that always said
  the two were worth the same would stray by 0.138, less than one play.
- **For a search that scores positions with the head** (the idea of the same
  day, a search on the head's estimates and the network taught its choices):
  the head ranks two answers at a position little better than chance, and
  not at all where the policy hesitates, so a search on it would follow its
  errors. Played on, the answers are told apart on average (one pair's noise
  is two and a half times the spread of their worth, and the mean of four
  pairs would follow it at about 0.8), but these plays go on in the duel
  itself with the hidden cards as they are, as a search cannot, and four
  pairs cost four and a half duels a decision.

## Results of the value head with more results (read on 2026-10-23)

On the development machine from 12:17 to 13:22 (`tools/value_phase.py fit`
and `report`, `analysis/credit-offline/value-phase/` and
`value-phase-report.txt`): `all71`'s last checkpoint, the policy fixed, the
games of 48 updates (24,960 duels) taught to both copies, three phases of
the variant (after the 16th, 32nd and 48th updates, about 49,900 positions
each, two passes), read on 4,160 held duels. **The first reading is not
met, so the second is not made.**

1. **Not shown to predict better.** Over every decision of the held games
   (522,609) the variant explains 0.392 of a result and the control 0.396:
   −0.004 ± 0.002. The head they were both begun as explains 0.400 there.

Told beside it: at each transaction's first decision (473,282) the variant
0.406, the control 0.409 (−0.003 ± 0.002), the head 0.412. On each update's
games before they were taught the two copies stayed within 0.009 of each
other throughout (the variant 0.0007 under the control after the first
phase, 0.001 over after the second, on average); the phases' loss went from
0.250 to 0.239.

What it says: **the head is not short of results.** Taught three times as
many results again, at the positions it learns them best from, it explains
no more of them on games it has not seen; nor did the control, taught as
PPO teaches it on the same games, gain on the head it began as. With the
critic that saw the hidden cards and was taught five times as long (0.400
against 0.398), the head explains about as much of a result as its data
and what a seat sees allow, and the noise of the credit is the luck of what
follows, which neither more results nor a pair played on from the decision
takes out (above).
