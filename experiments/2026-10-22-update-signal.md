# The signal of an update: how much of it is noise, how much the deal

**Status:** COMPLETED (made beside
[Copy of the specialist](2026-10-22-copy-of-the-specialist.md), as it
trained nothing and touched nothing of that one, and the only one active
once that was closed the same day)

**Proposed:** 2026-10-22

**Activated:** 2026-10-22

**Last updated:** 2026-10-22 (design and readings fixed before anything was
built; built, tried small, run and read on the development machine that day,
with a follow-up at three λ; closed that day)

Proposed on 2026-10-22, when
[Seventy-one decks](2026-10-19-seventy-one-decks.md) had been read: the next
step is to make learning faster, the raw speed of an update giving less and
less, and the first suspect is the signal, weak as seen many times, which
would explain why a matchup takes some 200,000 duels to be played well
enough (`decks-01` in its own mirror). Two measurements were proposed to
tell what kind of weakness it is before building a remedy, as the remedies
differ. They were confirmed with their readings before anything was built,
and were to be made beside the copy of the specialist if they could be made
quickly on the development machine: an exception to the rule of one
experiment at a time, accepted because it leaves the current experiment
unchanged and saves time while giving more information.

## What is known of the signal

- A duel's only reward is its result (1 a win, −1 a loss, −0.5 a draw),
  which each seat's seventy or so transactions share; the credit is spread
  by GAE at λ = 0.95 against the value head's estimates.
- The head explains about 0.35 of a result on games it has not seen
  ([The learner itself](2026-10-08-value-head-and-board.md)), where the
  seat's view allowed about 0.43 (2026-10-08).
- Over 8 worlds about one in ten of the leads the search showed was real
  ([Learning the search's leads](2026-10-07-search-leads.md)).
- An update is 520 duels, some 62,000 transactions, taken in one pass of
  about 270 minibatches of 256 decisions drawn from all of them, at 1e-4.

## Research questions

1. Is an update's step limited by noise: is the direction it moves the
   network in mostly the signal of its 520 duels, or mostly their noise?
2. How much of the noise is the deal's luck, which training on each deal
   played twice would cancel?

## Design (fixed on 2026-10-22)

Of `all71` of seventy-one decks, its last checkpoint (`model-27599.pt`,
R3), in the run's own settings and games (its `run.json`: half of an
update's duels against itself, the others against the pilots, its 71 decks
as an update draws them), on the development machine's processor, with
nothing trained (`tools/update_signal.py`):

1. **Noise.** The games of the run's next eight updates (4,160 duels),
   played as the run would have played them. Of each update, the gradient
   of its policy term on eight groups of 65 of its duels (each half against
   itself), as the update's own objective has it where it begins: 64
   groups. What two groups' gradients share estimates the signal's squared
   length, what each has beyond it the noise's; the **noise scale** is how
   many duels an update would need for its step to be half signal and half
   noise (McCandlish et al. 2018, *An empirical model of large-batch
   training*). Its standard error by leaving a group out at a time.
2. **Luck.** 512 deals of the games of the run's next update, each played
   16 times, the network's answers drawn afresh each time (8,192 duels).
   For every decision, its advantage as the trainer computes it, and the
   result of another play of the same deal for the same seat: the share of
   the advantages' variance that result accounts for, with one
   coefficient, which is what training on each deal played twice would
   cancel at no more games, both plays training. Its standard error by
   drawing the deals again.

## Readings (fixed on 2026-10-22, before anything was built)

1. *The update's step.* **Limited by noise** when the noise scale is at
   least 1,040 duels (twice an update) and more than two standard errors
   over 520. **Not limited by noise** when it is at most 260 (half an
   update) and more than two standard errors under 520: then more updates
   per duel would pay, not a stronger signal. Otherwise neither.
2. *Pairing deals.* **Pays** when another play's result accounts for at
   least a quarter of the advantages' variance, by two standard errors:
   about a third more duels' worth of signal from each duel. **Does not
   pay** when it accounts for at most a tenth, by two standard errors.
   Otherwise not decided.

Told beside them, without a mark: the first at R1 (`model-18959.pt`), and
for the value head's term; the first on an update of the Blackwing mirror
alone, one pairing (the network's Blackwing against itself and against the
Blackwing pilot); the deal's share of a result's variance; what the value
head explains at a seat's first decision; the second with the mean of the
other 15 plays of a deal, which is what more plays of a deal would give;
each of the second's split by whether the other seat is the network or a
pilot.

What each would point to, as proposed: limited by noise and pairing pays,
paired deals are built and tried in a run for how fast it learns; limited
by noise and pairing does not pay, the noise is in the credit and not the
deal, and the value head (the gap from 0.35 to 0.43) or λ is next; not
limited by noise, more updates per duel and the network's size, where the
ablations of ygo-agent's trainer that are still owed come in.

## The build (2026-10-22)

`tools/update_signal.py`, with `tests/test_update_signal.py` (the measures
on numbers whose answers are known, and the readings' marks). The trainer
is built as the run's `run.json` says, its settings checked against it, on
the processor, with the checkpoint loaded at the update after its own; the
games are the coordinator's for the run's next rollout seeds, played by the
trainer's own `_learner_games`; the advantages are the optimizer's
`prepare_samples`, and the gradients its `objective`, a group's taken a part
at a time and weighted by its transactions, as one minibatch of the whole
group would give them. Tried small that day, to time it: one update in four
groups (48 s), 64 deals played three times (7 s).

## The run

Made on the development machine on 2026-10-22, from 10:34, the four
measurements one after the other and the report
(`analysis/update-signal/report.txt`): the two readings at R3, then the
first at R1 and on the Blackwing mirror alone. At the pins of `77bac339`,
whose games are the run's but where Batteryman Fuel Cell's cost would take
its last target (the policies' `0ed8ee1`).

## Results (read on 2026-10-22)

Made on the development machine from 10:34 to 10:56, in
`analysis/update-signal` (the report in `report.txt`, what is told beside it
in `beside.txt`). The tool was tidied afterwards, its numbers the same: the
shared signal's error is told beside the noise scale since.

The readings:

1. **The update's step: neither, its signal not measured.** The noise scale
   of the policy's term is 15,481 ± 20,945 duels, an update being 520. The
   error is larger than the scale because what the 64 groups' gradients
   share is 0.8 standard errors from none (two groups' gradients at a
   cosine of 0.004). With that shared part two standard errors larger, the
   noise scale would still be 4,530 duels, nine updates' worth.
2. **Pairing deals: does not pay.** Another play's result accounts for
   0.037 ± 0.004 of the advantages' variance; the mean of the other 15
   plays, 0.064 ± 0.006.

Beside them, without a mark:

- **The deal is most of a result**: 0.583 ± 0.018 of a result's variance is
  the deal's (0.514 ± 0.025 against itself, 0.628 ± 0.027 against the
  pilots). The value head explains 0.185 ± 0.021 of a result at a seat's
  first decision, a third of what the deal holds. Yet an advantage carries
  little of the result: with GAE at λ = 0.95 the result reaches the first
  of a seat's seventy transactions at 0.95^69, about 3% of its weight, and
  about a quarter on average over them. The advantages are mostly the
  value head's own estimates from one transaction to the next, which
  another play of the deal does not share (pairing: 0.028 ± 0.005 against
  itself, 0.046 ± 0.008 against the pilots).
- **One matchup alone**: on updates of the Blackwing mirror alone (the same
  checkpoint, its Blackwing against itself and against the Blackwing
  pilot), the policy's noise scale is 758 ± 168 duels, its shared signal
  4.7 standard errors from none: an update of that one matchup is about
  two fifths signal. Over the 71 decks the shared signal is at most a
  sixth of that (the bound of 4,530): their signals do not add up into one
  direction within an update, each deck having some seven of its duels.
- **At R1** the policy's noise scale is 39,196 ± 177,055 duels, its shared
  signal 0.4 standard errors from none (5,894 with it two errors larger).
- **The value head's term**: 2,041 ± 1,545 duels at R3 (1.3 standard errors
  from none), 461 ± 242 at R1 and 130 ± 29 on the Blackwing mirror alone.

What it says: pairing deals is not the lever. At λ = 0.95 the advantages
hold little of the deal's luck, which is most of a result; they are the
value head's step-to-step estimates. In the mix of 71 decks an update's
direction is mostly noise as far as 4,160 duels can tell, and an update
of one matchup is near its noise scale. The noise is in the credit, and in
how little of an update each matchup has.

## The follow-up at three λ (fixed on 2026-10-22, before any run of it)

Proposed with the results above, the noise being in the credit: the same
checkpoint and games, each advantage computed at λ = 0.95 (the run's),
0.98 and 1 on the same plays, which needs no training. At λ = 1 the
advantages carry the result in full, so that pairing might pay there and
the step's signal could change; and with four times the duels the first
reading might be settled.

- **Noise.** The games of the run's next 32 updates (16,640 duels), each
  update's gradients on its eight groups at the three λ: 256 groups. Every
  group's gradient at three λ would not fit in memory, so the errors are
  by leaving out a block of 16 groups, two updates, at a time
  (`noise-lambdas`).
- **Luck.** The same 512 deals played 16 times each as above, the
  advantages computed at the three λ (`luck --lambdas`).

The readings:

1. *The step at λ = 0.95, settled.* As the first reading above: limited by
   noise at 1,040 duels or more and more than two standard errors over 520;
   not limited at 260 or less and more than two standard errors under 520;
   otherwise neither.
2. *Pairing deals at λ = 1.* As the second reading above: pays when another
   play's result accounts for at least a quarter of the advantages'
   variance, by two standard errors; does not pay at a tenth or less, by two
   standard errors.
3. *λ = 1 against 0.95, for the step.* The signal per duel, one over the
   noise scale: **more signal** at λ = 1 when it is at least twice 0.95's
   and the difference is more than two standard errors; **less** when at
   most half, by as much; otherwise not told apart.

Told beside them, without a mark: λ = 0.98 for all three; the Blackwing
mirror alone at the three λ; at each λ, the advantages' variance and how
much of it the seat's own result accounts for. The noise scale tells how
consistent the step's direction is, not whether it points the right way: a
λ under 1 buys consistency with the value head's bias, so more signal at
0.95 would not by itself make it the better choice.

Tried small end to end before the run (two updates in two blocks, 32 deals
played three times), to see it work and time it: about 125 s an update at
the three λ.

### The follow-up's results (read on 2026-10-22)

Made on the development machine from 11:21 to 12:50, in
`analysis/update-signal` (`report-lambdas.txt`).

The readings:

1. **The step at λ = 0.95: limited by noise.** With four times the duels its
   signal is measured, 3.2 standard errors from none: the noise scale is
   20,018 ± 6,347 duels, some 38 updates, so that an update of 520 duels is
   about 2.5% signal (5.0e-5 ± 1.6e-5 a duel).
2. **Pairing deals at λ = 1: does not pay.** Another play's result accounts
   for 0.049 ± 0.009 of the advantages' variance (the others' mean 0.080),
   though the seat's own result is 0.57 of it there: by the time most
   decisions are made, the value head already holds most of what the deal
   decides.
3. **λ = 1 against 0.95: not told apart.** A signal per duel of 1.7e-5 at
   λ = 1 against 5.0e-5, the gap −3.3e-5 ± 2.1e-5: a third, but 1.6
   standard errors from none.

Beside them, without a mark:

- **λ = 0.98** is 0.95's again: a noise scale of 23,124 ± 9,791 duels,
  pairing 0.046 ± 0.006.
- **One matchup** (the Blackwing mirror alone, eight updates) tells λ apart
  where the mix cannot: the noise scale is 758 ± 96 duels at 0.95 (the
  first measurement's again), 1,401 ± 204 at 0.98 and 4,049 ± 1,165 at 1,
  the step at λ = 1 with a fifth of the signal per duel.
- **What the advantages are**: their variance is 0.09 at λ = 0.95, 0.18 at
  0.98 and 0.58 at 1, the seat's own result a quarter of it at 0.95 (0.25 ±
  0.01), two fifths at 0.98 and 0.57 at 1.
- **The value head's term** has a noise scale of about 3,500 duels at
  every λ in the mix and 130 to 235 on the mirror alone.

What it says: an update of the mix of 71 decks is limited by noise, about
a fortieth of its step signal, and one matchup's update is near its noise
scale. Neither pairing deals nor taking the result in full gives a cleaner
step: the more of an advantage is the value head's from one transaction to
the next, the more consistent the step, so the credit the head gives is
what carries the signal. A better critic is the lever left on the map the
readings were confirmed with.

## What follows

Closed on 2026-10-22, the day it was read, with every reading of it and of
its follow-up taken, for the lever its readings point to: a critic that
sees, in training only, the cards the seat cannot see
([A critic that sees the hidden cards](2026-10-22-privileged-critic.md)).
Crossing the input's rule for that critic was accepted on two conditions:
the critic is used in training only, and the model plays without seeing the
hidden cards.
