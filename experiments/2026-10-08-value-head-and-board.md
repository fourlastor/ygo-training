# The learner itself: the value head and the board

**Status:** COMPLETED

**Proposed:** 2026-10-08

**Activated:** 2026-10-08

**Completed:** 2026-10-09

**Last updated:** 2026-10-09

On 2026-10-08 [Learning the search's leads](2026-10-07-search-leads.md) was
closed, to look at the learner itself before anything else is built on it,
and the design and the readings below were fixed the same day. The run was
made on the CUDA machine that evening and read on 2026-10-09:
[The run](#the-run). A follow-up run with its own readings was made and
read the same day: [The follow-up run](#the-follow-up-run). It was closed
that evening for
[PPO with a head that predicts](2026-10-09-ppo-with-a-head-that-predicts.md),
which starts from its arm **Once, few**.

## Research questions

1. Trained so that it cannot learn the results of its games by heart, does
   the value head tell on games it has not seen who is winning, and how
   much of what can be told?
2. Does the copy of the pilot lose or gain by that way of training?
3. Does a network that sees the board as its cards copy its pilot more
   closely than one that sees the board pooled, and does a head on it tell
   more, of results and of the search's leads?

## Why

What the arms of the last two experiments learned from recorded games,
they mostly learned by heart
([After the readings](2026-10-07-search-leads.md#after-the-readings-every-update-and-the-rules-changes-apart)):
the lead head explains on unseen games what it explained after a fifth of
the searched games and then fits the ones it has seen; the search's answers
are found at 29% of the deviations trained on and 5 to 8% of others. The
value head is the plainest case, and it stands in the way of PPO, whose
advantages are measured against it.

- **It predicts nothing on games new to it.** At the first steps of the
  transactions of 520 new games the seed's version 89 explains -0.10 of
  the results
  ([the seed's run](2026-10-05-search-imitation.md#what-the-run-shows)),
  and -0.06 to -0.08 in the first pass of three arms of the last
  experiment. Within an update its loss averages 0.2 where it starts near
  1.0.
- **How it is trained.** A game's result is the target at the first step
  of every one of its transactions, about 72 a game, and an update goes
  four times through its 520 games. One result is shown some 290 times,
  with the game's cards and the seat's own earlier answers to tell the
  game by. The next update has other games.
- **The same is known of Go.** A value network trained on every position
  of 160,000 games learned the games: an error of 0.19 on them and 0.37 on
  others. Trained on one position from each of 30 million games it had
  0.226 and 0.234 (Silver et al., *Nature* 2016).
- **There is much to be told.** A first look on 2026-10-08, by a script
  that is not kept, at the held-out log of the search-imitation experiment
  (1,535 games against the earlier pilots, 81,443 examined decisions): the
  pilot's answer wins between none and all of the 8 worlds it is played
  out in, and the spread of those rates, less what 8 samples add, is 43%
  of the variance of one result: what a perfect predictor explains, 26% in
  the first two turns and 60% from the eleventh. Those rates explain 36%
  of the real results of other games, where 37% follows from the 43%. A
  network of 7,000 weights on the log's `situation` (life points, counts
  of cards, turn, phase) explains 20% on games it was not trained on, and
  29% with the opponent's deck. The 192 games at the present pin give 43%
  again. These are to be measured again by a tool that is kept.
- **The board.** Every visible card is encoded by one layer and the cards
  are averaged into one vector before anything compares two of them; an
  answer is scored from that vector and its own card
  (`ygo_training/model.py`). `individual-tokens`, where each answer
  attends to the cards on the field, has no valid comparison: one was
  retired with the earlier pilots before it produced evidence, the other
  ran on the engine without set codes
  ([board tokens](2026-09-28-ygo-policies-board-tokens.md)). Imitation
  levels off at 95% agreement with its pilot though every update has new
  games, 2 points of win rate under it.
- **What a hundred million games give, measured outside this repository.**
  ygo-agent's released model (self-play PPO on one deck against itself,
  over 100M games per its author, 4.3M parameters) was set on 2026-10-08
  against a pilot of its deck written that evening, in the policies'
  engine: it wins 1,659 of 2,000 games, 83.0% ± 0.8%. The pilot wins 919
  of 1,000 against a player that takes the first option, the model 992.
  Its value head does what ours does not: over 200,000 decisions of games
  against itself the seat to move won 4% of the time where the head gave
  it under 10%, 48% at 40 to 50% and 98% over 90%. Of its training, the
  steps from 11.3 to 16.5 billion are worth 4.9 ± 1.2 points against the
  pilot and those from 16.5 to 22.75 billion nothing (-1.1 ± 1.3).
  With the bench's one-step search on top, the same pilot wins 59 of 200
  games against the model where alone it wins 33 (+13.0 ± 3.3 points deal
  by deal), and 164 of 200 against the pilot itself: as often as the model
  beats that pilot, and still seven games in ten behind the model when the
  two meet. In the search's try-outs the pilot stood in for the model, so
  the search planned against other answers than the ones it met. With the
  model itself playing its seat there, measured on 2026-10-09 on the same
  deals, the pilot with a search wins 76 of 200, 38.0% ± 3.4%: 9.5 ± 3.4
  points more than with the pilot standing in on the same server (57 of
  200). A wrong picture of the opponent is part of the gap and not all of
  it: given the model's own answers, asked 181,000 times a game, the pilot
  with a search still loses six games in ten. The pilot itself was then
  improved the way the policies improve theirs, on 2026-10-09: four rules
  a search pointed to, each kept for what it won on other deals. The
  improved pilot wins 61.1% of 8,192 games against the first, and takes
  22.6% of 2,000 games from the model where the first takes 17.0% (5.5 ±
  1.3 points, the two on different deals): the model still wins 77.5% ±
  0.9%. The record, with how the model is run and what was checked, is
  `benchmarks/ygo-agent/README.md` and `SEARCH.md` on the branch
  `ygo-agent-benchmark-work` of ygo-policies.
- **The size of the network is not the question.** Of its 7.3M parameters
  6.3M are the lookup of card identities, of which the cards in play use
  a few hundred rows; about 1M compute. It fits 520 results in an update
  and 7,680 games' leads in eight passes.

### A trial on the development machine

On 2026-10-08, on the CPU, from scratch, 130 games an update against the
13 decks, the seed's command otherwise. The head's fit is the share of the
results it explains at the transactions' first steps of each update's
games before it trains on them.

| Updates | Four passes, as now: fit | loss in the update | agreement | One pass: fit | loss in the update | agreement |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1-10 | -0.06 | 0.42 | 80.4% | +0.12 | 0.75 | 77.8% |
| 11-20 | -0.05 | 0.30 | 87.5% | +0.18 | 0.66 | 84.1% |
| 21-30 | -0.08 | 0.31 | 88.2% | +0.23 | 0.63 | 86.1% |
| 31-40 | -0.06 | 0.29 | 89.2% | +0.25 | 0.60 | 87.2% |

Both runs have the same 5,200 games. With four passes the head fits an
update's results, a loss of 0.3 from 0.9, and explains none of the next
update's: -0.32 at worst. With one pass its loss stays near what it
explains, and by the last ten updates it explains a quarter of the results
of games it has not seen, still rising. The copy agrees with its pilot as
much after 10 updates of four passes as after 40 of one: for the same
optimizer steps one pass takes four times the games, which the pilot plays
for nothing. Their figures are in `analysis/learner-trial/` on this
machine (`value-asnow` and `value-once`).

The two models after their 40 updates, on the 192 validation games played
here at the present pin, which neither has seen: their values at the 9,993
examined decisions that open a transaction against the rate at which the
pilot's answer wins its 8 worlds there (`evaluate-search`, the measure
built for this experiment; 297 examined decisions open none and are left
out). A perfect predictor explains 43.0% of one result there.

| | Four passes | One pass |
| --- | ---: | ---: |
| Correlation with the worlds' rates | +0.51 | +0.65 |
| Of what can be told, by a straight line through its values | 30.2% | 49.9% |
| Of what can be told, by its values as they stand | -36.4% | 43.3% |
| Of one result, by the straight line | 13.0% | 21.5% |
| By turn, of what can be told: 1-2, 3-4, 5-6, 7-10, 11 on | 22, 25, 32, 44, 27% | 51, 41, 49, 65, 42% |
| Deals won of the 192 by its first choices, pilot 61.5% | 40.1% | 27.1% |

- The head trained with four passes is not without knowledge: its values
  rank the positions at a correlation of 0.51. They are far too sure, and
  as they stand they explain less than nothing, which is what a baseline is
  used for. That is what the explained variance on new games had shown,
  and why the correlation is now reported beside it.
- The head trained with one pass ranks better and its values can be taken
  as they are: half of what can be told after 5,200 games.
- **The copy is the other way round at equal games.** After the same 5,200
  games the model of four passes has made four times the optimizer steps
  and plays better, 40.1% of the deals against 27.1%; both are far from
  their pilot on so few games. The run on the CUDA machine compares them
  at equal steps, where the trial's agreement was equal, and its reading
  of the copy decides this. If one pass copies worse there, the arm to add
  keeps four passes for the policy and gives the value head each game once.

## Design (fixed on 2026-10-08)

Imitation of the Blackwing pilot from scratch, as the seed was trained
(`format13-imitation-seed.sh`: the pilot plays 520 games an update against
the 13 decks, the value loss at 0.5, training seed 2012), with one thing
changed in each arm. The games are the pilot's and do not depend on the
model, so every arm has the same games in the same order.

| Arm | What changes | Updates |
| --- | --- | ---: |
| Again | Nothing: the seed's recipe, four passes through an update's games. | 90 |
| Once | One pass: every game is seen once (`--ppo-epochs 1`). | 360 |
| Once, few | One pass, and a game's result is the target at 4 of its transactions, drawn at random. | 120 |
| Once, reading | One pass, and the value head reads the shared state without shaping it. | 360 |
| Once, cards | One pass, and the board as its cards (`--board-representation individual-tokens`). | 360 |

- **Again** is the control, and the readings that name the seed
  are applied to it. The design as confirmed took the existing seed for
  the control; that run imitated games against the pilots ten of the
  opponents had until 2026-10-07, so the recipe is trained again against
  the present ones, on the games every other arm has. That change was made
  after the confirmation. The existing seed's version 89 is measured
  beside it.
- 360 updates of one pass are the optimizer steps of 90 updates of four,
  on four times the games: 187,200.
- **Once** is the main arm. **Once, few** asks whether a game's 72
  positions in one pass are still too many of one result, and is read on
  the value head only, so it stops at 120 updates. **Once, reading** asks
  whether the head needs to shape what it reads, and whether the copy is
  better left alone by it. **Once, cards** is the board against the pooled
  board of **Once**.
- Then, from the last checkpoints of **Once** and **Once, cards**, one
  pass through the 7,680 logged games of the last experiment with the lead
  head at its weight of 75 and the pilot's answers as labels: what the
  head explains before each update on games new to it, pooled against
  cards.
- A checkpoint is kept every 40 updates, and the suite is played then;
  every 10 for **Again**.

About five and a half hours on the CUDA machine, unattended.

## Measures

- **The value head's fit**: the share of the results it explains at the
  transactions' first steps of an update's 520 games before it trains on
  them, over the last 40 updates of a run: 20,800 games new to it. For
  **Again** over its last 10 updates, which are the same optimizer steps.
  For the existing seed, before the first of four updates as it was
  trained, on 520 games of the present pilots.
- **Against the worlds**, on the 1,536 held-out logged games
  (`evaluate-search`): the head's value at each examined decision that
  opens a transaction against the rate at which the pilot's answer wins
  its 8 worlds there. Of the spread of those rates, the part that is real
  is what remains without the spread of a mean of 8 results; the head is
  read as the share of that part a straight line through its values
  explains, and the share its values explain as they stand. Beside it, by
  `tools/policy_search_predictability.py`, the share of one result that a
  perfect predictor explains there, and what a small network on the log's
  `situation` tells, taught on the 7,680 logged training games.
- **The copy**: deals won of the 1,536 held-out deals by the model's first
  choices, deal by deal against the seed's version 89 and between arms;
  the suite of 1,664 games at the last two evaluations; agreement with the
  pilot on the games of the last 40 updates.
- **The lead head**: the share of the measured leads it explains before
  each update of the one pass, over its second half.

## Readings (fixed on 2026-10-08)

Each arm is one training run: the errors are those of the games measured.
They were proposed after the first figures of the trial above, which
already had one pass near the first mark on a quarter of the games an
update; the second mark is the one that asks something.

- *The value head predicts*: **Once** explains at least 0.20 of the
  results of games new to it. *It reads the position*: at least 0.30,
  more than the small network explains with the opponent's deck and two
  thirds of what there is.
- *The copy is no worse for it*: **Once** wins no fewer held-out deals
  than the seed's recipe, **Again**, by more than two standard errors. *It
  is better*: at least 2 points more, and more than two standard errors.
- *The head must shape what it reads*: **Once** explains at least 0.05
  more than **Once, reading**.
- *Fewer positions of a game are needed*: **Once, few** explains at least
  0.05 more than **Once** over updates 81 to 120.
- *The board as its cards helps the copy*: **Once, cards** wins at least 2
  points more of the held-out deals than **Once**, and more than two
  standard errors. *It helps the heads*: its value head explains at least
  0.05 more, or its lead head at least 0.05 more on games new to it.

If **Once** does not predict: the results themselves are then too poor a
lesson at this amount, and the next arm teaches the head the rates of the
worlds, which the logs hold for 53 positions of each searched game.

## The run

On the CUDA machine on 2026-10-08, by
[`learner-value-board.sh`](commands/learner-value-board.sh) at ygo-policies
`1b85b63`. Its figures are in `analysis/learner/` on the development
machine, and the tables below are `tools/learner_report.py` on them
(`analysis/learner/report.json`) with the runs' own figures beside it.

- **Every run had the same games in the same order.** The fingerprint of
  an update's decisions is the same in the four runs of one pass at every
  update they have, and the 90 updates of **Again** are the first 90 of
  theirs.
- A run of one pass took 73 to 75 minutes for its 360 updates and 187,200
  games; **Again** 44 minutes for 90 updates and 46,800 games.
- **On the suite the pilot's reference for these runs is 60.3%, not
  55.5%.** A suite's deals follow from `--seed`. A run plays the suite with
  its training seed, 2012, and on those deals the pilot wins 1,003 of
  1,664 (`evaluate-suite --pilot --seed 2012`, on the development machine
  on 2026-10-09). The 55.5% of the
  [last record](2026-10-07-search-leads.md) is of the default seed's
  deals, 2011, on which the seed's 53.2% there was measured too. The
  64.9% against the earlier pilots is of seed 2012, from the seed's own
  script: on the same deals the improved opponents cost the pilot 4.6
  points of the suite, not 9.4.

### The value heads

The fit is the share of the results a head explains at the transactions'
first steps of games new to it, and the correlation is with those
results. The other columns are its values against the worlds on the 1,536
held-out logged games, at the last checkpoint measured.

| Run | Updates | Fit | Correlation | With the worlds' rates | Of what can be told, by a straight line | As its values stand | Of one result |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| The seed, version 89 | its own | -0.113 | +0.39 | +0.558 | 36.2% | -21.0% | 15.8% |
| Again | 81-90 | -0.142 ± 0.014 | +0.38 | +0.579 | 38.9% | -16.2% | 17.0% |
| Once | 321-360 | +0.183 ± 0.007 | +0.51 | +0.747 | 64.7% | 51.9% | 28.4% |
| Once, few | 81-120 | +0.320 ± 0.006 | +0.57 | +0.805 | 75.2% | 73.1% | 32.9% |
| Once, reading | 321-360 | +0.240 ± 0.004 | +0.49 | +0.679 | 53.6% | 53.4% | 23.5% |
| Once, cards | 321-360 | +0.168 ± 0.008 | +0.50 | +0.723 | 60.6% | 43.4% | 26.6% |

On those games, at 77,977 decisions that open a transaction, a perfect
predictor explains 43.7% of one result. The small network on the log's
`situation`, taught on 6,912 of the logged training games, explains 21.1%
of one result taught the games' results and 23.5% taught the worlds'
rates; with the opponent's deck 29.7% and 32.8%, which is 67.9% and 75.1%
of what can be told.

The fit over a run, by blocks of 40 updates:

| Updates | 1-40 | 41-80 | 81-120 | 121-160 | 161-200 | 201-240 | 241-280 | 281-320 | 321-360 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Once | +0.184 | +0.163 | +0.141 | +0.144 | +0.139 | +0.150 | +0.145 | +0.160 | +0.183 |
| Once, few | +0.237 | +0.301 | +0.320 | | | | | | |
| Once, reading | +0.133 | +0.195 | +0.214 | +0.219 | +0.230 | +0.242 | +0.232 | +0.230 | +0.240 |
| Once, cards | +0.153 | +0.129 | +0.111 | +0.127 | +0.130 | +0.140 | +0.135 | +0.143 | +0.168 |

- **One pass is what makes a head predict at all.** Four passes leave it
  at -0.14, as the seed is at -0.11; every run of one pass is between
  +0.17 and +0.32.
- **A game's result at four of its positions teaches more than at all
  72.** With an eighteenth of the targets (2,080 an update against 37,095)
  **Once, few** explains 0.32 of the results of new games after 120
  updates, where **Once** explains 0.14 over the same updates and 0.18 at
  its end. Against the worlds it tells three quarters of what can be told,
  and its values can be taken as they stand (73%). That is what the small
  network tells with the opponent's deck when it is taught the worlds'
  rates themselves, in every band of turns: 72, 70, 79, 79 and 76% for
  turns 1-2, 3-4, 5-6, 7-10 and from the eleventh.
- **Once ranks the positions and is too sure of them.** A correlation of
  0.51 with the results allows 0.26 explained, and it explains 0.18; its
  fit does not rise over the run. The same shows against the worlds, 65%
  by a straight line and 52% as its values stand. A reading of this, not
  measured: the 72 positions of a game come in many steps of the one pass,
  and the later ones find the game's result already learned.
- **The head that reads without shaping is not too sure and ranks no
  better.** It explains 0.24 at a correlation of 0.49; against the worlds
  it tells 54% where **Once** tells 65%.
- The board as its cards gives the head nothing: 0.168 against 0.183.

### The copies

Agreement is with the pilot on games new to the run, over the updates of
the table above. The suite is the mean of a run's last two evaluations,
1,664 games each, where the pilot wins 60.3%. Of the 1,536 held-out deals
the pilot wins 60.7% (933) and the search 72.5%.

| Run | Agreement | Suite | Held-out deals after 120 updates | 240 | 360 | Under the pilot at its last |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| The seed, version 89 | | | | | 55.1% | -5.6 ± 1.0 |
| Again, 90 updates | 94.7% | 56.8% | | | 57.2% | -3.6 ± 1.0 |
| Once | 95.3% | 57.8% | 52.5% | 58.7% | 57.2% | -3.5 ± 0.9 |
| Once, few | 93.7% | 52.9% | 54.8% | | | -6.0 ± 1.1 |
| Once, reading | 95.8% | 58.6% | 55.2% | 59.2% | 58.2% | -2.5 ± 0.8 |
| Once, cards | 95.5% | 58.9% | 54.6% | 59.0% | 59.7% | -1.0 ± 0.9 |

Deal by deal on the held-out deals:

| | After 120 updates | 240 | 360 | Over the three |
| --- | ---: | ---: | ---: | ---: |
| Once, cards over Once | +2.1 ± 1.1 | +0.4 ± 1.0 | +2.5 ± 0.9 | +1.6 ± 0.6 |
| Once, reading over Once | +2.7 ± 1.0 | +0.5 ± 0.9 | +1.0 ± 0.9 | +1.4 ± 0.6 |
| Once, few over Once | +2.3 ± 1.0 | | | |
| Once over Again, at their last | | | +0.1 ± 0.9 | |
| Again over the seed | | | +2.0 ± 1.0 | |

- **One pass copies as well as four at the same optimizer steps**, on four
  times the games, which the pilot plays for nothing: 57.2% of the
  held-out deals both, 95.3% against 94.7% agreement, 57.8% against 56.8%
  of the suite.
- **Four positions for the value head take nothing from the copy.** Over
  the same updates **Once, few** agrees with its pilot as **Once** does
  (87.5, 92.2 and 93.7% by blocks of 40 against 87.7, 92.0 and 93.4%),
  wins as much of the suite (48.0, 52.1 and 53.8% against 48.7, 52.1 and
  53.0%) and 2.3 ± 1.0 points more of the held-out deals. Its lower
  figures in the table are those of a run stopped at 120 updates.
- **The board as its cards: the best copy at the end, and not shown over
  the run.** After 360 updates it wins 2.5 ± 0.9 points more of the
  held-out deals than **Once** and stands 1.0 ± 0.9 under its pilot where
  **Once**, **Once, reading** and **Again** stand 2.5 to 3.6 under; on
  the suite it is 1.4 under. After 240 updates the two were level, and
  over the three checkpoints the cards have 1.6 ± 0.6. A run's own
  checkpoints differ as much: **Once** wins 58.7% after 240 updates and
  57.2% after 360. Early in the run the cards agree sooner: 90.2% against
  87.7% over the first 40 updates.
- **Again** is 2.0 ± 1.0 points over the seed, which imitated games
  against the pilots the opponents had before.
- By opponent the copies are furthest from their pilot against
  Dragunity, Gladiator Beasts and Monarchs: of 128 held-out deals each the
  pilot wins 74, 90 and 42, and the three runs of 360 updates 65 to 68,
  78 to 83 and 35 to 38.

### The lead heads

One pass through the 7,680 logged games from the last checkpoints of
**Once** and **Once, cards**, 16 updates; the fit is the share of the
measured leads the head explains before an update, over the second half
of the pass.

| | Fit | Correlation on the 384 validation games |
| --- | ---: | ---: |
| From Once | +0.268 ± 0.005 | +0.463 |
| From Once, cards | +0.262 ± 0.005 | +0.451 |

The head of the last experiment, on the seed, reached 0.25 in its first
pass and 0.46 on the validation games. A body trained on four times the
games, each seen once, and the board as its cards leave what a lead head
tells where it was.

### The readings

| Reading | Mark | Measured | |
| --- | --- | ---: | --- |
| The value head predicts | Once explains 0.20 | +0.183 ± 0.007 | no |
| It reads the position | 0.30 | +0.183 | no |
| The copy is no worse for it | Once within two standard errors of Again | +0.1 ± 0.9 | yes |
| The copy is better | 2 points and two standard errors | +0.1 ± 0.9 | no |
| The head must shape what it reads | Once 0.05 over Once, reading | -0.057 ± 0.006 | no |
| Fewer positions of a game are needed | Once, few 0.05 over Once, updates 81 to 120 | +0.179 ± 0.005 | yes |
| The board as its cards helps the copy | 2 points and two standard errors over Once | +2.5 ± 0.9 | yes |
| It helps the heads | 0.05, value or lead | -0.015 ± 0.005, -0.006 | no |

Three of them need a word beside the mark.

- **The first is failed by the arm it was fixed on and passed by the arm
  beside it.** **Once** is 0.017 short of 0.20. **Once, few**, the same
  lesson at four positions a game, passes both marks, 0.32, at a third of
  the updates. What was written for a failure, that the results are then
  too poor a lesson and the head should be taught the worlds' rates, does
  not follow: the same results teach the head once a game no longer
  repeats them 72 times.
- **The board as its cards passes on the last checkpoint only.** Over the
  three checkpoints measured the difference is 1.6 ± 0.6, under the 2
  points; it is one training run against one.
- **The head need not shape what it reads, and the two measures part.**
  Reading only, it explains more of the results as its values stand and
  ranks the positions worse.

## What the run answers

1. *Trained so that it cannot learn its games by heart, does the value
   head tell who is winning?* Yes, where a game's result is its target at
   four positions and every game is seen once: 0.32 of the results of
   games new to it, three quarters of what can be told at the decisions
   the search examined, with values that can be used as they stand. Seen
   once at every position it ranks the positions and is too sure (0.18);
   seen four times it explains less than nothing (-0.14).
2. *Does the copy lose or gain by it?* Neither. At the same optimizer
   steps one pass copies as well as four, and four positions for the head
   copy as well as all of them over the updates both ran.
3. *The board as its cards?* The copy ends 2.5 ± 0.9 points over the
   pooled board and a point under its pilot; over the run the difference
   is half that and not secured. The heads gain nothing from it.

No run has both: the head that predicts is on a copy stopped at 120
updates, and the best copy has a head at 0.17.

## The follow-up run

One more run was agreed on 2026-10-09 before anything is built on
these arms: the head that predicts and the best copy in one network. It
was made on the CUDA machine the same day, by
[`learner-follow-up.sh`](commands/learner-follow-up.sh) at ygo-policies
`1b85b63`. Its figures are in `analysis/learner-follow-up/` on the
development machine, and the tables below are
`tools/learner_follow_up_report.py` on them (`report.json` there).

Four runs of one pass and 360 updates, the board by the positions a
game's result is taught at. **Once** and **Once, cards** are the runs
above; **Once, few** was taken on from its 120 updates; **Once, cards,
few**, the board as its cards and the result at four positions, is new.
Each was measured on the held-out deals after 240, 280, 320 and 360
updates, so that a copy is read over four checkpoints of a run where it
was read at one. The fingerprint of an update's decisions is the same in
the four runs at every update.

Its readings were proposed on 2026-10-09 before the run and fixed the
same day, before any of it was seen. A copy is read deal by
deal over the four checkpoints: the share of them at which a run wins a
deal.

- *With both, the value head predicts*: **Once, cards, few** explains at
  least 0.30 of the results of games new to it over its last 40 updates,
  as **Once, few** did after 120.
- *With both, the copy is no worse than with the cards alone*: it wins no
  fewer held-out deals than **Once, cards** by more than two standard
  errors.
- *On the pooled board four positions cost the copy nothing*: the same of
  **Once, few** against **Once**.
- *The board as its cards helps the copy*: the two runs on the cards win
  at least 2 points more than the two on the pooled board, and more than
  two standard errors.

What PPO starts from was to follow: **Once, cards, few** if the cards
help, and **Once, few** if they do not, the pooled board being the simpler
one and that of every checkpoint so far.

### The value heads, after 360 updates

Over a run's last 40 updates and, against the worlds, at its last
checkpoint: the columns of the table above.

| Run | Fit | Correlation | With the worlds' rates | Of what can be told, by a straight line | As its values stand | Of one result |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Once | +0.183 ± 0.007 | +0.51 | +0.747 | 64.7% | 51.9% | 28.4% |
| Once, few | +0.349 ± 0.006 | +0.60 | +0.839 | 81.7% | 81.3% | 35.8% |
| Once, cards | +0.168 ± 0.008 | +0.50 | +0.723 | 60.6% | 43.4% | 26.6% |
| Once, cards, few | +0.344 ± 0.006 | +0.59 | +0.835 | 80.8% | 80.1% | 35.4% |

The fit by blocks of 40 updates:

| Updates | 1-40 | 41-80 | 81-120 | 121-160 | 161-200 | 201-240 | 241-280 | 281-320 | 321-360 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Once, few | +0.237 | +0.301 | +0.320 | +0.321 | +0.326 | +0.342 | +0.337 | +0.338 | +0.349 |
| Once, cards, few | +0.225 | +0.293 | +0.316 | +0.317 | +0.321 | +0.337 | +0.333 | +0.332 | +0.344 |

- **The head taught at four positions goes on learning, slowly.** 0.32
  after 120 updates and 0.35 after 360. Against the worlds it now tells
  four fifths of what can be told, 82% where it told 75% after 120
  updates, and its values stand as they are (81%). That is more than the
  small network tells when it is taught the worlds' rates themselves with
  the opponent's deck (75.1%), in every band of turns: 78, 78, 83, 85 and
  83% for turns 1-2, 3-4, 5-6, 7-10 and from the eleventh.
- **The board makes no difference to the head**: 0.344 on the cards and
  0.349 on the pooled board, 81% and 82% against the worlds.

### The copies, over four checkpoints

Of the 1,536 held-out deals the pilot wins 60.7%. The suite is the mean
of a run's last two evaluations, where the pilot wins 60.3%.

| Run | Suite | Held-out deals after 240 updates | 280 | 320 | 360 | Over the four | Under the pilot at its last |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Once | 57.8% | 58.7% | 57.2% | 58.3% | 57.2% | 57.8% | -3.5 ± 0.9 |
| Once, few | 57.1% | 58.1% | 57.3% | 59.2% | 59.6% | 58.6% | -1.1 ± 0.9 |
| Once, cards | 58.9% | 59.0% | 58.5% | 60.0% | 59.7% | 59.3% | -1.0 ± 0.9 |
| Once, cards, few | 58.9% | 59.5% | 60.2% | 59.0% | 58.8% | 59.4% | -2.0 ± 0.9 |

Deal by deal, the share of the four checkpoints at which a run wins a
deal:

| | Points |
| --- | ---: |
| Once, few over Once | +0.7 ± 0.5 |
| Once, cards, few over Once, cards | +0.1 ± 0.4 |
| Once, cards over Once | +1.4 ± 0.6 |
| Once, cards, few over Once, few | +0.8 ± 0.5 |
| The two on the cards over the two on the pooled board | +1.1 ± 0.4 |

- **Four positions for the head cost the copy nothing on either board.**
- **The cards are worth a point, not two.** 1.1 ± 0.4 over two pairs of
  runs and four checkpoints of each. The 2.5 ± 0.9 of the first run's
  last checkpoint was one difference between two runs whose own
  checkpoints move by 1.4 to 2.3 points.
- **Every copy is under its pilot**: by 1.3 to 2.9 points over the four
  checkpoints.

### Its readings

| Reading | Mark | Measured | |
| --- | --- | ---: | --- |
| With both, the value head predicts | Once, cards, few explains 0.30 | +0.344 ± 0.006 | yes |
| With both, the copy is no worse than with the cards alone | within two standard errors of Once, cards | +0.1 ± 0.4 | yes |
| On the pooled board four positions cost the copy nothing | Once, few within two standard errors of Once | +0.7 ± 0.5 | yes |
| The board as its cards helps the copy | 2 points and two standard errors over the pooled board | +1.1 ± 0.4 | no |

The last fails on its size and not on its sign: the difference is 2.7
standard errors from nothing and half the mark.

### What it answers

The head that predicts and a copy as good as any are in one network, on
either board: four positions take nothing from the copy, and the head is
the same on both. The cards add a point to the copy, which is less than
was asked of them. By what was fixed before the run, **PPO starts from
Once, few after 360 updates**: the pooled board, a head that explains
0.35 of the results of games new to it, and 58.6% of the held-out deals
over its four checkpoints, 59.6% at its last, where the pilot wins 60.7%.

## Conclusion and what follows

Closed on 2026-10-09, when
[PPO with a head that predicts](2026-10-09-ppo-with-a-head-that-predicts.md)
became the active experiment.

1. **A value head tells who is winning once it cannot learn its games by
   heart.** Every game seen once, and its result the head's target at four
   of its positions: it then explains 0.35 of the results of games new to
   it and tells four fifths of what can be told at the decisions the
   search examined, with values that stand as they are. At every position
   of a game seen once it explains 0.18, and with the four passes of every
   run before this one, less than nothing.
2. **The copy loses nothing by it.** One pass copies as well as four at
   the same optimizer steps, and four positions for the head as well as
   all of them.
3. **The board as its cards is worth a point to the copy** (1.1 ± 0.4 over
   two pairs of runs and four checkpoints of each), under the two that
   were asked of it, and nothing to a head. The pooled board stays.
4. **Every copy is one to three points under its pilot**, which this
   experiment did not set out to close.

What further work starts from is **Once, few** after 360 updates
(`runs/learner/once-few/checkpoints/model-00359.pt` on the CUDA machine,
format 14, board `mean-pool`): 58.6% of the held-out deals over its last
four checkpoints where its pilot wins 60.7%, and a head at 0.35.

**PPO's own lesson for the value head is the one that fails here.** Its
update gives the head a target at every transaction of an update's games
and goes several times through them. `--ppo-value-positions K`, built on
2026-10-09, teaches the head at `K` transactions of each seat of a duel,
and `--ppo-value-target result` the duel's result where PPO would teach
its own return; an update reports the head against the results of its
games before it trains on them (`duel_outcome_explained_variance`). The
next experiment is the first to use them.

Beside this experiment, and decided the same day: the trainer moves to a
duel environment that steps many duels at once and gives the model whole
batches, built in `crates/ygo-env`. With the network playing, a worker
today waits for its answer nine tenths of its time. It is in behind
`--pool-threads`, off by default: evaluations and an update's games can
go through it ([docs/commands.md](../docs/commands.md)). An update that
imitates the pilot through it is the workers' update to the last digit,
so this experiment's runs are the same either way; the follow-up's
command does not use it. On the CUDA machine
(`experiments/commands/pool-check.sh`, 2026-10-09) three imitating
updates of 520 games ended in the same model by both routes, in 7.2 s an
update where the workers take 9.9 s, and an update whose games the model
plays took 13.3 s at one pass where it takes 28.8 s.

## The tooling

Written on 2026-10-08; the commands are in
[docs/commands.md](../docs/commands.md).

- **The value head's target.** `train --imitation-value-positions K`: of
  each seat's transactions in a duel, K drawn at random carry the outcome
  as the head's target in an imitation-only update, where every one did.
  `--imitation-value-detach`: the head reads the shared state and sends no
  gradient into it. An update also reports how many transactions carried
  the outcome and the correlation of the head's values with the outcomes
  before it trained, beside the explained variance, which a head too sure
  of itself loses though it ranks the positions rightly.
- **Checkpoints.** `train --keep-checkpoints N` keeps the checkpoint of
  every Nth update and the latest and removes the others of its own
  invocation as it goes: 360 updates would otherwise leave 30 GB a run.
- **The measure against the worlds.** The records give, for every examined
  decision, the mean result of the pilot's answer over the worlds of the
  first stage (`rate`), and `evaluate-search` sets the model's value
  against it at the decisions that open a transaction, over all of them
  and by turn.
- **The references.** `tools/policy_search_predictability.py TRAIN
  MEASURE`: of one result, the share a perfect predictor explains at the
  decisions of the kinds that open a transaction, and what a network of
  6,000 weights on the log's `situation` tells, taught the games' results
  or the worlds' rates, without and with the opponent's deck.
- **The run.** [`learner-value-board.sh`](commands/learner-value-board.sh)
  does the whole of the design: the references, the seed's own value head
  on new games, the five runs from scratch, the two passes with the lead
  head, and the measurements. Finished parts are skipped and an
  interrupted run goes on.
- **The report.** `tools/learner_report.py` reads the folder the run
  leaves: the value heads, the copies, the lead heads, and the readings
  with their marks.
- **The follow-up** (2026-10-09).
  [`learner-follow-up.sh`](commands/learner-follow-up.sh) goes on from the
  runs of the first script and `tools/learner_follow_up_report.py` applies
  its readings. Checked on this machine on the CPU, on 26 games an update
  and eight logged games: four runs from scratch, every run taken on from
  4 updates to 8 with the games a run of 8 has, a second start that does
  nothing over, and the report.

Checked on this machine, on the CPU:

- The references at the earlier pilots, the small network taught on 1,382
  games of the search-imitation experiment's held-out log and measured on
  the 384 development games: a perfect predictor explains 44.1% of one
  result at 19,170 decisions (26.6% in the first two turns, 57.6% from the
  eleventh). The network tells 45.6% of what can be told when it is taught
  the games' results, 20.1% of one result, and 64.5% with the opponent's
  deck, 28.5% of one result; taught the worlds' rates 50.7% and 73.8%,
  22.4% and 32.5% of one result. The first look above stands.
- On worths drawn at random and rates of 8 worlds drawn from them, a head
  that knows every worth is read as telling all there is, one that is
  twice as sure as telling as much and explaining nothing as it stands,
  and one that knows nothing as telling nothing (a test).
- The script runs through on 26 games an update and four logged games a
  run: five runs from scratch, the two with the lead head, every
  measurement. A second start does nothing over, a run made longer goes
  on from its latest checkpoint, and the checkpoints left are those of
  every second update and the latest, as asked there.
