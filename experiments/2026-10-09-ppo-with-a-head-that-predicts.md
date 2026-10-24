# PPO with a head that predicts

**Status:** COMPLETED

**Proposed:** 2026-10-09

**Activated:** 2026-10-09

**Completed:** 2026-10-11

**Last updated:** 2026-10-20 (the specialist copy's teacher chosen and its command tried small)

Proposed on 2026-10-09, when the follow-up of
[the learner itself](2026-10-08-value-head-and-board.md) had been read.
That experiment was closed the same evening and this one opened, with the
design and the readings below fixed as they were proposed, before any of
its runs. The five runs were made on the CUDA machine on 2026-10-10 and
read the same day: the result is below, and after it a follow-up, run and
read that day, with its third seed and a second long run read on
2026-10-11. Every reading fixed here has been taken. It was closed on
2026-10-11 for
[Self-play, and the named plays](2026-10-11-self-play.md), which
starts from the second long run's last checkpoint.

## Research questions

1. Does the value head go on predicting under PPO, on the games the model
   plays itself, when it is taught as the last experiment found a head
   must be taught?
2. With such a head, does PPO still undo a copy of the pilot that has no
   anchor?
3. Does it add to the copy, without an anchor or with one?

## Why

PPO has been tried twice on this engine from an imitation of the
Blackwing pilot, and has not made a player yet.

- **Without an anchor it undoes the copy.** From the imitation seed the
  long branch fell from 50.4% to 35.5% of the suite in 90 updates, its
  agreement with the pilot from 92% to 33%, mostly by activating in chains
  where the pilot passes
  ([archetype-fix re-baseline](2026-09-28-archetype-fix-rebaseline.md)).
  The entropy term was ruled out there, since without it the drift is
  faster, and no single term of an update drives it: it is a slow
  accumulation through shared parameters, held back only where a
  decision's own credit is strong.
- **With an anchor it holds and adds nothing.** The pilot's answers on the
  learner's own states, at a weight of 0.1 or 1, stop the drift for as
  long as they stay: taken away after 50 updates, the copy lost 12 points
  in 20. Over 110 updates and 57,200 duels the anchored learner and one
  trained on those labels alone ended within a point of each other and of
  the pilot ([PPO regression remedies](2026-09-29-ppo-regression-remedies.md)).
- **Every one of those runs taught the value head in the way that
  fails.** Four passes over an update's 520 games, the head's target at
  every transaction. The last experiment measured what that gives: a head
  that explains less than nothing of the results of games new to it
  (-0.14 trained so again, -0.11 for the seed), where a head that sees
  every game once with its result at four of its positions explains 0.35,
  and the copy loses nothing by it.
- **And their credit leaned on that head.** With `--duel-gae-lambda 0.95`,
  which every one of them used, the advantage of a transaction is mostly
  how the head's estimate moves over the twenty or so transactions after
  it, and little of how the duel ended. The head's own target was that
  return too, its later estimates of the same duel added to its result:
  the `duel_explained_variance` of 0.70 those runs report is against such
  returns and says nothing of results.

That the head of those PPO runs told nothing of new games is an inference
from the imitating runs: their figures hold no measure of it, and the
first arm below takes one. If it holds, an answer's own credit was noise
nearly everywhere, and a head that predicts is the plainest way to make it
stronger, which is what the earlier record left open.

## Design (fixed on 2026-10-09)

Five runs of PPO from one checkpoint, **Once, few** of the last
experiment after 360 updates
(`runs/learner/once-few/checkpoints/model-00359.pt` on the CUDA machine:
format 14, the pooled board, a head at 0.35, 58.6% of the held-out deals
over its last four checkpoints where its pilot wins 60.7%). Each is 120
updates of 520 games the model plays with sampled answers against the
thirteen decks' pilots, 62,400 games, with the seed, the learning rate
and the entropy term of the runs before (2012, 3e-4, 0.01), and no anchor
unless said.

| Arm | Passes | The head is taught at | Its target | An answer's credit | Options |
| --- | ---: | --- | --- | --- | --- |
| **As before** | 4 | every transaction | PPO's return | leans on the head | `--ppo-epochs 4` |
| **One pass** | 1 | every transaction | PPO's return | leans on the head | `--ppo-epochs 1` |
| **Head kept** | 1 | four transactions of a seat | the duel's result | leans on the head | `--ppo-epochs 1 --ppo-value-positions 4 --ppo-value-target result` |
| **Head kept, plain credit** | 1 | four transactions of a seat | the duel's result | the result minus the head's estimate | the same and `--duel-gae-lambda 1` |
| **Head kept, anchored** | 1 | four transactions of a seat | the duel's result | leans on the head | the same as **Head kept** and `--teacher-imitation-weight 0.1` |

- **As before** is PPO as every earlier run made it. It says whether the
  drift is there from this start, against these pilots, and what its head
  tells.
- **One pass** changes the number of passes alone, and **Head kept** the
  head's lesson with it: the two changes the last experiment found a head
  needs.
- **Head kept, plain credit** does not lean on the head: it says whether
  a head that predicts is better leaned on or only subtracted.
- **Head kept, anchored** is there for the case that PPO still drifts:
  whether, held in place, it adds anything now.
- The games go through the library's pool in every arm
  ([docs/commands.md](../docs/commands.md)), so the arms draw their
  sampled answers in the same way and none is set beside a run that went
  through the workers. Their first update has the same games.
- An arm of four passes takes four times the optimizer steps of an update:
  **As before** after 30 updates has taken as many as the others after
  120.

Each arm is measured after 30, 60, 90 and 120 updates on the 1,536
held-out logged games of the last two experiments (`evaluate-search`).

## Measures

- **The head**: the share of the results it explains at the transactions'
  first steps of an update's games before it trains on them
  (`duel_outcome_explained_variance`), over an arm's last 40 updates:
  20,800 games the model plays itself.
- **The copy**: deals won of the 1,536 held-out deals by the model's first
  choices. A copy is read deal by deal, by the share of its checkpoints at
  which it wins a deal: an arm's after 60, 90 and 120 updates against the
  four of its start (after 240, 280, 320 and 360 updates of its own run).
  A run's own checkpoints move by 1.4 to 2.3 points, and the start's last
  is its best of four.
- **Where it leaves its pilot**: on the same games, how often the first
  choice is another answer than the pilot's where the search kept the
  pilot's, by the kind of the decision. At the start: 4.9% of all of them,
  0.7% in chains, 11.7% with the turn's choices open and 4.1% in battle.
- **Beside them**: the entropy of the sampled answers, how far an update
  moves the policy, the games won while sampling, and the suite.

## Readings (fixed on 2026-10-09)

Proposed and confirmed on 2026-10-09, before any of the runs. Each arm is
one training run: the errors are those of the deals measured.

- *Under PPO the head keeps predicting*: **Head kept** explains at least
  0.30 of the results of its own games before it trains on them, over its
  last 40 updates.
- *PPO no longer undoes the copy*: **Head kept** wins no fewer held-out
  deals than its start by more than two standard errors.
- *PPO adds to the copy*: **Head kept** wins at least 2 points more than
  its start, and more than two standard errors. The same is read for
  **Head kept, anchored**.
- *The head's lesson is what holds it*: **Head kept** wins at least 2
  points more than **As before**, and more than two standard errors.

**One pass** and **Head kept, plain credit** carry no mark: they say which
of the changes did what the marked arms show.

What would follow:

- **The head predicts and the copy holds**: PPO can be run long, to see
  whether it climbs. A million games are about seven hours through the
  pool.
- **The copy does not hold and the anchored arm adds to it**: PPO adds
  something only while it is held in place, and the anchor is the next
  question.
- **The copy does not hold and nothing adds to it**: PPO is left, for a
  learner that uses the head in another way.
- **The head does not predict**: the lesson does not survive the games
  the model samples itself, and that is looked at before anything else.

## Beside the experiment: their trainer on the same card (2026-10-10)

Measured for the benchmark of ygo-agent before any arm here ran: the
record is `benchmarks/ygo-agent/README.md`, its section on how fast it
learns, on the branch `ygo-agent-benchmark-work` of ygo-policies, and the
two reports are in `analysis/curve/` on the development machine. Their own
trainer, from scratch, by self-play on the Blue-Eyes deck against itself,
ten hours on the CUDA machine: 206.6 million steps, about 980,000 games.

- It is past the benchmark's pilot after 28,000 games and twenty minutes,
  and past that pilot as improved on 2026-10-09 after 37,000.
- After 356,000 games and four hours it scores against either pilot what
  the released model scores (84% and 77%), and after 764,000 games and
  eight hours it wins 64.5% ± 3.0% of its games against the released model
  itself in that matchup.

What it bears on here:

- **PPO by self-play learns this game at a scale one card reaches**, on
  one matchup. The hundred million games of the released model are those
  of its 32 decks.
- **Their recipe shares three choices with the arms here**: one pass over
  an update's steps, an answer's credit leaning on the value head
  (lambda 0.95), no anchor. Its head is taught PPO's return at every step,
  as **One pass** teaches it.
- **It differs in the rest**, read in `scripts/cleanba.py` of ygo-agent at
  `26293f8`:
  - one network plays both seats and learns from both;
  - it starts from nothing;
  - its 32 duels play on from one update to the next, and an update is
    the next 128 steps of each, 4,096 steps or some twenty games, with
    credit cut at the end of the stretch and the head's estimate standing
    for the rest; an update here is 520 whole games;
  - the learning rate is 1e-4, a minibatch 512 steps, the clip 0.25;
  - an answer keeps the credit of the result where the play went better
    than the head expected (UPGO);
  - the network has 4.3M parameters and carries a state through the game;
  - the environment and the input are their own.

  Its reward is ours, 1 or -1 at a game's end.
- **A failure of the arms here would no longer say that PPO is to be
  left**, as it is written above from before this was measured. It would
  ask which of those differences it takes. The readings stand as they
  are.
- **Decided on 2026-10-10**: what follows this experiment is to find which
  of those differences matters and to bring it into this trainer. Training
  with their trainer on our decks, measured against our pilots through the
  benchmark's bridge, is the fallback if that is not found.

## The run

On the CUDA machine, after a pull and after the follow-up of the last
experiment, whose run and measurements it starts from:

```sh
bash experiments/commands/ppo-head.sh
```

About four hours: the five runs, then each on the held-out games, then
the readings. `analysis/ppo-head/` comes back, with the readings in
`report.txt`. A run that is complete is left as it is and an interrupted
one goes on.

## The tooling

Written on 2026-10-09.

- **`--ppo-value-target result`** (`ygo_training/optimization.py`): in an
  update whose games the model plays, the head's target is how its seat's
  duel ended, where PPO's return would hold the head's own later
  estimates. The advantages are measured against the head as before.
  With `--ppo-value-positions`, of the last experiment, it is the
  imitating updates' lesson for the head, made in PPO.
- **[`ppo-head.sh`](commands/ppo-head.sh)** and
  **`tools/ppo_head_report.py`**: the arms, the measurements and the
  readings. The report says first whether the arms' first update had the
  same games.
- **Tried on the development machine**: the five arms for four updates of
  26 games from a small copy, measured twice each on eight held-out games.
  The four arms without an anchor had the same first games, and the
  anchored one the same decisions with the pilot's answers beside them.
  The report's figures were checked on the follow-up's measurements put in
  the arms' places.

## Result (2026-10-10)

The five runs and their measurements took three hours on the CUDA machine;
`analysis/ppo-head/` is on the development machine. The arms' first update
had the same games.

### The readings

| Reading | Measured | Mark | Met |
| --- | --- | --- | --- |
| Under PPO the head keeps predicting | **Head kept** explains 0.349 ± 0.006 of the results of its own games | 0.30 | yes |
| PPO no longer undoes the copy | **Head kept** over its start: +2.8 ± 1.0 | no fewer by two standard errors | yes |
| PPO adds to the copy | **Head kept** over its start: +2.8 ± 1.0 | 2 points and two standard errors | yes |
| The same with the anchor | **Head kept, anchored** over its start: +2.4 ± 0.7 | 2 points and two standard errors | yes |
| The head's lesson is what holds it | **Head kept** over **As before**: +14.0 ± 0.9 | 2 points and two standard errors | yes |

All five are met. The last is met by its mark and not by its name: the two
arms it sets beside each other differ by the number of passes and by the
head's lesson, and **One pass**, which changes the passes alone, holds the
copy as well.

### The arms

| Arm | The head explains | After 30 | 60 | 90 | 120 | Read | Over its start | Another answer than the pilot's |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| The start, after 240 to 360 updates of its own | 0.35 | 58.1% | 57.3% | 59.2% | 59.6% | 58.6% | | 4.9% |
| **As before** | 0.372 | 45.6% | 46.7% | 47.3% | 48.0% | 47.4% | -11.2 ± 1.0 | 43.3% |
| **One pass** | 0.367 | 59.1% | 60.2% | 63.3% | 65.2% | 62.9% | +4.3 ± 1.0 | 15.6% |
| **Head kept** | 0.349 | 60.3% | 59.9% | 63.0% | 61.3% | 61.4% | +2.8 ± 1.0 | 13.0% |
| **Head kept, plain credit** | 0.354 | 50.0% | 55.2% | 53.6% | 48.5% | 52.4% | -6.1 ± 0.9 | 15.0% |
| **Head kept, anchored** | 0.345 | 58.2% | 60.3% | 60.1% | 62.5% | 61.0% | +2.4 ± 0.7 | 6.8% |

The percentages are the 1,536 held-out deals won, of which the pilot wins
60.7% and the pilot with its search 72.5%. An arm is read by its
checkpoints after 60, 90 and 120 updates and the start by its four. The
head's figure is over an arm's last 40 updates. The last column is at an
arm's last checkpoint, at the decisions where the search kept the pilot's
answer.

### What the arms say

- **The number of passes is what undid the copy.** **As before** and
  **One pass** differ in nothing else, and the second wins 15.5 ± 0.9
  points more of the deals. It is not the number of optimizer steps:
  **As before** has taken 16,000 after 30 updates and wins 45.6%, and
  **One pass** has taken as many after 90 and wins 63.3%. In four passes
  an update
  moves the policy by a KL of 0.05 where one pass moves it by 0.02.
  **As before** drifts as the earlier records describe: the entropy of its
  answers goes from 0.16 to 0.67 in 40 updates, it activates at 9% of its
  answers in a chain at first and at 45% after 70 updates, and it wins 53%
  of its sampled games at first and 30 to 34% from the twentieth update
  on.
- **The head predicted in every arm**, in every ten updates of every run
  (0.33 to 0.39), **As before** among them while its copy fell. The
  inference this experiment began from does not hold: PPO's own lesson,
  the return at every transaction, keeps a head predicting that starts so,
  in four passes too. Why it differs from the imitating updates' result at
  every transaction, which four passes learn by heart, is not measured:
  the return of a transaction is mostly the head's own later estimates of
  the duel, and holds little of a result to learn. Whether that lesson
  would have taught the head of the earlier runs' seed, which explained
  -0.11, is not measured either. The drift is not for want of a head that
  predicts.
- **The head's lesson adds nothing to one pass**: **Head kept** against
  **One pass** is -1.5 ± 0.8.
- **An answer's credit has to lean on the head.** With the result less the
  head's estimate for credit the copy loses 6.1 ± 0.9 points, 9.0 ± 0.8
  under **Head kept**. Those advantages spread by 0.80 where the leaning
  ones spread by 0.34: most of a result is the luck of its game, which
  every answer of the game is then credited with alike.
- **The anchor is not needed, and adds nothing**: -0.5 ± 0.9 against
  **Head kept**. It keeps the first choices at the pilot's (93% agreement
  on the held-out games, where the other arms of one pass end at 81 to
  84% and the start has 95%).
- **PPO adds to a copy, for the first time on this engine.** The three
  arms of one pass whose credit leans on the head all gain, and all win
  more of their sampled games as they go (**One pass** 56% in its first
  ten updates, 61 to 63% in its last fifty). **One pass** after 120
  updates wins 4.4 ± 1.3 points more of the held-out deals than the pilot
  it copies does on the same deals, and 2.6 ± 1.3 after 90; that is the
  best of the twenty checkpoints measured here. On the suite it wins 62.7%
  and 64.7% after 90 and 120 updates, where the pilot wins 60.3%. Its gain
  over the start is spread over the opponents: by its checkpoints after 90
  and 120 updates, 5 points or more against seven of the twelve, and no
  more than 1.2 lost against any.
- **What it changed.** At the last checkpoint the first choice of
  **One pass** is another answer than the pilot's at 15.6% of the
  decisions where the search kept the pilot's, and at 32.1% of those with
  the turn's choices open; in chains at 4.5%. At the search's 2,357
  deviations it gives the search's answer at 13.8% where its start gave it
  at 2.0% (**Head kept** 15.1%, **Head kept, anchored** 6.2%). Its games
  end a turn sooner (11.9 turns against 13.0). The arms did not change the
  same answers: **One pass** special-summons more often than its start
  (2.4 times a game against 1.9) and **Head kept** less (1.4), attacking
  more (7.6 times against 6.1).
- **The measure by the search's worlds does not show the gain.** The other
  answers of **One pass** lose 1.0 ± 0.1 points each by the leads the
  search measured, where the start's lost 2.0 ± 0.2, but it gives four
  times as many (7.3 a game against 1.8), and their sum is -7.1 ± 1.0 a
  game against -3.7 ± 0.5. That measure values one answer changed with the
  pilot playing on from it. Not looked into.

### Limits

- **One training run an arm.** The errors are those of the deals. **One
  pass** and **Head kept** are no further apart than a run's own
  checkpoints are, and no order between them is shown.
- **62,400 games.** Whether PPO goes on adding is not measured: the sampled
  games of **One pass** level at 61 to 63% over its last fifty updates.
- **A head that does not predict was not among the arms.** Every arm began
  with one that does and kept it.

### Speed

Over a whole run with its evaluations and checkpoints the arms of one pass
played and learned 115,000 to 131,000 games an hour (14.3 to 16.3 s an
update), and **As before** 100,000. ygo-agent's own trainer makes 98,000
duels an hour on the same card.

### The default number of passes

Decided on 2026-10-10, on this result: the trainer's default is one pass
over an update's games (`--ppo-epochs 1`), where it was four
(`TrainingProtocolDefaults` in `ygo_training/config.py`). The scripts of
earlier experiments that took the default now name four, so that they make
the runs their records describe and go on with the runs they made: the
imitation seed and its value check, and the arms of the search-imitation
and search-leads experiments.

## Follow-up: one pass, again and further (readings fixed on 2026-10-10)

Agreed on 2026-10-10 as what is prepared next. The readings below were
proposed that day and confirmed as they were proposed, before any of its
runs. Nothing of it has run.

### Questions

1. Was the gain of **One pass** one run's luck?
2. Run long, does PPO in one pass go on adding to the copy, hold what it
   added, or come undone?
3. Beside them: what does the learning rate of ygo-agent's trainer do at
   this length?

### Design

| Run | From | Updates | What differs from **One pass** |
| --- | --- | ---: | --- |
| **One pass, another seed** (`one-pass-seed`) | the start, **Once, few** after 360 updates | 120 | its games are drawn with seed 2013 (`--training-seed 2013`) |
| **One pass, their rate** (`one-pass-rate`) | the same | 120 | a learning rate of 1e-4 (`--resume-learning-rate 1e-4`) |
| **One pass, long** (`one-pass-long`) | **One pass** after its 120 updates | 1,920 in all, 998,400 games | nothing: it is that run, continued |

- The suite of **One pass, another seed** stays on the deals of seed 2012,
  where the pilot wins 60.3%: `--seed` is unchanged.
- A run that resumes from a checkpoint keeps the learning rate saved with
  its optimizer, whatever `--learning-rate` says: the rate is changed by
  `--resume-learning-rate`.
- **One pass, long** is made in a folder of its own from the last
  checkpoint of **One pass**. Tried at a small size, that gives the games
  and the losses of the first run continued in place, update for update.
- The two short runs are measured as the arms were, after 30, 60, 90 and
  120 updates, and read over the last three against the start's four. The
  long run keeps a checkpoint every 120 updates, plays the suite then, and
  each is measured on the 1,536 held-out deals; it is read by its last
  three, after 1,680, 1,800 and 1,920 updates.

### Readings (fixed on 2026-10-10)

Proposed and confirmed on 2026-10-10, before any of the runs. Each run is
one training run: the errors are those of the deals measured.

- *The gain is not one run's luck*: **One pass, another seed** wins at
  least 2 points more of the held-out deals than its start, and more than
  two standard errors. This is the first experiment's mark for a copy
  PPO adds to.
- *Run long, one pass does not undo what it added*: the long run's last
  three checkpoints win no fewer deals than its checkpoint after 120
  updates (65.2%) by more than two standard errors.
- *PPO goes on adding*: they win at least 2 points more than that
  checkpoint, and more than two standard errors.
- *It is past its pilot*: they win at least 2 points more than the pilot
  does on the same deals (60.7%), and more than two standard errors.

**One pass, their rate** carries no mark: it is read beside how far the two
seeds are apart. The pilot with its search wins 72.5% of those deals, which
is a reference and no mark.

The checkpoint after 120 updates is what the long run is set against, not
the three the first experiment read **One pass** by (62.9%): those hold
its climb, and a run that only stayed where it was after 120 updates would
pass a mark set against them.

What would follow:

- **It goes on adding**: PPO against the pilots is a learner that climbs.
  The differences from ygo-agent's trainer are then tried for what they
  add to a base that works, and a longer run is worth its hours.
- **It holds and adds no more**: this recipe against fixed pilots gave
  what it gives in its first 60,000 games, and the differences are what is
  tried next: their rate if it helps here, self-play on both seats, a
  state carried through the game, a larger network.
- **It comes undone**: one pass delays the drift and does not stop it. The
  anchored arm, which stayed at its pilot's answers, and the size of a
  step are looked at before anything else.
- **The second seed does not gain**: the first experiment's gain is not
  established, and a third seed is run before the long run is read for
  more than itself.

### The run

On the CUDA machine, after a pull:

```sh
bash experiments/commands/ppo-long.sh
```

About nine and a half hours: 75 minutes for the two short runs and their
measurements, after which their part of the report is printed, and the
rest for the long run. `PARTS=long` or `PARTS=short` makes one of the two.
A run that is complete is left as it is, an interrupted one goes on, and
with a larger `LONG` the long run goes on and is measured on.
`analysis/ppo-long/` comes back, with the readings in `report.txt`.

### The tooling

Written on 2026-10-10.

- **[`ppo-long.sh`](commands/ppo-long.sh)** and
  **`tools/ppo_long_report.py`**: the three runs, the measurements and the
  readings. The long run's readings are taken only when it has had its
  updates.
- **A measurement keeps the pilot's result on every deal**
  (`deals.references.pilot.outcomes`, `ygo_training/evaluation.py`): the
  long run's last checkpoints are set beside the pilot deal by deal.
- **Tried on the development machine**: the three runs from a small copy,
  the short ones for four updates of 26 games and the long one to ten,
  each checkpoint on eight held-out games; then the long run made longer,
  killed part of the way and gone on with. The report's figures were
  checked on the first experiment's measurements put in the runs' places.
- **A run that is stopped loses no update's figures**, since the two
  commits after this was prepared (`ygo_training/checkpoints.py` and the
  loop of `train`). A checkpoint is written under another name and takes
  its own when it is whole and its update has been evaluated, its line of
  figures following at once. Before that, a run killed while it evaluated
  an update went on from that update's checkpoint without its line and
  its evaluation, and one killed while it saved could leave a checkpoint
  that does not load. Tried on the development machine: the long run at a
  small size killed while an update was evaluated and gone on with has
  the lines, the games, the losses, the suite and the weights of the same
  run left alone, update for update.

### Result of the follow-up (2026-10-10)

Run on the CUDA machine on 2026-10-10 and read that evening from
`analysis/ppo-long/` (`report.txt`). The two short runs took half an hour
each; the long run 7.8 hours for its 1,800 further updates, which is
120,000 games an hour with its suites.

| Reading | Measured | Mark | Met |
| --- | --- | --- | --- |
| The gain is not one run's luck | **One pass, another seed** over its start: +1.5 ± 1.0 | 2 points and two standard errors | **no** |
| Run long, one pass does not undo what it added | its last three checkpoints over the one after 120 updates: +11.7 ± 1.1 | not under by two standard errors | yes |
| PPO goes on adding | +11.7 ± 1.1 | 2 points and two standard errors | yes |
| It is past its pilot | its last three checkpoints over the pilot: +16.1 ± 1.2 | 2 points and two standard errors | yes |

**The runs of 120 updates.** Held-out deals won, and over the last three
checkpoints against the start's four (58.6%):

| Run | 30 | 60 | 90 | 120 | Read | Over its start | Suite, last two |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **One pass** (seed 2012, 3e-4) | 59.1% | 60.2% | 63.3% | 65.2% | 62.9% | +4.3 ± 1.0 | 63.7% |
| **One pass, another seed** (2013) | 60.2% | 57.9% | 60.5% | 61.8% | 60.1% | +1.5 ± 1.0 | 60.9% |
| **One pass, their rate** (1e-4) | 60.9% | 64.6% | 66.3% | 67.0% | 66.0% | +7.4 ± 0.9 | 66.2% |

- **The second seed gains less than the first, and under the mark**: 2.8
  ± 0.8 points fewer deals than **One pass** over the same updates. A
  seed moves a run of this length by about that much. What was fixed for
  this case holds: the first experiment's gain is not established as more
  than one run's, and a third seed is run before the long run is read for
  more than itself. It was run that night and
  [meets the mark](#the-third-seed-2026-10-11).
- **Their learning rate does better at this length** than either seed at
  ours: +3.1 ± 0.8 over **One pass**, which is about how far the two
  seeds are apart, and +5.9 ± 0.8 over the second seed. An update moves it
  a quarter as far (0.004 of KL against 0.016 and 0.019) and it leaves
  fewer of the pilot's answers the search kept (10.0% against 15.6% and
  14.1%). One run, and no mark.

**The long run**, **One pass** and its continuation as one run of 1,920
updates and 998,400 games:

| Updates | Games | Held-out deals | Over the pilot | Suite | Wins while sampling | Entropy |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 120 | 62,400 | 65.2% | +4.4 ± 1.3 | 64.7% | 61.6% | 0.11 |
| 240 | 124,800 | 66.0% | +5.3 ± 1.3 | 66.3% | 64.2% | 0.10 |
| 480 | 249,600 | 70.1% | +9.3 ± 1.4 | 69.7% | 67.1% | 0.10 |
| 720 | 374,400 | 71.7% | +11.0 ± 1.4 | 70.4% | 69.0% | 0.11 |
| 960 | 499,200 | 74.0% | +13.3 ± 1.4 | 73.7% | 71.2% | 0.15 |
| 1,200 | 624,000 | 73.9% | +13.2 ± 1.3 | 76.6% | 72.9% | 0.16 |
| 1,440 | 748,800 | 75.3% | +14.5 ± 1.4 | 72.7% | 73.4% | 0.18 |
| 1,680 | 873,600 | 76.4% | +15.7 ± 1.3 | 76.7% | 75.0% | 0.18 |
| 1,800 | 936,000 | 76.4% | +15.7 ± 1.4 | 76.9% | 75.3% | 0.19 |
| 1,920 | 998,400 | 77.6% | +16.9 ± 1.3 | 75.4% | 75.3% | 0.18 |

- **It climbs through its whole length**, more slowly at the end: its
  last three checkpoints win 76.8% of the held-out deals, 18.2 ± 1.0
  points more than its start and 16.1 ± 1.2 more than the pilot (60.7%).
  Of the four checkpoints after 1,560 updates the deals are 76.6%, 76.4%,
  76.4% and 77.6%, and the suite 76.8%, 76.7%, 76.9% and 75.4%.
- **It is past the pilot with its search** (72.5%) from 960 updates on. At
  the end the two win 974 deals both and lose 205 both; the model wins
  218 the search loses, and the search 139 the model loses.
- **Against every opponent** it wins more than its start, on the held-out
  deals and on the suite:

  | Opponent | Pilot | Pilot with its search | Start | After 120 | After 960 | After 1,920 |
  | --- | ---: | ---: | ---: | ---: | ---: | ---: |
  | Burn | 81.2% | 96.1% | 82.8% | 92.2% | 92.2% | 98.4% |
  | Crystal | 60.2% | 77.3% | 60.2% | 69.5% | 78.1% | 77.3% |
  | Dragunity | 57.8% | 71.1% | 56.2% | 66.4% | 71.1% | 80.5% |
  | Gishki | 65.6% | 82.0% | 65.6% | 62.5% | 79.7% | 89.1% |
  | Gladiator | 70.3% | 82.8% | 67.2% | 66.4% | 72.7% | 76.6% |
  | Heroes | 39.1% | 63.3% | 34.4% | 40.6% | 54.7% | 57.8% |
  | Infernity | 49.2% | 51.6% | 45.3% | 50.0% | 63.3% | 66.4% |
  | Lightsworn | 37.5% | 56.2% | 35.2% | 57.0% | 60.2% | 64.1% |
  | Monarch | 32.8% | 34.4% | 32.0% | 40.6% | 58.6% | 64.8% |
  | Morphtronic | 73.4% | 84.4% | 76.6% | 74.2% | 81.2% | 84.4% |
  | Rock-Block | 67.2% | 72.7% | 64.8% | 71.1% | 80.5% | 74.2% |
  | Spellcaster | 94.5% | 97.7% | 95.3% | 91.4% | 96.1% | 97.7% |

  128 held-out deals an opponent, so a row's last digit is not to be
  read. The search is still ahead against Gladiator and Heroes. In the
  suite's mirror it wins 75.0% of 128 games where its start won 50.0%. It
  wins 82.6% of the deals it opens and 72.7% of those it plays second
  (64.5% and 54.8% at its start).
- **How it plays.** In its own games its first choice is the pilot's at
  78.4% of its decisions, where the start's was at 95.3%. Where the
  search kept the pilot's answer it gives another at 19.9% of the
  decisions: 40.2% of the commands of a Main Phase, 21.5% of those of a
  Battle Phase, 7.3% of the chain windows. At the search's 2,357
  deviations it gives the search's answer at 18.9% and the pilot's at
  55.5% (2.0% and 89.5% at the start). Its games end a turn and a half
  sooner than the start's (11.6 turns against 13.0).
- **The search's measure does not see the gain.** By the worlds of the
  search, in which the pilot plays on after the one answer, the model's
  first choice where it leaves the pilot's is worth -0.6 ± 0.1 points an
  answer and -5.6 ± 1.0 a game, as at the start (-3.7) and after 120
  updates (-7.1). It wins 17 points more deals than the pilot all the
  same. What it gained is not single answers that pay with the pilot
  playing on after them: it is paid by the model's own play after them.
  Not looked into further.
- **The learner stays in order over the million games.** The value head
  explains 0.34 to 0.37 of the results of its own games throughout, and
  on the held-out games tells 79% of what can be told at the decisions
  the search examined. The entropy of the answers rises slowly, from 0.10
  to 0.18, the opposite of a policy closing up; an update moves the
  policy by 0.013 of KL in its first 120 updates and 0.007 in its last.

**What it says.** The long run answers its own questions: one pass does
not undo what it added, PPO against the pilots goes on adding for a
million games, and the checkpoint it ends in is well past its pilot and
past its pilot's search. Its last checkpoint
(`runs/ppo-long/one-pass-long/checkpoints/model-02279.pt`, 77.6% of the
held-out deals) is the strongest Blackwing player this project has made.
What is not shown is that another run would do the same: there is one long
run, and of the two seeds measured at 120 updates one passed the mark for
a gain and one did not. The third seed, read the next day, follows.

### The third seed (2026-10-11)

What the plan fixed for a second seed under its mark: **One pass** once
more from the same start, its games drawn with seed 2014. Run on the CUDA
machine in the night of 2026-10-10 (half an hour) and read on 2026-10-11
from `analysis/ppo-seed-2014/` (`report.txt`), as the second seed was and
by its mark.

| Reading | Measured | Mark | Met |
| --- | --- | --- | --- |
| The gain is not one run's luck | the third seed over its start: +3.1 ± 0.9 | 2 points and two standard errors | **yes** |

Held-out deals won, and over the last three checkpoints against the
start's four (58.6%; the pilot wins 60.7% of those deals):

| Run of **One pass** | 30 | 60 | 90 | 120 | Read | Over its start | Suite, last two |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| The first (seed 2012) | 59.1% | 60.2% | 63.3% | 65.2% | 62.9% | +4.3 ± 1.0 | 63.7% |
| The second (2013) | 60.2% | 57.9% | 60.5% | 61.8% | 60.1% | +1.5 ± 1.0 | 60.9% |
| The third (2014) | 59.9% | 60.0% | 61.1% | 63.8% | 61.7% | +3.1 ± 0.9 | 62.5% |

- **Three runs on other games each end above their start, and two of
  them past the mark.** The mark was fixed for one run; what three say
  together was not fixed, so they are told as they are. Over their nine
  checkpoints the three win 61.5% of the deals, 3.0 ± 0.8 points more
  than the start, and the first run's 4.3 is the highest of the three.
  After their 120 updates they are 4.4, 1.0 and 3.1 points over the pilot
  on the same deals (± 1.3 or 1.4 each), where the start's last
  checkpoint is 1.1 ± 0.9 under it.
- **A run of this length has a point or two of luck of its own.** The
  third seed is 1.2 ± 0.8 under the first and 1.6 ± 0.8 over the second.
  The three gains are spread by 1.4 points (their standard deviation),
  where the deals they are measured on would spread them by 0.6.
- **What that says of a run read alone.** The first experiment's arms
  were one run each. Four passes (11.2 points under the start) and credit
  that does not lean on the head (6.1 under) are far outside what a seed
  does. The differences of under two points between one pass, the head's
  lesson and the anchor are inside it: none of the three is shown to be
  better than another.
- **Their rate beside the three.** The run at 1e-4 gains 7.4 ± 0.9: 3.1,
  5.9 and 4.3 points more than the three seeds (each ± 0.8), where the
  lowest and the highest of them are 2.8 apart. It is one run at that
  rate, with no second seed of its own.
- **As a learner the third run is like the other two.** Its head explains
  0.38 of the results of its own games, an update moves it by 0.018 of KL
  (0.016 and 0.019), and where the search kept the pilot's answer it gives
  another at 13.6% of the decisions (15.6% and 14.1%).

**What it says.** The reading the second seed did not meet is met by the
third. That PPO in one pass adds to the copy in its first 120 updates is
not one run's luck: it did in three runs on other games, by one and a
half to four points, about three on the average. How much a run adds is
what a seed moves, so the first experiment's 4.3 points are to be read as
about three. What three seeds of 120 updates do not show is that another
long run climbs as the first did: there is one run of that length, and it
went on from the seed that had gained most. That is the first reading of
the second long run, below.

### The second long run (2026-10-11)

Decided on 2026-10-10 and made that night on the CUDA machine: **One pass,
their rate** (a learning rate of 1e-4, the deals of seed 2012) continued
from its 120 updates to 1,920, 998,400 games in all, as **One pass** was.
Its two readings were confirmed before the run and are fixed
([Conclusion and what follows](#conclusion-and-what-follows)). Read on
2026-10-11.

**How the folder is read.** The run was made with the script as it stood
one commit before it took a run's name (`NAME`): it went on from the right
checkpoint, the last of **One pass, their rate**, and its description
says so by the rate its optimizer came with (1e-4, which no option of the
script sets on a run that goes on). But it is called `one-pass-long`, in
`runs/ppo-long-rate/`, and the folder of its figures
(`analysis/ppo-long-rate/`) sets it after the first 120 updates of **One
pass**, the run at 3e-4: the first four rows of that folder's `report.txt`
and its figures over its checkpoint after 120 updates are of the wrong
run. The same files under the names the report expects are in
`analysis/ppo-long-rate-read/` on the development machine, with the report
made again (`tools/ppo_long_report.py ... --first one-pass-rate --beside
analysis/ppo-long`); its figures were also computed apart from the tool,
from the deals. The trainer is the same at both commits.

| Reading | Measured | Mark | Met |
| --- | --- | --- | --- |
| A long run climbs again | its last three checkpoints over the pilot: +20.9 ± 1.2; over its own checkpoint after 120 updates: +14.6 ± 1.1 | 2 points and two standard errors, each | yes |
| Their rate is better run long | over the first long run's checkpoints after the same updates, deal by deal: +4.8 ± 0.7 | 2 points and two standard errors | yes |

Held-out deals won by the two long runs after the same updates, and the
second over the first deal by deal:

| Updates | Games | At 3e-4 | At 1e-4 | Over it | Suite at 1e-4 | Entropy |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 120 | 62,400 | 65.2% | 67.0% | +1.8 ± 1.2 | 66.2% | 0.10 |
| 240 | 124,800 | 66.0% | 71.0% | +5.0 ± 1.2 | 68.4% | 0.12 |
| 480 | 249,600 | 70.1% | 74.9% | +4.9 ± 1.2 | 71.8% | 0.12 |
| 720 | 374,400 | 71.7% | 77.1% | +5.4 ± 1.2 | 75.1% | 0.13 |
| 960 | 499,200 | 74.0% | 78.0% | +4.0 ± 1.2 | 75.7% | 0.16 |
| 1,200 | 624,000 | 73.9% | 77.9% | +4.0 ± 1.2 | 76.3% | 0.16 |
| 1,440 | 748,800 | 75.3% | 78.6% | +3.4 ± 1.1 | 78.0% | 0.17 |
| 1,680 | 873,600 | 76.4% | 81.5% | +5.1 ± 1.1 | 78.8% | 0.17 |
| 1,800 | 936,000 | 76.4% | 82.0% | +5.5 ± 1.1 | 79.4% | 0.21 |
| 1,920 | 998,400 | 77.6% | 81.4% | +3.8 ± 1.1 | 79.4% | 0.20 |

- **It climbs through its whole length too**, and is ahead of the first
  long run at every one of the nineteen points both were measured at, by
  1.8 to 6.6 points. Its last three checkpoints win 81.6% of the held-out
  deals: 23.0 ± 1.0 points more than its start, 20.9 ± 1.2 more than the
  pilot (60.7%) and 9.2 ± 1.1 more than the pilot with its search (72.5%),
  which it is past from 480 updates on.
- **Against every opponent** its last checkpoint wins more than its start,
  and against none clearly fewer than the first long run's:

  | Opponent | Pilot | Pilot with its search | Start | At 3e-4 | At 1e-4 |
  | --- | ---: | ---: | ---: | ---: | ---: |
  | Burn | 81.2% | 96.1% | 82.8% | 98.4% | 96.9% |
  | Crystal | 60.2% | 77.3% | 60.2% | 77.3% | 89.1% |
  | Dragunity | 57.8% | 71.1% | 56.2% | 80.5% | 80.5% |
  | Gishki | 65.6% | 82.0% | 65.6% | 89.1% | 86.7% |
  | Gladiator | 70.3% | 82.8% | 67.2% | 76.6% | 85.9% |
  | Heroes | 39.1% | 63.3% | 34.4% | 57.8% | 63.3% |
  | Infernity | 49.2% | 51.6% | 45.3% | 66.4% | 65.6% |
  | Lightsworn | 37.5% | 56.2% | 35.2% | 64.1% | 67.2% |
  | Monarch | 32.8% | 34.4% | 32.0% | 64.8% | 73.4% |
  | Morphtronic | 73.4% | 84.4% | 76.6% | 84.4% | 89.8% |
  | Rock-Block | 67.2% | 72.7% | 64.8% | 74.2% | 82.0% |
  | Spellcaster | 94.5% | 97.7% | 95.3% | 97.7% | 96.1% |

  128 held-out deals an opponent. The search is level against Heroes, two
  deals ahead against Spellcaster and behind everywhere else; against
  Gladiator, where it was ahead of the first long run, it is four deals
  behind. In the suite's mirror the checkpoint wins 80.5% of 128 games
  (75.0% at 3e-4). It wins 85.8% of the deals it opens and 77.0% of those
  it plays second.
- **It wins more and stays nearer its pilot.** Where the search kept the
  pilot's answer it gives another at 15.8% of the decisions, the first
  long run at 19.9%: 32.5% of the commands of a Main Phase (40.2%), 18.1%
  of those of a Battle Phase (21.5%), 4.6% of the chain windows (7.3%).
  In its own games its first choice is the pilot's at 79.9% of its
  decisions (78.4%). At the search's 2,357 deviations it gives the
  search's answer at 18.6% and the pilot's at 60.2% (18.9% and 55.5%).
  Its games are half a turn shorter again (11.1 turns, 11.6, the start
  13.0).
- **The learner stays in order.** An update moves the policy by 0.004 of
  KL throughout, a third to a half of what it moves at 3e-4 (0.013
  falling to 0.007). The value head explains 0.33 to 0.37 of the results
  of its own games, and the entropy of the answers rises from 0.10 to
  0.20 as in the first run.
- The 1,800 further updates took 7.3 hours, 128,000 games an hour with
  the suites.

**What it says.** Both readings are met. A second long run on other games
climbs as the first did, so the first was not one run's luck in that; and
the rate that did better over 120 updates does better over a million
games, by 4.8 ± 0.7 points of the held-out deals at the end. It is one
run at each rate, both on the deals of seed 2012, and how far two long
runs at one rate lie apart is not measured: the error is that of the
deals. The learning rate is the first of the differences from ygo-agent's
trainer tried here, and it is worth those points. The run's last
checkpoint, `runs/ppo-long-rate/one-pass-long/checkpoints/model-02279.pt`
on the CUDA machine (81.4% of the held-out deals; `model-02159.pt`, after
1,800 updates, 82.0%), is now the strongest Blackwing player made here.
It has the file name of the first long run's last checkpoint, in another
folder; on the development machine it is `runs/model-02279-rate.pt`. What
it wins against decks outside training and what it does with the named
plays are told below
([decks outside training](#beside-the-follow-up-decks-outside-training-2026-10-10),
[duels in a room](#beside-the-follow-up-duels-in-a-room-2026-10-10)).

### Beside the follow-up: decks outside training (2026-10-10)

Asked while the long run was at 960 of its 1,920 updates: is what PPO
added a way of beating the thirteen pilots it trains against, or of
playing its deck? `eval_suites/ygo-policies-blackwing-unseen.json` is 128
games against the pilot of each of the eighteen decks of ygo-policies
outside those thirteen, 2,304 games on the deals of seed 2012. Two
checkpoints were copied to the development machine and played there, on
its processor: the start of every run here (**Once, few** after 360
updates) and **One pass, long** after 960 updates, **a checkpoint of a run
that had not ended**. The figures are in `analysis/unseen/` there.

| Games against | Pilot | The start | After 960 updates | Over the start | Over the pilot |
| --- | ---: | ---: | ---: | ---: | ---: |
| The thirteen decks of training (the suite, 1,664) | 60.3% | 57.9% | 73.7% | +15.8 | +13.4 |
| The eighteen other decks (2,304) | 63.8% | 58.6% | 68.8% | +10.2 ± 1.4 | +4.9 ± 1.4 |

- **Two thirds of what PPO added is there against decks it never met**:
  10.2 points of the 15.8. It wins more than the start against fifteen of
  the eighteen, most against Machina (20.3% to 43.8%, the pilot 21.9%),
  Harpie (57.8% to 76.6%) and Tele-DAD (51.6% to 69.5%, the pilot 52.3%),
  and more than the pilot against fifteen.
- **The start is further under its pilot against these decks** (5.2
  points, where it is 2.4 under against the decks of training): most of
  their cards it was never shown.
- **Against Ojama the networks stay far under the pilot**: the pilot wins
  95.3%, the start 80.5%, the later checkpoint 76.6%. Nobody wins against
  Countdown (3, 1 and 0 games of 128).
- The errors are those of the games, and the three are set beside each
  other as rates: `evaluate-suite` keeps no result by deal, though the
  deals are the same. A deck is 128 games.
- Played on the processor, the suite of training has the wins of the CUDA
  machine's card in each of the thirteen matchups (1,226 of 1,664).

**With the long run ended**, the same suite was played on the CUDA
machine by its checkpoints after 480, 960, 1,440, 1,680, 1,800 and 1,920
updates, and again on the development machine by the pilot, the start and
the checkpoint after 960, each measurement now keeping every game's
result (`analysis/unseen/ppo-long.json` and `analysis/unseen/paired/`;
`tools/suite_beside.py` sets them beside each other). The checkpoint
after 960 updates wins the same games on the card as on the processor,
all 2,304 of them. The long run is read by its last three checkpoints, as
on the held-out deals, and the differences are game by game.

| Games against | Pilot | The start | After 480 | 960 | 1,440 | 1,680 | 1,800 | 1,920 | Last three | Over the start | Over the pilot |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| The thirteen decks of training (the suite, 1,664) | 60.3% | 57.9% | 69.7% | 73.7% | 72.7% | 76.7% | 76.9% | 75.4% | 76.3% | +18.4 | +16.0 |
| The eighteen other decks (2,304) | 63.8% | 58.6% | 65.8% | 68.8% | 68.8% | 70.3% | 71.3% | 69.6% | 70.4% | +11.8 ± 0.9 | +6.6 ± 0.8 |

- **Two thirds of what a million games added carries to decks it never
  met**, as at the halfway point: 11.8 points of 18.4. Against them it
  goes on gaining in the second half of the run too, 1.6 ± 0.7 over its
  checkpoint after 960 updates, where the suite of training gains 2.6.
- **It wins more than its start against thirteen of the eighteen** by
  more than two standard errors, and fewer against none: most against
  Tele-DAD (51.6% to 75.3%), Karakuri (62.5% to 81.8%), Gravekeeper (39.1%
  to 57.8%), Draconic Might (44.5% to 63.0%), Harpie (57.8% to 75.8%) and
  Machina (20.3% to 37.8%, the pilot 21.9%). It is past the pilot against
  ten of them by that margin.
- **Ojama is the one deck it stays under its pilot against** (82.0% where
  the pilot wins 95.3%, 13.3 ± 2.7 under), and PPO did not move it: the
  start wins 80.5% there. Against Six Samurai, Destiny HERO and Countdown
  it is where its start and its pilot are; against Toon 4.2 ± 2.4 under
  the pilot.

**The second long run (2026-10-11)**, the one at a learning rate of 1e-4,
played the same suite on the CUDA machine by its checkpoints after the
same updates (`analysis/unseen/ppo-long-rate.json`: the same deals, and the
same revisions of the engine, the scripts and the card data). The suite
of training is the one each run plays as it goes.

| Games against | Pilot | The start | After 480 | 960 | 1,440 | 1,680 | 1,800 | 1,920 | Last three | Over the start | Over the pilot | Over the first long run |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| The thirteen decks of training (the suite, 1,664) | 60.3% | 57.9% | 71.8% | 75.7% | 78.0% | 78.8% | 79.4% | 79.4% | 79.2% | +21.3 | +18.9 | +2.9 |
| The eighteen other decks (2,304) | 63.8% | 58.6% | 71.7% | 74.0% | 74.2% | 74.0% | 73.8% | 73.3% | 73.7% | +15.0 ± 0.9 | +9.8 ± 0.8 | +3.3 ± 0.5 |

- **What the rate of 1e-4 added is there against decks it never met
  too**: its last three checkpoints win 3.3 ± 0.5 points more of those
  games than the first long run's, and it is ahead at each of the six
  points, by 2.5 to 6.0. Seven tenths of what this run added to its start
  carries over, 15.0 points of 21.3.
- **Against these decks it stops gaining halfway.** Its last three
  checkpoints are where its checkpoint after 960 updates is (0.3 ± 0.7
  under), while against the decks of training it gains 3.5 points more
  over that half, and 3.6 ± 0.9 on the held-out deals. The first long run
  gained 1.6 ± 0.7 against them over its second half. One run each: the
  two are not told apart in this by more than their errors.
- **It wins more than its start against fifteen of the eighteen** by more
  than two standard errors, and more than the pilot against thirteen:
  most over the pilot against Tele-DAD (79.2%, the pilot 52.3%), Machina
  (39.6%, the pilot 21.9%), Gravekeeper (64.1%, 46.9%) and X-Saber (76.3%,
  60.2%). Over the first long run it gains most against X-Saber (65.6% to
  76.3%), and it wins 24 of its 384 games against Countdown, where the
  first long run won two.
- **Ojama is still the one deck it stays under its pilot against**: 86.2%
  where the pilot wins 95.3%, 9.1 ± 2.5 under.

### Beside the follow-up: head to head (2026-10-11)

The trainer seats a checkpoint in the other seat since that day
([docs/commands.md](../docs/commands.md), its section on who plays the
other seat). It is built for what comes next
([Conclusion and what follows](#conclusion-and-what-follows)) and was
first used on the checkpoints at hand:
`eval_suites/blackwing-against-checkpoint.json` is 2,048 deals of the
mirror, the measured seat opening every other one, each network taking its
first choice at its own decisions. Played on the development machine
(`analysis/head-to-head/` there). No reading hangs on it.

| Measured | Against the second long run's last checkpoint | Against the first long run's last |
| --- | ---: | ---: |
| The deck's pilot | 21.4% ± 0.9 | 25.1% ± 1.0 |
| The copy PPO started from | 18.2% ± 0.9 | 22.5% ± 0.9 |
| The first long run after 960 updates | 42.6% ± 1.1 | 48.4% ± 1.1 |
| The first long run's last checkpoint | 44.0% ± 1.1 | |
| The second long run's last checkpoint | | 52.6% ± 1.1 |

- **The second long run's last checkpoint beats the first's**: with each
  in turn the one held, every deal is played from either seat, and it
  wins 54.3% ± 0.7 of the 4,096 games. It wins 480 deals from both seats
  and the other 304; in 1,264 the same seat wins whoever sits in it.
- **The measure tells apart what the pilots tell apart, and more widely.**
  The copy is 23 points of the held-out deals under the strongest
  checkpoint and wins 18% of the mirror against it; the first long run's
  last checkpoint is 4 points under and wins 44% to 47%.
- **More PPO against the pilots is worth less here than against them.**
  The second half of the first long run added 3.6 points of the held-out
  deals (74.0% to 77.6%). In the mirror its last checkpoint is not shown
  to be ahead of its checkpoint after 960 updates: that one wins 48.4% ±
  1.1 against it, and against the second long run's last checkpoint the
  two do alike (1.4 ± 1.4 apart, game by game).
- The seat that opens wins more, whoever it is: the second long run's
  checkpoint wins 59.9% of the deals it opens against the first's and
  45.4% of those it plays second.
- The copy is under its pilot here as it is against the pilots (3.2 ± 0.8
  and 2.6 ± 0.8 under, game by game).

### Beside the follow-up: duels in a room (2026-10-10)

The same checkpoint, **One pass, long** after 960 updates, was seated in a
room of EDOPro (`edopro`, [docs/commands.md](../docs/commands.md)) and
played a duel there, the other seat taking the Blackwing deck too. It won
in seven turns. That is no rate: one duel, against play unlike any of the
pilots'. The duel is in `edopro-logs/1791651168-blackwing-model-01319.log`
on the development machine: of 114 decisions of its own the model gave its
pilot's answer at 103, and among the others it kept Monster Reborn for a
turn and took the other player's Blackwing Armed Wing with it.

That matchup against the deck's own pilot, for what a rate says of it:
2,048 games of the mirror on the deals of seed 2012, played on the
development machine. The pilot wins 51.0% ± 1.1 of them from the seat a
model takes, the start 47.9% ± 1.1 and the checkpoint 69.6% ± 1.0: 76.5%
of the games it opens and 62.7% of those it plays second, as it sat in
the room. The suite's 128 games of the mirror had said 74.2%.

The duel was then played again in the trainer's environment from the
replay EDOPro itself kept of it (the seed, both decks and every answer of
both players: all 313 answers are taken by the engine as they were given),
with the deck's pilot and the checkpoint asked at every decision of either
seat. `analysis/edopro/` on the development machine holds it.

- **The seat in the room answered as the checkpoint answers in the
  trainer**: at each of its 114 decisions the answer played is the
  checkpoint's first choice there. That is the seat's fidelity on a duel
  of a real room, where the test has it on a scripted one.
- **The other seat's own moves beside the pilot's**: of its 51 decisions
  the pilot gives the same answer at 34 and the checkpoint at 38, cards
  alike counted as one answer. At nine both would answer otherwise and
  name the same answer: Pot of Duality on the first turn, Gale for Black
  Whirlwind's search, Main Phase 2 before a turn is ended. The
  checkpoint's value head, asked from the other seat, has that seat ahead
  until the fifth turn (0.71 of a win in the first turns, 0.86 in the
  fourth, 0.56 in the fifth, 0.09 in the seventh).

Three more duels followed that evening against the same checkpoint: two
more mirrors and one in which the other seat played the Monarch deck. The
checkpoint went first in each and won all three, so the four duels played.
Each was read again as the first was, from EDOPro's own replay.

| The other seat's deck | Turns | The bot's decisions | the checkpoint's first choice in the trainer | its pilot's | The other seat's decisions | the pilot's answer | the checkpoint's |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Blackwing | 13 | 106 | 106 | 86 | 47 | 33 | 32 |
| Blackwing | 15 | 135 | 135 | 117 | 68 | 48 | 49 |
| Monarch | 31 | 392 | 391 | 324 | 76 | 50 | not asked |

- Against Monarch the checkpoint rated itself behind for the first third
  of the duel (its value head from its own seat: 0.51 of a win at the
  first turn, 0.10 at the twelfth), ahead from the sixteenth (0.84 at the
  twenty-second), and won at the thirty-first with 3,400 Life Points.
- The other seat's answers leave its pilot's mostly at the commands of a
  Main Phase (the pilot's answer at 37 of 72 over the three duels, at 50
  of 58 chain windows); four of the differences in each duel are only the
  phase a turn is ended through.

**A room shows the trainer's decision only where its card scripts are the
trainer's.** EDOPro plays its own card scripts and card database, and
those of this installation are of April 2025, the trainer's of October
2026. Every one of the four duels plays again in the trainer, answer for
answer, so the games are the same; what differs is what some answers are
called. The trainer's scripts give the activations of Mystical Space
Typhoon and Pot of Duality a description and EDOPro's give them none, and
the model's input says of every answer whether it names an effect. Each
bot's own trace was fed to the seat again and each decision set beside the
trainer's:

- the first duel's 114 decisions are the same in every field;
- the other three differ in the description of 36, 80 and 411 answers (and
  once in a selection's hint), which moves the checkpoint's probabilities
  by more than a point at 1, 1 and 4 decisions;
- its first choice changes at one decision of the 747: whether to activate
  Mystical Space Typhoon in an End Phase, 0.52 to 0.48 in the trainer and
  0.48 to 0.52 in the room, where it passed.

The test holds the seat to an episode on a room that is scripted from the
trainer's own engine, and cannot see this. In a room of EDOPro the seat is
as faithful as the room's scripts are the trainer's.

**The Monarch list in a room.** The pool's list names Gorz the Emissary of
Darkness by an alternate-artwork code (44330100, an alias of 44330098)
that this EDOPro's database does not hold, so the list loads there with 39
cards. In the trainer it has its forty: the engine scripts an alternate
code by its alias, and the Monarch pilot reads it by its alias. The place
was filled with a Solemn Judgment from the Side Deck, so the bot, told the
pool's list, faced one card it was not told of; asked with either list,
the checkpoint gives the decision above the same probabilities to the
fourth decimal. Lightsworn's list has the same with Honest (37742479 for
37742478); no other deck of the pool names a card this EDOPro lacks.

**The room's files after EDOPro updated itself (2026-10-10, late).** This
EDOPro had never fetched its repository of updates, which brings an
engine, the scripts and the card rows together: the machine's git
configuration rewrites `https://github.com/` to SSH, EDOPro reads that
configuration too and has no key to offer (`error.log`: authentication
required). With the rewrite unset it fetched the repository (its commit
`490a5387` of that day). Read as EDOPro reads them, the repository's files
before the ones it came with, and set beside the trainer's pins for the
580 cards of the pool's 31 lists:

| | As it came (April 2025) | Fetched on 2026-10-10 |
| --- | ---: | ---: |
| Cards the database lacks | 2 | 0 |
| Cards told otherwise (a set code) | 4 | 0 |
| Card scripts that are another, of 563 | 43 | 0 |
| Helper scripts that are another, of 25 | 20 | 1 |
| Pilot games that are the trainer's, of 240 | 16 | 240 |

The games are Blackwing's pilot against the pilot of each of the other 30
decks, eight a deck, played by the policies' bench on EDOPro's engine,
scripts and card rows and on the trainer's: a game is the trainer's when
it asks the same questions and takes the same answers to its end (106,113
decisions in all). As it came, 208 games are another and 16 fail, those
against the two lists that name a card it lacks; the 43 scripts are more
than the two the duels above had shown. The one helper that still differs
is five days ahead of the trainer's pin, in a function that places a Field
Spell by an effect, which no script of the pool calls. So the room now
plays the trainer's duels, and both lists load with their forty cards.

`tools/edopro_files.py EDOPRO` is that comparison. EDOPro follows the
scripts as they change every time it starts and the trainer's pin does
not, so it is run again before a session. It tells of the files; a duel
played in the updated room and read with `edopro-review --trace` is still
owed.

**The other seat's Blackwing list in the three mirrors** was the copy in
its EDOPro's deck folder, of July, and not the pool's: 35 of its 40 Main
Deck cards are the pool's (two more Kalut, a Book of Moon, an Icarus
Attack and a Cold Wave, where the pool's list has Jin, Zephyros, a
Mystical Space Typhoon, a Dimensional Prison and a Bottomless Trap Hole).
The reviews said so (88% of the Main Deck) and it was not raised when they
were read; it was found again when the deck folder was looked at. The bot
played the pool's list. With which list it was told the other seat played
is not kept; each duel's trace was fed to the seat again with the
checkpoint told the pool's list and told the list played: its first choice
is the same at all 355 decisions, and a probability moves by a point at
two. `--opponent` takes a `.ydk` file for a list that is not the pool's.

**In a web page (2026-10-11).** A web client (`beat-claudio-yu-gi-oh`, a
repository of its own) seats a checkpoint since that day: the seat and the
input are `native/ygo-model-input`'s, around them its own build of the
engine and its own room, and the network is computed by a forward pass
written again in Rust from the weights `tools/export_checkpoint.py`
writes. Checked with the long run's last checkpoint, from the development
machine:

- **The forward pass** gives the scores of the trainer's network for the
  130 decisions the export answers, within 0.00005 where a score reaches
  200: for that checkpoint, for the one after 960 updates, and for a
  checkpoint whose answers attend to the cards on the field (a small
  trial's, in which the attention moves a score by up to 4.3).
- **Whole duels.** 32 duels were played there and each was played again
  here with the checkpoint in the trainer's own seat, answering with its
  first choice, against the other player's answers as the page kept them:
  8 by the page's engine as a program, against the pilots of eight decks,
  and 24 through the WebAssembly module the page loads, with the card data
  and scripts it is given, against a player who only passes, 6 of them
  with the page reloaded every five questions. All 4,956 answers of the
  network's seat are the same byte for byte, 2,825 of them decisions of
  the network, and the other player's answers fit to the last one kept. So
  the page's engine (OCGCore as its authors publish it, where the trainer
  builds the policies' fork), its scripts and its card data play the
  trainer's duels, and its seat shows the network what the trainer shows.
  What the room sent the seat in the first 8 duels was also fed to the
  trainer's seat: its 834 decisions are the trainer's in every field.
  `tools/web_client_duels.py CHECKPOINT SAVED...` is that comparison, for
  any duel the page saves ([docs/commands.md](../docs/commands.md)).
- **What had to be set alike**: the seed of the pilot asked alongside,
  which places the cards and breaks its ties by that seed. With another
  seed a card goes to another zone and nothing else differs; zones are no
  part of the network's input. A duel is kept there up to the last
  question its player was asked, and is compared that far.
- **Against a player who does nothing**, the checkpoint wins slowly. The
  player of the 24 duels passes every chance and ends every turn. Of the
  18 without reloads the checkpoint wins all, in 12 turns on the average
  and up to 25; the Blackwing pilot in its place wins 12 of 12 in 7 (5 to
  11). In one mirror it sets a monster and its traps, then summons nothing
  for six turns of its own, entering the Battle Phase and leaving it each
  time, where its pilot would summon; it rates its own chances at 78%
  rising to 98% over those turns. No opponent of training does nothing,
  and a win counts the same however long it takes. Not looked into
  further.

**Duels on the page (2026-10-11).** The long run's last checkpoint was
played on that page, first against a Machina Gadget deck and then in the
mirror, and every game was reported won against it (corrected on
2026-10-12: how many duels were played is not known. That every game was
won was read from the report of them; a screenshot of the page shows one
duel won and one lost against a loaded network, and three won against
Claudi-oh's own decks. The page keeps finished duels only, and one record
for whatever network is loaded). The checkpoint's play was judged very
poor, with a number of plain mistakes. Three were named: it activated
Allure of Darkness with no DARK monster in hand as the first action of a
turn and lost a hand of Spells and Traps it could have set first; it uses
a card at the first chance, Trap Hole on a monster that threatens nothing;
and it activated Threatening Roar when the other seat entered the Battle
Phase with no monster. It was added that the list has no answer to Machina
Fortress but Dark Hole and Icarus Attack.

The page plays the checkpoint's own answers (above), so the three plays
were looked for in the trainer: 2,240 games on the development machine,
160 against the pilot of each of the thirteen decks of training and of
Machina, with every decision of the checkpoint's seat kept beside its
pilot's answer. The same games were played by the copy PPO started from
and by the checkpoint after 960 updates and 500,000 games, which is the
one played in the four duels above. The last checkpoint wins 74% of its
games and gives its pilot's answer at 80% of its decisions; the one after
960 updates 70% and 79%; the copy 54% and 97%. The last column is the
second long run's last checkpoint, told under the bullets.

| | The pilot | The copy PPO started from | After 960 updates | After 1,920 | The second long run, after 1,920 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Turns with Allure of Darkness and a DARK monster in hand: activated | | 641 of 675 | 513 of 755 | 538 of 565 | 562 of 631 |
| Turns with nothing in hand but it: activated | never | 2 of 74 | 29 of 32 | 63 of 72 | 31 of 41 |
| Turns with only Spells and Traps beside it: activated | never | 3 of 112 | 71 of 83 | 16 of 20 | 38 of 57 |
| ... with cards still to set | | 3 | 67 | 14 | 35 |
| Trap Hole at a Summon that allows it | 92% | 93% | 93% | 78% | 74% |
| ... of a monster with 1,000 to 1,499 ATK | 95% | 91% | 94% | 62% | 57% |
| Threatening Roar at their Battle Phase, monsters of theirs in Attack Position | 39% | 35% | 2 of 6,801 | 9 of 4,807 | 456 of 6,402 |
| Threatening Roar in their Main Phase 1 | 15 of 10,646 windows | 16 of 9,386 | 469 of 11,858 | 720 of 10,646 | 144 of 12,465 |
| ... spent before they had an attacker, in answer to nothing | never | 0 | 1 | 24 | 0 |

- **Allure of Darkness is PPO's own, and it is two plays.** The pilot
  activates it only with a DARK monster in hand, and so does its copy.
  With nothing else in hand both checkpoints activate it, which risks
  nothing and which the pilot never does. With only Spells and Traps
  beside it they activate it too, and nearly always before those are set:
  in 14 of 16 such turns at the end, in 67 of 71 after 960 updates. That
  is the first of the plays named above. Of the last checkpoint's 101
  activations with no DARK monster in hand 16 sent the hand to the
  Graveyard, one of them a hand with other cards in it; after 960 updates
  17 of 134, seven of them with other cards. It comes up four times as
  often after 960 updates (83 turns in the 2,240 games against 20),
  because that checkpoint keeps the card in a third of the turns it has a
  DARK monster for it.
- **Trap Hole at the first chance is the pilot's rule**, which answers
  every Summon the card allows, and after 960 updates the checkpoint still
  plays it so (882 of 894 at the first window of a game). In the second
  half of the run PPO made it less so, most for the weakest monsters. The
  same holds at the end for Bottomless Trap Hole (68% where the pilot 78%)
  and Dimensional Prison (55% and 71%).
- **Threatening Roar was moved by PPO from their Battle Phase to their
  Main Phase 1**, and had been after 960 updates (469 of its 497
  activations). Of the last checkpoint's 785 activations 720 are there,
  266 of them right after a card of theirs came to the field and 263 in
  answer to a card of theirs, and 53 are in their Draw Phase. 24 are at a
  moment they had no monster to attack with and nothing of theirs was
  answered, all but two of them close to an even choice for it (0.51 to
  0.60), and 22 of them against Burn or Spellcaster; after 960 updates
  there is one such. At a Battle Phase they entered with no monster the
  last checkpoint activated the card at 1 of 260 windows, and at none of
  894 against a player who does nothing else (none of 624 after 960
  updates): the play as it was named is rare here. A Threatening Roar
  spent before they have a monster to attack with comes once in about
  ninety games at the end.
- **Machina** it loses to as well: 29% of 160 games against that deck's
  pilot (34% after 960 updates, the copy 21%).

**The second long run's last checkpoint** played the same 2,240 games on
2026-10-11, once it had been copied to the development machine
(`runs/model-02279-rate.pt` there: its last update is that run's, by the
fingerprint of the update's games). It is the strongest player made here.
It wins 77% of the games and gives its pilot's answer at 81% of its
decisions.

- **Allure of Darkness before its cards are set it has too.** With only
  Spells and Traps beside it, it activates the card in 38 of 57 turns, in
  35 of them with cards still to set. Of its 70 activations with no DARK
  monster in hand six sent the hand to the Graveyard, three of them a
  hand with other cards in it.
- **Trap Hole it plays as the first long run's last checkpoint does**: at
  74% of the Summons that allow it, where its pilot answers 89% of those,
  at 57% for a monster of 1,000 to 1,499 ATK, and at 728 of 876 first
  windows of a game. Bottomless Trap Hole 68% (the pilot 77%), Dimensional
  Prison 74% (72%).
- **Threatening Roar it did not move.** 456 of its 601 activations are at
  the start of their Battle Phase with monsters of theirs in Attack
  Position, where the pilot plays the card, though at 7% of those windows
  where the pilot takes 42%. 144 are in their Main Phase 1, and 143 of
  those answer a card of theirs on the chain, 63 times a Mystical Space
  Typhoon. None is in their Draw Phase, none is spent before they had an
  attacker in answer to nothing, and none comes at a Battle Phase they
  entered with no monster (357 windows, and 621 against a player who does
  nothing). So showing the card early was the first long run's own path,
  and no habit of PPO.
- **Machina** it wins 42% of the 160 games against, and 39.6% of the 384
  of the suite of other decks
  ([above](#beside-the-follow-up-decks-outside-training-2026-10-10)),
  where the first long run's last checkpoints win 37.8%.
- **Against a player who does nothing** (120 games, half in the mirror
  and half against the Monarch deck) it wins all in 10.6 turns on the
  average; in the same games the first long run's last checkpoint takes
  14.0 and its checkpoint after 960 updates 11.9.

**What it says.** The three plays are the checkpoint's: one a habit PPO
found and carries too far, one its pilot's rule, one a timing PPO chose.
None costs it anything that shows against pilots, which do not look for
them: a pilot plays on the same whether Threatening Roar is shown before
its attack or at it, and Allure of Darkness costs it a hand in about one
game in a thousand. So more games against the pilots give PPO no reason to
mend them, and the rate against pilots (77.6% of the held-out deals) does
not say how the checkpoint plays an opponent that looks for such things.
The checkpoint after 960 updates is no other player in this: it has the
three plays, Trap Hole as the pilot plays it and Allure of Darkness before
its cards are set four times as often, and only Threatening Roar spent
early is the later checkpoint's. The second long run's last checkpoint has
two of the three: Allure of Darkness before its cards are set, which both
runs came to, and Trap Hole at most first chances, which both loosened
alike from the pilot's rule. Threatening Roar it plays at their attack.
Winning 4.8 points more of the held-out deals has not mended the two.

## Conclusion and what follows

**What it answered.** PPO from a copy of the pilot adds to the copy when
an update makes one pass over its games, and undoes it in four. The value
head predicted either way, so the inference the experiment began from did
not hold: it was the number of passes. Run long, one pass goes on adding
for a million games and is past the pilot with its search; at a learning
rate of 1e-4 it ends higher, at 81.6% of the held-out deals where the pilot
wins 60.7%, and seven tenths of what it added carries to decks it never
met. A run of 120 updates has a point or two of luck of its own. What the
rate against pilots does not see was found in duels against the
checkpoints: their play was judged poor, and plays were named that are
theirs and cost nothing against a pilot (how many of those duels were won
is not known: the correction under the duels on the page).

**Closed on 2026-10-11**, with every reading here taken, and with these
decisions of that day:

- **PPO runs at a learning rate of 1e-4 from here on.** Its scripts name
  it; the trainer's default stays 3e-4, which the imitating runs are made
  with.
- **The copy of the specialist is put on hold** (it is prepared, below,
  with its readings fixed).
- **What comes next is the named misplays**
  ([above](#beside-the-follow-up-duels-in-a-room-2026-10-10)), which the
  rate against pilots does not see. Self-play is what is tried first:
  [Self-play, and the named plays](2026-10-11-self-play.md).
- **The order of it, agreed that day.** First a checkpoint can be seated
  in the other seat, and the table of the three plays becomes a command.
  Then the strongest checkpoint is measured with what the rate against
  pilots does not see, as the base: head to head against other
  checkpoints, and by an exploiter, a learner trained for a fixed budget
  against the checkpoint held as it is. Then self-play, with its readings
  fixed before any run of it.
- **Who sits at each seat in that run, decided that day:** the latest
  network at both, every decision of a game being the learner's, with part
  of each round still against the pilots. An earlier checkpoint held at
  one seat is what head to head and the exploiter need, and becomes an
  opponent of training only if a later checkpoint is found to lose to an
  earlier one. ygo-agent's trainer seats the latest network at both
  (`scripts/cleanba.py`: one set of weights answers both seats and every
  step is trained on).

**Prepared on 2026-10-11**, built and tried on the development machine;
nothing of it is run on the CUDA machine yet:

- The trainer seats the learner itself (`--self-play-duels`) or a
  checkpoint held as it is (`--opponent-checkpoint`) in the other seat, in
  shares of an update's duels, and a suite's case can be played against
  the held checkpoint ([docs/commands.md](../docs/commands.md)). A run
  without these options plays the games it played.
- `tools/named_plays.py` is the table of the named plays, for any
  checkpoints beside each other. It gives every figure of the table
  [above](#beside-the-follow-up-duels-in-a-room-2026-10-10), which was
  counted by hand.
- The base, `experiments/commands/self-play.sh`: the second long run's
  last checkpoint measured with what the rate against pilots does not
  see, before any arm of self-play is. Head to head, its run's earlier
  checkpoints, itself and the deck's pilot against it held in the other
  seat; and two exploiters, learners that begin as it and play every duel
  of 240 updates against it. No reading hangs on the base. About three
  hours there by estimate, not by measurement.

The arms of self-play and their readings were confirmed that day and are
fixed in that experiment's record.

The third seed is read ([above](#the-third-seed-2026-10-11)); it was made
with

```sh
PARTS=short SHORT=one-pass-seed SEED=2014 RUNS=runs/ppo-seed-2014 \
  OUT=analysis/ppo-seed-2014 bash experiments/commands/ppo-long.sh
```

**The second long run** is read too
([above](#the-second-long-run-2026-10-11)), and both its readings are met:
nothing more is owed in this record. It was decided on 2026-10-10 when the
follow-up's result was read, for that night and before anything else was
started: **One pass, their rate** continued from its 120 updates to 1,920,
as **One pass** was, and set beside it. Its command:

```sh
PARTS=long NAME=one-pass-rate FIRST=runs/ppo-long/one-pass-rate \
  FIRST_OUT=analysis/ppo-long BESIDE=analysis/ppo-long \
  RUNS=runs/ppo-long-rate OUT=analysis/ppo-long-rate \
  bash experiments/commands/ppo-long.sh
```

On the CUDA machine the run it made is in
`runs/ppo-long-rate/one-pass-long`, from before the script took `NAME`.
Given again there with the script as it is now, this command looks for
`runs/ppo-long-rate/one-pass-rate-long`, does not find it, and trains the
run anew.

It asked two things with one night: whether a long run climbs again on
other games than the first (the two play other games from their second
update on), and whether the rate that did better over 120 updates does
better run long. If it ends lower than the first, its rate and its own
luck are not told apart, and the run of the second seed at the first rate
is what tells them (`NAME=one-pass-seed`, with
`OPTIONS="--training-seed 2013"`). Its readings, proposed and confirmed on
2026-10-10 before the run and fixed, on its checkpoints after 1,680, 1,800
and 1,920 updates:

1. *A long run climbs again*: they win at least 2 points and two standard
   errors more of the held-out deals than the pilot, and than this run's
   own checkpoint after 120 updates. These are the first long run's marks.
2. *Their rate is better run long*: they win at least 2 points and two
   standard errors more of the deals than the first long run's checkpoints
   after the same updates, deal by deal. At least as far under, the rate
   is worse run long; between the two, no difference is shown.

What was to come after it was decided on 2026-10-10 as an experiment of
its own, and is on hold since 2026-10-11: whether what the long run's last
checkpoint has over its pilot is taken into a new network by imitating the
checkpoint, in a fraction of the games it took to learn. Its readings stay
fixed for when it is taken up. It bears on how a network for several decks
is to be made (from the pilots of several decks, or from a specialist of
each) and on starting a network of another shape from this one's play. The
trainer imitates a checkpoint since that day (`--imitation-teacher`), and
the run is prepared: `experiments/commands/specialist-copy.sh`, with
`tools/specialist_copy_report.py`. Its teacher was named when there was
one long run: that run's last checkpoint (77.6% of the held-out deals).
The second long run's is stronger (81.4%) and in another folder under the
same file name; which of the two is copied was to be settled before the
run.

**The teacher, chosen on 2026-10-20: neither, `decks-01`** of
[more decks](2026-10-14-more-decks.md), its last checkpoint (1,920 updates
of PPO, half of its duels against itself; 78.3% of the held-out deals by its
last three). Its mirror curve is measured, so a copy can be set on its own
teacher's curve, and it has the strength head to head in the mirror that the
long runs, which played only the pilots, do not. It is of format 14 and
reads the other seat's deck list, so the run is made at `05d27141`, the last
revision that loads it, with the script as it is (`TEACHER` named; the
held-out games are of the policies' `1b85b63`, that revision's pin), on the
CUDA machine once [seventy-one decks](2026-10-19-seventy-one-decks.md) is
closed. Its readings are those above, the teacher's lead over the pilot
being `decks-01`'s. One command from the repository root makes it there:

```sh
bash experiments/commands/specialist-copy-decks-01.sh
```

It makes the checkout of `05d27141` beside the repository when it is not
there (`tools/old_checkout.sh`) and runs that revision's script on this
checkout's files, into `runs/specialist-copy-decks-01` and
`analysis/specialist-copy-decks-01`; then, with this checkout's code, the
fourth reading below, the copies and `decks-01`'s checkpoints folded for
the mirror (`mirror/` there). About three hours in all. Tried small on the
development machine's processor on 2026-10-20 as written (8 of the
held-out games, two updates of 8 duels, both copies, `decks-01`'s points
120 and 720 alone): the teacher measured and copied was `decks-01`'s last
checkpoint, and all four readings were taken. Every file it
reads was there on the CUDA machine that day: the 1,536 held-out games of
`1b85b63`, `once-few`'s checkpoints after 40 to 200 updates and its
measurements, the teacher.

Its readings were confirmed on 2026-10-10, before any run, and are fixed.
Two networks from scratch imitate the checkpoint as `once-few` imitated
the pilot (the same first network, the same deals, one pass, 200 updates):
`choice`, taught the checkpoint's first choice at each decision, and
`distribution`, taught what it gives every answer. A copy is read on the
1,536 held-out deals, deal by deal, by the share of its checkpoints after
160 and 200 updates at which it wins a deal.

1. *A copy of the specialist is past the pilot*: `choice` wins at least 2
   points and two standard errors more of the deals than the pilot.
2. *It holds half of what its teacher has over the pilot*: `choice` wins
   no fewer deals than the pilot and half the teacher's lead over the
   pilot come to, deal by deal. The estimate decides, and its standard
   error is told beside it.
3. *The teacher's distribution teaches more than its choice*:
   `distribution` wins at least 2 points and two standard errors more of
   the deals than `choice`.

The first two are of `choice` because it is taught as the pilot's copy was.

A fourth, proposed when `decks-01` was chosen and **confirmed on 2026-10-20,
before any run**: whether the copy keeps its teacher's strength head to head
in the mirror, which the held-out deals, played against the pilots, do not
tell. `choice`'s checkpoint after 200 updates, folded for the mirror,
against `decks-01` at its copy's end and after each 120 updates of its PPO,
every deal of the mirror from either seat (more decks, in its section on the
mirror along the training of Blackwing alone), level, over and under as
`tools/blackwing_mirror_curve.py` tells them (at two standard errors from
50%):

4. *The copy keeps its teacher's strength in the mirror* when it is level
   with or over `decks-01` after 720 updates, where that curve stops
   rising; it *keeps little of it* when it is under `decks-01` after 120
   updates; otherwise it *keeps some*. `distribution` is set beside it,
   without a mark.

Beside it, also decided and confirmed on 2026-10-10 before any run:
whether one network holds several decks, which either way of making a
network for several decks needs. It is tested with pilots alone
(`experiments/commands/several-decks.sh`, with
`tools/several_decks_report.py`; the trainer has a learner of several
decks since that day, `--learner blackwing,lightsworn`). One deck more was
chosen for now, Lightsworn: a deck played in a way of its own, and the one
that tends to run out of cards. A network of each deck imitates its pilot
as `once-few` did, 200 updates of 520 games, and one network imitates both
from the same first network for 400, which is as many games of each deck.
A deck is read on its screen (1,664 games against the thirteen opponents),
game by game, by the share of a network's checkpoints at which it wins a
game: the deck's own after 160 and 200 updates, the network of both after
320 and 400.

1. *The network of both plays a deck no worse than the deck's own
   network*, for each of the two decks: it is worse when it wins at least
   2 points fewer of the screen's games and by more than two standard
   errors.
2. *One network holds the two decks*: it is worse on neither.

How often each network's first choice is its pilot's is told beside them,
without a mark.
