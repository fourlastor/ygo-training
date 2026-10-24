# Imitating the search

**Status:** COMPLETED

**Proposed:** 2026-10-05

**Activated:** 2026-10-06

**Last updated:** 2026-10-07

**Completed:** 2026-10-07

## Conclusion

Imitating the answers the search played does not make a stronger player
than imitating the pilot, on up to 7,680 searched games and up to eight
times through them. The learner comes to give the search's answer at 9% of
the deviations of games it has not seen, 15% of those the search decided
most firmly, and wins as often as its control (+1.1 ± 1.1 points at best,
with the search 9.7 above its pilot). More games raise what is learned,
and by little; on every set the learner ends by giving more other answers
than are worth it. No reading that asks for more than that is met, and no
checkpoint is promoted: the imitation seed's version 89 is the one to start
from.

What the search measured for every answer it tried, and not only the one it
played, is taken up in
[Learning the search's leads](2026-10-07-search-leads.md).

## Research questions

1. Does a learner that imitates games played by the one-step search learn
   when to leave the pilot's answer, on games it has not seen?
2. Does that make it win more than a learner that imitates the pilot, and
   how much of the search's 16 to 18 points does it take?
3. How many searched games does that need?

After the run of 2026-10-06 (see [Result](#result)):

1. Barely. On 7,680 games it gives the search's answer at 7.3% of the
   held-out deviations, where a learner that imitated the pilot on the same
   deals gives it at 2.8%. It gives it at 13 to 97% of the deviations of the
   games it trained on. Of the held-out deviations the search decided most
   firmly it follows one in nine.
2. No: 65.3% against 64.8% (+0.5 ± 1.1 points), where the search is 9.7
   points above the pilot. Both learners are still under the pilot.
3. More games do not raise the first: it is 7 to 9% on every set from 960
   games to 7,680. The two larger sets were still rising when their 64
   updates ended, so where they level off is not known.

Read by the search's own worlds on 2026-10-07 (see
[The models' answers by the search's worlds](#the-models-answers-by-the-searchs-worlds)),
the largest set stands apart. There the Search arm's other answers than the
pilot's are worth 2.2 ± 0.5 points a game more than its control's, where
the search measured them; on the smaller sets they are worth the same. On
the largest set that gap built up over the updates and was still growing at
the last. The smaller sets have it early and lose it as they go through
their games again.

Trained on to 128 updates, outside the protocol
([The largest set trained on](#the-largest-set-trained-on)), the largest
set's Search arm gives the search's answer at 9.1% of the held-out
deviations, the gap in what its other answers are worth falls back to 1.2 ±
0.6, and it wins 66.2% of the deals against its control's 65.1% (+1.1 ±
1.1).

## Why

The [PPO regression remedies](2026-09-29-ppo-regression-remedies.md)
experiment brought a Blackwing learner to its teacher's level by imitation
and no further: over 110 updates anchored PPO added nothing in aggregate. A
one-step search on top of the same teacher wins 80.1% where the teacher wins
62.4% (Monarch: 86.8% against 71.1%), against the same opponents. Its
answers are labels better than the teacher's, and imitating them does not
depend on the credit PPO assigns.

The difficulty is that the search agrees with the pilot almost everywhere.
It leaves the pilot's answer in 3 to 5% of the decisions it examines, 1.8 to
1.9 times a game, and simple descriptions of a decision do not tell those
apart: no kind of alternative that comes up often is taken in more than 9%
of its offers, and a boosted-tree model on the situation keeps 80% of the
deviations only by keeping 40% of the decisions. Whether a network that
sees the whole position learns the difference is the question.

Those marks are of the pilots and revisions of early October. At the ones
pinned now, on the 384 Blackwing games the search's speed-up was checked on
(32 against each of the 12 decks, seed 730000):

| | Wins | Rate |
| --- | ---: | ---: |
| Pilot | 246 of 384 | 64.1% |
| Search | 296 of 384 | 77.1% |

The search wins 60 deals the pilot loses and loses 10 the pilot wins. It
leaves the pilot's answer 533 times, 1.39 times a game and at 2.6% of the
20,181 decisions it examines: 360 in the Main Phase, 130 in chain windows,
38 in the Battle Phase, 5 at a yes/no question or a battle position. 121
of the 384 games hold no deviation. The pilot's rules were improved from
searches on this seed, so the held-out run, on another, gives the marks the
experiment is read against; the gap to expect is nearer 13 points than 17.

## Design

Agreed on 2026-10-05, to start as soon as the port of the duel environment
is done.

- **Seed.** The pilot imitated to its plateau, as in the remedies
  experiment (the teacher plays, the learner imitates), in the new
  observation format.
- **Searched games.** `policy-bench search` (strict, 8, 32 and 96 worlds,
  z 1.645) plays the learner's deck against the 12 other decks on a plain
  sample of deals. The games are generated once, outside training, and kept
  as replay records. The trainer plays each record again in its own duel
  environment and reads, at every decision of the searching seat, the
  learner's input and the answer that was played. That answer is the label:
  the search's where it left the pilot's answer, the pilot's everywhere
  else, including the decisions the search does not examine (an open chain,
  a face-down monster on the other side when strict, card selections).
- **Arms**, from the seed, with the same number of updates and passes:

  | Arm | Games imitated |
  | --- | --- |
  | Pilot | the pilot's own games on the same deals (each search row holds both games) |
  | Search | the searched games |

  The difference between the two is what the search's answers add, with
  the deals, the amount of training and the opponents held equal.
- **Training** (agreed on 2026-10-06). Every run does 64
  updates of 480 games from the seed, with the seed's recipe otherwise (4
  epochs, minibatch 256, value weight 0.5), whatever the size of its set:
  the sizes then differ only in how many different games stand behind the
  same number of gradient steps. A pass over the four sets is 2, 4, 8 and
  16 updates, so the largest set is gone through 4 times and the smallest
  32. Since nothing says in advance how long is right, every run is
  measured at updates 4, 8, 16, 32 and 64. By the timings of the earlier
  runs an update should cost about 20 seconds on the CUDA machine and a
  measurement about 3 minutes: 5 hours for the 8 runs.
- **No games selected by their result, and no pilot-labelled games beside
  the searched ones.** The proposal first discussed was to search only the
  deals the pilot loses and to train on those next to the pilot's games.
  Those deals hold half of the deviations (49% for Blackwing, 46% for
  Monarch) for two fifths of the search's time, so searching them alone
  finds 15% more deviations per CPU-second and misses half. The half it
  misses stays labelled with the pilot's answer, in positions of the kind
  the search corrects, and a deal's first, lost game labels the position of
  its first deviation both ways. The searched games need no rehearsal
  beside them: 95 to 97% of their examined decisions already carry the
  pilot's answer.
- **Size.** Nested sets of 960, 1,920, 3,840 and 7,680 searched games (80,
  160, 320 and 640 against each of the 12 decks, half of them going first),
  for a learning curve. The first design said 1,000 to 8,000; these are the
  nearest numbers that give every deck and seat an equal share. A searched
  game took 54 (Monarch) to 65 (Blackwing) CPU-seconds in the strict runs.
  Since then the search skips the decisions its pilot's answer wins
  outright, and its runtime was reworked (ygo-policies `bc8c0f7`): a
  Blackwing game now takes 28.9 CPU-seconds, 48.6 before the rework, with
  the same games. 7,680 games are about 62 CPU-hours, three hours on 22
  workers, and hold about 10,700 deviations at 1.39 a game; the smallest
  set about 1,300.
- **Held-out games.** A separate search run on another seed, 64 games
  against each of the 12 decks, with the search's log, so that the decisions
  it examined are known. Its rows also give the pilot's and the search's
  marks for the pilots as pinned then. Agreed on 2026-10-06, now that it
  costs 40 minutes: 128 games a deck, not 64, 1,536 in all with about 2,100
  deviations. A quarter of a 13-point gap is 3.3 points. If two models'
  results differ on a fifth to a third of shared deals (the pilot's and the
  search's differ on 18%), that is 1.7 to 2.2 standard errors of their
  difference on 768 deals and 2.4 to 3.0 on 1,536. What varies between two
  training runs comes on top: the replicate PPO runs of the remedies
  experiment had a standard deviation of 3.1 points on 1,664 games.
- **Decks.** Blackwing first, one model per deck. Monarch second: its
  pilot's mistakes are frequent and simple (Frog effects fired in chain
  windows).

## Measures

- **The answer key.** The held-out searched games are played again with the
  model watching the searching seat:
  - at the deviations, how often the model's first choice is the search's
    answer, and how often it is still the pilot's;
  - at the examined decisions where the search kept the pilot's answer, how
    often the model's first choice is another one;
  - both by kind of decision (chain window, Main Phase action, ending the
    turn or the phase, Battle Phase).
- **Held-out win rate** (argmax) against the 12 other decks on the held-out
  run's own deals, so that model, pilot and search are compared deal by
  deal.
- **The learning curve** of both over the four sizes.

## Readings (fixed on 2026-10-06)

The thresholds were proposed together with the number of updates. The
number was agreed; the thresholds were shown and left as they are. No run
of the experiment existed when they were fixed.

The main comparison is the Search arm against the Pilot arm on 7,680 games
after all 64 updates, on the 1,536 held-out deals, deal by deal and
deviation by deviation. `tools/search_imitation_report.py` gives each
difference with its standard error.

- *The search's answers are learnable* if the Search arm gives the search's
  answer at 5 points more of the held-out deviations than the Pilot arm
  does, and at more than three standard errors, while at the decisions the
  search examined and kept it gives another answer than the pilot's no more
  than 1 point more often.
- *It plays better* if it wins more of the held-out deals than the Pilot
  arm by a quarter of the gap between the search and the pilot on those
  deals (3.3 points if the gap is 13), and by more than two standard
  errors.
- *More games would help* if both differences still grow from 3,840 to
  7,680 games.
- *Not learnable this way* if on 7,680 games the Search arm's rate at the
  deviations stays within 3 points of the Pilot arm's at updates 16, 32 and
  64. The follow-ups are under Later.
- The other measurements are the learning curves, and qualify the main
  comparison. Still rising at update 64: longer training would help. Below
  an earlier update's: the run overfits, and the earlier value is reported
  as such, not claimed.

## The protocol, frozen on 2026-10-06

| | |
| --- | --- |
| Pilots, engine, scripts, cards | ygo-policies `bc8c0f7` (engine `1c49ef7`, CardScripts `37f270d`, BabelCdb `fdf92ae`) |
| Checkpoint format, board | 13, `mean-pool` |
| Search | strict, 8, 32 and 96 worlds, z 1.645 |
| Deck and opponents | Blackwing against the 12 other decks of the 13-deck pool |
| Games to train on | seed 800000, 640 a deck; sets of the first 960, 1,920, 3,840 and 7,680 |
| Held-out games | seed 900000, 128 a deck, with the search's log |
| Seed | the pilot imitated: 520 duels an update against the 13 decks, 4 epochs, value weight 0.5, training seed 2012, evaluated every 10 updates, to its plateau |
| Arms | 64 updates of 480 recorded games from the seed, same recipe, training seed 2012 |
| Measured | the seed, and every run at updates 4, 8, 16, 32 and 64, on the held-out run |

The seed is at its plateau when its last evaluation is not more than 2
points above the one before. It starts with 50 updates and goes on by 20
until then, at most to 110, where the run stops for a look instead. (The
seed script has had another rule since 2026-10-07, after this one stopped
the seed too early: see [Current next step](#current-next-step).)

The whole experiment is one command on the CUDA machine,
[`commands/search-imitation.sh`](commands/search-imitation.sh): the seed
([`format13-imitation-seed.sh`](commands/format13-imitation-seed.sh)), the
searched games ([`search-imitation-games.sh`](commands/search-imitation-games.sh),
a deck at a time), then the arms, their measurements and the report
([`search-imitation-arms.sh`](commands/search-imitation-arms.sh)). Each
step skips what is finished and goes on with what is not. Nothing of it
runs on the development machine.

No checkpoint of format 13 existed when this was frozen. The seed run wrote
the first, and its last checkpoint (version 49) is the only one the arms
loaded. Since the run, the CUDA machine holds that seed run
(`runs/format13-blackwing-imitation-seed-2012`) and the eight arms
(`runs/search-imitation/`, versions 50 to 113). None is promoted. The seed
stood 5.5 points under its pilot at version 49
([What the run shows](#what-the-run-shows)). The seed run was continued on
2026-10-07 and levels off at version 89, 2.1 points under its pilot
([Current next step](#current-next-step)): that is the checkpoint to start
anything further from. The arms started from version 49, and their
checkpoints are measurements.

## The run

Everything ran on the CUDA machine as one command, from 2026-10-06 14:31 to
2026-10-07 03:20 by the times of its files, with the trainer at `90495d4e`
and ygo-policies at `bc8c0f7`, both clean in every run's `run.json`. The
pilot's reference on the evaluation suite came out as on the development
machine, case by case (1,080 of 1,664 games): the two machines play the
same duels. The outputs are under `analysis/format13-seed/` and
`analysis/search-imitation/`; the rows of the two search runs and the
checkpoints stayed on that machine.

| Step | Took | Expected | |
| --- | --- | --- | --- |
| Seed | 25 min | 30 min | 50 updates |
| Held-out games | 1 h 26 min | 40 min | 1,536 games with the log |
| Games to train on | 6 h 34 min | 3 h | a searched game takes 72 seconds of a worker there, against 28.9 CPU-seconds on the development machine |
| Arms | 4 h 25 min | 5 h | 21 seconds an update, 10 minutes to measure a run at its five updates |

**The searched games.**

| Run | Games | Pilot | Search | Gain | Deviations |
| --- | ---: | ---: | ---: | ---: | ---: |
| Held out, seed 900000 | 1,536 | 67.6% | 77.3% | +9.7 ± 1.0 | 2,225, 1.45 a game |
| To train on, seed 800000 | 7,680 | 67.1% | 78.3% | +11.2 ± 0.5 | 11,050, 1.44 a game |

On the held-out deals the search wins 199 that the pilot loses and loses 50
that it wins. A quarter of that gap, which the second reading asks for, is
2.4 points. The search leaves the pilot's answer at 2.7% of the 81,443
held-out decisions it examined.

**The seed.** Its evaluations on the suite were 49.1%, 56.9%, 57.9%, 60.1%
and 59.4% at versions 9 to 49, so the rule stopped it at version 49, the
first place it could. The pilot wins 64.9% of that suite.

## Result

The main comparison: both arms on 7,680 games after 64 updates, on the
1,536 held-out deals, their 2,225 deviations and the 79,218 decisions the
search examined and kept.

| | Seed | Pilot arm | Search arm | Search − Pilot |
| --- | ---: | ---: | ---: | ---: |
| Wins | 951, 61.9% | 996, 64.8% | 1,003, 65.3% | +0.5 ± 1.1 |
| At the deviations: the search's answer | 87, 3.9% | 62, 2.8% | 163, 7.3% | +4.5 ± 0.5 |
| At the deviations: still the pilot's | 85.6% | 89.0% | 82.4% | |
| Examined and kept: another answer than the pilot's | 6.2% | 5.1% | 6.1% | +1.0 |

The pilot wins 1,038 of those deals (67.6%) and the search 1,187 (77.3%).
The Search arm follows the search at 126 deviations where the Pilot arm
does not, the Pilot arm at 25 where the Search arm does not. Each wins
about as many deals the other loses, 139 and 132.

Against the readings:

- *The search's answers are learnable:* **not met.** 4.5 points where 5
  were asked, though at eight standard errors where three were asked; and
  1.0 point more of other answers at the kept decisions, which is the
  limit. That point is the Pilot arm coming closer to the pilot (6.2% to
  5.1%) while the Search arm stays where the seed was.
- *It plays better:* **not met.** +0.5 ± 1.1 points where 2.4 points and
  two standard errors were asked.
- *More games would help:* **not met.** From 3,840 to 7,680 games the
  difference at the deviations goes from +5.9 to +4.5 points. The one in
  wins goes from -0.8 to +0.5, and neither is a standard error from
  nothing.
- *Not learnable this way:* **not met by its letter.** The difference at
  the deviations is +2.5, +3.2 and +4.5 points at updates 16, 32 and 64:
  within 3 points at the first only.
- *The learning curves.* The Search arm on 7,680 games is still rising at
  update 64 (5.4%, 5.9%, 7.3% at updates 16, 32 and 64). The smaller sets
  are 2, 4 and 8 times further in passes over their games and show where
  that leads: to 8.6% at most, with more answers changed where the search
  changed none, and fewer wins.

So the result lies between the two readings it could have met. The Search
arm takes something from the search's answers that holds on games it has
not seen: the answer at one deviation in twenty more than its control. None
of it shows in the results of its games.

**Every size, at the held-out deviations.** The Search arm's first choice
is the search's answer (the seed: 3.9%; the Pilot arm stays between 2.5%
and 3.6% throughout):

| Games | Update 4 | 8 | 16 | 32 | 64 | Passes by 64 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 960 | 5.3% | 6.8% | 8.6% | 8.5% | 7.9% | 32 |
| 1,920 | 4.8% | 5.9% | 7.7% | 7.8% | 7.7% | 16 |
| 3,840 | 4.8% | 5.2% | 6.7% | 7.6% | 8.6% | 8 |
| 7,680 | 5.6% | 5.4% | 5.4% | 5.9% | 7.3% | 4 |

After the same number of passes the four sets stand at the same place: 6.8%,
7.7%, 7.6% and 7.3% after four.

**Every size, after 64 updates.** Search arm, then Pilot arm:

| Games | The search's answer at the deviations | Another answer where examined and kept | Wins | Wins, Search − Pilot | Search arm at the deviations of its own games, last pass |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 960 | 7.9%, 3.6% | 8.5%, 6.4% | 59.8%, 62.3% | -2.5 ± 1.2 | 97.1% |
| 1,920 | 7.7%, 2.9% | 7.9%, 6.1% | 60.7%, 62.2% | -1.5 ± 1.1 | 84.9% |
| 3,840 | 8.6%, 2.7% | 7.6%, 5.5% | 62.7%, 63.5% | -0.8 ± 1.2 | 45.3% |
| 7,680 | 7.3%, 2.8% | 6.1%, 5.1% | 65.3%, 64.8% | +0.5 ± 1.1 | 12.7% |

Over the 20 measurements the difference in wins lies between -2.7 and +2.1
points, with a standard error of 1.0 to 1.2.

## What the run shows

- **The deviations are memorised, not learned.** The last column above is
  measured during training, before each update, on the games of that update
  (`imitation_model_departure_agreement`). On 960 games the model gives the
  search's answer at 97% of the deviations of games it has gone through 31
  times, and at 7.9% of the held-out ones. In the first pass, where an
  update's games are new to it, the rate is 4.8% to 5.4% on every set,
  which agrees with the held-out measurement, made by other code. The
  labels reach the model, and it fits them on the games it has seen and
  hardly beyond.
- **More games do not raise it, and more passes cost.** The rate on games
  not seen depends on the passes and not on how many games there are. What
  grows with the passes is the number of other answers where the search
  changed nothing (2.0 points more than the Pilot arm on 960 games, 1.0 on
  7,680), and with it the Search arm falls behind its control, by 2.5
  points on 960 games.
- **Where it follows.** On 7,680 games, by what the search did:

  | Deviation | Held out | Seed | Pilot arm | Search arm |
  | --- | ---: | ---: | ---: | ---: |
  | Main Phase: ends the turn instead | 404 | 0.2% | 0.2% | 6.2% |
  | Main Phase: another action | 1,139 | 5.1% | 3.2% | 8.2% |
  | Chain window: activates where the pilot passes | 384 | 2.1% | 2.1% | 4.2% |
  | Chain window: passes where the pilot activates | 89 | 10.1% | 9.0% | 15.7% |
  | Battle Phase | 170 | 5.3% | 4.1% | 7.1% |

  The other 39 deviations are yes/no questions, battle positions and one
  activation in the place of another. The pilot's answer remains the Search
  arm's first choice at 78% to 96% of each kind. The seed has seen no
  answer of the search: where it agrees, it failed to copy the pilot, and
  imitating the pilot longer removes some of that.
- **What it has from the search, and what the seed had already.** The seed
  gave the search's answer under 1% of its weight at 1,893 of the 2,225
  held-out deviations, and 20% or more at 134. All 87 of its own agreements
  are among those 134: a copy's doubts, of which the Pilot arms keep 49 to
  52. Of the Search arm's 163 agreements on 7,680 games, 62 are among them
  too and 75 among the 1,893. Those a run can only have from the games it
  imitated: the Pilot arms give the search's answer at 0.1 to 0.4% of them.
  The Search arm's rate there, by how often it has gone through its set
  (each ± 0.2 to 0.5; `search_imitation_report.py --records`):

  | Times through the set | 960 games | 1,920 | 3,840 | 7,680 |
  | --- | ---: | ---: | ---: | ---: |
  | 1 | | 1.0% | 1.6% | 1.6% |
  | 2 | 1.1% | 1.7% | 3.1% | 2.6% |
  | 4 | 3.4% | 4.3% | 3.8% | 4.0% |
  | 8 | 4.9% | 4.7% | 5.5% | |
  | 16 | 5.0% | 4.2% | | |
  | 32 | 4.5% | | | |

  The two smaller sets level off at 4 to 5%. The two larger were still
  rising when their 64 updates ended (1.6%, 2.6% and 4.0% after 16, 32 and
  64 updates on 7,680 games), so the run does not say where they level
  off, nor whether more games raise that level. By the z the search took
  them at, in two near halves (963 under 2.5, 930 at 2.5 and more), the
  rates after 64 updates are 3.1% and 5.9% on 960 games, 3.6% and 4.8% on
  1,920, 4.7% and 6.5% on 3,840, and 1.9% and 6.1% on 7,680 (each ± 0.4 to
  0.8). The largest set tells the firm from the marginal best. That may be
  its size or its four passes: after four passes the others stood at 2.6%
  and 4.2%, 3.1% and 5.5%, and 2.5% and 5.1%.
- **It plays differently from the pilot, not better.** Against the pilot's
  own results on the 1,536 deals the Search arm wins 115 deals the pilot
  loses and loses 150 it wins; the Pilot arm 76 and 118. The Search arm
  changes the result of 265 deals and the Pilot arm of 194, to the same
  balance.
- **The seed was not at the pilot's level, and the stopping rule let it
  pass.** On the held-out deals the seed wins 61.9% where the pilot wins
  67.6% (-5.7 ± 1.0), and 5.5 points less on the suite. The Pilot arm,
  which goes on imitating the same pilot, is at 64.8% after its 64 updates
  (-2.7 ± 0.9) and still rising: 62.0%, 63.7%, 64.8% at updates 16, 32 and
  64. The rule compared two evaluations, each good to a point, and never
  asked how far under the pilot the seed stood, as the
  [remedies](2026-09-29-ppo-regression-remedies.md) readings had. The two
  arms are still on an equal footing, since they start from the same seed
  and do the same updates. But both spent those updates partly on catching
  up with the pilot. The model the search's answers were added to gives
  another answer than the pilot's at 5 to 6% of the decisions the search
  examined and kept; the search itself changes 2.7%.
- **The value head predicts nothing on games it has not seen.** Its
  explained variance, measured before each update on that update's games,
  is below zero in every update of the seed run after the first (-0.12 on
  average), and between -0.28 and +0.06 in the first pass of every arm. It
  is high only on games seen before, up to 0.95 on the 960 games. Within an
  update the head fits the results of that update's games: its loss
  averages 0.19 over the four epochs, where the variance of the results is
  0.9, and a game's result is the target at every one of its transactions,
  some 75 of them. The three imitation runs of the remedies experiment
  averaged -0.10, +0.05 and -0.16. In imitation updates a game's result has
  not been a target the value head learns from.

  These explained variances were taken over every step of an update's
  games, where the head has a target at the first step of each transaction
  only: 92% of the steps. A review by another agent pointed that out on
  2026-10-07, and the trainer now measures at the first steps (a metrics
  record with `imitation_transactions` holds the newer figure). The saved
  figures cannot be turned into that one; they bound it. With 8% of the
  steps left out, -0.12 over all of them is between -0.22 and -0.03 at the
  first steps, unless the head is confidently wrong at the later ones. In
  the first ten updates of a small run made here to compare the two (130
  games an update) they differ by 0.02 at most, either way. Measured on
  2026-10-07 from the seed's version 89, by
  [`format13-value-check.sh`](commands/format13-value-check.sh) (trainer
  `8bdffe5d`): -0.10 at the first steps on 520 games new to it, and -0.07,
  -0.13 and -0.14 before the three updates after. The finding stands.

## The labels

`tools/policy_search_firmness.py` reads a logged search run for how firmly
each deviation was decided. On the held-out run's log (copied back on
2026-10-07), where it reproduces the search's choice at every one of the
81,443 examined decisions:

| The search's test | Decisions | An alternative taken | One ahead, not taken |
| --- | ---: | ---: | ---: |
| The pilot's answer wins all 8 first worlds, nothing else tried | 32,516 | | |
| Decided after 8 worlds | 36,873 | | |
| After 32 | 6,906 | 727 | 1,746 |
| After 96 | 5,148 | 1,498 | 2,443 |

- **The deviations are the top of a continuum.** At 4,189 decisions an
  alternative ended ahead of the pilot's answer and was not taken, nearly
  twice as many as the 2,225 where one was. A fifth of the deviations were taken at
  a z under 2, where the bar is 1.645; the median is 2.5.
- **Most of a lead is the worlds that showed it.** Where the test went on
  after 32 worlds, the leading alternative was 9.0 points of win rate ahead
  of the pilot's answer in those worlds, and 1.3 points ahead in the next
  64, which had no part in choosing it (5,148 decisions; the two leads
  correlate at 0.27).
- **About half of the deviations would not be found again.** With the
  worlds drawn again from those each was played in, the same test takes
  the same alternative 57% of the time on average, and under one time in
  two for 47% of the deviations. The deviations are those that passed, so
  their leads flatter them and the true rates are lower.

The 384 logged games of the development seed, read first, give the same
figures (56%, and 9.1 points against 0.9).

A test at a one-sided 5% level, made at some 49,000 decisions of these
games, passes many of them by the luck of its worlds. That costs the search
little: an alternative taken by luck is on average about as good as the
pilot's answer. An imitator is asked to reproduce it all the same, and
positions of the same kind carry both labels.

**How much of the shortfall that explains.** `search_imitation_report.py
--records` sets each run's rate apart by the z a deviation was taken at.
On 7,680 games after 64 updates:

| The search's z | Held-out deviations | Found again, at most | Median gain | Seed | Pilot arm | Search arm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Under 2 | 464 | 28% | 6 points | 2.4% | 1.9% | 4.3% |
| 2 to 2.5 | 634 | 49% | 10 points | 3.6% | 2.5% | 6.2% |
| 2.5 to 3 | 586 | 68% | 19 points | 4.3% | 3.4% | 7.7% |
| 3 and more | 541 | 80% | 25 points | 5.2% | 3.1% | 10.9% |

The Search arm follows the firmly decided deviations two and a half times
as often as the marginal ones. On the smaller sets it is 1.4 to 2.1 times,
with 9.8% to 11.6% at a z of 3 and more. So the luck in the labels does
cost. It is not the main thing: of the deviations decided most firmly,
which are worth the most, the learner follows one in nine. Nor is the
answer nearly its first choice there. At 65% of those 541 it gives the
search's answer less than 5% (the median is 1.5%), so reading the model at
a lower bar than its first choice would not find them.

A quick look at whether the leads can be told from a coarse description of
the decision, with a script not kept: a small network on the log's own
`situation` (phase, Life Points, counts of cards) and on what the two
answers are, fitted to the first-stage lead of every alternative tried
(114,753 in the held-out games) and asked about games it was not fitted on.
It ranks alternatives by how much worse they are (correlation 0.38) and
finds none that are better: the half percent it ranks highest lead by 0.4 ±
0.6 points. The mean lead of the same kind of alternative in other games
does no better than +2.4 ± 0.7 points on its highest half percent.
Whatever makes an alternative better is not in such a description.

## The models' answers by the search's worlds

Measured on 2026-10-07 with
[`commands/search-imitation-tried.sh`](commands/search-imitation-tried.sh)
(trainer `24fcc146`; 20 minutes on the CUDA machine, nothing trained). At
each of the 81,443 decisions the search examined in the held-out games, a
model's first choice is read by the lead the search measured for it over the
pilot's answer in its first 8 worlds ([The tooling](#the-tooling)): for the
eight arms after their 64 updates, the seed they started from, and the seed
continued to version 89. Leads are in points of win rate.

| Model | Other answers a game | Where the pilot's won all 8 worlds | With a lead | Ahead / level / behind | Mean lead | A game's leads added up | Deals won, against the pilot |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Seed, version 49 | 2.65 | 1,580 | 2,494 | 425 / 1,479 / 590 | -2.2 ± 0.3 | -3.6 ± 0.5 | -5.7 |
| Seed, version 89 | 2.15 | 1,295 | 2,006 | 313 / 1,204 / 489 | -2.8 ± 0.3 | -3.6 ± 0.5 | -4.8 |
| Pilot arm, 960 games | 2.75 | 1,603 | 2,627 | 437 / 1,580 / 610 | -2.3 ± 0.3 | -3.9 ± 0.5 | -5.3 |
| Search arm, 960 | 3.98 | 2,124 | 3,976 | 825 / 2,181 / 970 | -1.3 ± 0.2 | -3.4 ± 0.7 | -7.7 |
| Pilot arm, 1,920 | 2.58 | 1,538 | 2,418 | 367 / 1,475 / 576 | -2.7 ± 0.3 | -4.2 ± 0.5 | -5.4 |
| Search arm, 1,920 | 3.65 | 1,934 | 3,657 | 746 / 2,000 / 911 | -1.6 ± 0.3 | -3.8 ± 0.7 | -6.9 |
| Pilot arm, 3,840 | 2.21 | 1,305 | 2,095 | 327 / 1,285 / 483 | -2.4 ± 0.3 | -3.3 ± 0.5 | -4.0 |
| Search arm, 3,840 | 3.51 | 1,835 | 3,546 | 766 / 1,907 / 873 | -1.2 ± 0.3 | -2.9 ± 0.6 | -4.9 |
| Pilot arm, 7,680 | 2.01 | 1,186 | 1,908 | 322 / 1,175 / 411 | -1.8 ± 0.3 | -2.2 ± 0.4 | -2.7 |
| Search arm, 7,680 | 2.64 | 1,399 | 2,653 | 577 / 1,506 / 570 | -0.0 ± 0.3 | -0.0 ± 0.5 | -2.3 |
| The search | 1.45 | | 2,225 | 2,225 / 0 / 0 | +23.0 ± 0.3 | +33.3 ± 0.9 | +9.7 |

Game by game, the Search arm's added-up leads are above the Pilot arm's by
0.5 ± 0.7 points on 960 games, 0.4 ± 0.6 on 1,920, 0.4 ± 0.6 on 3,840 and
2.2 ± 0.5 on 7,680.

- **What the measure sees.** An answer other than the pilot's gets a lead
  where the search tried it, which is where the pilot's answer lost at
  least one of the 8 worlds: 61 to 65% of a model's other answers at the
  examined decisions. The rest are given where the pilot's answer won all
  8. Nothing was tried there, an answer can only be level or behind, and
  the Search arm gives more of those than its control on every set. So a
  model's sum is too high by what it loses there, and the Search arm's by
  more. How much is not known. The alternatives the search tried cost more
  the better the pilot's answer does: on average they lead by 2.9 points
  where the pilot's answer lost all 8 worlds, and trail by 9.8 where it won
  7 of them, losing 15% of the worlds that it won. Answers at decisions the
  search did not examine, the selections of cards among them, are not seen
  at all. The search's own choices lead by 23 points in the worlds they
  were taken on and win 9.7 points more deals, so a lead is no count of
  games.
- **Every learner's other answers lose on average**: 1.8 to 2.8 points each
  for the seeds and the Pilot arms, where the alternatives the search tried
  lose 5.7. Over half are level (54 to 62%), the 8 worlds ending the same
  either way. A game's leads add up to -2.2 to -4.2 points for those six
  models, whose shortfall under the pilot on the same deals is 2.7 to 5.7.
- **Up to 3,840 games the Search arm gives more other answers, not better
  sums.** It gives 40 to 60% more of them than its control. Each is less
  costly on average (-1.2 to -1.6 points against -2.3 to -2.7) and a game's
  leads come to the same. It also gives 400 to 530 more where the pilot's
  answer won every world, and wins 0.8 to 2.5 points fewer deals.
- **On 7,680 games its other answers are level with the pilot's, and its
  control's are not.** A mean lead of -0.0 ± 0.3 against -1.8 ± 0.3, and
  a game's leads 2.2 ± 0.5 points above the Pilot arm's: better in 349
  games, worse in 276, and 3.5 ± 0.6 above the seed's, where the Pilot
  arm is 1.3 ± 0.5 above it. Most of it is in the Main Phase (+0.6 ± 0.5 a
  game against -0.9 ± 0.3), with smaller differences the same way in chain
  windows and the Battle Phase. It gives 213 more answers than its control
  where the pilot's won every world, fewer than on any smaller set. This is
  the one measurement of the experiment in which what the Search arm
  learned is worth something on games it has not seen, and it is there on
  the largest set only. Each arm is a single training run, and the error is
  that of the held-out games alone. In deals won it is +0.5 ± 1.1.
- **Nothing better lies below the first choice.** Taking a model's heaviest
  alternative once it has half, a fifth, a tenth or a twentieth of the
  weight of the pilot's answer gives more other answers and a lower sum at
  each step, for every one of the ten models: -0.0, -0.5, -2.6, -4.9 and
  -8.4 points a game for the Search arm on 7,680 games, -2.2, -2.5, -3.1,
  -3.9 and -4.8 for its control.
- **The seed at version 89 gives fewer other answers and loses as much by
  them.** 2.15 a game against 2.65 at version 49, and a game's leads the
  same (-0.1 ± 0.5 game by game). It wins 62.8% of the held-out deals
  against 61.9%.

### By update

The same measure at every update the arms were measured after, run on
2026-10-07 (trainer `524fbd7e`, 84 minutes;
`analysis/search-imitation/tried-by-update/`). A game's leads added up, the
Search arm's above the Pilot arm's game by game, by how often each had gone
through its set. Down a column are a set's updates 4, 8, 16, 32 and 64.

| Times through the set | 960 games | 1,920 | 3,840 | 7,680 |
| --- | ---: | ---: | ---: | ---: |
| 1/4 | | | | +0.7 ± 0.5 |
| 1/2 | | | +0.7 ± 0.5 | +1.3 ± 0.5 |
| 1 | | +2.0 ± 0.5 | +1.0 ± 0.5 | +1.1 ± 0.5 |
| 2 | +0.8 ± 0.5 | +0.5 ± 0.6 | +1.3 ± 0.6 | +1.5 ± 0.5 |
| 4 | -0.0 ± 0.6 | -0.0 ± 0.7 | +0.4 ± 0.6 | +2.2 ± 0.5 |
| 8 | -0.9 ± 0.7 | -0.4 ± 0.8 | +0.4 ± 0.6 | |
| 16 | +0.7 ± 0.7 | +0.4 ± 0.6 | | |
| 32 | +0.5 ± 0.7 | | | |

- **On 7,680 games the gap built up.** It is 0.7, 1.3, 1.1, 1.5 and 2.2
  points at the five updates, and grew by 1.5 ± 0.7 from the first to the
  last, game by game. The Search arm's own sum went from -2.5 to -0.0 (2.5
  ± 0.6), the Pilot arm's from -3.2 to -2.2 (1.0 ± 0.5). It is made in the
  Main Phase, where the Search arm's leads add up to -0.6, -0.5, -0.3, +0.2
  and +0.6 points a game and the Pilot arm's stay between -0.9 and -1.3.
  It was still rising at the last update. Before the run a gap that builds
  up over the passes was to count as something learned, and one there at
  the last update only as nothing to build on.
- **Every Search arm starts ahead, and the smaller sets lose it.** After 4
  updates, 1,920 games imitated, the Search arm is 0.7 to 2.0 points above
  its control on every set. On 960 and 1,920 games that is gone by the
  fourth time through the set. On 3,840 it holds for two (+1.3) and falls
  to +0.4. What grows on those sets is the number of other answers: from
  2.8 a game to 4.0 on 960 games, 2.7 to 3.7 on 1,920 and 2.6 to 3.5 on
  3,840, and of those given where the pilot's answer wins outright, from
  1.0 to 1.2 and more. On 7,680 the Search arm gives 2.6 to 2.8 a game at
  every update, and 0.9 to 1.05 of them outright.
- **So it is the number of games, and not only the fewer passes.** Four
  times through its set the gap is -0.0, -0.0, +0.4 and +2.2 from the
  smallest set to the largest, and twice through it +0.8, +0.5, +1.3 and
  +1.5. After the same number of updates, from 8 on, the largest set is
  ahead of every smaller one or level with it. Each is still one run of
  each arm.
- **In deals won it does not show.** On 7,680 games the Search arm is 0.7,
  -0.5, 2.1, 0.5 and 0.5 points above its control at the five updates,
  each ± 1.1. Both arms were still gaining at the last: 64.2% to 65.3% and
  63.7% to 64.8% from update 32 to 64, with the pilot at 67.6%.

### The largest set trained on

The two arms on 7,680 games were trained 64 updates further on 2026-10-07,
to eight times through the set, and measured after updates 96 and 128
(trainer `d662d12e`; `analysis/search-imitation/tried-longer/`). This is
outside the frozen protocol, whose main comparison is after 64 updates, and
no reading of [Readings](#readings-fixed-on-2026-10-06) is claimed from it.
The checkpoint of update 64, measured again, gives the leads it gave
before, game for game.

| Update | Times through the set | Search's answer at the held-out deviations | Where the seed gave it none: all, z under 2.5, 2.5 and more | Other answers a game | A game's leads added up | Search arm above its control | Deals won | Difference |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 4 | 7.3%, 2.8% | 4.0%, 1.9%, 6.1% | 2.64, 2.01 | -0.0, -2.2 | +2.2 ± 0.5 | 65.3%, 64.8% | +0.5 ± 1.1 |
| 96 | 6 | 8.5%, 2.7% | 5.2%, 3.1%, 7.4% | 2.95, 2.00 | -0.8, -1.6 | +0.8 ± 0.6 | 65.0%, 64.6% | +0.4 ± 1.1 |
| 128 | 8 | 9.1%, 2.6% | 5.8%, 3.5%, 8.2% | 2.96, 1.92 | -0.8, -2.0 | +1.2 ± 0.6 | 66.2%, 65.1% | +1.1 ± 1.1 |

Where two figures stand, the first is the Search arm's and the second the
Pilot arm's. The third column is the Search arm's alone; the Pilot arm
stays at 0.2% there.

- **It goes on learning the search's answers, past where the smaller sets
  stopped.** 9.1% of the held-out deviations after 128 updates, 6.5 ± 0.6
  points above its control, and 14.6% of those the search took at a z of 3
  and more. Where the seed gave the search's answer no weight it is at 5.8%
  and still rising, where the two smaller sets level off at 4 to 5% and
  3,840 games stood at 5.5% after as many passes. Of the firmer half it
  gives 8.2%, against 5.3%, 6.1% and 6.5% on the smaller sets after eight
  passes. So more games do raise what is learned of the search's answers,
  slowly.
- **What its other answers are worth turned after the fourth pass.** The
  gap over its control is 0.8 and 1.2 points a game where it was 2.2: 1.5
  ± 0.6 less at update 96, 1.0 ± 0.7 less at 128. The Main Phase kept what
  it had: +0.6, +0.6 and +0.5 points a game at the three updates, against
  -0.9, -0.7 and -1.0 for the control. Chain windows lost it. The Search
  arm gives 823 other answers there where it gave 635, and their leads add
  up to -0.6 points a game where they came to -0.1. It does activate at
  more of the places where the search does, 7.8% of 384 against 4.2%, and
  more often where the search does not. It is the turn 3,840 games took
  after two passes, two passes later.
- **In deals won nothing shows.** +1.1 ± 1.1 at update 128, and +0.7 ± 0.7
  over the three updates together. The Search arm's 66.2% is its highest,
  1.4 points under the pilot.
- **By the letter of the readings.** The 5 points at the deviations that
  *learnable* asks for are passed at updates 96 and 128, at ten standard
  errors. Its second condition is not met: the Search arm gives another
  answer than the pilot's at 6.6% of the decisions the search examined and
  kept, 1.5 and 1.7 points more often than its control, where 1 was
  allowed. *It plays better* asks for 2.4 points in deals won.

On the games it trained on, the Search arm gives the search's answer at
27.6% of the deviations in the eighth pass, before each update, where it
gave 12.7% in the fourth: three times its rate on the held-out ones.

## What it comes to

Imitating the games the search plays, as they are, does not make a stronger
player than imitating the pilot's, on any number of games up to 7,680 and
any number of passes up to 32. A small part of the search's answers carries
over to games not seen, and the rest of what the model learns of them is the
games themselves.

The losses-only arm is a part of the same labels, and the run gives no
reason to expect more from it. More searched games and longer training on
the largest set are another matter than they looked. By the rate at the
deviations and by deals won the sets do not differ. By the search's worlds
the largest is the one on which the Search arm's other answers are worth
more than its control's, the more the longer it trained, where the smaller
sets lose what they had early as they go through their games again
([By update](#by-update)). That is one run of each arm, and 7,680 searched
games take six and a half hours. Trained twice as long, the largest set
learns more of the search's answers than any smaller one, and what its
other answers are worth turns down as theirs did, later
([The largest set trained on](#the-largest-set-trained-on)). More games
help, and by little. With eight times the games the gap lasts four passes
where it lasted under two, and reaches 2.2 points a game by a measure that
flatters; in deals won there is nothing to show.

Three things stand in the way, in this order:

- **What there is to learn from.** The 7,680 games hold 11,050 deviations,
  spread over some sixty kinds of change, and about half of them would be
  found again at most. What the learner takes from them looks like a
  tendency (it ends the turn more readily, for one) more than a knowledge
  of the positions it applies to: it follows one firmly decided deviation
  in nine.
- **The labels.** Half of them are the luck of the worlds, and the
  decisions where an alternative was ahead and not taken are labelled the
  other way. That accounts for little of the shortfall: the rate is 7.3%
  over all the deviations and 10.9% where they are firm.
- **The learner.** It does not yet reproduce the pilot, and its value head
  learns nothing from results.

## Current next step

The held-out run has been read by firmness ([The labels](#the-labels)); it
speaks against going straight to a large run of anything. The first five
points are done, all on 2026-10-07. With them the experiment's three
questions are answered and it is closed. The last two points are what is
tried in its place:
[Learning the search's leads](2026-10-07-search-leads.md).

1. **The pilot imitated to a real plateau: done, at version 89.**
   [`format13-imitation-seed.sh`](commands/format13-imitation-seed.sh) now
   goes on, 20 updates at a time and at most to version 249, while the last
   two evaluations are on average more than a point above the two before,
   and fails if the seed then stands more than 3 points under its pilot. By
   that rule the seed was not at a plateau at version 49 (its last two
   evaluations were 2.4 points above the two before). Run again on the CUDA
   machine on 2026-10-07 it trained 40 more updates in 21 minutes:

   | Version | 39 | 49 | 59 | 69 | 79 | 89 | Pilot |
   | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
   | Suite, 1,664 games | 60.1% | 59.4% | 62.4% | 62.3% | 62.6% | 63.0% | 64.9% |

   The last two evaluations are 0.4 points above the two before, and 2.1
   points under the pilot; the last four together are 2.3 under (1.8
   without the mirror match). The learner agrees with its pilot at 94.9% of
   the decisions of the last ten updates' games, against 94.0% in the ten
   before version 49, and the imitation loss is flat at 0.09. Imitation in
   this format ends a little under the pilot, as it did in the last one. The
   value head still predicts nothing: its explained variance on each
   update's new games averages -0.12 over the 90 updates and is never above
   0.004 (taken over every step: see
   [What the run shows](#what-the-run-shows)). The arms script keeps to the
   seed it measured, so the eight runs are not trained further.
2. **The seed's value head, measured where it is trained: done.**
   [`format13-value-check.sh`](commands/format13-value-check.sh) branches a
   run from the seed's last checkpoint and makes four imitation-only
   updates on the games the seed run would play next. Each reports the
   head's explained variance at the transactions' first steps before it
   optimizes; the seed run is not touched. From version 89: -0.10, then
   -0.07, -0.13 and -0.14 ([What the run shows](#what-the-run-shows)).
3. **A measure that reads a model's own answers by the search's worlds:
   done.** Built and run on 2026-10-07:
   [The models' answers by the search's worlds](#the-models-answers-by-the-searchs-worlds).
   It tells the Search arm on 7,680 games from its control at four standard
   errors, where deals won tell them apart at half of one.
4. **The same measure at the earlier updates: done.**

   ```sh
   POINTS="4 8 16 32 64" TRIED_OUT=analysis/search-imitation/tried-by-update \
     bash experiments/commands/search-imitation-tried.sh
   ```

   The finding in the Search arm's favour rested on a single run of each
   arm after its last update. A gap that builds up over the passes was to
   count as something learned, and one there at the last update only as
   nothing to build on. It builds up, on the largest set alone:
   [By update](#by-update).
5. **The largest set's two arms trained on: done**, to update 128
   ([The largest set trained on](#the-largest-set-trained-on)). Both were
   stopped while they were still gaining: in deals won, in the search's
   answers they give where the seed gave none (1.6%, 2.6% and 4.0% after
   updates 16, 32 and 64), and the Search arm in what its other answers are
   worth against its control's (1.1, 1.5 and 2.2 points a game). The 64
   updates were the frozen protocol's; what comes after is a look outside
   it, and no reading of [Readings](#readings-fixed-on-2026-10-06) applies
   to it.

   ```sh
   SIZES=7680 UPDATES=128 bash experiments/commands/search-imitation-arms.sh
   SIZES=7680 POINTS="64 96 128" TRIED_OUT=analysis/search-imitation/tried-longer \
     bash experiments/commands/search-imitation-tried.sh
   ```

   The first trains each of the two arms 64 updates further, to eight times
   through the set (about 25 minutes each), and leaves the other arms and
   their measurements as they are. The second measures them after updates
   64, 96 and 128 (16 minutes) and leaves
   `analysis/search-imitation/tried-longer/` to copy back. The 128 new
   checkpoints take 11 GB. A run that has only imitated may now be
   continued with a larger `--imitation-only-updates`; before, the arms'
   64 were fixed in their `run.json` and anything after them would have
   been PPO. It was to say whether the gap goes on growing or turns as it
   did on 3,840 games after two passes, whether the search's answers level
   off at the 4 to 5% of the smaller sets or above, and whether either
   shows in deals won. The gap turned, the answers go on rising above that
   level, and deals won show nothing. Going on to update 256 would follow
   the answers further and is not proposed: the gap has turned.
6. **Leads where the pilot's answer wins every first world.** The measure
   leaves out more than a third of a model's other answers, those that can
   only cost, because the search tries nothing there. An option of
   `policy-bench search` would play the alternatives through the first
   stage there too, for the log alone. The search takes the pilot's answer
   there either way and its worlds are seeded from the deal and the
   decision, so the games must come out with the digests they have, which
   `tools/compare_bench_rows.py` checks. In the held-out run that is
   641,024 playouts more on 2,979,976 (a fifth: 1 hour 45 minutes on the
   CUDA machine instead of 1 hour 26). A model's sum over the examined
   decisions is then complete, and no longer flattered. Built on 2026-10-07
   as `--log full` (ygo-policies `568711d`), for the next experiment.
7. **The search's measurements as the target, tried small.** For every
   decision the search examined, how the pilot's answer and each
   alternative did in the worlds, in place of the one answer played: 32
   decisions a game where there are 1.4 deviations, and 75 alternatives,
   graded, with nothing labelled the other way. It also gives the value
   head a target that is a mean over worlds. Whether a network learns those
   leads from the position is open, and the split by firmness says the
   labels' luck is not the main obstacle, so not at full size first. The
   1,920-game set again with `--log true` is the same deals as the two
   1,920-game arms, which then are its controls on the same held-out run
   (1.6 hours of search on the CUDA machine). The other 5,760 games (5
   hours) only if that shows. It would be a new experiment with its own
   record, read by the measure of point 3, which is sharp enough for a set
   that small: its standard error there is 0.6 to 0.7 points a game. What
   points 5 and 6 show goes into its design.

## What had to be there first

All done on 2026-10-06:

- The port of the duel environment to the ygo-policies tooling, with the
  learner's input built from the pilots' own view and imitation of
  pilot-played games working in it.
- `policy-bench search --record true` writes each duel of a row as a replay
  record, and `crates/ygo-env` plays a record again with the searching seat
  watched. Each of its decisions comes with the answer played and the
  pilot's own, and its `answer` is the `decision` of the row's deviations.
- The engine, the scripts and the card database brought up to date with
  their upstreams (ygo-policies `215f7d0`), so that the marks and the
  searched games are measured once. See the port in the
  [remedies record](2026-09-29-ppo-regression-remedies.md).
- The trainer reads such records as the games of imitation-only updates,
  and measures checkpoints on a held-out run. See
  [The tooling](#the-tooling).
- The faster search (ygo-policies `bc8c0f7`, the pin since). The engine,
  the scripts and the card database are untouched. The 192 plain and 16
  searched Blackwing games kept from the pin before come out identical
  (`tools/compare_bench_rows.py`): results, answers, deviations and
  digests. So does the trainer's small end-to-end run.
- The remedies experiment closed.

## The tooling

Written on 2026-10-06; the commands are in
[docs/commands.md](../docs/commands.md).

- **The games.** `policy-bench search --record true` writes one row a deal,
  with the pilot's game and the searched game as records. The same command
  at the same revisions writes the same rows, with or without `--log true`.
  The bench numbers a matchup's games from the run's seed on, so the
  training run and the held-out run need seeds further apart than their
  games per matchup, or they share deals. Proposed: 800000 to train on and
  900000 held out. 730000 is left to neither: the earlier searches ran on
  it, and the pilots' rules were then improved from their logs.
- **The arms.** `train --imitation-records ROWS` makes the imitation-only
  updates play the rows' records again and imitate the answers they hold:
  `--imitation-game search` for the Search arm, `baseline` for the Pilot
  arm. Both go through the same deals in the same updates. The games are
  taken in an order that does not depend on how the search run was
  scheduled, and `--imitation-games N` takes the first `N` of it: an equal
  share of every deck and seat, a smaller set within a larger. An update
  imitates the next `--duels` games of a shuffled pass.
- **Records and revisions.** A record is refused unless it plays again with
  the digest it was recorded with and leaves its pilot where its row says.
  A record made at other revisions can play again without an error and
  still show the network other inputs than the recorded duel had.
- **The measures.** `evaluate-search CHECKPOINTS --records HELDOUT` plays the
  held-out run's deals with the model, beside the pilot's and the search's
  outcomes on the same deals, and asks the model at every decision of the
  held-out searched duels: the answer key of [Measures](#measures), overall,
  by kind of decision and by what the pilot answered and the search played
  instead, with every deviation listed. A checkpoint is refused if its own
  run imitated any of the held-out run's deals.
- **During training**, each update reports how often the model's first
  choice at the deviations of its games is the search's answer or still the
  pilot's, measured before the update. In the first pass those games are
  new to it.
- **A model's answers by the search's worlds** (2026-10-07). A logged run
  holds, for every answer the search examined, how the pilot's answer and
  each alternative it tried did in each world. `evaluate-search` now reads
  the model's first choice by that at every examined decision: the pilot's
  answer, another where the pilot's won every world (nothing else was tried
  there), or an alternative the search tried, with the lead it measured for
  it in the worlds of its first stage, which every alternative goes through
  whatever comes of it. The same is given for the model's heaviest
  alternative taken at lower bars than first choice, for the search's own
  choices and for any other answer the search tried. The leads exist only
  where the pilot's answer lost some world, which flatters every
  alternative alike: a model is read beside another, not by the sign of its
  lead. The alternatives are listed again from the decision as the network
  is offered it, the way the bench lists them, and a log that names others
  is refused. On the 384 logged games of the development seed and the two
  the test makes with the bench, they are the logged ones at every examined
  decision (20,181 and 90). There the search's own choices lead by 23.3
  points and any other answer it tried by -5.8.

Checked on the 384 logged games of the search's speed-up (seed 730000),
which the trainer plays again with their digests at the pin: the answer key
is asked all 533 deviations, and every one of the 20,181 decisions the
search examined is a decision the network is asked, kind by kind (9,278 in
chain windows, 6,488 in the Main Phase, 3,404 in the Battle Phase, 605
battle positions, 406 yes/no questions). The network is asked 29,369
decisions in those games, 76 a game, so the search examines two thirds of
them. The answer key gives every kind of decision apart.

A check that the labels reach the model, on the CPU: both arms were trained
from scratch for 30 passes over the same 16 deals, whose searched games hold
31 deviations, and then asked about those searched games.

| Arm | Search's answer at the deviations | Pilot's | Another answer where the search kept the pilot's |
| --- | --- | --- | --- |
| Search | 54.8% | 32.3% | 3.5% |
| Pilot | 6.5% | 77.4% | 11.6% |

This is memorisation of 16 games and says nothing of games not seen. It
does show the deviations are learned last: after 20 passes the Search arm
gave the recorded answer at 89% of all decisions and at 19% of the
deviations.

## Later

- **Losses only**, the first proposal, as a third arm: the pilot's games of
  every deal plus the searched games of the deals the pilot lost. It is a
  subset of the same labels, and after this run there is no reason to
  expect more of it.
- **Labels on the learner's own positions**, with the search inside the
  trainer. In the remedies experiment imitating the teacher's own games did
  as well as that (58.8% against 60.0%), and the labels would be the same
  test's decisions.
- The worlds' results as targets: now the last point of
  [Current next step](#current-next-step).
- A second step of improvement, with the search on top of the learner.
- The multi-deck generalist (an earlier record, not included here) with the
  search as its teachers.
