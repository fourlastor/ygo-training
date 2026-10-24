# Self-play, a second seed

**Status:** COMPLETED

**Proposed:** 2026-10-13

**Activated:** 2026-10-13

**Completed:** 2026-10-14

**Last updated:** 2026-10-14 (closed the day it was read: the second pair
lost against the pilots by the mark, is the better player of the mirror
and is under the control; after 1,920 updates self-play leaves an
exploiter nothing)

Chosen and fixed on 2026-10-13, before any run of it, when
[Self-play, further](2026-10-12-self-play-further.md) was closed with its
three readings met. It makes the two things that experiment left open when
it began: a second seed of both arms, and exploiters of the arms'
checkpoints after 1,920 updates.

## Research questions

1. Does a second pair of arms, of another training seed, give what the
   first gave after 960 updates: does the learner that plays half of its
   duels against itself keep what the start wins against the pilots, and
   is it the better player of the mirror?
2. Is it under the control against the pilots in the second pair too?
3. What does a learner that looks for its habits find in the first
   pair's checkpoints after 1,920 updates: does self-play still leave it
   less than the control does?

## Why

- **The price against the pilots rests on one pair of runs.** With
  itself was 2.1 ± 0.7 points under Pilots alone on the held-out deals
  after 960 updates and 2.8 ± 0.7 after 1,920, which is the same pair
  measured twice. Three seeds of one design of PPO lay a point or two
  apart after 120 updates
  ([PPO with a head that predicts](2026-10-09-ppo-with-a-head-that-predicts.md)):
  a run has that much luck of its own, and two points between two runs
  are no more than two runs of one design can differ by.
- **What 960 more updates did to what an exploiter finds was no part of
  the run.** After 960 a learner that looks for its victim's habits won
  51.1% of its games against With itself's checkpoint and 67.0% against
  the control's.
- **They are runs that do not wait for each other**, and since
  2026-10-13 the CUDA machine makes such runs at the same time through
  NVIDIA's MPS: three at once play twice the duels an hour of one.

## Design (fixed on 2026-10-13)

**The second pair.** The two arms as the first pair was made
([the design](2026-10-11-self-play.md#design-fixed-on-2026-10-11)),
from the same start, the second long run's last checkpoint
(`runs/ppo-long-rate/one-pass-long/checkpoints/model-02279.pt`), 960
updates each: `with-itself` with 260 of an update's 520 duels against
itself and the others against the thirteen pilots, `pilots-alone` with
every duel against the pilots; a learning rate of 1e-4, one pass over an
update's games, a checkpoint and the suite every 120 updates. **The
training seed is 2013** where the first pair's was 2012: other deals and
other draws. The suite, the held-out deals and the mirror's deals are
those the first pair was measured on. The pair has folders of its own,
`runs/self-play-2013` and `analysis/self-play-2013`, and is measured as
the first pair was after 960 updates: the start and each arm's eight
checkpoints on the held-out games, the head to head, the decks outside
training and the named plays.

**The exploiters.** Two exploiters, of the training seeds 2012 and 2013,
of each arm's checkpoint after 1,920 updates, of the first pair
(`runs/self-play/with-itself/checkpoints/model-04199.pt` and the same
under `pilots-alone`): a learner that starts as its victim and plays
every duel of 240 updates against it held as it is, with a checkpoint
every 30 updates, and its last three checkpoints with its victim on the
2,048 deals of the mirror, each in turn the one held. As the exploiters
of the checkpoints after 960 updates were made. Their runs are beside
those (`runs/self-play/exploiter-ARM-SEED.1920`), and they are measured
in the folder of the arms' 1,920 updates, `analysis/self-play-further`.

**The trainer is not the one the first pair was made with.** Since
2026-10-13 it gives the LSTM a packed batch and, on a card, answers the
calls of an update's games from CUDA graphs with the duels in two groups
([the record before](2026-10-12-self-play-further.md#the-trainer-since-the-run-2026-10-13)).
Both arms of the second pair are made with it, and all four exploiters,
so that what is read within the pair and between the arms' exploiters is
of one trainer. What the second pair wins, set beside what the first
won, is of another seed and of another trainer at once. The measurements
ask their model as they did.

**At once.** The two arms go at the same time through MPS, and the
exploiters three at a time (`AT_ONCE` of the script). A run then takes
longer than with the card to itself; the runs measured side by side
through MPS played the games they play alone.

## Readings (fixed on 2026-10-13)

Fixed before any run. The first three are taken by each arm's checkpoints
after 720, 840 and 960 updates, of the second pair; the fourth by each
exploiter's checkpoints after 180, 210 and 240.

1. *Self-play keeps what it has against the pilots, in the second pair
   too.* With itself's three checkpoints against the start on the 1,536
   held-out deals, deal by deal: it has **lost** when it wins at least 2
   points and two standard errors fewer. The rule of the first
   experiment, by which the first pair kept it (1.4 ± 0.9 under).
2. *Self-play makes the better player of the mirror, in the second pair
   too.* With itself's three against Pilots alone's checkpoint after 960
   updates, of the same seed, on the 2,048 deals of the mirror, each in
   turn the one held: 12,288 games. **Better** at 52% of the games and
   two standard errors over half, **worse** at 48% and two under. The
   rule of the first experiment (the first pair: 67.5% ± 0.5).
3. *With itself is under the control against the pilots.* New. With
   itself's three checkpoints against Pilots alone's three, of the same
   seed, on the 1,536 held-out deals, deal by deal: **under** when it
   wins at least 1 point and two standard errors fewer, **over** when at
   least 1 point and two standard errors more, and between the two not
   shown. The first pair after 960 updates, 2.1 ± 0.7 under, is under by
   this rule.
4. *Self-play still leaves less for an exploiter.* Two exploiters of
   each arm's checkpoint after 1,920 updates, each by its last three
   checkpoints against its victim, every deal from either seat: **less**
   when With itself's win at least 3 points and two standard errors
   fewer of their games than Pilots alone's of theirs, and each of its
   two fewer than each of the other two; **more** when the same the
   other way. The rule of the first experiment (after 960 updates: 51.1%
   and 67.0%, less).

Told beside them without a mark: the decks outside training; the named
plays; and what the start's exploiters found, against each arm's
checkpoint after 1,920 updates.

What the third would say. *Under* in the second pair too: the two points
are the design's and not one run's luck, since two pairs of two runs
agree. *Not shown*: what the first pair has is within what a run's luck
gives, and is not to be counted on either way. *Over*: it was the first
pair's luck.

## The run

From the repository root on the CUDA machine, with nothing else running
there:

```sh
bash experiments/commands/self-play-second-seed.sh
```

It makes the second pair with its measurements and then the exploiters
with theirs, each by `experiments/commands/self-play.sh`:

```sh
AT_ONCE=2 ARM_SEED=2013 RUNS=runs/self-play-2013 OUT=analysis/self-play-2013 PARTS=arms bash experiments/commands/self-play.sh
AT_ONCE=3 SINCE=960 ARM=1920 OUT=analysis/self-play-further PARTS=exploiters bash experiments/commands/self-play.sh
```

About ten hours by reckoning, six for the pair and four for the
exploiters. With the trainer as it now is, a run with half of its duels
against itself was timed alone and three at once (15.5 s and 23.0 s an
update); two at once, a run against the pilots alone and an exploiter
were not. The exploiters are made whatever came of the pair, and what did
not come to its end is told at the end; started again, the script takes
its runs on from where they stand and skips what is there. A run's
output is in a log beside it (`runs/self-play-2013/with-itself.log`), and
its `run.json` says how its model was asked (`model_calls`: calls of up
to 160 decisions from graphs, two groups that go on together from 64).

`analysis/self-play-2013` and `analysis/self-play-further` are what is
copied back, and they are read with

```sh
uv run python tools/self_play_report.py analysis/self-play-2013
uv run python tools/self_play_report.py analysis/self-play-further --arm 1920 --since 960
```

The pin of `native/ygo-policies` stays at `1b85b63` while this is open:
the held-out deals are logged games of that revision.

## The second pair as it came out (2026-10-14)

Made on the CUDA machine from 19:08 on 2026-10-13, both arms at once
through MPS, and measured until 00:49 on 2026-10-14. Read on 2026-10-14
from `analysis/self-play-2013` there. The report made there is the one
made here from the same files, and the figures of the three readings were
worked out a second time from each deal's result, apart from the report,
and are the same.

An arm after so many updates: what it wins of the 1,536 held-out deals,
of the suite against the pilots and of the mirror against the start held
in the other seat. The start wins 81.4%, 79.4% and 48.2%.

| Updates | With itself: held-out | the pilots | the start | Pilots alone: held-out | the pilots | the start |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 120 | 79.2% | 77.3% | 62.1% | 81.0% | 80.6% | 51.2% |
| 240 | 76.3% | 76.2% | 64.3% | 80.1% | 80.2% | 45.9% |
| 360 | 77.5% | 76.9% | 67.3% | 82.6% | 79.2% | 47.8% |
| 480 | 77.5% | 78.1% | 68.9% | 82.0% | 80.7% | 51.2% |
| 600 | 78.0% | 77.9% | 68.7% | 81.9% | 80.5% | 50.1% |
| 720 | 78.5% | 78.7% | 67.9% | 81.5% | 80.9% | 51.1% |
| 840 | 77.6% | 78.4% | 70.5% | 81.9% | 81.9% | 51.6% |
| 960 | 80.1% | 78.6% | 70.6% | 81.7% | 81.3% | 54.3% |

The readings, by each arm's checkpoints after 720, 840 and 960 updates:

1. **With itself has lost against the pilots, by the mark fixed.** It
   wins 78.7% of the held-out deals, 2.6 ± 0.9 points fewer than the
   start, deal by deal: at least 2 points and two standard errors. The
   first pair after as many updates was 1.4 ± 0.9 under, within the mark.
2. **It is the better player of the mirror**: 67.3% ± 0.5 of 12,288 games
   against Pilots alone, every deal from either seat (75.0% of the deals
   it opens, 59.6% of the others). The first pair: 67.5% ± 0.5.
3. **It is under the control against the pilots**: 3.0 ± 0.7 points fewer
   of the held-out deals than Pilots alone, deal by deal. The first pair
   was 2.1 ± 0.7 under after 960 updates and 2.8 ± 0.7 after 1,920.

Told beside them: Pilots alone wins 81.7% of the held-out deals, 0.3 ±
0.8 over the start. On the decks outside training (2,304 games) the start
wins 73.3%, With itself 71.5% (1.8 ± 0.7 under the start) and Pilots alone
73.9%: With itself is 2.3 ± 0.6 under it there, where the first pair was
2.3 ± 0.5 under after 960 updates.

**The named plays** (`tools/named_plays.py`, the same 2,240 games and
counts as for the first pair):

- **Threatening Roar spent before the other player had an attacker,
  answering nothing: 2 and 0 times**, where the first pair's arms had come
  to it 110 and 164 times within their first 960 updates and the start
  never did. The habit was that pair's own, and not what training of
  either kind leads to.
- **Allure of Darkness on a turn with only Spells and Traps beside it**:
  activated 7 times of 26 by With itself and 15 of 27 by Pilots alone,
  where the start activates it 38 times of 57.
- **Trap Hole** is taken at 78% and 80% of the windows that offer it (the
  start 74%; its pilot 90% and 92%), at the first window of a game 87% and
  86% of the times.
- All three win every game against a player who does nothing, and With
  itself wins 83% of the 160 against the Blackwing pilot (the start 74%,
  Pilots alone 77%).

## The exploiters after 1,920 updates as they came out (2026-10-14)

Made on the CUDA machine from 00:49 to 03:56 on 2026-10-14, three at once
through MPS, and measured until 04:28. Read from
`analysis/self-play-further` there; the figures of the fourth reading
were worked out a second time from each deal's result, and are the same.

An exploiter's checkpoints after 180, 210 and 240 updates with its
victim, each in turn the one held: 12,288 games an exploiter, every deal
from either seat. Beside them, the exploiters of the checkpoints after
960 updates ([the first experiment](2026-10-11-self-play.md#the-exploiters-of-the-arms-as-they-came-out-2026-10-12)).

| Exploiters of | Seed 2012 | Seed 2013 | Together | After 960 updates |
| --- | ---: | ---: | ---: | ---: |
| With itself's checkpoint after 1,920 updates | 51.3% | 50.3% | 50.8% | 51.1% |
| Pilots alone's checkpoint after 1,920 updates | 68.5% | 69.4% | 69.0% | 67.0% |
| the start, as made before and measured again | 69.3% | 68.3% | 68.8% | 68.8% |

4. **Self-play still leaves less for an exploiter, and nothing.** With
   itself's exploiters win 50.8% of their games and Pilots alone's 69.0%:
   18.2 ± 0.5 fewer, deal by deal, and each of its two fewer than each of
   the other's. A learner that spends 240 updates, 124,800 games, looking
   for With itself's habits ends 0.8 points over an even game with it,
   and at every evaluation it won between 48.6% and 52.5% of the deals
   against it (a deal from one seat).

What the start's exploiters found, against each arm's checkpoint after
1,920 updates, every deal from either seat:

| Against | The start | Its exploiters | Over the start | After 960 updates |
| --- | ---: | ---: | ---: | ---: |
| the start itself | 50.0% | 69.4% ± 0.5 | +19.4 ± 0.5 | +19.4 |
| With itself | 27.5% ± 0.6 | 38.1% ± 0.5 | +10.6 ± 0.8 | +11.9 |
| Pilots alone | 46.1% ± 0.7 | 66.7% ± 0.5 | +20.6 ± 0.8 | +19.8 |

What an exploiter found in the start is still whole in Pilots alone after
1,920 updates against the pilots, and still does not beat With itself,
which beats the start in 72.5% of their games.

**Time.** Nine hours and twenty minutes for both, where ten were
reckoned. With itself took 4.84 hours (17.6 s an update, most of it
beside the other arm) and Pilots alone 3.08 (10.9 s); the exploiters of
With itself 2.01 and 2.02 hours and those of Pilots alone 1.80 and 1.32,
the last mostly alone. Every call of the games was answered from graphs
and none failed, and the MPS daemons' logs name 4 and 8 clients.

## What it says

1. *The price against the pilots is self-play's, not one run's.* Two pairs
   of runs now put the learner that plays half of its duels against
   itself 2 to 3 points under the same updates against the pilots alone,
   on the held-out deals (2.1, 3.0, and 2.8 after twice the updates) and
   on the decks outside training (2.3, 2.3, 1.5). Whether it then stays
   within 2 points of its start is the seed's: the first pair did and
   the second did not.
2. *The mirror is the same in both pairs*: 67.5% and 67.3% against the
   control.
3. *It leaves an exploiter nothing, and twice the updates did not change
   that*: 50.8% where the control's exploiters win 69.0%.
4. *A habit seen in one pair of runs need not be the design's*: Threatening
   Roar spent early came within 960 updates in both arms of the first
   pair, and in neither of the second.

The second pair was made by another trainer than the first (the packed
LSTM, the calls from graphs): within it the readings stand; set beside
the first pair, the seed and the trainer differ at once.

What it does not say: why the duels against itself cost against the pilots
(they are half of an update's duels, so the learner meets the thirteen
pilots half as often); whether a smaller share against itself keeps the
mirror and costs less; and how these checkpoints fare in recorded duels
against them (four duels against With itself's checkpoint after 960
updates, which the network won three of, are in the first experiment's
record).

## The tooling

- `experiments/commands/self-play-second-seed.sh` is the two commands
  above, one after the other.
- `experiments/commands/self-play.sh` takes `ARM_SEED` (both arms of
  another training seed, refused in the folder of another seed's arms),
  `AT_ONCE` (so many of a part's trainings at the same time, through an
  MPS daemon the script starts and tells to quit:
  `experiments/commands/at-once.sh`), and makes the exploiters of the
  arms' checkpoints after a longer arm in run folders of their own, where
  it would have taken on those of the first 960 updates.
- `tools/self_play_report.py` takes the third reading where it has both
  arms on the held-out deals
  (`Against the pilots, With itself is under Pilots alone`, the mark
  `UNDER`), and knows an exploiter in a folder named by its arm's length.
- Tried on the development machine at a small size on 2026-10-13, on the
  processor: arms of a second seed two at once; arms taken on and the
  exploiters of the longer arms three at once, beside those of the
  shorter; each again, with nothing redone. The script of both was held
  to those two commands with a stand-in for the inner one. What a card
  does where a run names nothing, graphs and two groups, cannot be run
  on that machine: it is the way measured on the CUDA machine with the
  options named.

## What follows

Run and read on 2026-10-14, the morning after it was started, and closed
the same day with every reading taken. Three came out as in the first pair
(the mirror, the price against the control, nothing left for an exploiter)
and the fourth did not (the second pair lost against the pilots by the
mark, where the first had kept them).

What it leaves open: why the duels against itself cost against the pilots,
which a smaller share of them, or a control that meets the pilots as
seldom, would tell; and how the checkpoints after 1,920 updates and those
of the second pair fare in recorded duels against them. The duels recorded
against the arms are four against With itself's checkpoint after 960
updates on 2026-10-12, of which the network won three
([read then](2026-10-11-self-play.md#four-recorded-duels-against-with-itself-2026-10-12)).
The experiment chosen on 2026-10-13 to follow it is one network for more
decks with half of its duels against itself
(`experiments/commands/more-decks.sh`), its design and readings fixed that
day.
