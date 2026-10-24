# Self-play, further

**Status:** COMPLETED

**Proposed:** 2026-10-12

**Activated:** 2026-10-12

**Completed:** 2026-10-13

**Last updated:** 2026-10-13 (closed, with its three readings
met, for the two things it left open)

Chosen on 2026-10-12, before any run of it, when
[Self-play, and the named plays](2026-10-11-self-play.md) was
closed with every reading taken. The design and the readings were first
written that day in the next step of
[One network, several decks](2026-10-12-one-network-several-decks.md),
which was the active experiment, and were moved here as they stood when
that one was read and this record opened.

## Research questions

Both arms of the experiment on self-play go on from 960 updates to 1,920.

1. Does the learner that plays half of its duels against itself still
   keep what the start wins against the pilots?
2. Is it still the better player of the mirror than the same updates
   against the pilots alone make?
3. Has it gained on itself: does it beat its own checkpoint of 960
   updates before?

## Why

- **What 960 updates gave is large, and whether it goes on is not
  known.** With itself's last checkpoint wins 67.5% of the mirror against
  the control and leaves a learner that looks for its habits next to
  nothing. Its rate against the start in the mirror was still rising
  when the run ended (63.5% after 120 updates, 70% to 71% after 720 to
  960).
- **The price against the pilots rests on one pair of runs.** With itself
  ended 1.4 ± 0.9 points under the start on the held-out deals and 2.1 ±
  0.7 under the control. Twice the updates say whether that grows, stays
  or goes away. A second seed of both arms, which says whether it was one
  run's luck, was the other choice for the night and is left for another.
- **It needs nothing built but one measurement.** The arms go on from
  where they stand.

## Design (fixed on 2026-10-12)

The two arms as they were made
([the design before](2026-10-11-self-play.md#design-fixed-on-2026-10-11)),
each taken on from its checkpoint after 960 updates for 960 more:
`with-itself` with 260 of an update's 520 duels against itself and the
others against the thirteen pilots, `pilots-alone` with every duel against
the pilots; a learning rate of 1e-4, one pass over an update's games,
seed 2012, a checkpoint and the suite every 120 updates. The start is the
second long run's last checkpoint, as before.

Every measurement is made anew, in a folder of its own
(`analysis/self-play-further/`): the start and each arm's sixteen
checkpoints on the held-out games, the head to head, the decks outside
training and the plays.

## Readings (fixed on 2026-10-12)

Proposed and agreed before any run, along with the choice of this run
for the night over the other. They are taken by each arm's
checkpoints after 1,680, 1,800 and 1,920 updates, with the rules of the
experiment this goes on from:

1. *Self-play still keeps what it has against the pilots.* With itself's
   three checkpoints against the start on the 1,536 held-out deals, deal
   by deal: it has **lost** when it wins at least 2 points and two
   standard errors fewer.
2. *Self-play still makes the better player of the mirror.* With itself's
   three against Pilots alone's checkpoint after 1,920 updates, on the
   2,048 deals of the mirror, each in turn the one held: 12,288 games.
   **Better** at 52% of the games and two standard errors over half,
   **worse** at 48% and two under.
3. *With itself has gained on itself.* Its three against its own
   checkpoint after 960 updates, the same way: 12,288 games. **Gained**
   at 52% and two standard errors over half, **lost** at 48% and two
   under, and between the two not shown.

Told beside them without a mark: Pilots alone against the start and With
itself against Pilots alone on the held-out deals, which is where the two
points between the arms are seen to grow or not; the decks outside
training; the named plays.

## The run

From the repository root on the CUDA machine, with nothing else running
there:

```sh
ARM=1920 SINCE=960 OUT=analysis/self-play-further PARTS=arms bash experiments/commands/self-play.sh
```

About twelve hours and a quarter by what the arms took before: 6.9 for
With itself's 960 more updates, 4.1 for Pilots alone's, and 1.3 for the
measurements, of which the held-out games of 33 checkpoints are 44
minutes. A run or a measurement that is there is not made again, and an
interrupted run goes on. `analysis/self-play-further/` is what is copied
back, and it is read with

```sh
uv run python tools/self_play_report.py analysis/self-play-further --arm 1920 --since 960
```

The pin of `native/ygo-policies` stays at `1b85b63` while this is open:
the held-out deals are logged games of that revision.

## The arms as they came out (2026-10-13)

Made on the CUDA machine from the evening of 2026-10-12 to 08:48 on
2026-10-13, and read from `analysis/self-play-further/` as it was copied
back that morning. The report made there is the one made here from the
same files, and the figures of the three readings were worked out a second
time from each deal's result, apart from the report, and are the same.

An arm after so many updates, as
[before](2026-10-11-self-play.md#the-arms-as-they-came-out-2026-10-12):
what it wins of the 1,536 held-out deals, of the suite against the pilots
and of the mirror against the start held in the other seat. The start
wins 81.4%, 79.4% and 48.2%. The row of 960 updates is where the first
run ended.

| Updates | With itself: held-out | the pilots | the start | Pilots alone: held-out | the pilots | the start |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 960 | 79.9% | 80.3% | 70.4% | 82.4% | 81.4% | 54.0% |
| 1,080 | 79.3% | 80.1% | 73.0% | 81.4% | 80.4% | 52.8% |
| 1,200 | 78.4% | 80.3% | 71.0% | 81.9% | 81.9% | 53.6% |
| 1,320 | 81.2% | 79.0% | 71.1% | 83.2% | 81.5% | 57.7% |
| 1,440 | 82.3% | 81.2% | 73.4% | 83.9% | 80.9% | 55.8% |
| 1,560 | 80.9% | 79.9% | 72.7% | 82.7% | 81.4% | 50.3% |
| 1,680 | 80.5% | 78.8% | 71.1% | 84.6% | 82.6% | 52.4% |
| 1,800 | 81.0% | 81.1% | 73.1% | 83.1% | 82.3% | 53.6% |
| 1,920 | 81.0% | 81.7% | 72.5% | 83.3% | 82.6% | 52.9% |

- **The first reading: it still keeps what the start wins against the
  pilots, and stands nearer the start than it did.** With itself's
  checkpoints after 1,680, 1,800 and 1,920 updates win 80.8% of the
  held-out deals, 0.5 ± 0.9 fewer than the start deal by deal, where 2
  points and two standard errors fewer would have said it had lost them.
  After 960 updates it was 1.4 ± 0.9 fewer.
- **Beside it: the two points between the arms have not gone.** Pilots
  alone wins 83.7% of the held-out deals, 2.3 ± 0.9 more than the start
  (0.7 ± 0.8 after 960), and With itself 2.8 ± 0.7 fewer than Pilots
  alone (2.1 ± 0.7 after 960). On the decks outside training (2,304
  games) the start wins 73.3%, With itself 72.8% (0.5 ± 0.7 fewer) and
  Pilots alone 74.3% (1.0 ± 0.7 more): With itself 1.5 ± 0.5 fewer than
  Pilots alone (2.3 ± 0.5 after 960). Of the held-out deals With itself
  wins 0.9 points more than after 960 updates and Pilots alone 1.6 more,
  and what lies between them is two to three points on the decks of
  training and one to two outside them. Neither of those is told apart
  from its own figure of 960 updates before.
- **The second reading: it is the better player of the mirror, by more
  than before.** With itself's three checkpoints win 71.2% ± 0.5 of the
  12,288 games against Pilots alone's last checkpoint, every deal from
  either seat, where the three after 960 updates won 67.5% ± 0.5 against
  that arm's last of then: 79.0% of the games it opens and 63.5% of the
  others, and each of the three between 70.1% and 73.3% whichever of the
  two is held. The mark was 52% and two standard errors over half.
- **The third reading: it has gained on itself, a little and surely.**
  The three win 52.7% ± 0.4 of the 12,288 games against With itself's own
  checkpoint after 960 updates, every deal from either seat: 52.7%, 52.3%
  and 53.2% each. The mark was 52% and two standard errors over half, and
  it is six over. Against the start in the mirror it wins 71.1%, 73.1%
  and 72.5% of the deals at its last three evaluations, where it won
  70.9%, 71.4% and 70.4% after 720 to 960; Pilots alone 52.4%, 53.6% and
  52.9%, as before.
- **Where the points against the pilots lie** (the suite at an arm's last
  three evaluations, 128 deals a deck; the start by its one):

  | Against the pilot of | The start | With itself | Pilots alone | With itself less Pilots alone | After 960 |
  | --- | ---: | ---: | ---: | ---: | ---: |
  | `blackwing` | 80.5% | 85.7% | 83.3% | +2.3 | +3.9 |
  | `burn` | 93.8% | 93.0% | 96.9% | -3.9 | -0.3 |
  | `crystal` | 86.7% | 83.6% | 84.6% | -1.0 | -7.0 |
  | `dragunity` | 85.2% | 81.0% | 86.7% | -5.7 | -8.3 |
  | `gishki` | 88.3% | 88.3% | 91.4% | -3.1 | -2.1 |
  | `gladiator` | 76.6% | 78.9% | 81.8% | -2.9 | +1.0 |
  | `heroes` | 64.1% | 68.5% | 65.1% | +3.4 | +4.2 |
  | `infernity` | 66.4% | 70.6% | 70.3% | +0.3 | -4.4 |
  | `lightsworn` | 59.4% | 61.2% | 65.1% | -3.9 | -2.6 |
  | `monarch` | 72.7% | 74.0% | 77.6% | -3.6 | -4.9 |
  | `morphtronic` | 83.6% | 83.9% | 87.2% | -3.4 | -1.0 |
  | `rock-block` | 80.5% | 82.0% | 84.9% | -2.9 | +0.8 |
  | `spellcaster` | 94.5% | 96.4% | 97.4% | -1.0 | 0.0 |
  | All thirteen | 79.4% | 80.5% | 82.5% | -2.0 | -1.6 |

  With itself still wins more against the Blackwing pilot and against
  `heroes`, and fewer against most of the others by three or four points
  each. What stood out after 960 updates, `crystal` and `dragunity`, is
  smaller now, and the rest is spread. A deck has 128 deals, so a figure
  there is good to three or four points.
- **In its own games against itself the player who opens wins 58% to
  59%**, as from the first updates, and a game takes 15 to 16 turns where
  it took 14.2 at the end of the first run.

### The named plays, after twice the updates (without a mark)

`tools/named_plays.py`, as before: the same 2,240 games against pilots and
120 against a player who does nothing, by the start and by each arm's
checkpoint after 1,920 updates.

| | The start | With itself | Pilots alone |
| --- | ---: | ---: | ---: |
| Games won, of 2,240 | 77% | 79% | 79% |
| ... against the Blackwing pilot, of 160 | 74% | 86% | 81% |
| ... against the Machina pilot, of 160 | 42% | 43% | 41% |
| Allure of Darkness, turns with only Spells and Traps beside it: activated | 38 of 57 | 15 of 16 | 12 of 14 |
| ... with cards still to set | 35 | 15 | 11 |
| Activated with no DARK monster in hand: the hand lost | 6 of 70 | 9 of 71 | 16 of 76 |
| ... a hand with other cards in it | 3 | 1 | 3 |
| Trap Hole at a window that offers it (its pilot) | 74% (89%) | 76% (90%) | 70% (91%) |
| ... a monster of 1,000 to 1,499 ATK | 57% | 54% | 54% |
| ... at the first window of a game | 83% | 83% | 76% |
| Bottomless Trap Hole at a window that offers it | 68% | 68% | 59% |
| Dimensional Prison at a window that offers it | 74% | 72% | 61% |
| Threatening Roar: activations | 601 | 639 | 781 |
| ... at their Battle Phase, monsters of theirs in Attack Position | 456 of 6,402 | 394 of 6,281 | 486 of 4,694 |
| ... in their Main Phase 1 (of those, answering a card on the chain) | 144 (143) | 227 (206) | 219 (191) |
| ... in their Draw Phase | 0 of 6,515 | 13 of 6,371 | 75 of 5,695 |
| ... spent before they had an attacker, answering nothing | 0 | 25 | 76 |
| Against a player who does nothing: won, in turns on the average | 120 of 120, 10.6 | 120 of 120, 10.6 | 120 of 120, 9.9 |

- **Threatening Roar spent early has nearly gone again in With itself,
  and halved in Pilots alone.** After 960 updates both arms spent it
  before the other player had a monster to attack with, answering
  nothing, 110 and 164 times in the 2,240 games, which the start never
  did. After 1,920 it is 25 and 76 times, and in the other player's Draw
  Phase 13 and 75 times where it was 116 and 164. The habit came within
  960 updates in both arms and went on its way out in the next 960,
  further where half of the duels are against itself.
- **Trap Hole and Allure of Darkness are played as before.** Trap Hole
  is taken at 76% and 70% of the windows that offer it (the start 74%,
  its pilot 90% and 91%), at the first window of a game 83% and 76% of
  the times. The turn with only Spells and Traps beside Allure of
  Darkness comes 16 and 14 times where the start had 57, and the card is
  then activated in 15 and 12 of them, in 15 and 11 with cards still to
  set.
- **Both arms win 79% of these games** where the start wins 77%, and With
  itself 86% of the 160 against the Blackwing pilot (the start 74%,
  Pilots alone 81%).

**Time.** Twelve hours and a half, where twelve and a quarter were
reckoned: 7.06 hours for With itself's 960 more updates (25.9 s an update
of 81,200 decisions, where the first 960 took 25.3 s of 79,800), 4.08 for
Pilots alone's (14.7 s of 47,400), and 1.3 for the measurements.

## What it says

1. *Twice the updates undid nothing.* Half of each update's duels against
   itself, for 1,920 updates on from the strongest checkpoint, still
   keeps what that checkpoint wins against the pilots, by the mark fixed,
   and stands half a point under it where it stood a point and a half.
2. *It goes on making the better player of the mirror.* 71.2% against the
   control where it was 67.5%, and 52.7% against its own self of 960
   updates before: a small gain, and six standard errors of it.
3. *The two points against the pilots beside the control are still
   there.* 2.8 ± 0.7 on the held-out deals and 1.5 ± 0.5 on the decks
   outside training, where they were 2.1 ± 0.7 and 2.3 ± 0.5. Twice the
   updates neither closed nor widened it past what one pair of runs
   tells. It is still one pair of runs.

What it does not say: what an exploiter finds in the checkpoints after
1,920 updates, which was no part of this run; whether a second seed of
both arms gives the same two points; and how either fares in recorded
duels.

## The tooling

- `experiments/commands/self-play.sh` takes an arm on from where it
  stands when `ARM` is longer than it has run, as it takes on a run that
  was interrupted. `SINCE` adds With itself's last three checkpoints with
  its own checkpoint after so many updates, each in turn the one held
  (`head-to-head/with-itself.NNNN.since.json` and `.back.json`).
- `tools/self_play_report.py --since` reads those by the marks of the
  mirror, and refuses a pair that is not With itself's checkpoint of that
  length with its own earlier one.
- Tried on the development machine at a small size on 2026-10-12: arms of
  6 updates taken on to 12, to the end. Both kept their learning rate and
  With itself its duels against itself.

## The trainer since the run (2026-10-13)

Everything above was run and measured before these, on the CUDA machine.

- **Three changes leave an update as it is and shorten it** (`c604aa06`):
  a refresh of the field decoded once for both seats, a transaction's
  members given their advantage in one read, the next minibatch built
  while the card works on this one. An arm taken on with them is the arm
  it would have been.
- **The LSTM is given a packed batch**, which a card works out another
  way, in a tenth of the time: an update against the pilots is 2.41 ±
  0.14 s shorter of 13.94 s. On a card that moves what the LSTM gives in
  the third decimal, as far as the card's reduced precision already has
  it from exact arithmetic, and a checkpoint's score of an answer by
  about a thousandth. **An arm taken on after it is another run than it
  would have been, and a checkpoint measured on the card after it is not
  asked to the digit as it was for the readings above.** On a processor
  nothing changes. **A measurement on the card came out the same**: the
  screen's 1,664 games by the first choice of With itself's checkpoint
  after 960 updates, played both ways there, are the same games (1,336
  wins, none ended otherwise).

- **On a card the model is asked another way in an update's games**
  (decided the same day). Its calls of up to half the pool's
  duels are answered at sizes fixed in advance, from CUDA graphs, and a
  thread's duels go in two groups, one taken forward while the model
  answers the other. A run alone with half of its duels against itself
  takes 15.5 s an update where it took 18.6 s, and three such runs at
  once through NVIDIA's MPS 23.0 s each: twice the duels an hour of the
  one. Filled out to fixed sizes a decision's scores differ in their last
  digits from those of its own batch (up to 1.3e-5 where measured), and a
  draw within that of its boundary goes the other way: now and then
  another game of the same model, which the runs measured did not have
  in seven updates. **An arm taken on with it is another run than it
  would have been**, as it already was after the packed batch. **The
  measurements are not touched**: a suite, the held-out games and a
  run's own evaluations are played with every call as it is and every
  duel together. A run says how its model was asked (`model_calls` in
  its `run.json`), and `--inference-graphs 0 --pool-groups 1` is the
  trainer as it was.
- **A script's trainings can go at once** (`AT_ONCE` of
  `experiments/commands/self-play.sh`, through an MPS server on a card):
  each run is then half as long again as alone, with three.

`docs/commands.md` has the figures, in its parts on the three changes of
2026-10-13, the LSTM given a packed batch, calls from CUDA graphs and
the duels in two groups, and an experiment's trainings at once.

## What follows

Closed on 2026-10-13, the day the run was read, with its three readings
met. What it left open when it began is the experiment after it,
[Self-play, a second seed](2026-10-13-self-play-second-seed.md):
a second seed of both arms, and exploiters of the arms' checkpoints after
1,920 updates, which are measured in this experiment's folder
(`analysis/self-play-further`).

The checkpoints after 1,920 updates are on the CUDA machine as
`runs/self-play/with-itself/checkpoints/model-04199.pt` and the same
under `pilots-alone`.
