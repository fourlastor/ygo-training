# A critic that sees the hidden cards, in training only

**Status:** COMPLETED

**Proposed:** 2026-10-22

**Activated:** 2026-10-22

**Closed:** 2026-10-22, once the third phase was read: the second setting is
not run.

**Last updated:** 2026-10-22 (the third phase read: not told apart from
`all71` at R1, +0.1 ± 0.2 points; closed)

Proposed on 2026-10-22, when [The signal of an update](2026-10-22-update-signal.md)
had shown that an update of the mix of 71 decks is limited by noise (its
noise scale some 20,000 duels, an update of 520 about 2.5% signal), that
neither pairing deals nor taking a duel's result in full gives a cleaner
step, and that the more of an advantage is the value head's estimate from
one transaction to the next, the more consistent the step: the credit the
critic gives carries the signal. The value head explains 0.19 of a result
at a seat's first decision, where the deal holds 0.58: most of the gap is
the cards the seat cannot see. A critic that sees them, in training only,
gives credit to what the players do rather than to what is drawn or
revealed, which it already knows. Crossing the input's rule for it was
accepted on two conditions: the critic is used in training only, and the
model plays without seeing the hidden cards. **The plan and its readings
were fixed on 2026-10-22, before anything was built**.

## Research questions

1. How much more of a duel's result does a critic that sees the hidden
   cards explain than the policy's value head?
2. Does it make an update's signal stronger: more signal a duel in the
   step its advantages give?
3. If it does, does PPO with it learn faster per duel? (Its run is designed
   after the first two.)

## Design (fixed on 2026-10-22)

- **A network of its own**, separate from the policy, used only inside the
  training update. Its input is two packings of the same decision: the
  seat's view exactly as the policy has it, its history included, and the
  same position with nothing hidden, the opponent's hand and every
  face-down card shown. Seeing both, it knows what the seat knows as well as
  the truth, which keeps the policy's gradient unbiased (Baisero and Amato,
  2022: a critic of the state alone can be biased under partial
  observability, one of the history and the state is not).
- **Where the hidden view comes from**: built in `crates/ygo-env` at each
  decision the learner is asked, from a second projection given the
  engine's messages unredacted. `native/ygo-model-input`, which other
  projects share, is not changed.
- **Never at play**: it is not in what is exported for play; a test holds
  that changing the hidden cards leaves the policy's answers as they are;
  the exception is written into the input's rule in `AGENTS.md`.
- **Two versions**, measured before one is chosen: the hidden hands and
  face-down cards; the same and the order of the next cards of each deck,
  which makes the draws known too.

Phases:

1. **Fitted alone**: the critic trained on `all71`'s own games (its last
   checkpoint, R3 of [Seventy-one decks](2026-10-19-seventy-one-decks.md)),
   about 40 updates' worth (some 20,000 duels), the policy as it is, and
   read on games it never trained on against the policy's value head.
2. **The update's signal**: the noise scale of an update's step as
   [The signal of an update](2026-10-22-update-signal.md) measured it, the
   advantages from the critic.
3. **A run**, only where the second phase shows a stronger signal: PPO with
   the critic against PPO with the value head, from the same start, at the
   same games; one deck (Blackwing) is leaned to, a matchup's signal being
   measurable there and a run hours. Its design and readings are fixed
   after the first two phases. Its shape, agreed on 2026-10-22: no new
   control runs, the runs already made are the baselines. The critic's arm
   goes on from `all71`'s checkpoint at R1 (1,920 updates of PPO) with the
   same seed and games, to R2 (4,440), measured beside `all71`'s own
   checkpoints there; where that is not decided, an arm from `half`'s 1,920
   updates to its 3,840, against its own; other seeds only where the two
   settings disagree, both arms at each. Before it, a few updates from R1 at
   today's code are checked to play `all71`'s games. **Run whatever the
   second phase shows** (decided on 2026-10-22, while it ran). The second
   phase measures the noise of an update's step, not its bias: a critic
   wrong in a consistent way, or right where the head is wrong, changes
   where the step points, which at these duels a step cannot be measured to
   show; only a run's result sums it. What others found was in a run's
   result too, and neither reports how well its critic predicted:
   AlphaStar's value function given the opponent's observations, in
   training only, took a best response against fixed opponents to 82% of
   its games, 22% without (Vinyals et al. 2019, Fig. 3K); PerfectDou's
   critic given the other players' hands, 1e9 steps each, against
   DouZero of as many: 73.8% beside plain PPO's 50.9% without its
   shaped reward, 73.2% beside 71.7% with it (Yang et al. 2022, Table 4).
   In DouDizhu every card is dealt at the start, and the hands are all of
   a game's luck.

## Readings (fixed on 2026-10-22, before anything was built)

1. *It sees more*: the critic explains at least 0.10 more of a result than
   the value head on the games it never trained on, over all their
   decisions, by two standard errors.
2. *A stronger signal*: with the critic's advantages, the update's signal
   per duel is at least twice the value head's 5.0e-5 (at λ = 0.95), the
   gap more than two standard errors.

Told beside them, without a mark: both by turn, and for both versions.

## The build (2026-10-22)

- **The policies** (`native/ygo-policies`): `Projection::observation_whole`,
  the observation with every card the projection knows named, the other
  player's face-down cards and Extra Deck as well; a test beside the
  projection's.
- **The environment** (`crates/ygo-env`): a setup's `whole` gives each of
  its caller's seats a `Sight` (`src/seat.rs`), a projection fed the
  engine's messages unredacted; at each decision the network is shown,
  the same decision is packed from it with the seat's own trail, and kept
  beside the network's under the same row (`Pool::batch_whole`,
  `ygo_pool_batch_whole`). With `whole_deck` the next cards of each Deck
  are named as well, top first, from the engine. Played with and without
  it, 236 decisions of eight duels: the network's arrays the same byte for
  byte; the seat leaves 3.4 cards a decision unknown, the whole view none
  (`tests/test_pool.py`).
- **The critic** (`ygo_training/critic.py`): `SightedCritic`, two readers of
  the policy network's shape, the seat's view and the whole one, begun as
  the policy's network, and a value of their two states together
  (`tests/test_critic.py`).
- **Fitted alone** (`tools/privileged_critic.py fit`): the policy as the
  checkpoint has it; after each update's games the critic is taught, in
  one pass of minibatches of 256 at 3e-4, the results of eight of each
  seat's transactions a duel, drawn at random (a duel's result is one
  outcome its positions tell again and again; the value head taught every
  position learnt the results by heart, taught four did not). On each
  update's games, before it is taught them, what it explains is kept
  beside the value head's; then it reads every transaction of four held
  updates' games, far from those (`--away 10000`). An implementation
  choice the design left open: the number of positions.
- **The signal** (`tools/update_signal.py noise-critic`): the update's step
  with the value head's advantages and with the critic's, on the same
  games and blocks (`report-critic` takes the second reading).
- **`AGENTS.md`**: the input's rule has its exception, never the policy's.

## Results of the first phase (read on 2026-10-22)

**The hidden hands and face-down cards** (`analysis/privileged-critic/hands`,
on the development machine in 11 minutes): **not shown to see 0.10 more.**
Over every decision of 2,080 held duels (240,380 transactions) the critic
explains 0.368 of a result and the value head 0.398: the critic's lead is
−0.030 ± 0.010. By turn it is over the head only at the first (0.131
against 0.115) and a little under it from the second on (0.206 against
0.226 at the second, 0.399 against 0.440 from the eleventh). On each
update's games before it was taught them, it came from nothing to the
head's level by the end (0.497 against 0.496 at the fortieth update).

**The hidden cards and the next five of each Deck**, told beside it
(`analysis/privileged-critic/deck`, as long): the same. Over every decision
of the same held duels the critic explains 0.369 and the head 0.398, a lead
of −0.030 ± 0.011; by turn over the head at the first (0.141 against
0.115), under it from the second on; on each update's games before it was
taught them, level with the head by the end (0.502 against 0.496).

What it says: in 40 updates the critic learns from nothing to what the
value head explains, and knowing the cards in hand, the face-down cards
and the draws to come does not take it further. What it is shown is
likely not what holds it back: at a seat's first decision the deal is 0.58
of a result (measured by playing deals again, [The signal of an
update](2026-10-22-update-signal.md)), and the critic that sees both hands
and the next five draws of each Deck explains 0.14 there. Those are not
all of the deal, but they are the cards of its first turns. Likelier it is
how much it has learnt, from 250,000 positions with a value begun anew,
where the head learnt from all of a long run's.

### The critic begun as the value head (added on 2026-10-22)

Told beside the first reading, which stays as it was taken, without a mark
of its own: the critic's value begun as the policy's value head, what it
reads of the seat's view the head's weights and of the whole duel nothing
yet (`fit --warm`), so that it begins at the head's level and has only what
the hidden cards add to learn; and fitted five times as long, 200 updates,
in both versions. Then the second phase with each. On the CUDA machine from
13:44, at `16f7feef` (its checkout taken there from the development
machine's commits directly, by a branch `critic-build` of each repository
that holds them), in tmux `critic` (`~/ygo-training/critic-runs.sh`).

**What there is to see** (`tools/privileged_critic.py hidden`, the held
games of the reading; `analysis/privileged-critic/hidden.txt`): of every
decision, by turn, the other player's cards the seat does not see. At
the first turn its whole hand, about five cards, and nothing face-down;
from the second on three to four a decision, about two in hand and one
to one and a half face-down (3.28 + 1.35 at the second turn, 1.90 + 1.26
at the seventh to tenth, 2.28 + 1.58 from the eleventh); at 93% to 100%
of the decisions at least one. The hidden cards fall by about a third over
a duel, and what the critic gains from them falls to nothing: it is not
that there is nothing left to see.

**Begun as the head, 200 updates** (on the CUDA machine, 13:44 to 14:36,
`analysis/privileged-critic/warm-hands` and `warm-deck` there), read on the
same held duels:

| Version | The critic | The head | Its lead | Turn 1 | Turn 2 | Turns 3-4 | Turns 5-6 | Later |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Hands and face-down cards | 0.400 | 0.398 | +0.002 ± 0.008 | +0.067 | +0.022 | +0.005 | −0.004 | −0.003 to −0.005 |
| And the next five of each Deck | 0.405 | 0.398 | +0.007 ± 0.008 | +0.073 | +0.026 | +0.013 | +0.004 | −0.001 |

On each update's games before it was taught them, both stayed 0.01 to 0.03
over the head throughout (0.522 against 0.501 at the 200th update with the
hands). Begun where the head is and taught five times as long, the critic
gains on it at the start of a duel only, where the head is weakest; knowing
the draws to come adds a little more there and nothing later.

## Results of the second phase (read on 2026-10-22)

On the CUDA machine from 14:41 to 15:07, at `7d4d1cd5`
(`tools/update_signal.py noise-critic` and `report-critic`;
`analysis/update-signal/noise-critic-warm-hands.json` and `-deck.json`, the
reports beside them): `all71`'s last checkpoint, the games of 32 updates
(16,640 duels) in 16 blocks of 256 groups of 65 duels, the step's
advantages from the value head and from each critic begun as the head and
fitted 200 updates (above), on the same games and groups. **Not shown to
be stronger, in either version.**

| Advantages from | Noise scale | Signal per duel | Its gap to the head's |
| --- | --- | --- | --- |
| The value head | 19,239 ± 5,860 duels | 5.20e-05 ± 1.58e-05 | |
| The critic, the hands and face-down cards | 21,476 ± 7,971 duels | 4.66e-05 ± 1.74e-05 | −0.54e-05 ± 0.99e-05 |
| The critic, and the next five of each Deck | 22,574 ± 8,349 duels | 4.43e-05 ± 1.66e-05 | −0.77e-05 ± 0.78e-05 |

The head's own is where the follow-up of [The signal of an
update](2026-10-22-update-signal.md) found it (5.0e-05). The mark, twice
that with the gap over two standard errors, is far: with either critic the
step is at best as consistent as with the head, each gap within one
standard error of nothing, on the low side. What the critic adds to its
value at a duel's first turns does not reach the credit. The third phase
runs all the same, by the decision above: this measures the noise of a step
and not where it points.

## The third phase: a run (design and reading fixed on 2026-10-22)

Fixed before anything of it was built. One change from the shape agreed
before it: the arm goes from `all71`'s copy to R1, not from R1 to R2. There
`all71` gained most (on the forty 11.2 points in its first 1,920 updates of
PPO, 2.8 in the next 2,520; on the eighteen 6.7 and 1.3), so a learner
faster by as much is about twice as far ahead there (learning twice as fast:
about 2 points at R1, under 1 at R2), and the run is 600 updates shorter.

- **One arm**, `critic`: PPO on from `all71`'s copy
  (`all71-copy/checkpoints/model-17039.pt`, 17,040 updates of imitation)
  for 1,920 updates, as `all71` went on from it, with its settings and
  seed (2012), at today's code.
- **The critic**: the version with the next five cards of each Deck, the
  second phase favouring neither (the version it favours, or that one, as
  confirmed); begun as the copy's value head; taught after each update's
  games as in the first phase, the results of eight of each seat's
  transactions a duel drawn at random, in one pass of minibatches of 256
  at 3e-4, with an optimizer of its own. The policy's advantages come from
  it, its values in the head's place in GAE (λ 0.95 over transactions);
  the value head is taught as before, its target its own return, so that
  only the credit changes. The critic is kept beside each checkpoint
  (`critic-NNNNN.pt`), never in one.
- **Before it**: three updates from the copy at today's code without the
  critic, checked to play `all71`'s first three (the fingerprints of every
  decision). Batteryman Fuel Cell's script is upstream's fix now, where
  `all71` had the mend: its duels may differ.
- **Read as `all71`'s R1**: the checkpoints after 1,680, 1,800 and 1,920
  updates on the thirteen screens, the eighteen and the forty, game by game
  beside `all71`'s own there. About seven to eight hours on the CUDA
  machine, by reckoning, and an hour of measuring.

### Its reading (fixed on 2026-10-22, before the run)

3. *It learns faster*: `critic` against `all71` at R1, both by their
   checkpoints after 1,680, 1,800 and 1,920 updates of PPO, game by game
   over the three suites together (the thirteen screens, the eighteen and
   the forty): **faster** when the arm is over `all71` by 2 points and two
   standard errors more, **slower** when it is under by as many, otherwise
   **not told apart**, and then the second setting: `half` from its copy
   to its 1,920 updates, against its own.

Told beside it, without a mark: each suite, the critic's lead over the
value head along the run, and the training win rate. A doubling of the
speed of learning would land about at the mark: this arm alone decides a
large effect only.

### The run (2026-10-22)

Started on the CUDA machine at 15:25, at `b8e6df72` with the submodules at
their pins (its checkout taken from the development machine's commits by
the branch `critic-build`, as before), nothing else running there, in tmux
`critic-run`, from the repository root:

```sh
PYTHON=.venv/bin/python bash experiments/commands/privileged-critic.sh
```

**The check passed**: three updates from the copy at this code, without the
critic, played `all71`'s first three (versions 17,040 to 17,042), the
fingerprints of every decision and of what each update trains on the same:
this code plays its games and makes its updates, the engine's fix of a
monster whose substitute is destroyed and upstream's Batteryman Fuel Cell
notwithstanding. The arm's first update, the critic begun as the head,
explained its games' results as the head did (0.442 and 0.442). Its first
updates took 12 to 15 seconds, the critic's part 2.2 to 3.2 of them, where
`all71`'s first 1,920 took 9.0 on average: by reckoning about seven hours.

## Results of the third phase (read on 2026-10-22)

The arm made its 1,920 updates by 21:21, 11.1 seconds each on average
against `all71`'s 9.0, and was measured from 21:21 to 22:44
(`analysis/privileged-critic/run`, its `report.txt` and `report.json`, on
both machines). `all71` measured again at this code played every one of
the measuring's 101,088 games (33,696 deals, three checkpoints each) as
[Seventy-one decks](2026-10-19-seventy-one-decks.md) measured them.

**3. It learns faster: not told apart.** Over the three suites the arm
wins 54.9% against `all71`'s 54.8%, +0.1 ± 0.2 points over 33,696 games:
by two standard errors between −0.2 and +0.5 points, where the mark was 2.

| Suite | `critic` | `all71` | Gap | Games |
| --- | --- | --- | --- | --- |
| The thirteen screens | 55.1% | 54.8% | +0.2 ± 0.2 | 21,632 |
| The eighteen other decks of thirty-one | 50.9% | 51.4% | −0.5 ± 0.5 | 3,744 |
| The forty new decks | 56.5% | 56.4% | +0.1 ± 0.4 | 8,320 |

Told beside it, without a mark:

- **Each screen** is within a point of `all71`'s (± 0.6 to 0.9, 1,664
  games a screen), but Blackwing's, +2.7 ± 0.8. One of thirteen beyond
  three standard errors turns up by chance about once in a hundred: told,
  not read.
- **Along the run**, on each update's games before the critic is taught
  them, the critic explains 0.007 more of a result than the value head in
  the first 240 updates and 0.025 more in the last 240 (0.449 against
  0.424). Against the pilots in training the arm is level with `all71`:
  −0.15 ± 0.07 points paired by update over the run, 57.1% each in the
  last 240.
- **How it plays**: its choices stay more spread (entropy 0.265 against
  0.233 in the last 240 updates), and it passes less at a chain window,
  18,205 passes an update in the last 480 against 22,115 (−18%), where it
  activates as often (3,046 against 3,072); its decisions are 6.5% fewer,
  its duels as long (13.4 turns). The credit is in keeping with it: the
  critic's values are a little under what follows them (the raw
  advantages average +0.017 throughout, `all71`'s 0.000; 47% of them over
  nothing once normalized, against 55%), and a pass at a chain window is
  told worse (−0.025 normalized, against −0.008). Duels cut off: 4 against
  6.

What it says: the critic changes a little how the learner plays, not how
often it wins; at R1 the two runs are within half a point. The three phases
say the same: the critic that sees the hidden cards explains a little more
of a result than the value head, at a duel's first turns; the step it
gives is no more consistent; and a run with it learns no faster. Why the
hidden cards add so little to what it explains is not shown. Two readings
fit what was measured: what is still hidden from it (the draws after the
next five, the players' own sampled choices) holds most of a result; or it
has not learnt what the hidden cards do, its reader of the whole duel
begun from nothing and taught by results alone, where at a duel's first
decision the deal is 0.58 of a result and the critic explains 0.18 at most.

**Closed** that evening: the reading named the second setting next, `half`
from its copy to its 1,920 updates with the critic, against its own, but it
could find a large effect only, and this setting puts any effect under half
a point. Next, [How cards work](2026-09-08-engine-transition-prediction.md).
