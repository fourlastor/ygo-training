# Three of ygo-agent's training settings, one at a time

**Status:** COMPLETED

**Proposed:** 2026-10-16

**Activated:** 2026-10-16

**Completed:** 2026-10-16

**Last updated:** 2026-10-16 (closed the morning it was read, UPGO to be
tried again as an arm of the first run on the new input; run and read:
none of the three is told apart from the control, UPGO 1.9 points over it
and smaller updates 1.6 under)

Chosen on 2026-10-16, to fill the CUDA machine's night beside the
measurements beyond their decks of the networks of
[One network for more decks](2026-10-14-more-decks.md): of the list of
things for which ygo-agent's trainer did better, the learning rate had
been tested and theirs kept, for its better results; further differences
in training were to be tested, each against one of the runs already made
as a control, and three at once, the throughput being highest with three
runs in parallel. The design and the readings below were fixed the same
night, before any run.

## Research question

Does any of three of the settings ygo-agent's trainer has and this one did
not, each on its own, make PPO of Blackwing win more of the held-out deals
at the same number of games: updates of fewer games, their minibatch and
clip, or UPGO?

## Why

- **The list was written down** in
  [PPO with a head that predicts](2026-10-09-ppo-with-a-head-that-predicts.md)
  (its item on how theirs differs in the rest, read in their
  `scripts/cleanba.py` at `26293f8`), and it was decided on 2026-10-10
  that what followed was to find which of those differences matters and
  bring it in. Where each stands:

  | Theirs | Here |
  |---|---|
  | Learning rate 1e-4 | Tried first: 4.8 points over a million games; the rate of PPO here since 2026-10-11 |
  | One pass over an update's steps | One pass since 2026-10-10: four undid the copy |
  | One network in both seats, learning from both | Half of the duels against itself, tried in three experiments; all of them not tried |
  | From nothing | From a copy of the pilot; not worth trying |
  | An update of 128 steps of each of 32 duels that go on from update to update, some 20 games, its credit cut at the end of the stretch | 520 whole games: **tried here, as updates of fewer games** |
  | Minibatches of 512 steps, a clip of 0.25 | 256 and 0.2: **tried here** |
  | UPGO | **Tried here** |
  | 4.3M weights, cards and answers that attend to each other, a state carried through the game | The network: for when the input changes, not here |

## Design (fixed on 2026-10-16)

`experiments/commands/training-differences.sh`. **The control** is
`decks-01` of the experiment on more decks: Blackwing alone, its pilot
imitated 240 updates from scratch (`decks-01-copy`), then PPO at 1e-4, one
pass, 520 duels an update, 260 against itself and the others against the
pilots of the thirteen decks of training, minibatches of 256, a clip of
0.2, seed 2012; its checkpoints after 720, 840 and 960 of those updates.
**Three runs** begin from the same copy (`model-00239.pt`) and are made as
the control was, but for one thing:

1. **`small-updates`**: 130 duels an update, 65 against itself, four
   updates for each of the control's. The same games and the same steps
   of the optimizer a game; the policy the games are played by is new four
   times as often. Theirs carries its duels from one update to the next
   and cuts the credit at the end of a stretch; here every update still
   holds whole games.
2. **`their-step`**: minibatches of 512 and a clip of 0.25
   (`--minibatch 512 --clip 0.25`): half the steps of the optimizer a game.
3. **`upgo`**: their UPGO added to each transaction's advantage (`--upgo`,
   built on 2026-10-16 and held to a transcription of their loop): the
   seat's return followed on through the transactions after it for as long
   as each went at least as well as the head expected, less the head's
   estimate. The head's target is unchanged, as theirs is.

Each runs to the games of 960 of the control's updates (3,840 updates for
`small-updates`), the three at once through MPS, keeping a checkpoint every
120 of its updates. **The runs train under the rule against repeated
commands, merged on 2026-10-16, and the control did not**: Blackwing alone
had no game cut off in its run, so the rule should seldom touch it. The
measuring part plays the control's checkpoints again under the rule, and
it is that measurement the runs are read against; the report sets it
beside the control's measurement before the rule.

## Readings (fixed on 2026-10-16)

Each run on its own against the control, by their checkpoints after the
games of 720, 840 and 960 of the control's updates, on the 1,536 held-out
deals, deal by deal, a deal read by the share of a run's three checkpoints
that win it:

- **better** when it wins at least 2 points and two standard errors more;
- **worse** when it wins at least 2 points and two standard errors fewer;
- otherwise **not told apart**.

Told beside them without a mark: each checkpoint; the control under the
rule against the control before it; what each run's updates were like
(the seconds of each of the control's updates' worth, the KL an update
moves the policy by, how many ratios are clipped, the entropy, and what
UPGO added).

## The run

On the CUDA machine, from the repository root, after the pushes (the
`main` of `native/ygo-model-input` first, then this repository's), a pull
and `git submodule update --init --recursive` (the pin of
`native/ygo-model-input` moved with the rule), as one line after the
measurements beyond their decks of the networks of more decks:

```sh
PARTS=beyond bash experiments/commands/more-decks.sh; AT_ONCE=3 bash experiments/commands/training-differences.sh && PARTS=measure bash experiments/commands/training-differences.sh
```

By reckoning: an hour for the first part, five to six and a half hours for
the three runs (`small-updates` sets the pace: an update of fewer games
costs more a game, and how much more on the card is not known), twenty
minutes for the measuring part. `analysis/training-differences` comes back,
the readings in `report.txt` there (`tools/training_differences_report.py`),
and `analysis/more-decks/beyond`.

## The tooling

- `train --clip` and `--upgo` (`docs/commands.md`), `tests/test_upgo.py`.
- `experiments/commands/training-differences.sh` and
  `tools/training_differences_report.py`, its marks held by
  `tests/test_training_differences_report.py`.
- Tried small on the development machine's processor on 2026-10-16, on
  arms of one and two decks made small for it: the three runs three at
  once, each with the settings it is told (in its `run.json`), the
  measuring part and the report; and the measurements beyond their decks
  of `PARTS=beyond`, to their report.

## Results (read on 2026-10-16)

The three runs ran on the CUDA machine from 02:32 to 10:05 on 2026-10-16,
at `a7523f68`, three at once through MPS until UPGO ended at 07:26 and
their minibatch and clip at 07:55, smaller updates alone after that. The
measuring part ran from 10:08 to 10:23; `analysis/training-differences`,
the report in `report.txt` (its figures of the updates as
`tools/training_differences_report.py` gives them since `fc83cf98`, which
reads the control's over its first 960 updates, not all its 1,920).

The 1,536 held-out deals, by the games of the control's updates each
checkpoint had played, the control under the rule against repeated
commands as the runs are:

| Run | 720 | 840 | 960 | The three | Against the control |
|---|---|---|---|---|---|
| The control | 74.2% | 72.8% | 74.3% | 73.8% | |
| Smaller updates | 71.7% | 70.7% | 74.0% | 72.1% | −1.6 ± 0.7 |
| Their minibatch and clip | 71.0% | 73.2% | 73.8% | 72.7% | −1.1 ± 0.7 |
| UPGO | 74.5% | 76.5% | 76.0% | 75.7% | +1.9 ± 0.8 |

**By the readings none of the three is told apart from the control**: each
is under the 2-point mark, either way. UPGO comes closest (1.9 points over,
2.4 standard errors) and smaller updates lean the other way (1.6 under,
2.3 standard errors). Each is one run of one seed, and runs of one
training on other seeds have ended a point or two apart.

Beside the readings, without a mark:

- **The rule against repeated commands changes nothing for the control on
  these deals**: its three checkpoints win the very deals under it that
  they won before it (+0.0).
- **What the updates were like**, over the updates the three checkpoints
  read had made, each run three at once with the others for most of it:

  | Run | Seconds a control update's worth | KL an update | Ratios clipped | Entropy |
  |---|---|---|---|---|
  | The control (its first 960) | 21.5 | 0.0046 | 2.8% | 0.105 |
  | Smaller updates | 28.5 | 0.0032 | 2.2% | 0.101 |
  | Their minibatch and clip | 20.2 | 0.0043 | 1.8% | 0.082 |
  | UPGO | 18.4 | 0.0058 | 3.5% | 0.123 |

  Smaller updates cost a third more a game. UPGO moves the policy more an
  update and keeps it less sure, and added 0.054 to a transaction's
  advantage on average, before the advantages are normalised. Their
  minibatch and clip leave it surer.

## What it says

None of three of ygo-agent's training settings, each tried on its own,
makes PPO of Blackwing win 2 points more of the held-out deals in its
first 500,000 games. UPGO is the one that might: it is 1.9 points over,
which one run cannot tell from a seed's spread. Updates of fewer games
cost time and win nothing; their minibatch and clip change nothing.

Of the list of differences what is left untried is the network (its size,
cards and answers that attend to each other, a state carried through the
game) and all of the duels against itself.

## What follows

Closed on 2026-10-16, the morning it was read: UPGO is tried again as an
arm of the first run made on the model's new input, with and without it
side by side, rather than with another seed of this. The other two are
left as they are. Next, in the order agreed on 2026-10-15: the deck lists
dropped from the model's input.
