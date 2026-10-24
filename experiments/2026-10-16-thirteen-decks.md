# Thirteen decks on the new input: longer, with UPGO, and against itself alone

**Status:** COMPLETED

**Proposed:** 2026-10-16

**Activated:** 2026-10-16

**Completed:** 2026-10-17

**Last updated:** 2026-10-17 (closed the day it was read; measured and
read: without the lists equivalent, more updates gained, UPGO and itself
alone not told apart; the measuring part's abort and the engine's fix told
under the run)

The first network made on the model's input without the deck lists
(checkpoint format 15). Chosen on 2026-10-16, after
[One network for more decks](2026-10-14-more-decks.md) found the network
of thirteen still learning when it stopped (to extend its training to
3,840 updates and measure what that adds, and to try imitation followed by
self-play alone on it), and
[Three of ygo-agent's training settings](2026-10-16-training-differences.md)
left UPGO 1.9 points over its control (to try it again and see whether it
improves the network). From scratch, the networks of earlier formats not
loading on the new input.

## Research questions

1. Does a network of the thirteen decks lose anything without the deck
   lists under PPO, which the copies of the experiment without them did
   not measure?
2. How much do PPO updates from 1,920 to 3,840 add to a network of the
   thirteen decks?
3. Does UPGO make it better?
4. What does playing every duel against itself, and none against the
   pilots, do to it?

## Design (chosen on 2026-10-16)

`experiments/commands/thirteen-decks.sh`, on the CUDA machine, seed 2012
throughout:

1. **`copy`**: the pilots of the thirteen decks of training imitated from
   scratch, as `decks-13-copy` was (240 updates a deck, 3,120 in all, at
   3e-4, every game seen once with its result at four positions).
2. Three runs of PPO on from its last checkpoint, 3,840 updates each at
   1e-4, one pass over an update's 520 duels, the three at once through
   MPS:
   - **`half`**: 260 duels against itself (any two of its decks, one in
     each seat) and the others its decks against the pilots of the
     thirteen decks, as `decks-13` was;
   - **`half-upgo`**: the same with UPGO;
   - **`itself`**: all 520 against itself.

A run keeps a checkpoint every 120 updates; none is evaluated as it goes.
The rule against repeated commands is in force for all of it, where
`decks-13` had not had it, and for the arms since their second start its
extension to activations at a chain window (below).

## Readings (fixed on 2026-10-16, before any run)

Each by a run's checkpoints after 3,600, 3,720 and 3,840 PPO updates unless
said, game by game on the same games, a game read by the share of the three
checkpoints that win it:

1. *Nothing lost without the lists, under PPO.* `half` after 1,680, 1,800
   and 1,920 updates against `decks-13` after as many (as the experiment on
   more decks measured it), over the thirteen screens together (21,632
   games): **equivalent** when the gap less two standard errors is above −2
   points, **worse** when it wins at least 2 points and two standard errors
   fewer, otherwise not decided. Its copy is set beside `decks-13-copy` the
   same way, without a mark. The rule against repeated commands differs
   between the two as well.
2. *More updates add.* `half` after 3,600 to 3,840 updates against itself
   after 1,680 to 1,920, over the thirteen screens: it has **gained** when
   it wins at least 2 points and two standard errors more.
3. *UPGO.* `half-upgo` against `half` over the thirteen screens: **better**
   at 2 points and two standard errors more, **worse** at as many fewer,
   otherwise not told apart.
4. *Against itself alone, what the pilots say.* `itself` against `half`
   over the thirteen screens: **better** at 2 points and two standard
   errors more, **worse** at as many fewer, otherwise not told apart.
5. *Against itself alone, head to head.* `itself` against `half`, every
   pairing of the thirteen decks, 16 deals each played from either seat,
   the other run's checkpoint held in the other seat and their
   checkpoints matched (16,224 games): `itself` is **better** when it wins
   at least 52% and two standard errors over half, **worse** at 48% and
   two standard errors under, otherwise not told apart.

Told beside them without a mark: every run's checkpoints on the held-out
deals of Blackwing; the eighteen decks outside training played by each run
(as the networks of more decks were measured beyond their decks);
deck by deck; and the runs' own figures.

## The run

Started on the CUDA machine on 2026-10-16 at 11:21, at `a2ed5485` with
`native/ygo-model-input` at `58785cd`, from the repository root:

```sh
AT_ONCE=3 bash experiments/commands/thirteen-decks.sh
```

By reckoning about a day: some four hours for the copy alone, then 17 to
21 hours for the three runs at once (`itself` has the network in both
seats of every duel, which costs more an update), with every deck's calls
answered from the graphs now that no list is in the input.

The measuring part was built while the runs went, and runs once they are
made, on the CUDA machine, after a pull and
`git submodule update --init --recursive`:

```sh
PARTS=measure bash experiments/commands/thirteen-decks.sh
```

It measures at the revision after the decision of 2026-10-16 to play every
game through the pool and build the model's input in
`native/ygo-model-input` alone, which was checked to give the same games
and measurements as the code the runs were made with (`EXPERIMENT_LOG.md`).
Into `analysis/thirteen-decks`: the copy at its last checkpoint and each
arm at its last three (half also after 1,680 to 1,920 updates) on the
thirteen screens, the same with half's at its end on the held-out deals of
Blackwing and on the eighteen decks outside training, and `itself` against
`half` on every pairing of the thirteen decks
(`eval_suites/thirteen-decks-head-to-head.json`, 169 cases of 16 deals),
each in turn the one held. The report (`tools/thirteen_decks_report.py`)
takes the five readings, `decks-13`, its copy and the pilots read from
`analysis/more-decks`, where the experiment on more decks left them on
the same games. About two hours by reckoning. Tried small on the
development machine on 2026-10-16 (two decks, arms of six updates, suites
of a few games): every measurement and the report.

**The arms started again (2026-10-16).** The copy ended at 14:47 after 3
hours 26 minutes, and the three arms started then, three at once at 13.4 s
an update (`half`, `half-upgo`) and 16.7 s (`itself`). From about its
seventieth update `half` had about one duel an update that ran to the cap
of 4,096 decisions, and `itself` from its thirty-sixth a third of one; not
`half-upgo`. Played again here with `half`'s checkpoint after 240
updates, they were Lyla, Lightsworn Sorceress activated again and again at
a chain window of the seat's own Main Phase, which the rule against
repeated commands left alone, the seat brought back to the same window
each time; such a duel then played alone while the others waited, and an
update of `half` took 16.3 s. It was decided that day to extend the rule
to activations at a chain window and to start the arms again from the
same copy, which only pilots play the games of and the rule does not
touch. With the rule, the same three updates of `half`'s games had no duel
cut off where two had been (`docs/commands.md`, where the rule is told).
They began again at 17:41, at `313854d6`: those are the
experiment's arms. The first ones are in `runs/thirteen-decks/looped`, not
measured. A first start at 16:29 had not done it: under the CUDA machine's
bash 5.3 the trainings a script runs at once did not hear Ctrl-C, so the
first arms went on into the folders they had just been moved out of, and
the new ones took them up from there; `itself` went on with the old rule.
What those trainings wrote, two of them in a folder for some minutes, is
in `runs/thirteen-decks/mixed`, not measured either.
`experiments/commands/at-once.sh` stops its runs on Ctrl-C since.

**The measuring part (2026-10-17).** The arms ended at 08:21 (`half`,
`half-upgo`) and about 10:30 (`itself`). The measuring part stopped at
`half` on the decks outside training: in one of its games, Countdown played
by its checkpoint after 3,720 updates against the Infernity pilot, a stall
duel at turn 53, the engine's searches for Synchro materials filled the
duel's 256 MB arena, and the engine's exception aborted the process. Lua
held 3.85 MB of it: the policies' engine never gave memory back. The rest
was measured without those decks (`UNSEEN=`), which are beside the
readings, not among them. The engine reuses what Lua frees since (the
fork's `77dc180`, which the policies' `a10d7ec` pins), the checks finding
every game as before (14,880 games of the pilots, 32 with and without
their search, 3,744 of `half` on those decks), and a duel whose arena still
fills ends cut off; at that pin the decks outside training can be measured,
the game that aborted the process played to its end.

Tried small on the development machine's processor on 2026-10-16 (two
decks, a copy of 4 updates and runs of 3): the copy, then the three at
once, each with what it is told (`half` 12 of 24 duels against itself and
12 against pilots, `half-upgo` the same with UPGO's figures, `itself` all
24), in format 15.

## Results (read on 2026-10-17)

Measured on the CUDA machine from 10:41 to 13:04 on 2026-10-17, the decks
outside training from 13:10 to 13:40 at the engine of `a10d7ec` (above),
nothing else running there; `analysis/thirteen-decks`, the report in
`report.txt` (`tools/thirteen_decks_report.py`).

The readings:

1. **Nothing lost without the lists: equivalent.** `half` after 1,680 to
   1,920 updates against `decks-13` after as many, over the thirteen
   screens (21,632 games): −0.0 ± 0.2, both 61.0%. Beside it, the copies:
   −0.6 ± 0.2 (48.6% against 49.1%).
2. **More updates add: gained.** `half` after 3,600 to 3,840 updates
   against itself after 1,680 to 1,920: +2.6 ± 0.2 (63.6% against 61.0%).
3. **UPGO: not told apart.** `half-upgo` against `half`: −0.2 ± 0.2
   (63.4%).
4. **Against itself alone, by the pilots: not told apart.** `itself`
   against `half`: −0.7 ± 0.2 (62.9%).
5. **Against itself alone, head to head: not told apart.** `itself` wins
   50.7% ± 0.3 of the 16,224 games against `half` (50.9%, 50.5% and 50.7%
   by checkpoint).

Beside them, without a mark:

- **Every arm beats every deck's pilot on its screen**, the pilots winning
  50.7% of the thirteen screens together: `half` from Burn's 27.6% (its
  pilot 22.8%) to Heroes' 79.7% (65.2%).
- **The held-out deals of Blackwing**: the copy 59.7%, `half` 74.5%,
  `half-upgo` 74.7%, `itself` 74.6%.
- **The eighteen decks outside training** (3,744 games a checkpoint): the
  copy 27.7% (`decks-13`'s copy 27.5%), `half` 35.5%, `half-upgo` 34.2%,
  `itself` 34.3%, where `decks-13` after 1,920 updates won 34.1% and those
  decks' own pilots win 46.6%.
- **The runs' own figures** over their last updates, three at once: `half`
  11.6 s an update (162,000 duels an hour), `half-upgo` 12.5 s (150,000),
  `itself` 11.6 s (162,000); the copy alone 4.0 s (474,000). Against the
  pilots in training `half` 61.7%, `half-upgo` 61.9%; the opening seat
  against itself 57.3%, 56.6% and 57.5%.

What it says: the input without the deck lists loses nothing under PPO,
and a network of thirteen decks is still learning at 3,840 updates, 2.6
points over its second 1,920. UPGO adds nothing here. Every duel against
itself and none against the pilots does as well against the pilots, and
head to head, as half of each. On decks it never trained on, the longer
training adds about a point and a half (34.1% to 35.5%), still eleven
points under those decks' own pilots.

## What follows

Closed on 2026-10-17, the day it was read, with every reading taken. Next,
[Card descriptions and hidden identities](2026-10-17-card-descriptions.md),
whose design and readings were fixed that day: `half` after 1,680 to 1,920
updates is its control.
