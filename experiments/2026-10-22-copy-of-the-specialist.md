# Copy of the specialist: decks-01 imitated into a new network

**Status:** COMPLETED

**Proposed:** 2026-10-10

**Activated:** 2026-10-22

**Last updated:** 2026-10-22 (opened when seventy-one decks was closed,
started on the CUDA machine, read, and closed the same day)

Proposed on 2026-10-10, during
[PPO with a head that predicts](2026-10-09-ppo-with-a-head-that-predicts.md),
where it was prepared and its readings were confirmed. Of two ways to a
network that plays several decks, the pilots of several decks imitated or a
specialist of each deck imitated into one network, the second might start a
new checkpoint well above what the policies' pilots give, if imitating a
specialist takes what it has over its pilot into a new network in a fraction
of the games it took to learn. Put on hold on 2026-10-11; its teacher chosen
on 2026-10-20, with a fourth reading confirmed that day; opened on
2026-10-22, when [Seventy-one decks](2026-10-19-seventy-one-decks.md) was
closed. The design and the readings below are those of that record's section
on its conclusion and what follows, unchanged.

## Research questions

1. Does a network that imitates the specialist's first choice, from scratch
   and in the games the pilot's copy took, play the held-out deals better
   than the pilot, and with half of what the specialist has over it?
2. Does imitating what the specialist gives every answer teach more than
   its first choice?
3. Does the copy keep the specialist's strength head to head in the
   Blackwing mirror, which the held-out deals, played against the pilots,
   do not tell?

## Design (prepared on 2026-10-10, the teacher chosen on 2026-10-20)

The teacher is `decks-01` of [more decks](2026-10-14-more-decks.md), the
network of Blackwing alone, by its last checkpoint
(`runs/more-decks/decks-01/checkpoints/model-02159.pt`: 1,920 updates of
PPO, half of its duels against itself; 78.3% of the held-out deals by its
last three, where the pilot wins 60.7%), whose curve in the mirror is
measured. The copies, `choice` and `distribution`, are made from scratch as
`once-few` imitated the pilot (seed 2012, one pass, the value taught at four
positions a game), and the teacher is measured on the held-out deals beside
them; `once-few` is the pilot's copy at the same updates. How they are made
and read is told with the readings.

`decks-01` is of format 14 and reads the other seat's deck list, so the
copies are made at `05d27141`, the last revision that loads it, by that
revision's script (`experiments/commands/specialist-copy.sh`) in a checkout
beside this one (`tools/old_checkout.sh`), on this checkout's files: the
1,536 held-out games, of the policies' `1b85b63`, that revision's pin; the
pilot's copy and its measurements; the teacher. Then, with this checkout's
code, the copies and `decks-01`'s checkpoints, folded for the mirror
(`tools/fold_deck_lists.py`), are set on `decks-01`'s curve
(`experiments/commands/blackwing-mirror-curve.sh`).

## Readings (fixed before any run: three on 2026-10-10, the fourth on 2026-10-20)

Its readings were confirmed on 2026-10-10, before any run, and are
fixed. Two networks from scratch imitate the checkpoint as `once-few`
imitated the pilot (the same first network, the same deals, one pass, 200
updates): `choice`, taught the checkpoint's first choice at each decision,
and `distribution`, taught what it gives every answer. A copy is read on
the 1,536 held-out deals, deal by deal, by the share of its checkpoints
after 160 and 200 updates at which it wins a deal.

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

A fourth, proposed when `decks-01` was chosen and **fixed on 2026-10-20,
before any run**: whether the copy keeps its teacher's strength head to head
in the mirror, which the held-out deals, played against the pilots, do not
tell. `choice`'s checkpoint after 200 updates, folded for the mirror,
against `decks-01` at its copy's end and after each 120 updates of its PPO,
every deal of the mirror from either seat (more decks, its section on the
mirror along the training of Blackwing alone), level, over and under as
`tools/blackwing_mirror_curve.py` tells them (at two standard errors from
50%):

4. *The copy keeps its teacher's strength in the mirror* when it is level
   with or over `decks-01` after 720 updates, where that curve stops
   rising; it *keeps little of it* when it is under `decks-01` after 120
   updates; otherwise it *keeps some*. `distribution` is set beside it,
   without a mark.

## The build

The trainer imitates a checkpoint since 2026-10-10 (`--imitation-teacher`,
`be1b9df9`, `1d3112e8` and `a6a0c027`), with
`experiments/commands/specialist-copy.sh` and
`tools/specialist_copy_report.py`. For `decks-01`, on 2026-10-20:
`experiments/commands/specialist-copy-decks-01.sh` and
`tools/old_checkout.sh` (`13fdf290`), and the fourth reading (`e2c2515f`,
`tools/blackwing_mirror_curve.py --keeps 720 --little 120`). Tried small on
the development machine's processor that day as written: 8 of the held-out
games, two updates of 8 duels, both copies, `decks-01`'s points 120 and 720
alone, and all four readings taken.

## The run

Started on the CUDA machine on 2026-10-22, at `15bbaa86` with the
submodules at their pins, nothing else running there, from the repository
root:

```sh
bash experiments/commands/specialist-copy-decks-01.sh
```

About three hours by reckoning: the checkout of `05d27141` made and built,
the teacher measured, the two copies made and measured, then the mirror;
into `runs/specialist-copy-decks-01` and `analysis/specialist-copy-decks-01`,
the fourth reading in `mirror/reading.txt` there.

## Results (read on 2026-10-22)

Made on the CUDA machine from 09:49 to 12:09, the copies at `05d27141` in
the checkout beside the repository and the mirror at `15bbaa86`, into
`analysis/specialist-copy-decks-01` (`report.txt`, the fourth reading in
`mirror/reading.txt`).

The readings, a copy read deal by deal on the 1,536 held-out deals by its
checkpoints after 160 and 200 updates:

1. **A copy of the specialist is past the pilot: yes.** `choice` wins 75.4%
   of the deals, 14.6 ± 1.3 points over the pilot's 60.7%.
2. **It holds half of what its teacher has over the pilot: yes.** It is
   6.6 ± 0.9 points over the pilot and half of the teacher's lead, and 1.5 ±
   0.9 under the teacher itself (76.9%): it holds nine tenths of the
   teacher's 16.1 points over the pilot.
3. **The teacher's distribution teaches more than its choice: no.**
   `distribution` wins 75.9%, 0.6 ± 0.7 over `choice`.
4. **It keeps some of its teacher's strength in the mirror.** `choice` after
   200 updates is over `decks-01` up to 360 updates of its PPO, level with it
   after 480 (49.0% ± 0.7) and under it from 600, with 41.0% against its last
   checkpoint. `distribution`, beside it, is level with `decks-01` after 600
   (48.7%) and wins 42.5% against its last.

Beside them, without a mark:

- **The teacher's answers teach more than the pilot's.** On the same deals
  and in as many updates, the pilot's copy (`once-few`) wins 57.3%: the
  copies of the teacher are 18.1 ± 1.2 and 18.6 ± 1.2 points over it.
- **How fast**: after 40 updates `choice` wins 63.9% of the deals, past the
  pilot, and after 120, 75.1%. Its first choice is its teacher's at 93.2% of
  the decisions after 40 updates and 96.7% after 200; where the teacher
  leaves its pilot, at 20.1% of its decisions, at 81.1% and 89.6%.
- **Its value head** explains 0.28 of its own games' results after 200
  updates, the pilot's copy's 0.33.
- **What it cost**: the teacher had its pilot's copy and then 1,920 updates
  of PPO, about a million games; its copy was taught in 104,000 games, the
  teacher's own against the pilots.

What it says: imitating a specialist takes nine tenths of what it has over
its pilot into a new network from scratch, in about a tenth of the games it
took the specialist to learn it; what the specialist gives every answer
adds nothing over its first choice. In its own mirror, which the games it
was taught in never played, the copy keeps what the specialist had a
quarter of the way through its PPO.

## What follows

Closed on 2026-10-22, the day it was read, with every reading taken. The
experiment made beside it that day,
[The signal of an update](2026-10-22-update-signal.md), is the one active
from then. Not decided: what of the specialists-then-copies way to a network
of several decks comes next (a copy that also learns other decks from their
pilots, or specialists of more decks).
