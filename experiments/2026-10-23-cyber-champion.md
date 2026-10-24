# A Cyber champion: the Blackwing recipe from the improved pilot

**Status:** COMPLETED (closed on 2026-10-24, when the design of the next
experiment, built on what it found, was fixed)

**Proposed:** 2026-10-23

**Activated:** 2026-10-23, when [Overlap](2026-10-23-overlap.md) was read
and closed. **The design and the readings were fixed before anything was
run.**

**Last updated:** 2026-10-24 (run and read: a champion, 74.4% ± 0.5% of
the mirror against `all71`; the pilot not shown to matter; `cyber-old`, played
in the mirror, a champion too at 76.4%)

Proposed on 2026-10-23, once Cyber's improved pilot closed 56% of its gap to
`all71` ([The improved pilots](2026-10-23-improved-pilots.md)): to run
Blackwing's recipe again for Cyber, as far as Blackwing's went before it
stopped improving, and see whether the improved pilot and self-play make a
single-deck champion, faster with the speed work done since.

The recipe is that of the network of Blackwing alone, `decks-01` of
[More decks](2026-10-14-more-decks.md): its pilot imitated from scratch for
240 updates at 3e-4, then PPO at 1e-4, half of each update's 520 duels the
mirror against itself and the others against the thirteen decks' pilots.
In the mirror against the network of thirteen it reached 70% after 720
updates of PPO, some 200,000 games of the mirror (374,400 duels in all), and
stayed at 69% to 73% to its 1,920; the first 120 updates brought two thirds
of it.

## Design (fixed on 2026-10-23)

- **Two arms, the recipe for Cyber to 720 updates of PPO.** `cyber-new`
  copies the improved pilot (the policies' branch `pilots-against-the-network`
  at `d696a4b`, played in a worktree of its own, the pin not moved);
  `cyber-old` copies the pilot as pinned. Without the second, a strong
  network would not say whether the pilot made it.
- **Today's settings.** The copies as the recipe has them (minibatches of 256,
  no duels in flight). PPO with `fast`'s, the default on a card since
  2026-10-23 (1,040 duels in flight, minibatches of 512, the calls as they
  are in two groups), and with the next update's games beside the pass if
  [Overlap](2026-10-23-overlap.md) reads equivalent and faster. Overlap read
  equivalent and not faster (1.17 times), so PPO runs with `fast`'s settings
  alone. Seed 2012 for both arms: the same deals.
- **Measured afterwards** (`experiments/commands/cyber-champion.sh`): each
  arm's checkpoints after every 120 updates of PPO on Cyber's yardstick
  (`eval_suites/cyber-yardstick.json`: Cyber's 208 games of the forty decks'
  suite against the thirteen decks' pilots, the deals `all71` and the pilots
  played there, checked to give the pinned pilot its 18.8% again); and
  `cyber-new`'s last three checkpoints against `all71` in the Cyber mirror
  (`eval_suites/cyber-against-checkpoint.json`, 2,048 deals), each deal from
  both seats.

## Readings (fixed on 2026-10-23, before anything was run)

1. *A champion*: `cyber-new`'s last three checkpoints against `all71` head
   to head in the Cyber mirror, every deal from both seats, a deal's games
   together: **a champion** when it wins at least 52% by two standard errors
   (the share less two standard errors at 52% or more), **weaker** at 48% or
   less by two standard errors, otherwise **not told apart**.
2. *The pilot matters*: `cyber-new` against `cyber-old` on Cyber's yardstick,
   after 120 updates of PPO and after 720, the same games paired: it
   **matters** at a point when `cyber-new` is at least 2 points over by two
   standard errors; otherwise not shown there.

Told beside them, without a mark: both arms' yardstick curve every 120
updates beside `all71`'s 57.5%, the pinned pilot's 18.8% and the improved
pilot's 40.4% on the same games; the hours each arm took
(`tools/cyber_champion_report.py`). A point of the yardstick is 208 games,
some ±3.4 points alone and about ±4 for a paired difference.

## The build (2026-10-23)

- `experiments/commands/cyber-champion.sh`: the worktree (its submodules from
  this checkout's, the policies at the branch, refused unless at `d696a4b`),
  both arms in turn, and the measurements; tried small on the development
  machine's processor, both arms and every measurement to the report.
- The defaults of a run on a card leave a copy as it was: a run with
  updates of imitation takes no duels in flight and minibatches of 256
  where its command line does not say.

## The run (2026-10-23)

Begun on the CUDA machine the evening overlap was closed, at `ac1b90a4`'s
code, by `experiments/commands/cyber-champion.sh` as it stands (no
`OPTIONS`). It is the first training run since `fast`'s settings were made
the default on a card: the script names them for PPO, and leaves a copy as
it was.

`cyber-old` plays from that machine's checkout, whose engine is built from
a profile of its own games by a config of the machine's own
(`tools/engine_pgo_build.sh`). `cyber-new` plays from the worktree, which
that config does not reach, so its engine is built as it is. Their games
are the same either way, as the profile build's check found; only the hours
of the two arms are not alike, the engine's part of an update a tenth
faster in `cyber-old`.

## Results (2026-10-24)

Both arms ran from 20:40 to 00:02 and were measured by 00:07, at
`ac1b90a4`'s code (the report and the runs' metrics copied to
`analysis/cyber-champion/`). Nothing failed: no duel stopped by the engine
or cut off, no traceback, none of the memory warnings overlap had, at most
0.7 GiB of the card in use.

1. **A champion: a champion.** `cyber-new`'s checkpoints after 480, 600 and
   720 updates of PPO win 74.4% ± 0.5% of the Cyber mirror against `all71`,
   2,048 deals each from both seats. With two standard errors taken away
   that is 73.4%, over 52%. By checkpoint 70.4%, 76.7% and 76.0%, from
   either seat alike.
2. **The pilot matters: not shown at either point.** `cyber-new` against
   `cyber-old` on Cyber's yardstick: +7.7 ± 3.3 points after 120 updates of
   PPO (51.4% against 43.8%; less two standard errors 1.1, under 2), +0.5 ±
   3.6 after 720 (63.0% against 62.5%).

Beside them, without a mark:

- **The yardstick every 120 updates**, where `all71` wins 57.5% of the same
  games, the pilot as pinned 18.8% and the improved pilot 40.4%:
  `cyber-new` 51.4%, 50.5%, 57.2%, 60.6%, 60.6%, 63.0%; `cyber-old` 43.8%,
  51.0%, 57.7%, 57.7%, 60.6%, 62.5%.
- **The hours:** `cyber-new` 0.23 of copy and 1.76 of PPO, `cyber-old` 0.21
  and 1.16; both arms and the measurement 3 hours 28 minutes.
- **Why `cyber-new`'s PPO took longer:** its games against itself grew long
  between 360 and 540 updates, to 52 turns on average over 421 to 480,
  with about a fifth of them won by a deck-out, and came back to 20 to 27
  turns. `cyber-old`'s grew later and less, to 19 turns over its last 60.
  The pace per decision held at about 25,000 decisions a second.
- **Against `all71` the games stay short:** 12 to 13 turns on average; 21 of
  the last checkpoint's 4,096 were decided by a deck-out.

What it says: the recipe makes a Cyber champion in two hours of the card,
as it made Blackwing's (`decks-01`, 70% of the mirror against the network
of thirteen). From the improved pilot it starts higher, but the pinned
pilot's arm catches up by 720 updates, so the pilot is not shown to have
made the champion. Whether `cyber-old` is a champion too was not measured:
the design plays only `cyber-new` in the mirror.

## `cyber-old` in the mirror (2026-10-24)

Not in the design. Added once the readings were in, so that whether the
pinned pilot's arm is a champion too could be told. `cyber-old`'s
checkpoints after 480, 600 and 720 updates of PPO against `all71`, the same
2,048 deals from both seats, played on the CUDA machine in 3 minutes
(`analysis/cyber-champion/cyber-old/mirror/`).

- **By reading 1's rule, a champion too:** 76.4% ± 0.5% of the mirror. By
  checkpoint 75.4%, 77.3% and 76.5%, from either seat alike, in games of
  11 to 12 turns.
- **The two arms paired deal by deal, without a mark:** `cyber-new` 2.0 ±
  0.5 points under `cyber-old` (74.4% against 76.4%). The error is the
  deals' alone. One run of each arm says nothing of how far two runs of one
  recipe differ.

What it says: the pinned pilot's arm is as strong a champion, or a little
stronger. What made the champion is the recipe, a deck's own pilot copied
and then PPO with half its duels against itself, not the improved pilot.

## Closed (2026-10-24)

With both readings in and `cyber-old` played in the mirror, the recipe is
known to work for a champion of one deck, in a network of one deck. The
design of the next night's run on the recipe was fixed that night. The
record is closed with it; the next experiment is
[A network of two decks at the champions' games of each mirror](2026-10-24-two-decks.md).
