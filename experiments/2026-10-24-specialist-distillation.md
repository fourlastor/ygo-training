# Specialist distillation: two champions imitated into one network, then their decks against each other

**Status:** COMPLETED (closed on 2026-10-24, once its report was in)

**Proposed:** 2026-10-24

**Activated:** 2026-10-24, when
[A network of two decks at the champions' games of each mirror](2026-10-24-two-decks.md)
was closed. Of two next steps, a network of four decks by the champion's
recipe and this one, this was taken first: it bears on making champions
apart, on many machines at once, and needs no new champion. **The design
and the readings were fixed before anything was built.**

**Last updated:** 2026-10-24 (read and closed: the copy keeps Blackwing's
mirror, Cyber's not decided; the gap closed after 124,800 games of the two
decks against each other, against the network of both's 312,000)

A single-deck champion is a champion of its mirror alone: against a deck it
never played it is well under a network that played the pairing against
itself (two decks, its section on a champion outside its mirror).
Champions made apart, one a machine, would still need their pairings
trained. This asks what that costs once the champions are imitated into
one network: whether the network keeps each champion's mirror, taught by
imitation in the champion's own games, and how many games of the two decks
against each other it then needs to play their matchup as well as the
network of both, which learnt everything together.

[The copy of the specialist](2026-10-22-copy-of-the-specialist.md)
imitated `decks-01` in the games it played against the pilots alone. It
kept nine tenths of its lead over the pilot there, and in the mirror, which
those games never played, only what its teacher had a quarter of the way
through its PPO. Here half of the games imitated are each champion's mirror
against itself, as half of a champion's own duels were.

For many decks the pairings are most of the work: 71 decks make 4,970
ordered pairs of two decks and 71 mirrors. What decides whether champions
made apart pay is the second reading: whether a pair needs fewer games on a
network that already plays each deck as its champion does.

## Research question

Does one network of Blackwing and Cyber, imitating each deck's champion in
the champion's own games, keep each champion's mirror; and how many games of
the two decks against each other does it then need to play their matchup
as well as the network of both?

## Design (fixed on 2026-10-24)

- **The copy (`taught`).** One network of Blackwing and Cyber from scratch,
  each deck's games played by its champion by its first choice and
  imitated: Blackwing alone (`runs/two-decks/blackwing`) and `cyber-old`
  (`runs/cyber-champion/cyber-old`), each by its last checkpoint, after 720
  updates of PPO. 480 updates of 520 duels (240 a deck, as the network of
  both's copy), at 3e-4, in minibatches of 256 with no duels in flight, one
  pass, the value taught at four positions a game. Of an update's duels,
  260 are a champion against itself in its own mirror, 130 a deck, both
  seats played by it and imitated; 260 are a champion against the thirteen
  decks' pilots, 10 in each of the 26 pairings. These are the two halves a
  champion was made in. Each deck is taught in 124,800 of its champion's
  games, half of them its mirror, as many as the network of both's copy
  taught it of its pilot's. A checkpoint every 80 updates.
- **The cross pairings (`cross`).** PPO at 1e-4 from the copy's last
  checkpoint for 1,440 updates, with `fast`'s settings. Of an update's 520
  duels, 260 are against itself with the two decks against each other
  alone, 130 each way, so it never plays a mirror; 260 are against the
  thirteen decks' pilots, 10 in each of its 26 pairings, as the network of
  both had. It meets the two decks against each other 260 times an update,
  twice as often as the network of both, which met them 374,400 times in
  its 2,880 updates; its 1,440 updates meet them as often. A checkpoint
  every 240 updates, 62,400 games of the two decks against each other from
  one to the next, as the network of both's every 480.
- Seed 2012, on the CUDA machine, one after the other.
- **Measured afterwards** (`experiments/commands/specialist-distillation.sh`),
  each pairing on the 2,048 deals of its suite, every deal from both seats:
  the copy by its checkpoints after 320, 400 and 480 updates against `all71`
  in each mirror, and head to head against each teacher in its mirror; the
  copy's last checkpoint and each checkpoint of `cross` against `all71` in
  Blackwing against Cyber, as the network of both was measured there: a
  network playing either deck against `all71` playing the other
  (`eval_suites/blackwing-against-cyber-checkpoint.json`); each checkpoint
  of `cross` against `all71` in each mirror.

## The yardstick (measured on 2026-10-24, before the design was fixed)

The network of both in Blackwing against Cyber against `all71`, by its copy
and its checkpoints every 480 updates of PPO: the last three as measured for
its record, the others that afternoon (`analysis/two-decks/cross`).

| Updates of PPO | Games of the two decks against each other | Its share |
| --- | --- | --- |
| 0 (its copy) | 0 | 28.9% ± 0.6% |
| 480 | 62,400 | 54.8% ± 0.6% |
| 960 | 124,800 | 58.3% ± 0.6% |
| 1,440 | 187,200 | 60.8% ± 0.6% |
| 1,920 | 249,600 | 62.1% ± 0.6% |
| 2,400 | 312,000 | 63.7% ± 0.6% |
| 2,880 | 374,400 | 64.7% ± 0.6% |

By the rule of the second reading, the network of both is first within 2
points of its last three's 63.5% after 2,400 updates, 312,000 games of the
two decks against each other (+0.2 ± 0.4 points); after 1,920 it is −1.4 ±
0.4. The network of the mirrors only, which never played the two decks
against each other, wins 54.0% by its last three checkpoints.

Played again that afternoon at `dfe6a871`, after the card scripts' move,
the network of both's games there after 1,920 updates and each teacher's
against `all71` in its mirror (Blackwing alone's in its record,
`cyber-old`'s in the Cyber champion's) end as they did, 12,288 of 12,288
(`analysis/specialist-distillation/check`): the measurements made before
are paired with those made now.

## Readings (fixed on 2026-10-24, before anything was built)

1. *The copy keeps each champion's mirror*: for each deck, the copy's share
   against `all71` in that deck's mirror, by its checkpoints after 320, 400
   and 480 updates, a deal's games at the three together, less its
   teacher's share on the same deals: it **keeps** the champion's mirror
   when the gap less two standard errors is above −2 points, **loses
   strength** when the gap plus two standard errors is below −2, otherwise
   **not decided**.
2. *The games of the two decks against each other it needs*: the first of
   the copy's last checkpoint and those of `cross` at which its share of
   Blackwing against Cyber against `all71`, less the network of both's
   63.5% (its last three checkpoints, the deals paired), less two standard
   errors, is above −2 points. The games of the two decks against each
   other it had played by then are the reading, set beside the network of
   both's 312,000. Where none of them meets it, up to 374,400: **not
   closed within the network of both's games**.

Told beside them, without a mark (`tools/specialist_distillation_report.py`):
the two curves at as many games of the two decks against each other; each
mirror against `all71` along `cross`, which plays none; the copy head to
head against each teacher; the hours of each part, and of the whole way
(the two champions, the copy and `cross`) beside the network of both's 6.6.
Added on 2026-10-24, after the run began and before anything was measured:
the copy's checkpoints before its last three, after 80, 160 and 240
updates, against `all71` in each mirror over its teacher, for how far into
the copy its strength comes, which tells how long a copy needs to be.

## The build (2026-10-24)

- `--imitation-teacher DECK=CHECKPOINT`, once for each deck of a learner of
  several (`TrainerConfig.imitation_teachers`): each deck's seats are played
  by its own teacher (`pool.taught_games`), and each is asked again at the
  decisions of its deck in the update (`PpoOptimizer._taught`). In an update
  that imitates networks, the duels against itself are played by them, a
  teacher in both seats, both imitated; where the pilots are imitated there
  are none, as before.
- `--self-play-pairs cross`: the duels against itself with two different
  decks alone, every ordered pair as often as any other
  (`schedules.allocate_matchup_seats`).
- Every kind of run before plays as it did, to the last weight and to every
  number it reports but the clock's, two updates each: a copy of one
  teacher, the pilots' copy of two decks, PPO of two decks with the duels
  against itself on every pair and on the mirrors alone, and PPO of one deck.
- `experiments/commands/specialist-distillation.sh` and
  `tools/specialist_distillation_report.py`, held by
  `tests/test_specialist_distillation_report.py`; tried small on the
  development machine's processor, every part, to the report.

## The run

Begun on the CUDA machine on 2026-10-24 at 15:08, at `a2c59c0b` with the
submodules at their pins, nothing else running there, from the repository
root:

```sh
bash experiments/commands/specialist-distillation.sh
```

Into `runs/specialist-distillation` and `analysis/specialist-distillation`.
An update of the copy takes about 26 seconds, not the 7 or so reckoned: each
mirror game is imitated from both seats, about 100,000 decisions an update,
and both teachers are asked at them. The copy takes about 3.5 hours, the
whole run about 7.

### The copy's speed (2026-10-24)

Of an update's 26 seconds, 15 are its games, 9.3 of them the teachers
answering in some 3,300 calls of 29 decisions on average, and 10 the
update, 3.8 of them the teachers asked again: once to measure them, and
once more in the training pass, where for `choice` their answers are not
used (some 15 minutes of the copy). Built that afternoon for later runs,
not in this one: the training pass asks the teachers for `distribution`
alone, which changes no result; and the teachers' games go by the pool's
groups and from CUDA graphs where a run names them, which can change a
first choice between two answers that score the same to their last
digits. Every kind of run before plays as it did, to the last weight. How
fast each way goes on the card is measured when it is free
(`experiments/commands/teacher-calls.sh`).

Measured on the CUDA machine that night, once the run was done, at
`7e568fbf`: 12 updates of this copy in each way, the seconds of an update
after its first.

| The teachers' games | Seconds an update | The teachers' calls | Their seconds |
| --- | --- | --- | --- |
| As begun, 16 slots, all together | 24.1 | 3,285 | 9.4 |
| An update's duels at once, 26 slots | 23.7 | 3,206 | 9.2 |
| In two groups | 24.2 | 3,840 | 11.1 |
| From CUDA graphs, up to 260 decisions | 18.6 | 3,206 | 4.3 |
| In two groups, from graphs up to 130 | 17.7 | 3,840 | 4.8 |

The graphs halve the teachers' time; two groups alone only make more calls,
and an update's duels at once change little. The training pass asking the
teachers no more made 24.1 seconds of the run's 26. With graphs and groups a
copy of 480 updates takes about 2.4 hours where this one took 3.5. Decided on
2026-10-25: the next copy names both.

## Results (2026-10-24)

The copy ran on the CUDA machine from 15:08 to 18:36 and `cross` to 22:16,
both measured by 22:50, the copy's earlier checkpoints by 22:56 (the report
and the runs' metrics copied to `analysis/specialist-distillation/`).
Nothing failed: no duel stopped by the engine or cut off, at most 0.62 GiB
of the card in use.

1. **The copy keeps each champion's mirror: Blackwing's; Cyber's not
   decided.** Against `all71` by its checkpoints after 320, 400 and 480
   updates, the deals paired with its teacher's: in the Blackwing mirror
   65.3% against Blackwing alone's 65.6%, −0.2 ± 0.5 points; in the Cyber
   mirror 75.0% against `cyber-old`'s 76.5%, −1.5 ± 0.5, with two standard
   errors taken away 2.5 under, past the margin.
2. **The gap closed after 124,800 games of the two decks against each
   other**, after 480 updates of `cross`, 1.04 hours of them: 63.0% ± 0.6%
   of Blackwing against Cyber against `all71`, −0.5 ± 0.7 points to the
   network of both's 63.5%. The network of both came within the same mark
   after 312,000, two and a half times as many.

Beside them, without a mark:

- **The two curves**, Blackwing against Cyber against `all71`, by the games
  of the two decks against each other:

  | Games of the two decks | Distilled, then paired | The network of both | Difference |
  | --- | --- | --- | --- |
  | 0 (each copy) | 52.1% | 28.9% | +23.1 |
  | 62,400 | 60.9% | 54.8% | +6.1 |
  | 124,800 | 63.0% | 58.3% | +4.7 |
  | 187,200 | 63.2% | 60.8% | +2.4 |
  | 249,600 | 65.6% | 62.1% | +3.5 |
  | 312,000 | 66.3% | 63.7% | +2.6 |
  | 374,400 | 67.2% | 64.7% | +2.5 |

  Each point ± 0.6%. The copy of the champions starts where they are, about
  52%; the network of both's copy of the pilots at 28.9%.
- **The copy along its updates**, each mirror against `all71` over its
  teacher: Blackwing −3.8, −2.5, −1.6, −0.7, −0.2 and +0.2 points after 80,
  160, 240, 320, 400 and 480 updates; Cyber −7.4, −3.6, −2.4, −0.7, −1.9 and
  −1.8. It came within a point of each teacher from 320 updates, some
  83,000 of each champion's games; Blackwing still rose at 480, Cyber had
  levelled off from 320.
- **Head to head with its teachers**: 49.6% ± 0.4% of the Blackwing mirror
  against Blackwing alone, 48.3% ± 0.4% of the Cyber mirror against
  `cyber-old`.
- **The mirrors along `cross`**, which plays none, over each teacher
  against `all71`: Blackwing's 1.5 to 3.3 points under from its first
  checkpoint on; Cyber's 1.2 to 2.1 points over up to 1,200 updates, and
  4.1 under after 1,440.
- **The hours**: the copy 3.46, 26 seconds an update (each mirror game
  imitated from both seats, both teachers asked at each decision); `cross`
  3.66, 1.04 of them to the second reading. The whole way to the mark: 7.72
  hours with the champions made one after the other (Blackwing alone 1.85,
  `cyber-old` 1.37), 6.35 with the champions made at once; the network of
  both to the same mark 5.41, its whole run 6.57.
- **The games against itself grew long**: from 7 turns in the first update
  to 15 to 19 from 240 on, the updates from 6 seconds to 9 to 11;
  Blackwing's seat won 65% to 71% of them.

What it says: champions imitated into one network, and then their decks
against each other, need two and a half times fewer games of the pairing
to play it as well as a network that learnt everything together, and end
better at it: 2.5 points at as many games. The copy keeps Blackwing's mirror
and comes within 1.5 points of Cyber's. For two decks the way is not
cheaper in hours: the copy took 3.5 hours and the champions were made
first. With many decks the pairings are most of the work, and champions can
be made at once on many machines: a pairing that needs 2.5 times fewer
games is what decides.

## Closed (2026-10-24)

Closed that night, once its report was in, as decided that evening: its
second reading sets how the network of four decks is made. The next
experiment is [A network of four decks](2026-10-24-four-decks.md).
