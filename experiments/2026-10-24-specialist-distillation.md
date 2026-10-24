# Specialist distillation: two champions imitated into one network, then their decks against each other

**Status:** ACTIVE

**Proposed:** 2026-10-24

**Activated:** 2026-10-24, when
[A network of two decks at the champions' games of each mirror](2026-10-24-two-decks.md)
was closed. Of two next steps, a network of four decks by the champion's
recipe and this one, this was taken first: it bears on making champions
apart, on many machines at once, and needs no new champion. **The design
and the readings were fixed before anything was built.**

**Last updated:** 2026-10-24 (built, and begun on the CUDA machine at 15:08)

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
