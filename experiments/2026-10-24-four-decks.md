# A network of four decks: two more champions, then four decks distilled and paired

**Status:** ACTIVE

**Proposed:** 2026-10-24

**Activated:** 2026-10-24, when
[Specialist distillation](2026-10-24-specialist-distillation.md) was read and
closed. Its design is in two parts: the champions of two more decks, fixed
that evening before anything was run; and the network of the four decks,
made the way specialist distillation found cheaper in games of the
pairings, its design fixed once the champions are made.

**Last updated:** 2026-10-24 (opened; the champions to begin on the CUDA
machine once specialist distillation's follow-up is done)

Specialist distillation found that champions imitated into one network, and
then their decks against each other, need two and a half times fewer games
of the pairing than a network that learns everything together. With many
decks the pairings are most of the work: four decks have twelve ordered
pairings to their four mirrors, 71 have 4,970 to 71. This asks whether that
way holds at four decks, and how the games each pairing needs change as
decks are added.

## Research question

Do champions of four decks, imitated into one network and then their decks
against each other, give one network that keeps each champion's mirror and
plays each pairing well; and how many games of each pairing does it need?

## Part 1: two more champions (design fixed on 2026-10-24)

- **Lightsworn and Infernity**, each by the recipe of Blackwing alone and
  the Cyber champion: its pilot copied from scratch for 240 updates of 520
  duels against the thirteen decks' pilots (3e-4, minibatches of 256, no
  duels in flight, one pass, the value taught at four positions a game),
  then PPO at 1e-4 for 720 updates with `fast`'s settings, half of each
  update's duels the mirror against itself and the others against the
  thirteen decks' pilots; a checkpoint every 120 updates; seed 2012. Two
  archetypes unlike Blackwing and Cyber, a deck that mills itself for its
  graveyard and a combo deck that plays with no cards in hand, both among
  the thirteen opponents; `all71` wins 75.6% and 66.8% of their games
  against the thirteen decks' pilots.
- **Cyber again, at seed 2013**, by the same recipe, pilot and settings as
  `cyber-old`, flag for flag: the two runs differ by their seed alone. Two
  decks' calibration set `cyber-new` against `cyber-old` at 57.3%, but those
  two had copied different pilots; this run tells the spread of seeds
  alone.
- One after the other on the CUDA machine
  (`experiments/commands/champions.sh`), about 2 hours each, from the night
  of 2026-10-24, after specialist distillation's follow-up.
- **Measured afterwards**: each run's last three checkpoints against `all71`
  in its deck's mirror, each deal from both seats
  (`eval_suites/lightsworn-against-checkpoint.json`,
  `eval_suites/infernity-against-checkpoint.json` and
  `eval_suites/cyber-against-checkpoint.json`, 2,048 deals each); Cyber at
  seed 2013 against `cyber-old` head to head, checkpoint for checkpoint,
  and `cyber-old` against `all71` again on the same deals.

### Readings (fixed on 2026-10-24, before anything was run)

1. *A champion*, for each run, as the Cyber champion's: its last three
   checkpoints against `all71` head to head in its deck's mirror, every
   deal from both seats, a deal's games together: **a champion** when it
   wins at least 52% by two standard errors, **weaker** at 48% or less by
   two standard errors, otherwise **not told apart**.

Told beside it, without a mark (`tools/champions_report.py`): Cyber at seed
2013 against `cyber-old` head to head, and its share against `all71` over
`cyber-old`'s on the same deals, the spread of two seeds of the recipe; the
hours.

## Part 2: the network of four decks (to be fixed once the champions are made)

Blackwing alone, `cyber-old` and the two new champions as its teachers,
their mirrors imitated as in specialist distillation, then the twelve
pairings against each other. Its design and readings are fixed before it is
built, with what the champions and the speed of the copy say
(`experiments/commands/teacher-calls.sh`, measured on the CUDA machine the
same night).

## Beside it: luna's queue

From 2026-10-25 the CUDA machine takes its work from a queue
([docs/luna-queue.md](../docs/luna-queue.md)). Behind this experiment's own
runs, as work given beforehand at low priority, the champions of the other
ten of the thirteen decks by the same recipe, 2.5 hours each: Gishki, Burn,
Crystal Beasts, Dragunity, Gladiator Beasts, HEROes, Monarchs, Morphtronics,
Rock-Block and Spellcasters (`runs/champions`, `analysis/champions`). Each is
read by the rule of a champion above, without a mark: the way to a network
of many decks needs a champion of each.
