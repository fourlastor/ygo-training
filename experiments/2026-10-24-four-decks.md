# A network of four decks: two more champions, then four decks distilled and paired

**Status:** ACTIVE

**Proposed:** 2026-10-24

**Activated:** 2026-10-24, when
[Specialist distillation](2026-10-24-specialist-distillation.md) was read and
closed. Its design is in two parts: the champions of two more decks, fixed
that evening before anything was run; and the network of the four decks,
made the way specialist distillation found cheaper in games of the
pairings, its design fixed once the champions are made.

**Last updated:** 2026-10-25 (part 1 read: all three champions; two seeds
of the recipe 48.1% head to head; the queue's first five champions too)

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

### Results (2026-10-25)

Run on the CUDA machine from 23:19 on 2026-10-24 to 02:45 the next morning,
at `7e568fbf` (the clocks went back an hour at 03:00 that night; the times
are local), into `runs/four-decks` and `analysis/four-decks`. Nothing
failed: no duel stopped by the engine or cut off.

1. **Lightsworn: a champion**, 56.5% ± 0.5% of its mirror against `all71`;
   1.17 hours.
2. **Infernity: a champion**, 63.0% ± 0.5%; 1.57 hours.
3. **Cyber at seed 2013: a champion**, 75.9% ± 0.4%; 1.44 hours.

Beside them, without a mark:

- **Two seeds of the recipe**: Cyber at seed 2013 wins 48.1% ± 0.4% head to
  head against `cyber-old`, and against `all71` 0.5 ± 0.5 points under it
  (75.9% against 76.4%, the deals paired). Two runs on the same pilot land
  within 2 points of each other head to head and half a point against
  `all71`. The calibration's 57.3% between `cyber-new` and `cyber-old`, which
  copied different pilots, is so mostly the pilot: `cyber-new` was 7.3 points
  ahead head to head though 2.0 under against `all71`.
- Lightsworn's champion is the narrowest over `all71`, which is strongest on
  Lightsworn among the thirteen (75.6% against their pilots).
- Cyber's mirror games at seed 2013 ran to 26 turns by its last updates,
  Infernity's to 12, Lightsworn's to 9.
- **The queue's champions**, read by the same rule: Gishki 65.5% ± 0.5%
  (1.57 hours), Burn 63.8% ± 0.4% (1.82), Crystal Beasts 63.6% ± 0.6% (0.98),
  Dragunity 62.8% ± 0.5% (1.53), Gladiator Beasts 67.7% ± 0.5% (1.63):
  champions, all five. Burn's beats `all71` in its mirror though `all71`
  plays Burn no better than its pilot against the thirteen.

## Part 2: the network of four decks (design fixed on 2026-10-25)

Specialist distillation's mirrors faded in a phase that played none, and a
pairing's gains slowed as it neared its level: a mirror is a pairing as any
other in what it needs. Hence two arms of as many games, one with every
pairing as often as any other and one with the games given to each pairing
by how far behind it is.

- **The copy (`taught`).** One network of Blackwing, Cyber, Lightsworn and
  Infernity from scratch, seed 2012, each deck's games played by its
  champion by its first choice and imitated: Blackwing alone, `cyber-old`
  (as in specialist distillation), and part 1's champions of Lightsworn and
  Infernity, each by its last checkpoint. 960 updates of 520 duels (240 a
  deck, 124,800 of each champion's games, as specialist distillation's copy
  had), at 3e-4, in minibatches of 256 with no duels in flight, one pass,
  the value taught at four positions a game. Of an update's duels, 260 are
  each champion against itself in its own mirror, 65 a deck, both seats
  imitated, and 260 against the thirteen decks' pilots, 5 in each of the 52
  pairings. Its teachers' games in two groups and from CUDA graphs (an
  update's duels at once, 26 slots; graphs up to 130 decisions), decided
  from the measurement of 2026-10-24: 17.7 seconds an update against 24.1. A
  checkpoint every 80 updates. A copy of 640 updates may cost under a point
  and save some 1.7 hours; 960 is kept so as to compare with specialist
  distillation, and 640 is to be measured later by an arm of PPO alone from
  the copy's checkpoint there.
- **Two arms of PPO from the copy's last checkpoint,** 2,880 updates each at
  1e-4 with `fast`'s settings, a checkpoint every 240. Of an update's 520
  duels, 260 against itself and 260 against the thirteen decks' pilots, 5
  in each pairing, in both.
  - **Uniform (`uniform`)**: its duels against itself on the 16 ordered
    pairings of its decks as often as any other, the mirrors among them.
  - **Weighted (`weighted`)**: its duels against itself shared out by
    weights (`train --self-play-weights`, built for it). Before every 240
    updates, its last checkpoint plays each ordered pairing against `all71`
    on the other side, 128 games each
    (`eval_suites/four-decks-pairings-quick-checkpoint.json`). A pairing's
    share is the learner's there, a mirror's from both seats, two decks' the
    mean of their two seatings, and the mean of this measurement and the one
    before it. Each pairing weighs (1 − share)² times its seatings, at least
    a quarter of its share where every ordered pairing counts the same; two
    decks' weight is split evenly between their seatings
    (`tools/pairing_weights.py`). A pairing where `all71` happens to be
    strong looks behind, and draws games, though the network may be near its
    level there: the second reading tells whether that costs.
- Four tasks of high priority on the CUDA machine's queue, one after the
  other (`experiments/commands/four-decks.sh`, `PARTS` copy, uniform,
  weighted and measure): about 5, 7 and 7 hours, and 1 to measure.
- **Measured afterwards**, every deal from both seats where a suite plays
  one deck against another: each teacher and the copy's checkpoints after
  800, 880 and 960 updates, and after 160, 320, 480 and 640, against `all71`
  in each deck's mirror; each arm's last three checkpoints on every ordered
  pairing against `all71`, 1,024 games each
  (`eval_suites/four-decks-pairings-checkpoint.json`), and every other
  checkpoint on the 128 games of each; the uniform arm after 1,920 updates in
  Blackwing against Cyber against `all71`, both ways, as specialist
  distillation's cross pairings were measured; the copy's and each arm's
  last checkpoint head to head against each champion in its mirror.

### Readings (fixed on 2026-10-25, before anything was built)

1. *The copy keeps each champion's mirror*: for each deck, the copy's share
   against `all71` in that deck's mirror, by its checkpoints after 800, 880
   and 960 updates, a deal's games at the three together, less its
   teacher's share on the same deals: it **keeps** the champion's mirror
   when the gap less two standard errors is above −2 points, **loses
   strength** when the gap plus two standard errors is below −2, otherwise
   **not decided**.
2. *Weighting by weakness pays*: the weighted arm against the uniform arm,
   by their checkpoints after 2,400, 2,640 and 2,880 updates, by the mean
   over the ten pairings of the learner's share against `all71` (a mirror
   from both seats, two decks the mean of their two seatings), the deals
   paired: it **pays** when the weighted arm is at least 1 point over by two
   standard errors, **costs** when it is at least 1 point under by two
   standard errors, otherwise **not told apart**. A point, not two: a mean of
   ten pairings has a smaller error than one pairing.
3. *Four decks against two, for one pairing*: Blackwing against Cyber
   against `all71`, the uniform arm after 1,920 updates against specialist
   distillation's cross pairings after 240, each having played the two
   decks against each other 62,400 times, the deals paired: sharing four
   decks **helps** a pairing when the uniform arm is at least 2 points over
   by two standard errors, **hurts** it when it is at least 2 points under
   by two standard errors, otherwise **not told apart**.

Told beside them, without a mark (`tools/four_decks_report.py`): each
pairing's curve in each arm; the weighted arm's weights along it; the
weakest pairing of each arm; the copy along its updates; the copy and each
arm head to head against each champion in its mirror; the hours, and those
the weighted arm spent measuring.

## Beside it: luna's queue

From 2026-10-25 the CUDA machine takes its work from a queue
([docs/luna-queue.md](../docs/luna-queue.md)). Behind this experiment's own
runs, as work given beforehand at low priority, the champions of the other
ten of the thirteen decks by the same recipe, 2.5 hours each: Gishki, Burn,
Crystal Beasts, Dragunity, Gladiator Beasts, HEROes, Monarchs, Morphtronics,
Rock-Block and Spellcasters (`runs/champions`, `analysis/champions`). Each is
read by the rule of a champion above, without a mark: the way to a network
of many decks needs a champion of each.
