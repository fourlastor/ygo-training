# A network of four decks: two more champions, then four decks distilled and paired

**Status:** ACTIVE

**Proposed:** 2026-10-24

**Activated:** 2026-10-24, when
[Specialist distillation](2026-10-24-specialist-distillation.md) was read and
closed. Its design is in two parts: the champions of two more decks, fixed
that evening before anything was run; and the network of the four decks,
made the way specialist distillation found cheaper in games of the
pairings, its design fixed once the champions are made.

**Last updated:** 2026-10-26 (part 2 read: the copy keeps two mirrors and
two are not decided; weighting not told apart from the uniform arm; four
decks not told apart from two; the network about each champion's equal
in its mirror, over `all71` on every pairing; nine of the queue's
champions)

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
  Dragunity 62.8% ± 0.5% (1.53), Gladiator Beasts 67.7% ± 0.5% (1.63),
  HEROes 68.3% ± 0.5% (1.22), Monarchs 70.1% ± 0.5% (1.51), Morphtronics
  59.2% ± 0.5% (0.94), Rock-Block 63.3% ± 0.5% (1.51): champions, all nine.
  Burn's beats `all71` in its mirror though `all71` plays Burn no better
  than its pilot against the thirteen.

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

### Results (2026-10-26)

Run on the CUDA machine's queue from 10:31 on 2026-10-25 to 02:37 the next
morning: the copy (4.27 hours) and the two arms at `6142c469`, the
measurements at `0b5f89ed` (1.3 hours), into `runs/four-decks/network` and
`analysis/four-decks/network`. No duel failed; one of the weighted arm's 1.5
million was cut off at the cap of answers.

1. **The copy keeps Lightsworn's and Infernity's mirrors; Blackwing's and
   Cyber's are not decided**, each against `all71` less its teacher, the
   deals paired:

   | Mirror | The copy | Its teacher | Gap | Reading |
   |---|---|---|---|---|
   | Blackwing | 64.4% | 65.6% | −1.2 ± 0.5 | not decided |
   | Cyber | 74.7% | 76.5% | −1.8 ± 0.5 | not decided |
   | Lightsworn | 57.2% | 57.7% | −0.4 ± 0.4 | keeps |
   | Infernity | 63.1% | 63.8% | −0.7 ± 0.4 | keeps |

   Specialist distillation's copy of two of them had Blackwing −0.2 ± 0.5
   and Cyber −1.5 ± 0.5.
2. **Weighting by weakness: not told apart.** By the mean over the ten
   pairings against `all71`, the weighted arm 63.4% and the uniform 63.0%,
   +0.4 ± 0.3 points.
3. **Four decks against two, for Blackwing against Cyber: not told
   apart.** The uniform arm after 1,920 updates 60.9%, specialist
   distillation's cross pairings after 240 updates 60.9%, +0.0 ± 0.8 points,
   each after as many games of the pairing.

Beside them, without a mark:

- **Every pairing against `all71`**, by each arm's last three checkpoints:

  | Pairing | Uniform | Weighted |
  |---|---|---|
  | Blackwing mirror | 65.4% | 66.0% |
  | Blackwing and Cyber | 63.2% | 63.3% |
  | Blackwing and Lightsworn | 58.1% | 59.2% |
  | Blackwing and Infernity | 61.6% | 62.5% |
  | Cyber mirror | 79.7% | 77.2% |
  | Cyber and Lightsworn | 57.2% | 59.8% |
  | Cyber and Infernity | 62.8% | 64.0% |
  | Lightsworn mirror | 56.4% | 56.8% |
  | Lightsworn and Infernity | 60.7% | 60.3% |
  | Infernity mirror | 65.5% | 65.0% |

  Each mirror is over the mark of a champion against `all71` in either arm.
- **Head to head against each champion in its mirror**, by each arm's last
  checkpoint: the uniform arm 50.1% against Blackwing's, 52.7% against
  Cyber's, 49.5% against Lightsworn's, 48.1% against Infernity's; the
  weighted arm 49.0%, 45.3%, 49.9% and 48.9%; the copy 50.0%, 49.3%, 49.2%
  and 48.5%.
- **The weighted arm's weights** went most to Lightsworn's pairings, its
  mirror 7.7% to 9.3% of the duels against itself where each ordered
  pairing has 6.25% in the uniform arm: `all71` is strongest on Lightsworn,
  so its pairings looked furthest behind. The Cyber mirror, the strongest,
  drew the fewest, and is the weighted arm's one pairing under the uniform's
  (77.2% against 79.7%, and 45.3% head to head with its champion).
- **Along the updates**, the mean over the pairings against `all71`: both
  arms from the copy's 58.1% to 61.3% after 480 updates; the uniform arm
  then between 61.4% and 64.2%, 64.2% after 2,880, the weighted arm between
  61.7% and 65.4%, 65.4% after 2,880.
- **The copy along its updates**, each mirror less its teacher: after 640
  updates Blackwing −0.7, Cyber −3.1, Lightsworn −0.6 and Infernity −1.1;
  after 960 −1.1, −2.3, −0.3 and −1.2.
- **Hours:** the copy 4.27, the uniform arm 5.26, the weighted arm 5.03, of
  which 0.21 measuring its weights.

What it says: one network of four decks, the champions copied and then
played against itself on every pairing as often as any other, plays each
mirror about as well as its champion (48% to 53% head to head) and beats
`all71` on all ten pairings (56.4% to 79.7%), in 9.5 hours of the CUDA
machine. Weighting the pairings by weakness against `all71` does not pay;
four decks cost Blackwing against Cyber nothing against two.

## Beside it: luna's queue

From 2026-10-25 the CUDA machine takes its work from a queue
([docs/luna-queue.md](../docs/luna-queue.md)). Behind this experiment's own
runs, as work given beforehand at low priority, the champions of the other
ten of the thirteen decks by the same recipe, 2.5 hours each: Gishki, Burn,
Crystal Beasts, Dragunity, Gladiator Beasts, HEROes, Monarchs, Morphtronics,
Rock-Block and Spellcasters (`runs/champions`, `analysis/champions`). Each is
read by the rule of a champion above, without a mark: the way to a network
of many decks needs a champion of each.
