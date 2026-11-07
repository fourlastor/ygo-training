# The expanded pool: from seventy-one decks to the cards around them

**Status:** ACTIVE

**Proposed:** 2026-11-01

**Activated:** 2026-11-02, when
[A network of the seventy-one decks](2026-10-28-seventy-one-champions.md) was read
and closed. Its design and readings are to be fixed before anything is built.

**Last updated:** 2026-11-06 (the batch again read: `distilled` keeps the mirrors and
the old decks while the new decks learn, so the batches after take its term and share)

The goal since 2026-10-24 has been a champion of all seventy-one decks and then the
pool around them. The first is released as
[seventy-one-champions](../models/seventy-one-champions.md). This record is the
second: a network that plays well beyond the decks it was made on.

## What there is to build on

- **The pool**: the 4,065 cards of the World Championship 2011 list, its 53 banned
  cards counted in: a model should know a card whatever the list of the day says,
  as agreed on 2026-11-02, and Heavy Storm, banned there and so in none of the
  seventy-one decks, is in half of the Decks below. The seventy-one decks hold 956
  of the 4,065, and 92.1% of the slots of the 2010 and 2011 tournament lists gathered
  in `analysis/meta-decks-2011`, counted in the pool (`tools/played_cards.py` reads
  them at the pool's limits).
- **What the games of the era field**: the opponents of the World Championship
  games of 2009 to 2011 and of the six Tag Force games, as Yugipedia's character
  pages list them, read through its API on 2026-11-02. 3,094 Decks of 40 to 60 cards
  are tied to a game by the pages themselves, what they hold of other games, Tag
  Force Special among them, and of the real world left out; 3,005 are different, of
  which 89 Turbo Decks, set aside for their Speed Spells, and 71 of Tag Force 6 with
  Xyz monsters. The lists stay in `analysis/wc-opponents/`, which no commit tracks.
  They come from the games and may be used, as agreed on 2026-11-02; the games'
  character names may not, in anything tracked.
  The seventy-one decks hold 62.2% of the World Championship Decks' slots in the
  pool and 57.1% of all of them, where they hold 92.1% of the tournament lists'. A
  World Championship Deck is held at 62% at the median, and 15% of them share half
  their main deck with one of ours, where 71% of the tournament lists do.

  | Added in turn | New cards | The pool held |
  |---|---|---|
  | The seventy-one decks | 956 | 23.5% |
  | World Championship 2009 to 2011 | 1,801 | 67.8% |
  | Tag Force 4 to 6 | 517 | 80.5% |
  | Tag Force 1 to 3 | 367 | 89.6% |
  | Turbo Decks | 18 | 90.0% |
  | Tournament lists | 6 | 90.2% |

  400 cards are in no Deck: 121 Spells, 114 Traps, 110 Effect, 46 Normal and 9 other
  monsters. Of the pool's cards 2,346 are in at least 5 of the Decks, 1,444 in at
  least 10. Each Deck chosen for the new cards it brings, the pool held and the
  Decks' slots held go from 23.5% and 57.1% to 38.5% and 72.6% with 25, 47.9% and
  81.3% with 50, 61.0% and 89.2% with 100, 75.1% and 96.6% with 200, 86.2% and 99.7%
  with 400; 523 bring all that the Decks hold. Chosen for the slots they bring, 25
  Decks hold 77.4% of the slots and 33.7% of the pool, and the two meet by 200.
  The games go past the pool too: Tag Force 6 has 5,311 cards and World Championship
  2011 4,231, by Yugipedia's counts. The standing Decks hold 833 cards outside the
  list, 691 of them in the card database, 4.6% of their slots, both older cards and
  newer ones.
- **The lists**: 278 of those 515 lists are complete, and 199 of them share at least
  half their main deck with a deck of the pool
  ([One deck, or one deck with its variants](2026-10-22-deck-variants.md), which
  stays FUTURE until this record decides whether it is the first step). Its sketch,
  agreed on 2026-10-26: real lists only, the copy on the decks' own lists and every
  game of PPO on variants, staged from one deck up. Not built: the conversion of
  the lists into `.ydk` files, and training on several lists of a deck.
- **What a network does with what it never met**: a network loses little to a list
  with cards it never trained on, some 2 points a copy
  ([Darker Than Black](2026-10-19-darker-than-black.md)), and much to a deck it never
  played, 18.8 points over training on it
  ([thirty-one decks](2026-10-18-thirty-one-decks.md)). The description of a card
  told it nothing measurable ([card descriptions](2026-10-17-card-descriptions.md)).
- **Adding to a network**: `decklight`, a public project on the same game, reports
  distilling new specialists into a network with a term to the network itself on
  everything else, which would add decks without copying all of them again: 85% of
  the specialists' gains kept and the earlier decks unchanged, from a weak base and
  against its own previous release. Not tried here.

- **Where the network stands on a Deck it never met**: the twelve Decks that bring
  the most new cards, 316 between them, chosen from the 1,907 different lists whole
  in the pool, each against the thirteen decks' pilots on 32 deals from both seats,
  nothing trained (`analysis/game-decks-probe.py`, 9,984 games an arm, none failed):
  `seventy-one-champions` wins 19.9% and the policies' generic pilot 14.4%, 5.5 ± 0.4
  points under it, the deals paired, over it by two standard errors on eight of the
  twelve and under it on none. So the generic pilot has nothing to teach it there.

## How the pool is taken on

The cost of a network grows with its decks and not with their pairings. PPO of the
seventy-one decks played some 1,200 duels of each pairing that was no mirror and
gained 7.6 points over all of them, where one matchup alone takes hundreds of
thousands: it learns decks and cards, not matchups. So the decks are taken on in
batches of some fifty, most duels with a new deck in one seat and the other drawn
from all of them, and with no share kept for the mirrors. Both seats of a duel teach
the network, so that the older decks go on being trained, at a share that falls as
the pool grows. Each batch gives a network to keep, and the batches stop when what
they add flattens. How many games a new deck takes from where the network stands
has never been measured, and it sets all of this.

What that thin spread cost was read on 2026-11-03: `seventy-one-champions` against
`fourteen-decks` on the fourteen decks, each against each, 32 deals from both seats,
12,544 games. Both had the same copy, 160 updates a deck, and some 321,000 games a
deck of PPO, the fourteen at 9,257 duels of each pairing and the seventy-one at 1,205.
`fourteen-decks` wins 52.0% ± 0.4%, 51.6% ± 0.5% with Cyber in neither seat, and plays
each of the fourteen decks better than the other does, from Crystal Beasts by 0.8
points to Cyber by 10.4, each ± 2.3. It leads against the pilots on all fourteen, by
2.5 points on average. The two are level in the mirrors against the champions (49.1%
and 49.4%) and against a deck neither met (56.6% and 56.7% of the 640 deals of each of
thirteen decks). So the spread cost about two points of a head to head, on matchups
the narrower network saw 7.7 times as often, against the 57 decks it gained. Giving
all 71 decks that volume of each pairing would take some 49 million duels, 10 to 11
days on the CUDA machine.

## The first batch test (design and readings fixed on 2026-11-02, before anything was built)

- **Decks**: the first twelve of the games' Decks by the new cards they bring,
  `game-0001` to `game-0012`, 316 new cards; the next twelve held out. The games'
  character names go nowhere tracked.
- **Run**: PPO from `seventy-one-champions` at 1e-4, with its optimizer fresh, 520
  duels an update at the settings of `fast`, every duel against itself and none
  against a pilot, for 4,320 updates, some 12 hours. Three quarters of the duels have
  a new deck in one seat and any of the 83 decks in the other, a quarter are between
  the old decks: 37 games a new deck and 8.4 an old deck an update. A checkpoint
  every 60 updates. Its first start was stopped after some 20 updates, before it
  counted for anything: `allocate_weighted_pairs` gave every duel left over to the
  largest remainders, so that of pairs each with less than a duel to its share the
  largest took them all, 55 games a new deck, 5 an old deck and none between old
  decks. A pair's share is now its whole duels and one more with the chance of what
  is left of it, which keeps the weights' proportions over the updates. By the same
  rule the mirrors of the seventy-one decks had 27.3% of their duels and not a
  quarter.
- **Measured** after 540, 1,080, 2,160 and 4,320 updates, some 20,000, 40,000, 80,000
  and 160,000 games a new deck, each by the mean of its last six kept checkpoints,
  and at the start by `seventy-one-champions` itself: each new deck against the
  frozen network at each of the 71 decks, 64 games a pairing from both seats (4,544
  a deck, `eval_suites/first-batch-pairings-checkpoint.json`), and every ordered
  pairing of the 71 decks head to head against the frozen network, 4 games each
  (20,164 games). Against the frozen network because these decks have no pilot or
  champion to be read against, and it plays the 71 decks at a level that does not
  move.

Readings:

1. *A new deck learns from here*: the mean over the twelve of each deck's share
   against the frozen network after 2,160 updates, less its share at the start, the
   deals paired: **learns** when at least 5 points by two standard errors, **does
   not** when at most 2 points by two standard errors, otherwise **not decided**.
2. *How many games a deck takes*: the fewest of the first three points whose gain is
   within 2 points of the last point's by two standard errors, the deals paired, sets
   a batch's budget. Where none is, the decks are still rising at 160,000 games.
3. *The old decks hold*: the trained network head to head against the frozen one on
   the 71 decks' pairings after 4,320 updates: **hold** when at least 49% by two
   standard errors, **slip** when at most 49% by two standard errors, otherwise **not
   decided**.

Told beside them, without a mark, after 4,320 updates: the next twelve Decks against
the frozen network as the first twelve, trained on nothing, to tell whether what was
learnt carries over; the old mirrors head to head against their champions, 128 deals
from both seats; the new decks against the thirteen decks' pilots, beside the 19.9%
they start from; the hours.

If the new decks learn and the old ones hold, the batches go on at reading 2's budget.
If the old decks slip, their share goes up, or a term holding the network to itself
on them is tried, as `decklight` reports. If the new decks do not learn, the
champions' recipe is tried on them.

## The first batch test: results (read on 2026-11-03)

The run took 10.1 hours on the CUDA machine and its measure 54 minutes, from
`seventy-one-champions` (`analysis/expanded-pool/first-batch/report.txt`).

1. *A new deck learns*: **learns**, +9.2 ± 0.2 points against the frozen network
   after 2,160 updates, from 16.4%. Each of the twelve gains, from +2.2
   (`game-0009`) to +21.2 (`game-0005`).
2. *How many games a deck takes*: **2,160 updates**, some 80,000 games a new deck,
   within 2 points of the last point's +10.5 ± 0.2 after 4,320; 540 updates (+5.4)
   and 1,080 (+7.6) are not.
3. *The old decks hold*: **not decided**, 49.6% ± 0.4% head to head against the
   frozen network after 4,320 updates, 49.4% to 49.6% at every point along the run.

Beside them:

- The next twelve Decks, trained on nothing, gained 1.3 ± 0.1 points from 12.1%:
  what the network learns of a deck carries little to one it does not train.
- The new decks against the thirteen decks' pilots: 32.2% from 20.9%, +11.2 ± 0.3.
- The old mirrors head to head against their champions: 44.1%, where
  `seventy-one-champions` stood at 50.4%, the lowest `deckout` and `reptilianne` at
  30.9%. The weights gave the 71 mirrors 0.35% of the duels, some 110 a mirror over
  the run, and they fell as the fourteen decks' did with no share of their own
  (44.9%).
- 220 duels of some 2.25 million failed, every one with `game-0001` or `game-0012`
  in a seat: Fake Feather 117, Beast Rising 89, seven other cards 14 between them.
  Beast Rising reads its target without checking there is one (line 45 of
  `c84962466.lua`); Fake Feather takes the first card of an empty group (line 40 of
  `c22628574.lua`) and at line 51 finds no Effect where it kept one.

The rules fixed with the readings cover neither new decks that learn while the old
ones are not decided nor the mirrors' fall, told beside without a mark.

## The first batch (design and readings fixed on 2026-11-03, before anything was built)

- **Start**: the first batch test's network, the mean of its last six kept checkpoints
  after 4,320 updates (step 37,440), its twelve Decks old decks from here.
- **Decks**: the next fifty of the games' Decks by the new cards they bring,
  `game-0013` to `game-0063` but `game-0037`, 718 new cards; the next twelve,
  `game-0064` to `game-0076` but `game-0065`, held out. The old decks are 83, the
  seventy-one and `game-0001` to `game-0012`. `game-0037` and `game-0065` each hold a
  card that names a card, Cloak and Dagger and Archfiend's Oath, as 7 of the 484 lists
  do: the question offers 4,096 answers, and the pool pads every decision in flight to
  the widest of them. The first start, with `game-0037`, ran out of the GPU's memory
  in its first update on 2026-11-03. Those lists wait until the pool asks such a
  question apart.
- **Run**: PPO at 1e-4 with its optimizer fresh, 520 duels an update at the settings
  of `fast`, every duel against itself: a quarter in the mirrors of all 133 decks, as
  the fourteen decks' `quarter` arm kept its mirrors, and of the rest three quarters
  with a new deck in one seat and any other deck in the other, a quarter between two
  old decks. An update gives a new deck 9.98 games, an old one 6.52 and a mirror 0.98
  duels, so that 8,040 updates give a new deck some 80,000 games, the test's budget:
  some 19 hours. A checkpoint every 60 updates (`experiments/commands/pool-batch.sh`).
- **Measured** after 4,020 and 8,040 updates, each by the mean of its last six kept
  checkpoints, and at the start by the network it starts from: each new deck against
  that network frozen at each of the 83 old decks, 32 games a pairing from both seats,
  and every ordered pairing of the old decks head to head against it, 8 games each
  (55,112 games); after 8,040, each of the seventy-one mirrors head to head against its
  champion, 128 deals from both seats (`eval_suites/pool-batch-one-*.json`,
  `tools/pool_batch_report.py`).

Readings, at 8,040 updates:

1. *The new decks learn*: the mean over the fifty of each deck's share against the
   frozen network, less at the start, the deals paired: **learns** when at least 5
   points by two standard errors, **does not** when at most 2 points by two standard
   errors, otherwise **not decided**.
2. *The old decks hold*: head to head against the frozen network on the old decks'
   pairings: **hold** when at least 49% by two standard errors, **slip** when at most
   49% by two standard errors, otherwise **not decided**.
3. *The mirrors come back*: the seventy-one mirrors against their champions, from
   44.1% at the start: **kept** when their mean is at least 48% by two standard errors
   and none is at most 45% by two standard errors, **lost** when the mean is at most
   48% by two standard errors or one is at most 45% by two standard errors, otherwise
   **not decided**.

Told beside them, without a mark: the same at 4,020 updates; the twelve held out
against the frozen network as the new decks, at the start and the end; the new decks
against the thirteen decks' pilots, at the start and the end; the hours.

If the mirrors do not come back, the batch is played again from
`seventy-one-champions` with all sixty-two Decks and the same share in the mirrors,
some 26 hours.

## Card identities, beside the batches (fixed on 2026-11-03, before anything was played)

What a deck learns in a batch is bound to its cards' identity keys, read by the trunk,
more than written into them. On a probe of the first batch test's twelve Decks against
twelve old decks, 1,152 games an arm, putting the 318 new cards' identity rows back to
their untrained values cost 0.4 ± 1.0 of a 10.1-point gain, and drawing them afresh 6.3
± 1.3 and 6.7 ± 1.3 points on two draws; training moved a row by about 1.1 against a key
of about 6.9. A card the network never trained on keeps its random key. Two steps beside
the batches, each read on its own:

1. *What an unseen card's key costs at play*, on this machine's CPU, with no training.
   The test's network (`runs/expanded-pool/first-batch-mean/4320.pt`, the first batch's
   start), its identity rows that no update moved rewritten (`tools/identity_keys.py`):
   blank, the mean of the rows training moved, or drawn afresh on two seeds; a row
   training moved, and row 0, the code of a card the seat cannot see, is never touched.
   In the twelve Decks the first batch holds out, 58% of the copies have such a low row
   and 26% both halves. Each arm plays those Decks against the network as it is at each of
   the 83 old decks, 8 deals from both seats from deal 60,000, 15,936 games, and the
   network as it is plays them against each arm (`experiments/commands/identity-keys.sh`).
   A random key is read as the mean of three draws, the network itself and the two drawn
   afresh, the deals paired.
   1. *Blank keys, playing unseen cards*: blank less the random keys, at the held-out
      Decks: **help** when at least +1.0 point by two standard errors, **hurt** when at
      most −1.0 point by two standard errors, otherwise **not decided**.
   2. *Blank keys, facing unseen cards*: the same, the arm at the old decks against the
      held-out Decks.

   Told beside them, without a mark: the mean keys by the same rule, the three random
   draws, what a key's luck is worth, and the held-out Decks one by one. If blank keys
   help, a new card starts from a blank key in step 2, and a network seated against decks
   it never met plays with blank keys; if they hurt, random keys stay.
2. *Identity rows that learn faster* (design and readings fixed on 2026-11-03, before
   anything was built). The first batch as it runs is the control. The fast arm is the same
   batch, from the same network on the same decks, weights and seed, with the two identity
   tables at ten times the learning rate in an optimizer group of their own
   (`--identity-learning-rate-scale 10`), queued after the first batch on the CUDA machine
   and played to 4,020 updates, some 40,000 games a new deck and some 10 hours
   (`experiments/commands/pool-batch-one-fast.sh`). Both are read at 4,020 updates against
   the same frozen start on the same games, deals paired (`tools/pool_batch_compare.py`).
   Each is one run, and what a run's own luck is worth is not in the errors.
   1. *It learns faster*: the fast arm's gain less the first batch's: **faster** when at
      least +1.0 point by two standard errors, **not faster** when at most +0.3 points by
      two standard errors, otherwise **not decided**.
   2. *The old decks are no worse*: the fast arm's head-to-head share on the old decks'
      pairings less the first batch's: **no worse** when at least −1.0 point by two
      standard errors, **worse** when at most −1.0 point by two standard errors, otherwise
      **not decided**.

   Told beside them, without a mark: both arms' mirrors against their champions at 4,020
   updates, the first batch's measured there for this, the held-out Decks, and where what is
   learnt lives. If it learns faster and the old decks are no worse, the fast arm goes on to
   8,040 updates as the first batch's network and the batches after take the faster tables;
   otherwise it stops at 4,020 and the second batch starts from the first.

## Card identities, step 1: results (read on 2026-11-03)

The arms played 15,936 games each on this machine's processor, 1 hour 52 minutes in all
(`analysis/identity-keys/report.txt`). With random keys the held-out Decks win 13.4%
against the old decks.

1. *Blank keys, playing unseen cards*: **not decided**, +0.8 ± 0.2 points against the
   random keys: more than nothing, not shown to reach a point.
2. *Blank keys, facing unseen cards*: **not decided**, +0.4 ± 0.1 points: under a point
   by two standard errors, though the reading has no verdict for that.

Beside them:

- The mean keys: +0.5 ± 0.2 playing, +0.3 ± 0.1 facing, both not decided.
- What a key's luck is worth: playing, the three random draws win 14.0% (the network's
  own keys), 13.2% and 13.0%, up to 1.0 point apart, some 0.5 points a draw beyond what
  their games tell; facing, 86.0%, 86.0% and 85.9%, nothing. With that luck in the
  error, blank keys playing are +0.8 ± 0.4. Against the keys the network has, blank
  keys are +0.2 ± 0.2 playing and +0.4 ± 0.2 facing.
- Deck by deck, a key's luck is worth several points. `game-0072` wins 9.8% with the
  network's own keys, 9.6% blank and 9.9% with the mean, but 5.4% and 5.6% with the two
  drawn afresh; `game-0076` wins 34.2% with its own keys, 28.7% and 31.8% drawn afresh
  and 28.2% blank. Blank keys gain most at `game-0066`, +4.1, and lose most at
  `game-0076`, −3.4.

Blank keys do not help by either reading, so nothing changes: a new card starts from a
random key in step 2, as in the first batch, and a network seated against decks it
never met plays with the keys it has.

## The first batch: results (read on 2026-11-04)

The run took 20.3 hours on the CUDA machine and its measure an hour
(`analysis/expanded-pool/batch-one/report.txt`).

1. *The new decks learn*: **learns**, +8.7 ± 0.1 points against the frozen network
   after 8,040 updates, from 17.4%. Every one of the fifty gained, from +0.2
   (`game-0019`) to +28.5 (`game-0040`); five gained under two points: `game-0019`,
   `game-0046` +0.4, `game-0025` +0.6, `game-0054` +1.6 and `game-0034` +1.8.
2. *The old decks hold*: **hold**, 49.5% ± 0.2% head to head against the frozen
   network.
3. *The mirrors come back*: **lost**, 45.7% ± 0.3% against their champions from 44.1%.
   Five are at most 45% by two standard errors, `deckout` 29.7%, `destiny-hero`
   33.2%, `heroes` 37.1%, `gusto` 37.9% and `monarch` 37.9%, and none is at least 50%
   by two. Deck by deck, 47 of the 71 stand higher than with the test's network.

Beside them:

- At 4,020 updates: the new decks +7.4 ± 0.1, the old decks 49.2%. The second half
  added 1.3 points.
- The twelve held out, trained on nothing: +1.0 ± 0.2 points from 13.5%.
- The new decks against the thirteen decks' pilots: 29.5% from 19.4%, +10.2 ± 0.2.
- A quarter of the duels over 133 mirrors gave each 0.98 duels an update, some 7,900
  over the run. The fourteen decks' quarter arm gave each of its mirrors some 9.3 an
  update and kept them (49.1%).
- The run stopped twice on the engine, at updates 3,871 and about 4,300: a duel that
  filled its 256 MB arena aborted the process when it was destroyed. It went on each
  time from its last checkpoint. The engine was mended (the fork's `da75a41` and
  `f0710af`, the policies' `5a0cc20`, `1c793600` here): a full arena now ends its duel
  and nothing else. From update 5,002 the run went on at `430b2ca8` without arenas, 11%
  slower (9.96 s an update against 8.9 to 9.06 s), and from 5,192 at `7306db2d` with
  the mended arenas (8.61 s). Every game is the same on the three engines except where
  a duel fills an arena: the policies' round robin of all pilots, 19,880 games, and
  the bench's games for moving the pin have the same digests. After the mending, three
  duels filled their arenas and ended there: `game-0048` against `game-0051`,
  `game-0021` against `game-0040` and `game-0055` against `game-0048`.
- 688 duels of some 4.2 million failed: Planet Pollutant Virus 553 (`game-0050`),
  Beast Rising 109 (`game-0001`), Fake Feather 14 (`game-0012`); two each for
  Batteryman Fuel Cell's other effect, Pyro Clock of Destiny and Orbital Bombardment;
  one each for Mind Control, Raging Cloudian and Lightforce Sword; and the three full
  arenas. The scripts are collected through the experiment and mended together at its
  end, as agreed on 2026-11-03.

The new decks learn and the old decks hold, but the mirrors do not come back: a
quarter of the duels over 133 mirrors raised them 1.6 points in 8,040 updates. As the
design fixed, the batch is played again from `seventy-one-champions` with all
sixty-two Decks and the same share in the mirrors.

## The batch again, in two arms (design and readings fixed on 2026-11-04, before anything was built)

The first batch lost the mirrors, and its design has the batch played again from
`seventy-one-champions` with all sixty-two Decks and the same share in the mirrors.
Beside it runs an arm that holds the old decks by distillation and not by their duels.
A quarter of the duels over 133 mirrors gave each 0.98 duels an update, and a pool of
the games' Decks would need ever more of them for the same (agreed on 2026-11-04).

- **Start**: `seventy-one-champions` (step 33,120), its mirrors at 50.4%.
- **Decks**: new, the sixty-two of the first batch test and the first batch,
  `game-0001` to `game-0063` but `game-0037`; old, the seventy-one; held out, the
  same twelve.
- **Runs**: PPO at 1e-4 with its optimizer fresh, 520 duels an update at the settings
  of `fast`, every duel against itself, 9,000 updates, a checkpoint every 60. The cards'
  identity tables are as the fast arm reads: at ten times the learning rate where it
  learns faster and the old decks are no worse, at the same rate otherwise. One arm
  after the other on the CUDA machine, some 23 hours each.
  - **`quarter`**: the first batch's weights, a quarter of the duels in the mirrors of
    all 133 decks and of the rest a quarter between two old decks. Between two old
    decks that is 167 duels an update, 0.98 for an old mirror; a new deck plays 8.85
    games an update, some 79,650 over the run.
  - **`distilled`**: the old decks' duels, their mirrors and their pairings, at 0.4 of
    `quarter`'s weights, what that leaves going to the new decks' pairings. Between
    two old decks that is 67 duels an update, 0.39 for an old mirror; a new deck plays
    11.2 games an update. At each decision of an old deck's seat in a duel between two
    old decks, the divergence of the network's answers from those of
    `seventy-one-champions`, frozen, is added to PPO's loss with the weight PPO gives a
    transaction.
- **Measured** after 4,500 and 9,000 updates by the mean of the last six kept
  checkpoints, and at the start by `seventy-one-champions`, as the first batch was
  (`eval_suites/pool-batch-replay-*.json`); the seventy-one mirrors after 9,000.

Readings, at 9,000 updates, for each arm unless said:

1. *The mirrors are kept*: the seventy-one mirrors against their champions: **kept**
   when their mean is at least 48% by two standard errors and none is at most 45% by
   two, **lost** when the mean is at most 48% by two or one is at most 45% by two,
   otherwise **not decided**.
2. *The old decks hold*: head to head against the frozen network on the old decks'
   pairings: **hold** when at least 49% by two standard errors, **slip** when at most
   49% by two, otherwise **not decided**.
3. *The new decks learn*: the mean over the sixty-two of each deck's share against the
   frozen network, less at the start, the deals paired: **learns** when at least 5
   points by two standard errors, **does not** when at most 2 points by two, otherwise
   **not decided**.
4. *`distilled` against `quarter` on the new decks*: `distilled`'s gain less
   `quarter`'s, the deals paired: **more** when at least +1.0 point by two standard
   errors, **not more** when at most +0.3 points by two, otherwise **not decided**.

Told beside them, without a mark: the same at 4,500 updates; the twelve held out and
the new decks against the thirteen decks' pilots, at the start and the end;
`distilled`'s divergence from `seventy-one-champions` at its old decks' decisions along
the run; the hours.

If `distilled` keeps the mirrors and holds the old decks, the batches after take its
term and its share. If only `quarter` does, the quarter stays. If neither does, the
mirrors need more, designed then. The fast arm stops at 4,020 whatever it reads.

## Card identities, step 2: results (read on 2026-11-04)

The fast arm took 10.1 hours to 4,020 updates on the CUDA machine
(`analysis/expanded-pool/batch-one-fast/compare.txt`), and the first batch was measured
at the same point for it (`analysis/expanded-pool/batch-one-4020/`). Its measure stopped
once, four measures at once running out of the card's memory; the one that failed was
played again alone, as the batch script now does (`db4f9130`).

1. *It learns faster*: **not decided**, +0.6 ± 0.1 points: +8.0 against +7.4 over the
   frozen network.
2. *The old decks are no worse*: **not decided**, −0.8 ± 0.2 points head to head: 48.4%
   against 49.2%.

Beside them:

- The mirrors against their champions: 42.4% against the first batch's 44.9% at 4,020
  updates, twenty-two decks under 45% by two standard errors against ten.
- The twelve held out, trained on nothing: +0.8 ± 0.2 points more than the first
  batch's, +1.1 against +0.3 from 13.5%.
- The new decks against the thirteen decks' pilots: 28.7% against 27.9%.
- Where what is learnt lives, on this machine's processor: each network at the fifty new
  decks against the start at each old deck, 8,300 games an arm, the deals paired
  (`experiments/commands/identity-rows.sh`, `analysis/identity-rows/report.txt`). The
  batch met first 715 rows of the low table and 268 of the high, rows the start had
  untrained. The first batch moved them 1.0 against keys of 6.9: putting them back to
  their untrained values costs 0.9 ± 0.3 points of its +6.1 ± 0.4 there, 15%, and
  drawing them afresh 3.9 ± 0.4, 64%. The fast arm moved them 5.4 and 5.0: putting them
  back costs 3.7 ± 0.4 of its +7.6 ± 0.4, 48%, and drawing them afresh 5.1 ± 0.4, 67%.
  The rows of the cards trained before the batch moved 0.65 in the first batch and 4.1
  to 4.2 in the fast arm, against keys of 7.0 to 7.5; the rest of the network moved as
  far in both, 62 and 60 against 639.

Neither reading is met, so the fast arm stops at 4,020 and the identity tables keep the
network's learning rate; the two arms of the batch again take them so, as
`pool-replay.sh` read at `quarter`'s start. At the network's rate most of what a batch
learns of its new cards lies outside their rows, bound to their random keys; at ten
times it half lies in them, and the old cards' keys move six times as far, which the
mirrors' fall fits but does not show.

## The batch again: `quarter`'s results (read on 2026-11-05)

`quarter` ran on the CUDA machine from 2026-11-04 18:54 at `db4f9130`, 22.3 hours and
its measure an hour more (`analysis/expanded-pool/replay-quarter/report.txt`). Its
updates gave a new deck 8.84 games and an old one 6.93, as designed.

1. *The mirrors are kept*: **lost**, 45.1% ± 0.3% against their champions. Seven are
   at most 45% by two standard errors, `deckout` 35.9%, `destiny-hero` 36.3%, `venom`
   36.3%, `zombie` 36.7%, `dragunity` 38.3%, `infernity` 38.7% and `blackwing` 39.5%,
   each ± 2.3% to 2.9%, and none is at least 50% by two.
2. *The old decks hold*: **not decided**, 49.1% ± 0.2% head to head against the frozen
   network.
3. *The new decks learn*: **learns**, +8.9 ± 0.1 points against the frozen network
   after 9,000 updates, from 14.7%. Every one of the sixty-two gained, from +0.7
   (`game-0019`) to +23.4 (`game-0045`); five gained under two points, `game-0019`,
   `game-0025` +0.9, `game-0055` +1.1, `game-0046` +1.7 and `game-0054` +1.8.

Beside them:

- At 4,500 updates: the new decks +7.6 ± 0.1, the old decks 48.5%.
- The twelve held out, trained on nothing: +1.6 ± 0.2 points from 10.4%.
- The new decks against the thirteen decks' pilots: 29.2% from 18.6%, +10.7 ± 0.2.
- The mirrors deal by deal on the check's deals, against the mean of the last thirteen
  kept checkpoints of `seventy-one-champions` at 20,160 updates, the nearest that
  measure has (50.8% there): `quarter` −5.6 ± 0.4 points, sixty-eight of the
  seventy-one lower; the first batch −5.0 ± 0.4; the first batch test −6.6 ± 0.4.
  `quarter` against the first batch, −0.6 ± 0.4 (`tools/mirror_pairs.py`).
- 635 duels of some 4.7 million failed: Planet Pollutant Virus 445, Beast Rising 137,
  Fake Feather 35, Pyro Clock of Destiny 4, Mind Control 2, Lightforce Sword and
  Orbital Bombardment one each; new since the first batch, one each, Enemy Controller
  (`gusto` against `gladiator`, two old decks), Ally of Justice Unlimiter, Fog Control
  and Armed Sea Hunter; and six duels that filled their arenas and ended there.

A quarter of the duels over 133 mirrors does not keep them. From
`seventy-one-champions` they fell to where the first batch brought them back to from
the first batch test's 44.1%, some five points under their champions; the share fits
as the level it holds the mirrors at whatever the start, which the two runs do not
show. The new decks learn as in the first batch. By the design the quarter does not
stay: `distilled` reads whether its term keeps the mirrors, and if it does not either,
the mirrors need more, designed then.

## The batch again: `distilled`'s results (read on 2026-11-06)

`distilled` ran on the CUDA machine from 2026-11-05 18:21 at `db4f9130`, 23.3 hours, 9.3
seconds an update against `quarter`'s 8.9, and its measure some 50 minutes more
(`analysis/expanded-pool/replay-distilled/report.txt` and `compare.txt`). Its updates
gave a new deck 11.21 games and an old mirror 0.39 duels, as designed, and distilled some
11,400 decisions each.

1. *The mirrors are kept*: **kept**, 50.6% ± 0.3% against their champions. None is at
   most 45% by two standard errors. The lowest are `deckout` 41.0% ± 3.0%, `dragunity`
   43.4% ± 2.7%, `fortune-lady` 43.4% ± 2.3%, `ojama` 44.1% ± 2.7% and `arcana` 44.5% ±
   2.5%. The highest are `volcanic` 60.9% and `gem-knight` 58.6%.
2. *The old decks hold*: **hold**, 50.5% ± 0.2% head to head against the frozen network.
3. *The new decks learn*: **learns**, +9.2 ± 0.1 points against the frozen network
   after 9,000 updates, from 14.7%. Every one of the sixty-two gained, from +0.8
   (`game-0025`) to +22.4 (`game-0040`); five gained under two points, `game-0025`,
   `game-0019` +1.0, `game-0046` +1.5, `game-0054` +1.5 and `game-0034` +1.8.
4. *`distilled` against `quarter` on the new decks*: **not decided**, +0.3 ± 0.1 points,
   +9.2 against +8.9.

Beside them:

- At 4,500 updates: the new decks +8.2 ± 0.1, the old decks 50.3%.
- The old decks head to head, against `quarter`'s: +1.4 ± 0.2 points, 50.5% against
  49.1%.
- The mirrors deal by deal on the check's deals, against `seventy-one-champions` at
  20,160 updates (50.8% there): −0.1 ± 0.3 points, thirty of the seventy-one lower;
  against `quarter`, +5.5 ± 0.4 points, two lower (`tools/mirror_pairs.py`).
- The twelve held out, trained on nothing: +1.5 ± 0.2 points from 10.4%, as `quarter`'s
  (−0.0 ± 0.2).
- The new decks against the thirteen decks' pilots: 29.7% from 18.6%, +11.1 ± 0.2.
- The divergence from `seventy-one-champions` at the old decks' decisions: 0.0008 at
  the first update, 0.020 over the first thousand, 0.029 over the last, still rising
  slowly (0.0285 over updates 5,001 to 6,000).
- 838 duels of some 4.7 million failed (`tools/failed_duels.py`): Planet Pollutant
  Virus 606, Beast Rising 161, Fake Feather 47, Lightforce Sword and Ally of Justice
  Unlimiter two each, Orbital Bombardment one; new since `quarter`, Evil HERO Infernal
  Gainer two (`game-0038`), and one each Mei-Kou, Master of Barriers (`spellcaster`, an
  old deck, against `game-0018`), Great Shogun Shien (`six-samurai`, an old deck,
  against `game-0001`) and the union procedure (`proc_union.lua:143`, `game-0002`
  against `game-0061`); and fourteen duels that filled their arenas and ended there.

The term keeps what the duels did not. With 0.39 duels an old mirror an update against
`quarter`'s 0.98, the mirrors stay deal by deal where `seventy-one-champions` had them,
and the old decks stay a little over even against it, at some 5% more time an update.
The new decks learn as in `quarter`; the reading does not tell apart the 27% more games
a new deck had. By the design, the batches after take the term and the share.

## Batch two (design and readings fixed on 2026-11-06, before anything was built)

`distilled` kept the mirrors and the old decks while the new decks learnt, so batch two
takes its term and its share (agreed on 2026-11-06).

- **Start**: `distilled`'s mean at 9,000 updates (step 42,120).
- **Decks**: new, fifty: `game-0064`, `game-0066` to `game-0092` and `game-0094` to
  `game-0115`, among them the twelve held out until now; `game-0065` and `game-0093`
  each hold a card that names a card, a question of 4,096 answers. Old, 133: the
  seventy-one and the sixty-two of the batch again, `game-0001` to `game-0063` but
  `game-0037`. Held out: `game-0116` to `game-0127`.
- **Run**: PPO at 1e-4 with its optimizer fresh, 520 duels an update at the settings of
  `fast`, every duel against itself, 6,840 updates, a checkpoint every 60, the cards'
  identity tables at the network's rate. The first batch's weights, a quarter of the
  duels in the mirrors of all 183 decks and of the rest a quarter between two old decks,
  with the old decks' duels at 0.4 of them and what that leaves going to the new decks'
  pairings: a new deck plays 11.77 games an update, some 80,500 over the run, an old deck
  3.39, an old mirror 0.28 duels, and 76.8 duels an update are between two old decks. At
  each decision of an old deck's seat in a duel between two old decks, the divergence of
  the network's answers from the start's, frozen, is added to PPO's loss with the weight
  PPO gives a transaction. Some 18 hours on the CUDA machine.
- **Measured** after 3,420 and 6,840 updates by the mean of the last six kept
  checkpoints, and at the start by the start (`eval_suites/pool-batch-two-*.json`); the
  seventy-one mirrors after 6,840. With 133 old decks the measure takes some two hours.

Readings, at 6,840 updates:

1. *The mirrors are kept*: the seventy-one mirrors against their champions: **kept**
   when their mean is at least 48% by two standard errors and none is at most 45% by
   two, **lost** when the mean is at most 48% by two or one is at most 45% by two,
   otherwise **not decided**.
2. *The old decks hold*: head to head against the start, frozen, on every ordered
   pairing of the 133 old decks: **hold** when at least 49% by two standard errors,
   **slip** when at most 49% by two, otherwise **not decided**.
3. *The new decks learn*: the mean over the fifty of each deck's share against the
   start, less at the start, the deals paired: **learns** when at least 5 points by two
   standard errors, **does not** when at most 2 points by two, otherwise **not
   decided**.

Told beside them, without a mark: the same at 3,420 updates; the old decks' share for
the seventy-one and for the sixty-two game Decks apart; the twelve held out and the new
decks against the thirteen decks' pilots, at the start and the end; the divergence along
the run; the hours.

If all three are met, the batches go on the same way. If the mirrors or the old decks
are not kept, what changes is designed then.

## Batch two's results (read on 2026-11-07)

Batch two ran on the CUDA machine from 2026-11-06 19:46 at `2c7f2b5d`, 18.2 hours, 9.6
seconds an update, and its measure some two and a half hours more
(`analysis/expanded-pool/batch-two/report.txt`). Its updates gave a new deck 11.78 games
and an old one 3.39, as designed.

1. *The mirrors are kept*: **lost**, on one mirror alone. Their mean is 50.2% ± 0.3%
   against their champions, but `deckout` is at most 45% by two standard errors, 37.1% ±
   2.9%. The next lowest are `arcana` 42.2% ± 2.4%, `rock-block` 43.0% ± 2.5%,
   `dragunity` 43.4% ± 2.6% and `burn` 43.8% ± 1.9%. The highest are `karakuri` 58.6%
   and `ice-barrier` 57.0%.
2. *The old decks hold*: **hold**, 50.2% ± 0.1% head to head against the start, frozen,
   on every ordered pairing of the 133 old decks (141,512 games).
3. *The new decks learn*: **learns**, +8.9 ± 0.1 points against the start after 6,840
   updates, from 30.4%. Every one of the fifty gained, from +2.3 (`game-0107` and
   `game-0067`) to +29.9 (`game-0085`).

Beside them:

- At 3,420 updates: the new decks +7.7 ± 0.1, the old decks 50.3%.
- The old decks at 6,840 by the two decks of a pairing: both of the seventy-one 50.0% ±
  0.2%, both game Decks 50.5% ± 0.3%, one of each 50.2% ± 0.2%. Told by the deck the
  network plays, as the report told them until `ad031e77`, the seventy-one stood at
  63.3% and the game Decks at 35.2%. That split shows what the seventy-one are worth
  against the game Decks, whichever network plays them.
- The mirrors deal by deal on the check's deals (`tools/mirror_pairs.py`):
  - Against `seventy-one-champions` at 20,160 updates (50.8% there): −0.5 ± 0.3 points,
    thirty-seven of the seventy-one lower.
  - Against `distilled`: −0.4 ± 0.3, thirty-three lower.
  - The most lost against `seventy-one-champions` are `cyber`, −11.3 ± 3.7 (`distilled`
    −6.2), and `volcanic`, −8.2 ± 3.4.
- Scored as reading 1 scores them, `seventy-one-champions`' own check has two mirrors
  at most 45% by two standard errors: `deckout` 38.7% ± 2.9% and `rock-block` 40.2% ±
  2.4%. `distilled` has none; its `deckout` stood at 41.0% ± 3.0%.
  - Deal by deal, batch two's `deckout` is −0.8 ± 3.2 points from
    `seventy-one-champions`' and −3.1 ± 2.9 from `distilled`'s.
  - The reading scores a draw as lost to both networks. `deckout`'s mirror drew 19 of
    its 256 games here and 15 for each of the other two networks; the other seventy
    mirrors drew 13 between them.
- The twelve held out, trained on nothing: +1.4 ± 0.2 points from 33.5%.
- The new decks against the thirteen decks' pilots: 33.5% from 24.9%, +8.6 ± 0.2.
- The divergence from the start at the old decks' decisions: 0.0008 at the first update
  and 0.015 over the first thousand. Over the last 840 it was 0.023, still rising slowly
  (0.0226 over updates 5,001 to 6,000).
- 257 duels of some 3.56 million failed (`tools/failed_duels.py`).
  - Ninety-two filled their arenas and ended there. Eighty-five of them had `game-0085`
    in a seat, fifty-seven of those against `game-0043`. They came at an even rate from
    the first thousand updates on. `game-0085` plays traps that become monsters
    (Embodiment of Apophis, Metal Reflect Slime, Zoma the Spirit) beside Solemn Wishes
    and Fairy Box.
  - Cards seen before: Planet Pollutant Virus 88, Beast Rising 33, Mind Control 18, Fake
    Feather 3, and one each for Pyro Clock of Destiny, Mei-Kou, Master of Barriers and
    Orbital Bombardment.
  - With `game-0070`, every one at an effect that is not there: Great Shogun Shien 5,
    and four other Six Samurai cards 6 between them.
  - New since `distilled`: Lord British Space Fighter 4 (`game-0080`), Neo Flamvell
    Shaman 2 (`game-0109`), and one each for Domino Effect (`game-0019`) and Fire
    Princess (`burn`, an old deck). One duel ended on a recursive event trigger.

The new decks learn and the old decks hold, as in `distilled`. The mirrors stay where
`seventy-one-champions` had them, −0.5 ± 0.3 points deal by deal. Reading 1 is lost on
its floor alone: `deckout` at 37.1%, where `seventy-one-champions`' own check stands at
38.7% and fails the same floor, as its `rock-block` does. One mirror's 128 deals carry
some three points of error, so a mirror that `seventy-one-champions` plays at 38% to 41%
can fall under the floor in one run and not in the next. `distilled`'s `deckout`, at
41.0%, did not. By the design, what changes is designed now, before anything else is
built.

**Current next step:** what changes after batch two is to be designed and agreed before
anything is built. The next batch would start from batch two's network: the mean of its
last six kept checkpoints at 6,840 updates (step 48,960), the network measured here.
