# The expanded pool: from seventy-one decks to the cards around them

**Status:** ACTIVE

**Proposed:** 2026-11-01

**Activated:** 2026-11-02, when
[A network of the seventy-one decks](2026-10-28-seventy-one-champions.md) was read
and closed. Its design and readings are to be fixed before anything is built.

**Last updated:** 2026-11-14 (the board as its cards made from scratch at fourteen decks
read: not decided, `tokens` winning 49.85% of the deals against `pooled`, so the
seventy-one are not made from scratch with it; on 2026-11-13, batch seven fixed, from
`pooled`: the 113 usable lists not
yet taken at half batch six's length, none held out, the new decks' mark at 2 points;
before it the same day, batch six's `switched` read and the two arms head to head:
the switch not shown, `switched` winning 50.0% of the deals against `pooled`, a fifth of
a point behind it on the new decks and 6% slower an update, the old decks holding, the
new decks learning and the mirrors not decided in both; before that, on 2026-11-12, the
copy's width and rate read at seventy-one decks: twice
as wide and a third of the rate from halfway each fit the teachers' answers better, half
as wide worse, and head to head against the teachers twice as wide wins 2.3 points more
of its deals and half as wide 3.4 fewer; the board as its cards from
scratch at fourteen decks fixed,
a pair of networks made by the fourteen decks' recipe to tell whether the seventy-one are
made so; on 2026-11-11 batch six's `pooled` read, the old decks holding, the new decks
learning +3.7 points and the mirrors not decided, and `switched` stopped on a selection
with no answer and gone on; before that, on 2026-11-10, batch six fixed: a hundred new decks in two arms, the pool's
network as it is and given the board as its cards, read head to head; two measurements of
a review of the learning recorded beside the batches; batch five read: the mirrors kept
and the old decks hold, the new decks not decided at +4.6 points; the screen's second
session read: the board as
its cards ahead by 3.7 points for Blackwing and 2.5 for Cyber after twice the training,
neither a large lever, and the switch of a trained network not shown; before them, the
screen of the network under PPO read: for Cyber the board as its cards is stronger,
58.4% head to head, and the history is not shown to be needed)

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

## Batch three (design and readings fixed on 2026-11-07, before anything was built)

Batch two learnt its new decks and held the old ones, and its mirrors stayed where
`seventy-one-champions` had them deal by deal. Batch three goes on the same way, with two
changes, agreed on 2026-11-07. Reading 1 is read against `seventy-one-champions` deal by
deal, with no floor on any one mirror, since its own check fails one. The old decks are
measured at the end alone.

- **Start**: batch two's mean at 6,840 updates (step 48,960).
- **Decks**:
  - New, fifty: `game-0116` to `game-0165`, among them the twelve held out until now.
  - Old, 183: the seventy-one and the 112 game Decks of the batches before, `game-0001`
    to `game-0115` but `game-0037`, `game-0065` and `game-0093`.
  - Held out: `game-0166` to `game-0179` but `game-0167` and `game-0171`, which each
    hold a card that names a card.
- **Run**: as batch two's, over all 233 decks.
  - PPO at 1e-4 with its optimizer fresh, 520 duels an update at the settings of `fast`,
    every duel against itself, a checkpoint every 60, the cards' identity tables at the
    network's rate.
  - The first batch's weights, the old decks' duels at 0.4 of them and what that leaves
    going to the new decks' pairings. At each decision of an old deck's seat in a duel
    between two old decks, the divergence from the start's answers, frozen, is added to
    PPO's loss.
  - 7,200 updates. A new deck plays 11.10 games an update, some 80,000 over the run, an
    old deck 2.65 and an old mirror 0.22 duels; 79.8 duels an update are between two old
    decks. Some 19 hours on the CUDA machine.
- **Measured** at the start by the start, and after 3,600 and 7,200 updates by the mean of
  the last six kept checkpoints (`eval_suites/pool-batch-three-*.json`).
  - After both: each new deck against the start at each old deck.
  - After 7,200 alone: every ordered pairing of the 183 old decks head to head against
    the start, and the seventy-one mirrors against their champions.
  - At the start and after 7,200: the twelve held out, and the new decks against the
    thirteen decks' pilots.
  - Some three and a half hours.

Readings, at 7,200 updates:

1. *The mirrors are kept*: the seventy-one mirrors against their champions, deal by deal
   against `seventy-one-champions`' check at 20,160 updates on its deals, a draw half a
   win (`tools/mirror_pairs.py`): **kept** when the difference is at least −2 points by
   two standard errors, **lost** when at most −2 points by two, otherwise **not
   decided**. By this reading batch two's mirrors were kept (−0.5 ± 0.3), the first
   batch's and `quarter`'s lost (−5.0 ± 0.4 and −5.6 ± 0.4).
2. *The old decks hold*: head to head against the start, frozen, on every ordered
   pairing of the 183 old decks: **hold** when at least 49% by two standard errors,
   **slip** when at most 49% by two, otherwise **not decided**.
3. *The new decks learn*: the mean over the fifty of each deck's share against the
   start, less at the start, the deals paired: **learns** when at least 5 points by two
   standard errors, **does not** when at most 2 points by two, otherwise **not
   decided**.

Told beside them, without a mark: the new decks at 3,600 updates; the old decks by the
two decks of a pairing; the mirrors' mean against their champions and the lowest, and
deal by deal against batch two's; the twelve held out and the new decks against the
thirteen decks' pilots, at the start and the end; the divergence along the run; the
hours.

If all three are met, the batches go on the same way. If the mirrors or the old decks
are not kept, what changes is designed then.

## Batch three's results (read on 2026-11-08)

Batch three ran on the CUDA machine from 2026-11-07 17:06 at `62d1fd06`, 19.4 hours, 9.7
seconds an update, and its measure 2.2 hours more (`analysis/expanded-pool/batch-three/report.txt`).
Its first updates gave a new deck 11.08 games and an old one 2.65, as designed.

1. *The mirrors are kept*: **not decided**, −1.7 ± 0.3 points deal by deal against
   `seventy-one-champions`' check, forty-eight of the seventy-one lower. Against their
   champions they stand at 49.0% ± 0.3%. The most lost are `cyber` −14.5, `reptilianne`
   −8.6, `scrap` −7.8, `chaos` −7.0 and `macro-dd` −6.6; the most won `gishki` +4.7 and
   `gemini` +3.9.
2. *The old decks hold*: **hold**, 50.2% ± 0.1% head to head against the start, frozen,
   on every ordered pairing of the 183 old decks (267,912 games).
3. *The new decks learn*: **learns**, +6.9 ± 0.1 points against the start after 7,200
   updates, from 39.2%. Every one of the fifty gained, from +0.9 (`game-0152` and
   `game-0150`) to +23.1 (`game-0151`); three gained under two points, those two and
   `game-0154` +1.3.

Beside them:

- At 3,600 updates: the new decks +6.0 ± 0.1.
- The old decks at 7,200 by the two decks of a pairing: both of the seventy-one 49.9% ±
  0.2%, both game Decks 50.4% ± 0.2%, one of each 50.2% ± 0.1%.
- The mirrors deal by deal against each network before (`tools/mirror_pairs.py`):
  - `distilled` against `seventy-one-champions`: −0.1 ± 0.3 points.
  - Batch two against `distilled`: −0.4 ± 0.3.
  - Batch three against batch two: −1.2 ± 0.3, forty-eight lower.
  - `cyber` has fallen at each: −6.2, −11.3 and −14.5 against `seventy-one-champions`.
- The fall does not follow the cards a deck shares with the new decks. A mirror's change
  against batch two and the share of its deck's cards that the fifty new decks also
  hold correlate at +0.03. The third of the seventy-one sharing least, 0% to 42% of
  their cards, fell 1.2 points; the third sharing most, 62% to 93%, fell 1.1.
- The lowest mirrors against their champions: `deckout` 36.7%, `rock-block` 39.8%,
  `macro-dd` and `reptilianne` 42.2%, `dragunity` 42.6%.
- The twelve held out, trained on nothing: +1.0 ± 0.1 points from 36.6%.
- The new decks against the thirteen decks' pilots: 38.1% from 31.5%, +6.6 ± 0.2.
- The divergence from the start at the old decks' decisions: 0.0007 at the first update,
  0.013 over the first thousand and 0.021 over the last, level since some 5,000.
- 186 duels of some 3.74 million failed (`tools/failed_duels.py`).
  - Fifty-three filled their arenas and ended there, fifty-one of them with `game-0085`
    in a seat, fifty of those against a new deck.
  - Cards seen before: Planet Pollutant Virus 75, Beast Rising 25, Neo Flamvell Shaman
    4, Pyro Clock of Destiny 3, Fake Feather 3 and Orbital Bombardment 1, which
    `game-0117` holds too.
  - New since batch two: Gozen Match 8 and Rivalry of Warlords 7, both with
    `game-0099` at line 71 of their scripts; Star Blast 3 (`game-0013`); one each for
    Level Up! (`game-0095`) and Second Booster (`game-0110`). Two duels ended on a
    recursive event trigger.

The new decks learn and the old decks hold head to head against the start, but the
seventy-one mirrors fell 1.2 points from batch two's, and in all lie 1.7 points under
`seventy-one-champions`. The divergence holds each batch to its own start, so what a
batch loses in the mirrors stays lost in the batches after it. The new decks start
higher than batch two's (39.2% against 30.4%) and gain less, +6.9 against +8.9: the
start knows more of their cards. By the design, what changes is designed now, before
anything else is built.

## Batch four (design and readings fixed on 2026-11-08, before anything was built)

Batch four goes on as batch three with one change, agreed on 2026-11-08. In a duel between
two of the seventy-one, their decisions are held to `seventy-one-champions`, frozen,
instead of to the start, as `distilled` held them. Every other duel between two old decks
is held to the start as before. Since the hold no longer follows each batch's start, what
the seventy-one lose cannot carry into the next batch.

- **Start**: batch three's mean at 7,200 updates (step 56,160).
- **Decks**:
  - New, fifty: `game-0166` to `game-0217` but `game-0167` and `game-0171`, among them
    the twelve held out until now.
  - Old, 233: the seventy-one and the 162 game Decks of the batches before, `game-0001`
    to `game-0165` but `game-0037`, `game-0065` and `game-0093`.
  - Held out: `game-0218` to `game-0229`.
- **Run**: as batch three's, over all 283 decks.
  - PPO at 1e-4 with its optimizer fresh, 520 duels an update at the settings of `fast`,
    every duel against itself, a checkpoint every 60, the cards' identity tables at the
    network's rate.
  - The first batch's weights, the old decks' duels at 0.4 of them and what that leaves
    going to the new decks' pairings.
  - At each decision of an old deck's seat in a duel between two old decks, the
    divergence from a frozen network's answers is added to PPO's loss, at the weight PPO
    gives a transaction. Where both decks are of the seventy-one, the network is
    `seventy-one-champions` (`runs/seventy-one-champions/ppo-mean/21900.pt`, step
    33,120, the one `distilled` began from and was held to); otherwise it is the start.
  - 7,440 updates. A new deck plays 10.67 games an update, some 79,400 over the run, an
    old deck 2.17 and an old mirror 0.18 duels. Of the 81.8 duels an update between two
    old decks, 16.6 are between two of the seventy-one. Some 20 hours on the CUDA
    machine.
- **Measured** as batch three's, at the start by the start and after 3,720 and 7,440
  updates by the mean of the last six kept checkpoints (`eval_suites/pool-batch-four-*.json`):
  the old decks and the mirrors after 7,440 alone. Some three hours.

Readings, at 7,440 updates, as batch three's:

1. *The mirrors are kept*: the seventy-one mirrors against their champions, deal by deal
   against `seventy-one-champions`' check at 20,160 updates on its deals, a draw half a
   win (`tools/mirror_pairs.py`): **kept** when the difference is at least −2 points by
   two standard errors, **lost** when at most −2 points by two, otherwise **not
   decided**.
2. *The old decks hold*: head to head against the start, frozen, on every ordered
   pairing of the 233 old decks: **hold** when at least 49% by two standard errors,
   **slip** when at most 49% by two, otherwise **not decided**.
3. *The new decks learn*: the mean over the fifty of each deck's share against the
   start, less at the start, the deals paired: **learns** when at least 5 points by two
   standard errors, **does not** when at most 2 points by two, otherwise **not
   decided**.

Told beside them, without a mark: the new decks at 3,720 updates; the old decks by the
two decks of a pairing; the mirrors' mean against their champions and the lowest, and
deal by deal against batch three's; the twelve held out and the new decks against the
thirteen decks' pilots, at the start and the end; the divergence from each frozen network
along the run; the hours.

If all three are met, the batches go on the same way. If the mirrors or the old decks
are not kept, what changes is designed then.

## Batch four's results (read on 2026-11-09)

Batch four ran on the CUDA machine from 2026-11-08 15:09 at `5c0ef6eb`, 19.9 hours, 9.6
seconds an update, and its measure 3.0 hours more (`analysis/expanded-pool/batch-four/report.txt`).
Its first updates gave a new deck 10.68 games and an old one 2.17, as designed.

1. *The mirrors are kept*: **kept**, −0.9 ± 0.3 points deal by deal against
   `seventy-one-champions`' check, forty of the seventy-one lower. Against their
   champions they stand at 49.8% ± 0.3%. The most lost are `cyber` −11.7, `scrap` −8.2,
   `spirit` −7.4, `nordic` −7.0 and `chaos` −5.9; the most won `venom` +5.9 and `gusto`
   +4.7.
2. *The old decks hold*: **hold**, 50.2% ± 0.1% head to head against the start, frozen,
   on every ordered pairing of the 233 old decks (434,312 games).
3. *The new decks learn*: **learns**, +6.0 ± 0.1 points against the start after 7,440
   updates, from 43.8%. Every one of the fifty gained, from +0.6 (`game-0189`) to
   +22.2 (`game-0208`); five gained under two points: `game-0189`, `game-0169` and
   `game-0178` +1.2, `game-0168` and `game-0213` +1.9.

Beside them:

- At 3,720 updates: the new decks +5.3 ± 0.1.
- The old decks at 7,440 by the two decks of a pairing: both of the seventy-one 50.0% ±
  0.2%, both game Decks 50.3% ± 0.1%, one of each 50.2% ± 0.1%.
- The mirrors deal by deal against each network before (`tools/mirror_pairs.py`):
  - Batch four against batch three: +0.7 ± 0.3 points, twenty-six of the seventy-one
    lower.
  - Against `seventy-one-champions`, network by network: `distilled` −0.1, batch two
    −0.5, batch three −1.7, batch four −0.9, each ± 0.3.
  - `cyber` against `seventy-one-champions`: −6.2, −11.3, −14.5 and now −11.7.
- The lowest mirrors against their champions: `deckout` 37.5%, `arcana` 41.8%,
  `rock-block` 42.2%, `dragunity` 44.1%, `herald` 44.5%.
- The divergence at the old decks' decisions in duels between two old decks, some
  14,400 an update, 2,850 of them the seventy-one's in their duels with each other:
  - From the start: 0.010 over the first 500 updates and 0.018 over the last 500,
    where batch three's went from 0.011 to 0.021.
  - The seventy-one's from `seventy-one-champions`: 0.060 over the first ten updates,
    0.054 by the hundredth, 0.058 from the 250th to the 1,000th and 0.056 over the
    last 1,440.
- The twelve held out, `game-0218` to `game-0229`, trained on nothing: +0.1 ± 0.1
  points from 46.0%.
- The new decks against the thirteen decks' pilots: 40.3% from 34.9%, +5.4 ± 0.2.
- 461 duels of some 3.87 million failed (`tools/failed_duels.py`).
  - 298 filled their arenas and ended there: 243 with `game-0195` in a seat, a deck of
    Cyber Dragon, DNA Surgery and Chimeratech, and 50 with `game-0085`. They come at a
    steady rate, 91, 66, 61 and 80 over the run's four quarters, not as a habit learnt.
  - Cards seen before: Planet Pollutant Virus 54, Beast Rising 24, Fake Feather 17,
    Rivalry of Warlords 16, now with `game-0188` too, Pyro Clock of Destiny 2 and
    Orbital Bombardment 1; one recursive event trigger.
  - New: Gauntlet Warrior 23 (`game-0201`), Dark Diviner 13 (`game-0211`), Mind Control
    8 (`game-0216`; `game-0050` holds it too), and one each of The Lady in Wight
    (`game-0200`), Serial Spell (`game-0212`), Ally of Justice Unlimiter (`game-0217`)
    and Enemy Controller (`gladiator`).

With the seventy-one held to `seventy-one-champions` in their duels with each other, the
mirrors are kept, 0.7 points over batch three's and 0.9 under `seventy-one-champions`.
No arm without that hold ran beside it, so how much of the rise the hold made is not
measured; the divergence from `seventy-one-champions` at those decisions fell little,
from 0.060 to 0.056. The new decks learn and the old decks hold, as in every batch since
`distilled`. Each batch's new decks start higher and gain less: 30.4% and +8.9 points in
batch two, 39.2% and +6.9 in batch three, 43.8% and +6.0 here, and the next twelve stand
at 46.0% before any training, the start knowing more of their cards each time. The decks
trained on now hold 2,869 of the pool's 4,065 cards, 70.6%, where they held 2,675
(65.8%) after batch three; the next fifty lists bring 129 more, to 73.8%. All three
readings are met, so by the design the batches go on the same way.

## Batch five (design and readings fixed on 2026-11-09, before anything was built)

Batch five goes on as batch four, agreed on 2026-11-09, with nothing changed but its
decks and its length.

Its first attempt, from 2026-11-09 14:29 at `946abdbc` with `game-0218` to `game-0267`
new, stopped after nine updates: the engine read an attacker that was gone (step 9 of a
battle, `processor.cpp:2147`), and the process with it. Last Turn, in `game-0233`, can be
activated after an attack is declared; its `Duel.ForceAttack` plays a battle of its own
inside that one, which clears the attacker when it ends, and the battle it was played in
goes on without one. Upstream's engine has the same code. Last Turn is the only card of
the pool that forces an attack and `game-0233` the only list that holds it, so
`game-0233` is left out, as the lists with a card that names a card are, agreed on
2026-11-09; with it go the only two cards it alone brings, Last Turn and Jowgen the
Spiritualist. The engine's fault waits for the experiment's end with the scripts that
failed. The attempt's folder is set aside in `runs/expanded-pool/aborted-2026-11-09` on
the CUDA machine.

- **Start**: batch four's mean at 7,440 updates (step 63,600).
- **Decks**:
  - New, fifty: `game-0218` to `game-0268` but `game-0233`, among them the twelve held
    out until now.
  - Old, 283: the seventy-one and the 212 game Decks of the batches before, `game-0001`
    to `game-0217` but `game-0037`, `game-0065`, `game-0093`, `game-0167` and
    `game-0171`.
  - Held out: `game-0269` to `game-0280`.
- **Run**: as batch four's, over all 333 decks.
  - PPO at 1e-4 with its optimizer fresh, 520 duels an update at the settings of `fast`,
    every duel against itself, a checkpoint every 60, the cards' identity tables at the
    network's rate.
  - The first batch's weights, the old decks' duels at 0.4 of them and what that leaves
    going to the new decks' pairings.
  - At each decision of an old deck's seat in a duel between two old decks, the
    divergence from a frozen network's answers is added to PPO's loss, at the weight PPO
    gives a transaction: `seventy-one-champions`' where both decks are of the
    seventy-one, otherwise the start's.
  - 7,680 updates. A new deck plays 10.36 games an update, some 79,600 over the run, an
    old deck 1.84 and an old mirror 0.16 duels. Of the 83.2 duels an update between two
    old decks, 13.5 are between two of the seventy-one. Some 21 hours on the CUDA
    machine.
- **Measured** as batch four's, at the start by the start and after 3,840 and 7,680
  updates by the mean of the last six kept checkpoints (`eval_suites/pool-batch-five-*.json`):
  the old decks and the mirrors after 7,680 alone. The old decks' pairings grow to
  640,712 games, and the measure to some four and a half hours.

Readings, at 7,680 updates, as batch four's:

1. *The mirrors are kept*: the seventy-one mirrors against their champions, deal by deal
   against `seventy-one-champions`' check at 20,160 updates on its deals, a draw half a
   win (`tools/mirror_pairs.py`): **kept** when the difference is at least −2 points by
   two standard errors, **lost** when at most −2 points by two, otherwise **not
   decided**.
2. *The old decks hold*: head to head against the start, frozen, on every ordered
   pairing of the 283 old decks: **hold** when at least 49% by two standard errors,
   **slip** when at most 49% by two, otherwise **not decided**.
3. *The new decks learn*: the mean over the fifty of each deck's share against the
   start, less at the start, the deals paired: **learns** when at least 5 points by two
   standard errors, **does not** when at most 2 points by two, otherwise **not
   decided**.

Told beside them, without a mark: the new decks at 3,840 updates; the old decks by the
two decks of a pairing; the mirrors' mean against their champions and the lowest, and
deal by deal against batch four's; the twelve held out and the new decks against the
thirteen decks' pilots, at the start and the end; the divergence from each frozen network
along the run; the hours.

If all three are met, the batches go on the same way. If the mirrors or the old decks
are not kept, what changes is designed then.

## The network under PPO, a screen beside the batches (design and readings fixed on 2026-11-09, before anything was built)

Every result under PPO since format 13 is of one network: the board read as the mean of
its cards, each answer scored alone, and the seat's own last 32 answers read by an LSTM
that holds 254,976 of its 738,410 weights. Two other shapes are built, and were read on
copies only: the board as its cards, each answer attending to them
(`--board-representation individual-tokens`, +1.1 ± 0.4 points on a copy of the Blackwing
pilot, [The learner itself](2026-10-08-value-head-and-board.md)), and the network without
its history (`--leave-out history`, a copy of the Blackwing pilot without it equivalent
outside training, +0.5 ± 0.8, and not decided on the decks of training, −0.3 ± 0.9,
[Without the deck lists](2026-10-15-without-deck-lists.md)). Neither has been through
PPO. A review of the project proposed it on 2026-11-09, and it was agreed the same day to
run beside the batches on a rented card, asking nothing of the CUDA machine. The question
is narrow: whether the shape changes what PPO makes of one deck, not what one network
holds of hundreds of decks.

- **The fair start.** A board of tokens cannot begin from weights trained on the pooled
  board, so every arm is made from scratch by the champions' recipe
  (`experiments/commands/champions.sh`) at one code, pins and seed (2012). The three
  shapes own the same parameters and begin as the same network. The control is today's
  shape made again the same way, not a champion there is.
- **The arms**, all of Cyber: `cyber`, the control; `cyber-tokens`; `cyber-no-history`.
  Each is the deck's pilot imitated from scratch for 240 updates of 520 duels against the
  thirteen decks' pilots, then 720 updates of PPO at 1e-4, 260 of an update's duels the
  mirror against itself, with fast's settings; a shape is given at both stages. Cyber's
  mirror has moved most from network to network.
- **Measured** by each run's last three checkpoints (480, 600 and 720 updates of PPO):
  each variant head to head against the control, checkpoint for checkpoint, on Cyber's
  mirror suite (2,048 deals, each from both seats, 12,288 games), a deal's games together;
  each run against `all71` in the mirror, as a champion is read.

What one pair can find: two runs of the recipe on the same pilot were 1.9 ± 0.4 points
apart head to head ([Four decks](2026-10-24-four-decks.md)), so a difference between two
runs is taken to have a standard deviation of 1.9 to 2.4 points beside the deals' 0.4. By
simulation, a mark at 54% by two standard errors of the deals is passed by a true effect
of 4 points 34% to 37% of the time, of 6 points 69% to 73%, of 8 points 91% to 95%, and by
none 1% to 2%. The screen finds a large lever and nothing finer; two points would take
some ten pairs an arm.

Readings, at 720 updates, each arm by what it would decide:

1. *The board as its cards*: `cyber-tokens` against the control: **stronger** when it
   wins at least 54% by two standard errors of the deals, **weaker** when at most 46% by
   two, otherwise **not a large lever for one deck**.
2. *The history*: `cyber-no-history` against the control: the history is **needed** when
   it wins at most 46% by two standard errors of the deals, otherwise **not shown to be
   needed**, which frees nothing: one pair can show that the history cannot go, not that
   it can.

Told beside them, without a mark: each run against `all71` in the mirror, and the
seconds an update and the hours of each.

If the board as its cards is stronger, cards that attend to each other are worth a build
and a new format, which is every seat's too: the web client's and the 3D game's. If it is
not a large lever, the shape of the board is not what limits one deck, and whether one
network holds hundreds of decks as well as it could stays open. The history stays unless
a larger test, ten pairs, shows that it can go.

**Where it runs**: on a pod rented for it, the three runs at once, the card and its price
agreed before the pod is created; some three to four hours. Nothing of it runs on the CUDA
machine.

## The network under PPO: the screen's results (read on 2026-11-10)

Run at `359322b7` (`experiments/commands/network-screen.sh`) on a pod rented for it, an
RTX 4090 of Runpod's Secure Cloud at $0.89 an hour with 36 threads. The three runs went at
once through MPS from 2026-11-09 22:24, were measured on the same card, and were read at
01:43 on 2026-11-10.

- Each run trained in its own shape at both stages, as its checkpoints and metrics carry
  it, all three with the same 7,243,755 parameters.
- Their PPO played the designed mix: some 258 of an update's 520 duels were the mirror
  against itself, and no duel failed.
- A run took 2.7 hours, copy and PPO. An update of PPO took 10.9 seconds for the control
  and 11.1 for each variant, the three sharing the card. The pod ran 3.5 hours, about
  $3.10.

1. *The board as its cards*: **stronger**. `cyber-tokens` wins 58.4% ± 0.4% against the
   control over their last three checkpoints (2,048 deals, each from both seats, 12,288
   games). It is over 54% at each of them: 59.4%, 58.2% and 57.6% at 480, 600 and 720
   updates of PPO, ± 0.7 each.
2. *The history*: **not shown to be needed**. `cyber-no-history` wins 45.5% ± 0.5%: under
   46%, but not by two standard errors. By checkpoint it wins 42.3%, 46.0% and 48.0%, the
   gap closing as PPO goes on.

Told beside, without a mark:

- **Against `all71` in the mirror:**
  - the control 74.7% ± 0.5%, where the two arms of
    [Cyber champion](2026-10-23-cyber-champion.md) stood at 74.4% and 76.4%;
  - `cyber-tokens` 75.5% ± 0.4%, +0.8 ± 0.6 points over the control, the deals paired;
  - `cyber-no-history` 72.5% ± 0.5%, −2.2 ± 0.6.

  Head to head, the board as its cards is eight points over the control. Against the
  older network it is under one point over it.
- **The copies end alike:** the imitation loss over their last 20 updates is 0.155 for the
  control, 0.154 for the board as its cards and 0.163 without the history.

The design took the board as its cards, if stronger, for a build and a new format. The
shape tested needs neither:

- Both shapes own the same four matrices of the attention (`board_query`, `board_key`,
  `board_value`, `board_fusion`); the pooled board does not read them.
- The shape is in a checkpoint's metadata, the format stays 15, and no array of the input
  changes.
- `tools/export_checkpoint.py` already names the four matrices for a forward pass written
  elsewhere.

What a seat needs is that one step in its own forward pass, held to the trainer by
`outputs.json`: each answer attends to the cards in the monster zones and the spell and
trap zones, and adds what it reads through `board_fusion`. Whether the web client's and
the 3D game's seats have the step is not checked here. With `board_fusion` at zero, the
step leaves the network as it was. The history stays. The screen does not tell whether
the lever holds for other decks, or for one network of hundreds of decks, which is the
pool's question.

## The network under PPO, a second session (design and readings fixed on 2026-11-10, before anything was built)

The screen found the board as its cards stronger for Cyber: 58.4% head to head against
the control, under a point over it against `all71`. Three questions follow from it. They
were agreed on 2026-11-10 to run in one session on a rented card before batch five is
read, so that their answers stand beside batch five's when the next batch is designed:

- whether the edge holds for a second deck;
- whether it holds after twice the training, or the control closes it;
- whether a network trained on the pooled board takes the attention from where it
  stands. The pool's network cannot be made again cheaply, so this would be its route.
  The attention is only added to each answer (`action + board_fusion(context)`): with
  `board_fusion` at zero, the switched network answers as the one it came from.

- **The arms**, five runs at once:
  - `blackwing` and `blackwing-tokens`: Blackwing from scratch by the champions' recipe at
    seed 2012, as the screen's Cyber arms were made: 240 updates of copy, then 720 of PPO.
  - `cyber` and `cyber-tokens` go on: the screen's two runs continue from 720 to 1,440
    updates of PPO, everything else as it was.
  - `cyber-switched`: `cyber` at 720 updates of PPO, given the board as its cards from
    there on with `board_fusion` set to zero (`tools/switch_board.py`), then 720 updates of
    PPO as the control's continuation: from the same checkpoint, with its optimizer's
    state and the same rate.
- **Measured** as the screen was, each run by its last three checkpoints (Blackwing's at
  480, 600 and 720 updates of PPO, Cyber's at 1,200, 1,320 and 1,440): head to head against
  its control, checkpoint for checkpoint, on the deck's mirror suite (2,048 deals, each
  from both seats), a deal's games together; each run against `all71` in the mirror.

Readings, with the screen's marks:

1. *A second deck*: `blackwing-tokens` against `blackwing`: **stronger** when it wins at
   least 54% by two standard errors of the deals, **weaker** when at most 46% by two,
   otherwise **not a large lever for Blackwing**.
2. *Twice the training*: `cyber-tokens` against `cyber` at 1,200 to 1,440 updates: the edge
   **holds** when it is stronger by the same mark; otherwise it is **not shown to hold**,
   its share told beside the screen's 58.4%.
3. *The switch*: `cyber-switched` against `cyber` at the same checkpoints: the switch
   **pays** when it wins at least 54% by two standard errors, **harms** when at most 46%
   by two, otherwise it is **not shown**. A gain smaller than the screen's eight points
   would often be missed: a true four points passes about a third of the time.

Told beside them, without a mark: each run against `all71`, the seconds an update and the
hours.

**Checks with no training**, on the development machine while the session is prepared,
read by the same paired deals on Cyber's mirror suite:

- the two copies' last checkpoints (240 updates) head to head: whether the edge was there
  before PPO, told without a mark;
- `cyber-tokens` and `cyber` at 720 updates of PPO, each against `cyber-old` at 720, the
  arm of [Cyber champion](2026-10-23-cyber-champion.md) made by the same recipe at the
  same seed: the edge **carries to a third network** when `cyber-tokens` wins more against
  it than the control does, by two standard errors of the deals paired; otherwise it is
  **not shown to carry**, and the edge head to head may be one network's over its sibling.

What follows from the readings is decided beside batch five's results.

**Where it runs**: a pod rented for it, an RTX 4090 or an RTX 3090 of Runpod's Secure
Cloud, whichever is available (agreed on 2026-11-10), the five runs at once; some five
hours. The checks run on the development machine. Nothing of it runs on the CUDA machine.

## The second session: the checks with no training (read on 2026-11-10)

Played on the development machine's processor with the code of `a13bf9fd`
(`experiments/commands/network-screen-checks.sh`), read by
`tools/network_screen_checks.py`, each on Cyber's mirror suite (2,048 deals, each from
both seats).

- *The edge against a third network*: it **carries to a third network**. Against
  `cyber-old` at 720 updates, `cyber-tokens` wins 54.8% ± 0.7% and the control 46.2% ±
  0.7%: +8.5 ± 1.0 points, the deals paired. The edge head to head is found again against
  a third network of the same recipe, so it is strength in the mirror, not one network's
  over its sibling.
- Told without a mark, the copies head to head: the board as its cards wins 51.6% ± 0.6%
  against the control's copy. Most of the edge came under PPO.
- Beside them: the control itself wins 46.2% against `cyber-old`, two runs of one recipe
  3.8 points apart, where two runs on the same pilot were 1.9 ± 0.4 apart.

## Batch five's results (read on 2026-11-10)

Batch five ran on the CUDA machine from 2026-11-09 14:49 at `a6166be6`, 20.3 hours, 9.5
seconds an update, and its measure 3.9 hours more (`analysis/expanded-pool/batch-five/report.txt`).
Its first updates gave a new deck 10.31 games and an old one 1.85, as designed, all 333
decks seated and `game-0233` in none.

1. *The mirrors are kept*: **kept**, −1.1 ± 0.4 points deal by deal against
   `seventy-one-champions`' check, forty-five of the seventy-one lower. Against their
   champions they stand at 49.6% ± 0.3%. The most lost are `cyber` −8.6, `spirit` −7.4,
   `chaos` −7.0, `nordic` and `volcanic` −6.6; the most won `gusto` +8.6, `gishki` and
   `naturia` +5.5.
2. *The old decks hold*: **hold**, 50.1% ± 0.1% head to head against the start, frozen,
   on every ordered pairing of the 283 old decks (640,712 games).
3. *The new decks learn*: **not decided**, +4.6 ± 0.1 points against the start after
   7,680 updates, from 45.4%: under 5 points by two standard errors, over 2 by two.
   Every one of the fifty gained, from +0.8 (`game-0231`) to +16.8 (`game-0225`); nine
   gained under two points, the least `game-0231` +0.8, `game-0232` +1.0 and `game-0240`
   +1.1.

Beside them:

- At 3,840 updates, half the games: the new decks +4.0 ± 0.1, seven eighths of the end's
  gain.
- The old decks at 7,680 by the two decks of a pairing: both of the seventy-one 50.1% ±
  0.2%, both game Decks 50.2% ± 0.1%, one of each 50.1% ± 0.1%.
- The mirrors deal by deal against each network before (`tools/mirror_pairs.py`):
  - Batch five against batch four: −0.1 ± 0.3 points, thirty-six of the seventy-one
    lower.
  - Against `seventy-one-champions`, network by network: `distilled` −0.1, batch two
    −0.5, batch three −1.7, batch four −0.9, batch five −1.1, each ± 0.3 or 0.4.
  - `cyber` against `seventy-one-champions`: −6.2, −11.3, −14.5, −11.7 and now −8.6.
- The lowest mirrors against their champions: `deckout` 34.4%, `pyramid` 43.0%,
  `dragunity` and `infernity` 43.4%, `macro-dd` 44.5%.
- The divergence at the old decks' decisions in duels between two old decks, some
  14,500 an update, 2,300 of them the seventy-one's in their duels with each other:
  - From the start: 0.009 over the first 500 updates and 0.017 over the last 500, where
    batch four's went from 0.010 to 0.018.
  - The seventy-one's from `seventy-one-champions`: 0.052 over the first ten updates,
    0.056 over the first hundred, 0.059 from the 250th to the 1,000th and 0.060 over the
    last 1,440.
- The twelve held out, `game-0269` to `game-0280`, trained on nothing: +0.4 ± 0.1 points
  from 54.4%.
- The new decks against the thirteen decks' pilots: 41.4% from 37.3%, +4.1 ± 0.2.
- 296 duels of some 3.99 million failed (`tools/failed_duels.py`).
  - 88 filled their arenas and ended there: 47 with `game-0195` in a seat, 27 with
    `game-0085` and 13 with `game-0262`, new in this batch; 21, 21, 24 and 22 over the
    run's four quarters.
  - Cards seen before: Fake Feather 106, at its two lines, now with `game-0261` too;
    Planet Pollutant Virus 49, Beast Rising 25, Neo Flamvell Shaman 6, now with
    `game-0218` too, Gauntlet Warrior 4, Pyro Clock of Destiny and Lord British Space
    Fighter 2 each, and one each of Mind Control, Serial Spell, Rivalry of Warlords and
    Orbital Bombardment; one recursive event trigger.
  - New: Human-Wave Tactics 7 (`game-0073`), and one each of Nettles (`game-0230`) and
    Shock Troops of the Ice Barrier (`game-0221`).

The mirrors are kept and the old decks hold, as in batch four. The new decks gain less
again: 30.4% and +8.9 points in batch two, 39.2% and +6.9 in batch three, 43.8% and +6.0
in batch four, 45.4% and +4.6 here, ending at 39.3%, 46.1%, 49.8% and 50.0% against the
start. The next twelve stand at 54.4% before any training. The decks trained on now hold
2,997 of the pool's 4,065 cards, 73.7%. Of the 484 game lists usable as they stand, 214
are left after batch five's, the twelve held out among them; they bring some 260 cards
more, to about 80%: the next fifty some 96, the next hundred some 146. The third reading
is not decided, so by the design the batches do not go on the same way by themselves:
batch six is designed with the second session's readings beside these.

## The second session's results (read on 2026-11-10)

Run at `a13bf9fd` (`experiments/commands/network-screen-two.sh`) on a pod rented for it,
an RTX 4090 of Runpod's Secure Cloud at $0.89 an hour with 32 threads. The five runs went
at once through MPS from 2026-11-10 12:00 to 15:41, and the two decks were measured at
once on the same card after them.

- Each run trained in its own shape, as its checkpoints and metrics carry it;
  `cyber-switched` in the board as its cards from its first update.
- No duel failed.
- An update of PPO took 15.1 seconds for `blackwing` and 13.1 for `blackwing-tokens`,
  17.6, 17.9 and 18.1 for `cyber`, `cyber-tokens` and `cyber-switched`, the five
  sharing the card; an update of Blackwing's copies 9.7 and 10.6. The pod ran 4.6 hours,
  about $4.10.

1. *A second deck*: **not a large lever for Blackwing**. `blackwing-tokens` wins 53.7% ±
   0.4% against `blackwing` over their last three checkpoints (2,048 deals, each from both
   seats): 53.1%, 55.5% and 52.6% at 480, 600 and 720 updates of PPO, ± 0.6 or 0.7 each.
2. *Twice the training*: **not shown to hold**. `cyber-tokens` wins 52.5% ± 0.5% against
   `cyber` at 1,200 to 1,440 updates: 52.5%, 51.8% and 53.1%, ± 0.7 each, where the
   screen found 58.4% at 480 to 720.
3. *The switch*: **not shown**. `cyber-switched` wins 50.0% ± 0.4% against `cyber` at the
   same checkpoints: 49.7%, 49.8% and 50.5%, ± 0.7 each. By two standard errors it is
   within 0.8 points of the control.

Told beside, each against `all71` in the mirror, the deals paired against the control:

- `blackwing` 64.9% ± 0.5%; `blackwing-tokens` 66.4% ± 0.5%, +1.5 ± 0.5 points.
- `cyber` 78.5% ± 0.4%, from 74.7% at 480 to 720; `cyber-tokens` 78.8% ± 0.4%, +0.3 ±
  0.5, from 75.5%; `cyber-switched` 78.2% ± 0.4%, −0.3 ± 0.5.

The board as its cards, in a run made in that shape from the start, stays ahead of the
pooled board for both decks, by less than the screen's mark: Blackwing's by 3.7 points
head to head, Cyber's by 2.5 after twice the training, where the screen's checkpoints
had 8.4. Both are over 50% by more than five standard errors. Against `all71` it adds
1.5 points for Blackwing and 0.3 for Cyber. Cyber's control closed most of the gap as PPO
went on, both runs rising some four points against `all71`. A network trained on the
pooled board and given the attention at 720 updates gains nothing measured in the 720
that follow: within 0.8 points of its control by two standard errors, where the run made
in that shape from the start keeps 2.5. The pool's network could take the attention by
that route alone, or be made again from the start.

## Beside the batches: two measurements of a review of the learning (recorded on 2026-11-10)

A review of what could improve the learning, made on 2026-11-09 beside batch five,
measured two things on the development machine, each in a worktree of its own whose code
is not committed. Neither trains anything.

- **What a seat has seen and is not shown.** Over all71's four held updates (264,111
  decisions), a seat is not shown a card it has seen at 23.2% ± 0.7% of its decisions:
  - at 18.0% a card in the other player's hand, 0.25 of its 2.43 unseen hand cards at
    a decision;
  - at 7.0% a card face-down on the field;
  - at none in the first turn and at 27.9% from the eleventh.

  The count is an upper bound: every card seen counts, whether or not a person could
  still tell which it is. What showing them would be worth is not measured. The count
  needs a mark kept on each card through the policies' projection, a patch of that
  repository's which is not taken.
- **An attending reader for the critic that sees the hidden cards**
  ([Privileged critic](2026-10-22-privileged-critic.md)): two readings, fixed before
  the fit, of a critic whose cards attend to each other, hands and face-down cards
  among them, against the policy's head. Fitted warm on the same games, 200 updates
  from all71's last checkpoint, read on 2,080 held duels. Neither is met.
  - Over every held decision the head explains 0.404 of a result, the attending
    critic 0.002 ± 0.005 more (the mark: 0.10 more by two standard errors), and the
    critic of that record fitted again 0.012 ± 0.006 more. At 400 updates the same:
    0.004 ± 0.009 and 0.011 ± 0.009.
  - Ranking two answers played on from one position (2,000 decisions, four pairs
    each), the head follows their worth at 0.163 ± 0.035 and the attending critic at
    0.189 ± 0.036: 0.026 ± 0.013 more, where the mark was twice the head's.

  Nothing follows for the CUDA machine.

## Batch six, in two arms (design and readings fixed on 2026-11-10, before anything was built)

Batch six goes on as batch five with three changes, agreed on 2026-11-10:

- a hundred new decks instead of fifty, at the same length;
- the third reading's mark at 3 points: the new decks now start near 45% and end near
  50% against the start;
- two arms, one of them given the board as its cards at its start. The second session
  did not show the switch for one deck; the arms ask it of the pool's network, so that
  whether the switch is kept is decided on their results.

- **Start**: batch five's mean at 7,680 updates (step 71,280).
  - `pooled`: as it is.
  - `switched`: given the board as its cards with `board_fusion` at zero
    (`tools/switch_board.py`), so that it answers as `pooled`'s start does.
- **Decks**:
  - New, a hundred: `game-0269` to `game-0369` but `game-0335`, the twelve held out
    until now among them; of the lists usable as they stand, those that bring the most
    new cards.
  - Old, 333: the seventy-one and the 262 game Decks of the batches before, `game-0001`
    to `game-0268` but `game-0037`, `game-0065`, `game-0093`, `game-0167`, `game-0171`
    and `game-0233`.
  - Held out: `game-0370` to `game-0381`.
- **Run**: as batch five's, over all 433 decks.
  - The two arms are the same but for the shape of their start: the same seed and
    weights, and the same frozen networks held to, batch five's mean and
    `seventy-one-champions`, each answering in its own shape.
  - 7,680 updates. A new deck plays 5.65 games an update, some 43,400 over the run,
    half of batch five's; an old deck plays 1.43 and an old mirror 0.12 duels. Of the
    79.0 duels an update between two old decks, 10.3 are between two of the
    seventy-one.
  - Some 20 hours an arm on the CUDA machine, `pooled` first.
- **Measured** as batch five's, each arm into its own folder, but the new decks' pairings
  are played at 16 games each instead of 32: twice the decks, the same games.
  - `switched` takes `pooled`'s measures at 0, its start answering the same, except the
    pilots' suite, which it plays itself. Those games must be the same, game by game, or
    the switch is not what it was taken for.
  - Some 5 hours for `pooled`'s measure and 4 for `switched`'s.
- **Head to head**: after both, the two arms' means at 7,680 updates.
  - Each of the 433 decks plays 32 others drawn at random (seed 2012), two games each,
    one from each seat.
  - Each game is played once with each arm in the learner's seat: 27,712 deals, a
    deal's two games together.
  - The same at 3,840 updates, told beside.

Readings, at 7,680 updates. The first three are each arm's own, as batch five's but the
third's mark:

1. *The mirrors are kept*: as batch five's.
2. *The old decks hold*: as batch five's, on every ordered pairing of the 333 old decks.
3. *The new decks learn*: the mean over the hundred of each deck's share against the
   start, less at the start, the deals paired: **learns** when at least 3 points by two
   standard errors, **does not** when at most 1 point by two, otherwise **not decided**.
4. *The switch*: `switched` head to head against `pooled`: **pays** when it wins at least
   51% of the deals by two standard errors, **harms** when at most 49% by two, otherwise
   **not shown**.

Told beside them, without a mark:

- for each arm, what batch five's told;
- the two arms' differences, deal by deal, on the measures they share: the new decks
  against the start, the old decks, the twelve held out, the pilots, and the mirrors
  against their champions;
- the head to head at 3,840 updates;
- the hours.

Which arm the pool goes on from is decided on these results, as agreed on 2026-11-10.

On 2026-11-11 `game-0379`, one of the twelve held out, was left out of the held out:
`pooled`'s held-out measure at 0 failed on it.
- Its three Flame Rulers each count as two tributes for a FIRE monster, and it plays
  Level 10 FIRE monsters.
- The policies' decision (`ygo-policies-ocgcore`) builds a tribute selection as a
  selection of cards, one tribute a card. Where the one card is the only way to pay both
  tributes, the selection offers no answer and the round fails.
- Some two dozen trained decks hold such a card, `fish` and `arcana` among the
  seventy-one, and none has failed a round in the batches. Where such a card is offered
  beside other monsters, the seat cannot give it alone: it gives as many cards as
  tributes, one more than the engine takes, the pilots as the network. Every comparison of
  the experiment shares it.
- The fault waits for the experiment's end with the scripts that failed, as agreed on
  2026-11-11, and `game-0379` stays out of the batches until then. Its fix changes those
  decks' questions, so it moves the policies' pin and may cost the digest of a champion
  recording that holds such a selection.

The held out are eleven, the same for both arms. A held-out measure made on other decks
is set aside and played again (`experiments/commands/pool-batch-six.sh`). `pooled`'s
first task fails at the end of its measure, so a task of its own finishes the measure
before `switched` runs.

On 2026-11-11 at 23:15 `switched` stopped at its 590th update on such a selection, in a
duel the failed round did not name. From `ba221814` a selection the seat's rules leave no
answer ends its duel there, as a card's failed script does:
- the duel is kept with its record among the run's failed duels;
- in training it counts as a duel cut off, a loss to the seats that train;
- in a measure it counts as a draw cut off.

`pooled` met none in its run or its measures, so it plays the same either way.
`switched` goes on from its last checkpoint at that commit, and the head to head after
it.

## Batch six: `pooled`'s results (read on 2026-11-11)

`pooled` ran on the CUDA machine from 2026-11-10 16:46 at `2138255d`, 22.6 hours, 10.6
seconds an update. Its measure took some six hours more, the held out played again by a
task of its own at `0368d3f4` (`analysis/expanded-pool/batch-six-pooled/report.txt`). Its
first updates were as designed: a new deck 5.64 games an update and an old one 1.43, all
433 decks seated and `game-0335` in none, some 12,700 decisions an update distilled and
1,500 of them held to `seventy-one-champions`.

1. *The mirrors are kept*: **not decided**, −1.3 ± 0.4 points deal by deal against
   `seventy-one-champions`' check, forty-six of the seventy-one lower: above −2 points,
   but not by two standard errors. Against their champions they stand at 49.4% ± 0.3%.
   The most lost are `cyber` −8.6, `volcanic` −7.8, `x-saber` −7.0, `blackwing` and
   `scrap` −6.6; the most won `dark-world`, `gishki` and `gusto` +5.1.
2. *The old decks hold*: **hold**, 50.2% ± 0.1% head to head against the start, frozen,
   on every ordered pairing of the 333 old decks (887,112 games).
3. *The new decks learn*: **learns**, +3.7 ± 0.1 points against the start after 7,680
   updates, from 51.8%. None of the hundred lost ground and twenty-nine gained under two
   points, the least `game-0323` +0.0, `game-0352` +0.1 and `game-0356` +0.3, the most
   `game-0306` +9.6, `game-0337` +12.3 and `game-0365` +14.0.

Beside them:

- At 3,840 updates, half the games: the new decks +3.1 ± 0.1, five sixths of the end's
  gain.
- The old decks at 7,680 by the two decks of a pairing: both of the seventy-one 50.3% ±
  0.2%, both game Decks 50.3% ± 0.1%, one of each 50.1% ± 0.1%.
- The mirrors deal by deal against batch five's: −0.2 ± 0.3 points, forty of the
  seventy-one lower. Against `seventy-one-champions`, batch by batch from batch four:
  −0.9, −1.1 and −1.3, each ± 0.3 or 0.4; `cyber` −8.6, as in batch five.
- The lowest mirrors against their champions: `deckout` 38.7%, `blackwing` 41.4%,
  `herald` 42.2%, `macro-dd` 43.4%, `gladiator` 44.1%.
- The divergence at the old decks' decisions in duels between two old decks, some
  13,500 an update, 1,780 of them the seventy-one's in their duels with each other:
  - From the start: 0.010 over the first 500 updates and 0.019 over the last 500, where
    batch five's went from 0.009 to 0.017.
  - The seventy-one's from `seventy-one-champions`: 0.061 over the first ten updates and
    over the first hundred, 0.064 from the 250th to the 1,000th and 0.067 over the last
    1,440, where batch five's went from 0.052 to 0.060.
- The eleven held out, trained on nothing: +0.4 ± 0.1 points from 50.6%.
- The new decks against the thirteen decks' pilots: 47.5% from 44.0%, +3.5 ± 0.1.
- 490 duels of some 3.99 million failed (`tools/failed_duels.py`).
  - 151 filled their arenas and ended there: 53 with `game-0195` in a seat, 44 with
    `game-0085`, 24 with `game-0289`, 18 with `game-0304` and 13 with `game-0340`, new in
    this batch, and 12 with `game-0262`; 33, 35, 38 and 45 over the run's four quarters.
  - Planet Pollutant Virus 239, now with `game-0321` too, against 49 in batch five.
  - Cards seen before: Fake Feather 39, at its two lines, Beast Rising 21, Neo Flamvell
    Shaman 10, Pyro Clock of Destiny, Mind Control and Rivalry of Warlords 4 each,
    Orbital Bombardment 2 and Gauntlet Warrior 1.
  - New: Infestation Wave 5 (`game-0293`), Ultimate Obedient Fiend 3 (`game-0273`),
    Compulsory Evacuation Device 2 (`six-samurai`), and one each of Mei-Kou, Master of
    Barriers (`spellcaster`), Lightforce Sword (`game-0020`), Gozen Match (`game-0319`),
    The Six Samurai - Yariza (`game-0070`) and Exiled Force (`game-0354`).

The 433 decks of the batch, the same in both arms, hold 3,149 of the pool's 4,065
cards, 77.5%, from 2,997 after batch five, counted as then: the names in their main and
extra decks. The 113 usable lists not yet taken would bring 120 more, to 3,269 (80.4%),
and the nine kept out until the fixes held for the experiment's end 19 that no other
list holds, to 3,288 (80.9%).

The old decks hold and the new decks learn by the batch's mark. The mirrors are not
decided. They stand where batch five left them (−0.2 ± 0.3), but against
`seventy-one-champions` the batches since batch four read −0.9, −1.1 and −1.3, and the
seventy-one's divergence from it rose again. The new decks start higher and gain less:
45.4% and +4.6 points in batch five, 51.8% and +3.7 here, ending at 55.5% against the
start. `switched`'s readings and the head to head were read on 2026-11-13 (below).

## The board as its cards from scratch, at fourteen decks (design and readings fixed on 2026-11-12, before anything was built)

Made from scratch, the board as its cards was ahead for single decks: 53.7% head to head
for Blackwing and 52.5% for Cyber after twice the training, each about a point over its
control against `all71`. Switched into a trained network it was not shown (50.0%), and
batch six's `switched` arm asks the same of the pool's network. What remains open is a
network of many decks made from scratch with the board as its cards. Made for the
seventy-one by `seventy-one-champions`' recipe, that takes some five days of the CUDA
machine. It was agreed on 2026-11-12 to ask it first of the fourteen decks of
[A network of fourteen decks](2026-10-26-fourteen-decks.md), in a day and a half, and to
make the seventy-one only where the board as its cards is stronger there.

- **The arms**, `pooled` and `tokens`: two networks of the fourteen decks from scratch,
  seed 2012, the same in everything but the board, which `tokens` takes as its cards at
  both stages (`--board-representation individual-tokens`). The two are trained at once,
  as that record's `quarter` and `half` were.
- **The copy**: each deck's games are its champion's recorded games, played again with its
  answers and imitated. The recordings are those of the seventy-one, whose teachers for
  these fourteen decks are the fourteen decks' network's own. 2,240 updates of 520 duels,
  160 a deck: 260 in the mirrors, both seats imitated, and 260 against the thirteen decks'
  pilots. At 3e-4, in minibatches of 256 with no duels in flight, every game seen once and
  its result at four of its positions; a checkpoint every 80 updates. Each deck's copy
  takes 41,600 of its 44,000 recorded mirror games and 3,200 of the 4,000 against each
  pilot, as `seventy-one-champions`' copy did.
- **PPO**: from each copy's last checkpoint, as that record's `quarter` arm: 4,320
  updates of 520 duels at 1e-4 with `fast`'s settings, every duel against itself and none
  against a pilot, a quarter of them in the mirrors (each mirror weighed 13, each other
  ordered pairing 3). A checkpoint every 240 updates of the trainer's count, which goes on
  from the copy's 2,240. Each deck has some 320,900 games, as many as
  `seventy-one-champions`' PPO gave each of its decks.
- **Measured** on the 196 ordered pairings of the fourteen decks, 64 games each
  (`eval_suites/fourteen-decks-pairings-quick-checkpoint.json`, seed 2012):
  - `tokens` head to head against `pooled` at their last three checkpoints (after 4,000,
    4,240 and 4,320 updates of PPO), checkpoint for checkpoint. Each game is played once
    with each arm in the learner's seat, and a deal's games at the three checkpoints are
    read together, as the screens read theirs: 12,544 deals of six games.
  - The same for the two copies' last checkpoints and after 1,360 and 2,800 updates of
    PPO.
  - Each arm's last three checkpoints against `all71`, on the same deals as that record's
    `quarter` arm.

Reading, with the marks of batch six's head to head, so that the switch and the network
made from scratch read alike:

1. *The board as its cards from scratch*: `tokens` against `pooled` at the last three
   checkpoints is **stronger** when it wins at least 51% of the deals by two standard
   errors and **weaker** when at most 49% by two; otherwise it is **not decided**.

What follows was agreed on 2026-11-12. The seventy-one are made from scratch with the
board as its cards only when the reading is **stronger**, after the fixes held for the
experiment's end and against `seventy-one-champions`. Otherwise the route from scratch is
dropped.

Told beside it, without a mark:

- the head to head of the copies and after 1,360 and 2,800 updates of PPO: whether an edge
  comes from the copy or from PPO, and whether it grows or closes as Cyber's did;
- each arm against `all71`, and the two deal by deal;
- `pooled` against that record's `quarter` arm against `all71` on the same deals: whether
  a copy from recordings, at today's pins, lands where the copy that asked its champions
  did;
- the copies' imitation losses, the seconds an update and the hours.

**Where and when**: on the CUDA machine's queue at high priority, after batch six's head
to head. By reckoning some 10 hours for the two copies, 14 for PPO and 2 to measure. That
record's copy, which asked its champions, took 13.4 hours; its two arms of PPO took 12.6
and 12.8 hours at once; `seventy-one-champions`' copy from recordings took 11.2 seconds an
update. A copy from recordings stops at the first game that plays again with another
digest, so the test runs before the pins move. The fixes held for the experiment's end
reach the fourteen decks only through four scripts that failed a few times in all the
batches (Compulsory Evacuation Device, Enemy Controller, Fire Princess and Mei-Kou,
Master of Barriers), so its answer stands after them.

## The copy's width and rate, at seventy-one decks (design and readings fixed on 2026-11-12, before anything was built)

`seventy-one-champions`' copy fitted each of the fourteen decks less well than the
fourteen decks' copy did, on the same teachers at the same updates a deck: 46.4% head to
head against them where the other had 48.4%. Its record left open what limits it, the
network's size or a rate held at 3e-4 through five times the steps. Counted over the
fourteen decks' decisions, each deck alike as that record counts, the two copies' share of
the teachers' answers is level to ten updates a deck, the seventy-one's copy ahead at
first. It falls behind after that: 94.70% against 95.20% at 10 to 20 updates a deck,
95.38% against 96.14% at 20 to 40, and the gap goes on widening. The seventy-one's copy
stands at 94.80% over all its decks at 10 to 20.

The network has never been a variable but for its board, and its width is not yet an
option. It becomes one with this test (`train --embedding-dim --hidden-dim`): a
checkpoint of other widths says so, and whatever plays one builds the network it says. A
checkpoint of today's widths is described as before.

- **The arms**, four at once on a rented card. Each copies the seventy-one from their
  recordings from scratch as `seventy-one-champions`' copy did: seed 2012, the same
  recorded games in the same order, 520 duels an update, 260 in the mirrors with both
  seats imitated and 260 against the thirteen decks' pilots, minibatches of 256 with no
  duels in flight, every game seen once and its result at four of its positions; a
  checkpoint every 142 updates.
  - `control`: as that copy, at 3e-4. It also shows whether the card plays that copy's
    updates again.
  - `wide`: twice as wide: a card's identity 192 wide and a decision 384
    (today 96 and 192).
  - `narrow`: half as wide: 48 and 96.
  - `slow`: `control` from its checkpoint after ten updates a deck (710 updates), on at
    1e-4, a third of the rate, from the same state and on the same games after it.
- **Length**: twenty updates a deck, 1,420 updates; `slow` the last 710 of them.
- **Measured** from each run's own metrics, with no game played for it: the share of the
  teachers' answers the network picks at the decisions it imitates, each deck alike, over
  ten to twenty updates a deck (updates 711 to 1,420). Each arm is read against `control`
  update by update, the two imitating the same games, by the mean difference and two
  standard errors of its means over the window's ten stretches of 71 updates.

Readings, the same for each of `wide`, `narrow` and `slow` against `control`. The mark is
half of the gap between the two copies over that window:

1. it **fits better** when it picks the teachers' answers at least 0.25 points more often
   by two standard errors, and **fits worse** when at least 0.25 points less by two;
   otherwise it is **not a lever at this length**.

Read together: `wide` fitting better says the width holds the copy back at seventy-one
decks, and more so for the pool's hundreds. `narrow` not fitting worse says today's width
has room to spare. `slow` fitting better says a rate lowered as the copy goes fits each
deck better. What follows is decided on the results, beside those of the fourteen decks
made from scratch.

Agreed on 2026-11-12 while the copies ran, before any reading: where any of the three is
not a lever at this length, the four copies go on to forty updates a deck (2,840
updates) on the same card, each from where it stands, and are read again by the same rule
over twenty to forty updates a deck (updates 1,421 to 2,840). The mark is then half the
gap between the two copies there, 0.38 points (95.38% against 96.14%), and the readings at
twenty updates a deck are told beside. Otherwise the card is stopped once what it made is
here.

Told beside them, without a mark:

- on the fourteen decks alone, each arm and `control` beside the fourteen decks' copy over
  the same window: how much of the gap a change closes;
- `control` against `seventy-one-champions`' copy update by update, whether the rented
  card plays its updates again;
- where an arm fits better or worse, its last checkpoint head to head against the teachers
  in the fourteen decks' mirrors, on the CUDA machine afterwards;
- the seconds an update and the hours.

**Where**: an RTX 4090 of Runpod's Secure Cloud, $0.89 an hour on 2026-11-12, with 32
threads or more, the four at once. By reckoning some 8 to 10 hours: on the CUDA machine
an update of that copy took 12.5 seconds, 6.9 of them training on its card. The copies
use the seventy-one's recordings, which play only at today's pins, so it runs before the
fixes move them. Nothing of it runs on the CUDA machine but the head to head beside.

## The copy's width and rate: results (read on 2026-11-12)

The four copies ran at once on a rented RTX 4090 with 16 threads (none with 32 was free),
18.4 to 18.8 seconds an update each, some 7.3 hours a copy and 7.6 hours of the card
(`experiments/commands/copy-width-rate.sh`). Over ten to twenty updates a deck (updates
711 to 1,420), each deck alike, update by update against `control` on the same games,
two standard errors from the means of ten stretches of 71 updates:

1. `wide` **fits better**: 95.43% of the teachers' answers against `control`'s 94.82%,
   +0.61 ± 0.01 points.
2. `narrow` **fits worse**: 93.67%, −1.15 ± 0.01 points.
3. `slow` **fits better**: 95.27%, +0.45 ± 0.02 points.

Each was decided at twenty updates a deck, so the copies did not go on to forty, and the
card was stopped once what it made was here: the metrics and the report in
`analysis/copy-width-rate/`, each copy's last checkpoint in `runs/copy-width-rate/`.

Beside them, without a mark:

- On the fourteen decks alone, over the same window: `control` 94.72%, `wide` 95.30%,
  `narrow` 93.58%, `slow` 95.22%, and the fourteen decks' copy 95.20% at the same updates
  a deck. Twice the width closes the gap between the two copies and goes 0.10 points past
  it; a third of the rate from halfway closes it.
- `control` less `seventy-one-champions`' copy, update by update over all 1,420 updates:
  +0.02 ± 0.00 points. The rented card plays that copy's updates again but for that, not
  bit for bit.
- Twice the width cost the card no time with four copies on it: 18.8 seconds an update
  against `control`'s 18.5. Its checkpoint is 182 MB against 85 MB.
- Each copy's last checkpoint (update 1,420) head to head against the teachers in the
  fourteen decks' mirrors, each deck's suite of 2,048 deals at seed 2012 from both
  seats, 229,376 games with none failed. Played on the development machine's processor,
  whose suites play the CUDA machine's games (measured on 2026-10-10) and where the
  teachers are too, instead of waiting for the CUDA machine
  (`experiments/commands/copy-width-rate-heads.sh`). `control` 42.6% ± 0.2%, `wide`
  44.9%, `narrow` 39.3%, `slow` 43.2%. Less `control`, deal by deal:
  - `wide` +2.32 ± 0.19 points, ahead on thirteen of the fourteen decks, from −0.3
    (Heroes) to +4.2 (Gladiator);
  - `narrow` −3.37 ± 0.20 points, behind on all fourteen, from −0.3 (Crystal) to −7.3
    (Heroes);
  - `slow` +0.55 ± 0.17 points, from −1.0 (Monarch) to +1.9 (Dragunity).

  At twenty updates a deck, an eighth of a copy: `seventy-one-champions`' copy had 46.4%
  against the same teachers after 160.

As the design reads them: the width holds the copy back at seventy-one decks, and by
that reading more so at the pool's hundreds; today's width has no room to spare; and a
rate lowered as the copy goes fits each deck better. All three are the share of the
teachers' answers in the games imitated, and the head to head goes the same way: the
closer fit plays the teachers better, by 2.3 points twice as wide and 0.6 at the lower
rate, and half as wide plays them 3.4 points worse. Whether the gains last to a copy's
end, and whether the two add up (`wide` was not tried at another rate), is not measured.
What follows is decided on these results beside those of the fourteen decks made from
scratch.

## Batch six: `switched`'s results and the head to head (read on 2026-11-13)

`switched` ran on the CUDA machine from 2026-11-11 21:25 at `0368d3f4` and, from its
590th update, from 23:39 at `d323c925`, which holds `ba221814` (above). Its 7,680 updates
took 24.0 hours, 11.2 seconds an update against `pooled`'s 10.6: the board as its cards
costs the run 6%. Its measure ended on 2026-11-13 at 02:16
(`analysis/expanded-pool/batch-six-switched/report.txt`) and the head to head at 02:56
(`experiments/commands/pool-batch-six-arms.sh`,
`analysis/expanded-pool/batch-six-arms/report.txt`). Over the whole run its mix was
`pooled`'s: all 433 decks seated and `game-0335` in none, a new deck and an old one as
many games an update, and some 13,500 decisions an update distilled, 1,780 of them held
to `seventy-one-champions`. Its first updates were as designed: by the 30th the board's
fusion had grown from zero to 0.94 in norm.

1. *The mirrors are kept*: **not decided**, −1.3 ± 0.4 points deal by deal against
   `seventy-one-champions`' check, forty-four of the seventy-one lower, as `pooled`'s.
   Against their champions they stand at 49.3% ± 0.3%. The most lost are `cyber` −10.9,
   `garden` and `volcanic` −7.8, `karakuri` and `nordic` −7.0; the most won `gishki`
   +6.2, `gusto` and `naturia` +5.1.
2. *The old decks hold*: **hold**, 50.2% ± 0.1% head to head against the start, frozen.
3. *The new decks learn*: **learns**, +3.5 ± 0.1 points against the start after 7,680
   updates, from 51.8%. The least `game-0352` +0.1, `game-0356` and `game-0323` +0.4, the
   most `game-0292` +9.3, `game-0337` +12.2 and `game-0365` +13.9.
4. *The switch*: **not shown**. `switched` wins 50.0% ± 0.1% of the 27,712 deals head to
   head against `pooled` at 7,680 updates (50.05% ± 0.12%), within three tenths of a point
   of even by two standard errors.

Beside them:

- The head to head at 3,840 updates: 49.9% ± 0.1%.
- `switched` less `pooled`, deal by deal at 7,680 updates:
  - the new decks against the start −0.18 ± 0.05 points: `switched` learns them a fifth
    of a point less;
  - the old decks −0.02 ± 0.03, the eleven held out −0.01 ± 0.11;
  - the new decks against the pilots −0.07 ± 0.12;
  - the mirrors against their champions −0.05 ± 0.30, thirty-four of the seventy-one
    lower.
- `switched`'s start played the pilots' 83,200 games at 0 as `pooled`'s did, every one:
  given the board as its cards with `board_fusion` at zero, it answered as the network it
  came from.
- At 3,840 updates the new decks +3.1 ± 0.1, as `pooled`'s.
- The mirrors deal by deal against batch five's: −0.3 ± 0.3 points, forty-three of the
  seventy-one lower. The lowest against their champions: `deckout` 35.5%, `malefic`
  42.6%, `rock-block` 43.0%, `pyramid` 43.4%, `blackwing` 44.1%.
- The divergence at the old decks' decisions, read as `pooled`'s:
  - from the start, 0.0105 over the first 500 updates and 0.0194 over the last 500,
    against `pooled`'s 0.0101 and 0.0189;
  - the seventy-one's from `seventy-one-champions`, 0.061, 0.062, 0.065 and 0.068 over
    the same four stretches as `pooled`'s 0.061, 0.061, 0.064 and 0.067.
- The eleven held out, trained on nothing: +0.3 ± 0.1 points from 50.6%.
- The new decks against the thirteen decks' pilots: 47.5% from 44.0%, +3.4 ± 0.1.
- 468 duels of some 3.99 million failed (`tools/failed_duels.py`).
  - 132 filled their arenas and ended there: 45 with `game-0195` in a seat, 29 with
    `game-0085`, 21 with `game-0304`, 17 with `game-0289` and 10 with `game-0262`; 32,
    34, 30 and 36 over the run's four quarters.
  - Planet Pollutant Virus 241.
  - Cards seen before: Fake Feather 32, at its two lines, Beast Rising 27, Neo Flamvell
    Shaman 7, Pyro Clock of Destiny and Infestation Wave 5 each, Mind Control and Rivalry
    of Warlords 3 each, Ultimate Obedient Fiend and Batteryman Fuel Cell 2 each, and one
    each of six more.
  - New: Diamond-Dust Cyclone (`game-0053`) and Gravi-Crush Dragon (`game-0302`), one
    each.
  - The selection with no answer at the 590th update (`game-0186`), cut off as
    `ba221814` does; there was no other.

The two arms are not told apart. Head to head the switch is not shown, and on the
measures they share `switched` is behind by a fifth of a point on the new decks alone,
at 6% more time an update. Switched into the pool's trained network, as into Cyber's,
the board as its cards is not shown to pay. Whether it pays made from scratch is asked of
the fourteen decks (above).

## Batch seven (design and readings fixed on 2026-11-13, before anything was built)

The pool goes on from `pooled`, agreed on 2026-11-13 on batch six's results. Batch seven
goes on as `pooled` with three changes, agreed the same day:

- its new decks are every usable list not yet taken, so none is held out: the games'
  lists end with it, but for the nine kept out until the fixes held for the experiment's
  end;
- half batch six's length: in every batch so far the new decks took 84% to 88% of their
  gain in the first half of the updates;
- the third reading's mark at 2 points: batch six's new decks had gained 3.1 points by
  its 3,840th update, from 51.8%, and these lists each bring one card that the lists
  before them in the order do not, and play fewer games.

- **Start**: `pooled`'s mean at 7,680 updates (step 78,960).
- **Decks**:
  - New, 113: `game-0370` to `game-0484` but `game-0379` and `game-0397`, the eleven held
    out until now among them. They bring 120 cards that no deck of the batches before
    holds.
    - `game-0397` holds Archfiend's Oath, a card that names a card, and is left out as
      the six lists before it that hold one are.
    - `game-0379` stays out until the double-tribute fix (above).
  - Old, 433: the seventy-one and the 362 game Decks of the batches before, `game-0001`
    to `game-0369` but `game-0037`, `game-0065`, `game-0093`, `game-0167`, `game-0171`,
    `game-0233` and `game-0335`.
  - Held out: none.
- **Run**: as `pooled`'s, over all 546 decks.
  - Held to the same two frozen networks: at an old deck's decisions in a duel between
    two old decks, the start, and in a duel between two of the seventy-one,
    `seventy-one-champions`.
  - 3,840 updates. A new deck plays 4.88 games an update, some 18,700 over the run,
    where batch six's had played some 21,700 by its 3,840th update; an old deck plays
    1.13 and an old mirror 0.10 duels. Of the 80.2 duels an update between two old
    decks, 7.8 are between two of the seventy-one.
  - Some 11 hours on the CUDA machine.
- **Measured** as `pooled`'s, at the start by the start and after 1,920 and 3,840
  updates by the mean of the last six kept checkpoints, the old decks and the mirrors
  after 3,840 alone (`eval_suites/pool-batch-seven-*.json`).
  - With no deck held out, there is no held-out measure.
  - The new decks' pairings grow to 782,864 games at each point and the old decks' to
    1,499,912, the measure to some seven hours.

Readings, at 3,840 updates, as batch six's but the third's mark:

1. *The mirrors are kept*: as batch six's, deal by deal against
   `seventy-one-champions`' check.
2. *The old decks hold*: as batch six's, on every ordered pairing of the 433 old decks.
3. *The new decks learn*: the mean over the 113 of each deck's share against the start,
   less at the start, the deals paired: **learns** when at least 2 points by two
   standard errors, **does not** when at most 1 point by two, otherwise **not
   decided**.

Told beside them, without a mark: what `pooled`'s told but the held out; the new decks at
1,920 updates; the mirrors deal by deal against `pooled`'s; the failed duels; the hours.

The batch's 546 decks hold 3,269 of the pool's 4,065 cards (80.4%), counted as before.
The nine lists kept out hold 19 more that no other list holds. What follows the batch is
decided on its results and the fourteen decks'.

## The board as its cards from scratch, at fourteen decks: results (read on 2026-11-14)

The two arms ran at once on the CUDA machine from 2026-11-13 02:56 at `9f0f8c9c` to
2026-11-14 00:11, 21.3 hours against the 30 reckoned
(`experiments/commands/fourteen-tokens.sh`, `analysis/fourteen-tokens/report.txt`). No
duel failed in the copies or in PPO, and every game of the measures played, 225,792 in
all with 16 drawn.

1. *The board as its cards from scratch*: **not decided**. `tokens` wins 49.85% ± 0.14%
   of the 12,544 deals head to head against `pooled` at 4,000, 4,240 and 4,320 updates
   of PPO, within half a point of even by two standard errors.

Beside it:

- The head to head along the way: 50.1% ± 0.2% at the copies' last checkpoints, 50.1% ±
  0.2% after 1,360 updates of PPO and 50.5% ± 0.2% after 2,800. No edge comes from the
  copy, and along PPO the two stay within a point of even.
- Against `all71` at the last three checkpoints: `pooled` 56.1% ± 0.4%, `tokens` 56.0% ±
  0.4%; `tokens` less `pooled`, deal by deal, −0.1 ± 0.3 points.
- `pooled` less the fourteen decks' `quarter` arm against `all71` on the same deals
  (56.2% ± 0.4%): −0.1 ± 0.3 points. A copy from recordings, at today's pins, lands
  where the copy that asked its champions did.
- The copies over their last 80 updates: `pooled`'s imitation loss 0.113 and 97.6% of
  the teachers' answers, `tokens`' 0.117 and 97.4%. Each took 12.0 seconds an update and
  7.5 hours, the two at once, where that record's one copy, which asked its champions,
  took 13.4 hours.
- PPO: `pooled` 10.4 seconds an update and 12.4 hours, `tokens` 10.6 and 12.8, the two
  at once. The board as its cards costs 3% an update here, 6% in batch six's
  `switched`.

Made from scratch at fourteen decks, as switched into a trained network, the board as its
cards is not shown to pay; it was ahead only for single decks. As agreed on 2026-11-12,
the seventy-one are not made from scratch with the board as its cards. Whether they are
made from scratch at all, with today's board and the copy's width or rate changed
(above), is decided on these results and the copy's.

**Current next step:** batch seven (above) runs on the CUDA machine since 2026-11-14
00:11 at `5b645ced`, some 18 hours with its measure.

The board as its cards made from scratch at fourteen decks is read (above), as are the
copy's width and rate with the copies' head to head against the teachers. Whether the
seventy-one are made from scratch, with today's board and a wider network or a rate
lowered as the copy goes, is decided on them. The scripts that failed and the selection
with no answer wait for the experiment's end.
