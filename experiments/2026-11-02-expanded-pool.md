# The expanded pool: from seventy-one decks to the cards around them

**Status:** ACTIVE

**Proposed:** 2026-11-01

**Activated:** 2026-11-02, when
[A network of the seventy-one decks](2026-10-28-seventy-one-champions.md) was read
and closed. Its design and readings are to be fixed before anything is built.

**Last updated:** 2026-11-02 (opened; the opponents of the World Championship and
Tag Force games measured, which hold 89.6% of the pool)

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

**Current next step:** the test, built and queued on the CUDA machine on 2026-11-02.
