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

**Current next step:** the design and readings of its first experiment, fixed
before anything is built. The games' Decks are the way to the pool and the
tournament lists are not, and those Decks have no pilots, so that neither imitation
nor the champions' recipe applies to them as they stand.
