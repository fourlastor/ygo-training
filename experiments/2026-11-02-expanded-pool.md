# The expanded pool: from seventy-one decks to the cards around them

**Status:** ACTIVE

**Proposed:** 2026-11-01

**Activated:** 2026-11-02, when
[A network of the seventy-one decks](2026-10-28-seventy-one-champions.md) was read
and closed. Its design and readings are to be fixed before anything is built.

**Last updated:** 2026-11-02 (opened)

The goal since 2026-10-24 has been a champion of all seventy-one decks and then the
pool around them. The first is released as
[seventy-one-champions](../models/seventy-one-champions.md). This record is the
second: a network that plays well beyond the decks it was made on.

## What there is to build on

- **The pool**: the 4,013 cards World Championship 2011 allows. The seventy-one
  decks hold 955 of them, and 92.7% of the slots of the 2010 and 2011 tournament
  lists gathered in `analysis/meta-decks-2011`, counted at the pool's limits
  (`tools/played_cards.py`). By the cards it is a quarter of the pool, by what was
  played nearly all of it.
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

**Current next step:** what the pool is taken to be, and then the design and
readings of its first experiment, fixed before anything is built.
