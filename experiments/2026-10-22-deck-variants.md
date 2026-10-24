# One deck, or one deck with its variants

**Status:** FUTURE

**Proposed:** 2026-10-22

**Last updated:** 2026-10-22 (proposed; not designed, its readings not yet
proposed)

Proposed on 2026-10-22, while
[A critic that sees the hidden cards](2026-10-22-privileged-critic.md) was
active, as a future experiment: to measure how a network of one deck learns
against a network of the same deck with its variants. It came up over
ygodata.com's split by format of DuelingBook's ranked matches, which counts
how often a deck's builds and staples are seen but holds no full lists, and
over the tournament lists of 2010 and 2011 already gathered in
`analysis/meta-decks-2011`.

## Question

Does a network of one deck learn it better, or faster, when its games deal
the deck's real variants, the lists actually played, than when they deal its
one list? And what does each play well: its own list, the variants it
trained on, and variants it never saw?

## What there is to build on

- **The lists**: `analysis/meta-decks-2011/decks.json` holds 515 tournament
  lists of September 2010 to September 2011, 252 of them complete with every
  main and Extra Deck card resolved (208 TCG, 41 OCG, 3 Worlds). Counted
  over their names' spellings: Plants (Tengu, Quickdraw Dandywarrior and
  Synchro builds) 31, Six Samurai 27, Gravekeeper's 18, Blackwing 14 (with
  the OCG's Graveyard Blackwing), X-Saber 14, Master Hyperion 9, Chaos 7.
- **Playing a list**: a setup's `lists` deals a `.ydk` in place of a deck's
  own list, the deck's pilot still asked alongside (`evaluate-suite
  --learner-list`, as [Darker Than Black](2026-10-19-darker-than-black.md)
  played a variant of Blackwing). That experiment found that a network keeps
  its edge with another list of its deck, and plays the cards it never
  trained on about as often as the others.
- **Not built**: training on several lists of a deck (the schedule giving a
  learner deck its lists, an update's games dealing them), and the
  conversion of the gathered records into `.ydk` files.

## A sketch, to be designed

Two networks of one deck, Blackwing by the lists there are and the curve of
`decks-01` there is to set them on (or Six Samurai or Plants, which have
more lists): its pilot's copy, then PPO, at the same games; one dealt its
own list, the other its own and its variants, some held out. Measured on
the deck's held-out deals with its own list, on the variants trained on,
on variants held out, and head to head in the mirror. The pilot imitated
was written for the deck's own list; with a variant it plays cards it may
not know, as with Darker Than Black, where it lost 6.3 points with the
variant list.

The readings are to be proposed and fixed before anything is built.
