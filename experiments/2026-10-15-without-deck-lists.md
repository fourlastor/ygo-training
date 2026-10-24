# Without the deck lists

**Status:** COMPLETED

**Proposed:** 2026-10-15

**Activated:** 2026-10-15

**Completed:** 2026-10-15

**Last updated:** 2026-10-15 (closed the night it was read: the deck lists
are dropped from the model's input and its own earlier answers kept;
measured and read: the lists can go by the readings, and without its earlier
answers the copy is equivalent outside training and not decided on the
held-out deals)

Proposed on 2026-10-15, after it was found that a deck list longer than 48
cards keeps the model's calls off the CUDA graphs: remove the deck list (at
least the other player's) and test whether that changes anything, with
equivalence expected. Both training and held-out deals were to be tested,
held-out decks being where dropping the list could also improve play; the
criterion for dropping it is equivalence, not better play. Read here as:
measured on the decks of training and on decks held out of training, whose
lists were never among what the network learnt from. It waited for
[One network for more decks](2026-10-14-more-decks.md) to be read and
closed, which was the evening of 2026-10-15. **The design and the readings
below were fixed on 2026-10-15, before any run**, with PPO left out for now.

## Research questions

1. Does a network that is not told the other player's deck list play as
   well as one that is, on the decks of training and on decks held out of
   training?
2. Does a network of several decks told neither list play as well as one
   told both? A network of one deck is told the same list of its own at
   every decision, so it cannot say.
3. Does a network that is not told its own earlier answers play as well
   as one that is (added on 2026-10-15)?

## Why

- **What the lists are.** Every decision's input holds two deck lists,
  the seat's own and the other player's, each the Main Deck then the Extra
  Deck. The model reads each only as the mean of its cards' encodings
  (`PolicyValueNet._deck`), beside the state.
- **What they tell.** The other player's list says which deck it faces
  before any card of theirs has been seen; after a card or two, the field
  and the graveyard say it too. A network's own list never changes when it
  plays one deck. When it plays several, the list says which, as its hand
  does from the first decision, and its own Extra Deck is among the
  state's cards anyway.
- **What they cost.** A call answered from a CUDA graph is filled out to
  200 card rows a decision, and the two lists are 104 of them (the state
  96, the lists 48 and 56). A call that holds a list longer than 48 cards
  is answered without the graphs (`docs/commands.md`, its section on a deck
  list of more than 48 cards missing the graphs), as nearly every call of a
  network of more decks is now. At the legal sizes, lists of 75 cards, they
  would be 150 rows of 310.
- **At a table.** `edopro` has to be told the deck it faces
  (`--opponent`, required) because the input holds the other player's
  list, which a room does not tell.
- **Outside training.** Against a deck held out of training the network
  is told a list it never met. Without the lists it is simply not told,
  and it may play better there. Better play is told beside the readings; it
  is not the mark.

## A first look, without training

Whether networks trained with the lists use the other player's.
`analysis/deck-lists/swapped.py` plays a command of the trainer with the
model's reading of that list replaced, decision by decision: by the list
of the next of the thirteen decks of training (`rotate`: it believes it
faces another deck it has met), or by the mean of the thirteen (`mean`: no
deck in particular). Which deck a decision's list is, is told by its
cards. Told the true list it plays the trainer's own games, game for game
(checked on 24 held-out deals and 36 games outside training).
`bash analysis/deck-lists/run.sh OUT CHECKPOINT...` measures, on the
processor here:

- the 1,536 held-out deals (Blackwing against twelve other decks of
  training), told the true list, the next deck's and the mean;
- the 2,304 games against the eighteen decks outside training
  (`eval_suites/ygo-policies-blackwing-unseen.json`), told the true list
  and the mean of the thirteen of training in its place;

then sets them side by side (`compare.py`). Its rule, confirmed with the
readings below: a checkpoint USES the list when, told the next deck's
list, it wins at least 2 points and two standard errors fewer of the
held-out deals than when told the true one, deal by deal. The rest is told
beside it without a mark. The scripts are in `analysis/deck-lists/`, which
the repository does not keep.

**Taken on 2026-10-15** (agreed the night before, for the morning: at
least the inference), on the last checkpoints of `decks-01-copy` and
`decks-01` of [One network for more decks](2026-10-14-more-decks.md):
Blackwing alone, its copy after 240 updates and PPO with half of its duels
against itself after 1,920 more. Twenty minutes on the processor here
(`analysis/deck-lists/first-look/`). Each told another list, less told the
true one, game by game on the same deals:

| | Held-out deals, true list | Told the next deck's | Told the mean of training's | Outside training, true list | Told the mean of training's |
|---|---|---|---|---|---|
| `decks-01-copy` | 59.7% of 1,536 | −0.7 ± 0.6 | −1.2 ± 0.6 | 59.6% of 2,304 | +0.2 ± 0.3 |
| `decks-01` | 76.9% of 1,536 | **−2.7 ± 1.0** | −0.7 ± 0.9 | 71.9% of 2,304 | −0.8 ± 0.5 |

By the rule the copy is not shown to use the list, and the network after
PPO USES it: told another deck's list it wins 2.7 points fewer (2.6
standard errors; 148 deals lost that it won, 107 won that it lost). Told no
deck in particular, the mean of the thirteen lists, it loses under a point,
within its error; and against the decks outside training, the mean of
lists it has met in place of one it never met gives nothing back (−0.8 ±
0.5, where it might have been expected to do better without the list). A
network trained with the lists is misled by a wrong one more than a right
one helps it. Whether a network trained without them plays as well is the
experiment's own question, which this does not answer.

## Design (confirmed on 2026-10-15)

The model leaves out the parts of its input it is told to
(`train --leave-out`, built on 2026-10-15): each is given nothing where it
was, the network keeps every weight it has, and checkpoints keep what
their model left out. The input stays as it is, and
`native/ygo-model-input` with it, until a part is dropped. The controls
are copies the experiment on more decks made with the same seed and
settings. An imitating run's games are its pilots', so a copy that reads
less begins as the same network and plays the very games its control did:
the report checks both from the runs' own figures.

1. **`blackwing-own-list`**: the Blackwing pilot imitated from scratch as
   `decks-01-copy` was (240 updates at 3e-4, every game seen once with its
   result at four positions, 520 duels an update against the pilots of the
   thirteen decks, seed 2012), reading its own list and not the other
   player's. Its control is `decks-01-copy`.
2. **`three-no-lists`**: the Blackwing, Lightsworn and Gishki pilots
   imitated as `decks-03-copy` was (720 updates), reading neither list.
   Its control is `decks-03-copy`.
3. **`blackwing-no-history`** (added on 2026-10-15, confirmed with its
   readings): the Blackwing pilot imitated as `decks-01-copy` was, reading
   both lists and not its own earlier answers (the last 32, read by an
   LSTM). Its control is `decks-01-copy`. Added once the loop of
   [the experiment on more decks](2026-10-14-more-decks.md) was found, the
   last 32 answers being a further candidate for dropping, tested in this
   experiment too.

About two and a half hours on the card for the three
(`experiments/commands/without-deck-lists.sh`, then `PARTS=measure`; the
report is `tools/deck_lists_report.py`). It starts on the evening of
2026-10-15, once the arms of the experiment on more decks are made, after
that experiment's own measuring part. This record turns ACTIVE when that
experiment is read and closed, one being active at a time. Its copies are
imitating runs, which the rule against repeated commands (branch
`no-repeat`) cannot touch, and its measurements are taken before that rule
is merged, as those of the experiment on more decks are. Each is measured
with its control, at its last checkpoint: on the 1,536 held-out deals; on
the 2,304 games against the eighteen decks outside training, playing
Blackwing; and, for the network of three, on the screens of its three decks.

Left out for now, to be reconsidered if needed: PPO on from
`blackwing-own-list` as `decks-01` was run (half of its duels against
itself, at 1e-4) to 960 updates, about four hours alone. Its control would
be `decks-01`'s checkpoint after 960 updates, which the experiment on more
decks measures on the held-out deals anyway.

## Readings (confirmed on 2026-10-15)

Equivalence is the mark for dropping a list, not better play. In each
reading the gap is the network that reads less minus its control, game by
game on the same deals, with its standard error.

1. *The other player's list, on the decks of training.*
   `blackwing-own-list` against `decks-01-copy` on the 1,536 held-out
   deals. The list can go (**equivalent**) when the gap less two standard
   errors is above −2 points. It stays (**worse**) when the network that
   reads less wins at least 2 points and two standard errors fewer.
   Otherwise the reading is not decided.
2. *The other player's list, outside training.* The same two networks on
   the 2,304 games against the eighteen decks outside training, by the
   same rule.
3. *Both lists, in a network of three.* `three-no-lists` against
   `decks-03-copy` over the screens of its three decks together (4,992
   games), by the same rule.
4. *Its own earlier answers, on the decks of training* (confirmed on
   2026-10-15 with the third copy). `blackwing-no-history` against
   `decks-01-copy` on the 1,536 held-out deals, by the same rule.
5. *Its own earlier answers, outside training* (confirmed with the fourth).
   The same two networks on the 2,304 games against the eighteen decks
   outside training, by the same rule.

Told beside them without a mark: each gap where it is over (better play);
the network of three on the held-out deals and outside training; and each
network's agreement with its pilots. A copy set beside another of the same
seed was read with a standard error of about 0.7 points on a screen
([One network, several decks](2026-10-12-one-network-several-decks.md)),
so a network that loses nothing is found equivalent by these rules about
four times in five.

## The run

On the CUDA machine, from the repository root, started at 19:11 on
2026-10-15 at `e4dbbd4e`, as one line after the measuring part of the
experiment on more decks:

```sh
PARTS=measure bash experiments/commands/more-decks.sh; AT_ONCE=3 bash experiments/commands/without-deck-lists.sh && PARTS=measure bash experiments/commands/without-deck-lists.sh
```

The copies began at 22:19, three at once, once that measuring part had
ended; the two of Blackwing were made by 22:51 and the network of three by
23:27. The measuring part ran from 23:27 to 23:40;
`analysis/without-deck-lists` (its report in `report.txt`).

## Results (read on 2026-10-15)

Each network that reads less, less its control, game by game on the same
deals, at the last checkpoint of each:

| Reading | Reads less | Control | Gap | By the rule |
|---|---|---|---|---|
| 1. The other player's list, decks of training (1,536 held-out deals) | 59.5% | 59.7% | −0.2 ± 0.8 | **equivalent** |
| 2. The other player's list, outside training (2,304 games) | 59.6% | 59.6% | +0.0 ± 0.8 | **equivalent** |
| 3. Both lists, the network of three, its three screens (4,992 games) | 54.3% | 53.7% | +0.6 ± 0.5 | **equivalent** |
| 4. Its own earlier answers, decks of training (1,536 held-out deals) | 59.4% | 59.7% | −0.3 ± 0.9 | not decided |
| 5. Its own earlier answers, outside training (2,304 games) | 60.1% | 59.6% | +0.5 ± 0.8 | **equivalent** |

Reading 4 misses equivalence by a tenth of a point: its gap less two
standard errors is −2.1 points. Told beside them without a mark, the
network of three that reads neither list on the held-out deals, 58.2%
against 59.3% (−1.1 ± 0.9), and outside training, 60.9% against 60.5%
(+0.4 ± 0.7). Each copy began as its control did and was taught on the
very games it was (no update of other games), and answers as its pilots
as often: 95.1% against 95.0%, 95.4% against 95.3%, 94.9% against 95.0%.

What it says, by the rules confirmed before the run:

- **The deck lists can go.** A copy that never reads the other player's
  list plays as well as one that does, on the decks of training and on
  those outside it; a copy of three decks that reads neither list plays as
  well on their screens.
- **Its own earlier answers can go outside training, and on the decks of
  training it is not decided**, by a tenth of a point.

What it does not say: these are copies, and PPO was left out. A network
trained under PPO without the lists, or without its earlier answers, is
not measured. The first look found that the network of Blackwing after
PPO, trained with the lists, loses under a point when told no deck in
particular in place of the other player's list (−0.7 ± 0.9), and is
misled by a wrong one (−2.7 ± 1.0). What its own list is worth to a
network of many decks under PPO is not measured either.

## What follows

Closed on 2026-10-15, the night it was read, with every reading taken:

- **The deck lists are dropped from the model's input**: both, the other
  player's and its own, as one change in three places (`AGENTS.md`).
- **Its own earlier answers are kept**: the reading on the decks of training
  is not decided, and the last 32 answers do not keep a call off the graphs
  as a long list does.
- **The order** agreed that night: the rule against repeated commands is
  merged first, into `main` of `native/ygo-model-input` as well, rather than
  pushed as a branch of its own, being needed everywhere in any case; then
  two measurements of the checkpoints of
  [One network for more decks](2026-10-14-more-decks.md), which read the
  lists as the new input will not (the networks playing
  the eighteen decks outside training themselves, and the network of
  thirteen against the network of Blackwing alone in the mirror); then the
  input change.

PPO without the lists stays unmeasured. The first network made on the new
input will show whether anything is missing: set against the numbers
here, at the same updates, it is read as these copies were.
