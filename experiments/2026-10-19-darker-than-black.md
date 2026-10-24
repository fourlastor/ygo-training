# Darker Than Black: does a network learn decks, or cards?

**Status:** COMPLETED

**Proposed:** 2026-10-19

**Activated:** 2026-10-19

**Last updated:** 2026-10-19 (design and reading fixed before anything was
built; built and tried small, run, read and closed that day)

Proposed on 2026-10-19, when
[Thirty-one decks](2026-10-18-thirty-one-decks.md) had been read: whether
a network learns decks or cards, to be found out by seeing whether a
variant of a certain deck plays many points worse than the original. The
variant is Darker Than Black, another Blackwing list of the same format,
kept here unchanged as `data/decks/darker-than-black.ydk`. No pilot was written for
it, and how it stands against any other deck is not known. The measure
proposed was the difference between what Blackwing's pilot wins with
Darker Than Black against the original Blackwing deck (the policy piloting
it) and what the model wins with it. The original list was set beside it
as the baseline, with the thirteen opponents of the Blackwing screen in
place of the mirror alone, and four times the screen's games. **The design
and the reading below were fixed on 2026-10-19, before anything was
built.**

## Research question

Does a network that plays Blackwing's list better than its pilot keep that
edge with another list of the same deck?

## What the list is to the networks (2026-10-19)

- **The main deck, 40 cards:** 24 copies of cards of Blackwing's list (20
  of the list's own copies, and more of Blackwing - Kalut the Moon Shadow,
  Bottomless Trap Hole and Dimensional Prison); 4 of cards the other twelve
  decks of training hold (Lightning Vortex, Mirror Force, Pot of Avarice,
  Swords of Revealing Light); 3 that only the thirty-one decks hold
  (Smashing Ground ×2, Solidarity); and 9 of cards no deck of the pool
  holds (Blackwing - Elphin the Raven ×2, Blackwing - Mistral the Silver
  Shield ×2, Blackwing - Breeze the Zephyr, Blackback, Delta Crow - Anti
  Reverse, Mind Control, Battlestorm). Of Blackwing's list it leaves out
  20 copies: Jin the Rain Shadow ×3, Zephyros the Elite ×2, Pot of Duality
  ×2, Seven Tools of the Bandit ×2, Threatening Roar ×2, Trap Hole ×2,
  Allure of Darkness, Book of Moon, Dark Hole, Giant Trunade, Torrential
  Tribute, and one each of Blizzard the Far North and Sirocco the Dawn.
- **The extra deck, 15 cards** (Blackwing's list has 8): 7 of cards of
  Blackwing's (Black-Winged Dragon, Silverwind the Ascendant, Armed Wing
  ×2, Armor Master ×3); 5 the other twelve decks hold (Ally of Justice
  Catastor ×2, Brionac, Colossal Fighter, Stardust Dragon); Shooting Star
  Dragon, which only the thirty-one hold; and Dark End Dragon and Goyo
  Guardian, which none holds.
- Of a card it never trained on a network knows what is printed (its kind,
  attribute, type, level, ATK and DEF, its archetype) and an identity
  never trained. The policies' code names none of the nine either: the
  Blackwing pilot plays them by its general rules.

## Design (fixed on 2026-10-19)

- **The networks**, no training: `half` of
  [Thirteen decks](2026-10-16-thirteen-decks.md) (thirteen decks, Blackwing
  one of them) and `all` of [Thirty-one decks](2026-10-18-thirty-one-decks.md),
  each by its checkpoints after 1,680, 1,800 and 1,920 updates of PPO.
- **The games:** the wide Blackwing screen
  (`eval_suites/ygo-policies-blackwing-wide.json`), Blackwing against each
  of the thirteen decks' pilots, 512 games a pairing, 6,656 in all; the
  first 128 of each pairing are the screen's own. Played with Blackwing's
  list and with Darker Than Black, by each network and by Blackwing's
  pilot, on the same seed.
- **A network's edge with a list:** its share of a game (the share of its
  three checkpoints that win it) less the pilot's win, game by game. The
  network and the pilot play the list on the same deals against the same
  opponents, so that what the list is worth against them, which nobody
  knows, is the same for both.
- On the screen, with 128 games a pairing, Blackwing's pilot wins 60.3%
  with its list, `half` 71.9% (+11.6) and `all` 69.3% (+9.0).

## Reading (fixed on 2026-10-19, before anything was built)

For each of `half` and `all`: its edge over Blackwing's pilot with Darker
Than Black less its edge with Blackwing's list, over the thirteen
opponents: **kept** when the gap less two standard errors is above
−2 points, **lost** when the gap and two standard errors are below
−2 points, otherwise not decided.

Told beside it without a mark: the mirror on its own (against Blackwing's
pilot with its list), each opponent, and each player's own rate with each
list. If the edge is lost, what follows is telling the cards never trained
on from the list itself: a second list of the same idea made only of cards
the networks trained on, which would be built.

## The build (2026-10-19)

- **A list read from a file:** `evaluate-suite --learner-list PATH` plays
  the learner's seat with the deck list of that `.ydk` file in place of
  its deck's own; the pilot of its deck plays it (`--pilot`) or is asked
  alongside the model, and the result names the list (its path and
  SHA-256). In the library a duel's setup takes a list read from a file
  for either seat (`lists`, `crates/ygo-env/src/episode.rs`). Held by
  `crates/ygo-env/tests/duels.rs` (each seat's list read from its own file
  plays the duel of the bench; Darker Than Black in Blackwing's seat is
  another duel, played by Blackwing's pilot, and the same duel with a
  caller that follows that pilot alongside it) and by
  `tests/test_learner_list.py` (through the pool: the Blackwing list from
  its file gives the games of `blackwing`).
- **The experiment** is a suite, a list, a script and a report:
  `eval_suites/ygo-policies-blackwing-wide.json`,
  `data/decks/darker-than-black.ydk`,
  `experiments/commands/darker-than-black.sh` and
  `tools/darker_than_black_report.py`
  (`tests/test_darker_than_black_report.py`). Tried small on the
  development machine's processor that day: the script as written but
  small (a suite of the mirror and Monarch, four games each, three
  checkpoints of format 15 at hand for each network), its six measurements
  and the report.

## The run

Run on the CUDA machine on 2026-10-19 from 18:26 to 18:38, at `953c2685`
with the submodules at their pins, nothing else running there, from the
repository root:

```sh
bash experiments/commands/darker-than-black.sh
```

`analysis/darker-than-black` was brought back that evening, the reading in
`report.txt` there.

## Results (read on 2026-10-19)

The reading, each network by its checkpoints after 1,680, 1,800 and 1,920
updates, game by game against Blackwing's pilot on the same games, over
the thirteen opponents, 6,656 games a list:

- **`half`: kept.** Its edge over the pilot is +12.2 ± 0.5 with
  Blackwing's list and +19.2 ± 0.6 with Darker Than Black: +7.0 ± 0.8.
- **`all`: kept.** +8.6 ± 0.5 and +16.6 ± 0.6: +8.0 ± 0.8.

Beside it, without a mark:

- **The networks win as much with the variant; the pilot does not.** With
  Blackwing's list `half` wins 71.9% of the games, `all` 68.3% and the
  pilot 59.7%; with Darker Than Black 72.5% (+0.6 ± 0.7), 69.9%
  (+1.6 ± 0.7) and 53.4% (−6.3 ± 0.9). The edges grow because the pilot
  plays the variant worse, not because the networks play it better: the
  pilot was written for Blackwing's list, plays the nine cards no deck
  holds by its general rules, and misses cards of the list its rules lean
  on. The mark **kept** says the edge is not lost, and nothing more of
  what the variant is worth.
- **In the mirror** (against Blackwing's pilot with its list, 512 games):
  the pilot wins 50.4% with the list and 41.6% with the variant, `half`
  63.8% and 63.4%, `all` 59.8% and 61.3%.
- **Against each of the thirteen opponents** both networks' edges are
  larger with the variant, least against Spellcaster (`half` +4.4 with the
  list, +6.1 with the variant) and Morphtronic, most against Dragunity
  (+10.9 and +23.2), Heroes and Rock-block.

What it says: the networks play a list of Blackwing they never saw, with
twelve (`half`) or nine (`all`) of its forty main-deck copies cards they
never trained on, at the rate they play the list they trained on, where
the pilot written for that list loses six points with it. Nothing in it
says that they learned the list rather than its cards. How strong the
variant is in itself is not known, so it does not show that they play it
as well as it allows: only that they do not lose with it what its pilot
loses.

## What the networks do with the cards they never trained on (2026-10-19)

Asked the same evening, to tell whether a card a network never trained on
costs it a deck's worth of training: on the same deals of the wide screen
with Darker Than Black, each network by its last checkpoint (`half` after
1,920 updates, `all` after 1,920 too), at the decisions of a turn (Idle,
Battle and Chain) where an answer summons, sets or activates a card of the
list, how often the network takes one, beside how often the pilot asked
alongside it would (`tools/card_use.py`, `tests/test_card_use.py`; played
on the development machine's processor, the checkpoints brought from the
CUDA machine; `analysis/darker-than-black/card-use/`):

| Cards of the list, by what they are to the network | `half`: offered, network, pilot | `all`: offered, network, pilot |
|---|---|---|
| Of Blackwing's list | 410,225: 14.6%, 17.5% | 408,509: 14.5%, 16.4% |
| Of another deck trained on | 73,171: 13.0%, 12.4% | 75,703: 24.1%, 16.2% |
| Never trained on | 119,540: 19.2%, 11.9% | 104,584: 16.1%, 9.8% |

- **The networks play the cards they never trained on**, as often as the
  others and more often than the pilot, which plays them by its general
  rules: Mind Control 40.8% of its chances for `half` and 23.2% for `all`
  (the pilot never), Blackback 38.7% and 43.8% (the pilot 11.1% and 9.1%),
  Delta Crow - Anti Reverse 15.0% and 7.8% (6.0% and 1.8%). Solidarity,
  which `half` never trained on and `all` did, 18.3% and 40.6% (the pilot
  0.1% and 0.0%).
- **Nor do they follow the pilot on the cards they trained on**: Blackwing
  Armed Wing 56.9% of its chances for `half` where the pilot takes 2.5%,
  Blackwing Armor Master 20.4% where it takes 63.1%, Brionac 4.0% where it
  takes 39.5%: what PPO made of the copy.
- **This tells use, not worth**, which the next measure reads.

**What those cards are worth to the networks** (asked that evening): the
same deals played by each network's last checkpoint as they are, and again
with the network barred from summoning, setting and activating the cards
of a group at the decisions of a turn, a decision that offers nothing else
left whole (`tools/forbidden_cards.py`, `forbid` of `pool.evaluate_games`,
`tests/test_forbidden_cards.py`; `analysis/darker-than-black/forbidden/`).
What a group is worth is what the network loses without it, against cards
left dead in the hand:

| Barred | `half`: copies, won, on the same deals | `all`: copies, won, on the same deals |
|---|---|---|
| Nothing | 72.1% | 71.4% |
| The cards never trained on | 12 + 3 extra: 46.8%, −25.3 ± 0.6 | 9 + 2 extra: 53.0%, −18.4 ± 0.6 |
| The cards of another deck trained on | 4 + 5 extra: 61.5%, −10.6 ± 0.5 | 7 + 6 extra: 51.0%, −20.4 ± 0.6 |

- **The cards a network never trained on are worth to it about what the
  cards it trained on are:** some 2.1 points a main-deck copy for both
  networks, against 2.6 (`half`) and 2.9 (`all`) for those of the other
  decks, whose count holds their extra-deck cards too. Barred, they change
  11.5 and 8.7 of a game's answers.
- What a network gets from them is not told apart from what any monster
  or spell gives: these are mostly Blackwings, of an archetype it knows,
  and spells and traps of simple effects. A card of a mechanism no card
  of training shares is not measured.

## Which cards people play (2026-10-19)

A second question that evening, which cards people actually play, was put
to Codex with a prompt: tournament decklists of September 2010 to
September 2011, TCG and OCG, counted by card (`analysis/meta-decks-2011/`:
the lists, the count `cards.csv` and a `README.md` with the sources, how
they were read and their biases; mostly the published top cuts of large
events, 7% of the slots of names not resolved, most of them Japanese). 432
lists enter the count (301 TCG, 128 OCG, 3 of Worlds), 23,255 main- and
extra-deck slots, 726 cards that the WC2011 pool allows. Against the cards
of the pool's decks (`tools/played_cards.py`,
`tests/test_played_cards.py`; the 72 decks read from the policies'
`origin/ygo-agent-benchmark-work` as our checkout last fetched it,
2026-10-10):

| Decks | Cards of the 726 | Their share of the slots | Of the 100 most played |
|---|---|---|---|
| The thirteen of training | 170 | 60.5% | 52 |
| The 31 of the pin | 280 | 84.2% | 86 |
| The 72 of the policies' branch | 432 | 92.8% | 96 |

- **Within our format:** cards the pool forbids (a count of 0) or does not
  hold are not counted. Counted again at the pool's limits, a list's copies
  of a card no more than the WC2011 list allows it (Book of Moon, limited
  to one, is played above that in 97 lists), the slots are 2.1% fewer and
  the shares barely move: 59.8%, 83.9% and 92.7% (`--decks`).
- **What people play is few cards:** 38 fill half of the slots, 115 80%,
  186 90%, 269 95%.
- **Held by none of the 72:** 294 cards, 7.2% of the slots; the most
  played of them Mind Control (0.59% of the slots, in 30% of the lists),
  Ancient Fairy Dragon, Arcanite Magician, Blackwing - Vayu the Emblem of
  Honor, Ally of Justice Decisive Armor, Dark End Dragon and D.D. Crow
  (`analysis/meta-decks-2011/coverage.txt`).

- **At the pin moved to the policies' `main` (`82a7c61`, 2026-10-19)** the
  pool holds 71 decks, each with its pilot: the same 432 cards, 92.8% of
  the slots. The duels of the 31 decks it held before are unchanged: the
  bench's 192 Blackwing games of the procedure (docs/commands.md, its
  section on moving the pin) identical, and the pilots' 3,744 games of the
  decks outside the thirteen and 6,656 of the wide Blackwing screen ended
  every one as they did at `a10d7ec` (`analysis/pins/`).

What it says, with what the cards never trained on are worth to the
networks: covering what people play is not training on the whole pool.
The decks the policies already have hold nine tenths of it, and a card of
a kind the networks know is worth to them about what a card they trained
on is.

## What follows

Closed on 2026-10-19, the day it was run and read, with every reading
taken and the games' own estimates for the run after it. What it left: a
network plays the cards it sees, the ones it never trained on too, so what
training has to cover is cards and not lists; and people play few cards,
most of which the policies' decks already hold. The pin was moved to the
policies' `main` and its 71 decks that evening (`4b308c40`). Next,
[Seventy-one decks](2026-10-19-seventy-one-decks.md): a network of every
deck of the pool, its design and readings fixed the same day.
