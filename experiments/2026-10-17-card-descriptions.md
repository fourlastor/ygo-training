# Card descriptions and hidden identities, for decks outside training

**Status:** COMPLETED

**Proposed:** 2026-10-17

**Activated:** 2026-10-17

**Completed:** 2026-10-18

**Last updated:** 2026-10-18 (closed the day it was read; run and read:
neither the descriptions nor the identities hidden, together or alone, told
apart from the control on the decks outside training, the decks of training
kept. Built, tried small, and started on the CUDA machine on 2026-10-17.
Opened when [Thirteen decks on the new input](2026-10-16-thirteen-decks.md)
was closed, its design and readings fixed before anything of it was built;
one change from that design in the build, the description added to the
identity rather than set beside it, below)

Proposed on 2026-10-17, when the question came up of how the networks could
come to play decks they never trained on, the longer goal being a network
able to play any reasonable deck. The network of thirteen decks wins 34.1%
of the games of the eighteen decks outside training where those decks' own
pilots win 46.6% (the follow-up of
[One network for more decks](2026-10-14-more-decks.md)), and the one made
after it on the new input, after twice the updates, 35.5%. The question was
how a card effect embedding would work; an open model, voyage-4-nano, was
chosen to make one, and the vectors were measured first, without training
(below), to see whether texts of similar meaning get vectors near each
other. An experiment on it followed. Two measures of the decks outside
training were asked for as well: how many of their cards the decks of
training hold, the overlap being unknown and the results hard to judge
without it, and how near their cards' descriptions come to those trained on,
since a card whose effect is near a trained card's should be played like it.
Both are told beside the readings. **The design and the readings below were
fixed on 2026-10-17, before anything was built**.

## Research questions

1. Does a network of the thirteen decks that is told what each card's text
   says, and trained with some cards' identities hidden, play the decks
   outside training better than one that is not?
2. Does it play the decks of training as well?
3. Which of the two carries what it gains: the descriptions, or the
   identities hidden?

## Why

- **What a network knows of a card.** Its identity, read from its code in
  two halves of sixteen bits, each a vector learnt from the games of the
  cards that share it (`CardIdentityEmbedding`), and its printed data:
  kind and subtypes, attribute, race, level, archetypes, ATK and DEF. What
  the card does is not in the input. A card no deck of training holds is
  shown by halves learnt for other cards or never learnt, and by its
  printed data.
- **The decks outside training are mostly such cards**: in fifteen of the
  eighteen, more than half (below).
- **A description** is a fixed vector of what the card's text says, the
  same for the cards trained on and the others: a new card whose text
  reads as a trained one's is near it, and means to the network what it
  learnt that one to mean.
- **Identities hidden.** A network that always has a card's identity can
  learn each card by it and leave the description unread. Hidden for a
  share of the cards in training, it leaves the description to tell what
  those cards do; at play, a card never trained on is then shown as those
  were, without an identity, rather than by halves learnt for other cards.

## A first look, without training (2026-10-17)

Before anything was designed the vectors were measured against what the
cards' scripts declare: `bash experiments/commands/card-embeddings.sh`,
about three minutes on a processor (`tools/card_embedding_check.py`,
`tools/card_text_embeddings.py`, which fetches voyage-4-nano at its commit
where it is not cached). The texts are the 3,761 effect texts of the
policies' pool (`data/wc2011.lflist.conf` there, 4,066 cards), a card's
own name put as `this card` and the database's note that a text is
unofficial left out as it is read. Measured first on 3,754: the texts had
left out the seven cards of the pool whose names are only treated as
another's, taken for alternate artworks (Harpie Lady 1, 2 and 3 and Cyber
Harpie Lady, as Harpie Lady; A Legendary Ocean, as Umi; Neo-Spacian
Twinkle Moss and Marine Dolphin), and measured again the same day with
them, every figure within 0.002 (`analysis/card-descriptions/first-look`).
What a card's script declares (its
categories, types, codes and properties, and the operations of the duel it
calls) judges whether cards near each other in the vectors do the same
things:

| Vectors | Numbers | What a card's 10 nearest share with it | Read out by a line | A twin nearest |
|---|---|---|---|---|
| random | 128 | 0.099 | 0.499 | 0% |
| the words of the texts (TF-IDF, 128 main directions) | 128 | 0.391 | 0.961 | 67% |
| voyage-4-nano | 2,048 | 0.360 | 0.953 | 73% |
| voyage-4-nano, 128 main directions | 128 | 0.369 | 0.953 | 77% |
| the words and voyage's 128 | 256 | 0.414 | 0.965 | 74% |

What the 10 nearest share is the Jaccard of the scripts' declarations;
read out by a line, the mean area under the curve of a linear readout of
each declaration, fitted on half the cards and tried on the others; a twin
is a card whose script is another's but for its numbers (95 cards), and
the share is of those whose twin is the nearest. The words and voyage
together are this experiment's descriptions. In them two cards at random
are at a cosine of −0.02, a card's nearest at 0.68; cards of the same
effect are each other's nearest at 0.80 to 0.95 (Solemn Judgment and
Solemn Warning 0.90, Sangan and Witch of the Black Forest 0.93), and an
effect shared with clauses more is further (Monster Reborn and Premature
Burial 0.29, 109 cards nearer). Vectors made from the scripts, each source
judged by the other, did no better than the texts; by the rule that the
texts are preferred where the scripts do no better, being quicker to produce
and not tied to the engine's internals, the scripts are left out unless
asked (`SCRIPTS=1`).

The eighteen decks outside training by what they share with the thirteen
of training, against what `decks-13` kept of their pilots' wins after
1,680 to 1,920 updates (its share of the games won over the pilot's), by
the experiment's report (`tools/card_descriptions_report.py`) on those
games, the least covered first:

| Deck | Same cards | Same effect (0.8 or more) | Nearest, on average | Pilot's wins kept |
|---|---|---|---|---|
| Arcana | 12% | 18% | 0.51 | 88% |
| Watt | 5% | 11% | 0.55 | 62% |
| Pyramid | 18% | 22% | 0.56 | 51% |
| Ojama | 9% | 13% | 0.59 | 47% |
| Destiny HERO | 25% | 28% | 0.60 | 54% |
| Toon | 15% | 28% | 0.60 | 41% |
| Fortune Lady | 35% | 40% | 0.61 | 50% |
| Harpie | 32% | 35% | 0.62 | 72% |
| Six Samurai | 36% | 36% | 0.62 | 65% |
| Countdown | 32% | 32% | 0.65 | 55% |
| Draconic Might | 29% | 29% | 0.67 | 81% |
| Gravekeeper | 38% | 42% | 0.68 | 86% |
| Karakuri | 32% | 36% | 0.68 | 74% |
| X-Saber | 40% | 42% | 0.70 | 82% |
| Machina | 46% | 49% | 0.75 | 89% |
| Tele-DAD | 56% | 56% | 0.77 | 76% |
| Quickdraw Plant | 60% | 60% | 0.82 | 61% |
| Verdict | 68% | 72% | 0.91 | 100% |

A deck's cards are counted with their copies and its Extra Deck, an
alternate artwork as its original; the same effect is a card trained on,
or one whose nearest trained description is at 0.8 or more; the nearest on
average is that closeness over a deck's cards, 1 for a card trained on and
a card without a text (a Normal Monster) not counted. A first count made
by hand that morning counted a Normal Monster as nothing near, and had the
seven cards above without a text. Few cards of these decks have a twin
among those trained on (Sakuretsu Armor has Dimensional Prison, at 0.93);
many have a cousin at 0.6 to 0.8 (Shining Angel and UFO Turtle 0.75, both
bringing a monster from the deck when destroyed by battle) and a few
nothing near (Final Countdown's nearest at 0.39, Double Summon's at 0.31).
What the network kept goes with each of the three measures about equally
(a correlation of +0.51 to +0.52), a quarter of the spread between decks;
the rest is the decks' own (Countdown wins by outlasting, Six Samurai by
swarming).

## Design (fixed on 2026-10-17)

Three networks of the thirteen decks, each made from scratch as `half` of
[Thirteen decks](2026-10-16-thirteen-decks.md) was: the pilots of the
thirteen decks imitated (240 updates a deck, 3,120 in all, at 3e-4, every
game seen once with its result at four positions), then PPO from that copy
at 1e-4, one pass over an update's 520 duels, 260 against itself and the
others against the pilots of the thirteen decks, to 1,920 updates; seed
2012, the three at once through MPS, under the rule against repeated
commands as `half` was.

| Arm | Descriptions | Identities hidden |
|---|---|---|
| A (`both`) | yes | yes |
| B (`descriptions`) | yes | no |
| C (`hidden`) | no | yes |
| the control, `half`, already trained | no | no |

- **The descriptions**: the words and voyage-4-nano's 128 main directions
  of the first look, 256 numbers a card of the pool, made once and kept in
  the repository as a data file (about 2 MB), so that no machine needs the
  model. The model holds them as a fixed table, not trained, looked up by
  the code each card of the batch already carries: an alternate artwork
  reads its original's row, a card face down (code 0) or one not in the
  table a row of zeros. One trained layer turns a row into what the card
  encoder takes, beside the identity, the kinds and the numbers. The batch
  arrays do not change; the checkpoint carries the table, in format 16.
- **Identities hidden**: in each update a quarter of the pool's cards,
  drawn again each update, have their identity blanked, their printed data
  and description kept: the same cards in that update's games and in its
  training, in the copy and in PPO alike. The checkpoint keeps which cards
  it was trained on. At play, A and C show every card never trained on
  with its identity blanked; B shows it with its identity as it is, as the
  control does.
- **The control** costs no training: `half` has checkpoints after 1,680,
  1,800 and 1,920 updates, those the readings take. B and C tell which of
  the two parts of A carries what it does.
- **In the build, one change, for confirmation before the run**: the
  description is added to the card's identity, through a layer of 256
  numbers to the identity's 96 without a bias, rather than set beside it.
  Beside it, every layer that reads a card would grow by as many inputs,
  and the arms told the descriptions would be wider networks than the
  control: what they gain could then be a wider network's. Added, every
  layer is the control's and the description's is the one more; under the
  seed an arm begins as the control's network, that layer drawn after it
  (`tests/test_card_table.py`), and a card without a description reads its
  identity alone. The identity is read in two places, the encoder of a
  card and the seat's earlier answers, and the description is added in
  both.
- **How the identities are hidden**: a quarter of the table's codes, 3,690
  of the database's 14,759, drawn from the run's seed and the update: about a
  quarter of the cards that come in an update's games. A
  hidden card's identity is read as one learnt vector, its printed data and
  description as they are. A card is trained on when an update's decisions
  show it with its identity, and the checkpoint keeps those.
- **The engine.** The arms are made at the policies' `a10d7ec`, whose
  engine reuses what Lua frees; `half` was made at `1b85b63`, whose engine
  aborted the process when a duel filled its arena, which no duel of its
  training did. The checks found every game as before (14,880 of the
  pilots, 3,744 of `half` on the decks outside training; the thirteen-decks
  record).

By reckoning about thirteen and a half hours for the copies and PPO, then
about two and a half of measuring.

## Readings (fixed on 2026-10-17, before anything was built)

Each by an arm's checkpoints after 1,680, 1,800 and 1,920 PPO updates
against the control's same three, game by game on the same games, a game
read by the share of the three checkpoints that win it:

1. *The decks outside training.* A against the control on the eighteen
   decks outside training, each played against the pilots of the thirteen
   decks of training, 16 games a pairing
   (`eval_suites/ygo-policies-unseen-played.json`, 3,744 games a
   checkpoint): **better** at 2 points and two standard errors more,
   **worse** at 2 points and two standard errors fewer, otherwise not told
   apart.
2. *The decks of training kept.* A against the control over the thirteen
   screens (21,632 games): **equivalent** when the gap less two standard
   errors is above −2 points, **worse** when the gap plus two standard
   errors is below −2 points, otherwise not decided.
3. *Which part carries it.* B against the control, and C against the
   control, on the decks outside training, each by the rule of reading 1.

Told beside them without a mark: the copies on the decks outside training;
deck by deck on the eighteen; each arm's gap to those decks' own pilots,
who win 46.6% of the same games; for every deck outside training the three
measures of the first look and the share of its pilot's wins each arm
keeps; and A against the control on the nine decks least covered by the
nearest on average and on the nine most covered, game by game.

## The build (2026-10-17)

Built on the development machine and tried small there:

1. **The table** (`1a1c8afc`, after `93735ed8` gave the check's texts the
   seven cards above): `tools/card_descriptions.py` made
   `data/card-descriptions.npz` (1.8 MB, SHA-256 `a12ab8a4…0cac`) of the
   first look's texts and voyage-4-nano's vectors of them: every code of
   the card database (14,759) with the row of its description, 3,906 codes
   with one of 3,761 descriptions. An alternate artwork has its original's
   row, the pool's or one a deck holds that the pool does not list (decks of
   training hold Monster Reborn, Stardust Dragon and Black Rose Dragon so,
   among others); cards whose texts read the same share one; a Normal
   Monster, a token and a card outside the pool have none. The file keeps
   the mean and the spread each number was set by, from which the report
   reads the descriptions as the first look measured them.
2. **The model** (`e60ff4fb`): `ygo_training/card_table.py`, and `train
   --card-table FILE` with `--card-descriptions` and `--hide-identities
   SHARE` (`docs/commands.md`, beside `--leave-out`). A model told more of
   a card is written in format 16 and one told nothing more still in format
   15, and both load; a checkpoint loaded to play hides what a model hides
   at play, an update hides its own draw, and `evaluate-suite` builds its
   model as its first checkpoint says. `tools/export_checkpoint.py` refuses
   a model told more. `tests/test_card_table.py` holds the table (every
   card with an effect of every deck has a description), the model (it
   begins as the control's under the seed, the description is added, a
   hidden identity is read as one, at play those never trained on, answered
   at the graphs' fixed sizes as by itself), its checkpoints, an update's
   draw (the same for the same update, another for the next, the cards
   shown kept as trained on), and a run of two updates measured with
   `evaluate-suite`. The whole suite passes.
3. **The script and the report**: `experiments/commands/card-descriptions.sh`
   and `tools/card_descriptions_report.py` (`tests/test_card_descriptions_report.py`).

Tried small on the development machine's processor on 2026-10-17, the
script as written but small (two decks, copies of 4 updates, PPO of 3,
eight duels an update, suites of a few games): a control made by
`experiments/commands/thirteen-decks.sh` the same way, then the three arms
at once, each its copy and its PPO, then the measuring part and the
report. The two arms that hide identities hid 3,690 cards an update and
kept 60 cards as trained on after the copy and 62 after PPO; the control,
of format 15, was measured beside them. What the descriptions and the
hiding cost an update on the card is not known before the run.

## The run

Started on the CUDA machine on 2026-10-17 at 15:39, at `e6eac82e` with the
submodules at their pins, after the build and its one change were handed
over, from the repository root:

```sh
AT_ONCE=3 bash experiments/commands/card-descriptions.sh
```

then, once the arms are made:

```sh
PARTS=measure bash experiments/commands/card-descriptions.sh
```

The reckoning of thirteen and a half hours was short. Over their updates
180 to 229 the copies told the descriptions took 9.8 s an update and
`hidden`'s 6.3 s, the difference all in the pass (`docs/commands.md`,
the section on what is left to try for speed). The arms were made at 08:17
on 2026-10-18, in 16 hours and 38 minutes. In PPO the calls of the games
were answered from the CUDA graphs as `half`'s were, some 82% of them in
every arm, and no graph failed. The measuring part ran from 09:54 to 11:54
that morning, nothing else running there.

The web client cannot seat a checkpoint of format 16 until it looks the
descriptions up as the model does; the experiment does not need it.

## Results (read on 2026-10-18)

Measured on the CUDA machine from 09:54 to 11:54 on 2026-10-18 at
`e6eac82e`; `analysis/card-descriptions`, the report in `report.txt`
(`tools/card_descriptions_report.py`). The control's checkpoints were
measured again beside the arms'.

The readings, each by the arm's checkpoints after 1,680, 1,800 and 1,920
updates against the control's, game by game:

1. **The decks outside training: not told apart.** `both` against the
   control: +0.5 ± 0.5 over 3,744 games (35.1% against 34.6%).
2. **The decks of training kept: equivalent.** `both` against the control
   over the thirteen screens: −0.3 ± 0.2 over 21,632 games (60.7% against
   61.0%).
3. **Which part carries it: neither told apart.** `descriptions` against
   the control on the decks outside training: +1.0 ± 0.5 (35.6%);
   `hidden`: −1.5 ± 0.5 (33.1%).

Beside them, without a mark:

- **The copies part further than the networks after PPO.** On the decks
  outside training the control's copy wins 27.7%; `descriptions`' copy
  2.4 ± 0.7 points more (30.0%), `both`'s 0.6 ± 0.7 (28.3%), and
  `hidden`'s 3.8 ± 0.7 fewer (23.9%). PPO takes the descriptions' lead
  over the control from 2.4 points to 1.0, and the hidden identities'
  loss from 3.8 to 1.5.
- **Every network is still eleven to thirteen points under those decks'
  own pilots** (46.6% of the same games): `both` −11.5 ± 0.7,
  `descriptions` −11.0 ± 0.7, `hidden` −13.5 ± 0.7, the control
  −12.0 ± 0.7.
- **By how near their cards come, `both` against the control**: +0.8 ±
  0.6 on the nine decks least covered (Arcana, Watt, Pyramid, Ojama,
  Destiny HERO, Toon, Fortune Lady, Harpie, Six Samurai), +0.3 ± 0.9 on
  the nine most. Not apart either.
- **Deck by deck the networks are far apart now and then**, on 208 games
  a deck, each read by three checkpoints, some three points of standard
  error between two of them: `both` over the control on Pyramid (13.1% against 6.9%),
  Karakuri (41.2% against 33.2%) and X-Saber (52.6% against 45.7%),
  under it on Quickdraw Plant (30.9% against 37.7%) and Countdown (43.6%
  against 50.5%); `hidden` under it on Watt (5.9% against 13.3%). The
  share of the pilot's wins kept is in `report.txt`, with what each deck
  shares with the decks of training.
- **On the screens all four are alike**: 60.6% to 61.0%.
- **The runs' own figures** over their last 120 updates: `both` 13.7 s an
  update, `descriptions` 15.0 s, `hidden` 13.4 s, the control 14.3 s at
  the same stage of its own run; against the pilots in training 59.4%,
  59.7%, 58.4% and the control 60.0%.

What it says: a fixed vector of each card's text, added to its identity,
does not make a network of thirteen decks play the decks it never trained
on measurably better, with the identities of a quarter of the cards hidden
in training or without, and it costs nothing on the decks it trained on.
Under imitation the descriptions are worth a little on new decks, and
hiding identities without them costs; PPO evens both out. The networks are
as far under those decks' own pilots as before. Of what was tried so far
on decks outside training, more decks in training is the one that has
moved them: 9.2 points from one deck to thirteen, in the experiment on
more decks.

## What follows

Closed on 2026-10-18, the day it was read, with every reading taken: the
networks are equivalent, and the descriptions slow training for an advantage
of the copies that PPO cancels out, so they are probably not worth keeping.
The descriptions and the hidden identities are left out of the runs that
follow; format 16 stays in the trainer as an option, the table in `data/`.
Next, [Thirty-one decks](2026-10-18-thirty-one-decks.md), whose design and
readings were fixed the same day: `half` is its control too.
