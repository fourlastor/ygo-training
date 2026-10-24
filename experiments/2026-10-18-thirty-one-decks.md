# Thirty-one decks: more decks for the decks never trained on

**Status:** COMPLETED

**Proposed:** 2026-10-18

**Activated:** 2026-10-18

**Last updated:** 2026-10-19 (design and readings fixed before anything
was built; built and tried small; started on the CUDA machine on
2026-10-18; run, read and closed on 2026-10-19)

Proposed on 2026-10-18, when
[Card descriptions and hidden identities](2026-10-17-card-descriptions.md)
was closed. Of what has been tried on the decks outside training, training
on more decks is what has moved them: the network of thirteen decks played
them 9.2 points better than the network of Blackwing alone, and 3.3 better
than the one of eight
([One network for more decks](2026-10-14-more-decks.md), its follow-up
beyond its decks), where UPGO, every duel against itself, and the cards'
descriptions with their identities hidden were not told apart. This is the
next step of that on the thirty-one decks of the policies' pool, all the
pin holds (the policies' branch `ygo-agent-benchmark-work` holds 72, which
are not on their `main`). **The design and the readings below were fixed
on 2026-10-18, before anything was built**, starting with the thirty-one
decks.

## Research questions

1. Does a network of twenty-two decks play decks it never trained on better
   than one of thirteen?
2. Do a network of twenty-two decks and one of all thirty-one still play the
   thirteen as well?
3. Does a network of all thirty-one play the eighteen it now trains on as
   well as their pilots do, as the network of thirteen plays its own?

## Design (fixed on 2026-10-18)

Three networks, each made from scratch as `half` of
[Thirteen decks](2026-10-16-thirteen-decks.md) was: its decks' pilots
imitated (240 updates a deck, at 3e-4, every game seen once with its result
at four positions), then PPO from that copy at 1e-4, one pass over an
update's 520 duels, 260 against itself (any two of its decks, one in each
seat) and the others its decks against their pilots, to 1,920 updates;
seed 2012, the three at once through MPS. Each plays its own decks and
meets their pilots alone:

| Arm | Its decks | Its copy | Held out |
|---|---|---|---|
| `plus-a` | the thirteen and half A, 22 | 5,280 updates | half B |
| `plus-b` | the thirteen and half B, 22 | 5,280 updates | half A |
| `all` | the thirty-one | 7,440 updates | none |
| the control, `half`, already trained | the thirteen | 3,120 updates | all eighteen |

- **The halves** of the eighteen decks outside the thirteen: half A
  Arcana, Draconic Might, Harpie, Karakuri, Pyramid, Quickdraw Plant, Six
  Samurai, Tele-DAD and X-Saber; half B Countdown, Destiny HERO, Fortune
  Lady, Gravekeeper, Machina, Ojama, Toon, Verdict and Watt. Of every way
  of parting the eighteen into nine and nine, the one whose halves are most
  alike in what their pilots win of the games of the decks outside training
  (46.8% and 46.4%), in what `half` wins of them (34.7% and 34.5%) and in
  how near their cards come to the thirteen's (0.66 both), by the
  measurements of the experiment on card descriptions
  (`analysis/card-descriptions/report.json`). A deck of one half is never
  seen by the arm of the other, neither played nor met; between them the
  two arms hold out every one of the eighteen.
- **Fewer games a deck.** In an update a deck is the learner's in some 24
  of the 520 duels with twenty-two decks and 17 with thirty-one, against 40
  in `half`: the second reading asks what that costs the thirteen.
- **The control** costs no training: `half` has checkpoints after 1,680,
  1,800 and 1,920 updates, those the readings take, and its checkpoints are
  measured again beside the arms'.

## Readings (fixed on 2026-10-18, before anything was built)

Each by an arm's checkpoints after 1,680, 1,800 and 1,920 PPO updates,
game by game on the same games, a game read by the share of the three
checkpoints that win it:

1. *More decks help the decks never trained on.* `plus-a` on the decks of
   half B and `plus-b` on those of half A, the eighteen together, each
   against the pilots of the thirteen decks, 16 games a pairing
   (`eval_suites/ygo-policies-unseen-played.json`, 3,744 games), against
   `half` on the same games: **better** at 2 points and two standard
   errors more, **worse** at as many fewer, otherwise not told apart.
2. *The thirteen decks kept.* Each of `plus-a`, `plus-b` and `all` against
   `half` over the thirteen screens (21,632 games): **equivalent** when the
   gap less two standard errors is above −2 points, **worse** when the gap
   plus two standard errors is below −2 points, otherwise not decided.
3. *A network of all the decks on the eighteen it now trains on.* `all` on
   the eighteen decks against those decks' own pilots, the same 3,744
   games: **over** at 2 points and two standard errors more, **under** at
   as many fewer, otherwise not told apart.

Told beside them without a mark: each network of twenty-two decks on the
nine new decks it trained on, against their pilots and against the arm
that held them out; the copies on the decks outside the thirteen; deck by
deck; and the runs' own figures.

## The build (2026-10-18)

The trainer is as it was; the experiment is a script and a report:
`experiments/commands/thirty-one-decks.sh` and
`tools/thirty_one_decks_report.py` (`tests/test_thirty_one_decks_report.py`).
Tried small on the development machine's processor on 2026-10-18, the
script as written but small (two decks of training and one deck to each
half, copies of 2 updates a deck, PPO of 3, eight duels an update, suites
of a few games, a small control made by
`experiments/commands/thirteen-decks.sh` the same way): the three arms at
once, each its copy and its PPO, each playing and meeting its own decks
alone, then the measuring part and the report.

## The run

Started on the CUDA machine on 2026-10-18 at 12:24, at `352ffe44` with the
submodules at their pins, nothing else running there, from the repository
root:

```sh
AT_ONCE=3 bash experiments/commands/thirty-one-decks.sh
```

then, once the arms are made:

```sh
PARTS=measure bash experiments/commands/thirty-one-decks.sh
```

By reckoning from the last two runs' pace about a day for the first (the
copies some ten hours for `plus-a` and `plus-b` and thirteen for `all`,
their PPO some seven each), and two to three hours for the second;
`analysis/thirty-one-decks` comes back, the readings in `report.txt` there.
The copies' first updates took 7.2 to 8.0 s, more than the reckoning's.

The arms were made at 12:00 on 2026-10-19, in 23 hours and 36 minutes.
The copies took 7.6 to 8.0 s an update (`plus-a` 11.1 hours, `plus-b`
11.8, `all` 16.5), their PPO 13.4 to 16.1 s an update over the three at
once (8.0, 8.6 and 7.1 hours; `all` ran its last three hours alone). In
PPO 86% to 92% of the calls of the games were answered from the CUDA
graphs, no graph failed, and two of the 998,400 duels of `all` and none of
the others' were stopped at the limit. The measuring part was started in
the same session and ran from 15:46 to 17:39 that day, nothing else
running there.

## Results (read on 2026-10-19)

Measured on the CUDA machine from 15:46 to 17:39 on 2026-10-19 at
`352ffe44`; `analysis/thirty-one-decks`, the report in `report.txt`
(`tools/thirty_one_decks_report.py`). The control's checkpoints were
measured again beside the arms'.

The readings, each by the arm's checkpoints after 1,680, 1,800 and 1,920
updates against the control's, game by game:

1. **More decks for the decks never trained on: not told apart.** The arm
   that held each deck out against `half`, the eighteen together:
   −0.0 ± 0.6 over 3,744 games (34.6% against 34.6%); half A by `plus-b`
   +0.1 ± 0.8, half B by `plus-a` −0.2 ± 0.8.
2. **The thirteen decks kept: `plus-a` equivalent, `plus-b` not decided,
   `all` worse.** Over the thirteen screens, against `half`'s 61.0%, each
   over 21,632 games: `plus-a` −1.5 ± 0.2 (59.5%), `plus-b` −2.4 ± 0.2
   (58.6%; with two standard errors added its gap is −1.9, just short of
   the mark for worse), `all` −3.4 ± 0.2 (57.6%).
3. **A network of all the decks on the eighteen it now trains on: over
   their pilots.** `all` against those decks' own pilots: +5.7 ± 0.7 over
   3,744 games (52.3% against 46.6%).

Beside them, without a mark:

- **Training on a deck is what plays it.** On the nine decks of its half
  an arm of twenty-two decks wins 53.4%: 18.8 ± 0.6 points over the arm
  that held them out, and 6.8 ± 0.7 over their own pilots. Each of the
  eighteen is played better by the arm that trained on it than by the one
  that held it out, from 4.0 points on Verdict to 38.1 on Six Samurai, and
  better than by its pilot on all but Countdown (70.5% against 84.6%).
  `all`, with fewer games of each deck, is 1.1 ± 0.5 under the arms of
  twenty-two on them (52.3%), and over the pilot on fourteen of the
  eighteen, even with it on Toon, Verdict and Watt (within a point), under
  it on Countdown (75.5% against 84.6%).
- **A deck held out is played as `half` plays it**, twelve points under its
  own pilot (−12.0 ± 0.7 both). The copies of more decks come a little
  further on decks they never saw (29.4%, `half`'s copy 27.7%: +1.7 ±
  0.7), and PPO evens it out, as it did the descriptions'.
- **Deck by deck** the held-out arm and `half` are far apart now and then,
  on 208 games a deck read by three checkpoints, some three points of
  standard error between two networks: from −10.7 on Countdown (39.7%
  against 50.5%) to +11.7 on Fortune Lady (28.0% against 16.3%), the rest
  in `report.txt`.
- **On the thirteen screens the cost is spread.** `all` is under `half` on
  every one of the thirteen, most on Gladiator (−6.9), Heroes (−5.4),
  Rock-block (−5.2) and Gishki (−4.6), least on Burn (−0.4) and
  Morphtronic (−0.9). It is at the same number of updates, so with fewer
  games of each deck (17 in an update with thirty-one decks, 24 with
  twenty-two, 40 in `half`); whether more updates give it back is not
  measured.
- **The runs' own figures** over their last 120 updates: `plus-a` 14.6 s
  an update, `plus-b` 13.7 s, `all` 11.6 s (alone by then), `half` 14.3 s
  at the same stage of its run; against their own decks' pilots in
  training 59.3%, 54.9%, 55.5% and `half` 60.0%, each against other
  pilots.

What it says: from thirteen decks on, more decks in training no longer
make a network play the decks it never trained on better. A network of
twenty-two decks plays the nine it held out exactly as the network of
thirteen does, twelve points under their pilots. What moves a deck is
training on it: 18.8 points over holding it out, and 6.8 over its pilot.
A network of all thirty-one plays the eighteen new decks 5.7 points over
their pilots, a point under the networks of twenty-two that trained on
them, and pays for it on the thirteen: 3.4 points under `half` there at the
same number of updates, under it on every one of them.

## What follows

Closed on 2026-10-19, the day it was read, with every reading taken and
with what follows as the next experiment.

Beside the readings there is the question of whether every deck added
makes a network take longer to train, as an update gives each deck fewer
duels. By `half`'s own course the cost on the thirteen is about what fewer
games predict: its second 1,920 updates, twice the games of each deck,
gained it 2.6 points, and at that rate 17 duels of a deck in an update in
place of 40 come to −3.3 (−3.4 measured), 24 to −2.0 (−1.5 and −2.4). A
rough check, by one point of one course: a network of more decks would
want as many more updates to give each deck as many games, about 4,600 for
thirty-one decks in place of 1,920, or fewer as the decks share what they
learn (in the experiment on more decks, the network of thirteen played
Blackwing some five points under the network of Blackwing alone on a
thirteenth of its games). Not measured: continuing `all` to some 4,600
updates would tell.

Another question came first: whether a network learns decks or cards, by a
variant of the Blackwing list that no pilot was written for,
[Darker Than Black](2026-10-19-darker-than-black.md).
