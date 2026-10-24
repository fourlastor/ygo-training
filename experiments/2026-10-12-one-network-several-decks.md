# One network, several decks

**Status:** COMPLETED

**Proposed:** 2026-10-10

**Activated:** 2026-10-12

**Completed:** 2026-10-12

**Last updated:** 2026-10-12 (run and read: one network holds the two decks, and in the follow-up a third)

Decided on 2026-10-10, beside
[PPO with a head that predicts](2026-10-09-ppo-with-a-head-that-predicts.md),
and prepared that day. Taken up on 2026-10-12, when
[Self-play, and the named plays](2026-10-11-self-play.md) was
closed: a run of two hours, to be read the same day, before the night's
run that repeats self-play.

## Research question

Does a network that imitates the pilots of two decks copy each of them as
well as a network that imitates one?

## Why

- **Everything measured so far is one deck.** The checkpoints play
  Blackwing, and what they play against a network is the Blackwing mirror.
  The web page a checkpoint is seated on has 31 decks.
- **Either way of making a network for several decks needs this first.**
  Such a network can be taught from the pilots of several decks or from a
  specialist of each
  ([the copy of a specialist](2026-10-09-ppo-with-a-head-that-predicts.md#conclusion-and-what-follows),
  on hold). Both ask one network to hold several decks, so that is tested
  with what is cheapest and best known: the pilots, imitated.
- **The second deck is Lightsworn**, chosen on 2026-10-10: one deck more
  for now, a deck played in a way of its own, and the one that tends to
  run out of cards.

## Design (fixed on 2026-10-10)

- **A network of its own for each deck** (`alone-blackwing`,
  `alone-lightsworn`), from scratch: the deck's pilot imitated as
  `once-few` imitated Blackwing's, one pass over an update's games, a
  game's result taught at four of its positions with a weight of 0.5, 200
  updates of 520 games against the thirteen opponents of training, seed
  2012.
- **One network of both** (`together`), from the same first network and
  with the same lesson: each deck in an equal share of an update's games,
  for 400 updates, so that it has had as many games of each deck as that
  deck's own network. An update holds at most 1,024 games, which is why
  the games of the second deck are more updates and not larger ones. A
  game of an imitating update is its own deck's pilot's, and is the game
  a single-deck learner has of that deal.
- **A deck is read on its screen**: 1,664 games against the pilots of the
  thirteen opponents, by a network's first choice. The held-out logged
  games exist for Blackwing alone.

## Readings (fixed on 2026-10-10)

Agreed before any run (the marks as proposed, and one deck more for now:
Lightsworn), and written then in the record of the experiment that was
active. A deck is read on its screen, game by game, by the share of a
network's checkpoints at which it wins a game: the deck's own after 160
and 200 updates, the network of both after 320 and 400.

1. *The network of both plays a deck no worse than the deck's own
   network*, for each of the two decks: it is **worse** when it wins at
   least 2 points fewer of the screen's games and by more than two
   standard errors.
2. *One network holds the two decks*: it is worse on neither.

How often each network's first choice is its pilot's is told beside them,
without a mark.

What the screen can tell: with 1,664 games a difference of two rates has
a standard error of about 1.7 points taken apart, and less game by game,
as the readings take it.

## The run

From the repository root on the CUDA machine, with nothing else running
there:

```sh
bash experiments/commands/several-decks.sh
```

About two hours, by what an imitating update takes there (7.2 s): 27
minutes for each deck's own network with its five evaluations, 53 for the
network of both, ten for the measurements. A run or a measurement that is
there is not made again, and an interrupted run goes on. `runs/several-decks`
holds the runs, and `analysis/several-decks/` is what is copied back.

## The tooling

- **The trainer has a learner of several decks** since 2026-10-10:
  `train --learner blackwing,lightsworn`, through the pool, each deck
  against every opponent in equal shares of an update's games. A run of
  one deck plays the games it played. With no suite named a run is
  evaluated on the screens of all its decks, each case on the deals of
  its own screen, so a deck's figures are those of `evaluate-suite` on
  that screen.
- `experiments/commands/several-decks.sh` makes the three runs and the
  measurements, and `tools/several_decks_report.py` takes the readings.
  It also says whether `alone-blackwing` was taught on the games of the
  copy made before (`once-few`, by the figures of its updates in
  `analysis/learner-follow-up/once-few.metrics.jsonl` on that machine),
  which it should be: the lesson and the deals are the same.
- Tried on the development machine at a small size on 2026-10-10, and
  again on 2026-10-12 at the revision the run is made at.

## The run as it came out (2026-10-12)

Made on the CUDA machine in an hour and 55 minutes and read from
`analysis/several-decks/` as it was copied back. The figures of the
readings were worked out a second time from each game's result, apart
from the report, and are the same.

- **The networks are what they were meant to be.** All three start from
  the same first network. The network of both has had 104,000 games of
  each deck in its 400 updates, as each deck's own has had of its deck in
  200. Blackwing's own is taught on the games of the copy made before
  (`once-few`), update for update, and wins of the screen what that copy
  won at each of their five evaluations.

A deck after so many games of it: how often a network's first choice is
its pilot's on games new to it, and the games of the deck's screen it
wins. Blackwing's pilot wins 60.3% of its screen and Lightsworn's 64.4%.

| Games of the deck | Blackwing: its own, as its pilot | the network of both | its own, wins | the network of both | Lightsworn: its own, as its pilot | the network of both | its own, wins | the network of both |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 20,800 | 90.6% | 91.5% | 48.0% | 50.4% | 91.3% | 92.3% | 56.6% | 58.9% |
| 41,600 | 93.0% | 93.7% | 52.1% | 53.4% | 93.0% | 94.1% | 60.5% | 62.9% |
| 62,400 | 94.0% | 94.4% | 53.8% | 55.9% | 94.0% | 95.0% | 61.2% | 61.8% |
| 83,200 | 94.6% | 94.8% | 56.1% | 57.5% | 94.7% | 95.5% | 61.4% | 62.4% |
| 104,000 | 94.9% | 95.1% | 57.0% | 57.1% | 95.3% | 95.8% | 61.3% | 63.5% |

Game by game of a deck's screen, by the deck's own network after 160 and
200 updates and the network of both after 320 and 400:

| | Blackwing | Lightsworn |
| --- | ---: | ---: |
| The network of both over the deck's own | +0.8 ± 0.7 | +1.6 ± 0.7 |
| The deck's own over its pilot | -3.7 ± 0.8 | -3.0 ± 0.9 |
| The network of both over the pilot | -3.0 ± 0.8 | -1.4 ± 0.9 |

- **The first reading: the network of both plays each deck no worse than
  the deck's own network.** It is 0.8 ± 0.7 over Blackwing's and 1.6 ±
  0.7 over Lightsworn's, where 2 points and two standard errors under
  would have said worse.
- **The second reading: one network holds the two decks.** It is worse on
  neither.
- **Beside them: it is a little ahead on both**, at every count of a
  deck's games, and gives its pilot's answer a little more often
  (95.1% for 94.9%, 95.8% for 95.3%). It has had twice the games in all,
  of two decks: what one deck's games teach is of some use to the other.

**Time.** A network of one deck took 25 and 27 minutes (6.6 and 7.4 s an
update), the network of both 52 (7.0 s an update), the measurements ten.

## Conclusion

**What it answered.** One network that imitates the pilots of Blackwing
and Lightsworn copies each of them as well as a network that imitates one,
at the same number of games of the deck. A second deck costs the first
nothing, and takes as many games as it would alone.

**What it does not say.**

- *More than two decks.* The page has 31, and a third or a tenth may
  crowd the others where a second does not. (A third does not: the
  [follow-up](#follow-up-a-third-deck-readings-fixed-on-2026-10-12) of the
  same evening.)
- *Past the copy.* These networks imitate, and each is still three
  points under its pilot after 104,000 games. Whether a network of
  several decks gains past its pilots by PPO, or by playing itself, as
  the network of one deck did, is not tried: the trainer plays several
  decks past the imitating updates, and that is all that is known of it.
- *Other measures than the screen.* A deck is read on its 1,664 games
  against the thirteen pilots, by its first choice. The held-out logged
  games exist for Blackwing alone.
- *One run of each network.*

**Closed on 2026-10-12** with both readings taken, for the night's run,
chosen that day before this one was read:
[Self-play, further](2026-10-12-self-play-further.md). What this
experiment leaves to build on is open: a network of more decks taught
the same way, or PPO and self-play on from the network of both.

## Follow-up: a third deck (readings fixed on 2026-10-12)

Asked for the same evening, as a run short enough to make before the
night's ([Self-play, further](2026-10-12-self-play-further.md), which
stays the active experiment), and fixed before any run: the deck and the
mark as proposed. It takes up the first thing this experiment
does not say.

- **The deck is Gishki**, the other candidate of 2026-10-10: a deck of
  Ritual Monsters, whose pilot wins 40.8% of its screen, where the other
  two pilots win over 60%.
- **The networks.** Gishki's own, made as the other two were
  (`alone-gishki`, 200 updates of 520 games). One network of the three
  decks from the same first network (`together-three`), each deck in an
  equal share of an update's games, for 600 updates: as many games of
  each deck as its own network has had. Blackwing's and Lightsworn's own
  networks are the ones made above, read as they are.
- **The readings**, by a deck's own network after 160 and 200 updates
  and the network of the three after 480 and 600, game by game of the
  deck's screen:
  1. *The network of the three plays a deck no worse than the deck's own
     network*, for each of the three decks: it is **worse** when it wins
     at least 2 points fewer of the screen's games and by more than two
     standard errors.
  2. *One network holds the three decks*: it is worse on none.

  The network of two decks beside it, and how often each network's first
  choice is its pilot's, are told without a mark.

```sh
DECKS="blackwing lightsworn gishki" JOINT=together-three OUT=analysis/three-decks bash experiments/commands/several-decks.sh
```

About two hours: 25 minutes for Gishki's own network, 80 for the network
of the three, and a quarter of an hour of measurements, all of them made
anew in `analysis/three-decks/`. The script refuses to take the network
of two decks on as one of three. Tried on the development machine at a
small size that evening.

### The follow-up as it came out (2026-10-12)

Made on the CUDA machine in two hours, before the night's run, and read
from `analysis/three-decks/` as it was copied back. The figures of the
readings were worked out a second time from each game's result, apart
from the report, and are the same.

- **The networks are what they were meant to be.** The network of the
  three has had 103,954 games of Blackwing, 103,975 of Lightsworn and
  104,071 of Gishki in its 600 updates, where each deck's own has had
  104,000. Blackwing's and Lightsworn's own are the networks read above,
  and are read the same game for game.

Game by game of a deck's screen, by the deck's own network after 160 and
200 updates and the network of the three after 480 and 600:

| | Blackwing | Lightsworn | Gishki |
| --- | ---: | ---: | ---: |
| The pilot wins | 60.3% | 64.4% | 40.8% |
| The deck's own network | 56.6% | 61.4% | 39.3% |
| The network of the three | 56.9% | 62.3% | 39.7% |
| The network of the three over the deck's own | +0.3 ± 0.7 | +1.0 ± 0.8 | +0.4 ± 0.7 |
| ... over the pilot | -3.4 ± 0.8 | -2.0 ± 0.9 | -1.1 ± 0.9 |
| ... over the network of two | -0.4 ± 0.7 | -0.6 ± 0.7 | |
| First choice is its pilot's: the deck's own, the network of the three | 94.9%, 95.2% | 95.3%, 95.8% | 94.6%, 94.7% |

- **The first reading: the network of the three plays each deck no worse
  than the deck's own network.** It is over the deck's own on each of the
  three, by less than a standard error and a half.
- **The second reading: one network holds the three decks.** It is worse
  on none.
- **Beside them.** On Blackwing and Lightsworn the network of the three
  is half a point under the network of two, which is not told from
  nothing. What the network of two had over a deck's own (0.8 and 1.6
  points) is 0.3 and 1.0 here: a third deck has not added to it, and has
  taken nothing a screen shows.

**What it says.** A third deck costs the other two nothing either, and
Gishki, whose pilot wins far less than the other two, is copied as well
as they are. Each deck still takes the games it takes alone: the network
of the three had three times the updates.

**Time.** Gishki's own network took 26 minutes (7.0 s an update), the
network of the three 79 (7.2 s an update, with five evaluations of 89 s
on the three screens), the measurements a quarter of an hour.

## What is kept

The runs are on the CUDA machine under `runs/several-decks`
(`alone-blackwing`, `alone-lightsworn`, `alone-gishki`, `together` for
two decks and `together-three` for three, each with a checkpoint every 40
updates of a deck's games), and the measurements in
`analysis/several-decks/` and `analysis/three-decks/`.
