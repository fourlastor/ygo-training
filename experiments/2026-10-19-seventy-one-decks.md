# Seventy-one decks: every deck of the pool

**Status:** COMPLETED

**Proposed:** 2026-10-19

**Activated:** 2026-10-19

**Last updated:** 2026-10-22 (the run ended and was read, and closed that
day: the sections on its results and on what follows; the mirror on
decks-01's curve measured on the CUDA machine and the head to head with
`half` added); 2026-10-21 (the run stopped at its 1,839th update of PPO on
a card's script that failed, and goes on at new pins whose duels are the
old ones: the section on the run that stopped and goes on); 2026-10-20 (a
measurement added beside the readings: where `all71` stands in the
Blackwing mirror on the curve of the network of Blackwing alone; on
2026-10-19 the design and readings fixed before anything was built, built
and tried small that evening, and started on the CUDA machine)

Proposed on 2026-10-19, when
[Darker Than Black](2026-10-19-darker-than-black.md) had shown that a
network plays the cards it sees, the ones it never trained on too, and
that the decks of the policies hold most of the cards people played: of
432 tournament decklists of 2010 and 2011, 92.8% of the slots are cards of
the 71 decks the pin holds since that evening (`4b308c40`), 60.5% of the
thirteen of training. This is a network of all of them, given each deck as
many games as `half` of [Thirteen decks](2026-10-16-thirteen-decks.md)
gave each of its thirteen. **The design and the readings below were fixed
on 2026-10-19, before anything was built**, as were the games' own
estimates for its PPO, the closing of the experiment before it and a
checkpoint every 120 updates of PPO.

## Research questions

1. Given as many games of each deck as `half` had, does a network of all 71
   decks play the thirteen as well as `half` does?
2. Given as many games of each deck as `all` of
   [Thirty-one decks](2026-10-18-thirty-one-decks.md) had, does it play the
   eighteen other decks of those thirty-one as well as `all` does?
3. Does it play the forty decks new to the pool better than their own
   pilots?

## Design (fixed on 2026-10-19)

One network, `all71`, made from scratch as `half` and `all` were: its
decks' pilots imitated (240 updates a deck at 3e-4, every game seen once
with its result at four positions: 17,040 updates), then PPO from that
copy at 1e-4, one pass over an update's 520 duels, 260 against itself (any
two of its decks, one in each seat) and the others its decks against their
pilots; seed 2012. Its PPO takes the games' own estimates
(`--rollout-estimates kept`), which on the CUDA machine are the recomputed
ones to their last digits (`docs/commands.md`, its section on what is left
to try for speed).

- **To 10,560 updates of PPO**, which give each deck about as many games as
  `half`'s 1,920 gave each of its thirteen (some 7.3 of an update's 520
  duels a deck, against 40); a checkpoint every 120.
- **Read at three points**, each by the checkpoints of its last three steps
  of 120, game by game as before:
  - **R1**, after 1,920 updates: as many updates as `half` and `all`;
  - **R2**, after 4,440: about as many games of each deck as `all` had;
  - **R3**, after 10,560: about as many as `half` had.
- **The controls** cost no training: `half` and `all` by their checkpoints
  after 1,680, 1,800 and 1,920 updates, measured again beside it.
- **The forty new decks** are measured as the eighteen of thirty-one decks
  were: each against the pilots of the thirteen decks, 16 games a pairing
  (`eval_suites/ygo-policies-forty-played.json`, 8,320 games).

## Readings (fixed on 2026-10-19, before anything was built)

1. *The thirteen kept.* `all71` at R3 against `half` over the thirteen
   screens (21,632 games): **equivalent** when the gap less two standard
   errors is above −2 points, **worse** when the gap plus two standard
   errors is below −2 points, otherwise not decided.
2. *The eighteen kept.* `all71` at R2 against `all` on the eighteen decks
   against the thirteen decks' pilots
   (`eval_suites/ygo-policies-unseen-played.json`, 3,744 games), by the
   rule of 1.
3. *The new decks.* `all71` at R3 with the forty new decks against those
   decks' own pilots on the same 8,320 games: **over** at 2 points and two
   standard errors more, **under** at as many fewer, otherwise not told
   apart.

Told beside them without a mark: the thirteen screens at R1 and R2 (how
many updates it takes to come back to `half`), the eighteen at R1 and R3,
the forty at R1 and R2 and against `all`, which never trained on them;
deck by deck; the copy at its end; and the run's own figures.

Not in it: an arm holding out some of the new decks, which would tell
whether a deck never trained on, most of its cards covered, is played
nearly as well as one trained on (about two days more), left for later.

## The build (2026-10-19)

The trainer is as it was; the experiment is a suite, a script and a
report: `eval_suites/ygo-policies-forty-played.json`,
`experiments/commands/seventy-one-decks.sh` and
`tools/seventy_one_decks_report.py` (`tests/test_seventy_one_decks_report.py`).
The copy keeps a checkpoint every 2,400 updates and its last, which alone
is read; PPO one every 120 (88 of them, some 7.5 GB). Tried small on the
development machine's processor that evening, the script as written but
small: two small controls made with it (two decks and three), then a
network of four decks (two of the thirteen, one of the eighteen, one of
the forty) to 9 updates of PPO read at 3, 6 and 9, its measuring part
queued after it in the same command (`PARTS="arm measure"`), and the
report.

## The run

Started on the CUDA machine on 2026-10-19 at 20:28, at `0edcefcd` with the
submodules at their pins, nothing else running there, from the repository
root:

```sh
PARTS="arm measure" bash experiments/commands/seventy-one-decks.sh
```

The copy's first 35 updates took 4.7 s each, more than the 3.95 of
`half`'s copy: about 22 hours for the copy, then by reckoning 31 to 34 for
PPO and two for the measuring part, which follows in the same command;
`analysis/seventy-one-decks` comes back, the readings in `report.txt`
there.

### The run stopped, and goes on (2026-10-21)

It stopped on 2026-10-20 at 23:41, in its 1,839th update of PPO (version
18,879), `model-18878.pt` its last checkpoint: before R1's last checkpoint,
18,959 (18,719 and 18,839 are kept). A card's script had failed in one duel
of the update, and the trainer stopped there: Batteryman Fuel Cell, of the
Full Charge deck, new to the pool, in the update's duel 495, Batteryman
against Garden with the network in both seats (its deal seed 47410671, the
update's rollout seed 47410176). Its cost Tributed Batteryman Charger, which
the opponent's Mark of the Rose was equipped to, its only card on the
field; the Spell was destroyed, and the effect resolved with no target,
which the script does not check (`official/c74730899.lua`, line 57). The
first such failure in about 957,000 duels of the run's PPO and 8.9 million
of its copy.

Played again on the development machine, that duel alone failed as on the
card: an update's games follow from its rollout seed, and the network's
answers from a seed of each decision. What was done (all three, and a
failed duel logged with its seeds so that it can be played again):

- The policies play that script mended (`9a3e4f1`): the effect does nothing
  where it failed, which is what EDOPro does after logging the error, and
  no duel that ran to its end is played otherwise.
- A duel the engine stops, a card's script failing in it or its arena
  filling, ends as one the limit stops, and the run keeps it in
  `failed-duels.jsonl` with its seeds, its record and the checkpoint it
  was played with (`docs/commands.md`, its section on duels the engine
  stops).
- The run goes on at the new pins, told so (`--new-pins`, kept in
  `run.json` as `pin_moves`): the mend's, and those of the field written
  straight into its buffer of 2026-10-20 (`docs/commands.md`, its section
  on what is left to try for speed), whose games are the old ones, message
  for message.

On the CUDA machine, from the repository root, nothing else running there:

```sh
NEW_PINS=1 PARTS="arm measure" bash experiments/commands/seventy-one-decks.sh
```

## Added beside the readings (2026-10-20)

Where `all71` stands in the Blackwing mirror, head to head, on the curve
of the network of Blackwing alone, which the readings, every one against
pilots, do not tell. Measured on 2026-10-20, `decks-01` of
[more decks](2026-10-14-more-decks.md) is ahead of the network of
thirteen's last checkpoint there within its first 120 updates of PPO and
level at 69% to 73% from 720. By the end `all71` will have played about
1,100 games of this mirror and the network of thirteen played 5,900: what
`decks-01` played in its first 4 and 21 updates. A way to play those
checkpoints with today's code was to be built, for use when the experiment
is ready. Once the run has ended, on the development machine, `all71` by
its last three checkpoints at each of R1, R2 and R3 against `decks-01` at
its copy's end and after each 120 updates of its PPO, folded for the
mirror (`experiments/commands/blackwing-mirror-curve.sh`, about 75 minutes
a point; `docs/commands.md`, its section on a checkpoint of format 14
played by today's code), told beside the readings without a mark.

## Results (read on 2026-10-22)

Its PPO ended on the CUDA machine on 2026-10-22 at 02:00, and the measuring
part ran from 02:01 to 05:23, at `15bbaa86` with the submodules at their
pins; `analysis/seventy-one-decks`, the report in `report.txt`
(`tools/seventy_one_decks_report.py`). The controls' checkpoints were
measured again beside it. Once the run went on, no duel failed: none of its
updates counts one, and no measurement does.

The readings, each by the checkpoints of the point's last three steps of
120, game by game:

1. **The thirteen kept: not decided.** `all71` at R3 against `half` over the
   thirteen screens: −2.3 ± 0.2 over 21,632 games (58.8% against 61.0%).
   With two standard errors added its gap is −1.8, short of the mark for
   worse; with two taken away −2.7, short of equivalent.
2. **The eighteen kept: equivalent.** `all71` at R2 against `all` on the
   eighteen decks against the thirteen decks' pilots: +0.4 ± 0.5 over 3,744
   games (52.7% against 52.3%).
3. **The new decks: over their pilots.** `all71` at R3 with the forty new
   decks against those decks' own pilots: +16.0 ± 0.5 over 8,320 games
   (61.9% against 45.9%).

Beside them, without a mark:

- **On the thirteen along the run**: under `half` by 6.2 ± 0.2 at R1 (54.8%),
  3.5 at R2 (57.5%) and 2.3 at R3: 2.7 points from R1 to R2 in 2,520
  updates, 1.3 from R2 to R3 in 6,120. At R2, with about as many games of
  each deck as `all` had, it is where `all` is (57.5% against 57.6%). At R3
  it is under `half` on twelve of the screens, most on Gladiator (−5.0),
  Crystal (−3.7) and Infernity (−3.1), level on Burn (+0.2); and over `all`
  on ten, level with it on Blackwing.
- **On the eighteen**: −1.0 ± 0.5 against `all` at R1 (51.4%), +0.4 at R2 and
  +2.1 ± 0.5 at R3 (54.4%), there +7.8 ± 0.7 over their own pilots, where
  `all` was +5.7; `half`, which never trained on them, 34.6%.
- **On the forty**: over their pilots by 10.5 ± 0.5 at R1 (56.4%), 13.3 at R2
  (59.2%) and 16.0 at R3. `all`, which never trained on them, is 8.0 ± 0.5
  under those pilots (37.9%), 24.0 ± 0.5 under `all71` at R3. The copy at its
  end plays the forty about as their pilots do (45.2%: −0.7 ± 0.4) and the
  eighteen a little under them (44.7% against 46.6%).
- **Deck by deck**, on 208 games a deck, some three and a half points of
  standard error for one network: at R3 `all71` is over its own pilot on 56
  of the 58 decks, on every one of the forty, from 1.1 points on Destiny
  Board and 1.9 on Exodia to 38.7 on Cyber (57.5% against 18.8%). It is
  under its pilot on Toon (20.0% against 20.7%) and on Countdown by 20.2
  points (64.4% against 84.6%), the one deck PPO took down: 78.4% at the
  copy's end, 79.5% at R1, 70.5% at R2. On Destiny Board and Exodia, which
  also win by a card's own condition, PPO left it where its copy was (48.6%
  to 48.2%, 75.0% to 74.5%).
- **The run's own figures** over its last updates: 10.8 s an update and
  172,687 duels an hour (`half` 14.3 s and 130,566, `all` 11.6 s and
  161,816); against its decks' pilots in training 62.2% over its last 1,000
  updates, from 50.4% over its first 120.

What it says: a network of all 71 decks plays the forty decks new to the
pool 16 points over their own pilots, each of them over its pilot, and the
eighteen of thirty-one decks as `all` does with as many games of each, and
better with more. With as many games of each deck as `all` had, it plays
the thirteen as `all` does too; with as many as `half` had, it is 2.3
points under `half` there, at the mark, and came 1.3 points closer over its
last 6,120 updates. Countdown is the one deck PPO plays worse than the
copy did, and Destiny Board and Exodia two it adds nothing to.

### The Blackwing mirror on decks-01's curve (measured on 2026-10-22)

As the section added beside the readings says, but on the CUDA machine,
from 08:06 to 09:39, the three points at once: moved there from the
development machine for speed. The first two measurements, made on both
machines before the move, are the same game for game, 2,048 games each.
`analysis/blackwing-mirror-curve/` `all71-1920`, `all71-4440` and
`all71-10560` (`tools/blackwing_mirror_curve.py`): each point by its last
three checkpoints against `decks-01` at its copy's end and after each 120
updates of its PPO, every deal of the mirror from either seat.

- **`all71` stands where the network of thirteen stood**: at each point
  between `decks-01` at its copy's end and after 120 updates of its PPO (0
  and 33,600 games of the mirror). At R3 its share is within about a point
  of `decks-13`'s against every checkpoint of `decks-01`: 69.4% against
  69.5% over the copy, 43.1% against 42.9% after 120 updates, 28.3% against
  29.1% after 1,920; on about 1,100 games of this mirror to `decks-13`'s
  5,900.
- **It climbs along its run**: against `decks-01` after 120 updates 35.0% at
  R1, 37.3% at R2 and 43.1% at R3; against its last checkpoint 23.7%, 24.5%
  and 28.3%.
- **What its training is worth there**: it would be level with `decks-01`
  somewhere between those two points of its curve, by a straight line
  between them after some 25,000 of `decks-01`'s games of the mirror, and
  likely fewer, the curve being steepest at its start. `decks-01` gains
  nothing more on it from about 720 updates (some 200,000 games), where it
  wins about 70%.

What it says: in the Blackwing mirror a network of all 71 decks plays as
the network of thirteen does, on a fifth of its games of the mirror, so
what it learns of other matchups carries there; and both are worth at most
some tens of thousands of a specialist's games of the mirror, which goes on
to beat them seven games in ten.

### Head to head with `half` (added on 2026-10-22)

Added while the mirror's curve ran, to test the hypothesis, stated before
it was measured, that `half` beats `all71` at R3 by a wide margin: `all71`
at R3 against `half` on every pairing of the thirteen decks, 16 deals each
played from either seat, the other's checkpoint held in the other seat and
their checkpoints matched (27,359 with 4,799, 27,479 with 4,919, 27,599
with 5,039), as `itself` was read against `half` in
[Thirteen decks](2026-10-16-thirteen-decks.md): 16,224 games, on the
development machine, in `analysis/seventy-one-decks/head-to-head`.

- **`all71` wins 48.9% ± 0.3** of them (49.2%, 49.0% and 48.6% by
  checkpoint): under `half` by 3.7 standard errors, but by the rule `itself`
  was read with (better at 52% and two standard errors over, worse at 48%
  and two under) not told apart.
- **Deck by deck** it plays a deck 2.2 points worse than `half` does on the
  same deals, about what it is under `half` against the pilots (−2.3 on the
  screens): most on Gladiator (−6.3 ± 1.6), Monarch (−5.0 ± 1.7), Gishki
  (−3.7) and Crystal (−3.3), level on Blackwing (+0.3), Lightsworn, Heroes
  and Burn.
- **The mirrors are no different**: 47.8% ± 1.3 of their 208 deals, 49.0% ±
  0.3 on the other pairings. Between two networks of many decks there is
  nothing of the 70% that `decks-01` wins against the network of thirteen
  decks in the Blackwing mirror.

What it says: `half` beats `all71` head to head by the same couple of
points a deck by which it plays the thirteen better against their pilots,
not by more.

## What follows

Closed on 2026-10-22, the day it was read, with every reading taken and
the copy of the specialist as the next experiment:
[Copy of the specialist](2026-10-22-copy-of-the-specialist.md).

Beside the readings, the measurements can be read as saying that the more
decks a network learns, the harder it is to improve on more specialised
networks, and that a specialist's strength in its own matchup takes some
200,000 duels of it, which would make ygo-agent's 100 million duels of 32
decks reasonable (528 matchups counting the mirrors, about 190,000 each if
spread evenly). The head to head and the mirror say where that cost lies.
Between two networks of many decks it is the couple of points a deck it is
against the pilots, mirrors included. In the Blackwing mirror a network of
all 71 decks is level with the network of thirteen on a fifth of its games
of it, at about what 25,000 of the specialist's games of the mirror are
worth; the specialist's edge from there, to about 70%, came with its own
games of that matchup. What a matchup's training carries to another is not
measured beyond that mirror.

Not measured: whether `all71`'s gap to `half` on the thirteen closes with
more updates (it came 1.3 points closer over its last 6,120); an arm holding
out some of the new decks; why PPO takes Countdown down.

## The duels it cut (found on 2026-10-22)

`all71` cut 87 duels at the cap of 4,096 engine answers over its 10,560
updates of PPO (about one in 63,000), none of them by a script's failure,
and more as it trained: 4 in its first 1,320 updates, 12 to 17 in each
1,320 from the 4,000th. Their decks, from each update's deal and its
per-deck figures (the seats of the duels against itself that did not end;
the deck whose win rate fits one duel fewer): 50 against itself, Psychic
in 47 of them (with Countdown 13, Verdict 8, Chaos 7, Exodia and Alien 5
each, Malefic 4); 37 against a pilot, Psychic the learner's deck in 14 of
the 21 that can be told. Most trained 1,600 to 2,400 of the network's
decisions, against some 116 in a duel that ends.

What it is, seen in duels of Psychic against those decks played again here
with `all71`'s checkpoint after 10,320 updates, its answers drawn: a loop
the engine allows. Destructotron's effect has no limit a turn, and Brain
Research Lab places a Psychic Counter in place of its 1,000 LP. Its target
answered with Fiendish Chain, Destructotron's effects are negated; the
engine still offers its effect, negated as it resolves (as Lyla's under
Skill Drain was), and the network takes it again and again for nothing,
790 counters on the Lab in one duel when the cap ended it. Each pass adds a
counter, so the position is never the one of the pass before, and the rule
that keeps an answer given at the same position earlier in the turn from
being offered again does not hold it. A duel cut is a loss to both seats
against itself, and at λ 0.95 that reaches back only a few transactions:
PPO does not unlearn it. The network's first choice, as it is measured,
never came to it: Psychic's 208 games of the forty at R1, R2 and R3 all
ended in a win or a loss.

Decided the same day: leave it, but for one exception, where the engine
says so: not offering a monster the activation of its effect while its
effects are negated, an activation that is usually of no use though the
rules allow it. The engine says so in a card's status (`STATUS_DISABLED`
in `QUERY_STATUS`), which the policies' duel does not ask for; asking for
it, marking a negated card in the projection and keeping that activation
out of what the seat is offered (`Trail::offered`, with EDOPro's client
alike) is to be built once the run of
[How cards work](2026-09-08-engine-transition-prediction.md) is read,
which reuses `all71`'s duels as its baseline. The two cut in the same
update, 17,074, by `all71` and by the critic's arm are of another kind:
Crystal against the Cloudian pilot, 339 and 373 decisions, a duel no
battle can end, which reaches the cap before a Deck runs out (found on
2026-10-23, [Duels in flight](2026-10-23-duels-in-flight.md), its section
on the duels it cut).

**The rule built** on 2026-10-23, the morning How cards work was closed:
the field is asked for with each card's status, the projection keeps its
`STATUS_DISABLED`, and `Trail::offered` keeps the activation of a negated
monster's effect out of what the seat is offered where it may decline it
(the policies' `c6d7cc4`, `native/ygo-model-input`'s `ef1ba77`;
`docs/commands.md`). The 600 duels of Psychic of 2026-10-22's look, the
same seeds and draws with `all71`'s network after 10,320 updates: 5 ran to
the cap at the code before and none with the rule, and Destructotron's
effect, taken 3,987 times before (790 in one duel), is taken 276 times (7
at most in one); the two duels played again then, 47 and 490, end in seven
turns. The pilots' games are as they were (8,320 of the forty), and the
network's first choices give another result in 21 and 4 of the 1,664 games
of the Rock-Block screen, whose deck plays Skill Drain, and the Lightsworn
screen, whose Lyla met it (`analysis/negated-rule/`).
