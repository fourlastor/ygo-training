# One network for more decks, half of its duels against itself

**Status:** COMPLETED

**Proposed:** 2026-10-13

**Activated:** 2026-10-14

**Completed:** 2026-10-15

**Last updated:** 2026-10-20 (the mirror along the training of
Blackwing alone, the networks of five and eight decks on it, and
Blackwing against four other decks head to head; on 2026-10-15 closed
the evening it was read, without the controls against the pilots alone:
every network of more decks is worse on Blackwing, the network of
thirteen has gained on its copy and is over the pilots; earlier that
day, where the networks grow unsure, the games that loop, and which
networks' calls go through the graphs)

Chosen on 2026-10-13: six arms in parallel (through MPS, with graphs) on
many decks in several sizes, in a long run, possibly with self-play since
that seemed to pay off; six sizes without controls for now, the controls
to be made after if the run proved worth it; 1,920 updates. The readings
and the order of the decks were fixed the same evening, before any run.
It was first meant for a rented machine, six arms at once; on 2026-10-14,
when [Self-play, a second seed](2026-10-13-self-play-second-seed.md) was
closed, it was started on the CUDA machine three arms at a time, since
the experiments there train serially and this run belongs there too.

## Research questions

1. What does a deck more cost the decks a network already plays, at the
   same number of games in all, under PPO with half of its duels against
   itself? Read on Blackwing, which every network plays.
2. Does a network of all thirteen decks of training gain on its own copy
   of their pilots under that training?
3. Does it end over the decks' own pilots?

## Why

- **One network copies two or three decks as well as a network of one**
  ([One network, several decks](2026-10-12-one-network-several-decks.md)),
  at as many games of each deck. That was imitation alone: PPO and
  self-play were never tried for more than one deck, and self-play could
  not be (a learner of several decks did not play itself until
  2026-10-13).
- **ygo-agent trains one network on many decks.** Whether ours can is
  the question before training for more than Blackwing.
- **Self-play pays in the mirror and against a learner that looks for a
  network's habits, and costs 2 to 3 points against the pilots**
  (the [two pairs of runs](2026-10-13-self-play-second-seed.md)).

## Design (fixed on 2026-10-13)

`experiments/commands/more-decks.sh`. Six networks, of the first 1, 2, 3,
5, 8 and 13 decks of Blackwing, Lightsworn, Gishki, Burn, Crystal Beast,
Dragunity, Gladiator, Heroes, Infernity, Monarch, Morphtronic, Rock-Block
and Spellcaster: the three decks one network was first made of, then the
others of training by their names. A smaller network's decks are among a
larger one's, and Blackwing is in every one. Each is made in two runs
under `runs/more-decks`:

1. **`decks-NN-copy`**: the pilots of its decks imitated from scratch, as
   the networks of the several-decks experiment were (every game seen
   once, its result at four of its positions), each deck in an equal
   share of an update's 520 games, for 240 updates a deck: 240 for one
   deck, 3,120 for thirteen. The several-decks experiment had 200 a deck;
   240 puts every checkpoint kept by any arm on one grid of 120 updates.
   The learning rate is 3e-4, as for the imitating runs before.
2. **`decks-NN`**: PPO on from the copy's last checkpoint for 1,920
   updates at 1e-4, one pass over an update's games, 260 of its 520 duels
   against itself and the others its decks against the pilots of the
   thirteen decks, every pairing as often. Against itself a network of
   several decks has one of its decks in each seat, every pair of them as
   often as any other (built for this on 2026-10-13); the network of one
   deck plays the mirror, as the arms of self-play did.

Seed 2012 for every arm. **Every arm has as many PPO updates, so a
network of N decks has an Nth of the games of each deck that the network
of Blackwing alone has**: what is read is the price of each deck at the
same games in all. No run is evaluated as it goes; a copy keeps a
checkpoint every 240 updates and the other run every 120, and they are
measured afterwards on the same machine. The trainer is that of
2026-10-13 (the packed LSTM, the calls from graphs with the duels in two
groups). Only the network of Blackwing alone has its calls answered from
graphs: every other plays Lightsworn, whose list of 50 cards is wider than
the 48 the graphs are filled out to, and nearly all of its calls are
answered as they are, the same model's answers more slowly (found on
2026-10-15; `docs/commands.md`, its item on a deck list of more than 48
cards missing the graphs).

No control without self-play for now: the arms of pilots alone at each
size are for after, if this is worth it.

## Readings (fixed on 2026-10-13)

Proposed as rules and fixed before any run. Each is taken by a network's
checkpoints after 1,680, 1,800 and 1,920 of its PPO updates:

1. *What more decks cost Blackwing.* Each network of more decks against
   the network of Blackwing alone, on the 1,536 held-out deals, deal by
   deal: it is **worse** when it wins at least 2 points and two standard
   errors fewer. Each of the five is read on its own.
2. *The network of all thirteen gains on its copy.* Its three checkpoints
   against its copy's last checkpoint, over the screens of its thirteen
   decks together (21,632 games), game by game: it has **gained** when it
   wins at least 2 points and two standard errors more.
3. *The network of all thirteen is over the pilots.* Its three
   checkpoints against each deck's own pilot on that deck's screen, the
   thirteen together, game by game: **over** at 2 points and two standard
   errors more, **under** at 2 points and two standard errors fewer.

Told beside them without a mark: every network's copy and last three
checkpoints on the screen of each of its decks; what each wins of the
duels it trained on against the pilots, deck by deck, update by update
(sampled games, a diagnostic); and each network's checkpoints through its
run. How fast each learns, which was asked for on 2026-10-14 as the effect
of the size of the pool of decks on the rate of learning, is told by every
checkpoint each network kept of its PPO on the held-out deals, by its
updates and by the seats of Blackwing it had played by then.

## The run

On the CUDA machine, from the repository root, started at 14:28 on
2026-10-14 at `ff062dec`:

```sh
AT_ONCE=3 bash experiments/commands/more-decks.sh
```

The arms go three at a time through MPS, the smallest first, each its
copy and then its PPO. By reckoning about 32 hours: the three smallest
done in about 14, the network of thirteen last. The first updates of the
three copies took about 8 s each. A run's output is in
`runs/more-decks/decks-NN.log`; an interrupted script, started again,
takes every run on from where it stands.

The pin of `native/ygo-policies` stays at `1b85b63` while this is open:
the held-out deals are logged games of that revision.

The arms were made at 19:07 on 2026-10-15, in 28 hours and 40 minutes.
The measuring part ran on the same machine at `e4dbbd4e` from 19:11 to
22:19, nothing else running there; `analysis/more-decks` (its report in
`report.txt`).

## Results (read on 2026-10-15)

The three readings, by each network's checkpoints after 1,680, 1,800 and
1,920 PPO updates:

1. **Every network of more decks is worse on Blackwing.** Of the 1,536
   held-out deals the network of Blackwing alone wins 78.3%; two decks
   76.2% (2.1 ± 0.7 fewer, just past the mark), three 73.7% (4.6 ± 0.8
   fewer), five 73.4% (5.0 ± 0.8), eight 73.7% (4.6 ± 0.8), thirteen
   73.5% (4.8 ± 0.8). Each is one run of one seed, and runs of the
   same training on other seeds have ended a point or two apart; the
   price of three decks and more is well past that. **It does not grow from three
   decks to thirteen.**
2. **The network of thirteen has gained on its copy**: 11.9 ± 0.3 points
   over its thirteen screens together (21,632 games).
3. **It is over the decks' own pilots**: 10.4 ± 0.3 points over the
   thirteen screens together, and over the pilot on each of them:

   | Deck | Pilot | Copy | After PPO |
   |---|---|---|---|
   | Blackwing | 60.3% | 58.0% | 72.7% |
   | Lightsworn | 64.4% | 65.4% | 74.9% |
   | Gishki | 40.8% | 40.6% | 57.2% |
   | Burn | 22.8% | 18.3% | 26.1% |
   | Crystal Beast | 37.6% | 36.8% | 43.3% |
   | Dragunity | 59.1% | 55.9% | 74.4% |
   | Gladiator | 58.1% | 55.0% | 74.4% |
   | Heroes | 65.2% | 65.3% | 78.2% |
   | Infernity | 56.7% | 56.7% | 68.8% |
   | Monarch | 69.7% | 66.9% | 78.6% |
   | Morphtronic | 40.1% | 39.4% | 43.7% |
   | Rock-Block | 61.1% | 60.5% | 70.9% |
   | Spellcaster | 22.8% | 20.1% | 30.3% |

   (each deck's screen, 1,664 games; after PPO, the last three
   checkpoints together.)

Beside the readings, without a mark:

- **The copies begin alike**: 58.2% to 60.1% of the held-out deals (the
  pilot 60.7%), as imitation of three decks cost Blackwing nothing in
  [One network, several decks](2026-10-12-one-network-several-decks.md).
  The price is paid under PPO.
- **By updates, the networks part after 600.** Every network gains 4 to 7
  points in its first 120 updates and is at 68% to 73% after 600; from
  1,200 on, the network of thirteen stays at 72% to 74% where Blackwing
  alone goes on to 75% to 79% (`report.txt`, every checkpoint each
  network kept).
- **By games of Blackwing, a network of more decks learns Blackwing
  faster.** After 94,000 seats of Blackwing the network of thirteen wins
  73.0% of the held-out deals, and the network of Blackwing alone, after
  as many, 65.7%; that one reached 72.9% after 468,000. The other decks'
  games teach Blackwing too, though not as much as as many of its own.
- **The screens tell what the held-out deals tell.** On Blackwing's
  screen, 77.2% alone, 74.8% with two decks, 74.7% three, 74.1% five,
  72.8% eight, 72.7% thirteen. The other decks pay for the decks added
  as well: Lightsworn 78.2% in the network of two and 74.9% in that of
  thirteen, Gishki 63.8% in that of three and 57.2% in that of
  thirteen.

## Beside the readings: where the networks grow unsure (2026-10-15)

Asked once the training figures showed the entropy of the networks of
more decks rising (0.11 to 0.16-0.18 by the network of five)
where the network of one deck's stayed at 0.11. Measured on the processor
here with `analysis/entropy/` (the model wrapped, its answers unchanged:
at each decision, the entropy of its answers in nats), on the copies and
the checkpoints after 600 and 1,920 PPO updates of the networks of one,
two and three decks:

- **On the same decisions every network grows less sure under PPO, with
  more decks or without.** On the 117,411 decisions of the held-out deals'
  recorded duels, put to each: one deck 0.100 at its copy and 0.146 after
  1,920 updates, two decks 0.093 and 0.117, three 0.096 and 0.145. It is
  at the Main Phase's commands (0.20 to 0.29) and in battle (0.12 to
  0.22), hardly at chain windows. The training figure of the network of
  one deck stays flat because its decisions shift toward chain windows,
  where it is sure (from 49% of them to 52-55%).
- **What more decks add comes from their other decks**, in their own games
  against the pilots of their screens: Lightsworn 0.09 to 0.29 (chain
  windows 0.02 to 0.18, choices of cards 0.27 to 0.69), Gishki 0.12 to 0.23
  (the Main Phase 0.08 to 0.26, and the choice among a card's effects 0.00
  to 0.99, a coin). Against itself, any of its decks in each seat, the
  network of three is at 0.13 and Lightsworn at 0.11.
- **Lightsworn's games against Rock-Block loop.** After PPO, 21 to 26 of
  its 128 games against Rock-Block on its screen reach the cap of 4,096
  decisions (the copies': none), at as many turns as before (12 to 13) and
  five times the decisions a game: about 100 more commands in the Main
  Phase, 90 choices of cards and 170 chain windows. A screen counts such a
  game a draw, and the training a loss (`OutcomeRewardModel.cutoff`, -1,
  for both seats where it plays itself): about one game an update in the
  runs of two and three decks and none in the run of one, each with
  thousands of the update's decisions, which raises the training figure of
  entropy as well.
- **What repeats is an effect Skill Drain negates.** The network of three
  after 1,920 updates played 60 games as Lightsworn against Rock-Block's
  pilot on the processor here (`analysis/loop/`): 10 ran to the cap. In
  nine it activates Lyla, Lightsworn Sorceress 593 to 675 times in one Main
  Phase, in the tenth Honest 754 times, with Rock-Block's Skill Drain
  face-up in every one; in nine of the ten nothing on the field, no life
  point and no pile changes from the first activation to the cap. Lyla
  turns to Defense Position and destroys a Spell or Trap only if its effect
  resolves, and Honest returns to the hand the same way; Skill Drain
  negates the effects of face-up monsters, though they can still be
  activated. So each activation is negated, nothing changes, and it can be
  activated again. The pilots never do it (the copies' games: none), and at
  those decisions their answer is another play. In a loop the duel comes
  back to the same decision, so an activation in it looks no better and no
  worse than the decision before it; only the cut at the cap, thousands of
  decisions on, says it was a loss.
- **A rule against it, for the runs after this experiment** (decided on
  2026-10-15, AGENTS.md's input rules): a command of the Main
  or the Battle Phase that the network's seat gave at the same position
  earlier in the turn is not offered there again (`Trail::offered` in
  `native/ygo-model-input/seat`, so here and at a table alike). Built on the
  branch `no-repeat` and merged on 2026-10-16, once this experiment's
  measuring part and that of
  [Without the deck lists](2026-10-15-without-deck-lists.md) had run on the
  rules their networks trained with. The same 60 games by its first
  choice: none reaches the cap, the ten end in 9 to 25 turns (six won), 47
  of the other 50 are played as before, and Lightsworn wins 37 for 31.

## What it says

1. *One network can be trained for all thirteen decks of training.* After
   1,920 updates of PPO, half of its duels against itself, the network of
   thirteen is over each deck's own pilot, 10.4 points over them all, from
   a copy that began at about their level.
2. *At the same updates, a deck pays for the others.* Blackwing pays two
   points for a second deck and about five for a third, and no more for
   thirteen; on their screens Lightsworn loses 3 points and Gishki 7 from
   the smallest network that plays them to the network of thirteen.
3. *Counted by its own games, a deck learns faster among others.* The
   network of thirteen got as far on Blackwing in a fifth of the games of
   Blackwing the network of Blackwing alone needed: the other decks'
   games teach it too.

What it does not say: whether more updates or a larger network close the
price (the network of thirteen has been at 72% to 74% of the held-out
deals since 1,200 updates); what the duels against itself cost a network
of many decks against the pilots, which no control here measures; how a
network of many decks fares in recorded duels; and anything of another
seed.

## What follows

Closed on 2026-10-15, the evening it was read, with every reading taken.
What was left open, as decided that evening:

- **No controls against the pilots alone**: the goal of the training is
  play against persons, which specializing on the pilots does not
  necessarily serve, and a control costs 10 to 20 hours of training time.
- **The graphs filled out to lists of 55 cards wait** for
  [Without the deck lists](2026-10-15-without-deck-lists.md), the
  experiment made active as this was closed: if the input without deck
  lists proves equivalent or better, the lists are dropped and the
  graphs need not be filled out to 55 cards at all.
- **The rule against repeated commands** (branch `no-repeat`) is merged
  before the next run, once that experiment's measurements, taken
  without it, are in.

## Follow-up: beyond their decks (readings fixed on 2026-10-16)

Two measurements of this experiment's networks, which read the deck lists
as the input will not once they are dropped, taken before the input
changes. Proposed on 2026-10-15 when the record was closed, and their
readings fixed on 2026-10-16, before any of it ran.

- **The decks outside training, played by the networks themselves.** Each
  network's copy at its last checkpoint and its checkpoints after 1,680,
  1,800 and 1,920 PPO updates play each of the eighteen decks outside
  training against the pilots of the thirteen decks of training, 16 games
  a pairing, 3,744 games (`eval_suites/ygo-policies-unseen-played.json`),
  and each of those decks' own pilots plays the same games. The cards
  those decks alone hold were never trained on: the network knows them by
  what is printed on them, as it would a player's own deck.
- **The mirror, the network of thirteen against the network of Blackwing
  alone**, both playing Blackwing on the 2,048 deals of the mirror
  (`eval_suites/blackwing-against-checkpoint.json`), each deal from either
  seat, their checkpoints after 1,680, 1,800 and 1,920 updates matched:
  12,288 games.

Both are played under the rule against repeated commands, merged by then.
The readings, by the network's last three checkpoints, game by game on the
same games:

1. *Over or under the pilots*: the network of thirteen against the decks'
   own pilots, the eighteen decks together: **over** at 2 points and two
   standard errors more, **under** at 2 points and two standard errors
   fewer, otherwise not told apart.
2. *More decks help with a deck never trained on*: the network of thirteen
   against the network of Blackwing alone, the same games: at 2 points and
   two standard errors more.
3. *PPO carries over to a deck never trained on*: the network of thirteen
   against its own copy, the same games: at 2 points and two standard
   errors more.
4. *The mirror*: the network of thirteen is **worse** when it wins at most
   48% and two standard errors under half, **better** at 52% and two
   standard errors over, otherwise not told apart.

Told beside them without a mark: every network, copy and PPO; deck by deck;
the mirror checkpoint by checkpoint. On the CUDA machine, the first part of
the night's command of 2026-10-16
([its record](2026-10-16-training-differences.md)):

```sh
PARTS=beyond bash experiments/commands/more-decks.sh
```

About an hour by reckoning; `analysis/more-decks/beyond` comes back, the
readings in `report.txt` there (`tools/more_decks_beyond_report.py`). Tried
small on the processor of the development machine on 2026-10-16.

### Results of the follow-up (read on 2026-10-16)

Measured on the CUDA machine from 00:55 to 02:32 on 2026-10-16, at
`a7523f68`, nothing else running there (the three runs of
[the next experiment](2026-10-16-training-differences.md) followed it);
`analysis/more-decks/beyond`, the report in `report.txt`.

The decks outside training, played by the networks themselves, 3,744 games
each; those decks' own pilots win 46.6% of the same games:

| Network | Its copy | After PPO (the three) |
|---|---|---|
| Blackwing alone | 22.1% | 24.9% |
| 2 decks | 23.6% | 27.9% |
| 3 decks | 19.8% | 25.4% |
| 5 decks | 22.7% | 27.9% |
| 8 decks | 24.6% | 30.8% |
| 13 decks | 27.5% | 34.1% |

The readings:

1. **Under the pilots**: the network of thirteen wins 34.1% where the
   decks' own pilots win 46.6%, 12.5 ± 0.7 points fewer.
2. **More decks help**: 9.2 ± 0.5 points over the network of Blackwing
   alone (24.9%).
3. **PPO carries over**: 6.6 ± 0.6 points over its own copy (27.5%).
4. **Worse in the mirror**: it wins 27.0% ± 0.4 of the 12,288 games
   against the network of Blackwing alone (26.2%, 25.8% and 29.1% by
   checkpoint).

Beside the readings, without a mark:

- **Deck by deck it is far apart.** On Verdict it plays as its pilot does
  (79.5% against 79.3%), and on nine more decks it is within ten points
  (Arcana, Destiny Hero, Draconic Might, Gravekeeper, Machina, Ojama,
  Pyramid, Watt, X-Saber); on Countdown it is 38 points under, on Six
  Samurai 24, Quickdraw Plant 22, Fortune Lady 20 and Harpie 16 (208 games
  a deck, a standard error of some three or four points).
- **Past three decks, the more decks the better a new deck is played**:
  24.9%, 27.9%, 25.4%, 27.9%, 30.8%, 34.1%; the copies too rise from
  22.1% to 27.5%.
- **The mirror is the specialist's own ground.** The network of Blackwing
  alone played about 540,000 games of this mirror in training (all 260 of
  its duels an update against itself, and 20 against the Blackwing pilot),
  the network of thirteen about 5,900 (one pairing in 169 of either kind).
  Against the pilots on the held-out deals the two were 4.8 points apart;
  against each other in the mirror, 46.

What it says: a network that has played more decks plays one it never saw
better, and what PPO adds carries over to it, but it is still far from a
pilot written for that deck, the more so on some decks than others. And
what the duels against itself teach a network in its own mirror is not
learnt from other decks' games.

## The mirror along the training of Blackwing alone (2026-10-20)

Asked on 2026-10-20, when the mirror above was set beside the
network of every deck of the pool
([Seventy-one decks](2026-10-19-seventy-one-decks.md)), which will have
played about 1,100 games of it: when the network of Blackwing alone came
to lead the network of thirteen there, and after how many games of the
mirror. The network of thirteen at its last checkpoint (1,920 updates)
against the network of Blackwing alone at its copy's end and at each of
the sixteen checkpoints its PPO kept, on the 2,048 deals of the mirror,
each from either seat (4,096 games a point). Both networks are of format
14, so they were played at `a7523f68`, the revision the follow-up's
mirror was played at, in a checkout of their own with the submodules at
its pins, on the development machine's processor with the follow-up's
options (`--workers 24 --pool-threads 20 --pool-slots 16 --seed 2012`).
The follow-up's two files of 1,920 updates were played again first and
came out the same, game for game; 2,048 games took 50 s. Fixed before the
curve was measured: the network of Blackwing alone is level at a point
where its share is within two standard errors of 50%, ahead above that,
behind below it; the error is that of the mean over deals, a deal's two
games together.

| Its PPO updates | Games of the mirror by then | Blackwing alone's share | |
| ---: | ---: | ---: | --- |
| 0 (its copy) | 0 | 30.5% ± 0.7 | behind |
| 120 | 33,600 | 57.1% ± 0.7 | ahead |
| 240 | 67,200 | 62.5% ± 0.7 | ahead |
| 360 | 100,800 | 65.5% ± 0.7 | ahead |
| 480 | 134,400 | 66.9% ± 0.7 | ahead |
| 600 | 168,000 | 69.0% ± 0.6 | ahead |
| 720 | 201,600 | 70.1% ± 0.6 | ahead |
| 840 | 235,200 | 69.2% ± 0.7 | ahead |
| 960 | 268,800 | 71.6% ± 0.6 | ahead |
| 1,080 | 302,400 | 68.8% ± 0.7 | ahead |
| 1,200 | 336,000 | 71.4% ± 0.6 | ahead |
| 1,320 | 369,600 | 72.1% ± 0.6 | ahead |
| 1,440 | 403,200 | 71.3% ± 0.6 | ahead |
| 1,560 | 436,800 | 70.0% ± 0.7 | ahead |
| 1,680 | 470,400 | 72.6% ± 0.6 | ahead |
| 1,800 | 504,000 | 72.5% ± 0.6 | ahead |
| 1,920 | 537,600 | 70.9% ± 0.6 | ahead |

The games of the mirror are its 280 an update, 260 against itself and 20
against the Blackwing pilot. The network of thirteen had about 5,900 after
its 1,920 updates, and 76,800 duels of Blackwing in all.

What it says: the copy is 19.5 points behind the network of thirteen.
Within its first 120 updates of PPO, 33,600 games of the mirror and 62,400
duels in all, fewer duels of Blackwing than the network of thirteen had,
it is 7 points ahead; where in those 120 it passed is not told, no
checkpoint having been kept between. It reaches 70% after 720 updates
(about 200,000 games of the mirror) and stays at 69% to 73% to the end:
the first 120 updates bring two thirds of what its PPO adds in the
mirror, the first 720 all of it. With itself's curve against its own
start ([Self-play](2026-10-11-self-play.md)) had the same shape: 63.5%
after 120 updates, 70.9% after 720, as much after. The measurements are in
`analysis/mirror-curve` (`run.sh`, `read.py`, `curve.txt`).

Played again by today's code the same day, the checkpoints of both folded
for the mirror (`tools/fold_deck_lists.py`; `docs/commands.md`, its
section on a checkpoint of format 14 played by today's code): the network
of thirteen against each point ended 69,617 of the 69,632 games as at
`a7523f68`, and every share is within a tenth of a point of the one above.
Where another network stands on this curve is measured with
`experiments/commands/blackwing-mirror-curve.sh`
(`analysis/blackwing-mirror-curve`, this one in `decks-13` there).

### The other networks on that curve (readings fixed on 2026-10-20)

The question that day: does Blackwing's head-to-head strength in the
mirror come from playing the mirror against itself, or from any game of
Blackwing against itself, the network on another deck in the other seat
counted too? A third possibility, every matchup learnt on its own, the
mirror cannot tell. The networks of 2, 3, 5 and 8 decks played Blackwing
against themselves in different mixes, every ordered pair of their decks as
often: in Blackwing's seats, M in the mirror (both seats Blackwing) and A
with itself on another deck, after their 1,920 updates of PPO. `decks-01`
plays 520 seats of the mirror against itself an update, so the first answer
puts a network near decks-01 after M / 520 updates (p1), the second near
(M + A) / 520 (p2):

| Network | M | A | p1 | p2 |
| --- | ---: | ---: | ---: | ---: |
| `decks-02` | 249,600 | 249,600 | 480 | 960 |
| `decks-03` | 110,900 | 221,900 | 213 | 640 |
| `decks-05` | 39,900 | 159,700 | 77 | 384 |
| `decks-08` | 15,600 | 109,200 | 30 | 240 |
| `decks-13` | 5,900 | 70,900 | 11 | 148 |

Each network's last checkpoint, folded for the mirror, against `decks-01`
at its copy's end and at each of its sixteen checkpoints of PPO, every deal
from either seat (`experiments/commands/blackwing-mirror-curve.sh`; one
checkpoint a network, `decks-05` and `decks-08` first, the other two after
at a lower priority). Under and level as `tools/blackwing_mirror_curve.py`
tells them (the network's share and two standard errors under 50%; within
them). The readings, fixed before any was measured:

- *As its mirror games alone*: under `decks-01` at its first kept point
  above p1 (120 for `decks-05` and `decks-08`, 240 for `decks-03`, 600 for
  `decks-02`).
- *As all its games against itself*: level with or over `decks-01` at its
  last kept point at or below p2 (240 for `decks-08`, 360 for `decks-05`,
  600 for `decks-03`, 960 for `decks-02`).
- *Partly* when it is neither.

By the same rule `decks-13`, measured above, is as its mirror games alone:
under at 120, where p2 would have it level there (its two points fall
together; it can be nothing between). What the curve does not tell apart:
a game on another deck that counts for less, and a network of more decks
that plays the mirror worse for sharing its weights among them (against
the pilots, game for game of Blackwing, the networks of more decks learnt
Blackwing faster). For `decks-02` both points lie where the curve has
nearly flattened, so its reading is the weakest.

#### Read on 2026-10-20: `decks-05` and `decks-08`

Measured that afternoon on the development machine's processor at the code
of `afa94b1c` (`analysis/blackwing-mirror-curve/decks-05` and `decks-08`,
their reports there). The time then went to the matchups below:
`decks-05` stops at `decks-01`'s 720 updates and `decks-08` at 1,440, past
every point a reading takes. The rest, and `decks-02` and `decks-03`, were
dropped once the matchups were read.

| Network | Against the copy | 120 | 240 | 360 | 480 | 720 | Its place | p1 | p2 | The reading |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- |
| `decks-13` | 69.5% | 42.9% | 37.5% | 34.5% | 33.1% | 29.9% | between 0 and 120 | 11 | 148 | as its mirror games alone |
| `decks-08` | 72.0% | 45.2% | 40.1% | 37.5% | 35.4% | 31.6% | between 0 and 120 | 30 | 240 | as its mirror games alone |
| `decks-05` | 75.4% | 50.4% | 46.4% | 43.0% | 41.2% | 38.4% | level at 120 | 77 | 384 | partly |

Each share is good to 0.6 or 0.7 points. What it says: a game of Blackwing
against itself on another deck counts for little in the mirror. Counted as
mirror games, `decks-05`'s would have put it near `decks-01` after 384
updates; it stands near 120, where its mirror games alone put it near 77:
very roughly a seventh of a mirror game each. `decks-08`'s together counted
for less than 46,800 seats of the mirror, under 0.43 each.

### Blackwing against another deck, head to head (readings fixed on 2026-10-20)

The prediction on the mirror's first two networks: the network that played
the mirror against itself has the most experience of exploiting its own
deck, and the same may hold for every matchup, so a network that played
Blackwing against itself on another deck should do better in that matchup
than the network of Blackwing alone, provided the second deck is the same
in both matches. That tells more than the mirror of the other two networks:
those, and the late points of `decks-05` and `decks-08` there, were put
off, and then dropped.

For each of `decks-05`'s other decks X (Lightsworn, Gishki, Burn and
Crystal), Blackwing against X on the 2,048 deals of
`eval_suites/blackwing-against-X-checkpoint.json` (Blackwing opens half of
them), X played by a network held in the other seat, Blackwing by
`decks-05` and by `decks-01` on the same deals, each network at its last
checkpoint and folded for its pairing. Against itself on X, `decks-05` had
played Blackwing in 39,900 duels by its end (and 7,700 against X's pilot),
`decks-08` in 15,600 (4,800), `decks-13` in 5,900 (3,000), `decks-01` in
none (38,400 against X's pilot).

The readings, fixed before any was measured, by the difference of
Blackwing's shares deal by deal over the four decks together (8,192 games):

1. *Against `decks-05` playing X* (the prediction above): the matchup's
   own games count when `decks-05`'s share is over `decks-01`'s at 2
   points and two standard errors more; the deck's other games count for
   more when `decks-01`'s is over at as many; otherwise not told apart.
2. *Against `decks-13` playing X*: the same. Here neither has met the
   network on the other side, as `decks-01` had not met `decks-13` in the
   mirror; at (1) `decks-05`'s X has played its own Blackwing, which may
   help either side.

Beside them, without a mark: each deck alone, and `decks-08` and `decks-13`
playing Blackwing on the same deals.

#### Read on 2026-10-20

Measured that afternoon on the development machine's processor
(`experiments/commands/blackwing-matchups.sh`; `analysis/matchups`, its
report in `report.txt` there). The same networks against the four decks'
pilots, on 2,048 deals each (`eval_suites/blackwing-against-X-pilot.json`),
were added once the two readings had been measured, as the measure to set
them beside. Blackwing's share, each network playing it:

| Against | `decks-01` | `decks-05` | `decks-08` | `decks-13` | `decks-05` less `decks-01` |
| --- | ---: | ---: | ---: | ---: | ---: |
| `decks-05` playing the other deck | 65.4% | 64.0% | 61.2% | 61.5% | −1.4 ± 0.5 |
| `decks-13` playing the other deck | 70.9% | 69.3% | 66.8% | 67.2% | −1.6 ± 0.5 |
| The decks' pilots | 79.3% | 78.2% | 75.7% | 75.1% | −1.1 ± 0.5 |

Deck by deck, `decks-05` less `decks-01` (± 0.6 to 1.3): against `decks-05`
Burn +1.3, Crystal +2.6, Gishki −10.7, Lightsworn +1.4; against `decks-13`
+2.0, −0.6, −7.5, −0.4; against the pilots +2.1, −0.2, −9.6, +3.2.

1. **Against `decks-05` playing the other deck: not told apart** (−1.4 ±
   0.5 over 8,192 games).
2. **Against `decks-13` playing the other deck: not told apart** (−1.6 ±
   0.5).

What it says: outside the mirror a network is no better head to head than
it is against the pilots. `decks-05` stands 1.1 to 1.6 points under
`decks-01` whoever plays the other deck, deck by deck alike (about ten
under against Gishki every time), and `decks-08` and `decks-13` 3.6 to 4.2
under in each. Its 39,900 duels of Blackwing against itself on these decks
gave it nothing head to head beyond what it shows against their pilots:
the prediction is not borne out at the mark. In the mirror it was
otherwise: `decks-01` after 120 updates, 31,200 duels of the mirror
against itself, won 65.7% of the held-out deals against the pilots to the
network of thirteen's 72% to 74% at its end, and was 7 points over it in
the mirror. The mirror is the matchup where playing it against itself
shows head to head; these four are not. What makes it so is not told; in
the mirror the network plays the deck it faces, in both seats of every
game against itself.
