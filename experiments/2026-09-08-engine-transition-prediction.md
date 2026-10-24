# How cards work: predicting what the engine does after an answer

**Status:** COMPLETED

**Proposed:** 2026-09-08 (for the earlier pilots)

**Activated:** 2026-10-22, when
[A critic that sees the hidden cards](2026-10-22-privileged-critic.md) was
closed.

**Closed:** 2026-10-23, the morning the run was read: the breakdown of the
first reading and of the other targets is not made.

**Last updated:** 2026-10-23 (the run read: the head predicts, 0.840 ±
0.004 of the frequencies' log-loss taken away; PPO with it not told apart
from `all71` at R1, +0.3 ± 0.2 points; closed)

First drafted on 2026-09-08, before the earlier pilots were retired, and
brought in unchanged on 2026-10-04; that text is in this file's history.
Refreshed on 2026-10-22, while the run of
[A critic that sees the hidden cards](2026-10-22-privileged-critic.md) went,
as the next experiment should that one not work out: teaching the model to
predict what cards do.

Why now. The question since [The signal of an
update](2026-10-22-update-signal.md) is how to learn faster per duel. An
update is limited by noise, and its signal is the credit a value gives to
one result a duel for each seat. The critic tried to make that credit
cleaner; its first phase read it as held back by how much it can learn, not
by what it is shown: like the value head, it learns what every card is
worth from results alone. Its run, read the evening this was opened, learnt
no faster (+0.1 ± 0.2 points against `all71` at R1). This experiment adds
signal instead: a label at
every transaction, from the engine's own execution, of what the cards it
touched did. [Card descriptions](2026-10-17-card-descriptions.md), a fixed
vector of each card's text, did not make a network play measurably better;
this is learnt from the duels themselves, in the network's own terms. The
rejected Monte Carlo auxiliary (an earlier record, not included here)
predicted the duel's payoff, which PPO already has; this predicts what
happens next on the board.

## Research questions

1. Can the network learn, from the engine's execution, where the cards it
   sees go after an answer, and what becomes of their statistics and of the
   life points?
2. Trained to predict it beside PPO, does it learn to play faster per duel?
3. How much of it does the network already know after its copy of the
   pilots? (Told beside, without a mark.)

## Design (proposed on 2026-10-22; the readings confirmed rest on it)

- **What is predicted**: at the decision that ends a transaction of the
  seat, every choice of it made, what the seat sees change by the
  *horizon*: its next decision with no chain open, or the end of the turn,
  or of the duel, whichever comes first. A transaction ends where the other
  seat has something to answer, often before an effect resolves; the chain
  resolved, the seat sees what its answer and the answers to it did. Passing
  a chain window is an answer too: its label is what the chain it let
  resolve did.
- **The targets**, all of what the seat sees, at the horizon as at the
  decision:
  1. each card in the decision's state: its place at the horizon, as the
     seat sees it there (its location, whose side, and on the field its
     position), or where it was;
  2. each card on the field at both: the change of its current ATK and
     DEF, scaled as the input scales them;
  3. each player: the change of life points, scaled as the input scales
     them, and of the number of cards in each location (which tells of the
     cards that come into sight: drawn, or summoned from the Deck).
  The places by cross-entropy, the changes by squared error, each loss the
  mean over what it is of.
- **Where the labels come from**: `crates/ygo-env`, beside the seat,
  follows each card of the decision's state through the engine's messages
  to the horizon, as the seat's projection moves it (a move names where a
  card leaves and where it goes), and takes its place there as the seat
  sees it. A card is named by its place in the decision's state, as the
  network reads it; nothing of a label reaches the input, and
  `native/ygo-model-input` is not changed. The labels are audited against
  the seat's own view at the horizon: every card followed there is where
  its view shows it.
- **The head**: a network of its own beside the policy, as the critic is,
  kept beside each checkpoint and never in one. It reads the policy
  network's own card tokens, state and chosen answer at the decision, and
  its gradient reaches that network: what it is for is to shape what the
  network learns of the cards. The network's input, its checkpoint and its
  play are unchanged, and nothing that plays reads the head.
- **Taught at every transaction's last decision** of an update, in PPO's
  own minibatches: its labels are many and each tells something of its own,
  where a duel's one result is told again at every position of it.
- **Its weight**, fixed before the run on the copy's first update: the
  weight at which the head's gradient on the shared parameters is as large
  as PPO's.

### Phases

1. **The labels**, built and audited on a few hundred duels: their agreement
   with the seat's own view at the horizon, and how often each place and
   change occurs.
2. **The probe**, told beside without a mark: the head taught alone on
   `all71`'s copy, its network held as it is, and read on games it never
   trained on against the frequencies (below): what the copy's network
   already tells of the cards.
3. **The run**, of the shape of the critic's: one arm, `cards`, PPO on from
   `all71`'s copy for 1,920 updates as `all71` went on from it (its
   settings, seed and games) with the head taught beside it, measured as
   `all71`'s R1 was, against `all71` there. About seven to eight hours on
   the CUDA machine by the critic's pace, and an hour and a half of
   measuring.

## Readings (fixed on 2026-10-22, before anything was built)

1. *It learns how cards move*: the arm's head with R1's network, on the
   games of four held updates far from the run's, against the frequencies of
   each place by the card's location, its side and the kind of answer
   (counted on the run's own games): it predicts when it takes away at least
   half of the frequencies' log-loss of the places of the cards the seat
   sees, over all of them, less two standard errors (the duels drawn again).
2. *It learns faster*: `cards` against `all71` at R1, both by their
   checkpoints after 1,680, 1,800 and 1,920 updates of PPO, game by game
   over the thirteen screens, the eighteen and the forty together: faster
   when the gap less two standard errors is at least 2 points, slower when
   the gap and two standard errors are at most -2 points, otherwise not told
   apart.

Told beside them, without a mark: the probe on the copy; the other targets
(statistics, life points, counts) by turn and by the kind of answer; each
suite; the training win rate; the cost of an update. Reading 1 unmet, the
second says nothing of what knowing the cards is worth: the head would not
have learnt them.

## The build (2026-10-22, before the experiment is opened)

Built while the critic's run went, the readings confirmed; nothing of it
has run beyond trials.

- **The labels** (`crates/ygo-env/src/follow.rs`): a `Follower` takes the
  cards of a decision's state in the order the network reads them and moves
  each as the seat's projection does, given each message first, while the
  projection still holds the piles as they were (a card taken out of a pile
  at its index, the others after it closing up; put in at its index, those
  after it moving on; field zones fixed; the Deck a count). It loses a card
  the seat cannot follow: one of the other player's hand or set cards dealt
  again, or one a refresh of the field drops. A setup's `follow` gives each
  caller's seat its followers, of a transaction its last decision's; at the
  horizon each gives its label, which the pool keeps under the decision's
  row (`Pool::batch_labels`, `DuelPool.labels`). The places are 23: on the
  seat's side and on the other's, the hand, a monster face-up in Attack,
  face-up in Defense or face-down, a spell or trap face-up or face-down, the
  Graveyard, banished, the Deck, the Extra Deck, a monster's material; and
  out of the duel. The changes of ATK and DEF are of cards face-up on the
  field at both ends, a quarter of the count of the cards in each location
  of each seat, all scaled as the input scales them.
- **The audit**, the first trials: on 40 duels of random answers 92% of the
  decisions labelled (the others not their transaction's last), 0.4% of the
  cards lost, about 3% of them in another place at the horizon, and every
  card the seat knew at both ends (16,416) found where it was followed to
  with its identity; on two updates' and one held update's games of
  `all71` at R1, 2,763,451 of 2,763,451. The network's input and the duels
  are the same with the labels or without (`tests/test_pool.py`).
- **The head** (`ygo_training/card_head.py`): of each card, its token beside
  the decision's state and chosen answer, through a layer of the network's
  width, to the 23 places and the two changes; of the decision, its state and
  answer to the life points' and counts' changes. The network gives its own
  parts from the same pass (`PolicyValueNet.parts`), and the head is taught
  in PPO's minibatches at every labelled decision (`train --card-head`).
- **Choices of the build the design left open**: the weight scales the
  gradient the head gives the network (`card_head.scaled`), and the head
  itself learns as much whatever the weight: at 0 the network's update is
  the one without the head, bit for bit (`tests/test_card_head.py`). The head
  learns at 3e-4 with an optimizer of its own, as the critic did, and its
  gradient is clipped apart from the network's. The run's head begins as the
  probe leaves it (`--card-head-begun`), so that what it gives the network
  from the first update is of cards and not of its own beginning. The
  weight is measured over the whole of the copy's first update's games, the
  update's gradient as one: the head's and PPO's then move the network as
  much an update. The frequencies are counted by the card's location (of the
  input's 16), its side and the kind of answer (32), one each to begin with.
- **The tools**: `tools/card_head.py probe | weight | read | report`, and
  `experiments/commands/card-head.sh` (`PARTS="probe weight check arm read
  measure"`), which measures the arm and reads `all71` at R1 where the
  critic's run measured it at this code; the report is the critic's
  (`tools/privileged_critic_run_report.py --arm cards --cards ...`). Tried
  small on the development machine against a stand-in of `all71` made by
  its own script: every part ran, the check found the same games.

## The run (2026-10-22)

Started on the CUDA machine at 23:06, at `58670ff9` with the submodules at
their pins (its checkout taken from the development machine's commits by
the branch `critic-build`, as the critic's was), nothing else running
there, in tmux `cards-run`, from the repository root:

```sh
PYTHON=.venv/bin/python bash experiments/commands/card-head.sh
```

`all71` at R1 is read where the critic's run measured it the same evening
(`analysis/privileged-critic/run`), every game of it as seventy-one decks
measured it: only the arm is measured. About seven hours by reckoning.

**The probe**, told beside without a mark (23:06 to 23:12,
`analysis/card-head/run/probe`): on the held games (7,334,677 cards) the
head on the copy's network takes away −0.342 ± 0.034 of the frequencies'
log-loss (0.293 against 0.218): **not shown to predict**. It was still
learning at the 40th update (0.29, from 5.99 at the first). A choice of
the build not listed above: the probe teaches the head 8 of each seat's
labelled decisions a duel, drawn at random, as the critic was taught, where
the arm teaches it every one, about seven times as many; the probe tells
less of what the copy's network knows than it might. The labels on the way:
64,050,627 of 64,050,627 cards known at both ends found where they were
followed to. The probe is made again once the run's script has ended, its
head taught every labelled decision as the arm's is, told beside as the
first was (`analysis/card-head/run/probe-all`; on the CUDA machine in tmux
`probe-all`, `~/ygo-training/probe-all.sh`, which waits for the run).

**The weight** (`analysis/card-head/run/weight.json`): on the copy's first
update's games PPO's gradient on the shared parameters is 0.0273 and the
probe's head's 0.123 (their cosine −0.09): the weight 0.222.

**The check passed**: three updates from the copy at this code, without the
head, played `all71`'s first three (versions 17,040 to 17,042), the
fingerprints of every decision and of what each update trains on the same.
The arm began at 23:12, its updates about 10.8 seconds each. Its head,
begun as the probe left it, came down fast in PPO's minibatches, each read
before it is taught it: its log-loss of the places 0.272 at the first update,
0.107 at the fourth, 0.062 at the eleventh, about 0.05 from the thirtieth
(the frequencies on the probe's games about 0.21).

## Results (read on 2026-10-23)

The arm made its 1,920 updates by 04:45, 10.4 seconds each on average
against `all71`'s 9.0; the first reading was taken by 04:47, the arm was
measured by 05:28, and the probe made again from 05:29 to 05:36
(`analysis/card-head/run`, its `reading-1.json`, `report.txt` and
`report.json`, on both machines).

**1. It learns how cards move: it predicts.** The arm's head with the
network of its last checkpoint, on the games of four held updates
(6,752,496 cards), takes away 0.840 ± 0.004 of the frequencies' log-loss of
the places (0.0363 against 0.2268), where the mark was a half less two
standard errors. The labels on the way: 12,112,352 of 12,112,352 cards
known at both ends found where they were followed to.

**2. It learns faster: not told apart.** Over the three suites the arm
wins 55.1% against `all71`'s 54.8%, +0.3 ± 0.2 points over 33,696 games:
by two standard errors between nothing and +0.6 points, where the mark was
2.

| Suite | `cards` | `all71` | Gap | Games |
| --- | --- | --- | --- | --- |
| The thirteen screens | 55.2% | 54.8% | +0.4 ± 0.2 | 21,632 |
| The eighteen other decks of thirty-one | 50.0% | 51.4% | −1.3 ± 0.5 | 3,744 |
| The forty new decks | 57.1% | 56.4% | +0.8 ± 0.4 | 8,320 |

Told beside them, without a mark:

- **The probe made again**, its head taught every labelled decision as the
  arm's is (`analysis/card-head/run/probe-all`): on the held games it takes
  away 0.226 ± 0.024 of the frequencies' log-loss (0.169 against 0.218),
  where the first probe took away −0.342; it was still learning at the
  40th update (0.170 on that update's games before it was taught them).
  At about as many updates the arm's head read 0.05 in its minibatches:
  the copy's network, held as it is, gives a head a good deal less than a
  network that learns beside it.
- **Each screen** (1,664 games each, ± 0.6 to 0.9): Crystal −3.4,
  Blackwing +2.3, Heroes +2.2, Rock-Block +2.1, Morphtronic +1.7, the rest
  within 1.2. On Blackwing the critic's arm was over `all71` by as much
  (+2.7; the two arms between them −0.4 ± 0.8): the gap there is
  `all71`'s, a run of its own, not something either arm does. Set against
  each other, the two arms,
  neither faster, are +0.19 ± 0.17 points apart over the three suites, but
  screen by screen further than the games' error allows (Gishki +2.0,
  Crystal −2.5): a screen's gap of two or three points is within what two
  runs differ by on their own.
- **Along the run**: against the pilots in training level with `all71`
  (−0.10 ± 0.06 points paired by update, 57.2% against 57.1% in the last
  240). The head's log-loss of the places in PPO's minibatches 0.049 in the
  first 240 updates, 0.037 in the last 240; its squared errors of the
  other targets fell most in the first 240 and little after (the changes
  of ATK and DEF 0.0014 to 0.0004, of the life points 0.0018 to 0.0011, of
  the counts 0.0053 to 0.0039). By turn and by the kind of answer, as the
  readings' list asks, they are not taken: the tools read the places over
  all the cards only.
- **How it plays**: as `all71` (entropy 0.235 against 0.233 in the last 240
  updates; chain passes 6% more and decisions 4% more in the last 480).
  Duels cut off: 14 against 6, one an update at most.
- **The cost**: 10.4 seconds an update against 9.0, 16% more.

What it says: with the head beside it the network learnt to tell where the
cards it sees go after an answer, far beyond what the copy's network gave
a head, and PPO with it learns no faster: at R1 the arm is within about
half a point of `all71`. As this record put it before the run, the head
learnt and play did not: what holds play back is not where cards go. One
limit of the first reading: most cards do not move in a transaction, and
of those that do many go where the answer itself says (the card summoned,
set or activated); the head reads the answer, the frequencies do not, so
the share does not tell how much of the effects of the cards it learnt. A
breakdown by whether a card moved and whether the answer named it would.

**Closed** that morning, without that breakdown or the other targets by turn
and by the kind of answer. Next, the rule that no longer offers a face-up
monster's effect while its effects are negated, agreed on 2026-10-22 to be
built after this run ([Seventy-one decks](2026-10-19-seventy-one-decks.md),
its section on the duels it cut).

## What would tell

- The probe already near the arm's head: the copy's network knew how the
  cards move, and a gain of the run would be of something else (a
  regularizer), a loss of interference.
- The head learnt (reading 1) and no faster learning (reading 2 not told
  apart): what holds play back is not what cards do but when to do it.
- Faster: the knowledge of cards was missing, and the copy from scratch with
  the head is the next step (below).

## Not in it, for later

- The copy made with the head from scratch, imitation and PPO, which asks
  whether the representation learns faster from the start: on thirteen
  decks beside `half`, about eleven hours.
- The head taught in pretraining only and left out of PPO, which asks
  whether what it taught is kept (the first draft's second arm).
- Predicting what the seat does not see, the other player's hand (a belief
  of its own): a training target may be hidden, as the critic's input is,
  but it is another question.

## Research context

[UNREAL](https://arxiv.org/abs/1611.05397) (Jaderberg et al. 2016) learnt
faster on Atari and Labyrinth with predictions beside the policy that share
its network; [Self-Predictive Representations](https://arxiv.org/abs/2007.05929)
predict the network's own future representations. Neither is evidence that
these targets help a Yu-Gi-Oh! player: that is the run's to say. Neither
AlphaStar nor ygo-agent trains such a head.
