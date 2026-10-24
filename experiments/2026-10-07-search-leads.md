# Learning the search's leads

**Status:** COMPLETED

**Proposed:** 2026-10-07

**Activated:** 2026-10-07

**Completed:** 2026-10-08

**Last updated:** 2026-10-08

## Research questions

At a decision the one-step search examines, it plays the pilot's answer and
up to eleven alternatives out in the same worlds. An answer's *lead* is how
much more often it wins there than the pilot's answer, in points of win
rate.

1. Trained on the leads the search measured, does a network tell on games it
   has not seen which answers do better than the pilot's, and which worse?
2. Does a player that leaves its first choice where it predicts a lead win
   more than one that imitates the pilot, and than one that imitates the
   answers the search played?
3. How much of what the search gains over its pilot does it take?

## Why

Every count of this section is of the search-imitation experiment's games,
against the pilots the other decks had then;
[the opponents](#the-opponents) have changed since.

[Imitating the search](2026-10-05-search-imitation.md) taught a learner the
one answer the search played at each decision. On 7,680 searched games it
gives the search's answer at 9% of the deviations of games it has not seen,
what its own other answers are worth by the search's measurements is better
than its control's for a while and then turns, and it wins as often as a
learner that imitated the pilot (+1.1 ± 1.1 points at best). More games help,
and by little.

The answer played is the least of what the search knows. In the 1,536
held-out games it leaves its pilot 2,225 times, 1.45 times a game, and half
of those would not be taken again with other worlds. In the same games it
examined 81,443 decisions, 53 a game, and measured 114,753 alternatives
against the pilot's answer, 75 a game. Those measurements were not used:

- They are graded. An alternative is 20 points ahead, level, or 30 behind;
  the label says only whether the search's test passed.
- They say where not to leave the pilot. The alternatives the search tried
  lose 5.7 points on average, and the learners' own other answers 1 to 3.
  Nothing in a played answer says so.
- They are not selected. Every alternative the search tries goes through the
  whole first stage, 8 worlds, whatever comes of it, so its lead there is an
  unbiased measurement, where the answer played is what passed a test at a
  one-sided 5% level. An alternative that is truly a little better can pass
  that test less than half the time, and a learner that fitted such labels
  perfectly would then still answer as the pilot does.

Where the pilot's answer wins all 8 worlds the search tries nothing else:
32,516 of the 81,443 decisions, with 80,128 alternatives. A learner's other
answers there can only cost, the Search arms gave more of them than their
controls, and the earlier measure could not see them. `policy-bench search
--log full` has those alternatives tried as well, for the log alone. With
it a game holds about 127 measured alternatives where it holds 1.45
deviations.

Two experiments of the earlier lineage, on its environment and its
representation, stand behind this one:

- Action-conditioned Monte Carlo auxiliary loss (an earlier record, not included here)
  added a head that predicts the duel's outcome from the answer taken, and
  rejected it. It saw the answers the learner took and a whole duel's result
  for each, and its record says the result does not speak of a method with
  counterfactual labels. This is one. What is kept from it: the control
  carries the same head with its loss at zero, an older checkpoint loads
  with only the new head initialised, and the loss's weight is set from the
  size of its gradient before any result is seen.
- Counterfactual reanalysis (an earlier record, not included here)
  moved a policy towards the better of paired rollouts, a handful of
  comparisons at a time. The movement carried over to positions it had not
  seen. The player trained with it, on mirror games alone, lost to its
  control against the other decks, and its record leaves the method open
  for a pool of opponents like the one played here.

## The opponents

On 2026-10-06 and 07 the pilots of 24 decks were improved in ygo-policies
(`bc8c0f7` to `75e4b37`), ten of the twelve decks Blackwing plays against
here among them. Blackwing's pilot is as it was, and so are Infernity's and
Monarch's, the engine, the card scripts, the card database and the search.
The experiment is run against the improved pilots (ygo-policies `1b85b63`:
`75e4b37` with the full log's commit on top). That was decided on
2026-10-07, before any of its games existed, on the rule that an experiment
which needs no comparison with earlier games takes the pilots of the day.

- **It needs none.** Every comparison that decides is between arms trained
  here on the same games. The search-imitation experiment's numbers are of
  the earlier opponents: background, and no baseline. The deals are the
  same and the games are others.
- **The teacher and the network's input are the same.** The seed's pilot is
  unchanged, the network sees what it saw, and the checkpoints load as
  before. The seed learned that pilot in games against the earlier
  opponents, so it is measured against these before anything starts from
  it: the first step of the games' script.
- **A head taught on the earlier games would not serve against these.** A
  lead says how an answer does with both pilots playing on.
- **What is given up.** The searched games are no longer those of the
  search-imitation experiment, so nothing here is compared with its arms
  game by game; and its recorded games play again only at `bc8c0f7`, or at
  `568711d` (that pin with the full log, on the ygo-policies branch
  `log-full-bc8c0f7`).
- **The pin stays where it is until the experiment is read.** Blackwing's
  pilot is one of seven that were not improved. A change to it changes the
  teacher: that takes a new seed and new games.

Measured on this machine on 2026-10-07 at `1b85b63`, beside what the same
measured at the earlier pin:

| | Earlier pilots | Improved pilots |
| --- | ---: | ---: |
| The pilot on the suite, 1,664 games | 64.9% | 55.5% |
| The pilot on the held-out deals, 1,536 | 67.6% | 60.7% |
| The pilot on the deals to train on, 7,680 | 67.1% | 59.5% |
| The search over the pilot, held-out deals | +9.7 ± 1.0 | +11.7 ± 1.1 |
| The search over the pilot, deals to train on | +11.2 ± 0.5 | +13.6 ± 0.5 |
| The search over the pilot, 192 of the validation deals | not played | +8.9 ± 3.2 |

Added on 2026-10-09: the two rates of the suite are not of the same deals.
A suite's deals follow from `--seed`: the 64.9% is of seed 2012, which the
seed's script gave, and the 55.5% of the default, 2011. With seed 2012,
the deals a training run of seed 2012 evaluates on, the pilot wins 60.3%
of the suite at `1b85b63` (1,003 of 1,664). The seed's 53.2% below is of
the default seed's deals, as its pilot's 55.5% is.

- On 192 of the validation deals (the first 16 a deck) the search takes
  the pilot from 61.5% to 70.3%, and a searched game holds 53.6 examined
  decisions, 127 measured alternatives and 1.58 deviations, where the
  earlier held-out games held 53.0, 127 and 1.45. The pilot's answer wins
  every world of the first stage at 34% of the examined decisions, 40%
  before. There is as much to learn from as there was.
- The reference games of the pin (`analysis/pins/1b85b63`): against
  Infernity and Monarch the 32 plain games and both games of the 4 searched
  deals are the earlier ones, digest for digest; against the ten others
  116 of the 160 plain games go otherwise.
- The full log changes no game at this pin either: both games of 16 deals.
- The trainer builds and its tests pass unchanged, among them the one that
  seats two pilots in the environment against the bench's digests and the
  one that reads a run of the bench.

## Design

The games were agreed on 2026-10-07. The rest was proposed that day and
confirmed that evening, with the readings, before any arm was trained. Two
things follow from the games by rules fixed here, and the scripts apply
them: the mark of the readings, and whether the arms start from the seed
as it is.

- **Games.** The deals of the search-imitation experiment, played against
  the improved pilots with the full log, and a third run to choose on:

  | Run | Seed | Games | For |
  | --- | ---: | ---: | --- |
  | Train | 800000 | 7,680 | the arms |
  | Validation | 850000 | 384 | choosing the update and the margin |
  | Held out | 900000 | 1,536 | the readings |

  Blackwing against each of the 12 other decks, half of the games going
  first; the search strict, with 8, 32 and 96 worlds and a z of 1.645, as
  before ([`search-leads-games.sh`](commands/search-leads-games.sh)). Every
  row holds the pilot's game and the searched game of its deal, so the
  runs themselves say what the pilot wins against these opponents and what
  the search gains.
- **The lead head.** A second number for every answer, beside the score the
  policy gives it: the lead the network predicts for that answer over the
  pilot's, from the same view of the state and the answer. It is trained by
  squared error on the lead measured in the 8 worlds of the first stage, a
  draw counting as half a win, for every answer the search tried at every
  decision it examined; the pilot's own answer has the lead 0. Like cards
  in one pile are one answer, as the search tries them. Decisions the search
  does not examine, the selections of cards among them, and answers past
  its twelfth give no target. The worlds of the later stages are left out:
  only alternatives that did well in the first reach them.
- **Arms.** From the imitation seed at version 89, on the same deals in the
  same updates, all with the head:

  | Arm | Games played again | The policy imitates | Lead loss |
  | --- | --- | --- | --- |
  | Pilot | the pilot's | the pilot's answer | none |
  | Search | the searched | the answer played | none |
  | Leads | the searched | the pilot's answer | yes |
  | Leads, reading only | the searched | the pilot's answer | yes, with no gradient into what the head reads |

  The Leads arm is the experiment. Its policy stays a copy of the pilot, and
  what it has of the search is in its head. The Search arm is the earlier
  experiment's method, from the better seed and on these games. In the
  fourth the head reads the
  policy's view of an answer without shaping it: the policy is then a
  pilot's copy and nothing else, and the arm says what a head can know from
  a view made for imitation. Whether predicting leads helps a policy that
  imitates the search's answers is left for later.
- **The seed.** The arms start from version 89 if, against these
  opponents, it wins no more than 5 points less than its pilot on the
  suite. Against the earlier ones it won 2.1 points less, a rate on the
  suite's 1,664 games is good to 1.2 points, and version 49, which the
  earlier experiment's arms started from, won 5.5 less. Further under, it
  is first continued against these opponents until it levels off, and the
  arms start from that. The rule was set on 2026-10-07, before the seed was
  measured against these opponents; the arms' script holds it against what
  the games' script measured and trains nothing where it fails.
- **How the Leads arm plays.** Its policy's first choice, unless it
  predicts for another answer a lead over that choice of at least a margin;
  then the answer with the largest predicted lead. Only at the decisions a
  strict search examines, which are the only ones the head is taught at: a
  Main Phase or Battle Phase command, a chain window that may be passed, a
  battle position or a yes/no question, with more than one alternative, no
  chain under way and no face-down monster on the other side. The trainer
  tells them from the decision as the network sees it, and refuses a logged
  run in which the search examined other decisions. With an infinite margin
  the arm is its policy.
- **Training.** The earlier experiment's recipe: updates of 480 games, 4
  epochs, minibatch 256, value weight 0.5, the games gone through in
  shuffled passes. 128 updates, eight times through the set, measured after
  updates 16, 32, 64, 96 and 128: the Search arm's other answers were worth
  most after 64, and nothing says where a head trained on leads is. The
  weight of the lead loss is set once, before any arm is measured, so that
  its gradient on the parameters it shares with the policy is as large as
  the imitation loss's: a calibration run of four updates from the seed at
  weight 1, and the ratio of the two gradients over its last three. The
  first is left out because the head starts untrained there, and its first
  errors are not those it goes on making. The arm that reads only has the
  weight 1, which then changes nothing but the head's own step.
- **Choosing.** The update and the margin of the Leads arm are chosen on
  the validation games, as those at which the rule gains most there, among
  the five updates and margins of 5, 10, 15, 20 and 30 points; the arm that
  reads only likewise. Of equal gains the earlier update is taken, then the
  larger margin. The held-out games are read once, at that choice, and the
  Pilot and Search arms at the updates so chosen.

## Measures

- **What the answers taken are worth.** At every decision the search
  examined in the measured games, the answer a model gives where it is not
  the pilot's, by the lead the search measured for it: how many such
  answers a game, and their leads added up over a game. With the full log
  this covers the decisions where the pilot's answer wins every world, and
  a model that has not seen the games is not flattered by it: its sum
  estimates what its answers gain or lose against the pilot's, the pilots
  playing on. For the Leads arm, the same at every margin.
- **What the head knows.** Over every alternative tried in the held-out
  games: the predicted lead against the measured one, and the mean measured
  lead of the alternatives it ranks highest.
- **Deals won** on the held-out run's deals, the Leads arm playing by its
  rule, beside the pilot's and the search's own results.

## Readings (fixed on 2026-10-07)

Proposed and confirmed on 2026-10-07, before any arm was trained.

The main comparison is the Leads arm, at the update and the margin chosen
on the validation games, against the Pilot arm at the same update, on the
1,536 held-out deals, game by game.

- *The leads are learnable* if the rule gains the mark or more a game, and
  more than three standard errors above zero. The rule's gain is what the
  answers the Leads arm takes lead the first choices they replace by, by
  the search's measurements, added up over a game. The mark is a quarter of
  what the search gains over its pilot on the held-out deals, in deals won.
  Against the earlier opponents the search gained 9.7 points and the mark
  was 2.4; the held-out run gives the gain against these before any arm is
  trained, and the report computes the mark from that run's rows.
- *It plays better* if, playing by its rule, it wins more of the held-out
  deals than the Pilot arm by the mark and by more than two standard
  errors.
- *Not learnable this way* if on the validation games the rule gains
  nothing above zero at any update and margin. The held-out games are then
  not read for it.
- What the head knows, the Search arm and the arm that reads only qualify
  the main comparison and decide nothing.

## Results (2026-10-08)

Everything ran on the CUDA machine on 2026-10-07 and 08 at ygo-policies
`1b85b63` and trainer `934feca4`, both clean in every `run.json`: the games
in 9 hours 40, the arms in 3 hours 30. No game failed, and the seed's rule
let the arms start. The files are in `analysis/search-leads/`.

### The games, the mark, the seed and the weight

| Run | Deals | Pilot | Search | The search's gain |
| --- | ---: | ---: | ---: | ---: |
| Validation | 384 | 60.9% | 73.4% | +12.5 ± 2.2 |
| Held out | 1,536 | 60.7% | 72.5% | +11.7 ± 1.1 |
| Train | 7,680 | 59.5% | 73.1% | +13.6 ± 0.5 |

- Against the improved opponents the search gains more over its pilot than
  it did against the earlier ones on the same deals (9.7 and 11.2 points).
  A held-out game holds 53.4 examined decisions, 127 measured alternatives
  and 1.53 deviations.
- **The mark is 2.9 points.** The search wins 1,113 of the held-out deals
  and the pilot 933, which is 11.7 points more.
- **The seed** wins 53.2% of the suite (886 of 1,664) and its pilot 55.5%
  (923): 2.2 points under, where it was 2.1 under against the earlier
  opponents. Its first choice is the pilot's answer at 94.4%.
- **The weight of the lead loss is 75.2.** Before the four updates of the
  calibration run the imitation loss's gradient on the shared parameters
  was 1.32, 1.87, 1.05 and 0.81, and the lead loss's 0.091, 0.017, 0.017
  and 0.016.

### The arms in training

128 updates each, eight passes through the 7,680 games. The head's fit is
the variance of its targets it explains, measured before each update on
that update's games; in the first pass the run has not yet seen them.

| Arm | Imitation loss, first pass to eighth | Lead loss | The head's fit, first pass |
| --- | ---: | ---: | ---: |
| Pilot | 0.087 to 0.055 | | |
| Search | 0.201 to 0.100 | 0.036, untrained | |
| Leads | 0.109 to 0.071 | 0.0156 to 0.0103 | 0.19 |
| Leads, reading only | 0.096 to 0.058 | 0.0183 to 0.0171 | 0.17 |

At the deviations of the games it trains on the Search arm gives the
search's answer at 5.1% in the first pass and 28.9% in the eighth, as the
arm of the earlier experiment did.

### The validation run and the choice

What the Leads arm's rule gains a game on the 384 validation games, in
points, with the first choices it changes a game in brackets:

| Update | 5 points | 10 | 15 | 20 | 30 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | +1.4 ± 0.9 (0.89) | +0.7 ± 0.4 (0.18) | +0.3 ± 0.3 (0.08) | +0.3 ± 0.2 (0.04) | +0.0 ± 0.0 (0.01) |
| 32 | **+1.5 ± 1.0 (1.11)** | +1.1 ± 0.6 (0.25) | +0.6 ± 0.3 (0.07) | +0.3 ± 0.2 (0.03) | +0.2 ± 0.2 (0.01) |
| 64 | -0.1 ± 0.9 (1.24) | +0.7 ± 0.5 (0.30) | +0.4 ± 0.3 (0.12) | +0.3 ± 0.3 (0.07) | +0.0 ± 0.0 (0.00) |
| 96 | +0.6 ± 1.4 (2.15) | +0.9 ± 0.9 (0.77) | +0.1 ± 0.7 (0.32) | +0.2 ± 0.6 (0.18) | -0.1 ± 0.3 (0.05) |
| 128 | -1.8 ± 1.6 (2.48) | +1.4 ± 0.9 (0.95) | +1.5 ± 0.6 (0.42) | +0.7 ± 0.5 (0.21) | +0.1 ± 0.3 (0.06) |

Update 32 at 5 points and update 128 at 15 points gain the same to the
last digit: 562.5 points over the 384 games, 1.46 a game, with standard
errors of 1.0 and 0.6. The rule for equal gains took the earlier update:
**update 32, 5 points**. The arm that reads only gains most at the same
place, +0.7 ± 0.9. By their own first choices the four arms win between
55.5% and 59.1% of the validation deals at the five updates, each rate
good to 2.5 points and no arm ahead of another.

### The readings

At update 32 on the 1,536 held-out deals, where the pilot wins 60.7% and
the search 72.5%. An arm's other answers are its first choices that are not
the pilot's, at the decisions the search examined, and their leads are the
search's measurements added up over a game.

| Arm | Deals won | Other answers a game | Their leads a game | By its rule: deals won | Changed a game | The rule's gain a game |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| The seed | 55.1% | 2.19 | -7.3 ± 0.6 | | | |
| Pilot | 56.6% | 1.91 | -4.5 ± 0.5 | | | |
| Search | 60.2% | 2.36 | -2.8 ± 0.6 | | | |
| Leads | 57.4% | 2.13 | -3.9 ± 0.5 | 58.0% | 1.00 | +1.9 ± 0.4 |
| Leads, reading only | 59.2% | 1.87 | -3.6 ± 0.5 | 59.4% | 1.15 | +0.5 ± 0.4 |

- *The leads are learnable*: **not shown.** The rule gains 1.9 ± 0.4
  points a game: more than four standard errors above zero, and two thirds
  of the mark of 2.9.
- *It plays better*: **not shown.** By its rule the Leads arm wins 1.4 ±
  1.2 points more of the deals than the Pilot arm, where the reading asks
  for 2.9 and for two standard errors. Against its own first choices the
  rule wins it 0.7 ± 0.9.
- *Not learnable this way* did not apply: the rule gained on the validation
  run.

### What the head knows

It knows which answers lose and not which lead. Over the 195,129
alternatives of the held-out games, which lose 7.3 points on average, the
alternatives the Leads arm's head ranks highest against the pilot's answer:

| It ranks highest | It predicts | The search measured | Reading only: predicts | measured |
| --- | ---: | ---: | ---: | ---: |
| 0.1%, 195 alternatives | +25.2 | -2.1 ± 1.2 | +22.5 | +1.0 ± 1.0 |
| 0.5% | +14.7 | -0.6 ± 0.5 | +14.1 | -0.9 ± 0.5 |
| 1% | +11.2 | -0.7 ± 0.3 | +11.0 | -0.8 ± 0.3 |
| 5% | +4.9 | -1.0 ± 0.1 | +5.2 | -0.7 ± 0.1 |
| 25% | +1.3 | -1.0 ± 0.1 | +1.5 | -0.9 ± 0.1 |

- Its predictions correlate with the measured leads at 0.49 (0.43 for the
  head that reads only, 0.06 and 0.00 for the untrained heads of the two
  other arms). The quarter it ranks highest loses 1 point where an
  alternative loses 7.3 on average: it tells the answers that cost from
  those that do not matter.
- Where it predicts that an alternative beats the pilot's answer by 11 to
  25 points, the alternative is level with the pilot's answer or a little
  behind. The validation run says the same at every update, and its
  predictions there grow with training while what is measured does not
  (+38 predicted for its highest 0.1% after 128 updates, -2.8 measured).
- **The rule's gain is then mostly its own copy repaired, by inference.**
  On the held-out games the rule at 5 points takes the arm from 2.13 other
  answers a game to 2.93, and their leads from -3.9 to -2.0 a game: it
  leaves the pilot's answer more often than it returns to it, and what it
  leaves it for is level or a little behind by the table above. The gain
  must come from the fewer cases, 318 at most of the 1,540 changes, where
  the policy's first choice was already another answer than the pilot's
  and a poor one. Followed by its head the arm still answers worse than
  its pilot by the search's measure, and loses less by it. The changes
  have since been counted apart by where they start from: 230 of the 1,540
  start at another answer, and they carry the gain
  ([After the readings](#after-the-readings-every-update-and-the-rules-changes-apart)).
- At larger margins the rule changes less and gains more for each change:
  +1.3 ± 0.3 a game at 10 points with 0.25 changes a game, +0.8 ± 0.2 at
  15 with 0.09. An untrained head read by the same rule loses 86 to 109
  points a game at 5 points: the measure does not favour a rule.
- The head that reads only what the policy has learned knows less (0.43,
  and a rule that gains +0.5 ± 0.4): predicting leads shapes what the
  policy sees a little, and that helps the head.

### How much of a first-stage lead is real

On the 192 validation games played on the development machine, 24,301
alternatives, from how each did world by world:

- A lead over 8 worlds has a standard deviation of 17.8 points over the
  alternatives, and 12.3 of sampling. About half of its variance is real
  (0.52, and the same from 4 worlds against the other 4). The head
  explains a quarter on the held-out games, nearly half of what there is.
- What is real is on the losing side. Half of the alternatives are level
  with the pilot's answer in every world, a third are behind, and a tenth
  are ahead: 13 a game. An alternative one world ahead in four is 3.7 ±
  0.4 points behind in the four others, one two worlds ahead is level
  there (+0.2 ± 1.2), and only three or four worlds ahead keep a lead
  (+9.7 ± 3.6).
- The search takes its two best alternatives of the first stage on to more
  worlds, 10.7 alternatives a game. Those were 18.0 points ahead in the
  first stage and are 2.3 behind in the worlds that follow: one world ahead
  of eight becomes -3.9 ± 0.3, two -1.8 ± 0.6, three +3.9 ± 1.2 and more
  +16.8 ± 2.7. The 1.6 a game the search then takes are 15.3 ahead in
  those later worlds, the others 5.4 behind.

So of the 13 alternatives a game whose target says they lead, one or two
do. A head trained on these targets is given about ten leads that are not
there for each one that is.

### What qualifies the readings

- **The Search arm won most at this update**: 60.2% of the held-out deals
  against the Pilot arm's 56.6%, +3.5 ± 1.0, and its other answers are
  worth 1.7 ± 0.6 points a game more than the Pilot arm's. The validation
  run does not show it (-1.0 ± 2.0 at update 32, -0.6 over the five
  updates), the two runs together give +2.6 ± 0.9 at update 32, and the
  earlier experiment found +0.5 ± 1.1. It is one checkpoint of each arm on
  one set of deals and was not a reading. Measured at the other updates
  since: it was the largest of five gaps, which together give +1.5 ± 0.6
  ([After the readings](#after-the-readings-every-update-and-the-rules-changes-apart)).
- The policy of the arm that reads only, which imitates the pilot's answers
  on the searched games, wins 59.2%, +2.6 ± 0.9 over the Pilot arm, with
  the same reserve; the Leads arm's policy, the same with the lead loss,
  +0.7 ± 1.0. Over the five updates neither wins more than the Pilot arm.
- **At the search's deviations** the Search arm gives the search's answer
  at 5.1% of the 2,357 on the held-out games, the Pilot arm at 2.2%, the
  Leads arm at 2.8%. It is low for every kind of change, the commonest
  among them: where the pilot passes a chain window and the search
  activates, 420 deviations, 2.9% against 1.0%. That a kind of deviation
  is frequent does not make its cases recognisable.

### After the readings: every update, and the rule's changes apart

Measured on 2026-10-08 by
[`search-leads-more.sh`](commands/search-leads-more.sh) as of `88ee6343`,
40 minutes on the CUDA machine: every arm on the 1,536 held-out deals
after each of the five updates. The deals of the readings come out the
same, and none of the readings changes: what the rule gains at another
update is no second reading. The files are in
`analysis/search-leads/arms/more/`.

**Deals won** by each arm's own first choices, and in brackets against the
Pilot arm at the same update, deal by deal:

| Update | Pilot | Search | Leads | Leads, reading only |
| ---: | ---: | ---: | ---: | ---: |
| 16 | 58.6% | 59.0% (+0.5 ± 1.0) | 57.2% (-1.4 ± 1.0) | 57.6% (-1.0 ± 1.0) |
| 32 | 56.6% | 60.2% (+3.5 ± 1.0) | 57.4% (+0.7 ± 1.0) | 59.2% (+2.6 ± 0.9) |
| 64 | 58.4% | 60.0% (+1.6 ± 1.0) | 58.4% (+0.0 ± 0.9) | 57.9% (-0.5 ± 0.9) |
| 96 | 58.2% | 60.2% (+2.0 ± 1.1) | 59.6% (+1.4 ± 1.0) | 58.4% (+0.2 ± 1.0) |
| 128 | 58.9% | 59.0% (+0.1 ± 1.1) | 58.8% (-0.1 ± 1.0) | 59.6% (+0.7 ± 0.9) |
| All five | 58.2% | 59.7% (+1.5 ± 0.6) | 58.3% (+0.1 ± 0.5) | 58.5% (+0.4 ± 0.5) |

Over the five updates a deal's difference is averaged first, so the error
is the deals' alone; each arm is one training run.

- **The Search arm's +3.5 was its largest gap and the Pilot arm's lowest
  rate.** Over the five updates it wins 1.5 ± 0.6 points more of the
  held-out deals than the Pilot arm. The validation deals give -0.6 ± 1.2,
  and the 1,920 deals of the two runs together +1.1 ± 0.5. It stays a
  point under the pilot itself, 60.7%: it takes back part of what a copy
  loses to its pilot and nothing of what the search gains.
- **The two arms with a head win what the Pilot arm wins**: +0.2 ± 0.5 and
  +0.2 ± 0.4 over the two runs. The +2.6 of the arm that reads only was
  that update.
- By the search's measurements the Search arm's other answers are worth
  more than the Pilot arm's through the first four passes and no more
  after: +3.0, +1.7, +2.2, +0.2 and -0.8 points a game at the five updates
  (± 0.6 to 0.7), as in the earlier experiment. Those of the arm that
  reads only are worth +0.9 ± 0.3 more over the five, the Leads arm's
  +0.1 ± 0.3.
- At the 2,357 held-out deviations the Search arm gives the search's
  answer at 5.2, 5.1, 6.6, 8.0 and 7.6%, the Pilot arm at 1.7 to 2.2%. At
  the deviations of the games it trains on: 5.1, 6.9, 12.4 and 28.9% in
  the first, second, fourth and eighth pass.

**The Leads arm's rule at every update**: what it gains a game on the
held-out games, with the first choices it changes a game in brackets.

| Update | 5 points | 10 | 15 | 20 | 30 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | +1.2 ± 0.5 (0.87) | +0.8 ± 0.2 (0.17) | +0.4 ± 0.2 (0.07) | +0.3 ± 0.1 (0.03) | +0.2 ± 0.1 (0.01) |
| 32 | **+1.9 ± 0.4 (1.00)** | +1.3 ± 0.3 (0.25) | +0.8 ± 0.2 (0.09) | +0.3 ± 0.1 (0.03) | +0.2 ± 0.1 (0.01) |
| 64 | +0.9 ± 0.6 (1.12) | +0.9 ± 0.3 (0.31) | +0.4 ± 0.2 (0.11) | +0.3 ± 0.1 (0.05) | +0.1 ± 0.0 (0.01) |
| 96 | -0.6 ± 0.7 (2.06) | +0.7 ± 0.4 (0.72) | +0.9 ± 0.3 (0.29) | +0.4 ± 0.2 (0.13) | +0.1 ± 0.1 (0.03) |
| 128 | -3.6 ± 0.8 (2.26) | -0.5 ± 0.5 (0.86) | +0.7 ± 0.4 (0.40) | +0.6 ± 0.3 (0.20) | +0.2 ± 0.1 (0.05) |

No update and margin reaches the mark of 2.9; the choice made on the
validation games is the best of them. The point that tied with it there,
update 128 at 15 points, gains 0.7 ± 0.4. With more passes the rule at 5
points changes twice as much and loses: the head predicts leads on the
games it has seen that are not there on others.

**The changes apart by where they start**, the Leads arm at update 32:
how many, how many of them are ahead of, level with and behind the first
choice they replace, and what they gain a game.

| Margin | From the pilot's answer | Back to the pilot's answer | From one other answer to another |
| ---: | ---: | ---: | ---: |
| 5 | 1,310 (251 / 824 / 235): +0.4 ± 0.4 | 88 (49 / 29 / 10): +0.7 ± 0.1 | 142 (63 / 43 / 36): +0.8 ± 0.2 |
| 10 | 282 (60 / 182 / 40): +0.3 ± 0.2 | 37 (27 / 8 / 2): +0.4 ± 0.1 | 69 (38 / 18 / 13): +0.5 ± 0.2 |
| 15 | 78 (22 / 44 / 12): +0.2 ± 0.1 | 18 (12 / 4 / 2): +0.2 ± 0.1 | 38 (22 / 9 / 7): +0.3 ± 0.1 |
| 20 | 23 (7 / 13 / 3): +0.1 ± 0.1 | 7 (4 / 1 / 2): +0.0 ± 0.0 | 19 (10 / 3 / 6): +0.2 ± 0.1 |

- **The rule's gain is the policy's own departures repaired: measured.**
  Of its 1.9 points a game, 1.5 come from the 230 changes that start at
  another answer than the pilot's. The 1,310 that leave the pilot's answer
  gain 0.4 ± 0.4: what they go to is ahead 251 times and behind 235.
- It is so at every update. Leaving the pilot's answer gains -0.8 ± 0.4,
  +0.4 ± 0.4, -0.6 ± 0.5, -1.7 ± 0.6 and -4.8 ± 0.8 points a game at the
  five updates, the changes from another answer +2.0, +1.5, +1.5, +1.1 and
  +1.1. For the arm that reads only, -0.8 to -1.4 and +0.9 to +1.6.
- Where the head predicts 10 points or more over the pilot's answer, the
  answer is ahead more often than behind at update 32 (60 against 40, and
  22 against 12 from 15 points). At the other four updates it is not: 23
  against 32, 82 against 80, 258 against 257, 298 against 367.

**More games of this kind would add little to this head.** Its fit before
each update of the first pass is on games it has not yet seen (see
[The arms in training](#the-arms-in-training)): 0.18 after the first 480
games, 0.22 over the rest of the pass's first quarter, then 0.22, 0.25 and
0.25 over the three quarters that remain. It has most of what it gets
after a fifth of the games. On the games it has seen the fit then rises to
0.46 in the eighth pass, while the correlation on the validation games
goes from 0.46 to 0.44. The head that reads only: 0.19 to 0.22 in the
first pass and 0.25 in the eighth.

The value head, beside the question: in the first pass it explains -0.06
to -0.08 of the results of games new to it in the Pilot, the Search and
the reading-only arm, and +0.06 in the Leads arm, the one whose shared
parameters also carry the lead loss; in the eighth pass 0.64 to 0.73 on
games seen. As in the
[seed's run](2026-10-05-search-imitation.md#what-the-run-shows), it fits
the results of the games in front of it and takes nothing to others.

## Conclusion

1. *Does a network trained on the leads tell which answers do better than
   the pilot's, and which worse?* Which do worse, yes: its head ranks the
   alternatives of games it has not seen at a correlation of 0.49 with
   what the search measures, where the noise of 8 worlds allows about
   0.72. Which do better, no: the alternatives it rates highest against
   the pilot's answer are level with it, and where its rule leaves the
   pilot's answer it goes to a better answer as often as to a worse one.
2. *Does a player that follows the head win more?* Not shown: 58.0% of the
   held-out deals by its rule, against 56.6% for the arm that imitates the
   pilot (+1.4 ± 1.2), 60.2% for the one that imitates the search's
   answers, and 60.7% for the pilot itself. Over five updates the arms
   with a head win what the arm that imitates the pilot wins, and the arm
   that imitates the search's answers 1.5 ± 0.6 points more (1.1 ± 0.5
   with the validation deals), still under the pilot.
3. *How much of the search's gain does it take?* By the search's
   measurements the rule gains 1.9 ± 0.4 points a game of the 11.7, under
   the mark of 2.9: 1.5 by undoing the policy's own departures from its
   pilot, and 0.4 ± 0.4 by leaving the pilot's answer. It takes nothing of
   what the search found.

Both experiments on the one-step search end at the same place. The network
learns its pilot and, with the leads, what loses. Where the pilot's answer
can be bettered it learns neither from the answers the search played nor
from the leads it measured. For the leads there is a reason in the targets
themselves: over 8 worlds, most of the thirteen leads a game are noise.
That is not the whole of it. What the head explains on games it has not
seen is nearly all there after the first 1,500 of the 7,680, and from
there on the arms mostly learn the games in front of them: the leads, the
search's answers and the results alike. In the earlier experiment eight
times the searched games left the rate at unseen deviations where it was,
at 7 to 9%. That speaks against more examples, or cleaner ones, being what
is missing first.

## What follows

Closed on 2026-10-08, with the fourth point below taken up:
[The learner itself](2026-10-08-value-head-and-board.md). The second and
third stay as they are written, neither agreed.

1. **Two measurements on what exists: done on 2026-10-08**
   ([After the readings](#after-the-readings-every-update-and-the-rules-changes-apart)).
   The Search arm's lead in deals won is 1.5 ± 0.6 points over the five
   updates, not the 3.5 of one; the rule's gain is the policy's own
   departures repaired.

The two proposals of 2026-10-08, with what the measurements say of each,
and a third made the same day:

2. **One more arm on the same games**, whose targets for the alternatives
   the search took past the first stage are their leads in the worlds that
   follow, 24 or more that had no part in choosing them. The log holds
   those already; an hour or two on the CUDA machine. It asks whether the
   noise of the targets was what kept the head from the leads that are
   there. The measurements speak against expecting much: cleaner targets
   do what more examples do, and the head's fit on unseen games hardly
   moved from 1,500 games to 7,680. There is also a cost in the design:
   an alternative's later worlds are there only where it was ahead in the
   first stage, so a target made of them is lower, on average, the more
   often an alternative is taken further; and it changes the invariant
   that the head is taught the first stage only.
3. **The one-step search is left as a teacher for this network as it is**,
   and what it finds goes into the pilots by rules, as upstream does.
4. **The learner itself first**, proposed on 2026-10-08 and agreed that day:
   past the first fifth of the searched games the arms mostly learn the
   games in front of them, and the value head does the same with results.
   Two short imitation runs would say whether that is the network or how
   it is trained: a value head trained so that it cannot fit an update's
   results, and the board given as its cards (`individual-tokens`, which
   has no valid comparison yet) against the pooled board, each also read
   by what a lead head on it explains on unseen games.

## What had to be there first

- `policy-bench search --log full`, written and pinned on 2026-10-07
  (ygo-policies `568711d`). On 16 games it gives the answers, deviations
  and digests of a run without it and of a run with `--log true`, whose
  rows it reproduces entry for entry where an alternative was tried before.
  At the 179 decisions of those games that the pilot's answer wins
  outright, all 365 alternatives are now played through the 8 worlds: none
  is ahead, two thirds are level, and they lose 10 points on average. The
  192 plain and 32 searched reference games of the pin before come out
  identical (`analysis/pins/568711d`).
- The pin on the improved pilots, the same day: that commit on top of
  ygo-policies `75e4b37` is `1b85b63`. See [The opponents](#the-opponents).

## The tooling

Written on 2026-10-07; the commands are in
[docs/commands.md](../docs/commands.md).

- **The head.** Every answer has a predicted lead beside its score
  (`PolicyValueNet.evaluate`). Checkpoints are format 14; a format-13
  checkpoint loads with the head initialised the same way every time and
  the optimizer's state kept for the parameters there were, so every arm
  starts from the seed with the same head.
- **The targets.** `train --imitation-records ROWS --imitation-lead-weight
  W` reads the leads of the first stage from the rows' log and trains the
  head on them, at every decision the search examined; `--imitation-lead-
  detach` for the arm that reads only, and `--imitation-label pilot` for a
  policy that imitates the pilot on the searched games. An update reports
  the head's loss, its fit before the update, and the two gradients the
  calibration compares.
- **The measure.** `evaluate-search` asks the head at every decision the
  search examined and gives, for margins of 5 to 30 points, the first
  choices the rule changes, what the answers are worth and what the rule
  gains; and what the head knows: its predictions against the measured
  leads over every alternative, and the measured lead of those it ranks
  highest. With the full log a model's sum covers the decisions the pilot's
  answer wins outright. `--lead-margin M` also plays the run's deals by the
  rule.

- **The arms and the report.**
  [`search-leads-arms.sh`](commands/search-leads-arms.sh) does the whole of
  the design after the games: the calibration run and the weight, the four
  arms, the validation run at the five updates, the choice, and the
  held-out run at the choice with its deals played by the rule. It starts
  by holding the seed against its pilot on the suite. The weight and the
  choice are written once and kept, and it refuses games of other pilots
  and a folder begun from another seed. `tools/search_leads_report.py`
  applies the seed's rule, sets the weight, makes the choice and applies
  the readings as they are fixed above; the mark it computes from the
  pilot's and the search's results on the held-out deals.

Checked on this machine, on the CPU:

- The decisions the trainer takes a search to examine are the ones the
  search examined, at every one of the 5,131 examined decisions of 96
  logged games of the development seed and at none of the 2,306 others of
  the watched seat (551 in a chain, 1,231 facing a face-down monster, 524
  of another kind). A test plays two games with the bench and checks the
  same, and that every alternative of the full log is a target.
- A model resumed from a format-13 checkpoint and trained three small
  updates on 16 fully logged games (2,631 targets an update) lowers the
  head's loss from 0.031 to 0.025, the size of the noise of 8 worlds, and
  is measured and played by its rule on four other logged games. This says
  the parts work together and nothing of what a head learns.
- At the present pin both scripts run through on four deals a run: the
  games with the seed's reference, then the weight, four arms of three
  updates, the choice and the readings. A run left unfinished is taken up
  again, a second start does nothing over, and a seed made to look far
  under its pilot stops the arms before anything is trained.
- On the 192 validation deals played here at the present pin, the
  decisions the trainer takes a search to examine are again the ones the
  search examined, all 10,290, and one update on those games has the
  24,301 alternatives of the log and the pilot's answer at each of those
  decisions as its 34,591 targets.

