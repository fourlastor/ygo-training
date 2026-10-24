# Archetype-fix re-baseline

**Status:** COMPLETED

**Activated:** 2026-09-28

**Last updated:** 2026-09-29

## Why

Until `f22edb4e2`, OCGCore never received card set codes, so every
archetype-gated effect was silently disabled in every duel since 2026-07-16.
[`2026-09-28-ygo-policies-board-tokens.md`](2026-09-28-ygo-policies-board-tokens.md)
and [`2026-09-28-value-head-warmup.md`](2026-09-28-value-head-warmup.md) are
invalidated. This record re-establishes the references on the corrected
engine.

## Blocking issue found by the fix: combinatorial card selections

With archetypes working, Evigishki Mind Augus (11877465) can be Ritual Summoned.
Its effect sends OCGCore `MSG_SELECT_CARD` with 1-5 of 27 graveyard cards. The
environment expanded that single prompt into one action per subset: 101,583
actions. Serializing them exhausted memory. OCGCore itself presents one
selection prompt; the combinatorial expansion is ours
(`DecisionActions.CardSubsets`). It was decided that multi-card selections are
always presented sequentially: pick one card at a time, then finish. That is
how OCGCore's own select/unselect protocol and real clients work.

### Sequential selection (implemented)

- **Environment** (`duel/training/SequentialSelection.cs`): card, tribute and
  sum selections are presented one step at a time. Each step offers:
  - a pick for every remaining candidate that can still complete a legal
    answer (for sums, only picks from which a legal total is still
    reachable);
  - `finish` whenever the picked set is a legal answer, including the empty
    set when zero cards are allowed;
  - `cancel` when OCGCore permits it.

  A pick action names its candidate, and its members are the cards already
  picked plus the candidate, so the final pick carries the complete set.
  Every action's response is OCGCore's response for the set it would
  produce. Only the completing step (finish, or a pick that leaves nothing
  more to choose) reaches OCGCore, with the combined response. Single-card
  selections, including attack targets, are one step, identical to the
  previous encoding. Sort, counter and multi-zone place prompts are still
  enumerated. The largest such prompt seen over the suite is a 120-way
  Gishki sort.
- **Training:** each step is a decision (and counts toward the decision
  limit). The transaction tracker assigns consecutive steps of one engine
  prompt (same player and `PromptId`) to one causal transaction, so PPO
  treats the whole selection as a single joint action. Selection-level
  `cancel` stays masked for the neural policy; `finish` does not.
- **ygo-policies** (`native/ygo-policies` commit `cf8cc65`, merged into
  upstream `main`): the Seat decides these
  selections as a sequence of `SelectCards`/`SelectSum` steps with Toggle,
  Finish and Cancel choices, then answers OCGCore once. The previous subset
  enumeration truncated at 4096 subsets, smallest first, so large selections
  such as Mind Augus's never offered their larger answers. Card selections
  keep the same result as before (greedy on per-card scores equals
  maximising the summed score); sum selections are greedy over completable
  picks. The FFI answer JSON reports the pick order (`picks`). The Python
  bridge uses that order to answer each environment step and to label
  imitation steps.

Over the full suite (13 x 128 games), every game completed and every policy
answer was matched step by step. The largest prompt was 26 actions, apart
from the 120-way sort.

## Plan

1. Implement sequential card selection, with regression tests, and verify it
   against the policy benchmark.
2. Measure the `blackwing` teacher on the evaluation suite with
   `teacher_evaluation`, on the fixed opponent pool (`9e5790e`) and the
   corrected engine.
3. Run a 100-update value-warm-up baseline on that pool. Its protocol is
   frozen after steps 1-2.

Steps 1 and 2 are done. The baseline command is
[`commands/setcodes-baseline.txt`](commands/setcodes-baseline.txt). It is the
earlier value-warm-up pilot command with 100 updates and its own run
directory:

- ygo-policies `cf8cc65` and the corrected engine, not a pinned old pool;
- `mean-pool`, seed 2012, 520 duels per update, all 13 policy decks;
- 10 teacher-executed imitation updates with value weight 0.5;
- default entropy coefficient 0.01, no target-KL;
- evaluation every 10 updates on the ygo-policies screen.

Its version-9 evaluation, together with the teacher reference above, sets the
new hard subset before any PPO checkpoint is examined.

## Evidence

### Teacher reference on the corrected engine

`teacher_evaluation` on `eval_suites/ygo-policies-blackwing-screen.json`,
seed 2012, ygo-policies `cf8cc65` (the fixed pool `9e5790e` plus sequential
selection), with archetype set codes transported:

| Matchup | Teacher | Previous engine, `9e5790e` |
| --- | ---: | ---: |
| aggregate | 1060-604 (63.7%) | 71.5% |
| mirror | 58-70 (45.3%) | 52.3% |
| burn | 115-13 (89.8%) | 78.9% |
| crystal | 79-49 (61.7%) | 92.2% |
| dragunity | 79-49 (61.7%) | 97.7% |
| gishki | 99-29 (77.3%) | 99.2% |
| gladiator | 84-44 (65.6%) | 75.8% |
| heroes | 65-63 (50.8%) | 60.9% |
| infernity | 79-49 (61.7%) | 65.6% |
| lightsworn | 53-75 (41.4%) | 69.5% |
| monarch | 42-86 (32.8%) | 11.7% |
| morphtronic | 101-27 (78.9%) | 99.2% |
| rock-block | 80-48 (62.5%) | 43.8% |
| spellcaster | 126-2 (98.4%) | 82.0% |
| **hard subset (mirror, monarch, rock-block)** | 180-204 (46.9%) | 34.9% |

Archetype-dependent decks that were near-certain wins (Dragunity, Crystal
Beast, Morphtronic) are now real matchups, Lightsworn is much stronger, and
Blackwing's own archetype effects lift Monarch and Rock-Block. The
previously pre-registered hard subset was chosen on the defective engine; it
must be re-chosen from this teacher reference and a new learner's version-9
evaluation before any comparison uses it. On an older ygo-policies commit,
the independent reference harness reproduced the Monarch result on its own
seeds (37.1%).

### Teacher strength of every deck

Each deck's own policy plays the learner seat of that deck's evaluation suite
(`eval_suites/ygo-policies-<deck>-screen.json`: 13 opponents x 128 games,
seed 2012, ygo-policies `cf8cc65`, corrected engine):

| Learner deck | W-L-D (aggregate) | Mirror | Worst matchup |
| --- | --- | ---: | --- |
| monarch | 1160-499-5 (69.7%) | 46.9% | heroes 44.5% |
| heroes | 1119-545-0 (67.2%) | 56.2% | dragunity 48.4% |
| rock-block | 1088-575-1 (65.4%) | 51.6% | blackwing 43.0% |
| blackwing | 1060-604-0 (63.7%) | 45.3% | monarch 32.8% |
| lightsworn | 994-670-0 (59.7%) | 53.1% | monarch 25.0% |
| dragunity | 949-715-0 (57.0%) | 53.9% | rock-block 25.0% |
| infernity | 895-769-0 (53.8%) | 53.9% | lightsworn 34.4% |
| gladiator | 847-817-0 (50.9%) | 46.1% | monarch 20.3% |
| gishki | 718-946-0 (43.1%) | 43.0% | lightsworn 16.4% |
| morphtronic | 696-967-1 (41.8%) | 54.7% | rock-block 18.0% |
| crystal | 595-1069-0 (35.8%) | 52.3% | heroes 18.8% |
| burn | 485-1177-2 (29.1%) | 60.9% | gladiator 4.7% |
| spellcaster | 406-1258-0 (24.4%) | 48.4% | heroes 3.1% |

## Overnight sweep: strong, middle and weak decks

Three full 100-update runs were chosen instead of 13 shorter ones. They share
the Blackwing baseline protocol, except for the learner deck and its
evaluation suite:

- **Monarch** (strongest teacher, 69.7%):
  [`commands/setcodes-baseline-monarch.txt`](commands/setcodes-baseline-monarch.txt)
- **Blackwing** (upper middle, 63.7%):
  [`commands/setcodes-baseline.txt`](commands/setcodes-baseline.txt)
- **Spellcaster** (weakest teacher, 24.4%):
  [`commands/setcodes-baseline-spellcaster.txt`](commands/setcodes-baseline-spellcaster.txt)

[`commands/overnight-setcodes-baselines.sh`](commands/overnight-setcodes-baselines.sh)
runs them sequentially in that order, starting with Blackwing, continuing
past a failed run, and logging to `analysis/overnight/`. Expected duration is
about 3.4 hours per run.

Before launch, every deck's suite completed with policy seats on both sides.
The Monarch and Spellcaster learner seats' largest prompts were 37 and 24
actions. A two-update local training smoke test was run for each new learner.

These runs give one learner per strength tier on the same protocol. They are
diagnostics of how imitation and PPO behave across decks, not a controlled
comparison. Each learner is compared with its own deck's teacher row above.

### Overnight sweep result

All three runs completed 100 updates. Status output is in
`analysis/setcodes-baselines/`. Held-out win rates, each learner against its
own deck's teacher on the same suite games (128 games per matchup, 1,664 in
the aggregate; draws count half):

| Learner | Teacher | v9 (imitation) | mean v19-v99 | v99 | v99 - v9 |
| --- | ---: | ---: | ---: | ---: | ---: |
| monarch | 69.9% | 65.3% | 67.7% | 69.4% | +4.1 pp |
| blackwing | 63.7% | 50.8% | 40.5% | 39.4% | -11.4 pp |
| spellcaster | 24.4% | 13.2% | 10.6% | 8.4% | -4.8 pp |

Mirror (the opponent is the policy the student imitates):

| Learner | Teacher | v9 | v99 |
| --- | ---: | ---: | ---: |
| monarch | 48.8% | 42.6% | 36.3% |
| blackwing | 45.3% | 33.6% | 27.3% |
| spellcaster | 48.4% | 17.2% | 18.8% |

Sampled training win rate (learner rollouts, not comparable to held-out
evaluation) settled at about 50% for Monarch, 20-28% for Blackwing and 2-4%
for Spellcaster. Rollout entropy rose in all three runs, to about 1.4 nats
(Monarch, Blackwing) and 1.7 (Spellcaster).

Findings:

- **Imitation leaves every student below its teacher,** despite 93-96%
  argmax agreement at version 9. The gap is 4.6 points for Monarch, 12.9 for
  Blackwing and 11.2 for Spellcaster, where it halves the teacher's rate. The
  mirror shows it directly: the Spellcaster student wins 17% against the
  policy it copies. Imitation loss was still falling at version 9 in all runs.
- **PPO helped only the learner that keeps winning.** Monarch held about
  50% of its training games, recovered past its version-9 result by version
  29 and finished level with its teacher, gaining in Dragunity, Heroes,
  Lightsworn and Rock-Block while losing ground in the mirror and against
  Crystal. Blackwing fell in all 13 matchups relative to version 9, most
  within 20 updates, and never recovered. Spellcaster, which wins almost no
  training games, declined steadily after version 29.
- **The Blackwing decline is not specific to hard matchups.** It lost 5 to
  21 points in every matchup, so the pooled hard subset would not separate
  the effect of an intervention from the broad decline.
- The previous series (defective engine, weaker opponents, Blackwing v9 at
  66.9%) held its aggregate through 100 updates. Together with the Monarch
  run, this is consistent with PPO degrading when the learner mostly loses its
  training games, where a win/loss reward carries little signal. It is not
  established: the engine and the opponents changed between the series, and
  the three decks here differ in more than win rate.

**Hard subset:** the plan was to choose it from Blackwing's version 9 before
any PPO checkpoint was examined. Those checkpoints have now been seen, so a
subset chosen now would not be blind, and the decline is broad rather than
concentrated. Comparisons on this engine use the aggregate and the mirror
instead, per learner, against both the teacher and that learner's version 9.

**Next candidates** (to test one at a time on Blackwing):

1. Opponent weighting toward matchups the learner currently wins about half
   the time, which tests the win-rate explanation directly.
2. More imitation updates, since the version-9 gap to the teacher is larger
   than anything PPO has recovered.
3. An imitation (teacher) term kept during PPO, which limits drift where the
   reward carries little signal.

### Policy distance of the overnight runs

`policy_distance measure` on `analysis/policy-distance-states-seed-7000.pt`
(11,891 learner-seat states from 104 teacher duels, ygo-policies `cf8cc65`,
corrected engine), reference version 9, output
`analysis/setcodes-baselines/policy-distance-setcodes.json`. The states were
collected with the tool's default learner, Blackwing, so for the Monarch and
Spellcaster runs they are prompts from a deck those students never play; they
measure distance on a fixed distribution, not on each student's own play.

| Learner | KL from v9 at v19 / v99 | Agreement with v9 at v19 / v99 | Path KL at v99 | Step KL, updates 90-99 | Entropy v9 / v99 |
| --- | --- | --- | ---: | --- | --- |
| monarch | 0.45 / 0.90 | 74% / 60% | 2.50 | 0.018-0.019 | 0.46 / 0.74 |
| blackwing | 0.55 / 1.08 | 68% / 47% | 2.22 | 0.019-0.022 | 0.32 / 0.76 |
| spellcaster | 0.33 / 0.98 | 82% / 38% | 1.30 | 0.007-0.010 | 0.39 / 0.88 |

Findings:

- **Monarch and Blackwing moved about equally far from version 9,** at
  similar per-update step KL, with opposite outcomes: at version 19 Monarch
  had moved 0.45 and held its result, while Blackwing had moved 0.55 and lost
  11 points. How far the policy moves does not explain the difference; the
  direction does. This agrees with the earlier rejection of target-KL: a
  smaller step would slow the Blackwing decline, not reverse it.
- **Spellcaster's updates shrink** to about a third of the others' step KL
  late in the run, consistent with little reward signal, while its entropy
  keeps rising and agreement with version 9 falls to 38%. The policy drifts
  without a direction that wins games.

## Learning-signal diagnostic: branched action values

**Question.** With only a terminal win/loss reward, how much does one learner
decision say about the outcome, and how much of what PPO sees is noise?

**Tool.** `ygo_training/action_values.py`, since removed.
In teacher-played Blackwing duels (ygo-policies in both seats), sampled
learner decisions are branched with OCGCore arena snapshots (the
`fourlastor/ygopro-core` fork). Each neural action is tried in the exact
world and in re-dealt worlds, where `RandomizeHiddenWorld` shuffles what the
learner cannot see. The teacher then plays every branch to the end. All
actions of a state share the same worlds and engine random state, so their
outcomes are paired.

- The policy seats are rebuilt in each branch by replaying the message log
  they were fed.
- The randomizer no longer moves cards whose identity is public.
- **Validation:** at every sampled state, the exact-world branch that plays
  the teacher's own action must reproduce the live duel's winner, win reason
  and decision count. This covers snapshot restore and seat replay together.

An action's value is the learner's mean outcome (+1/-1) over re-dealt worlds
when the teacher plays the rest of the duel. It is a reference value, not the
value under the student's own play.

**Zone placement is not branched.** A first collection (325 states, all
validated) included zone placement. In 48 of 52 placement states every zone
had the same outcome in every world. In the rest, the teacher's zone
(option 0) differed and every other zone matched each other. Tracing each
diverging world to its first difference in play (ignoring zone positions)
found no zone-dependent card. Instead an opponent policy chose its target in
zone order:

| State | Placed | First difference in play |
| --- | --- | --- |
| 92, vs Lightsworn | set spell/trap | Lyla destroys Book of Moon or Seven Tools of the Bandit |
| 299, vs Blackwing | set spell/trap | Mystical Space Typhoon hits Trap Hole or Threatening Roar |
| 323, vs Gladiator | set spell/trap | Mystical Space Typhoon hits Trap Hole or Threatening Roar |
| 264, vs Dragunity | monster | Dragunity Legionnaire destroys Dragunity Dux or Blackwing - Bora the Spear |
| 313, vs Monarch | monster | none until 160 decisions later (not resolved) |

Every resolved case is a tie broken by zone order. Face-down targets have no
other discriminator, and the Legionnaire's two targets scored the same (its
target score is ATK, and Dux has 1,500 + 200 for itself = 1,700, as Bora
does). A positional pick among equals is exploitable in a way a real
opponent's guess is not. Zone sequence is also masked from the network's
state and action tensors, so the student cannot tell placement options apart
and cannot learn from their differences. `action_values` therefore skips
placement prompts (`SKIPPED_PROMPTS`).

**Run.** `collect --duels 104 --worlds 32 --sample-rate 0.05
--max-states-per-duel 4`, seed 7000, balanced over all 13 opponents,
ygo-policies `cf8cc65`, placement skipped. Output
`analysis/action-values-blackwing-seed-7000.pt`; `summary` reproduces the
figures below.

- 299 states, **all 299 validated**, 31,851 playouts.
- 37 states were inside an open chain, where OCGCore cannot re-deal. They have
  exact-world values only and are excluded below.
- 3 playouts needed a continuation fallback after a forced selection pick.

| Measure (262 states, 32 re-dealt worlds) | Result |
| --- | --- |
| outcome-locked (every action, every world, same outcome) | 72 (27%) |
| outcome variance explained by the action, mean / median / 90th pct | 3.6% / 1.6% / 9.9% |
| teacher's action vs the worst action, paired sign test p < 0.05 | 38 (15%) |
| best action (chosen on even worlds) minus teacher (odd worlds) | +0.016 +- 0.010 |
| alternatives with any value difference from the teacher's | 419 of 700 |
| single-duel outcomes needed to see that difference at 2 sigma, quartiles | 20 / 63 / 231 |

| Prompt | States | Outcome-locked | Median value spread |
| --- | ---: | ---: | ---: |
| chain response | 120 | 43 | 0.0625 |
| idle (main phase) | 91 | 12 | 0.25 |
| battle | 37 | 11 | 0.125 |

Findings:

- **A single decision carries very little of the duel outcome.** On average
  the action chosen explains under 4% of outcome variance; the rest is the
  rest of the duel, the opponent and the draws. A typical nonzero difference
  between two actions needs about 60 independent duel outcomes of that exact
  state-action to see it, while PPO sees each once. This quantifies why
  win/loss credit is dominated by noise, in won and lost duels alike.
- **Most decisions barely matter.** Even with 32 paired worlds only 15% of
  states separate the teacher's action from the worst one. Main-phase
  decisions are where choices matter most.
- **The teacher is close to locally optimal under its own continuation.** The
  best single deviation gains +0.016 +- 0.010 (under one win-rate point),
  not significant. The branched values mostly say how much each deviation
  from the teacher costs, not how to beat the teacher: a one-step deviation
  followed by teacher play cannot show gains that need the student's own
  follow-up.

Caveats: states come from teacher-played duels, not the student's; values
assume teacher continuation and a uniform re-deal of concealed cards; one
deck.

**Next:** score the baseline's v9 and v19 on these states
(`action_values score`), to test whether the plays PPO changed between them
lost value.

## Random tie-breaking in ygo-policies

Following the placement traces, ygo-policies (`main` at
`2cd3c4a`) breaks ties between equally scored choices at random instead of in
engine order, which follows zone order. This applies to card and target
selections, attack targets and zone placement. Placement, which no card in the
pool depends on, is now uniform.

- The generator is seeded per seat (`ygo_policy_create_seeded`). The Python
  bridge seeds each seat from the duel seed and player, so a duel replays
  exactly and different duels vary.
- A 26-duel `action_values` smoke run validated all 70 states. Rebuilt seats
  therefore make the same random choices as the live ones.

Teacher strength on the same suites and seeds:

| Learner deck | `cf8cc65` | Random ties (`2cd3c4a`), W-L-D | Change |
| --- | ---: | --- | ---: |
| monarch | 69.9% | 1136-523-5 (68.4%) | -1.4 pp |
| heroes | 67.2% | 1137-527-0 (68.3%) | +1.1 pp |
| rock-block | 65.4% | 1106-558-0 (66.5%) | +1.1 pp |
| blackwing | 63.7% | 1033-631-0 (62.1%) | -1.6 pp |
| lightsworn | 59.7% | 1007-657-0 (60.5%) | +0.8 pp |
| dragunity | 57.0% | 948-716-0 (57.0%) | -0.1 pp |
| infernity | 53.8% | 903-761-0 (54.3%) | +0.5 pp |
| gladiator | 50.9% | 866-798-0 (52.0%) | +1.1 pp |
| gishki | 43.1% | 728-936-0 (43.8%) | +0.6 pp |
| morphtronic | 41.9% | 677-986-1 (40.7%) | -1.1 pp |
| crystal | 35.8% | 584-1080-0 (35.1%) | -0.7 pp |
| burn | 29.2% | 468-1194-2 (28.2%) | -1.0 pp |
| spellcaster | 24.4% | 390-1274-0 (23.4%) | -1.0 pp |

Aggregates move by at most 1.6 points, about one standard error of a
1,664-game aggregate, and the strength order is unchanged. Single matchups
move by up to 11 points (Blackwing against Heroes, Gishki against
Gladiator); at 128 games per matchup (standard error about 4.4 points), most
of this is expected from changed game trajectories. The earlier teacher and
learner results were measured on `cf8cc65`. Comparisons with a new run use
the table above.

The action-value diagnostic was re-run on `2cd3c4a` with the same command.
`analysis/action-values-blackwing-seed-7000.pt` now holds this run:

- 314 states, all 314 validated; 40 were exact-only;
- the action explains 3.6% of outcome variance on average (median 1.6%);
- 61 of 274 states are outcome-locked;
- 39 separate the teacher's action from the worst one;
- best minus teacher is +0.007 +- 0.012;
- a typical difference needs 63 single-duel outcomes to detect (median).

The conclusions above are unchanged.

### Where PPO lost value: baseline checkpoints on the branched states

`action_values score` of the Blackwing baseline
(`runs/ygo-policies-setcodes-baseline-seed-2012`) on the 274 re-dealt states
of the `2cd3c4a` values file, run on the CUDA machine. Values are the mean
payoff (+1/-1) of the checkpoint's argmax action followed by teacher play;
+-1 standard error over states.

| | v9 | v19 | v99 |
| --- | ---: | ---: | ---: |
| agreement with the teacher | 92.0% | 63.1% | 39.4% |
| minus teacher, all states | +0.003 +- 0.004 | -0.031 +- 0.011 | -0.037 +- 0.012 |
| minus teacher, where it departs | +0.037 +- 0.050 (22) | -0.083 +- 0.029 (101) | -0.061 +- 0.019 (166) |

Value change on the states where the choice changed:

| Prompt | v9 to v19 | v19 to v99 |
| --- | --- | --- |
| all | **-0.101 +- 0.030** (91) | -0.016 +- 0.018 (110) |
| chain response | **-0.172 +- 0.034** (54 of 126) | +0.002 +- 0.029 (54) |
| idle (main phase) | -0.003 +- 0.061 (25 of 92) | -0.022 +- 0.028 (43) |
| battle | +0.086 +- 0.125 (8 of 40) | -0.083 +- 0.037 (12) |

Findings:

- **Imitation's departures from the teacher were harmless** (+0.037 +-
  0.050 over 22 states). The version-9 gap to the teacher is not visible on
  teacher states; it must come from states the student reaches itself.
- **The early PPO damage is in chain responses.** Between versions 9 and 19,
  PPO changed 54 of 126 chain decisions, and each change cost 0.17 payoff
  (about 9 win-rate points for that decision, with teacher continuation), five
  standard errors. Main-phase changes were neutral.
- **Later drift is mostly neutral.** From version 19 to 99 agreement with
  the teacher fell from 63% to 39%, but the changed choices cost -0.016 +-
  0.018 overall. Battle changes may have cost value (-0.083 +- 0.037, 12
  states), which is weak evidence.

Caveat: these are teacher-distribution states and teacher continuations, so
the values measure each decision in isolation, not the student's play as a
whole.

**Direction of the chain-response changes** (`score` transition breakdown,
same run). Almost every changed chain response is the student activating
where the teacher passes:

| | v19 | v99 |
| --- | --- | --- |
| chain `pass -> activate`, of all chain departures | 54 of 56 | 86 of 86 |
| value change | -0.177 +- 0.034 | -0.111 +- 0.024 |
| Icarus Attack | 24 | 30 |
| Book of Moon | 21 | 28 |
| Blackwing - Kalut the Moon Shadow | 6 | 8 |
| Threatening Roar | 1 | 11 |
| Mystical Space Typhoon | - | 5 |

PPO learned to fire its flexible responses (destruction, a position change,
an ATK boost, an attack stop) at the first window instead of holding them as
the teacher does. At version 99 it also often ends the turn instead of setting
a Spell/Trap (10 states, -0.081 +- 0.068).

Hypothesis, not yet tested: firing a response changes the board at once, so
the critic's value rises at the next step and the activation gets a positive
advantage. The cost of spending the card, having no answer when a real threat
arrives, shows up many decisions later, beyond GAE's effective horizon
(about 20 decisions at lambda 0.95), and is spread across everything in
between. Credit would then systematically favor immediate board changes over
holding a card.

### Critic credit for early chain responses

**Question.** PPO learned to activate responses where the teacher passes
(previous section). Does the credit PPO receives favor those activations?

**Tool.** `ygo_training/critic_credit.py`, since removed.
At chain windows of teacher-played Blackwing duels where the teacher passes
and the learner could activate a card, pass and every activation are branched
in 8 re-dealt worlds. The learner's observation is captured 1, 2, 4, ... 256
learner decisions later. `score` evaluates a checkpoint's critic there and
approximates the GAE credit (the lambda-return of those critic values) that
activating gets over passing.

- **Teacher continuation:** the teacher plays both sides of every branch. In
  the exact world, the pass branch reproduced the live duel's result and all
  captured observations for all 266 states.
- **Student continuation (`--continuation`):** a checkpoint plays the
  learner's side as in PPO rollouts: sampled at temperature 1, joint-staged,
  with common random numbers across a world's actions.

Run: `collect --worlds 8`, seed 7000, 104 duels, 266 states, 4,848 branches
per file. Activation minus pass, +-1 standard error over states:

| Continuation | Pass-branch mean payoff | True outcome difference |
| --- | ---: | ---: |
| teacher | - | -0.139 +- 0.023 |
| v9 student | -0.198 | **-0.119 +- 0.022** |
| v19 student | -0.515 | -0.012 +- 0.020 |

GAE credit implied by each critic (activation minus pass):

| Critic, continuation | lambda 0.9 | 0.95 | 0.99 | 0.999 |
| --- | ---: | ---: | ---: | ---: |
| v9, teacher | -0.075 | -0.089 | -0.112 | -0.136 |
| v9, v9 student | -0.092 | **-0.119** | -0.151 | -0.128 |
| v19, teacher | -0.023 | -0.021 | -0.039 | -0.118 |

(Standard errors 0.007-0.021.)

Findings:

- **At the start of PPO, both the true returns and the critic's credit said
  pass.** Under v9's own play, firing early costs -0.119, and the v9 critic's
  GAE credit at lambda 0.95 is -0.119 +- 0.015. It is correctly signed and the
  right size. Neither misdirected credit nor a student unable to use a held
  card explains the drift. The earlier hypotheses (a myopic critic, then a
  continuation mechanism) are rejected.
- **After the drift the incentive is gone.** Under v19's play, firing early
  costs nothing (-0.012 +- 0.020), and the v19 critic barely separates the
  branches. Once the policy has drifted, the drift is self-consistent: its
  follow-up play no longer gets value from a held card, and the pass-branch
  payoff has fallen from -0.20 to -0.52 from the same positions.
- On these states, GAE at lambda 0.95 lost little of the credit relative to
  lambda 0.999. The concern that early decisions lose their credit applies
  when the critic is poor. The v9 critic was good here; the v19 critic was not.

**Open question.** The expected policy gradient at version 9 points toward
passing, yet the first ten updates moved toward activating. The next check is
which update moved it (score v10 to v18), and then whether a term of the
update other than the advantage-weighted policy gradient drives it: the
entropy bonus, or shared-parameter side effects of other decisions'
gradients.

**Which updates moved the chain responses** (`action_values score`, v9 to v19
on the 274 states; `analysis/action-values-score-v9-v19.json`):

| Version | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| agreement with teacher | 92% | 87% | 83% | 81% | 82% | 80% | 78% | 77% | 69% | 62% | 63% |
| chain `pass -> activate` states | 2 | 5 | 12 | 14 | 14 | 18 | 21 | 19 | 42 | 59 | 54 |
| value minus teacher | +0.003 | 0.000 | -0.002 | -0.005 | -0.002 | -0.013 | -0.019 | -0.021 | -0.026 | -0.032 | -0.031 |

The drift is not a first-update jump. Premature activations accumulate over
every update and roughly double at updates 17-18. That matches the steady
rise in policy entropy (rollout entropy 0.50 at update 10 to 1.11 at update
18; on the fixed policy-distance states 0.32 to 0.71). A flattening policy
flips argmax choices in bursts once the dominant `pass` falls below the best
activation. Each changed chain response costs value throughout (-0.15 to
-0.22 per departure from version 12 on).

## Shared warm-up seed for PPO branches

The 10 teacher-executed imitation updates are identical for every run with
the same seed and teacher. PPO interventions now branch from one warm-up
checkpoint instead of repeating them, which halves the cost of a 20-version
run and makes branches comparable version for version.

- **Warm-up seed:** `runs/ygo-policies-setcodes-baseline-seed-2012/checkpoints/model-00009.pt`
  on the CUDA machine (the Blackwing baseline's version 9: 10 imitation
  updates with value weight 0.5, ygo-policies `cf8cc65`).
- A branch resumes it with `--resume` and drops the imitation flags. Its
  versions continue at 10, and its rollout seeds continue from step 10.
  `TrainerConfig` now allows the ygo-policies teacher profile with zero
  imitation updates for this purpose.
- The seed was imitated from the `cf8cc65` teacher, while branches now play
  `2cd3c4a` opponents. Both arms of a comparison share that, but the
  baseline's own versions 10-99 were played against `cf8cc65`. Each
  comparison therefore runs its own control branch rather than reusing the
  baseline.

**First branches: entropy bonus.** The chain-response drift tracks rising
policy entropy (above). [`commands/setcodes-branch-control.txt`](commands/setcodes-branch-control.txt)
(default entropy coefficient 0.01) and
[`commands/setcodes-branch-entropy0.txt`](commands/setcodes-branch-entropy0.txt)
(`--entropy-coefficient 0`) run 10 PPO updates each from the warm-up seed.
Both are scored with `action_values score` on v10-v19. If the entropy bonus
drives the drift, chain `pass -> activate` departures stay near version 9's
level without it and grow as before in the control.

### Entropy branches: result

Both branches ran 10 PPO updates from the warm-up seed against `2cd3c4a`
opponents. `action_values score` on the 274 states
(`analysis/action-values-score-branch-{control,entropy0}.json`):

| | Baseline (`cf8cc65` opponents) | Control branch | Entropy-0 branch |
| --- | ---: | ---: | ---: |
| v19 agreement with teacher | 63.1% | 79.9% | 73.0% |
| v19 chain `pass -> activate` states | 54 | 13 | 25 |
| v19 value minus teacher | -0.031 +- 0.011 | -0.002 +- 0.007 | -0.006 +- 0.008 |
| v19 evaluation (1,664 games) | 39.9% | 49.0% | 43.0% |

Findings:

- **The entropy bonus does not drive the drift.** Without it, premature
  activations still grew (2 at v9 to 25 at v19), more than in the control,
  and the evaluation was lower. Rejected.
- **The control drifted much less than the baseline** over the same updates
  from the same checkpoint: 13 against 54 premature activations, and 49.0%
  against 39.9% at v19. The two differ in the opponents' tie-breaking
  (random `2cd3c4a` against zone-order `cf8cc65`) and in run-to-run
  randomness. With one run of each, the cause cannot be separated. The size
  of run-to-run variation in this protocol is unknown, so no single-run
  comparison between arms is conclusive either.

**Next:** measure run-to-run variance. Replicate branches must vary only the
training randomness, but `--seed` also sets the evaluation games, so a
separate seed for training rollouts is needed first.

### Run-to-run variance: replicate control branches

`train --training-seed` sets the training randomness (rollout duels and
opponents, action sampling, minibatch order) separately from `--seed`, which
still sets the evaluation games. It defaults to `--seed`, so existing commands
are unchanged. `run.json` records both.

Three more control branches from the warm-up seed differ from the control
only in `--training-seed` (2013, 2014, 2015):
[`commands/branch-control-replicates.sh`](commands/branch-control-replicates.sh)
runs them in sequence. Together with the control (training seed 2012), they
give four replicates of the same protocol. Their spread at v19 (evaluation,
premature chain activations, agreement with the teacher) sets the smallest
difference a single-run comparison can detect.

### Replicate results

Four control branches (training seeds 2012-2015) and the entropy-0 branch at
v19. The decision measures come from `action_values score` on the 274 states
(`analysis/action-values-score-branch-*.json`):

| v19 | Control replicates, mean +- SD | Entropy 0 | Baseline (`cf8cc65` opponents) |
| --- | --- | ---: | ---: |
| evaluation, 1,664 games | 47.2% +- 3.1 (49.0, 44.7, 44.4, 50.7) | 43.0% | 39.9% |
| agreement with teacher | 80.0% +- 1.2 | 73.0% | 63.1% |
| chain `pass -> activate` states | 13.8 +- 1.5 (13, 16, 13, 13) | 25 | 54 |
| value minus teacher | -0.002 +- 0.001 | -0.006 | -0.031 |

Findings:

- **Run-to-run variation in win rate is large.** The standard deviation of
  3.1 points is 2.5 times the evaluation's binomial noise (1.2). A single-run
  comparison cannot detect differences under about 9 points; four runs per
  arm bring that to about 4-5.
- **Decision measures are precise.** Premature activations and teacher
  agreement vary little between replicates. These are the sensitive
  comparison between arms.
- **The entropy bonus slows the drift.** Without it, v19 has 25 premature
  activations (about seven SDs above the controls) and 73% agreement (six
  SDs below). This reverses the entropy hypothesis above.
- **The baseline's chain drift mostly came from its opponents.** Its 54
  premature activations are far outside the replicate range. Its main
  difference from the branches is training against zone-order `cf8cc65`
  opponents, which apparently rewarded firing responses early. It was also
  one continuous run rather than resumed, but resuming restores the same
  model and optimizer state. Against the current opponents, the controls lose
  no measurable value on these states (-0.002).

**Open:** the controls' v19 average (47.2%) is below the baseline's v9
(50.8%), but that v9 was evaluated against `cf8cc65` opponents. v9 must be
re-evaluated against `2cd3c4a` to know whether PPO still loses strength.

### Version 9 against the current opponents

`evaluate-suite` of the warm-up seed (baseline v9) on the Blackwing suite,
seed 2012, against `2cd3c4a` opponents: **839-825 (50.4%)**, against 50.8% on
`cf8cc65`. Per matchup, the two opponent versions differ by at most 4.7
points (Monarch 20.3% to 25.0%, Crystal 60.2% to 55.5%).

The four control replicates reached 47.2% +- 3.1 at v19 on the same games.
Ten PPO updates therefore changed the evaluation by about -3.2 +- 2.0 points
(replicate standard error 1.55, plus the evaluation's binomial noise 1.2): a
possible small decline, not significant.

**Conclusion.** Against the current opponents, the first ten PPO updates
are roughly neutral. The baseline's collapse (50.8% to 39.9% by v19, and 54
premature chain activations) did not reproduce in any of four replicates.
It came mainly from training against zone-order `cf8cc65` opponents, possibly
with run-to-run variance on top. The baseline's longer-run results (to v99)
carry the same confound. PPO also shows no improvement over its imitation
start, consistent with the learning-signal diagnostic (one decision explains
under 4% of outcome variance).

## Long control from the warm-up seed

Everything past version 19 so far was measured against `cf8cc65` opponents.
[`commands/setcodes-branch-control-long.txt`](commands/setcodes-branch-control-long.txt)
continues the control protocol for 90 PPO updates from the warm-up seed
(versions 10-99) against `2cd3c4a` opponents, evaluating every 10 versions.
Its v19 is directly comparable with the four replicates, and its later
versions show whether PPO eventually improves on or degrades from the
imitation start. It is scored with `action_values score` at every tenth
version.

Speed: the NumPy tensorizer reduced an update from about 115 s to 85 s (the
optimization step from about 55 s to 33 s) without changing results. Two
further attempts did not help the rollout, which is bound by the parent
process's serial collect-and-infer loop: 48 workers (45.1 s against 47.9 s;
also changes sampling seeds) and overlapping inference on a thread
(reproduced the fingerprints, 49.0 s; reverted, as collection and inference
contend for the GIL).

### Teacher strength against `2cd3c4a`

The same 13 suites and seed as the `cf8cc65` table, with random tie-breaking
(`python -m ygo_training.teacher_evaluation --suite
eval_suites/ygo-policies-<deck>-screen.json`, CPU only, about 18 s per
suite with 20 workers):

| Learner deck | W-L-D | Win rate | `cf8cc65` | Mirror | Worst matchup |
| --- | --- | ---: | ---: | ---: | --- |
| heroes | 1137-527-0 | 68.3% | 67.2% | 52.3% | rock-block 55.5% |
| monarch | 1136-523-5 | 68.3% | 69.7% | 41.4% | heroes 43.8% |
| rock-block | 1106-558-0 | 66.5% | 65.4% | 50.0% | blackwing 42.2% |
| blackwing | 1033-631-0 | 62.1% | 63.7% | 44.5% | monarch 25.8% |
| lightsworn | 1007-657-0 | 60.5% | 59.7% | 52.3% | monarch 28.1% |
| dragunity | 948-716-0 | 57.0% | 57.0% | 55.5% | rock-block 23.4% |
| infernity | 903-761-0 | 54.3% | 53.8% | 61.7% | lightsworn 32.8% |
| gladiator | 866-798-0 | 52.0% | 50.9% | 46.1% | monarch 23.4% |
| gishki | 728-936-0 | 43.8% | 43.1% | 42.2% | lightsworn 18.0% |
| morphtronic | 677-986-1 | 40.7% | 41.8% | 57.0% | lightsworn 17.2% |
| crystal | 584-1080-0 | 35.1% | 35.8% | 50.8% | heroes 15.6% |
| burn | 468-1194-2 | 28.1% | 29.1% | 63.3% | gladiator 3.1% |
| spellcaster | 390-1274-0 | 23.4% | 24.4% | 51.6% | heroes 3.9% |

Every deck moved by at most 1.6 points (standard error about 1.2), and the
order is unchanged except that Heroes and Monarch are now tied at the top.
Tie-breaking did not change the relative strength of the teachers; the
Blackwing teacher's reference on its suite is 62.1%.

### Long control branch: PPO degrades steadily

`setcodes-branch-control-long` reproduced the control branch exactly at v19
(49.0%), so the NumPy tensorizer did not change results over a full run.
Held-out suite win rate then fell almost monotonically: v19 49.0%, v29 45.1%,
v39 41.9%, v49 35.5%, v59 37.5%, v69 38.5%, v79 37.4%, v89 34.3%, v99 35.5%
(v9 50.4%; teacher 62.1%). Entropy rose from 0.50 to about 1.45 by v50 and
stayed there; sampled training win rate fell from 41% to about 22%.

`action_values score` on the 274 re-dealt Blackwing states
(`analysis/action-values-score-control-long.json`):

| Version | Agreement | Value minus teacher | Chain pass -> activate (states, value) | Battle departures |
| --- | ---: | ---: | --- | --- |
| v9 | 92.0% | +0.003 ± 0.004 | 2, -0.188 | 4, -0.078 |
| v19 | 79.9% | -0.002 ± 0.007 | 13, -0.139 | 10, +0.006 |
| v49 | 53.3% | -0.021 ± 0.010 | 50, -0.130 | 19, -0.059 |
| v79 | 46.4% | -0.038 ± 0.012 | 69, -0.120 | 20, -0.062 |
| v99 | 33.2% | -0.055 ± 0.012 | 98, -0.125 | 23, -0.136 |

The loss is dominated by one transition: the learner activates chain effects
where the teacher passes. It grows from 2 to 98 of the 126 valued chain
states, each costing about 0.125 in expected payoff, and never self-corrects
despite being measurably worse at every version. Idle departures stay about
neutral; battle departures turn negative late. PPO from the imitation start
does not improve on it against `2cd3c4a` opponents: v9 remains the best
Blackwing checkpoint, and more updates only move the policy away from the
teacher in the direction the branched values say is wrong.

Distribution shape on the 120 valued chain states where the teacher passes
(`action_values score` `policy_shape`, `analysis/action-values-shape-control-long.json`;
2.2 options on average, so uniform is 0.47):

| Version | P(pass) | P(best activation) | Normalized entropy | Median log ratio |
| --- | ---: | ---: | ---: | ---: |
| v9 | 0.969 | 0.028 | 0.08 | -10.76 |
| v19 | 0.765 | 0.206 | 0.58 | -1.78 |
| v49 | 0.550 | 0.400 | 0.79 | -0.24 |
| v69 | 0.463 | 0.481 | 0.82 | +0.16 |
| v89 | 0.369 | 0.566 | 0.79 | +0.73 |
| v99 | 0.294 | 0.636 | 0.73 | +1.09 |

Two phases. Until about v49 the near-deterministic imitation policy diffuses
toward uniform (entropy 0.08 to 0.79). After it, activation probability
keeps rising past uniform while entropy falls again: PPO is actively
learning to prefer activating, although the branched values say each such
activation costs about 0.125. The same directed pattern appears for Idle
states where the teacher sets a spell or trap (P(teacher) 0.78 to 0.23,
value-neutral) and, more weakly, Battle states where it enters Main Phase 2.
The late chain drift is therefore a credit signal, not only diffusion, and
warrants measuring the advantages PPO assigns to chain pass and activation.

### Chain credit under PPO

[`commands/chain-credit-probe.sh`](commands/chain-credit-probe.sh) resumed
the long branch at v49 and v89 for three updates each, logging the
advantages given to learner chain passes and activations
(`analysis/chain-credit-probe-v{49,89}-metrics.jsonl`). Both probes
reproduced the long branch's sample counts and training win rates exactly.

| Updates | Chain samples, pass / activate | Raw advantage, activate minus pass | Critic value, pass / activate | Outcome, pass / activate |
| --- | --- | --- | --- | --- |
| v50-52 | 8,447 / 6,259 | +0.012, +0.001, +0.007 (mean +0.007) | -0.51 / -0.59 | -0.47 / -0.55 |
| v90-92 | 5,774 / 6,719 | -0.039, -0.028, +0.001 (mean -0.022) | -0.38 / -0.47 | -0.44 / -0.55 |

(Per-sample advantage SD is about 0.4. Within-duel correlation puts the
effective sample near the 520 duels per update, so each update's difference
has a standard error of roughly 0.02-0.03.)

- **The critic does not credit activations with wins they did not cause.**
  Activations occur in worse games (outcome about 0.1 lower), and the critic
  predicts about the same gap for both groups.
- **The chain advantage does not favor activating.** It is flat at v50-52
  and favors passing at v90-92, yet the policy moved further toward
  activating from v89 to v99 (P(pass) on the teacher-pass states 0.37 to
  0.29). This repeats the version 9 finding: the chain-step policy gradient
  points toward passing or nowhere while the policy moves toward activating.
- The comparison pools all chain states, including those where the teacher
  activates, and compares different states rather than the same state. It
  cannot rule out a within-state bias confined to the teacher-pass subset.

With the entropy bonus also rejected (it slows the drift), the remaining
candidate is a side effect of shared parameters. The drift would then come
from other terms of the update (other decisions' policy gradients, the value
loss through the shared trunk) rather than from the chain credit itself.

**Next: which term of the update moves the chain decisions.**
`python -m ygo_training.update_attribution` resumes a checkpoint exactly as
`train --resume` does and measures one update on probes over the fixed
branched states: the log probability of passing where the teacher passes,
and the expected branched value by prompt. The terms are the policy term
split by the decision rooting each transaction (chain pass, chain
activation, idle, battle, other), the value loss and the entropy bonus.

- A first-order prediction projects each term's gradient, scaled as Adam
  scales it over the first epoch's exact minibatches and clipping, onto the
  probes' gradients. A first version held Adam's second moments fixed at
  the checkpoint's values; on the three checkpoints its predictions were
  10-50 times the actual changes (and of the wrong sign at v49), so it now
  advances them step by step.
- Exact ablations repeat the real update from the same rollout and the
  restored model and optimizer without one term at a time. They need no
  linearization, but removing a term also changes clipping and Adam's
  scaling of the others.
- Every update (the production one, its first epoch alone, each ablation)
  also reports the probes' change along its actual parameter displacement,
  which tests whether the probes are linear over one update.

Locally, the production arm matched a `train --resume` update exactly and a
restored repeat matched it bit for bit.
[`commands/update-attribution.sh`](commands/update-attribution.sh) runs it
from v9 (the first update of the drift) and v49 and v89 of the long branch.

### Update attribution: result

`update_attribution` on the first update from v9, v49 and v89
(`analysis/update-attribution-v{9,49,89}.json`). Each production arm
reproduced the long branch's update exactly. The probe is the mean log
probability of passing on the 120 teacher-pass chain states; from v9 to v89
it fell by 0.0154 per update on average.

| Update | Production | First epoch only | Without chain policy | Without idle policy | Without battle policy | Without value loss | Without entropy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| v9 to v10 | +0.005 | -0.022 | -1.476 | -0.019 | -0.006 | -0.012 | -0.019 |
| v49 to v50 | -0.019 | -0.063 | -0.027 | -0.150 | -0.052 | -0.111 | -0.058 |
| v89 to v90 | +0.027 | +0.030 | +0.013 | +0.008 | +0.083 | +0.027 | +0.016 |

- **The first-order prediction fails validation**, even with evolving second
  moments: first-epoch predictions of +0.29, +0.28 and +1.11 against actual
  -0.02, -0.06 and +0.03. At v49 and v89 the probes are close to linear over
  an epoch (their change along the actual displacement is -0.072 and +0.032
  against -0.063 and +0.030), but the predicted first-epoch displacement
  agrees poorly with the actual one (cosine 0.3-0.6). Gradients at the
  starting parameters do not predict where an epoch of Adam steps goes.
- **The ablations are exact but not additive.** At v49 the single-term
  effects (production minus ablation) sum to +0.30 while the production
  update moved -0.02. Nearly every minibatch is clipped (114-204 of
  128-204), and Adam scales each coordinate independently, so removing any
  term changes how far the others move.
- **One update is too noisy for the drift.** The production changes were
  +0.005, -0.019 and +0.027 around a mean of -0.015, and single-term
  effects change sign between checkpoints (battle policy +0.033 at v49,
  -0.056 at v89).
- **The chain credit anchors a near-deterministic policy.** At v9, without
  the chain samples' own policy gradient, one update moves the probe from
  -0.04 to -1.52 (approximate KL 0.24 against 0.08). The rest of the update
  pushes the chain states hard toward activating, and only the chain credit
  holds them. Whether that push comes from other decisions through shared
  parameters or from the entropy bonus on those states (amplified by Adam
  once nothing else acts on them) would need another arm.
- **The entropy bonus pushes toward passing** at every checkpoint (+0.024,
  +0.039, +0.010), consistent with the entropy-0 branch drifting further.

Single-update measurements cannot identify what drives the chain drift. It
is a slow accumulation, smaller than the update-to-update variation, of
strongly interacting terms acting through shared parameters, held back only
where a decision's own credit is strong.

**Open.** Remedies either strengthen the per-decision signal (branched
values as credit for chain decisions) or anchor the policy (persistent
teacher imitation, as the attack-transaction experiment used at 0.10; the
retention KL toward a checkpoint does not yet support joint-staged
transactions). Multi-update ablation branches scored with `action_values`
(chain `pass -> activate` has an SD of 1.5 across replicates at v19) could
still isolate the driver.

## Conclusion

- **Teacher references** on the engine with set codes: every ygo-policies
  deck's strength on its own screen, first against zone-order `cf8cc65`
  opponents and then against tie-breaking `2cd3c4a` ones (at most 1.6 points
  apart). The Blackwing teacher's reference is 62.1%.
- **Blackwing PPO degrades from its imitation seed.** Against `2cd3c4a`
  opponents, v9 plays at 50.4%; the long control branch falls to 35.5% by
  v99, and teacher agreement falls from 92% to 33%. The damage is dominated
  by chain activations where the teacher passes (2 to 98 of 126 chain
  states, each about 0.125 of payoff). Monarch held its teacher level.
- **Not the cause:** the critic's calibration, the chain credit itself, or
  the entropy bonus (which slows the drift). The drift is a slow
  accumulation of strongly interacting terms acting through shared
  parameters, held back only where a decision's own credit is strong.
- **Tools** now available: branched action values from OCGCore snapshots
  with re-dealt hidden worlds (`action_values`), critic credit
  (`critic_credit`), policy shape by decision, chain credit metrics, and
  update attribution with exact ablations (`update_attribution`).

Continued in [PPO regression remedies](2026-09-29-ppo-regression-remedies.md).
