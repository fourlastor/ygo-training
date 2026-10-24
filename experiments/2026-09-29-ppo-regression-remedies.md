# PPO regression remedies

**Status:** COMPLETED

**Activated:** 2026-09-29

**Completed:** 2026-10-06

**Last updated:** 2026-10-06

## Why

The [archetype-fix re-baseline](2026-09-28-archetype-fix-rebaseline.md)
found that PPO from the Blackwing imitation seed degrades against
ygo-policies `2cd3c4a` opponents: v9 plays at 50.4% (teacher 62.1%), the
long control branch falls to 35.5% by v99, and teacher agreement falls from
92% to 33%. The damage is dominated by chain activations where the teacher
passes. The critic is calibrated, the pooled chain advantages do not favor
activating, and no single term of an update drives the drift: it is a slow
accumulation through shared parameters, held back only where a decision's
own credit is strong.

The RL literature describes this failure and remedies that this protocol
does not use:

- **Forgetting in RL fine-tuning.** A pretrained policy deteriorates on
  states where fine-tuning gives it little signal (Wołczyk et al., ICML
  2024). Established practice anchors it: AlphaStar kept a KL penalty toward
  its supervised policy, RLHF toward the supervised model; kickstarting keeps
  an auxiliary loss toward the teacher.
- **Covariate shift of behavior cloning.** Imitating only the teacher's own
  states compounds errors once the learner leaves them. With an expert that
  can label any state, DAgger (Ross et al., 2011) trains on the learner's
  own states instead.
- **Expert cost-to-go as credit.** AggreVaTe(D) and THOR train on the value
  of each action under the expert's continuation, which is what the branched
  teacher-continuation values measure.
- **Critic design.** Card-game agents train a critic with privileged hidden
  information (PerfectDou, AlphaStar, Suphx); the unbiased form conditions on
  history and state (Baisero & Amato, 2022). Shared policy and value
  networks interfere (Phasic Policy Gradient; Andrychowicz et al., 2021).
- **Update size.** PPO's clipping does not enforce a trust region (Engstrom
  et al., 2020); practice stops at a KL near 0.01-0.02, and ygo-agent uses
  one epoch. Here an update takes 500-800 Adam steps and reaches KL
  0.04-0.08.
- **Credit along long lines.** UPGO (AlphaStar, ygo-agent) and expected
  traces from an action-value critic (Fan & Farina, 2026).
- **Scale.** ygo-agent trained self-play PPO on over 100M games; a 100-update
  run here uses about 50,000.

## Hypotheses (phase 1)

- **H1, covariate shift.** Part of v9's 12-point gap to the teacher comes
  from imitating teacher-visited states only. Imitating teacher labels of the
  learner's own states (DAgger, no PPO) raises held-out strength above v9.
- **H2, forgetting.** An imitation anchor on the learner's own states stops
  the regression: PPO with the anchor holds or improves on v9 where the
  control declines.

## Protocol (phase 1)

- **Teacher labels on learner states.** With `--teacher-imitation-weight`
  above zero, the learner deck's own ygo-policies policy watches the learner
  seat in PPO rollouts and answers each learner decision without playing it.
  Its answer, mapped to the neural action list, labels the step. Each seat
  breaks ties with its own stream, so the games are unchanged: a local
  one-update run with labels reproduced every rollout metric of the run
  without. Answers that are not neural actions (a masked `cancel`) or that
  the learner already departed from within a sequential selection get no
  label (0.3% of learner decisions locally).
- **Metrics.** `teacher_label_coverage`, `teacher_sampled_agreement` and
  `teacher_argmax_agreement` (the rollout model's argmax against the labels,
  before the update; `teacher agree` in `status --ppo`).
- **`--policy-loss-weight`** scales the PPO policy term (default 1, results
  unchanged). Zero with teacher imitation is DAgger; the critic still trains.
- **Arms**, each 20 updates (v10-v29) resumed from the warm-up seed
  (`runs/ygo-policies-setcodes-baseline-seed-2012/checkpoints/model-00009.pt`)
  with the long control branch's protocol otherwise (training seed 2012,
  520 duels, 4 epochs, evaluation every 10 updates on the Blackwing screen):

  | Arm | Flags |
  | --- | --- |
  | DAgger | `--teacher-imitation-weight 1 --policy-loss-weight 0 --entropy-coefficient 0` |
  | Anchor 0.1 | `--teacher-imitation-weight 0.1` |
  | Anchor 1 | `--teacher-imitation-weight 1` |

  The control is the long control branch (v19 49.0%, v29 45.1%) and its
  replicates at v19 (47.2% +- 3.1).
- **Measures.** Held-out evaluation at v19 and v29 (1,664 games; replicate
  SD 3.1 points); `action_values score` on the 274 branched states (teacher
  agreement, value minus teacher, chain `pass -> activate`; replicate SD 1.5
  at v19); training argmax agreement with the teacher.
- **Command:**
  [`commands/regression-remedies-phase1.sh`](commands/regression-remedies-phase1.sh).

## Later phases

2. A separate critic with privileged training-time inputs (hidden cards),
   conditioned on history and state.
3. AggreVaTe-style training from branched teacher-continuation values,
   starting with chain decisions.
4. Update size and credit: target KL, fewer epochs, rollout temperature,
   UPGO.

A null result from any of these says something only on a learner whose
teacher is known to be beatable. Both teachers are: a search on top of each
wins 16 to 18 points more (see the floor under the ceiling below).

## Evidence

### Phase 1 result

All three arms ran 20 updates. Their first rollout matched the control's
(52,095 samples, 41.2% training wins), so the teacher seat left the games
unchanged on the CUDA machine as well. Held-out evaluation on the Blackwing
suite (1,664 games, the same games for every checkpoint; standard error 1.2
points; v9 50.4%, teacher 62.1%):

| Arm | v19 | v29 |
| --- | ---: | ---: |
| Control (long branch) | 49.0% | 45.1% |
| DAgger | 54.4% | 60.0% |
| Anchor 0.1 | 57.1% | 58.5% |
| Anchor 1 | 57.6% | 60.3% |

The sampled rollout and update at v29 (every arm began at v10 with entropy
0.50-0.52 and 81.2% agreement):

| Arm | Training wins | Entropy | Samples | Step KL | Argmax agreement with teacher labels |
| --- | ---: | ---: | ---: | ---: | ---: |
| Control (long branch) | 29.4% | 1.23 | 35,864 | 0.035 | not measured |
| DAgger | 59.6% | 0.35 | 59,024 | 0.061 | 85.4% |
| Anchor 0.1 | 38.7% | 0.76 | 45,922 | 0.030 | 77.9% |
| Anchor 1 | 56.3% | 0.38 | 56,456 | 0.041 | 85.1% |

`action_values score` on the 274 teacher-visited states
(`analysis/action-values-score-remedies-phase1.json`):

| Checkpoint | Agreement | Value minus teacher | Chain `pass -> activate` states | P(pass) where the teacher passes |
| --- | ---: | ---: | ---: | ---: |
| v9 | 92.0% | +0.003 ± 0.004 | 2 | 0.969 |
| Control v19 | 79.9% | -0.002 ± 0.007 | 13 | 0.765 |
| Control v29 | 69.7% | -0.018 ± 0.009 | 33 | 0.636 |
| DAgger v19 | 93.4% | +0.002 ± 0.003 | 1 | 0.983 |
| DAgger v29 | 93.1% | +0.002 ± 0.003 | 2 | 0.987 |
| Anchor 0.1 v19 | 85.4% | +0.001 ± 0.006 | 5 | 0.940 |
| Anchor 0.1 v29 | 84.7% | +0.004 ± 0.006 | 2 | 0.896 |
| Anchor 1 v19 | 90.5% | +0.007 ± 0.006 | 0 | 0.980 |
| Anchor 1 v29 | 93.4% | +0.001 ± 0.003 | 1 | 0.983 |

Findings:

- **Every arm beats v9 and the control.** At v29 the arms are 8.1 to 9.9
  points above v9 (a difference of two evaluations has a standard error of
  about 1.7) and 13.4 to 15.2 above the control. At v19 they are 2.3 to 3.4
  replicate SDs above the control replicates' mean (47.2% ± 3.1). DAgger and
  Anchor 1 at v29 are about 2 points below the teacher, which is not
  significant.
- **H1's prediction holds; its cause is not isolated.** DAgger, with no
  policy-gradient term, gained 9.6 points in 20 updates. v9's argmax agrees
  with the teacher on 81.2% of its own sampled states, against 92.0% of the
  274 teacher-visited states (the warm-ups reported 93-96% on the `cf8cc65`
  teacher's states). DAgger raised the first to 85.4% and left the second
  unchanged. But the warm-up's imitation loss was still falling at v9, and
  the labels come from the `2cd3c4a` teacher while v9 imitated `cf8cc65`. The
  same number of teacher-executed imitation updates might have gained as
  much. That control has not been run.
- **H2 holds over 20 updates.** Both anchored arms improve where the control
  declines, and premature chain activations stay at 0 to 5 states (control
  13, then 33).
- **No evidence that PPO adds strength.** The three arms at v29 span 58.5%
  to 60.3%, no more than the evaluation's own noise. Anchor 1 ends next to
  DAgger in evaluation, entropy and agreement, so at weight 1 the imitation
  term dominates the update.
- **A weight of 0.1 slows the diffusion and does not stop it.** Entropy rose
  from 0.52 to 0.76 (control: 0.50 to 1.23), P(pass) on the teacher-pass
  chain states fell from 0.969 to 0.896, agreement on its own states fell
  from 81.2% to 77.9%, and sampled training wins returned to 38.7% after
  peaking at 48.7% (v21). Its argmax policy plays 20 points above its
  sampled one. Whether the argmax holds over more updates is untested.
- **The 274-state score does not resolve these gains.** v9 and all six arm
  checkpoints score within 0.007 of the teacher (standard errors 0.003 to
  0.006) while differing by up to 10 points held out. v9's whole gap to the
  teacher is 0.23 of payoff per game, about 0.002 per decision over the
  roughly 100 learner decisions of a game, which is below the score's
  resolution. Agreement on teacher-visited states does not order strength
  either: v9 has 92.0% at 50.4%, Anchor 0.1 at v29 has 84.7% at 58.5%. The
  score still detects the control's regression and the chain drift.

## Phase 1b: imitation control and longer arms

- **Imitation control.** 20 more teacher-executed imitation updates from the
  warm-up seed (v10-v29): the warm-up's own recipe (`--imitation-only-updates
  20`, value weight 0.5) against the current opponents and teacher. A
  `--resume` branch starts its schedule at zero, so all 20 are imitation
  updates; for the same reason it cannot be continued in place. It differs
  from DAgger in whose states are labeled (and in the critic's targets). Its
  first update also measures v9's agreement on the current teacher's own
  states, to compare with 81.2% on v9's.
  - Within about 3 points of v9 at v29 (the control replicates' SD): the
    learner-state labels closed the gap, as H1 says.
  - Within about 3 points of DAgger: v9 was undertrained, and covariate
    shift is not shown.
  - In between: both contribute.
- **Longer arms.** The three arms continue in place for 30 updates
  (v30-v59), evaluated at v39, v49 and v59. Locally, `--continue` reproduced
  a single invocation's rollout fingerprints and update metrics exactly.
  - DAgger's plateau is the reference for imitation alone.
  - PPO adds strength if an anchored arm's mean over v39-v59 exceeds
    DAgger's by more than about 4 points, or an arm exceeds the teacher
    (above about 65.5%, two standard errors of a difference). A miss means
    PPO left known headroom unused: a search on top of the teacher wins 17.7
    points more (see the floor under the ceiling below). On 2026-10-03 this
    reading was first withdrawn (the updated pilot is no better than the
    teacher) and then restored with that result, all before any phase 1b
    result was seen.
  - Anchor 0.1 is unstable if it falls back toward the control while its
    entropy keeps rising.
- **Command:**
  [`commands/regression-remedies-phase1b.sh`](commands/regression-remedies-phase1b.sh),
  which ends with
  [`commands/regression-remedies-report.sh`](commands/regression-remedies-report.sh):
  each run's metrics and status table, and `action_values score` of every
  evaluated checkpoint, under `analysis/remedies/`.

### Phase 1b result

All four runs completed (`analysis/remedies/`). Held-out evaluation on the
Blackwing suite (v9 50.4%, teacher 62.1%; the long control fell to 41.9% at
v39 and 37.5% at v59):

| Arm | v19 | v29 | v39 | v49 | v59 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Imitation control | 54.7% | 58.8% | | | |
| DAgger | 54.4% | 60.0% | 60.9% | 61.2% | 60.5% |
| Anchor 0.1 | 57.1% | 58.5% | 59.5% | 59.9% | 62.0% |
| Anchor 1 | 57.6% | 60.3% | 60.8% | 61.0% | 60.6% |

The last update of each run, and `action_values score` of its checkpoint on
the 274 teacher-visited states:

| Arm | Training wins | Entropy | Argmax agreement with the teacher on the rollout's states | Agreement on the 274 states | Value minus teacher | P(pass) where the teacher passes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Imitation control, v29 | 66.7% (the teacher plays) | | 86.2% (the teacher's states) | 92.3% | +0.001 ± 0.004 | 0.996 |
| DAgger, v59 | 61.2% | 0.31 | 86.6% | 93.1% | +0.002 ± 0.004 | 0.994 |
| Anchor 0.1, v59 | 46.2% | 0.71 | 78.4% | 86.9% | +0.011 ± 0.006 | 0.931 |
| Anchor 1, v59 | 58.3% | 0.32 | 86.5% | 93.1% | +0.001 ± 0.003 | 0.987 |

- **The imitation control matches DAgger** (58.8% against 60.0% at v29,
  54.7% against 54.4% at v19, within 3 points). By the reading set above,
  v9 was undertrained: the gain is more imitation, and covariate shift is
  not shown. H1's prediction held and its cause did not. The control's
  first update also measured v9 on the current teacher's own states: 84.2%
  agreement, against 81.2% on v9's own sampled states, a gap of 3 points and
  not the 11 the 274 states suggested.
- **Every arm levels off at the teacher's level and none passes it.** Means
  over v39 to v59: DAgger 60.9%, Anchor 0.1 60.4%, Anchor 1 60.8%, one to
  two points under the teacher's 62.1%. The best single evaluation is 62.0%
  (Anchor 0.1 at v59). Per matchup the arms are within a few points of the
  teacher everywhere; the largest deficits are against Gladiator (5 to 9
  points) and Morphtronic (4 to 9).
- **PPO adds no measurable strength.** The anchored arms differ from DAgger
  by under half a point, where the reading asked for 4, and no arm
  approaches 65.5%. The search shows what was there to find: against the 12
  other decks the arms win 62 to 64% and the strict search 80%.
- **H2 holds over 50 updates, and a weight of 0.1 is enough.** Anchor 0.1's
  diffusion stopped: entropy went from 0.76 at v29 to 0.71 at v59, P(pass)
  on the teacher-pass chain states from 0.896 back to 0.931, and sampled
  training wins from 38.7% to 46-51%. Premature chain activations stay at 2
  to 7 states.

## Best known pilots against the training opponents

ygo-policies `0cf005a` (2026-10-03) holds improved pilots: effects no longer
land on a pilot's own cards, shared card knowledge is generated from the
engine, and every list is legal under the WC2011 Forbidden & Limited list.
Its tier list (`DECK-TIER-LIST.md`, 31 decks, 256 games per pair) shows what
each deck reaches when its policy pilots it. That is the best pilot known,
not the deck's real ceiling: a stronger player or a search may top it. It bounds the
ceiling from below. The list pits updated pilots against updated pilots on
updated lists, while training still uses `2cd3c4a`.

For the same reference on this protocol, `teacher_evaluation
--learner-library` plays the learner seat with another build of the policies
library and leaves the opponents on the checkout's build. Each updated pilot
then meets the `2cd3c4a` opponents on its own suite, with the checkout's deck
lists and the games of the teacher reference
([`commands/updated-pilot-ceilings.sh`](commands/updated-pilot-ceilings.sh),
`analysis/updated-pilots-0cf005a/`). A second copy of the `2cd3c4a` build on
the learner seat reproduced the teacher reference exactly.

| Deck | Training teacher (`2cd3c4a`) | Updated pilot (`0cf005a`) | Known headroom | Updated pilot against the old one |
| --- | --- | --- | ---: | ---: |
| monarch | 1136-523-5 (68.4%) | 1263-401-0 (75.9%) | +7.5 | 57.0% |
| crystal | 584-1080-0 (35.1%) | 656-1008-0 (39.4%) | +4.3 | 56.2% |
| spellcaster | 390-1274-0 (23.4%) | 442-1222-0 (26.6%) | +3.1 | 57.0% |
| dragunity | 948-716-0 (57.0%) | 989-675-0 (59.4%) | +2.5 | 54.7% |
| lightsworn | 1007-657-0 (60.5%) | 1041-623-0 (62.6%) | +2.0 | 54.7% |
| gladiator | 866-798-0 (52.0%) | 896-768-0 (53.8%) | +1.8 | 46.9% |
| gishki | 728-936-0 (43.8%) | 746-918-0 (44.8%) | +1.1 | 40.6% |
| heroes | 1137-527-0 (68.3%) | 1152-512-0 (69.2%) | +0.9 | 53.9% |
| infernity | 903-761-0 (54.3%) | 913-751-0 (54.9%) | +0.6 | 62.5% |
| burn | 468-1194-2 (28.2%) | 470-1192-2 (28.3%) | +0.1 | 60.2% |
| morphtronic | 677-986-1 (40.7%) | 678-985-1 (40.8%) | +0.1 | 57.8% |
| blackwing | 1033-631-0 (62.1%) | 1034-630-0 (62.1%) | +0.1 | 43.8% |
| rock-block | 1106-558-0 (66.5%) | 1096-568-0 (65.9%) | -0.6 | 50.0% |

(Draws count half. The last column is the suite's mirror case, 128 games. Two
independent evaluations differ with a standard error of about 1.7 points;
these share their games, so the true error is smaller. Dragunity, Gladiator,
Infernity and Lightsworn still list Goyo Guardian here. It is forbidden in
WC2011, so the updated lists and pilots dropped it, and these four rows may
understate the updated pilot.)

Findings:

- **The updated Blackwing pilot is no better than the teacher.** It wins
  1034 of the 1,664 games where the training teacher wins 1033. The tier
  list agrees: Blackwing takes 63.8% of its 12 non-mirror matchups there,
  against 63.5% for the training teacher here. The phase 1 arms at v29
  (58.5% to 60.3%) are 2 to 4 points below the teacher. That no better pilot
  has been written does not make the teacher the ceiling: the search below
  plays the deck 17.7 points better.
- **Monarch has 7.5 points of known headroom** (4.8 standard errors even
  unpaired), Crystal 4.3 and Spellcaster 3.1. The other ten decks are within
  noise. The upstream change that stops effects from landing on a pilot's
  own cards accounts for most of Monarch's (reported there as 69.0% to
  75.6%): Tribute Summons into an empty field, and Ignition effects fired in
  chain windows against its own cards. These are decisions a learner can
  correct.
- **PPO has not used that headroom.** The
  [earlier Monarch run](2026-09-28-archetype-fix-rebaseline.md) recovered
  from its warm-up seed to its teacher's level and stopped there (69.4% at
  v99 against a 69.9% teacher, `cf8cc65` opponents).

Monarch's teacher is the one a better hand-written pilot is known for. With
the updated policies on the training machine that difference goes away. The
search below shows headroom above every pilot measured, updated or not.

## A floor under the ceiling: search on top of a pilot

A search can raise the lower bound above the pilots without anyone writing a
better one. `policy-bench search` (ygo-policies, branch `search-player`) plays
every game twice on the same seed: the pilot alone, and the pilot with a
one-step search at each of its decisions.

- **The search.** The duel is snapshotted (the fork's arena snapshots). Every
  alternative the policy listed is tried in worlds where the cards the seat
  cannot see are dealt again (`YGO_DuelSwapHiddenCards`): its own Deck in
  another order, the other player's Deck, hand and Set Spells and Traps among
  themselves. Both policies play each world out, rebuilt from what the live
  seats were fed. Alternatives share their worlds and are compared world by
  world. The seat leaves its pilot's answer only for an alternative ahead by
  a paired z of 1.645 after 96 worlds; alternatives that are not ahead are
  dropped after 8 and 32.
- **What is searched.** Main and Battle Phase actions, optional chain
  responses, yes/no and battle positions. Not the zone a card goes to, nor
  card selections, which stay the pilot's; two copies of a card in the same
  pile count once.
- **What the search knows that the seat does not.** The other player's Deck
  list and policy, as a learner trained against that deck does. Nothing
  about hidden cards in the strict runs: decisions in an open chain (where
  cards cannot be dealt again) and decisions taken while the other player
  has a face-down monster (which cannot be dealt again either) are the
  pilot's, and each world breaks ties its own way. A coin or a die would
  land as in the duel.
- **This protocol.** The runs marked training use this checkout's engine
  build, pilots (`2cd3c4a`), deck lists, scripts and card database. Plain
  games of the benchmark then reproduce the references above: over 3,072
  paired games against the 12 other decks, Monarch wins 70.6% and its updated
  pilot 76.8%; Blackwing wins 64.1% with either pilot. The two script loaders
  search different folders (the benchmark adds `pre-errata`, the training
  environment `unofficial` and `skill`), but every card of the 13 lists
  resolves to the same file; across all 33 updated lists only Red-Eyes
  Darkness Metal Dragon in Draconic Might differs.
- **Checks.** With `--validate`, every decision of the seat is played out
  from its snapshot in the world as it is: all 3,560 decisions of 16 Monarch
  games ended as the duels did, and the duels kept the digests of the plain
  games. A diagnostic build confirmed that a deal changes the hidden cards
  and keeps each player's cards the same (the other player's five-card hand
  changed entirely), and that the alternatives of a decision share a world
  (`analysis/policy-search/validation/`).
- **Command:**
  [`commands/policy-search-floor.sh`](commands/policy-search-floor.sh), then
  `tools/policy_search_summary.py`; rows and summary in
  `analysis/policy-search/`.

### Runs

Each run plays the deck against the 12 other decks of the pool. In the Pilots
and lists column, training is this checkout (`2cd3c4a` pilots and lists on
both sides); updated is ygo-policies `0cf005a` on both sides, with its lists,
on the same engine build. The gain is paired by game.

| Deck | Pilots and lists | Search | Games | Pilot | With search | Gain |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Monarch | training | strict | 768 | 71.1% | 86.8% | +15.8 ± 1.6 |
| Blackwing | training | strict | 768 | 62.4% | 80.1% | +17.7 ± 1.7 |
| Blackwing | updated | strict | 576 | 62.0% | 78.8% | +16.8 ± 1.8 |
| Monarch | updated | strict | 384 | 75.3% | 87.8% | +12.5 ± 2.2 |
| Monarch | training | not strict | 384 | 70.1% | 87.5% | +17.4 ± 2.4 |
| Blackwing | training | not strict | 384 | 62.8% | 86.5% | +23.7 ± 2.4 |
| Blackwing | training | strict, twice the worlds | 384 | 62.8% | 79.7% | +16.9 ± 2.4 |
| Monarch | training | strict, twice the worlds | 384 | 70.1% | 90.6% | +20.6 ± 2.3 |

The search leaves the pilot's answer in 3% (Blackwing) to 5% (Monarch) of
the decisions it examines, and in three quarters of the games at least
once.

### Monarch: the check with a known answer

The training pilot against the 12 other decks, 64 games each
(`analysis/policy-search/monarch-old-strict.jsonl`):

| Monarch against the 12 other decks | Win rate |
| --- | ---: |
| Training pilot (`2cd3c4a`), these 768 games | 71.1% |
| Updated pilot (`0cf005a`), 3,072 games on the benchmark's seeds | 76.8% |
| Training pilot with the strict search, the same 768 games | 86.8% |

- **The search gains 15.8 ± 1.6 points** over the pilot on the same games
  (150 turned into wins, 29 into losses), more than twice what the updated
  pilot gains (6.1). It gains against every deck but Spellcaster, which the
  pilot already beats in all 64 games, and most against Crystal and
  Gladiator (+25), Heroes (+23) and Infernity (+22).
- It searched 35 decisions a game and left 5 in an open chain and 4 facing a
  face-down monster to the pilot. It left the pilot's answer 1.9 times a
  game. A searched game took 2,260 playouts (54 s on one core); none failed.
- **It finds the mistakes the updated pilot fixed by hand, and more.** Of
  the 1,466 deviations, 13% turn an effect fired in a chain window or an
  accepted optional trigger into a pass (Swap Frog, Treeborn Frog, Dupe
  Frog, Battle Fader: 116 passes, 80 refusals). Four fifths are Main Phase
  choices: another monster Summoned or Set (38%), another action (20%),
  something more done before ending the turn or entering the Battle Phase
  (15%), or stopping earlier (8%).

### Blackwing: the learner's own deck

The training pilot, which is the teacher of this experiment, against the 12
other decks, 64 games each
(`analysis/policy-search/blackwing-old-strict.jsonl`):

| Blackwing against the 12 other decks | Win rate |
| --- | ---: |
| Training pilot (`2cd3c4a`), these 768 games | 62.4% |
| Training and updated pilot, 3,072 games on the benchmark's seeds | 64.1% both |
| Training pilot with the strict search, the same 768 games | 80.1% |

- **The search gains 17.7 ± 1.7 points** (168 games turned into wins, 32
  into losses), against every deck: most against Heroes (+36), Dragunity
  (+30) and Crystal (+23), least against Monarch (+6, from 32.8% to 39.1%)
  and against Burn and Spellcaster, which the pilot already beats.
- It searched 59 decisions a game and left 6 in an open chain and 15 facing
  a face-down monster to the pilot. It left the pilot's answer 1.8 times a
  game (2,726 playouts, 65 s on one core); none failed.
- **What it changes.** Of the 1,386 deviations, 22% activate a card in a
  chain window where the teacher passes (Icarus Attack 99, Book of Moon 94,
  Threatening Roar 47, Mystical Space Typhoon 43). 19% stop where the
  teacher goes on: the turn ends instead of another Summon, Set or
  activation (204), or the Battle Phase starts (56). 23% take another Main
  Phase action, 15% Summon or Set another monster, 6% are Battle Phase
  choices.
- The control branch drifted toward chain activations where the teacher
  passes, which lose 0.125 of payoff on average over the valued teacher-pass
  states. The search also activates there, but 0.4 times a game and only
  where the worlds favor it. It does not vindicate the drift.

### Strict and not strict

A search that is not strict also decides while the other player has a
face-down monster, and its worlds show what that monster is. Its games are
the first 32 per opponent of the strict run, so the two are compared on the
same games:

| Deck | Pilot | Strict | Not strict | Difference | Decisions the strict search left facing a set monster |
| --- | ---: | ---: | ---: | ---: | ---: |
| Monarch | 70.1% | 86.2% | 87.5% | +1.3 ± 0.7 | 4 a game |
| Blackwing | 62.8% | 80.5% | 86.5% | +6.0 ± 1.5 | 15 a game |

For Monarch the restriction costs almost nothing. For Blackwing it leaves
about a fifth of the decisions to the pilot, most against Monarch (39 a
game, 43.8% strict against 56.2%) and Lightsworn (22 a game, 56.2% against
71.9%).
A search that could deal face-down monsters again would land between the two
columns: the strict one is the clean floor, the other an upper mark for this
search.

### Twice the worlds

With 16, 64 and 192 worlds at the three stages, on the first 32 games per
opponent of the strict runs:

| Deck | Standard search | Twice the worlds | Difference |
| --- | ---: | ---: | ---: |
| Blackwing | 80.5% | 79.7% | -0.8 ± 2.0 |
| Monarch | 86.2% | 90.6% | +4.4 ± 1.4 |

Blackwing's floor does not move with the budget. Monarch's does, to 90.6%.

### What luck leaves: full foresight

`--foresight true` is not a player. It plays every alternative out in the
world as it is, with the real hidden cards, the draws to come and the duel's
own tie-breaking, in every decision including those the strict search
leaves to the pilot. It leaves the pilot's answer whenever that one loses
and another wins. On the 768 games of the strict runs:

| Deck | Pilot | Strict search | Full foresight | Lost whatever single answer changes |
| --- | ---: | ---: | ---: | ---: |
| Monarch | 71.1% | 86.8% | 97.7% | 18 games (2.3%) |
| Blackwing | 62.4% | 80.1% | 96.7% | 25 games (3.3%) |

- Foresight never loses a game the pilot wins, and every game it turns takes
  exactly one changed answer (204 for Monarch, 264 for Blackwing). The strict
  search won 3 and 6 games that no single change wins, so slightly more are
  winnable with several.
- **Luck that no play can overcome is 2 to 3% of the games.** Even
  Blackwing's worst matchup, Monarch (32.8% for the pilot, 39.1% with the
  strict search), is winnable in 89.1% of the games.
- **The rest of the gap is not all headroom.** The 17 points for Blackwing
  and 11 for Monarch between the strict search and foresight are games that
  took knowledge of hidden cards or of draws to come, or a better search.
  These runs cannot split them. For Blackwing, twice the worlds add nothing
  and the search that sees face-down monsters adds 6 points, which points to
  information rather than budget. For Monarch more worlds still help.

### Where the search's work goes

The search leaves the pilot's answer in 3 to 5% of the decisions it
examines. Whether the rest can be told apart beforehand was checked on the
Blackwing teacher: the overrides of the strict run against how often each
alternative is offered (192 traced plain games;
`tools/policy_search_offers.py`, `analysis/policy-search/offers/`), and a
run that logs every decision examined (`--log true`, 230 games, 13,476
decisions, 408 overrides;
`analysis/policy-search/blackwing-old-strict-logged-partial.jsonl`,
`tools/policy_search_decisions.py`).

- **Not by the kind of alternative.** Of 120 kinds offered (what is done
  with which card), none that comes up at least once in five games is taken
  in more than 9% of its offers, and those behind most overrides are offered
  constantly at 1 to 2% (Book of Moon and
  Icarus Attack in chain windows: 16 offers a game each, taken in under 1%;
  ending the turn: 14 offers, 2%). Keeping only the kinds taken in at least
  1% of their offers keeps 61% of the held-out overrides for 39% of the
  first-stage playouts; at 2%, 46% for 22%. Monarch is the same: 89% for
  56%, 74% for 41%.
- **A little by class.** Main Phase decisions are 29% of the decisions, 66%
  of the overrides and 51% of the playouts. Chain windows on the seat's own
  turn are 25% of the decisions and 8% of the overrides. Yes/no prompts are
  almost never overridden.
- **Modestly by situation.** A boosted-tree model on turn, phase, Life
  Points, board and hand counts and the cards involved, cross-validated by
  game, keeps 80% of the overrides in 40% of the decisions (52% of the
  first-stage playouts) and 90% in 57% (66%).
- **Exactly by the pilot's own first-stage result.** Where the pilot's
  answer wins all 8 first-stage worlds no alternative can be ahead: 42% of
  the decisions. The search now tries the pilot's answer first and skips the
  alternatives there. The 230 games are the earlier run's, answer for
  answer, on 19% fewer playouts (2,640 to 2,145 a game).
- **The rest is the price of resolution.** After that shortcut the pilot's
  first stage is 22% of the playouts, the alternatives' first stage 30% and
  the later stages 48%. 15% of the decisions go on past the first stage and
  a fifth of those end in an override; half of all overrides come from
  alternatives only one win ahead after 8 worlds.
- **Half of it lands in games the pilot wins anyway.** In the two strict
  runs of 768 games, the games the pilot alone loses are 37.6% of
  Blackwing's and 28.9% of Monarch's. They hold 49.2% and 46.0% of the
  deviations (2.4 and 3.0 a game, against 1.5 in the games it wins) and
  take 42.7% and 40.1% of the search's time. A search of those games alone
  would find 15% more deviations per CPU-second and miss half of them. The
  search wins 58.1% and 67.6% of them, and loses 6.7% and 5.3% of the games
  the pilot wins.

### The updated policies

With `0cf005a` pilots and lists on both sides, the setting of the tier list,
the search gains 16.8 ± 1.8 points for Blackwing (as much as on the training
pilot) and 12.5 ± 2.2 for Monarch (against 15.8 on the training pilot: the
hand-made fixes took a few points of the headroom and left most of it). The
tier list's rates are those of the decks as their policies pilot them, well
below what the decks can do against the same opponents.

### What the floor means

- **Both teachers are far below the ceiling against these opponents.** One
  step of search on top of the pilot wins 86.8% with Monarch and 80.1% with
  Blackwing, 16 and 18 points above the pilot. The headroom above the best
  hand-written pilots is larger than anything those pilots differ by.
- **PPO has used none of it.** The earlier Monarch run stopped at its
  teacher's level, the Blackwing control fell below its seed, and the
  anchored Blackwing arms reached the teacher's level at best. A learner that
  stops at the teacher has not hit a ceiling.
- **Everything here is against these opponents as their policies play
  them.** The playouts use the opponents' own policies, so the search is a
  best response to them. That is the right reference for this experiment,
  which trains and evaluates against them, but it is not the deck's ceiling
  against a player the search cannot model, where it would gain less.
- **Where the ceiling is.** For Blackwing against these opponents: the pilot
  wins 62%, the clean search 80%, the search that sees face-down monsters
  86.5%, and full foresight 97%. A legitimate player's ceiling lies between
  the clean search and foresight, and these runs cannot place it within
  that range. More worlds add nothing for Blackwing, so going further would
  take another search (deeper, or one that also decides card selections and
  deals face-down monsters again), not a bigger one. For Monarch: 71%, 87%
  (91% with twice the worlds), and 98%.
- **The search's answers are labels better than the teacher's.** They say
  which decisions matter (about two a game, each worth 0.25 to 0.3 of
  payoff over the worlds at the median) and what to play there. Imitating
  them is the AggreVaTe phase above with a stronger expert, and does not
  depend on the credit PPO assigns.

## Open at the end of phase 1

Training:

- The anchor labelled every learner decision in every update. Whether the
  policy holds once it is removed, or with labels on a fraction of the
  decisions, was not run. The three arms' v59 checkpoints are on the
  training machine. Phase 1c below runs both.
- Each arm is one run. The run-to-run spread of anchored arms was not
  measured (the unanchored control's was 3.1 points); the three arms within
  a point of each other from v39 on suggest it is small.
- The last one to two points between the arms and the teacher are not
  resolved: the teacher's evaluation keeps no per-game outcomes, so the
  comparison is unpaired.
- Imitation to a plateau was run for Blackwing only, and the imitation
  control stops at v29 while still improving. Monarch and Spellcaster were
  only imitated for 10 updates. Phase 1c below continues both.
- How much of the learner's 13% disagreement with the teacher on its own
  states is the teacher's random tie-breaking was not measured.
- Phases 2 to 4 above (privileged critic, branched-value credit, update
  size and UPGO) were not run.

Search:

- Only Blackwing and Monarch have search marks.
- The search leaves card selections, decisions in an open chain and (when
  strict) decisions facing a face-down monster to the pilot; the native swap
  cannot deal face-down monsters again. Full foresight changes one answer,
  and a few games take several.
- The decision analysis (first-stage result, situation model) is on
  Blackwing only; Monarch has a 53-game fragment
  (`analysis/policy-search/monarch-old-strict-logged-partial.jsonl`). The
  shortcut's exactness was checked on the same 230 Blackwing games.
- Cheaper stopping rules for the later stages, which take 48% of the
  playouts, were not evaluated. Rows with every world's result exist for 31
  games on the training Blackwing pilot and 582 on the updated one.
- The logged searches on the updated pilots for the ygo-policies agent were
  stopped: Blackwing has 582 of 768 games
  (`analysis/policy-search/blackwing-updated-logged-partial.jsonl`), Monarch
  none. They are reproducible from that repository's `main` with
  `policy-bench search --log true`.

Code:

- The training environment does not load `pre-errata` card scripts (only
  Red-Eyes Darkness Metal Dragon in Draconic Might differs today). The
  cleanup takes the tooling that runs duels from ygo-policies, which ends
  the difference.
- `train --continue` on a branch started with `--resume` restarted its
  warm-up and imitation-decay schedule at the checkpoint's version. No run
  up to phase 1b depended on it. Fixed for phase 1c, below.

## Phase 1c: the anchor removed or thinned, and imitation on other decks

Two of the open items are cheap only on the current setup: the cleanup
changes the observation format, and today's checkpoints will not load after
it. Agreed on 2026-10-04 to run before the cleanup, and run that day; the
result follows the protocol.

- **Does the 0.1 anchor have to stay?** Three branches from Anchor 0.1 at
  v59, 20 updates each (v60-v79), evaluated at v69 and v79 and scored on the
  274 states at v64, v69, v74 and v79:

  | Branch | Flags |
  | --- | --- |
  | Anchor kept | `--teacher-imitation-weight 0.1` |
  | Anchor off | `--teacher-imitation-weight 0` |
  | Anchor thinned | `--teacher-imitation-weight 0.1 --teacher-label-fraction 0.1` |

  A `--resume` branch restores the optimizer and the rollout index, so
  Anchor kept is the arm continued, and the reference for the other two.
  `--teacher-label-fraction` keeps the teacher's label on a fixed share of
  the learner's decisions, chosen from the duel seed and the decision's
  index alone. The teacher still answers every decision, so the games are
  the full anchor's: locally the thinned branch's first rollout reproduced
  every rollout metric of the kept branch, with labels on 10.8% of the
  decisions against 99.2%. The imitation term is a mean over the labelled
  transactions, so at the same weight a tenth of the labels pulls as hard
  in expectation, on a tenth of the evidence. (That last claim was wrong
  for transactions of several decisions: see the result.)

  Readings, against Anchor kept at the same version. The yardstick is the
  unanchored control from v9: after 10 and 20 updates it had lost 12 and 22
  points of agreement on the 274 states (replicate SD 1.5), 0.20 and 0.33
  of P(pass) where the teacher passes, and 1.4 and 5.3 points held out, with
  entropy up from 0.50 to 1.23.
  - *The drift resumes* if Anchor off's agreement on the 274 states is more
    than 4 points under Anchor kept's at v79 (two SDs of a difference
    between replicates), with P(pass) falling and the rollout entropy
    rising. The held-out rate lagged the agreement in the control, so a
    held-out difference is not required. The anchor then has to stay for as
    long as PPO runs.
  - *The policy holds without it* if agreement stays within 4 points and
    the held-out rate within 3.4 points (two standard errors of a
    difference) at v69 and v79. Fifty anchored updates then left a policy
    that PPO alone does not erode over 20 more.
  - *The thinned anchor holds* by the same test. If it holds where Anchor
    off drifts, the rehearsal can be sparse, which is what a costly teacher
    needs (a search label costs on the order of a second of CPU). If it
    drifts with Anchor off, the anchor needs its labels densely.
- **Does imitation distill other decks?** For the strongest and the weakest
  teacher, the warm-up seed (v9 of the
  [overnight baselines](2026-09-28-archetype-fix-rebaseline.md)) is evaluated
  against the current opponents and then imitated for 40 more
  teacher-executed updates (v10-v49, the imitation control's recipe),
  evaluated every 10 on the deck's own suite:

  | Deck | Teacher against `2cd3c4a` | v9 against `cf8cc65` |
  | --- | ---: | ---: |
  | Monarch | 68.4% | 65.3% |
  | Spellcaster | 23.4% | 13.2% |

  Monarch alone was agreed. Its seed is already within 5 points of its
  teacher, and PPO from it reached 69.4%, so it has little left to show.
  Spellcaster's seed wins about half as often as its teacher, which makes
  it the harder test. `DECKS=monarch` leaves it out.

  Readings, per deck, on the mean of the v39 and v49 evaluations (the
  Blackwing arms levelled off one to two points under their teacher):
  - *Distilled:* within 3 points of the teacher (two standard errors of the
    difference).
  - *Not yet at a plateau:* v49 more than 2 points above v39 and still more
    than 3 under the teacher. The run is continued in place.
  - *Levelled off short:* more than 3 points under the teacher with v39 and
    v49 within 2 points of each other. Imitation does not reach that teacher
    with this model and observation, and the generalist starts with a known
    gap on that deck. The per-matchup rates and the agreement on the
    teacher's states say where it is.
- **`train --continue` on a branch.** A run now records the version it began
  at (`schedule_origin` in `run.json`), and a continuation counts its
  warm-up and imitation-decay schedule from there. Runs recorded earlier
  count from their first recorded version. Locally, a branch resumed into a
  three-update imitation warm-up, stopped after one update and continued,
  reproduced the single invocation's rollout fingerprints and losses. An
  interrupted imitation branch can therefore be continued in place.
- **Command:**
  [`commands/regression-remedies-phase1c.sh`](commands/regression-remedies-phase1c.sh),
  about 3 hours on the training machine (2.4 without Spellcaster). It runs
  the imitation first, then the three branches, and gathers each run's
  metrics and status table, the two seed evaluations and the score of the
  branches' checkpoints under `analysis/remedies/`.

### Phase 1c result

All five runs completed on 2026-10-04 (`analysis/remedies/`).

**The anchor.** The three branches share their first rollout (46,789
samples, 45.8% training wins). Held-out evaluation on the Blackwing suite
(teacher 62.1%; the arm was at 62.0% at v59):

| Branch | v69 | v79 |
| --- | ---: | ---: |
| Anchor kept | 64.8% | 63.5% |
| Anchor off | 54.6% | 50.4% |
| Anchor thinned | 61.5% | 58.6% |

The rollout of the last update, and `action_values score` on the 274 states
(`analysis/remedies/action-values-score-phase1c.json`):

| Checkpoint | Training wins | Entropy | Argmax agreement with the teacher on its own states | Agreement on the 274 states | Chain `pass -> activate` states | P(pass) where the teacher passes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Anchor 0.1, v59 | 46.2% | 0.71 | 78.4% | 86.9% | 2 | 0.931 |
| Anchor kept, v79 | 47.1% | 0.73 | 78.4% | 86.1% | 10 | 0.859 |
| Anchor off, v79 | 36.0% | 1.16 | not measured | 64.6% | 40 | 0.586 |
| Anchor thinned, v79 | 43.1% | 0.82 | 74.0% (its labelled tenth) | 82.8% | 14 | 0.828 |

Agreement on the 274 states at v64, v69, v74 and v79: kept 86.1%, 81.8%,
86.1%, 86.1%; off 78.5%, 72.3%, 70.8%, 64.6%; thinned 86.5%, 80.3%, 80.3%,
82.8%. Value minus teacher stays within 0.011 of zero for every checkpoint,
as before.

- **The anchor has to stay.** Without it the drift resumes at once, at the
  unanchored control's pace: agreement on the 274 states falls 22 points in
  20 updates (the control from v9 lost 22), P(pass) falls from 0.931 to
  0.586, premature chain activations rise from 2 to 40 states, entropy from
  0.69 to 1.16, and the held-out rate falls from 62.0% to 50.4%. Fifty
  anchored updates did not leave a policy that PPO spares: the anchor is a
  standing counter-force, not a cure that can be withdrawn.
- **The thinned anchor slows the drift and does not hold it.** By the
  reading it holds on the 274 states (1.5 and 3.3 points under the kept
  branch) and fails held out at v79 (4.9 points under, against 3.3 at v69).
  Agreement on its own labelled states falls from 79.9% to 74.0% and its
  entropy rises to 0.82 without settling. Held out it keeps about two
  thirds of the distance between no anchor and the full one.
- **The thinned branch pulled less than intended, so it cannot say why it
  drifts.** The imitation term averages over labelled transactions, and a
  transaction holds 1.43 decisions on average (30% hold more than one).
  Thinning by decision leaves more than a tenth of the transactions with a
  label, so each labelled decision weighs 0.70 of what it does in the kept
  branch (0.447 against 0.639 on the shared first rollout). The branch
  tested a tenth of the labels at an effective weight of 0.07, and the two
  are not separated.
- **The kept branch gives the arm's best evaluations, and its first above
  the teacher**: 64.8% and 63.5% against 62.1%, neither past the 65.5% bar
  set in phase 1b. The arm has risen in every evaluation but the last since
  v19 (57.1%, 58.5%, 59.5%, 59.9%, 62.0%, 64.8%, 63.5%), while DAgger
  levelled off at 60.9% from v39. Over v59 to v79 its largest gains on the
  teacher are in the teacher's three worst matchups: Monarch +10.9 points
  (teacher 25.8%), Lightsworn +6.5 (34.4%), Heroes +4.4 (39.8%). This is
  one run, three evaluations of 128 games a matchup, with no DAgger
  evaluation after v59: an observation, not yet a result. Over the same 20
  updates its P(pass) on the teacher-pass chain states fell from 0.931 to
  0.859 and its premature chain activations rose from 2 to 10 states, both
  the arm's weakest values so far (0.893 and 7 states before).

**Imitation on other decks.** Held-out evaluation on each deck's own suite
(1,664 games):

| Deck | Teacher | v9, current opponents | v19 | v29 | v39 | v49 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Monarch | 68.4% | 64.7% | 67.1% | 67.5% | 67.6% | 67.2% |
| Spellcaster | 23.4% | 12.6% | 14.7% | 16.3% | 16.4% | 18.2% |

Argmax agreement with the teacher on the teacher's own states went from
79.3% to 81.4% for Monarch and from 83.6% to 86.8% for Spellcaster; the
imitation loss from 0.441 to 0.382 and from 0.345 to 0.253.

- **Monarch is distilled.** It stands at 67.4% over v39 and v49, one point
  under its teacher, and was there within ten updates.
- **Spellcaster is not, after 40 updates.** At v49 it is 5.2 points under
  its teacher (3.7 standard errors), up from 12.6%. By the reading it
  levelled off short, because v49 is 1.8 points above v39 where the reading
  asks for more than 2. That sits on the boundary: the series rose in three
  of its four steps and the imitation loss is still falling. The largest
  gaps at v49 are against Crystal (14 points), the mirror (12), Morphtronic
  (10) and Burn (8).
- **Agreement on the teacher's states does not order strength across decks
  either.** Spellcaster agrees with its teacher more often than Monarch
  does (86.8% against 81.4%) and reaches 78% of its teacher's win rate,
  where Monarch reaches 98%.

Open after phase 1c:

- Whether the anchored arm's rise continues, and whether it is PPO's. DAgger
  has no evaluation after v59, and the arm is a single run.
- Whether sparse labels hold the policy at an equal pull.
- Whether Spellcaster's imitation reaches its teacher with more updates, or
  needs labels on the learner's own states.

## Phase 1d: the anchored arm against DAgger, and Spellcaster's two branches

Agreed on 2026-10-05, for the first and third of the questions phase 1c left
open, while the current checkpoints still load. The thinned anchor is not
rerun: it matters only with a costly teacher, and that setup would need its
own test. Run on 2026-10-05; the result follows the protocol.

- **Does the anchored arm keep rising, and is the rise PPO's?** Anchor kept
  continues in place to v119 (40 updates). DAgger, the same labels on the
  learner's own states without the PPO term, continues in place over the
  same versions (v60-v119) as its control. Both are evaluated every 10
  updates and scored on the 274 states at v79 (DAgger also v59), v99 and
  v119.

  The test uses the four new evaluations of each arm, v89 to v119. Anchor
  kept's v69 and v79 are left out because they prompted the run. The bars
  are those of phase 1b:
  - *PPO adds strength* if Anchor kept's mean over v89-v119 is more than
    about 4 points above DAgger's over the same versions, or above 65.5%.
  - *A lead that holds* if it passes neither bar but all four evaluations
    are above DAgger's at the same version and their mean is above the
    teacher's 62.1%. The rise is then probably real and slow, and one run
    cannot size it: a second training seed would, and the generalist should
    budget well over 100 anchored updates before judging PPO.
  - *No lead* if the mean is within 2 points of DAgger's: the v69 and v79
    values were noise.
  - Anything between is recorded as undecided.

  DAgger's own v69 and v79 say whether imitation alone also moved after v59.
  The score says whether the kept branch's chain passes go on eroding
  (P(pass) 0.859 at v79) while its held-out rate holds.
- **Does Spellcaster's imitation reach its teacher, and whose states have to
  be labelled?** Two branches from the imitation run's v49 (18.2%), 40
  updates each (v50-v89), evaluated every 10 on the Spellcaster suite
  (teacher 23.4%):

  | Branch | Flags |
  | --- | --- |
  | More imitation | `--imitation-only-updates 40 --warmup-execution teacher --imitation-value-weight 0.5` |
  | DAgger | `--teacher-imitation-weight 1 --policy-loss-weight 0 --entropy-coefficient 0` |

  A continuation cannot lengthen an imitation schedule (the run's protocol
  is fixed), so more imitation is a branch too: it restores the optimizer
  and goes on with the same rollout seeds. This is phase 1b's comparison of
  the imitation control with DAgger, on a deck where the gap is large.

  Readings, on each branch's mean over v79 and v89:
  - *Imitation reaches the teacher* if More imitation is within 3 points of
    23.4%. The shortfall at v49 was then a matter of updates: Monarch took
    about 10, Blackwing about 30, Spellcaster more than 40.
  - *Whose states are labelled matters* if DAgger is more than about 3
    points above More imitation. Covariate shift, which Blackwing did not
    show, is then real on this deck, and a generalist should label the
    learner's own states from the start, as its anchor does.
  - *Neither* if both stay more than 3 points short and within 3 of each
    other: the gap closes neither with updates nor with the learner's own
    states, and is carried into the generalist as a limit of this model and
    observation on this deck, with the per-matchup gaps.
  - If only DAgger comes within 3 points of the teacher, without leading
    More imitation by 3, the two branches are not separated.
- **Command:**
  [`commands/regression-remedies-phase1d.sh`](commands/regression-remedies-phase1d.sh),
  about 4.5 hours on the training machine (3.4 with
  `SPELLCASTER_DAGGER=0`). It runs Anchor kept first, then DAgger, then the
  two Spellcaster branches, and gathers the metrics, status tables and score
  under `analysis/remedies/`.

### Phase 1d result

All four runs completed on 2026-10-05 (`analysis/remedies/`).

**The anchored arm against DAgger.** Held-out evaluation on the Blackwing
suite (teacher 62.1%):

| Arm | v59 | v69 | v79 | v89 | v99 | v109 | v119 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Anchor kept | 62.0% | 64.8% | 63.5% | 63.6% | 62.9% | 60.1% | 60.9% |
| DAgger | 60.5% | 58.6% | 61.1% | 59.9% | 61.5% | 61.4% | 62.4% |

- **No lead.** Over the four new evaluations Anchor kept averages 61.9% and
  DAgger 61.3%. They are 0.6 points apart, where the reading allows 2, and
  Anchor kept is above DAgger in two of the four and 0.2 points under the
  teacher. The lead that prompted the run was gone by v109.
- **Over 110 updates PPO has added nothing measurable to the anchored
  learner in aggregate** (57,200 duels in each arm).
- **A single evaluation is good to about a point.** DAgger's rollout
  statistics barely move from v39 to v119 (entropy 0.28 to 0.34, agreement
  85.9% to 87.9%), and its nine evaluations over those versions range from
  58.6% to 62.4%, with an SD of 1.1 points, the binomial error of 1,664
  games. The 64.8% at v69 was more than two such errors above where the arm
  settled. Comparisons between arms need the mean of several evaluations.
- **The two learners win against different decks.** Over v89 to v119 Anchor
  kept is ahead of DAgger against Monarch (+7.2 points), Heroes (+5.3) and
  Rock-Block (+5.3), and behind against Morphtronic (-5.7), Gishki (-4.7)
  and Spellcaster (-2.7): ahead against Heroes in all four evaluations,
  behind against Spellcaster in all four. DAgger stays close to the teacher
  in these matchups. Every evaluation replays the same games, so the four
  are not independent: this is a pattern to test, not a result.
- **The kept branch's chain passes did not go on eroding.** P(pass) where
  the teacher passes was 0.859 at v79, 0.914 at v99 and 0.900 at v119, with
  10, 4 and 7 premature activations. Its entropy stays between 0.65 and 0.78
  and its agreement on its own states between 77% and 80%: the anchored
  policy is in a steady state. DAgger's P(pass) is 1.000.

**Spellcaster.** Held-out evaluation on the Spellcaster suite (teacher
23.4%; both branches start from 18.2% at v49):

| Branch | v59 | v69 | v79 | v89 |
| --- | ---: | ---: | ---: | ---: |
| More imitation | 18.9% | 18.2% | 18.6% | 18.9% |
| DAgger | 18.6% | 19.2% | 20.4% | 19.4% |

- **Neither.** Over v79 and v89 More imitation stands at 18.7% and DAgger at
  19.9%: 4.7 and 3.5 points under the teacher, and 1.2 apart. More updates
  do not close the gap, and labels on the learner's own states close a
  quarter of it at most, inside the noise.
- **The imitation itself has almost stopped improving.** Between the first
  and the second 20 updates, agreement with the teacher on the teacher's
  states goes from 87.2% to 87.4% and the loss from 0.251 to 0.244.
- **The learner does not see something this pilot plays by.** The
  Spellcaster pilot's rules read the Spell Counters on its cards
  (`decks/spellcaster.rs` in ygo-policies), and the teacher's seat receives
  them. The learner's observation has no counters at all
  (`policy_observation.CardState` holds code, place, position, ATK and
  DEF). This was verified in the code; how much of the gap it explains was
  not measured.
- The largest gaps over v79 and v89 are in the mirror (34% for both
  branches against the teacher's 51.6%), against Crystal (33% and 36%
  against 44.5%), Morphtronic (33% and 36% against 41.4%) and Infernity
  (10% against 15.6%).

## The port to the ygo-policies tooling (2026-10-06)

The second half of the cleanup, done once the engine with snapshots and the
hidden-card swap, a duel-running library and searched duels as records were
in ygo-policies (`a05b472`). Duels now run in `crates/ygo-env`, a Rust
library on those crates, in place of the .NET worker. The C# code, the old
deck pool and the trainer's switched-off features are gone.

What it changes for experiments:

- **Checkpoint format 13.** Earlier checkpoints read another input and do
  not load. The code that ran them is not part of this repository.
- **The input is what a pilot is given at the same decision**, through the
  policies' own redaction and projection: current ATK and DEF, level,
  counters (which the Spellcaster pilot plays by and the old input lacked),
  whether a monster can attack, the chain, the attack under way, the card of
  the event being answered, and the card of the command a selection follows
  from. The engineered combat features and the composite attack action are
  gone.
- **Fewer things are asked.** The zone a card goes to is the pilot's choice,
  and a decision with a single answer takes it. Samples and agreement rates
  now count real decisions only, so they do not compare with earlier ones,
  which included both.
- **Transactions.** A command and the selections it leads to, until the
  other seat has something to answer, are one action for PPO and for both
  imitation losses. The imitation-only updates grouped only an attack with
  its target before.
- **History.** The seat's own earlier answers. It held both players'
  answers before, which a seat does not see.
- **Deals and pilots.** A seed now deals as `policy-bench` deals it, and
  the pilots are those of `a05b472`, not `2cd3c4a`. Earlier evaluations are
  not paired with new ones, and every teacher reference has to be measured
  again (`evaluate-suite --pilot` plays the same deals with the pilot in the
  learner's seat).
- **Tools.** `action_values` and `update_attribution` are removed: they work
  on the 274 states captured in the old format and valued by branching the
  C# engine. `policy-bench search` values alternatives now. The single
  `evaluate` command, the tournament, the shortlist, MLflow and
  `teacher_evaluation` are removed too.

Checks:

- Before the engine was replaced, the trainer was stripped of its
  switched-off features on the old engine. A small end-to-end run
  (imitation-only, anchored PPO with a resume and a continuation, DAgger,
  thinned labels, a suite, the references, the status table) gave the same
  754 metric values before and after.
- Two pilots seated in the new environment play the duel `policy-bench`
  plays: same digest of every question and answer, winner, turns and number
  of answers, on the crate's own tests and on the rows of a bench matchup.
- The rows of a small search with records (8 duels, 18 deviations) play
  again as recorded, with a departure from the pilot at each deviation's
  `decision` and nowhere else.
- A learner that follows its labels plays the pilot's duel.
- On the CPU, twelve imitation-only updates of 96 duels (Blackwing against
  four decks) bring the model to the pilot's answer on 86.5% of its real
  decisions. This is a check that the input can be learned from, not a
  measured strength: no held-out evaluation has been run on the new setup.

### The engine, the scripts and the card database brought up to date

On the same day, before anything was measured on the new setup, the pin
moved to ygo-policies `215f7d0`. The pilots are unchanged; what they run on
is not:

| | Before | After |
| --- | --- | --- |
| OCGCore (the fork) | `2678487` | `1c49ef7`: 48 upstream commits merged, and compiled scripts shared between duels |
| CardScripts | `180d49c` | `37f270d`, upstream of 2026-10-05 |
| BabelCdb | `738d6c7` | `fdf92ae`, upstream of 2026-10-05 |

It was done now because the references, the imitation seed and the searched
games of the next experiment are all tied to these revisions, and none
existed yet.

The policies repository checked the two steps apart. The merged engine
reproduced 504 games exactly. The new scripts and card database, on that
engine, across 33 policies and 16,896 paired duels, gave the same winner,
win condition and turn count in every duel, other answers in 56 (53 with
Destiny HERO - Dogma, whose summoning procedure was rewritten, 2 with Pot
of Duality, 1 with Fusion Gate) and another digest in 8,536, which its
report puts down to effect descriptions the scripts now give. The new card
database alone changed nothing.

For the Blackwing seat, measured here on 192 games against the twelve
opponents (16 each), at the old revisions and the new:

| | Changed |
| --- | --- |
| Winner, win condition, turn count | 0 games |
| Either pilot's answers | 0 games |
| The bench's digest | 114 games |
| The network's input | 140 of 15,975 decisions (0.9%), in 95 games |

The input differs in one value: activating Pot of Duality now comes with
its effect's description, so that answer's names-an-effect flag is set.
The format stays 13.

A record made at other revisions can play again without an error and still
show the network other inputs than the recorded duel had: the 192 old
records play to the same end at the new revisions, 114 of them with another
digest. A record belongs to the revisions it was made with, and its digest
tells whether it is played at them.

Checks at the new pin: the crate's and the Python tests pass; the 192 games
come out of the environment digest for digest as the bench plays them, and
play again from their records; 16 searched duels and their 16 pilot games
play again as recorded, with a departure at each of the 31 deviations and
nowhere else; the small end-to-end run completes, and its pilot references
(8 games) are those of the old revisions.

### The pilot's reference at the new pin

Measured at `215f7d0`; the same games are played at `bc8c0f7`.

The Blackwing pilot in the learner's seat on the Blackwing suite (13 decks,
128 games each, seed 2012), which is what a model's games on the same deals
are set beside from here on:

| | Wins | Rate |
| --- | ---: | ---: |
| All 13 | 1,080 of 1,664 | 64.9% |
| Without the mirror | 1,007 of 1,536 | 65.6% |

By opponent it runs from 33.6% (Monarch), 44.5% (Infernity) and 45.3%
(Lightsworn) to 87.5% (Burn) and 91.4% (Spellcaster); the mirror is 57.0%.
No game is drawn or cut off. `evaluate-suite --pilot` gives the same
numbers on any machine at this pin, in seconds. It is not paired with the
62.1% of the old setup: the deals, the engine and the pilots all differ.

### The faster search

Later on 2026-10-06 the pin moved once more, to ygo-policies `bc8c0f7`,
which reworks the search's runtime and makes the pilots' seats clonable.
The engine, the scripts and the card database are the same. The 192 plain
and 16 searched Blackwing games kept from `215f7d0` come out identical at
the new pin (results, answers, deviations, digests), and so does the small
end-to-end run, so the reference below and anything trained at `215f7d0`
stand. The search plays the same games in three fifths of the time. Its
marks at these revisions, on 384 games, are in the
[search-imitation record](2026-10-05-search-imitation.md): 77.1% where the
pilot wins 64.1%.

### What a run now records

`run.json` holds the revisions the duels ran on (the pilots' commit, the
engine's, the scripts' and the card database's), and a run is not continued
across a change of them. The evaluation outputs hold them too.

Still open on the new setup: no model has been evaluated at full size. The
first such run is the imitation seed of the next experiment
([`commands/format13-imitation-seed.sh`](commands/format13-imitation-seed.sh)),
which also shows whether imitation in the new input reaches this reference.

## Conclusion and what follows

Closed on 2026-10-06, when
[imitating the search](2026-10-05-search-imitation.md) became the active
experiment.

Phase 1 is answered, the runs on the old setup are finished, and the port
to the new one is done. An imitation
anchor on the learner's own states cures the regression, at a weight of
0.1 as well as 1, and only for as long as it stays. It brings the learner
to its teacher's level and no further: over 110 updates the anchored arm
and DAgger end within a point of each other and of the teacher, and PPO
adds no wins in aggregate. Imitation reaches the
teacher for Blackwing and Monarch and stops at four fifths of it for
Spellcaster, whose pilot plays by counters the learner cannot see. A
search on top of the same teacher wins 16 to 18 points more.

Agreed on 2026-10-04, in this order:

1. **Phases 1c and 1d**, above: run. A second training seed for the
   anchored arm was considered and left out (2026-10-05): a single-run
   answer is accepted, and the generalist tests PPO again.
2. **Clean up the repository** before any new experiment: done. The Godot
   client, the earlier pilots, unused files and tools, the Rust conversion
   branches and the C# code are gone; the project is at the repository root;
   decks and the tooling that runs duels come from ygo-policies (the port,
   above). The learner observes what the pilots observe, counters included.
3. **Then [imitating the search](2026-10-05-search-imitation.md)**: a
   learner imitates games the search plays, against one that imitates the
   pilot's games on the same deals. On 2026-10-04 this was the alternative
   kept for after the generalist. Its design was agreed on 2026-10-05, to
   start as soon as the port is done. The trainer's side of it (imitating
   recorded duels, measuring on a held-out search run) was written on
   2026-10-06.
4. **The multi-deck generalist (an earlier record, not included here)**:
   one model distilled from several decks' policies, with the 0.1 anchor,
   to see whether PPO moves past the teachers once the model knows more
   than one deck. The teachers stay the policies, so the gap to the search
   is known. Search marks for its other decks are measured after the
   cleanup.

The pilots' own rules are improved separately, by the agent working on
ygo-policies, from the logged overrides; the remaining logged searches and
the engine build's move are that agent's too. The search is in that
repository's `main` (`policy-bench search`).
