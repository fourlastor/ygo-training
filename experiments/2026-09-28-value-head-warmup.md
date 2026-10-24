# Value-head warm-up

**Status:** INVALIDATED

**Activated:** 2026-09-28

**Last updated:** 2026-09-28


## Invalidated: OCGCore never received archetype set codes

**Recorded 2026-09-28.** `YGO.Core/Duel.cs` did not pass card set codes to
OCGCore from 2026-07-16 (the commit that moved the native setup into Duel)
until `f22edb4e2`. OCGCore therefore treated every card as archetype-less, and
it silently disabled every setcode-gated effect: Black Whirlwind's Blackwing
search, Gishki Ritual Summons, archetype summon conditions and more. The fix
was verified against an independent harness. The blackwing-vs-monarch policy
benchmark now matches it exactly (190/512 on seeds 7000-7511), where it gave
70/512 before.

Every duel in this record was played on that engine, so every result and
teacher reference here is invalid as a measurement of these decks. Mechanisms
observed here, such as the value head's first-update jump and PPO clipping
limiting per-update distance, are plausible but must be re-checked on the
corrected engine before they are relied on. The re-baseline is
[`2026-09-28-archetype-fix-rebaseline.md`](2026-09-28-archetype-fix-rebaseline.md).

## Research question

If the value head is trained during the imitation warm-up, does the first
phase of PPO keep more of the imitated policy, and does the run end stronger?

## Motivation and hypothesis

In [`2026-09-28-ygo-policies-board-tokens.md`](2026-09-28-ygo-policies-board-tokens.md),
`imitation_update` trained only the policy logits. The value head was still at
its random initialization when PPO started. Explained variance at update 10 was
-0.008 (control) and -0.17 (treatment), rising to about 0.52-0.57 by update 11
and 0.72 by update 12. In the same updates:

- approximate KL was 0.126-0.139 at update 10;
- entropy roughly doubled at update 11 (0.40 to 0.77 in the control);
- held-out results fell, most sharply in the treatment (71.5% to 60.8% by
  version 19);
- neither arm beat its imitation checkpoint by version 99, and the hard subset
  fell from 28.6-34.4% to about 19%.

The hypothesis is that PPO's first updates use advantages computed from an
untrained critic, move the policy a long way in noise-driven directions, and
discard much of the imitated policy. Training the value head toward duel
outcomes during warm-up should give PPO usable advantages from its first
update.

The competing explanation is that PPO's step size is too large regardless of
the critic: KL stayed at 0.03-0.05 long after explained variance recovered. If
the pilot fixes the critic but not the collapse, step-size control is the next
pilot.

## Implementation

`--imitation-value-weight` (default 0, which reproduces earlier runs exactly)
adds a value-regression term to each imitation-only update. Every labelled
learner-seat step targets its own seat's terminal duel payoff, with the
standard cutoff payoff for unfinished duels. There is one target per causal
transaction, the same weighting PPO uses. Warm-up updates log
`imitation_value_loss` and `imitation_value_explained_variance`, measured on
each update's fresh teacher rollouts before optimization. The pilot uses 0.5,
which equals PPO's own value-loss coefficient.

In a CPU smoke test (3 warm-up updates of 52 duels), explained variance at the
first PPO update was 0.578 with weight 0.5, against 0.001 with weight 0. That
shows the mechanism works. It is not evidence of a benefit.

## Phase 1: single-arm pilot (diagnostic)

Single-arm pilots were chosen over paired 100-update runs until the early
collapse is understood. Pilot results are diagnostics, not paired
experimental evidence.

- Command: [`commands/value-warmup-pilot.txt`](commands/value-warmup-pilot.txt).
  It matches the board-token control command except for the run name, 30
  updates instead of 100, and `--imitation-value-weight 0.5`.
- Opponent pool: ygo-policies `6203688`, checked out explicitly on the
  training machine, so the finished board-token control arm is an exact
  reference. The same seed gives identical teacher warm-up rollouts. The only
  change is value training, which also passes gradients through the shared
  layers, so policy imitation can differ slightly.
- Reference: board-token control arm, updates 0-29.

Criteria, compared with the reference:

1. `duel_explained_variance` at update 10 clearly above zero (reference:
   -0.008).
2. `duel_approximate_kl` at update 10 clearly below 0.126, and entropy not
   roughly doubling at update 11 (reference: 0.40 to 0.77).
3. Held-out results at versions 19 and 29 close to version 9, rather than
   falling (reference: 66.9%, 66.8%, 65.8%; hard subset 28.6%, 25.3%, 25.3%).

If criteria 1-2 hold but 3 does not, or entropy keeps rising, the critic is
not the whole explanation, and the next pilot tests PPO step-size control.

## Phase 2: paired confirmation (planned)

If the pilot passes, run a paired 100-update experiment on the fixed opponent
pool (ygo-policies `9e5790e`): control at weight 0 against treatment at weight
0.5, `mean-pool`, seed 2012, with the board-token evaluation suite and
pre-registered hard subset. The pool change applies to both arms. Before
Phase 2, evaluate the `blackwing` policy itself on the new pool for teacher
reference numbers. Freeze the Phase 2 commands only after the pilot result is
recorded.

## Evidence

### Phase 1 pilot result

The pilot ran 30 updates on ygo-policies `6203688`, using the `mean-pool`
architecture, seed 2012, and imitation value weight 0.5. The reference is the
board-token control arm, which used the same seed, pool and warm-up rollouts,
but no value warm-up.

| Update | Explained variance, control / pilot | Approx. KL, control / pilot | Entropy, control / pilot | Samples, control / pilot |
| ---: | --- | --- | --- | --- |
| 10 | -0.008 / 0.555 | 0.126 / 0.066 | 0.400 / 0.418 | 63.3k / 58.1k |
| 11 | 0.566 / 0.758 | 0.076 / 0.051 | 0.769 / 0.389 | 41.2k / 65.7k |
| 15 | 0.784 / 0.806 | 0.035 / 0.030 | 0.800 / 0.551 | 41.3k / 51.1k |
| 19 | 0.783 / 0.790 | 0.036 / 0.032 | 0.846 / 0.693 | 40.6k / 46.5k |
| 24 | 0.754 / 0.754 | 0.037 / 0.033 | 0.996 / 0.782 | 37.9k / 44.5k |
| 29 | 0.773 / 0.769 | 0.033 / 0.034 | 1.072 / 0.784 | 36.4k / 44.7k |

Held-out evaluation:

| | Control v9 / v19 / v29 | Pilot v9 / v19 / v29 |
| --- | --- | --- |
| aggregate | 66.9% / 66.8% / 65.8% | 65.9% / 63.4% / 66.7% |
| hard subset (pooled) | 28.6% / 25.3% / 25.3% | 26.6% / 27.1% / 29.7% |
| other ten (pooled) | 78.4% / 79.2% / 78.0% | 77.7% / 74.3% / 77.8% |
| mirror | 37.5% / 28.1% / 33.6% | 31.2% / 34.4% / 34.4% |
| monarch (wins) | 16 / 7 / 2 | 14 / 16 / 11 |
| rock-block | 35.9% / 42.2% / 40.6% | 37.5% / 34.4% / 46.1% |
| deck-out losses / wins | 8/9, 1/9, 0/31 | 7/8, 3/10, 1/12 |

Criteria:

1. **Passed.** Explained variance at update 10 was 0.555, against -0.008.
2. **Passed.** Approximate KL at update 10 halved (0.066 against 0.126), and
   entropy did not jump at update 11 (0.389 against 0.769).
3. **Passed on the aggregate, within noise.** The pilot dipped at version 19
   (63.4%, about one standard error of an evaluation difference), then ended
   0.8 points above its own version 9. The hard subset was the only
   improvement: 29.7% against the control's 25.3% at version 29 (about 1.4
   standard errors), with Monarch at 11 wins against 2 (about 2.5 standard
   errors). The pilot's hard subset ended above its own version 9 result.

Interpretation:

- The untrained critic caused the first-update shock. Training it during
  warm-up removed the shock and changed the course of the whole run that
  followed.
- A slower drift remains. Pilot entropy still climbed from 0.39 to about 0.8
  and levelled off there. The control kept climbing, to 1.07. Approximate KL
  settled at 0.03-0.035 in both runs, so the ongoing PPO step size is
  unchanged. The value warm-up moves the run to a better starting path, but
  PPO still drifts in the same direction.
- The pilot had fewer Lightsworn deck-out wins through version 29 (12 against
  31). With one seed, this timing is partly chance.
- The pilot's larger sample counts come from more student decisions per duel,
  closer to the warm-up level (about 60k). That is consistent with the pilot
  keeping more of the teacher's playing style. This is inferred from counts,
  not verified with turn-level diagnostics.

**Decision:** keep `--imitation-value-weight 0.5` in all later pilots and
experiments. It costs nothing measurable and removes the first-update shock.

## Phase 1b: PPO step-size pilot (next)

Test KL-targeted early stopping of PPO epochs (`--target-kl`, off by default),
with the value warm-up kept on. Use the same pool (`6203688`), seed,
architecture and 30-update budget, and use the Phase 1 pilot as the reference.
The question is whether smaller steps lower the evaluation curve's dependence
on noise, or just slow the same drift. Compare runs against cumulative
approximate KL as well as update number: if the curves overlap by cumulative
KL, step size only changes speed.

Implementation: `--target-kl T` measures each minibatch's approximate KL
against the rollout policy before its optimizer step. It ends the update,
without stepping, once that KL exceeds 1.5 x T, following Stable-Baselines3.
Updates now also log `duel_final_approximate_kl` (the mean over the last,
possibly partial, epoch), `ppo_epochs_completed`, `ppo_optimizer_steps` and
`ppo_early_stopped`. `status --optimization` shows these and a running total
of end-of-update KL.

The logged `duel_approximate_kl` is averaged over all epochs, including the
first epoch, which measures almost no movement. It therefore understates how
far an update moves the policy. Earlier runs, including the Phase 1 pilot, do
not have end-of-update KL, so the cumulative-KL comparison is available only
between runs that record it. The Phase 1b pilot is compared with the Phase 1
pilot by update number and evaluation. Its cumulative KL is recorded for later
runs.

- Command: [`commands/target-kl-pilot.txt`](commands/target-kl-pilot.txt).
  It matches the Phase 1 pilot except for the run name and
  `--target-kl 0.02`. The target was chosen before any Phase 1b data. The
  Phase 1 pilot's epoch-averaged KL was 0.03-0.035, so 0.02 should end most
  updates part-way through their four epochs.
- Pool: ygo-policies `6203688`, as in Phase 1.

Criteria against the Phase 1 pilot:

1. PPO updates stop early (`ppo_epochs_completed` below 4) and
   `duel_final_approximate_kl` stays near the 0.03 stopping threshold.
2. Entropy stays clearly below the Phase 1 pilot's 0.75-0.88 over updates
   20-29.
3. Held-out results at versions 19 and 29 are at least as good as the Phase 1
   pilot's (63.4% and 66.7%; hard subset 27.1% and 29.7%). A lower curve with
   stable entropy would mean the steps are now too small for 30 updates to
   show improvement, not that smaller steps are harmful.

### Correction to the Phase 1 interpretation: the warm-up value head overfits

The identical warm-up rows of the two pilots show `imitation_value_loss`
falling from 0.33 to 0.16. Outcomes are ±1 with the learner winning about 71%
of the time, a variance of about 0.82, so the value head fits its own training
games well. Its `imitation_value_explained_variance` on each update's fresh
games stayed between -0.17 and 0.00, however. It memorizes the outcomes of the
520 games of each update, which are strongly correlated within a game, instead
of learning what a winning position looks like.

The Phase 1 explained variance of 0.555 at update 10 is measured against GAE
returns that include the value estimate itself, so it overstates predictive
power. Phase 1's measured improvements (a smaller first update, no entropy
jump) stand. The credit may belong to a well-scaled, smooth value estimate
rather than an accurate prediction of who wins. Possible refinements are one
value epoch per warm-up update, or a lower value weight. Neither has been
tested.

### Phase 1b result: `--target-kl 0.02` (rejected)

The target was far more aggressive than intended. Within an update,
minibatch KL rises steeply over the first few dozen steps, then levels off as
PPO clipping takes hold. The logged `approx KL` averages over all roughly 900
steps and hid this. Updates stopped after 0.02-0.11 epochs, about 5-25 of
roughly 900 optimizer steps. A local trace confirmed that minibatch KL is
exactly 0 before the first step, so the stopping mechanism itself is correct.

| | Phase 1 pilot | Target-KL pilot |
| --- | --- | --- |
| aggregate, v9 / v19 / v29 | 65.9% / 63.4% / 66.7% | 65.9% / 63.5% / 60.9% |
| hard subset, v19 / v29 | 27.1% / 29.7% | 25.8% / 23.4% |
| other ten, v19 / v29 | 74.3% / 77.8% | 74.8% / 72.2% |
| rollout entropy, updates 20-29 | 0.75-0.88 | 0.23-0.29 |
| samples per update, 20-29 | 41-46k | 63-74k |

Criteria 1 and 2 were met: updates stopped early, and entropy stayed low. In
fact entropy fell. Criterion 3 failed: version 29 was 5.8 points below the
Phase 1 pilot (about 3.5 standard errors), with a broad regression and a lower
hard subset.

### Policy-distance analysis

`python -m ygo_training.policy_distance` scored every checkpoint of the three
runs on 12,651 learner-seat states from 104 teacher-executed duels (seed 7000,
ygo-policies `6203688`). Distances are KL from each run's own version 9, plus
checkpoint-to-checkpoint step KL.

Per-update step KL:

| Update | Control | Phase 1 pilot | Target-KL pilot |
| ---: | ---: | ---: | ---: |
| 10 | 0.307 | 0.066 | 0.031 |
| 11 | 0.080 | 0.057 | 0.030 |
| 13 | 0.029 | 0.031 | 0.026 |
| 15-19 | 0.026-0.030 | 0.024-0.028 | 0.020-0.030 |
| 21-29 | 0.026-0.027 | 0.024-0.031 | 0.007-0.019 |

Results at matched distance from version 9:

| Checkpoint | KL from v9 | Path KL | Agreement with v9 | Aggregate | Hard subset |
| --- | ---: | ---: | ---: | ---: | ---: |
| Control v19 | 0.360 | 0.639 | 83.3% | 66.8% | 25.3% |
| Phase 1 pilot v29 | 0.361 | 0.655 | 84.2% | 66.7% | 29.7% |
| Phase 1 pilot v19 | 0.266 | 0.365 | 88.4% | 63.4% | 27.1% |
| Target-KL pilot v29 | 0.260 | 0.436 | 91.4% | 60.9% | 23.4% |

Findings:

1. **The control's first PPO update moved the policy about as far as ten
   ordinary updates** (0.307). The value warm-up reduced it to 0.066. After
   the first one or two updates, the control and the Phase 1 pilot move about
   the same per update (0.025-0.03).
2. **Per-update distance is already limited by PPO clipping.** The target-KL
   pilot used 2-10% of the optimizer steps, yet moved 0.020-0.030 per update
   through update 19, close to the full-step runs. The remaining steps add
   little further movement on these states. The earlier conclusion that PPO
   steps were too large does not hold after the first update. Early stopping
   mainly discarded most of each update's data, which made its direction
   noisier at a similar size.
3. **At matched distance, the Phase 1 pilot reaches the same aggregate as the
   control** (66.7% against 66.8% at KL from v9 of about 0.36). By this
   measure the value warm-up is the control without its one-update jump. Its
   hard subset at that point was higher (29.7% against 25.3%, about 1.4
   standard errors), which is suggestive only.
4. **Staying near the imitation policy does not keep strength.** The
   target-KL pilot is the closest to version 9 (91.4% agreement) and the
   weakest at matched distance (60.9% against 63.4%). The direction of change
   matters more than its size.
5. **Entropy hypothesis.** On the same teacher states, the Phase 1 pilot's
   entropy doubled (0.28 to 0.58), while the target-KL pilot's stayed flat
   (0.28 to 0.26), at similar per-update distances. The difference is the
   hundreds of later steps, where most samples are clipped and the entropy
   bonus (coefficient 0.01) still acts. The hypothesis is that these steps
   drive the steady entropy increase in full-step runs. It is untested.
6. In the control's long run, distance from version 9 levels off at about
   0.85-0.95 after version 49, while path KL keeps rising (1.38 to 2.59). The
   run wanders at a roughly fixed distance, and its results recover during
   that phase.

**Decisions:** keep the value warm-up (weight 0.5). Do not use `--target-kl`
as a step-size control: clipping already limits per-update distance, and
early stopping discards data. The next pilot should test the entropy bonus
directly: the value-warm-up protocol with a lower entropy coefficient.

## Phase 1c: entropy-bonus pilot (next)

This pilot tests finding 5, the entropy hypothesis. Protocol:

- the Phase 1 value-warm-up protocol with `--entropy-coefficient 0.001`, a
  tenth of the default 0.01;
- no target-KL;
- the same pool (`6203688`), seed, architecture and 30-update budget;
- the Phase 1 pilot as the reference.

0.001 was chosen over 0 because the target-KL pilot showed that without
upward pressure, entropy falls and results degrade. The question is whether
the bonus drives the steady climb, not whether any bonus is needed.

- Command: [`commands/entropy-pilot.txt`](commands/entropy-pilot.txt). It
  matches the Phase 1 pilot except for the run name and
  `--entropy-coefficient 0.001`. `--entropy-coefficient` is a new flag,
  defaulting to 0.01, the value every earlier run used.

Criteria, recorded before any Phase 1c data:

1. Rollout entropy over updates 20-29 stays clearly below the Phase 1 pilot's
   0.75-0.88, and does not fall like the target-KL pilot's (0.23-0.29).
2. On the policy-distance states (seed 7000), entropy does not double from
   version 9 (Phase 1 pilot: 0.28 to 0.58), and per-update step KL stays
   comparable to the Phase 1 pilot (0.024-0.031). Comparable step KL with
   lower entropy isolates the bonus as the cause.
3. Held-out results at versions 19 and 29 are at least as good as the Phase 1
   pilot's, both overall and by distance from version 9 (Phase 1 pilot at KL
   from v9 of 0.266 / 0.361: 63.4% / 66.7%; hard subset 27.1% / 29.7%).

If entropy stays low but results fall, the bonus is not what drives the drift
in results, even if it drives entropy. The next candidates are then the
matchup weighting or a teacher anchor.

### Phase 1c result: entropy coefficient 0.001

| | Phase 1 pilot | Entropy pilot |
| --- | --- | --- |
| rollout entropy, updates 12-21 | 0.40-0.75 | 0.36-0.50 |
| rollout entropy, updates 22-25 | 0.72-0.84 | 0.53-0.62 |
| rollout entropy, updates 26-29 | 0.78-0.88 | 0.71-0.84 |
| entropy on distance states, v9 / v19 / v29 | 0.278 / 0.515 / 0.581 | 0.278 / 0.357 / 0.595 |
| per-update step KL, updates 13-29 | 0.024-0.031 | 0.026-0.044 |
| aggregate, v19 / v29 | 63.4% / 66.7% | 65.5% / 65.1% |
| hard subset, v19 / v29 | 27.1% / 29.7% | 27.6% / 27.9% |
| monarch wins, v19 / v29 | 16 / 11 | 13 / 14 |
| KL from v9, v19 / v29 | 0.266 / 0.361 | 0.260 / 0.426 |

Criteria:

1. **Held only through about update 23.** Entropy then rose steeply and
   reached 0.84 by update 29, level with the Phase 1 pilot.
2. **Held at version 19** (0.357 against 0.515, at similar step KL), but not
   at version 29 (0.595, doubled from version 9).
3. **Passed within noise,** by update and at matched distance from version 9
   (v19: 65.5% at KL 0.260, against 63.4% at 0.266).

**Interpretation:** reducing the entropy bonus tenfold delayed the entropy
rise by about ten updates but did not prevent it, and it did not change
results. The bonus contributes to the early rate of rise, but finding 5
(the bonus acting on clipped late steps drives the climb) is not supported
as the main cause. In this pilot the late rise came with other changes over
the same updates, starting around updates 22-24:

- clip fraction 0.12 to 0.20;
- approximate KL 0.03 to 0.045;
- value explained variance 0.83 to 0.78;
- samples per update about 59k to 43k (fewer decisions per duel).

This is consistent with a behavioral shift driven by the policy gradient
itself, but that is not verified.

## Status after Phase 1

- **Adopted:** value warm-up at 0.5. It removes the first-update jump, which
  was the main cause of the early collapse.
- **Rejected:** `--target-kl` early stopping. It discards data, and clipping
  already limits per-update distance.
- **No effect:** lowering the entropy bonus to 0.001. The default 0.01 stays.
- **Not addressed:** the later problems (the control's dip around versions
  39-49, the Monarch collapse, the Lightsworn exploit) occur beyond the
  30-update pilot horizon.

**Next:** measure the `blackwing` policy's own per-matchup results on the
fixed pool (`9e5790e`) with the evaluation suite's seeds, then run a 100-update
value-warm-up baseline on that pool as the reference for the next
interventions, such as matchup weighting. A paired run without the value
warm-up is optional, because its mechanism is established.

## Teacher reference on the evaluation suite

`python -m ygo_training.teacher_evaluation` plays the evaluation suite's exact
games: the same seeds (`seed + case * 10,000,000 + game`), the same seat
alternation and the same Rock-Paper-Scissors handling. The `blackwing` policy
plays the learner seat. It runs locally without a checkpoint or GPU, in about
25 seconds for the whole suite. Seed 2012,
`eval_suites/ygo-policies-blackwing-screen.json`:

| Matchup | Teacher, `6203688` | Teacher, `9e5790e` |
| --- | ---: | ---: |
| aggregate | 1216-448 (73.1%) | 1189-475 (71.5%) |
| hard subset (pooled) | 147-237 (38.3%) | 134-250 (34.9%) |
| mirror | 67-61 (52.3%) | 67-61 (52.3%) |
| monarch | 24-104 (18.8%) | 15-113 (11.7%) |
| rock-block | 56-72 (43.8%) | 56-72 (43.8%) |
| lightsworn | 96-32 (75.0%) | 89-39 (69.5%) |
| infernity | 89-39 (69.5%) | 84-44 (65.6%) |
| gladiator | 101-27 (78.9%) | 97-31 (75.8%) |
| burn | 101-27 (78.9%) | 101-27 (78.9%) |
| crystal | 118-10 (92.2%) | 118-10 (92.2%) |
| dragunity | 125-3 (97.7%) | 125-3 (97.7%) |
| gishki | 127-1 (99.2%) | 127-1 (99.2%) |
| heroes | 78-50 (60.9%) | 78-50 (60.9%) |
| morphtronic | 127-1 (99.2%) | 127-1 (99.2%) |
| spellcaster | 107-21 (83.6%) | 105-23 (82.0%) |

- The teacher's mirror result (52.3%) is close to the 50% expected from a
  policy playing itself, a sanity check on the evaluation.
- On the evaluation seeds the teacher wins 18.8% against Monarch on the old
  pool and 11.7% on the fixed pool. The roughly 37% from earlier local
  policy-versus-policy games is outside the 95% range of this 128-game result
  (about 12-26%). Some setting of that local check differed, which has not
  been identified. Earlier notes in this record that cite 37% should be read
  against 18.8%.
- On the old pool, students trail the teacher mainly on the hard subset
  (teacher 38.3%; students 26-34% at version 9 and about 19-30% later). On
  some middle matchups PPO matched or passed the teacher, for example
  Gladiator 81.2% at control v89, and Rock-Block 46.1% at value-warm-up pilot
  v29. The Lightsworn results partly reflect the deck-out exploit.
- The fixed pool is harder for the teacher where its fixes apply: Monarch,
  Lightsworn, Infernity and Gladiator. The next baseline is therefore compared
  with the `9e5790e` column.
