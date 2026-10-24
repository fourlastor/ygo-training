# Individual board tokens with ygo-policies

**Status:** INVALIDATED

**Activated:** 2026-09-28

**Last updated:** 2026-09-28

**Completed:** 2026-09-28


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

Unchanged from
`2026-08-02-individual-board-tokens.md` (an earlier record, not included here):
does attending to individual visible board cards improve wide and multi-target
board decisions relative to format-12 mean pooling? That record's architecture,
required evidence, and selection rule apply here. Only the teacher, the
opponents, and the deck pool change.

## Why a new protocol

The earlier policy is no longer used for training. Imitation labels and
opponents now come from the rule-based per-deck policies in the
`native/ygo-policies` submodule (<https://github.com/fourlastor/ygo-policies>).
Each policy pilots the deck it was written for, so the opponent pool is exactly
that repository's deck set. This is a new lineage; nothing from the earlier
protocol is comparable.

## Integration

- The .NET environment exposes OCGCore's raw message stream
  (`CaptureRawMessages`, `DrainRawMessages`) and `QueryLocationRaw`.
- `ygo_training/ygo_policies.py` loads `libygo_policies.so` via ctypes. For each
  duel, one seat per policy-controlled player receives the whole unfiltered
  stream plus a synthesized `MSG_START`, and `MSG_UPDATE_DATA` for both players'
  hand, field, Graveyard, banished, and Extra Deck immediately before each of its
  prompts. Redaction happens inside the policy library, following the rules of
  EDOPro's server.
- A policy answer is OCGCore response bytes, mapped to the pending prompt's
  action by exact response bytes, or by index set for reordered card/sum
  selections. An answer that matches no OCGCore action fails the duel. OCGCore
  remains the only source of legal actions.
- `--deck-pool ygo-policies` builds the deck manifest from the policy catalog
  and uses policy ids as deck ids, because several deck names contain commas.
  The `blackwing` deck is card-for-card identical to the earlier `Blackwing
  Assassin` list.
- Composite attack expansion speculatively submits and cancels on the live duel,
  and the policy seats would see those messages. Policy opponents therefore
  require staged prompts, and this protocol uses `joint-staged`.

## Pinned opponent pool

Both arms use ygo-policies at submodule commit `6203688`. During the control
run, this commit was found to contain a Lightsworn deck bug that causes some of
its deck-out losses, plus several card-identity bugs. Both were fixed upstream
afterwards. The pointer stays at `6203688` so both arms face the same
opponents, so matchups those bugs affect, `blackwing-vs-lightsworn` especially,
are measured against flawed opponents. Any later experiment that moves the
pointer faces a different pool, so its results are not comparable with this
experiment.

The repository branch moved the pointer to `9e5790e` (Lightsworn deck-out
hardening plus card-identity fixes) while the treatment arm was still running,
to prepare the next experiment. That does not affect this experiment. The CUDA
training machine pulls only before a run starts, so both arms ran entirely on
`6203688`. Anyone reproducing this experiment must check out `6203688`.

## Frozen protocol

Both arms run 100 updates of 520 duels from identically seeded new format-12
models:

- updates 0-9: teacher-executed imitation. The `blackwing` policy plays the
  learner seat, and each opponent's own policy plays the other seat. Labels are
  the learner seat's choices among neural actions;
- updates 10-99: ordinary PPO against policy-driven opponents;
- the same balanced pool in both phases: all 13 policy decks, including the
  Blackwing mirror, 40 duels each per update, with no separate warm-up pool;
- evaluation every ten updates on
  [`eval_suites/ygo-policies-blackwing-screen.json`](../eval_suites/ygo-policies-blackwing-screen.json),
  128 argmax games against each policy deck.

Inherited explicit invariants: `joint-staged` transactions, OCGCore-published
current ATK/DEF, monster identities, declared decks, and
`action_value_auxiliary_weight=0`. The protocol has no checkpoint resume,
frozen learned opponent, counterfactual sidecar, puzzle training, or persistent
imitation.

Reviewed commands:

- [`commands/ygo-policies-board-tokens-control.txt`](commands/ygo-policies-board-tokens-control.txt)
- [`commands/ygo-policies-board-tokens-treatment.txt`](commands/ygo-policies-board-tokens-treatment.txt)

Before launching, build `YGO.TrainingWorker` (Release) and the policy library
(`cargo build --release --manifest-path native/ygo-policies/Cargo.toml -p
ygo-policies-ffi`), then from `training/`:

```sh
uv run python -m ygo_training.paired_run_preflight \
  --root .. \
  --control-command-file experiments/commands/ygo-policies-board-tokens-control.txt \
  --treatment-command-file experiments/commands/ygo-policies-board-tokens-treatment.txt \
  --output analysis/ygo-policies-board-token-preflight-seed-2012.json
```

After the first update of both arms, audit the same manifest with `--audit`, as
in the superseded record, before accepting either run as experimental data.

## Hard-matchup subset

This subset was chosen on 2026-09-28. At that point the only evaluated
checkpoint was the **control** arm's version 9, which had only imitation
training. No PPO checkpoint from either arm had been seen. The subset is the
three matchups the control student won less than 40% of the time:

- `blackwing-vs-blackwing` (the primary case);
- `blackwing-vs-monarch`;
- `blackwing-vs-rock-block`.

Report the pooled hard-subset win rate (3 × 128 = 384 games) for both arms at
every scheduled evaluation. It is the hard-matchup outcome named in the
selection rule. The aggregate is weighted down by three matchups already at
97-99%, and 128 games give each single case a 95% interval of about ±8 points.
Neither is sufficient alone. `blackwing-vs-heroes` (58.6%) is deliberately
excluded.

## Evidence

The development machine passed an end-to-end CPU smoke test (26 duels, one
imitation update, one PPO update, and a 3-case evaluation), which is not
experimental evidence. In the smoke test's imitation update, the teacher won
20 of 26 duels against the pool, a 77% win rate.

### Control arm, version 9 (end of imitation warm-up, before PPO)

The following numbers are for the **control** (`mean-pool`) arm only. The
treatment arm had not yet run. Warm-up rollouts for updates 0-9, with the
teacher executing, had training win rates of 68.3-74.8%. That is a rough
teacher reference, not a matched comparison, because it uses training seeds.
There were no decision-limit cutoffs. Deck-outs decided 8 losses and 9 wins.

| Matchup | W-L | Win rate |
| --- | --- | --- |
| aggregate | 1114-550 | 66.9% |
| blackwing (mirror, primary) | 48-80 | 37.5% |
| burn | 97-31 | 75.8% |
| crystal | 111-17 | 86.7% |
| dragunity | 124-4 | 96.9% |
| gishki | 126-2 | 98.4% |
| gladiator | 90-38 | 70.3% |
| heroes | 75-53 | 58.6% |
| infernity | 85-43 | 66.4% |
| lightsworn | 81-47 | 63.3% |
| monarch | 16-112 | 12.5% |
| morphtronic | 127-1 | 99.2% |
| rock-block | 46-82 | 35.9% |
| spellcaster | 88-40 | 68.8% |
| **hard subset (pooled)** | **110-274** | **28.6%** |

A mirror win rate of 37.5% means imitation alone left the student clearly
below its teacher: the policy playing itself would be expected to win about
50%. This checkpoint is the pre-PPO baseline for the control arm only. The
treatment arm's version 9 must be recorded separately.

*Correction, recorded before this section was otherwise changed:* this section
first stated that a large gap between the two arms' version-9 results would
indicate a setup error. That was wrong. The teacher executes every warm-up
duel, so both arms train on identical rollouts, and a version-9 gap measures
how well each representation copies the same teacher.

### Control arm, complete run (versions 9-99)

This section was recorded when the control finished and before the treatment
arm produced any evaluation. The hard subset is the pooled result of the
mirror, Monarch and Rock-Block. The other-ten column pools the remaining ten
matchups, 1,280 games.

| Version | Aggregate | Hard subset | Mirror | Monarch | Rock-Block | Other ten | Deck-out L/W |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 9 | 66.9% | 28.6% | 37.5% | 16 | 35.9% | 78.4% | 8/9 |
| 19 | 66.8% | 25.3% | 28.1% | 7 | 42.2% | 79.2% | 1/9 |
| 29 | 65.8% | 25.3% | 33.6% | 2 | 40.6% | 78.0% | 0/31 |
| 39 | 61.0% | 20.3% | 27.3% | 0 | 33.6% | 73.2% | 2/49 |
| 49 | 63.3% | 18.8% | 20.3% | 0 | 35.9% | 76.6% | 3/40 |
| 59 | 64.8% | 21.4% | 29.7% | 0 | 34.4% | 77.9% | 2/42 |
| 69 | 66.5% | 24.2% | 36.7% | 0 | 35.9% | 79.1% | 2/50 |
| 79 | 63.6% | 19.5% | 28.9% | 0 | 29.7% | 76.8% | 2/47 |
| 89 | 67.5% | 20.8% | 31.2% | 0 | 31.2% | 81.5% | 0/52 |
| 99 | 65.7% | 19.3% | 30.5% | 0 | 27.3% | 79.6% | 1/35 |

The Monarch column gives wins out of 128. Version 89 includes one draw, against
Burn. The best scheduled aggregate is version 89 at 67.5% (1123-540-1). The
endpoint is 65.7% (1093-571).

Control observations:

- **Ninety PPO updates did not beat the imitation checkpoint on the
  aggregate.** The aggregate stayed between 61.0% and 67.5%. PPO improved the
  other ten matchups by about 1-3 points, most clearly Gladiator (70.3% to
  78.9%), Burn and Crystal. It lost ground in every hard matchup, and the
  pooled hard subset fell from 28.6% to 19.3%.
- **Monarch collapsed and never recovered.** Wins fell from 16 to 7, then 2,
  then 0, and stayed at 0 from version 39 to 99. The student started at 12.5%,
  and the `blackwing` policy wins about 37% against Monarch in local
  policy-versus-policy games. With win/loss as the only reward, the student
  loses every Monarch duel, so PPO gets no signal about which plays would work
  there.
- **The student learned to exploit the pinned Lightsworn bug.** Deck-out wins
  came almost entirely from Lightsworn and rose from 9 to between 31 and 52.
  At version 69, 49 of the student's 89 Lightsworn wins came from Lightsworn
  decking itself out. The per-matchup breakdown was checked only up to version
  69. The Lightsworn win rate barely changed, so this is a change in how the
  student wins, driven by the bugged opponent, not new strength.
- **The trajectory was not monotonic.** The aggregate dropped at version 39,
  recovered by version 69, dropped again at version 79 and peaked at version
  89. The earlier action-value control and the opponent-diversity treatment
  also moved like this, by 5-8 points over 20-40 updates. Differences between
  the arms at any single evaluation are therefore weak evidence. Compare the
  full trajectories and the endpoints.

These are control-arm observations only. They do not yet say anything about
the representation question.

### Treatment arm, version 9 (end of imitation warm-up, before PPO)

The warm-up rows for updates 0-9 were identical in both arms: the same sample
counts (60,325 at update 0), training win rates and rewards. This confirms that
both arms learned from the same teacher-executed rollouts, so the version-9
evaluations differ only because of the representation.

| Matchup | Control v9 | Treatment v9 | Difference |
| --- | --- | --- | --- |
| aggregate | 1114-550 (66.9%) | 1190-474 (71.5%) | +4.6 pp |
| mirror (primary) | 48-80 (37.5%) | 62-66 (48.4%) | +10.9 pp |
| monarch | 16-112 (12.5%) | 14-114 (10.9%) | -1.6 pp |
| rock-block | 46-82 (35.9%) | 56-72 (43.8%) | +7.9 pp |
| **hard subset (pooled)** | 110-274 (28.6%) | 132-252 (34.4%) | +5.8 pp |
| other ten (pooled) | 1004-276 (78.4%) | 1058-222 (82.7%) | +4.3 pp |
| burn | 97-31 | 108-20 | +8.6 pp |
| crystal | 111-17 | 121-7 | +7.8 pp |
| dragunity | 124-4 | 126-2 | +1.6 pp |
| gishki | 126-2 | 126-2 | 0 |
| gladiator | 90-38 | 101-27 | +8.6 pp |
| heroes | 75-53 | 67-61 | -6.2 pp |
| infernity | 85-43 | 91-37 | +4.7 pp |
| lightsworn | 81-47 | 94-34 | +10.2 pp |
| morphtronic | 127-1 | 128-0 | +0.8 pp |
| spellcaster | 88-40 | 96-32 | +6.3 pp |

Treatment deck-outs decided 0 losses and 11 wins. Control deck-outs decided 8
losses and 9 wins.

The aggregate difference is about 2.9 standard errors. The treatment is ahead
in 10 of 13 matchups, tied against Gishki, and behind against Heroes and
Monarch. The treatment's 48.4% mirror result is close to the
roughly 50% expected from the teacher playing itself. This is evidence that
individual board tokens copy the teacher more faithfully from identical
imitation data. It is still a single seed, and it measures imitation fidelity,
not the PPO outcome that the selection rule judges. Whether the head start
survives PPO is the question for versions 19-99.

Imitation metrics on the warm-up rollouts support the same reading. The
treatment had lower `imitation_loss` at every update from 0 to 9, and higher
`imitation_model_teacher_agreement` at every update from 1 to 9. Update 0 is
measured from the random initialization.

| Update | Control loss / agreement | Treatment loss / agreement |
| ---: | --- | --- |
| 0 | 0.482 / 69.1% | 0.429 / 65.5% |
| 1 | 0.359 / 91.8% | 0.313 / 93.2% |
| 5 | 0.277 / 94.7% | 0.256 / 95.5% |
| 9 | 0.250 / 95.8% | 0.235 / 96.4% |

At update 9, the treatment disagrees with the teacher on about 14% fewer
decisions (3.6% versus 4.2%). Agreement is measured on training rollouts, while
the evaluation uses separate seeds. Both show the same direction.

### Treatment version 19 and PPO step size

This section was recorded when the treatment had reached version 19, before
any later treatment evaluation existed.

The treatment lost its imitation lead within the first ten PPO updates. It
fell from 71.5% to 60.8% (1011-653). The control was 66.8% at the same point.
The hard subset fell to 21.9% (control: 25.3%), and the other ten fell to 72.4%
(control: 79.2%). Monarch fell to 1 win.

PPO diagnostics for updates 10-19 show that the treatment's policy moved
further on every update:

| Update | Approx. KL, control / treatment | Clip fraction, control / treatment | Entropy, control / treatment |
| ---: | --- | --- | --- |
| 10 | 0.126 / 0.139 | 0.174 / 0.201 | 0.400 / 0.419 |
| 11 | 0.076 / 0.126 | 0.228 / 0.306 | 0.769 / 0.908 |
| 12 | 0.044 / 0.061 | 0.190 / 0.259 | 0.746 / 0.987 |
| 14 | 0.033 / 0.053 | 0.173 / 0.245 | 0.807 / 1.006 |
| 17 | 0.033 / 0.054 | 0.166 / 0.236 | 0.806 / 0.958 |
| 19 | 0.036 / 0.051 | 0.184 / 0.254 | 0.846 / 1.072 |

- The treatment's KL and clip fraction were higher than the control's at every
  update from 10 to 19. From update 11 on, its KL was about 1.4-1.7 times the
  control's. The learning rate and the rest of the protocol are identical, so
  the architecture itself makes each PPO step larger.
- Both arms started PPO with about the same entropy (0.40 and 0.42). A sharper
  imitation policy does not explain the difference. Entropy roughly doubled in
  the first PPO update in both arms, so PPO immediately discarded much of the
  imitated policy. After that the treatment kept getting more random, reaching
  1.07 by update 19, against 0.85 for the control.
- Both arms took large steps. KL above 0.1 in the first PPO update, then 0.03
  to 0.05, is well above the 0.01-0.02 usually targeted with PPO. This probably
  contributes to the non-monotonic trajectories in this and earlier
  experiments, but that has not been tested.

**Consequence for this experiment's conclusion:** with a fixed learning rate,
the representation change also changes the effective PPO step size. The paired
comparison therefore mixes two effects: individual tokens versus mean pooling,
and larger versus smaller policy steps. A treatment deficit at the end must not
be reported simply as individual tokens being worse. It may come from larger
steps rather than from the representation. The comparison stays as planned.
This caveat is part of how it will be interpreted.

**Implication for the next experiment:** control the step size directly, for
example by stopping each update's PPO epochs early once approximate KL passes
a fixed target (about 0.02), or by lowering the learning rate. This would make
the two representations take steps of comparable size, and it tests whether
smaller steps reduce the loss of the imitated policy and the mid-run swings in
both arms. It is cheaper to test than new matchup weighting, and it should be
settled first. Change only one thing per experiment.

### Treatment arm, complete run, and comparison

Both arms completed 100 updates. Aggregate held-out win rate at every
scheduled evaluation:

| Version | Control | Treatment | Treatment - control |
| ---: | ---: | ---: | ---: |
| 9 | 66.9% | 71.5% | +4.6 pp |
| 19 | 66.8% | 60.8% | -6.0 pp |
| 29 | 65.8% | 60.6% | -5.2 pp |
| 39 | 61.0% | 61.4% | +0.4 pp |
| 49 | 63.3% | 64.1% | +0.8 pp |
| 59 | 64.8% | 61.5% | -3.3 pp |
| 69 | 66.5% | 65.4% | -1.1 pp |
| 79 | 63.6% | 63.8% | +0.2 pp |
| 89 | 67.5% | 64.6% | -2.9 pp |
| 99 | 65.7% | 65.4% | -0.3 pp |

Across the nine PPO evaluations (versions 19-99), the control won 9,733 of
14,976 games (65.0%) and the treatment 9,443 (63.1%). These evaluations are not
independent, because the same evaluation seeds are used at every version and
consecutive checkpoints are related.

Endpoint, version 99:

| | Control | Treatment |
| --- | --- | --- |
| aggregate | 1093-571 (65.7%) | 1088-576 (65.4%) |
| hard subset (pooled) | 74-310 (19.3%) | 73-311 (19.0%) |
| other ten (pooled) | 1019-261 (79.6%) | 1015-265 (79.3%) |
| mirror | 39-89 (30.5%) | 45-83 (35.2%) |
| monarch | 0-128 | 0-128 |
| rock-block | 35-93 (27.3%) | 28-100 (21.9%) |
| lightsworn | 93-35 (72.7%) | 96-32 (75.0%) |
| deck-out losses / wins | 1 / 35 | 0 / 58 |

Hard-subset results for the treatment were transcribed only at versions 9
(34.4%), 19 (21.9%), 69 (24.2%, identical to the control) and 99 (19.0%).

Treatment deck-out wins rose late: 13-18 through version 69, then 42, 34 and
58 at versions 79-99. The per-matchup breakdown was checked only through
version 69, where the wins were almost all Lightsworn. The earlier observation
that the treatment exploited the pinned Lightsworn bug less therefore held
only until about version 69. The later rise is most likely the same exploit
found later, but that has not been verified.

## Conclusion

**The endpoint result is null.** At version 99 the two arms are level on the
aggregate (65.4% versus 65.7%), the hard subset (19.0% versus 19.3%) and the
other ten matchups. Averaged over the PPO evaluations the treatment was about
1.9 points behind, almost all of it from its larger early drop at versions
19-29. The selection rule requires a durable aggregate or hard-matchup
improvement, supported by board-width or legal-target-count strata, and
nothing here meets it. The required board-width and target-count strata were
not analyzed, because there was no aggregate difference for them to explain.

**Individual board tokens copy the teacher more faithfully.** From identical
imitation data, the treatment ended warm-up 4.6 points ahead (about 2.9
standard errors), with lower imitation loss and higher teacher agreement at
every update. That is a real representation effect in this run, but PPO
removed it.

**Why this is not evidence against individual tokens:** the comparison is
confounded by two problems that affect both arms and the treatment more:

1. The imitation warm-up never trained the value head. Explained variance at
   the first PPO update was -0.008 for the control and -0.17 for the
   treatment, so update 10 computed advantages from an untrained critic.
2. At a fixed learning rate, the treatment took larger PPO steps: about
   1.4-1.7 times the control's approximate KL in updates 11-19, and its
   entropy kept rising.

Both mechanisms hit hardest right at the start of PPO, which is where the
treatment lost its lead. The representation question should be run again only
after the start of PPO is stable.

Other findings carried forward:

- PPO did not beat the imitation checkpoint in either arm. Monarch fell to 0
  wins in both arms and never recovered. The hard subset ended at about 19% in
  both arms, compared with 28.6% (control) and 34.4% (treatment) after
  imitation.
- Both arms exploited the pinned Lightsworn deck-out bug. Later experiments
  use ygo-policies `9e5790e`, which reduces the bug but does not remove it.
- Mid-run swings of 5-8 points were again normal. Single evaluations are weak
  evidence.

**Next step:** [`2026-09-28-value-head-warmup.md`](2026-09-28-value-head-warmup.md)
tests training the value head during imitation, first with a single-arm pilot
compared against this experiment's control arm.
