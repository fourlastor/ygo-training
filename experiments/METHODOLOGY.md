# Training experiment and probe methodology

This document defines what each training diagnostic is for, what evidence it
can provide, and the parity checks required before its output is trusted.
Individual experiment records should link here instead of redefining the
general methodology. Experiment-specific hypotheses, seeds, protocols,
results, and conclusions still belong in their own files.

## Vocabulary

- **Checkpoint format** identifies the model and tensor schema. Format 7, for
  example, adds composite target-action tensors. It is unrelated to the number
  of puzzle or probe families.
- **Probe family** is one class of generated tactical state within a tool.
  `--samples-per-family 1024` means 1,024 examples of each such class.
- **Variant** is one member of a paired counterfactual. Three families with
  1,024 pairs and two variants per pair produce 6,144 OCGCore states.
- **Fixture** is a small deterministic scenario intended for regression or
  interpretation, not a population estimate.
- **Held-out evaluation** means complete argmax duels outside the sampled
  training rollouts. This remains the primary checkpoint-selection evidence.

## Evidence hierarchy

No single measurement is sufficient for promotion.

1. **Held-out argmax complete duels** establish whether the policy wins under
   the intended game distribution. Always report W-L-D, cutoff rate, deck
   assignment, game count, and combat diagnostics.
2. **Generated OCGCore tactical probes** estimate whether a behavior
   generalizes across constrained valid states. Report per-family counts,
   confidence intervals where applicable, seed, and relevant breakdowns.
3. **Paired counterfactual probes** establish whether changing one tactical
   fact moves logits and choices in the expected direction.
4. **Deterministic fixtures** provide readable failure traces and quick
   regression checks. They do not establish frequency.
5. **Human play** discovers missing scenarios and qualitative patterns. Turn
   observations into generated probes before treating frequency claims as
   statistical evidence.
6. **Gradient and pipeline audits** explain a behavior already localized by
   the preceding evidence. Gradient magnitude alone does not establish learned
   strategy.

Sampled training win rate and sampled-rollout combat rates are diagnostics,
not substitutes for held-out argmax evidence.

## Probe validity requirements

A checkpoint probe is valid only when its inference path is semantically
identical to production for the representation under test:

- load the correct checkpoint format into its matching model;
- obtain legal choices from OCGCore rather than inferring or filtering them;
- use the production visibility reducer, action conversion, combat
  augmentation, and neural masks;
- use the production immediate-attack expansion and submission path;
- pass every tensor used by production inference, including all format-7
  target-card tensors;
- preserve physical sequence only for the OCGCore response and mask it from
  neural inputs;
- preserve public battle position and normalized feature semantics;
- keep direct and packed tensorization equivalent;
- state whether monster metadata is natural, randomized, or synthetic;
- run a teacher or known-answer smoke where the tool supports it.

Whenever representation or action semantics change, audit every tool that
calls `PolicyValueNet` directly. Successful engine enumeration does not prove
inference parity. Old JSON output must be marked invalid if either legality or
inference parity was wrong.

## Complete-duel evaluation

Use `python -m ygo_training evaluate` and
`python -m ygo_training status ... --eval-only` for selection evidence.

Use it to answer:

- Does the checkpoint win complete duels?
- Are games terminating without decision cutoffs?
- Which broad combat failures occur in the evaluated distribution?
- Does the result survive both deck assignments or a mirror-deck control?

It cannot by itself show that a low error rate is robust. Favorable decks or
state distributions may simply avoid difficult choices. Composite
`target errors` also combine attacker selection, target selection, and
attack-versus-decline disagreement; use tactical probes to decompose them.

## Reproducibility and records

For every material probe result, record:

- checkpoint path, checkpoint format, and source run;
- git revision or the relevant uncommitted code state;
- tool and command;
- deck assignment;
- device, worker count, sample count, seed, and batch size;
- whether metadata was natural, randomized, or synthetic;
- output JSON path;
- validity status and any superseded reports.

Keep raw JSON outside the experiment index. Summarize only evidence that
changes an experimental decision in the relevant experiment file.
