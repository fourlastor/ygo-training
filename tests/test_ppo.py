import json
import math
import os
import random
import shutil
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
from tests.views import (
    ACTIVATE,
    HAND,
    MONSTER,
    NORMAL_SUMMON,
    PASS,
    SHOWN,
    SPELL_TRAP_ZONE,
    TRAP,
    batched,
    choice,
    named,
    row,
    step,
    view,
)
from ygo_training.checkpoints import (
    FORMAT_VERSION,
    discard,
    finish,
    load,
    load_inference_model,
    save,
    training_state_fingerprints,
    value_fingerprint,
    write,
)
from ygo_training.config import TrainerConfig
from ygo_training.model import PolicyValueNet
from ygo_training.optimization import PpoOptimizer, PreparedSample
from ygo_training.ppo import (
    PpoConfig,
    generalized_advantage,
    joint_ppo_loss_components,
    ppo_loss,
    ppo_loss_components,
    transaction_advantages,
)
from ygo_training.ppo_trainer import PpoTrainer
from ygo_training.rewards import OutcomeRewardModel
from ygo_training.run_repository import RunRepository
from ygo_training.schedules import (
    ROLLOUT_SEED_SLOTS,
    ROLLOUT_SEED_STRIDE,
    RUN_SEED_MULTIPLIER,
    TrainingSchedule,
    decayed_teacher_imitation_weight,
    episode_seed,
    policy_sample_seed,
    worker_rng_seed,
)


def test_terminal_reward_is_not_bootstrapped():
    advantages, returns = generalized_advantage(
        torch.tensor([1.0]), torch.tensor([0.2]), torch.tensor([1.0])
    )
    assert returns.item() == pytest.approx(1.0)
    assert advantages.item() == pytest.approx(0.8)


def test_full_trace_lambda_propagates_terminal_outcome_to_every_decision():
    advantages, returns = generalized_advantage(
        torch.tensor([0.0, 0.0, 1.0]),
        torch.zeros(3),
        torch.tensor([0.0, 0.0, 1.0]),
        gae_lambda=1.0,
    )
    assert advantages.tolist() == pytest.approx([1.0, 1.0, 1.0])
    assert returns.tolist() == pytest.approx([1.0, 1.0, 1.0])


def test_gae_lambda_attenuates_distant_terminal_outcomes():
    advantages, _ = generalized_advantage(
        torch.tensor([0.0, 0.0, 1.0]),
        torch.zeros(3),
        torch.tensor([0.0, 0.0, 1.0]),
        gae_lambda=0.5,
    )
    assert advantages.tolist() == pytest.approx([0.25, 0.5, 1.0])


def test_ppo_temperature_requires_a_positive_value():
    with pytest.raises(ValueError, match="rollout_temperature"):
        ppo_loss(
            torch.zeros((1, 2)),
            torch.tensor([0]),
            torch.tensor([0.0]),
            torch.tensor([1.0]),
            torch.tensor([0.0]),
            torch.tensor([0.0]),
            PpoConfig(rollout_temperature=0.0),
        )


def test_ppo_loss_components_sum_to_the_production_loss():
    logits = torch.tensor([[0.3, -0.2], [0.1, 0.4]])
    actions = torch.tensor([0, 1])
    old = torch.tensor([-0.5, -0.6])
    advantages = torch.tensor([0.7, -0.2])
    returns = torch.tensor([1.0, -1.0])
    values = torch.tensor([0.2, -0.4])
    config = PpoConfig(entropy_coefficient=0.03, value_coefficient=0.7)
    components = ppo_loss_components(logits, actions, old, advantages, returns, values, config)
    assert components["total"] == pytest.approx(
        components["policy"] + components["weighted_value"] + components["weighted_entropy"]
    )
    assert ppo_loss(logits, actions, old, advantages, returns, values, config) == pytest.approx(
        components["total"]
    )


def test_joint_ppo_multiplies_conditional_action_ratios_before_clipping():
    logits = torch.tensor([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    actions = torch.tensor([0, 1, 0])
    current = torch.distributions.Categorical(logits=logits).log_prob(actions)
    old = current - torch.tensor([math.log(1.1), math.log(1.2), math.log(0.9)])
    components = joint_ppo_loss_components(
        logits,
        actions,
        old,
        torch.tensor([0, 0, 1]),
        torch.tensor([0, 2]),
        torch.tensor([1.0, 1.0]),
        torch.tensor([0.0, 0.0]),
        torch.zeros(3),
        PpoConfig(entropy_coefficient=0.0, value_coefficient=0.0),
    )

    # First transaction ratio is 1.1 * 1.2 = 1.32 and is clipped to 1.2;
    # the singleton ratio is 0.9.
    assert components["policy"] == pytest.approx(-(1.2 + 0.9) / 2)
    assert components["clip_fraction"] == pytest.approx(0.5)


def test_ppo_sample_preparation_exposes_production_returns_and_advantages():
    model = torch.nn.Linear(1, 1)
    optimizer = PpoOptimizer(
        model,
        torch.optim.Adam(model.parameters()),
        TrainerConfig(duel_gae_lambda=0.95),
        OutcomeRewardModel(),
        "cpu",
    )
    samples, advantages, metrics = optimizer.prepare_samples(
        [
            {
                "terminal": True,
                "learner_player": 0,
                "winner": 0,
                "steps": [
                    {"player": 0, "value": 0.2, "name": "first"},
                    {"player": 0, "value": 0.3, "name": "last"},
                ],
            }
        ]
    )
    assert [sample[0]["name"] for sample in samples] == ["first", "last"]
    assert [sample[2] for sample in samples] == pytest.approx([0.965, 1.0])
    assert advantages.tolist() == pytest.approx([1.0, -1.0])
    assert metrics["samples"] == 2.0
    assert metrics["mean_win_reward"] == 1.0


def test_joint_staged_preparation_keeps_attack_members_as_one_optimization_unit():
    model = torch.nn.Linear(1, 1)
    optimizer = PpoOptimizer(
        model,
        torch.optim.Adam(model.parameters()),
        TrainerConfig(),
        OutcomeRewardModel(),
        "cpu",
    )
    samples, advantages, metrics = optimizer.prepare_samples(
        [
            {
                "terminal": True,
                "learner_player": 0,
                "winner": 0,
                "steps": [
                    {
                        "player": 0,
                        "value": 0.0,
                        "transaction": 4,
                    },
                    {
                        "player": 0,
                        "value": 0.0,
                        "transaction": 4,
                    },
                    {"player": 0, "value": 0.0},
                ],
            }
        ]
    )

    chunks = optimizer._chunks(samples, list(range(len(samples))), 2)
    assert any(set(chunk) == {0, 1} for chunk in chunks)
    assert advantages[0] == advantages[1]
    assert metrics["gae_unit"] == "transactions"
    assert metrics["policy_transactions"] == 2.0


def test_joint_staged_gae_skips_the_dependent_target_prompt():
    model = torch.nn.Linear(1, 1)
    optimizer = PpoOptimizer(
        model,
        torch.optim.Adam(model.parameters()),
        TrainerConfig(duel_gae_lambda=0.5),
        OutcomeRewardModel(),
        "cpu",
    )
    samples, advantages, _metrics = optimizer.prepare_samples(
        [
            {
                "terminal": True,
                "learner_player": 0,
                "winner": 0,
                "steps": [
                    {
                        "player": 0,
                        "value": 0.2,
                        "transaction": 4,
                    },
                    {
                        "player": 0,
                        # This prompt is part of the atomic attack transaction;
                        # its value must not enter transaction-level GAE.
                        "value": 100.0,
                        "transaction": 4,
                    },
                    {"player": 0, "value": 0.3},
                ],
            }
        ]
    )

    assert [sample.return_ for sample in samples] == pytest.approx([0.65, 0.65, 1.0])
    assert advantages.tolist() == pytest.approx([-1.0, -1.0, 1.0])
    assert _metrics["duel_raw_advantage_mean"] == pytest.approx(0.575)
    assert _metrics["duel_raw_advantage_std"] == pytest.approx(0.125)
    assert _metrics["duel_normalized_advantage_mean"] == pytest.approx(0.0)
    assert _metrics["duel_normalized_advantage_std"] == pytest.approx(1.0)


def test_causal_transaction_gae_uses_only_each_transaction_root_value():
    model = torch.nn.Linear(1, 1)
    optimizer = PpoOptimizer(
        model,
        torch.optim.Adam(model.parameters()),
        TrainerConfig(duel_gae_lambda=0.5),
        OutcomeRewardModel(),
        "cpu",
    )
    samples, advantages, metrics = optimizer.prepare_samples(
        [
            {
                "terminal": True,
                "learner_player": 0,
                "winner": 0,
                "steps": [
                    {
                        "player": 0,
                        "value": 0.2,
                        "transaction": 4,
                    },
                    {
                        "player": 0,
                        "value": 100.0,
                        "transaction": 4,
                    },
                    {
                        "player": 0,
                        "value": 0.3,
                        "transaction": 5,
                    },
                    {
                        "player": 0,
                        "value": 100.0,
                        "transaction": 5,
                    },
                    {
                        "player": 0,
                        "value": 0.4,
                        "transaction": 6,
                    },
                ],
            }
        ]
    )

    assert [sample.return_ for sample in samples] == pytest.approx(
        [0.5, 0.5, 0.7, 0.7, 1.0]
    )
    assert advantages[0] == advantages[1]
    assert advantages[2] == advantages[3]
    assert metrics["gae_unit"] == "transactions"
    assert metrics["policy_transactions"] == 3.0
    assert metrics["multi_step_policy_transactions"] == 2.0
    assert metrics["max_transaction_length"] == 2.0


def test_teacher_imitation_weight_decays_linearly():
    assert decayed_teacher_imitation_weight(0.02, 20, 0) == pytest.approx(0.02)
    assert decayed_teacher_imitation_weight(0.02, 20, 10) == pytest.approx(0.01)
    assert decayed_teacher_imitation_weight(0.02, 20, 20) == 0.0
    assert decayed_teacher_imitation_weight(0.02, 0, 20) == pytest.approx(0.02)


def test_terminal_outcome_target_is_separate_from_gae_return_and_shared_by_transaction():
    model = PolicyValueNet(embedding_dim=8, hidden_dim=16)
    optimizer = PpoOptimizer(
        model,
        torch.optim.Adam(model.parameters()),
        TrainerConfig(duel_gae_lambda=0.5),
        OutcomeRewardModel(),
        "cpu",
    )
    samples, _advantages, _metrics = optimizer.prepare_samples(
        [
            {
                "terminal": True,
                "learner_player": 0,
                "winner": 0,
                "steps": [
                    step(transaction=1),
                    step(transaction=1),
                    step(transaction=2),
                ],
            }
        ]
    )

    assert [sample.terminal_return for sample in samples] == [1.0, 1.0, 1.0]
    assert samples[0].return_ == pytest.approx(0.5)
    assert samples[1].return_ == pytest.approx(0.5)
    assert samples[2].return_ == pytest.approx(1.0)


def test_rollout_worker_episode_seeds_are_unique_over_training_scale():
    schedule = TrainingSchedule(seed=2011)
    engine_seeds = set()
    worker_seeds = set()
    policy_seeds = set()
    for update in range(200):
        _, rollout_seed, _, _ = schedule.next(
            teacher_weight=0.0,
            teacher_decay_updates=0,
            warmup_updates=0,
        )
        expected_slot = (2011 * RUN_SEED_MULTIPLIER + update) % ROLLOUT_SEED_SLOTS
        assert rollout_seed == expected_slot * ROLLOUT_SEED_STRIDE
        for episode in range(1024):
            seed = episode_seed(rollout_seed, episode)
            assert seed not in engine_seeds
            engine_seeds.add(seed)
        for worker in range(24):
            seed = worker_rng_seed(rollout_seed, worker)
            assert seed not in worker_seeds
            worker_seeds.add(seed)
        for worker in range(24):
            for sequence in range(32):
                sample_seed = policy_sample_seed(rollout_seed, worker, sequence)
                assert sample_seed not in policy_seeds
                policy_seeds.add(sample_seed)
    assert len(engine_seeds) == 200 * 1024
    assert len(worker_seeds) == 200 * 24
    assert len(policy_seeds) == 200 * 24 * 32
    assert max(engine_seeds) <= 0x7FFF_FFFF
    assert worker_seeds.isdisjoint(policy_seeds)

    second_run = TrainingSchedule(seed=2012)
    second_run_seeds = set()
    for _ in range(200):
        _, rollout_seed, _, _ = second_run.next(
            teacher_weight=0.0,
            teacher_decay_updates=0,
            warmup_updates=0,
        )
        for episode in range(1024):
            second_run_seeds.add(episode_seed(rollout_seed, episode))
    assert engine_seeds.isdisjoint(second_run_seeds)


def test_rotating_seed_coordinate_fits_declared_bounds_and_wraps():
    rollout_seed = (ROLLOUT_SEED_SLOTS - 1) * ROLLOUT_SEED_STRIDE
    assert episode_seed(rollout_seed, 1023) == 0x7FFF_FFFF
    assert worker_rng_seed(rollout_seed, 63) <= 0x7FFF_FFFF_FFFF_FFFF
    assert policy_sample_seed(rollout_seed, 63, 999_999) <= 0x7FFF_FFFF_FFFF_FFFF

    first = TrainingSchedule(seed=2011)
    first_seed = first.next(
        teacher_weight=0.0,
        teacher_decay_updates=0,
        warmup_updates=0,
    )[1]
    rotated = TrainingSchedule(seed=2011)
    rotated.set_rollout_index(ROLLOUT_SEED_SLOTS)
    rotated_seed = rotated.next(
        teacher_weight=0.0,
        teacher_decay_updates=0,
        warmup_updates=0,
    )[1]
    assert rotated_seed == first_seed

    with pytest.raises(ValueError, match="seed must be"):
        TrainingSchedule(seed=-1)


def test_training_checkpoint_restores_python_and_torch_rng_state(tmp_path):
    random.seed(123)
    torch.manual_seed(456)
    model = PolicyValueNet(embedding_dim=8, hidden_dim=16)
    optimizer = torch.optim.Adam(model.parameters())
    path = tmp_path / "checkpoint.pt"
    save(path, model, optimizer, 7, {"test": True})
    expected_python = random.random()
    expected_torch = torch.rand(4)

    random.seed(999)
    torch.manual_seed(999)
    restored = PolicyValueNet(embedding_dim=8, hidden_dim=16)
    restored_optimizer = torch.optim.Adam(restored.parameters())
    load(path, restored, restored_optimizer, map_location="cpu")

    assert random.random() == expected_python
    assert torch.equal(torch.rand(4), expected_torch)


def test_training_state_fingerprint_ignores_checkpoint_metadata(tmp_path):
    random.seed(123)
    torch.manual_seed(456)
    model = PolicyValueNet(embedding_dim=8, hidden_dim=16)
    optimizer = torch.optim.Adam(model.parameters())
    first = tmp_path / "first.pt"
    second = tmp_path / "second.pt"
    save(first, model, optimizer, 7, {"rollout_seconds": 1.0})
    save(second, model, optimizer, 7, {"rollout_seconds": 999.0})

    assert training_state_fingerprints(first) == training_state_fingerprints(second)


def test_value_fingerprint_is_stable_for_dict_order_and_sensitive_to_tensor_values():
    first = {"b": torch.tensor([1.0, 2.0]), "a": [3, 4]}
    reordered = {"a": [3, 4], "b": torch.tensor([1.0, 2.0])}
    changed = {"a": [3, 4], "b": torch.tensor([1.0, 2.5])}

    assert value_fingerprint(first) == value_fingerprint(reordered)
    assert value_fingerprint(first) != value_fingerprint(changed)


def test_minibatch_shuffle_depends_on_update_not_ambient_torch_rng():
    model = torch.nn.Linear(1, 1)
    optimizer = PpoOptimizer(
        model,
        torch.optim.Adam(model.parameters()),
        TrainerConfig(seed=2012),
        OutcomeRewardModel(),
        "cpu",
    )
    samples = [
        PreparedSample(
            step={"_transaction_key": ("sample", index)},
            advantage=0.0,
            return_=0.0,
        )
        for index in range(16)
    ]

    optimizer.begin_update(17)
    first = optimizer._chunks(samples, list(range(len(samples))), 4)
    torch.manual_seed(987654)
    torch.rand(100)
    optimizer.begin_update(17)
    repeated = optimizer._chunks(samples, list(range(len(samples))), 4)
    optimizer.begin_update(18)
    next_update = optimizer._chunks(samples, list(range(len(samples))), 4)

    assert repeated == first
    assert next_update != first


def test_rollout_estimates_are_recomputed_in_canonical_minibatches(monkeypatch):
    model = torch.nn.Linear(1, 1)
    optimizer = PpoOptimizer(
        model,
        torch.optim.Adam(model.parameters()),
        TrainerConfig(seed=2012, minibatch=2, rollout_temperature=1.0),
        OutcomeRewardModel(),
        "cpu",
    )
    steps = [
        {"chosen": index % 2, "log_prob": 99.0, "value": 99.0}
        for index in range(3)
    ]
    batch_sizes = []

    def forward_steps(_model, chunk):
        batch_sizes.append(len(chunk))
        return (
            torch.tensor([[0.0, 1.0]] * len(chunk)),
            torch.tensor([0.25] * len(chunk)),
        )

    monkeypatch.setattr(optimizer, "forward_steps", forward_steps)

    count = optimizer.recompute_rollout_estimates([{"steps": steps}])

    assert count == 3
    assert batch_sizes == [2, 1]
    assert [step["value"] for step in steps] == [0.25, 0.25, 0.25]
    assert all(step["log_prob"] != 99.0 for step in steps)


def test_resumed_rollout_index_advances_past_saved_updates():
    """The CLI calls this after loading a checkpoint, without spawning workers."""
    trainer = object.__new__(PpoTrainer)
    trainer._rollout_index = 0
    trainer.set_rollout_index(50)
    assert trainer._rollout_index == 50


def test_imitation_value_weight_zero_leaves_value_head_untrained():
    model, optimizer, trajectories = _value_warmup_setup(0.0)
    before = [parameter.detach().clone() for parameter in model.value.parameters()]

    metrics = optimizer.imitation_update(trajectories)

    assert metrics["imitation_value_weight"] == 0.0
    assert metrics["imitation_value_loss"] == 0.0
    for old, new in zip(before, model.value.parameters(), strict=True):
        assert torch.equal(old, new)


def test_imitation_value_weight_regresses_value_toward_duel_outcome():
    model, optimizer, trajectories = _value_warmup_setup(0.5)
    step = trajectories[0]["steps"][0]
    initial = float(optimizer.forward_steps(model, [step])[1][0])

    losses = [optimizer.imitation_update(trajectories)["imitation_value_loss"] for _ in range(30)]

    final = float(optimizer.forward_steps(model, [step])[1][0])
    # The learner seat won, so the target is +1.
    assert final > initial
    assert final > 0.5
    assert losses[-1] < losses[0]


def test_imitation_value_weight_requires_imitation_updates():
    with pytest.raises(ValueError, match="imitation-only"):
        TrainerConfig(imitation_value_weight=0.5)
    with pytest.raises(ValueError, match="non-negative"):
        TrainerConfig(imitation_only_updates=1, imitation_value_weight=-1.0)


_SQUARED_ERROR = torch.nn.functional.mse_loss


def _carried(optimizer, monkeypatch, update):
    """The values an update's value loss is taken at, where every step has one of its own."""
    taught = {"row": row(), "chosen": 0, "player": 0, "teacher": True}
    games = [
        {
            "terminal": True,
            "winner": winner,
            "steps": [{**taught, "transaction": number} for number in range(1, 7)],
        }
        for winner in (0, 1)
    ]
    marks = {
        id(step): (place + 1) / 100
        for place, step in enumerate(step for game in games for step in game["steps"])
    }
    attached = next(optimizer.model.parameters()).sum() * 0.0

    def forward_steps(_model, chunk):
        return (
            torch.zeros(len(chunk), 2) + attached,
            torch.tensor([marks[id(step)] for step in chunk]) + attached,
        )

    taken = []

    def measured(values, targets):
        taken.append(sorted(round(float(value), 2) for value in values.detach()))
        return _SQUARED_ERROR(values, targets)

    monkeypatch.setattr(optimizer, "forward_steps", forward_steps)
    monkeypatch.setattr(torch.nn.functional, "mse_loss", measured)
    optimizer.begin_update(update)
    return optimizer.imitation_update(games), taken


def test_a_duels_outcome_is_the_target_at_so_many_of_its_transactions(monkeypatch):
    optimizer = _optimizer(
        PolicyValueNet(embedding_dim=8, hidden_dim=16),
        ppo_epochs=1,
        minibatch=64,
        imitation_only_updates=1,
        imitation_value_weight=0.5,
        imitation_value_positions=2,
    )

    metrics, taken = _carried(optimizer, monkeypatch, 1)

    # Two games of six transactions: two of each carry the outcome.
    assert metrics["imitation_transactions"] == 12.0
    assert metrics["imitation_value_targets"] == 4.0
    assert [len(values) for values in taken] == [4]
    assert sum(value <= 0.06 for value in taken[0]) == 2
    # The head is still measured at every transaction, before the update.
    assert metrics["imitation_value_explained_variance"] != 0.0
    # The same update draws the same transactions, another update others.
    assert _carried(optimizer, monkeypatch, 1)[1] == taken
    assert any(_carried(optimizer, monkeypatch, update)[1] != taken for update in (2, 3, 4))


def test_every_transaction_carries_the_outcome_unless_told_otherwise(monkeypatch):
    optimizer = _optimizer(
        PolicyValueNet(embedding_dim=8, hidden_dim=16),
        ppo_epochs=1,
        minibatch=64,
        imitation_only_updates=1,
        imitation_value_weight=0.5,
    )

    metrics, taken = _carried(optimizer, monkeypatch, 1)

    assert metrics["imitation_value_targets"] == 12.0
    assert [len(values) for values in taken] == [12]
    # Won games first, lost games second, and a head that rises through both.
    assert metrics["imitation_value_correlation"] < 0.0


def test_a_value_head_that_only_reads_sends_nothing_into_the_state():
    torch.manual_seed(0)
    model = PolicyValueNet(embedding_dim=8, hidden_dim=16)
    tensor = batched(view(), view())

    model.evaluate(tensor, detach_value=True)[1].sum().backward()

    reached = {
        name.split(".")[0]
        for name, parameter in model.named_parameters()
        if parameter.grad is not None and float(parameter.grad.abs().sum()) > 0.0
    }
    assert reached == {"value"}
    model.zero_grad()
    model.evaluate(tensor)[1].sum().backward()
    assert float(model.state[0].weight.grad.abs().sum()) > 0.0


def test_a_value_head_that_only_reads_still_learns_the_outcome():
    torch.manual_seed(0)
    model = PolicyValueNet(embedding_dim=8, hidden_dim=16)
    optimizer = _optimizer(
        model,
        lr=1e-2,
        ppo_epochs=1,
        minibatch=8,
        imitation_only_updates=1,
        imitation_value_weight=0.5,
        imitation_value_detach=True,
    )
    taught = {"row": row(), "chosen": 0, "executed_chosen": 0, "player": 0, "teacher": True}
    trajectories = [{"terminal": True, "winner": 0, "steps": [dict(taught), dict(taught)]}]

    losses = [optimizer.imitation_update(trajectories)["imitation_value_loss"] for _ in range(30)]

    assert losses[-1] < losses[0]


def test_the_value_options_need_a_value_weight():
    with pytest.raises(ValueError, match="imitation_value_positions requires"):
        TrainerConfig(imitation_only_updates=1, imitation_value_positions=4)
    with pytest.raises(ValueError, match="imitation_value_detach requires"):
        TrainerConfig(imitation_only_updates=1, imitation_value_detach=True)
    with pytest.raises(ValueError, match="imitation_value_positions must be non-negative"):
        TrainerConfig(
            imitation_only_updates=1, imitation_value_weight=0.5, imitation_value_positions=-1
        )


def test_ppo_update_without_target_kl_runs_every_epoch():
    metrics = _kl_update(None)

    assert metrics["ppo_early_stopped"] == 0.0
    assert metrics["ppo_epochs_completed"] == 4.0
    assert metrics["ppo_optimizer_steps"] == 4.0
    assert metrics["ppo_target_kl"] == 0.0
    assert metrics["duel_final_approximate_kl"] >= 0.0


def test_ppo_update_stops_epochs_once_minibatch_kl_exceeds_target():
    metrics = _kl_update(1e-9)

    assert metrics["ppo_early_stopped"] == 1.0
    assert metrics["ppo_optimizer_steps"] < 4.0
    assert metrics["ppo_epochs_completed"] < 4.0
    assert metrics["ppo_target_kl"] == 1e-9


def test_target_kl_must_be_positive():
    with pytest.raises(ValueError, match="target_kl"):
        TrainerConfig(target_kl=0.0)


def test_entropy_coefficient_is_configurable_and_non_negative():
    assert TrainerConfig(entropy_coefficient=0.001).entropy_coefficient == 0.001
    with pytest.raises(ValueError, match="entropy_coefficient"):
        TrainerConfig(entropy_coefficient=-0.1)


def test_checkpoint_round_trip_preserves_step_and_model(tmp_path):
    model = PolicyValueNet(embedding_dim=8, hidden_dim=16)
    optimizer = torch.optim.Adam(model.parameters())
    path = tmp_path / "checkpoint.pt"
    save(path, model, optimizer, 7, {"test": True})
    assert torch.load(path, weights_only=False)["format_version"] == FORMAT_VERSION == 15
    restored = PolicyValueNet(embedding_dim=8, hidden_dim=16)
    payload = load(path, restored, map_location="cpu")
    assert payload["step"] == 7
    for original, recovered in zip(model.parameters(), restored.parameters()):
        assert torch.equal(original, recovered)


def _held(run):
    """The names of the files in a run's checkpoint folder."""
    return sorted(path.name for path in run.checkpoint_dir.iterdir())


def _latest(run):
    """The checkpoint a run goes on from, by `--continue` and by the scripts
    (the last `model-*.pt` by name), once every checkpoint and description
    a listing can find in its folder has been read."""
    for found in run.checkpoint_dir.glob("*.pt"):
        load(found, PolicyValueNet(embedding_dim=8, hidden_dim=16))
    for found in run.checkpoint_dir.glob("*.json"):
        json.loads(found.read_text())
    latest = run.latest_checkpoint()
    assert latest == max(run.checkpoint_dir.glob("model-*.pt"))
    return latest


@pytest.mark.parametrize("cut", ["checkpoint", "description"])
def test_a_save_cut_short_leaves_the_checkpoint_before_it_the_latest(tmp_path, monkeypatch, cut):
    model = PolicyValueNet(embedding_dim=8, hidden_dim=16)
    optimizer = torch.optim.Adam(model.parameters())
    run = RunRepository(tmp_path / "run")
    save(run.checkpoint(0), model, optimizer, 1, {})
    before = _held(run)
    assert before == ["model-00000.pt", "model-00000.pt.json"]
    killed = RunRepository(tmp_path / "killed")
    written = torch.save

    def stop(file):
        # Half of the file is there when the writing stops. A process killed
        # then leaves the run as it is copied here, and one interrupted goes
        # through what the save does about it.
        os.truncate(file, os.path.getsize(file) // 2)
        shutil.copytree(run.run_dir, killed.run_dir)
        raise KeyboardInterrupt

    def checkpoint_cut_short(payload, file):
        written(payload, file)
        stop(file)

    def description_cut_short(file, text):
        file.write_bytes(text.encode())
        stop(file)

    with monkeypatch.context() as patched, pytest.raises(KeyboardInterrupt):
        if cut == "checkpoint":
            patched.setattr(torch, "save", checkpoint_cut_short)
        else:
            patched.setattr(Path, "write_text", description_cut_short)
        save(run.checkpoint(1), model, optimizer, 2, {})

    # Interrupted, the save leaves nothing of its own.
    assert _held(run) == before
    assert _latest(run) == run.checkpoint(0)
    # Killed, it leaves what it had written, which is no checkpoint and no
    # description to anything that lists them...
    assert len(_held(killed)) > len(before)
    assert _latest(killed) == killed.checkpoint(0)
    # ...and which the run writes over when it goes on and saves again.
    save(killed.checkpoint(1), model, optimizer, 2, {})
    assert _held(killed) == [*before, "model-00001.pt", "model-00001.pt.json"]
    assert _latest(killed) == killed.checkpoint(1)


def test_a_checkpoint_written_takes_its_name_when_it_is_finished(tmp_path):
    model = PolicyValueNet(embedding_dim=8, hidden_dim=16)
    optimizer = torch.optim.Adam(model.parameters())
    run = RunRepository(tmp_path / "run")
    save(run.checkpoint(0), model, optimizer, 1, {})
    before = _held(run)

    # Written, it is there whole with its description, and no checkpoint yet
    # to anything that lists them.
    write(run.checkpoint(1), model, optimizer, 2, {})
    assert len(_held(run)) == len(before) + 2
    assert _latest(run) == run.checkpoint(0)
    discard(run.checkpoint(1))
    assert _held(run) == before

    write(run.checkpoint(1), model, optimizer, 2, {})
    finish(run.checkpoint(1))
    assert _held(run) == [*before, "model-00001.pt", "model-00001.pt.json"]
    assert _latest(run) == run.checkpoint(1)


def test_a_checkpoint_is_what_torch_writes_to_a_file_of_its_name(tmp_path):
    model = PolicyValueNet(embedding_dim=8, hidden_dim=16)
    path = tmp_path / "model-00007.pt"
    save(path, model, torch.optim.Adam(model.parameters()), 8, {"test": True})
    # Written under another name first, it is the same bytes all the same:
    # torch names the archive in a file after the name it writes it to.
    direct = tmp_path / "direct" / path.name
    direct.parent.mkdir()
    torch.save(torch.load(path, weights_only=False), direct)
    assert path.read_bytes() == direct.read_bytes()


def test_a_checkpoint_of_an_earlier_format_is_refused(tmp_path):
    model = PolicyValueNet(embedding_dim=8, hidden_dim=16)
    path = tmp_path / "old.pt"
    torch.save({"format_version": 12, "step": 1, "model": model.state_dict(), "metadata": {}}, path)
    with pytest.raises(ValueError, match="format 12 is not format 15"):
        load(path, model)


@pytest.mark.parametrize("earlier", [13, 14])
def test_a_checkpoint_that_read_the_deck_lists_is_refused(tmp_path, earlier):
    """Formats 13 and 14 read both deck lists, which format 15 dropped."""
    model = PolicyValueNet(embedding_dim=8, hidden_dim=16)
    path = tmp_path / "model-00003.pt"
    save(path, model, torch.optim.Adam(model.parameters()), 3, {})
    payload = torch.load(path, weights_only=False)
    payload["format_version"] = earlier
    torch.save(payload, path)
    with pytest.raises(ValueError, match=f"format {earlier} is not format 15"):
        load(path, PolicyValueNet(embedding_dim=8, hidden_dim=16))


def test_the_model_predicts_a_lead_for_every_answer_and_none_where_there_is_none():
    model = PolicyValueNet(embedding_dim=8, hidden_dim=16)
    tensor = batched(view(), view())
    logits, value, leads = model.evaluate(tensor)
    plain_logits, plain_value = model(tensor)
    assert torch.equal(logits, plain_logits) and torch.equal(value, plain_value)
    assert leads.shape == logits.shape
    assert torch.equal(leads[~tensor["action_mask"]], torch.zeros_like(leads[~tensor["action_mask"]]))
    assert leads[tensor["action_mask"]].abs().sum() > 0


def test_a_checkpoint_keeps_its_board_representation(tmp_path):
    control = PolicyValueNet(embedding_dim=8, hidden_dim=16, board_representation="mean-pool")
    path = tmp_path / "mean-pool.pt"
    save(path, control, torch.optim.Adam(control.parameters()), 1, {})

    load(path, control, expected_board_representation="mean-pool")
    with pytest.raises(ValueError, match="Checkpoint board representation"):
        load(path, control, expected_board_representation="individual-tokens")
    with pytest.raises(ValueError, match="but the model uses"):
        save(path, control, torch.optim.Adam(control.parameters()), 1, {
            "board_representation": "individual-tokens"
        })


def test_the_lstm_given_a_packed_batch_reads_the_padded_batchs_history():
    """The seat's earlier answers go to the LSTM packed, which a card works
    out its standard way.  On a processor the two are one code: what the
    model reads is what the LSTM gives the padded batch, to the last digit,
    whatever the rows' lengths and the length they are padded to."""
    from ygo_training.schema import HISTORY_SCALAR_COUNT

    torch.manual_seed(7)
    model = PolicyValueNet()
    generator = torch.Generator().manual_seed(11)
    for lengths, steps in (([0, 1, 6, 3, 6], 6), ([1, 0], 1), ([32], 32)):
        rows = len(lengths)
        tensor = {
            "history_lengths": torch.tensor(lengths),
            "history_card": torch.randint(0, 2000, (rows, steps), generator=generator),
            "history_location": torch.randint(0, 8, (rows, steps), generator=generator),
            "history_kind": torch.randint(0, 8, (rows, steps), generator=generator),
            "history_scalars": torch.randn((rows, steps, HISTORY_SCALAR_COUNT), generator=generator),
        }
        padded, _ = model.history(
            torch.cat(
                (
                    model.card_identity(tensor["history_card"]),
                    model.location_embedding(tensor["history_location"]),
                    model.action_kind_embedding(tensor["history_kind"]),
                    tensor["history_scalars"],
                ),
                dim=-1,
            )
        )
        last = padded[torch.arange(rows), tensor["history_lengths"].clamp_min(1) - 1]
        read = model._history(tensor)
        assert read.shape == (rows, 192)
        assert torch.equal(read, last * (tensor["history_lengths"] > 0).unsqueeze(-1))
        # A row with no earlier answer reads nothing.
        for place, length in enumerate(lengths):
            assert bool(read[place].any()) == (length > 0)


def test_inference_loader_builds_the_model_a_checkpoint_describes(tmp_path):
    source = PolicyValueNet(board_representation="individual-tokens")
    path = tmp_path / "individual.pt"
    save(path, source, torch.optim.Adam(source.parameters()), 1, {})

    restored, payload = load_inference_model(path)

    assert restored.board_representation == "individual-tokens"
    assert not restored.training
    assert payload["step"] == 1
    assert all(
        torch.equal(expected, actual)
        for expected, actual in zip(source.parameters(), restored.parameters(), strict=True)
    )


def test_a_checkpoint_with_another_input_schema_is_refused(tmp_path):
    model = PolicyValueNet(embedding_dim=8, hidden_dim=16)
    path = tmp_path / "checkpoint.pt"
    save(path, model, torch.optim.Adam(model.parameters()), 1, {})
    payload = torch.load(path, weights_only=False)
    payload["metadata"]["policy_input_schema"]["packed_schema"]["limits"]["cards"] = 1
    torch.save(payload, path)
    with pytest.raises(ValueError, match="digest mismatch"):
        load(path, model)


def _optimizer(model, **config):
    optimizer = PpoOptimizer(
        model,
        torch.optim.Adam(model.parameters(), lr=config.pop("lr", 1e-3)),
        TrainerConfig(**config),
        OutcomeRewardModel(),
        "cpu",
    )
    # The steps' decisions are kept where the tests' views are.
    optimizer.kept = SHOWN
    return optimizer


def test_imitation_update_reports_teacher_execution_agreement():
    optimizer = _optimizer(PolicyValueNet(embedding_dim=8, hidden_dim=16), ppo_epochs=1, minibatch=8)
    taught = {"row": row(), "chosen": 0, "player": 0, "teacher": True}

    metrics = optimizer.imitation_update(
        [{"steps": [{**taught, "executed_chosen": 0}, {**taught, "executed_chosen": 1}]}]
    )

    assert metrics["samples"] == 2.0
    assert metrics["imitation_teacher_agreement"] == pytest.approx(0.5)
    assert 0.0 <= metrics["imitation_model_teacher_agreement"] <= 1.0
    assert metrics["warmup"] == 1.0


def test_imitation_counts_a_transaction_once():
    """The steps of a transaction share one term of the imitation loss."""
    logits = torch.tensor([[2.0, 0.0], [0.0, 2.0], [1.0, 1.0]])
    chosen = torch.tensor([0, 1, 0])
    each = torch.nn.functional.cross_entropy(logits, chosen, reduction="none")
    apart = [{"_transaction_key": index} for index in range(3)]
    joined = [{"_transaction_key": 0}, {"_transaction_key": 0}, {"_transaction_key": 1}]

    assert PpoOptimizer._joint_imitation_loss(logits, chosen, apart) == pytest.approx(
        float(each.mean())
    )
    assert PpoOptimizer._joint_imitation_loss(logits, chosen, joined) == pytest.approx(
        float((each[0] + each[1] + each[2]) / 2)
    )


def test_a_teachers_distribution_is_crossed_only_over_the_answers_there_are():
    """A decision beside a longer one has no answer in the last column: the
    cross-entropy takes nothing from it, and is the teacher's entropy when
    the model gives the answers what the teacher does."""
    none = float("-inf")
    taught = torch.tensor([[1.0, 0.0, -1.0], [0.5, -0.5, none]])
    logits = torch.tensor([[0.2, 0.1, 0.0], [-1.0, 2.0, none]], requires_grad=True)
    probabilities = torch.softmax(taught, dim=-1)
    entropy = -(probabilities[0] * probabilities[0].log()).sum()

    same = PpoOptimizer._teacher_cross_entropy(taught, taught)
    crossed = PpoOptimizer._teacher_cross_entropy(logits, taught)

    assert float(same[0]) == pytest.approx(float(entropy))
    # Two answers, as likely as the teacher holds them: nothing of a third.
    two = torch.softmax(taught[1, :2], dim=-1)
    assert float(same[1]) == pytest.approx(float(-(two * two.log()).sum()))
    alone = -(two * torch.log_softmax(logits[1, :2].detach(), dim=-1)).sum()
    assert float(crossed[1]) == pytest.approx(float(alone))
    assert (crossed > same).all()
    crossed.sum().backward()
    assert torch.isfinite(logits.grad).all() and float(logits.grad[1, 2]) == 0.0


def _taught_by_a_network(targets: str):
    """A learner, a teacher that is another network, and a game the teacher
    played: decisions of two answers and of three, each a transaction."""
    torch.manual_seed(3)
    teacher = PolicyValueNet(embedding_dim=8, hidden_dim=16).eval()
    # A network as it is first drawn holds every answer about as likely:
    # this one is given preferences.
    with torch.no_grad():
        teacher.score[-1].weight.mul_(150)
    torch.manual_seed(4)
    model = PolicyValueNet(embedding_dim=8, hidden_dim=16)
    optimizer = _optimizer(
        model,
        lr=1e-2,
        ppo_epochs=1,
        minibatch=8,
        imitation_only_updates=1,
        pool_threads=1,
        imitation_teacher="teacher.pt",
        imitation_teacher_targets=targets,
    )
    optimizer.teachers = {optimizer.config.learner_deck: teacher}
    three = [
        choice(ACTIVATE, named(code=TRAP, location=SPELL_TRAP_ZONE)),
        choice(NORMAL_SUMMON, named(code=MONSTER, location=HAND)),
        choice(PASS),
    ]
    decisions = [view(), view(choices=three), view(turn=3), view(choices=three, turn=5)]
    with torch.no_grad():
        logits, _ = teacher(batched(*decisions))
    steps = [
        {
            "row": SHOWN.row(decision),
            "chosen": first,
            "executed_chosen": first,
            "player": 0,
            "teacher": True,
            "transaction": number,
        }
        for number, (decision, first) in enumerate(zip(decisions, logits.argmax(-1).tolist()))
    ]
    return model, teacher, optimizer, [{"terminal": True, "winner": 0, "steps": steps}], logits


@pytest.mark.parametrize("targets", ["choice", "distribution"])
def test_a_network_is_imitated_by_its_choice_or_by_its_distribution(targets):
    model, teacher, optimizer, game, taught = _taught_by_a_network(targets)
    before = value_fingerprint(teacher.state_dict())
    steps = game[0]["steps"]
    with torch.no_grad():
        logits, _ = model(SHOWN([step["row"] for step in steps]))
    probabilities = torch.softmax(taught, dim=-1)
    offered = torch.isfinite(taught)
    logarithms = torch.log_softmax(logits, dim=-1).masked_fill(~offered, 0.0)
    crossed = -(probabilities * logarithms).sum(-1)
    entropy = -(probabilities * probabilities.log().masked_fill(~offered, 0.0)).sum(-1)
    chosen = torch.tensor([step["chosen"] for step in steps])
    expected = {
        "choice": torch.nn.functional.cross_entropy(logits, chosen),
        "distribution": crossed.mean(),
    }[targets]

    first = optimizer.imitation_update(game)

    # One batch, every step a transaction: the loss is the mean of theirs.
    assert first["imitation_loss"] == pytest.approx(float(expected), rel=1e-5)
    assert first["imitation_teacher_entropy"] == pytest.approx(float(entropy.mean()), rel=1e-5)
    assert first["imitation_teacher_divergence"] == pytest.approx(
        float((crossed - entropy).mean()), rel=1e-4, abs=1e-6
    )
    assert first["imitation_teacher_entropy"] > 0.3
    last = [optimizer.imitation_update(game) for _ in range(200)][-1]
    assert last["imitation_model_teacher_agreement"] == 1.0
    if targets == "distribution":
        # The model comes to give every answer what the teacher gives it.
        assert last["imitation_teacher_divergence"] < 0.01 * first["imitation_teacher_divergence"]
        assert last["imitation_loss"] == pytest.approx(last["imitation_teacher_entropy"], rel=0.01)
    else:
        # Sure of the teacher's choice, where the teacher is not.
        assert last["imitation_loss"] < 0.05
        assert last["imitation_teacher_divergence"] > 0.3
    assert value_fingerprint(teacher.state_dict()) == before
    assert not teacher.training


@pytest.mark.parametrize(("targets", "asked"), [("choice", 1), ("distribution", 2)])
def test_the_training_pass_asks_the_teacher_only_for_its_distribution(targets, asked):
    """What the teacher gives the answers is measured before the update and
    is the target of a distribution: a choice is the one the game played,
    and the teacher is not asked again for it."""
    _, teacher, optimizer, game, _ = _taught_by_a_network(targets)
    calls = []
    teacher.register_forward_hook(lambda *_: calls.append(1))
    optimizer.imitation_update(game)
    # One minibatch: once to measure, and once more to learn its distribution.
    assert len(calls) == asked


def test_each_deck_is_set_against_its_own_teacher():
    """A game with a network in both seats, a deck in each: each seat's
    decisions are set against the teacher of its deck, by what it gives
    every answer."""
    torch.manual_seed(3)
    mine = PolicyValueNet(embedding_dim=8, hidden_dim=16).eval()
    torch.manual_seed(5)
    theirs = PolicyValueNet(embedding_dim=8, hidden_dim=16).eval()
    with torch.no_grad():
        mine.score[-1].weight.mul_(150)
        theirs.score[-1].weight.mul_(-150)
    three = [
        choice(ACTIVATE, named(code=TRAP, location=SPELL_TRAP_ZONE)),
        choice(NORMAL_SUMMON, named(code=MONSTER, location=HAND)),
        choice(PASS),
    ]
    decisions = [view(), view(choices=three), view(turn=3), view(choices=three, turn=5)]
    players = [0, 1, 0, 1]
    with torch.no_grad():
        first, _ = mine(batched(*decisions))
        second, _ = theirs(batched(*decisions))
    taught = torch.stack(
        [(first if player == 0 else second)[place] for place, player in enumerate(players)]
    )
    steps = [
        {
            "row": SHOWN.row(decision),
            "chosen": int(answer),
            "executed_chosen": int(answer),
            "player": player,
            "teacher": True,
            "transaction": number,
        }
        for number, (decision, player, answer) in enumerate(
            zip(decisions, players, taught.argmax(-1).tolist(), strict=True)
        )
    ]
    game = {
        "terminal": True,
        "winner": 0,
        "learner_player": 0,
        "learner_deck": "blackwing",
        "opponent_deck": "cyber",
        "steps": steps,
    }

    def update(teachers):
        torch.manual_seed(4)
        optimizer = _optimizer(
            PolicyValueNet(embedding_dim=8, hidden_dim=16),
            lr=1e-2,
            ppo_epochs=1,
            minibatch=8,
            imitation_only_updates=1,
            pool_threads=1,
            learner_decks=["blackwing", "cyber"],
            opponent_decks=["burn"],
            imitation_teachers={"blackwing": "mine.pt", "cyber": "theirs.pt"},
            imitation_teacher_targets="distribution",
        )
        optimizer.teachers = teachers
        return optimizer.imitation_update([game])

    both = update({"blackwing": mine, "cyber": theirs})
    probabilities = torch.softmax(taught, dim=-1)
    offered = torch.isfinite(taught)
    entropy = -(probabilities * probabilities.log().masked_fill(~offered, 0.0)).sum(-1)
    assert both["imitation_teacher_entropy"] == pytest.approx(float(entropy.mean()), rel=1e-5)
    assert both["imitation_samples_by_deck"] == {"blackwing": 2.0, "cyber": 2.0}
    # Had the one teacher answered for both decks, the lesson would be another.
    alone = update({"blackwing": mine, "cyber": mine})
    assert alone["imitation_teacher_entropy"] != pytest.approx(both["imitation_teacher_entropy"])
    assert alone["imitation_loss"] != pytest.approx(both["imitation_loss"])
    with pytest.raises(RuntimeError, match="No teacher plays cyber"):
        update({"blackwing": mine})


def test_individual_board_tokens_run_through_the_optimizer_forward():
    model = PolicyValueNet(embedding_dim=8, hidden_dim=16, board_representation="individual-tokens")
    optimizer = _optimizer(model, board_representation="individual-tokens")

    logits, values = optimizer.forward_steps(model, [step(), step()])

    assert logits.shape == (2, 2)
    assert values.shape == (2,)
    assert torch.isfinite(logits).all() and torch.isfinite(values).all()


def _value_warmup_setup(value_weight: float):
    torch.manual_seed(0)
    model = PolicyValueNet(embedding_dim=8, hidden_dim=16)
    optimizer = _optimizer(
        model,
        lr=1e-2,
        ppo_epochs=1,
        minibatch=8,
        imitation_only_updates=1,
        imitation_value_weight=value_weight,
    )
    taught = {"row": row(), "chosen": 0, "executed_chosen": 0, "player": 0, "teacher": True}
    trajectories = [{"terminal": True, "winner": 0, "steps": [dict(taught), dict(taught)]}]
    return model, optimizer, trajectories


def test_the_imitation_value_head_is_measured_where_it_is_trained(monkeypatch):
    """A transaction's later steps have no value target, and no say in the
    head's explained variance."""
    model = PolicyValueNet(embedding_dim=8, hidden_dim=16)
    optimizer = _optimizer(
        model, ppo_epochs=1, minibatch=64, imitation_only_updates=1, imitation_value_weight=0.5
    )
    taught = {"row": row(), "chosen": 0, "player": 0, "teacher": True}
    # Each game: a command and a selection it leads to, then another command.
    game = [{**taught, "transaction": 1}, {**taught, "transaction": 1}, {**taught, "transaction": 2}]
    won = {"terminal": True, "winner": 0, "steps": [dict(step) for step in game]}
    lost = {"terminal": True, "winner": 1, "steps": [dict(step) for step in game]}
    # The head is right at the first step of every transaction and as wrong
    # as it can be at the selection.
    values = {id(won): [1.0, -1.0, 1.0], id(lost): [-1.0, 1.0, -1.0]}
    predicted = {
        id(step): value
        for trajectory in (won, lost)
        for step, value in zip(trajectory["steps"], values[id(trajectory)], strict=True)
    }
    attached = next(model.parameters()).sum() * 0.0

    def forward_steps(_model, chunk):
        return (
            torch.zeros(len(chunk), 2) + attached,
            torch.tensor([predicted[id(step)] for step in chunk]) + attached,
        )

    monkeypatch.setattr(optimizer, "forward_steps", forward_steps)

    metrics = optimizer.imitation_update([won, lost])

    assert metrics["samples"] == 6.0
    assert metrics["imitation_transactions"] == 4.0
    assert metrics["imitation_value_explained_variance"] == pytest.approx(1.0)
    assert metrics["imitation_value_loss"] == pytest.approx(0.0)


def _lead_setup(**config):
    torch.manual_seed(0)
    model = PolicyValueNet(embedding_dim=8, hidden_dim=16)
    optimizer = _optimizer(
        model,
        lr=1e-2,
        ppo_epochs=1,
        minibatch=8,
        imitation_only_updates=1,
        imitation_records="rows.jsonl",
        **config,
    )
    taught = {"row": row(), "chosen": 0, "executed_chosen": 0, "player": 0, "teacher": True}
    # A search measured the second answer half a win ahead of the pilot's.
    measured = {**taught, "lead_targets": [[0, 0.0], [1, 0.5]]}
    trajectories = [{"terminal": True, "winner": 0, "steps": [dict(measured), dict(taught)]}]
    return model, optimizer, trajectories


def _leads(model, optimizer, trajectories):
    step = trajectories[0]["steps"][0]
    with torch.no_grad():
        return optimizer.evaluate_steps(model, [step])[2][0, :2].tolist()


def test_the_lead_head_is_measured_and_left_alone_without_a_weight():
    model, optimizer, trajectories = _lead_setup()
    before = [parameter.detach().clone() for parameter in model.lead.parameters()]

    metrics = optimizer.imitation_update(trajectories)

    assert metrics["imitation_lead_weight"] == 0.0
    assert metrics["imitation_lead_targets"] == 2.0
    assert metrics["imitation_lead_loss"] > 0.0
    assert metrics["imitation_shared_gradient_norm"] > 0.0
    assert metrics["imitation_lead_shared_gradient_norm"] > 0.0
    for old, new in zip(before, model.lead.parameters(), strict=True):
        assert torch.equal(old, new)
    # Duels without a search's log say nothing of leads.
    for taught in trajectories[0]["steps"]:
        taught.pop("lead_targets", None)
    assert not [key for key in optimizer.imitation_update(trajectories) if "lead" in key]


def test_a_lead_weight_teaches_the_head_what_the_search_measured():
    model, optimizer, trajectories = _lead_setup(imitation_lead_weight=1.0)
    first = _leads(model, optimizer, trajectories)

    losses = [optimizer.imitation_update(trajectories)["imitation_lead_loss"] for _ in range(60)]

    pilot, other = _leads(model, optimizer, trajectories)
    assert losses[-1] < losses[0] / 10
    assert abs(pilot) < 0.1 and abs(other - 0.5) < 0.1
    assert abs(first[1] - 0.5) > abs(other - 0.5)


def test_a_detached_lead_head_learns_without_pulling_on_what_it_reads():
    model, optimizer, trajectories = _lead_setup(
        imitation_lead_weight=1.0, imitation_lead_detach=True
    )
    before = [parameter.detach().clone() for parameter in model.lead.parameters()]

    metrics = optimizer.imitation_update(trajectories)

    assert metrics["imitation_lead_shared_gradient_norm"] == 0.0
    assert metrics["imitation_shared_gradient_norm"] > 0.0
    assert any(
        not torch.equal(old, new)
        for old, new in zip(before, model.lead.parameters(), strict=True)
    )


def test_the_lead_options_need_recorded_duels():
    with pytest.raises(ValueError, match="imitation_lead_weight requires imitation_records"):
        TrainerConfig(imitation_only_updates=1, imitation_lead_weight=1.0)
    with pytest.raises(ValueError, match="imitation_lead_detach requires"):
        TrainerConfig(imitation_only_updates=1, imitation_records="rows.jsonl", imitation_lead_detach=True)
    with pytest.raises(ValueError, match="imitation_label requires imitation_records"):
        TrainerConfig(imitation_only_updates=1, imitation_label="pilot")
    with pytest.raises(ValueError, match="imitation_label must be"):
        TrainerConfig(imitation_only_updates=1, imitation_records="rows.jsonl", imitation_label="search")
    with pytest.raises(ValueError, match="imitation_lead_weight must be non-negative"):
        TrainerConfig(imitation_only_updates=1, imitation_records="rows.jsonl", imitation_lead_weight=-1.0)


def _kl_update(target_kl: float | None) -> dict:
    torch.manual_seed(0)
    model = PolicyValueNet(embedding_dim=8, hidden_dim=16)
    optimizer = _optimizer(model, lr=0.05, ppo_epochs=4, minibatch=64, target_kl=target_kl)
    steps = [step(chosen=index % 2) for index in range(6)]
    return optimizer.update(
        [{"terminal": True, "learner_player": 0, "winner": 0, "steps": steps}]
    )


def test_teacher_labels_join_the_ppo_objective_by_their_weight():
    torch.manual_seed(0)
    model = PolicyValueNet(embedding_dim=8, hidden_dim=16)
    optimizer = _optimizer(model, ppo_epochs=1, minibatch=64, teacher_imitation_weight=0.1)
    labelled = [step(chosen=index % 2, teacher_chosen=1) for index in range(4)]
    trajectory = {"terminal": True, "learner_player": 0, "winner": 0, "steps": [*labelled, step()]}

    labels = optimizer.teacher_label_metrics([trajectory])
    metrics = optimizer.update([trajectory], 0.1)

    assert labels["teacher_label_coverage"] == pytest.approx(0.8)
    assert labels["teacher_sampled_agreement"] == pytest.approx(0.5)
    assert metrics["teacher_imitation_samples"] == 4.0
    assert metrics["teacher_imitation_loss"] > 0.0
    assert optimizer.update([trajectory], 0.0)["teacher_imitation_samples"] == 0.0


def test_imitation_of_recorded_duels_reports_where_they_left_the_pilot():
    optimizer = _optimizer(PolicyValueNet(embedding_dim=8, hidden_dim=16), ppo_epochs=1, minibatch=8)
    taught = {"row": row(), "player": 0, "teacher": True}
    kept = {**taught, "chosen": 0, "executed_chosen": 0, "pilot_chosen": 0}
    left = {**taught, "chosen": 1, "executed_chosen": 1, "pilot_chosen": 0}
    # A later step of a selection answered another way: the pilot was not asked.
    unread = {**taught, "chosen": 1, "executed_chosen": 1}

    metrics = optimizer.imitation_update([{"steps": [kept, dict(left), dict(left), unread]}])

    assert metrics["samples"] == 4.0
    assert metrics["imitation_departures"] == 2.0
    followed = metrics["imitation_model_departure_agreement"]
    refused = metrics["imitation_model_departure_pilot_agreement"]
    # Two answers and the same input twice: the model gives one of them at both.
    assert {followed, refused} == {0.0, 1.0}

    # The pilot's own recorded games leave it nowhere.
    metrics = optimizer.imitation_update([{"steps": [dict(kept), dict(kept)]}])
    assert metrics["imitation_departures"] == 0.0
    assert "imitation_model_departure_agreement" not in metrics
    # Games the pilot plays now say nothing of it.
    metrics = optimizer.imitation_update([{"steps": [{**taught, "chosen": 0}]}])
    assert "imitation_departures" not in metrics


def test_the_advantages_worked_out_on_the_processor_are_the_tensors_to_the_last_bit():
    """Every length, discount and decay; values as a network gives them, in
    32 bits; a duel that ends at its last transaction, and ends elsewhere."""
    drawn = random.Random(7)
    compared = 0
    for gamma, gae_lambda in ((1.0, 1.0), (1.0, 0.95), (0.99, 0.95), (0.97, 0.5), (1.0, 0.0)):
        for length in [1, 2, 3, 5, 17, 64, 211] + [drawn.randrange(1, 120) for _ in range(40)]:
            values = torch.tensor([drawn.uniform(-1.5, 1.5) for _ in range(length)]).tolist()
            rewards = [0.0] * length
            rewards[-1] += drawn.choice([1.0, -1.0, 0.0, -0.25])
            dones = [0.0] * length
            dones[-1] = 1.0
            if length > 4 and drawn.random() < 0.3:
                # Not what an update asks for, and still the same arithmetic.
                dones[drawn.randrange(length)] = 1.0
                rewards[drawn.randrange(length)] += drawn.uniform(-1, 1)
            advantages, returns = generalized_advantage(
                torch.tensor(rewards),
                torch.tensor(values),
                torch.tensor(dones),
                gamma=gamma,
                gae_lambda=gae_lambda,
            )
            assert transaction_advantages(
                rewards, values, dones, gamma=gamma, gae_lambda=gae_lambda
            ) == (advantages.tolist(), returns.tolist())
            compared += length
    assert compared > 10_000


def test_an_updates_targets_are_those_the_tensors_advantages_give(monkeypatch):
    """The returns, the advantages and what an update reports of them, with
    the advantages worked out on the processor and with the tensors'."""
    from ygo_training import optimization

    drawn = random.Random(11)
    trajectories = []
    for game in range(60):
        steps = []
        transaction = 0
        for _ in range(drawn.randrange(1, 70)):
            transaction += drawn.random() < 0.8
            steps.append(
                {
                    "player": drawn.randrange(2) if game % 3 == 0 else game % 2,
                    "transaction": transaction,
                    "value": torch.tensor(drawn.uniform(-1.2, 1.2)).item(),
                    "log_prob": -drawn.random(),
                    "chosen": 0,
                }
            )
        trajectories.append(
            {
                "terminal": game % 11 != 0,
                "learner_player": game % 2,
                "winner": drawn.choice([0, 1, 2]),
                "steps": steps,
            }
        )

    def prepared(gae_lambda: float):
        model = torch.nn.Linear(1, 1)
        optimizer = PpoOptimizer(
            model,
            torch.optim.Adam(model.parameters()),
            TrainerConfig(duel_gae_lambda=gae_lambda),
            OutcomeRewardModel(),
            "cpu",
        )
        samples, advantages, metrics = optimizer.prepare_samples(trajectories)
        return (
            [(sample.advantage, sample.return_, sample.terminal_return) for sample in samples],
            advantages.tolist(),
            metrics,
        )

    def with_tensors(rewards, values, dones, gamma=1.0, gae_lambda=1.0):
        advantages, returns = generalized_advantage(
            torch.tensor(rewards),
            torch.tensor(values),
            torch.tensor(dones),
            gamma=gamma,
            gae_lambda=gae_lambda,
        )
        return [float(value) for value in advantages], [float(value) for value in returns]

    for gae_lambda in (1.0, 0.95):
        ours = prepared(gae_lambda)
        monkeypatch.setattr(optimization, "transaction_advantages", with_tensors)
        theirs = prepared(gae_lambda)
        monkeypatch.undo()
        assert len(ours[0]) > 1_000 and ours == theirs


def test_the_value_loss_is_taken_at_the_transactions_named():
    logits = torch.tensor([[0.2, -0.1], [0.0, 0.3], [0.5, 0.5], [-0.4, 0.1]])
    actions = torch.tensor([0, 1, 0, 1])
    given = {
        "logits": logits,
        "actions": actions,
        "old_log_probs": torch.log_softmax(logits, dim=-1)[torch.arange(4), actions],
        "group_ids": torch.tensor([0, 1, 1, 2]),
        "root_positions": torch.tensor([0, 1, 3]),
        "advantages": torch.tensor([1.0, -1.0, 0.5]),
        "returns": torch.tensor([1.0, 0.0, -1.0]),
        "values": torch.tensor([0.5, 0.25, 9.0, -0.5]),
        "config": PpoConfig(),
    }

    every = joint_ppo_loss_components(**given)
    two = joint_ppo_loss_components(**given, value_roots=torch.tensor([0, 2]))
    none = joint_ppo_loss_components(**given, value_roots=torch.zeros(0, dtype=torch.long))

    # Roots at steps 0, 1 and 3: the squared errors are 0.25, 0.0625 and 0.25.
    assert every["value"] == pytest.approx((0.25 + 0.0625 + 0.25) / 3)
    assert two["value"] == pytest.approx(0.25)
    assert float(none["value"]) == 0.0
    # Which transactions teach the head is nothing to the policy's loss.
    for name in ("policy", "entropy", "approximate_kl", "clip_fraction"):
        assert float(two[name]) == float(every[name]) == float(none[name])
    assert float(none["total"]) == float(none["policy"] + none["weighted_entropy"])


def _ppo_carried(optimizer, monkeypatch, update, taught=None):
    """The values a PPO update's value loss is taken at, where every step has
    one of its own, and what the update reports.  ``taught`` receives the
    targets those values were held against."""
    games = [
        {
            "terminal": True,
            "learner_player": 0,
            "winner": winner,
            "steps": [
                {**step(chosen=number % 2), "transaction": number, "player": 0}
                for number in range(1, 7)
            ],
        }
        for winner in (0, 1)
    ]
    marks = {
        id(taken): (place + 1) / 100
        for place, taken in enumerate(taken for game in games for taken in game["steps"])
    }
    for game in games:
        for taken in game["steps"]:
            taken["value"] = marks[id(taken)]
    attached = next(optimizer.model.parameters()).sum() * 0.0

    def forward_steps(_model, chunk):
        # The optimizer works on copies of the steps: their values tell them apart.
        return (
            torch.zeros(len(chunk), 2) + attached,
            torch.tensor([float(taken["value"]) for taken in chunk]) + attached,
        )

    taken_at = []

    def measured(values, targets):
        taken_at.append(sorted(round(float(value), 2) for value in values.detach()))
        if taught is not None:
            taught.extend(float(target) for target in targets)
        return _SQUARED_ERROR(values, targets)

    monkeypatch.setattr(optimizer, "forward_steps", forward_steps)
    monkeypatch.setattr(torch.nn.functional, "mse_loss", measured)
    optimizer.begin_update(update)
    return optimizer.update(games), taken_at


def test_a_ppo_update_teaches_the_value_head_at_so_many_transactions_of_a_seat(monkeypatch):
    def made(**more):
        return _optimizer(
            PolicyValueNet(embedding_dim=8, hidden_dim=16), ppo_epochs=1, minibatch=64, **more
        )

    everywhere, at_all = _ppo_carried(made(), monkeypatch, 1)
    optimizer = made(ppo_value_positions=2)
    metrics, taken = _ppo_carried(optimizer, monkeypatch, 1)

    # Two games of six transactions of one seat.
    assert (everywhere["ppo_value_positions"], everywhere["ppo_value_targets"]) == (0.0, 12.0)
    assert [len(values) for values in at_all] == [12]
    assert (metrics["ppo_value_positions"], metrics["ppo_value_targets"]) == (2.0, 4.0)
    assert [len(values) for values in taken] == [4]
    # Two of each game: the first game's values are 0.01 to 0.06.
    assert sum(value <= 0.06 for value in taken[0]) == 2
    # Every transaction keeps its advantage and its part in the policy's loss.
    assert metrics["policy_transactions"] == everywhere["policy_transactions"] == 12.0
    assert metrics["duel_policy"] == everywhere["duel_policy"]
    assert metrics["duel_value"] != everywhere["duel_value"]
    # The same update draws the same transactions, another update others.
    assert _ppo_carried(optimizer, monkeypatch, 1)[1] == taken
    assert any(_ppo_carried(optimizer, monkeypatch, update)[1] != taken for update in (2, 3, 4))
    with pytest.raises(ValueError, match="ppo_value_positions must be non-negative"):
        TrainerConfig(ppo_value_positions=-1)


def test_a_ppo_update_can_teach_the_head_the_result_while_credit_leans_on_it(monkeypatch):
    def made(**more):
        return _optimizer(
            PolicyValueNet(embedding_dim=8, hidden_dim=16), ppo_epochs=1, minibatch=64, **more
        )

    # Shortened advantages: PPO's own target holds the head's later estimates.
    returned: list[float] = []
    as_set, _ = _ppo_carried(made(duel_gae_lambda=0.5), monkeypatch, 1, returned)
    resulted: list[float] = []
    by_result, _ = _ppo_carried(
        made(duel_gae_lambda=0.5, ppo_value_target="result"), monkeypatch, 1, resulted
    )
    assert (as_set["ppo_value_target"], by_result["ppo_value_target"]) == ("return", "result")
    # Two games of six transactions: one won, one lost.
    assert sorted(resulted) == [-1.0] * 6 + [1.0] * 6
    assert len(returned) == 12 and any(abs(target) != 1.0 for target in returned)
    # The advantages, and so the policy's side of the update, are the same.
    for name in ("duel_raw_advantage_std", "duel_policy", "duel_explained_variance"):
        assert by_result[name] == as_set[name]
    assert by_result["duel_value"] != as_set["duel_value"]
    # With whole advantages PPO's target is the result already.
    whole: list[float] = []
    plain, _ = _ppo_carried(made(duel_gae_lambda=1.0), monkeypatch, 1, whole)
    asked, _ = _ppo_carried(
        made(duel_gae_lambda=1.0, ppo_value_target="result"), monkeypatch, 1, []
    )
    assert sorted(whole) == pytest.approx([-1.0] * 6 + [1.0] * 6, abs=1e-6)
    assert asked["duel_value"] == pytest.approx(plain["duel_value"], abs=1e-6)
    with pytest.raises(ValueError, match="ppo_value_target must be one of"):
        TrainerConfig(ppo_value_target="estimate")


def test_an_update_says_how_much_of_the_results_the_head_explains():
    """Beside the returns, which hold the head's own estimates when the
    advantages are shortened: the plain result of each seat's duel."""
    drawn = random.Random(3)
    trajectories = []
    for game in range(80):
        winner = game % 2
        # A head that leans the right way, with noise.
        steps = [
            {
                "player": 0,
                "transaction": number,
                "value": (0.4 if winner == 0 else -0.4) + drawn.uniform(-0.5, 0.5),
                "log_prob": -0.5,
                "chosen": 0,
            }
            for number in range(1, 9)
        ]
        trajectories.append(
            {"terminal": True, "learner_player": 0, "winner": winner, "steps": steps}
        )

    def measured(gae_lambda):
        model = torch.nn.Linear(1, 1)
        optimizer = PpoOptimizer(
            model,
            torch.optim.Adam(model.parameters()),
            TrainerConfig(duel_gae_lambda=gae_lambda),
            OutcomeRewardModel(),
            "cpu",
        )
        return optimizer.prepare_samples(trajectories)[2]

    whole, shortened = measured(1.0), measured(0.5)

    # With whole advantages a return is the result, and the two agree.
    assert whole["duel_outcome_explained_variance"] == pytest.approx(
        whole["duel_explained_variance"], abs=1e-6
    )
    assert 0.2 < whole["duel_outcome_explained_variance"] < 0.9
    assert whole["duel_outcome_correlation"] > 0.6
    # Shortened, the returns lean on the head and flatter it; the result does not.
    assert shortened["duel_outcome_explained_variance"] == pytest.approx(
        whole["duel_outcome_explained_variance"]
    )
    assert shortened["duel_explained_variance"] != pytest.approx(
        whole["duel_explained_variance"], abs=1e-3
    )
