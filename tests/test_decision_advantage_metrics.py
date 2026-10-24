import math

import pytest
import torch

from ygo_training.optimization import PreparedSample, decision_advantage_metrics


def _sample(prompt: str, kind: str, *, value: float, terminal: float, log_prob: float) -> PreparedSample:
    step = {"prompt_type": prompt, "action_kind": kind, "value": value, "log_prob": log_prob}
    return PreparedSample(step, 0.0, 0.0, terminal)


def test_groups_chain_pass_and_activation() -> None:
    samples = [
        _sample("Chain", "Pass", value=0.2, terminal=1.0, log_prob=math.log(0.5)),
        _sample("Chain", "Activate", value=0.0, terminal=1.0, log_prob=math.log(0.25)),
        _sample("Chain", "Activate", value=0.4, terminal=-1.0, log_prob=math.log(0.75)),
        _sample("Idle", "Pass", value=0.0, terminal=1.0, log_prob=0.0),
    ]
    raw = torch.tensor([0.5, 1.0, -0.2, 3.0])
    normalized = torch.tensor([0.25, 0.5, -0.1, 1.5])
    metrics = decision_advantage_metrics(samples, raw, normalized)
    assert metrics["chain_pass_samples"] == 1
    assert metrics["chain_activate_samples"] == 2
    assert metrics["chain_activate_raw_advantage_mean"] == pytest.approx(0.4)
    assert metrics["chain_activate_normalized_advantage_sum"] == pytest.approx(0.4)
    assert metrics["chain_activate_positive_advantage_fraction"] == pytest.approx(0.5)
    assert metrics["chain_activate_value_mean"] == pytest.approx(0.2)
    assert metrics["chain_activate_terminal_return_mean"] == pytest.approx(0.0)
    assert metrics["chain_activate_probability_mean"] == pytest.approx(0.5)


def test_steps_without_metadata_are_ignored() -> None:
    samples = [PreparedSample({"value": 0.0, "log_prob": 0.0}, 0.0, 0.0)]
    metrics = decision_advantage_metrics(samples, torch.zeros(1), torch.zeros(1))
    assert metrics == {"chain_pass_samples": 0.0, "chain_activate_samples": 0.0}
