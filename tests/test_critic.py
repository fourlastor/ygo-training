"""`ygo_training/critic.py`: a critic that sees the hidden cards, read from
both packings of the network's decisions, and apart from the policy."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from ygo_training.config import TrainerConfig
from ygo_training.critic import SightedCritic, transaction_roots
from ygo_training.environment import library_path
from ygo_training.model import PolicyValueNet
from ygo_training.optimization import PpoOptimizer
from ygo_training.pool import DuelPool, _setups
from ygo_training.ppo import transaction_advantages
from ygo_training.ppo_trainer import PpoTrainer
from ygo_training.rewards import OutcomeRewardModel

ROOT = Path(__file__).resolve().parents[1]


def _built() -> bool:
    try:
        library_path(ROOT)
    except FileNotFoundError:
        return False
    return True


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not _built(), reason="the environment is not built: cargo build --release"),
]


def decisions() -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """The network's decisions of two duels, as the policy is given them and
    with nothing hidden."""
    games = [
        {"seed": 400 + index, "opponent": "monarch", "learner_player": index} for index in range(2)
    ]
    with DuelPool(ROOT, threads=1, slots=2) as pool:
        pool.submit(
            _setups("blackwing", games, {"control": "caller", "alone": True}, 4096, whole=True),
            keep=True,
        )
        rows: list[int] = []
        answers: Any = ()
        while True:
            round_ = pool.step(answers)
            rows += round_.row.tolist()
            answers = (
                round_.arrays["action_mask"].argmax(axis=1).astype(np.uint32) if len(round_) else ()
            )
            if round_.in_flight == 0 and not len(round_):
                break
        rows = rows[:64]
        seat, whole = pool.batch(rows), pool.batch(rows, whole=True)
    return tensors(seat), tensors(whole)


def tensors(arrays: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
    return {name: torch.from_numpy(array) for name, array in arrays.items()}


def test_the_critic_values_each_decision_from_both_views_and_leaves_the_policy_alone():
    torch.manual_seed(5)
    policy = PolicyValueNet()
    critic = SightedCritic.from_policy(policy)
    # Both readers begin as the policy's network, and are not it.
    for reader in (critic.seat, critic.whole):
        assert reader is not policy
        for mine, its in zip(reader.parameters(), policy.parameters(), strict=True):
            assert torch.equal(mine, its) and mine.data_ptr() != its.data_ptr()
    seat, whole = decisions()
    value = critic(seat, whole)
    assert value.shape == (len(seat["action_mask"]),)
    assert (value.abs() <= 1).all()
    value.pow(2).mean().backward()
    assert all(parameter.grad is None for parameter in policy.parameters())
    assert critic.value[0].weight.grad is not None
    assert critic.whole.state[0].weight.grad is not None
    # What it reads of the hidden cards moves its value.
    hidden = whole["state_card"] != seat["state_card"]
    assert hidden.any()
    with torch.no_grad():
        blind = critic(seat, {**whole, "state_card": seat["state_card"]})
        assert not torch.allclose(blind, critic(seat, whole))


def test_a_warm_critic_begins_by_telling_each_decisions_value_as_the_value_head_does():
    torch.manual_seed(6)
    policy = PolicyValueNet()
    critic = SightedCritic.from_policy(policy, warm=True)
    seat, whole = decisions()
    with torch.no_grad():
        _, head = policy(seat)
        assert torch.allclose(critic(seat, whole), head, atol=1e-6)
    # What it reads of the whole duel is there to be learnt.
    value = critic(seat, whole)
    value.pow(2).mean().backward()
    assert critic.value[0].weight.grad[:, policy.hidden_dim :].abs().sum() > 0


def test_where_a_critic_gives_the_credit_the_head_is_still_taught_its_own_return():
    """The advantages are measured against the critic's values, and the
    value head's target is the return of its own estimates, as it is
    without a critic."""
    config = TrainerConfig(duel_gae_lambda=0.95)
    model = torch.nn.Linear(1, 1)
    optimizer = PpoOptimizer(
        model, torch.optim.Adam(model.parameters()), config, OutcomeRewardModel(), "cpu"
    )
    # Two seats of one duel against itself, which the first wins: each
    # transaction its seat, the head's estimate and the critic's value.
    played = [(0, 0.1, 0.5), (1, -0.2, -0.4), (0, 0.3, 0.9), (1, 0.0, -0.8), (0, 0.6, 0.2)]

    def prepared(critic: bool) -> dict[int, list[tuple[float, float]]]:
        steps = []
        for transaction, (player, head, valued) in enumerate(played):
            step = {"player": player, "transaction": transaction, "value": head}
            if critic:
                step["critic_value"] = valued
            steps.append(step)
        trajectory = {
            "terminal": True,
            "winner": 0,
            "learner_player": 0,
            "other": "self",
            "steps": steps,
        }
        samples, _, _ = optimizer.prepare_samples([trajectory])
        by_seat: dict[int, list[tuple[float, float]]] = {0: [], 1: []}
        for sample in samples:
            by_seat[int(sample.step["player"])].append((sample.advantage, sample.return_))
        return by_seat

    with_critic, without = prepared(True), prepared(False)
    for seat, result in ((0, 1.0), (1, -1.0)):
        mine = [entry for entry in played if entry[0] == seat]
        rewards = [0.0] * (len(mine) - 1) + [result]
        dones = [0.0] * (len(mine) - 1) + [1.0]
        heads, critics = [entry[1] for entry in mine], [entry[2] for entry in mine]
        credited, _ = transaction_advantages(rewards, critics, dones, gae_lambda=0.95)
        advantages, returns = transaction_advantages(rewards, heads, dones, gae_lambda=0.95)
        assert credited != advantages
        assert with_critic[seat] == list(zip(credited, returns, strict=True))
        assert without[seat] == list(zip(advantages, returns, strict=True))


def critic_config(**more: Any) -> TrainerConfig:
    """A small run whose games the model plays, against pilots and itself."""
    return TrainerConfig(
        workers=2,
        duels=4,
        device="cpu",
        seed=21,
        opponent_decks=["monarch", "gishki"],
        self_play_duels=2,
        pool_threads=2,
        pool_slots=2,
        minibatch=32,
        ppo_epochs=1,
        **more,
    )


@pytest.mark.parametrize("limit", [4096, 30])
def test_the_transactions_a_critic_values_are_those_an_update_trains_on(limit):
    """Of every seat an update learns from, duels cut short among them."""
    torch.set_num_threads(2)
    with PpoTrainer(ROOT, critic_config(critic=True, limit=limit)) as trainer:
        trajectories, _ = trainer.collect()
        if limit < 4096:
            assert any(not trajectory["terminal"] for trajectory in trajectories)
        found = transaction_roots(trajectories, trainer.reward_model)
        samples, _, _ = trainer.policy_optimizer.prepare_samples(trajectories)
        firsts: dict[object, Any] = {}
        for sample in samples:
            firsts.setdefault(sample.step["_transaction_key"], sample)
        assert sorted((int(root.step["row"]), root.result) for root in found) == sorted(
            (int(sample.step["row"]), sample.terminal_return) for sample in firsts.values()
        )


def test_a_critic_leaves_the_games_as_they_are_and_begins_with_the_heads_credit():
    torch.set_num_threads(2)
    with (
        PpoTrainer(ROOT, critic_config(critic=True, critic_deck=2)) as sighted,
        PpoTrainer(ROOT, critic_config()) as blind,
    ):
        sighted.begin_critic()
        blind.begin_critic()
        assert blind.critic is None and blind.critic_description is None
        assert sighted.critic is not None
        # Begun as the value head, it gives every decision the head's value.
        trajectories, _ = sighted.collect()
        blind.collect()
        credit = sighted._credit(trajectories, 1)
        found = transaction_roots(trajectories, sighted.reward_model)
        assert len(found) == credit["critic_transactions"] > 50
        assert credit["critic_positions"] <= 8 * 2 * 4
        assert all(abs(root.step["critic_value"] - root.step["value"]) < 1e-4 for root in found)
        assert credit["critic_outcome_explained_variance"] == pytest.approx(
            credit["critic_head_outcome_explained_variance"], abs=1e-4
        )
        # The games are the same, and the update with the critic trains it.
        taught = [parameter.detach().clone() for parameter in sighted.critic.parameters()]
        first, second = sighted.train_update(step=2), blind.train_update(step=2)
        assert first["rollout_decision_fingerprint"] == second["rollout_decision_fingerprint"]
        assert first["critic_loss"] > 0 and "critic_loss" not in second
        assert any(
            not torch.equal(before, after)
            for before, after in zip(taught, sighted.critic.parameters(), strict=True)
        )


def test_a_run_with_a_critic_keeps_it_beside_its_checkpoints_and_goes_on_from_it(tmp_path):
    run = tmp_path / "run"
    common = [
        "--device", "cpu", "--workers", "2", "--pool-threads", "2", "--pool-slots", "2",
        "--duels", "2", "--ppo-epochs", "1", "--minibatch", "64", "--seed", "45",
        "--opponents", "monarch", "--eval-every", "0", "--run", str(run),
        "--keep-checkpoints", "2",
    ]  # fmt: skip
    critic = ["--critic", "--critic-deck", "3", "--critic-positions", "4"]

    def train(*more: str, check: bool = True) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-m", "ygo_training", "train", *common, *more],
            cwd=ROOT,
            text=True,
            capture_output=True,
            timeout=600,
            check=check,
        )

    train(*critic, "--updates", "3")
    checkpoints = run / "checkpoints"
    # Kept as the checkpoints are: one in two and the latest.
    assert sorted(path.name for path in checkpoints.glob("*.pt")) == [
        "critic-00001.pt",
        "critic-00002.pt",
        "model-00001.pt",
        "model-00002.pt",
    ]
    # Nothing that plays holds it.
    held = torch.load(checkpoints / "model-00002.pt", map_location="cpu", weights_only=False)
    assert not any(name.startswith(("seat.", "whole.")) for name in held["model"])
    assert "critic" not in held
    described = json.loads((run / "run.json").read_text())
    assert described["critic"] == {
        "deck": 3,
        "positions": 4,
        "learning_rate": 3e-4,
        "minibatch": 64,
        "begun_as": "the value head",
    }
    rows = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
    assert all(row["critic_positions"] <= 4 * 2 * 2 for row in rows)
    # A run goes on from its critic, and only with it.
    refused = train("--continue", "--updates", "1", check=False)
    assert refused.returncode != 0 and "critic" in refused.stderr
    kept = tmp_path / "critic-00002.pt"
    shutil.move(checkpoints / "critic-00002.pt", kept)
    missing = train(*critic, "--continue", "--updates", "1", check=False)
    assert missing.returncode != 0 and "No critic is kept beside" in missing.stderr
    shutil.move(kept, checkpoints / "critic-00002.pt")
    train(*critic, "--continue", "--updates", "1")
    assert (checkpoints / "critic-00003.pt").exists()
    saved = torch.load(checkpoints / "critic-00003.pt", map_location="cpu", weights_only=False)
    assert saved["step"] == 4 and saved["optimizer"]["state"]
