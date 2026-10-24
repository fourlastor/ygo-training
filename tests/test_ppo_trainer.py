"""Slow integration coverage: a run from the command line, and determinism."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from ygo_training.environment import LIBRARY
from ygo_training.ppo_trainer import (
    configure_reproducible_torch,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.integration
@pytest.mark.skipif(
    not (ROOT / LIBRARY).exists(), reason="the environment is not built: cargo build --release"
)
def test_a_run_through_the_pool_is_checkpointed_update_by_update(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "ygo_training",
            "train",
            "--root",
            ".",
            "--device",
            "cpu",
            # Two updates: the pool that played the first plays the second.
            "--workers",
            "2",
            "--duels",
            "2",
            "--updates",
            "2",
            "--pool-threads",
            "2",
            "--pool-slots",
            "2",
            "--ppo-epochs",
            "1",
            "--minibatch",
            "64",
            "--seed",
            "45",
            "--opponents",
            "monarch",
            "--eval-every",
            "0",
            "--run",
            str(tmp_path),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=120,
        check=True,
    )
    checkpoint = tmp_path / "checkpoints" / "model-00001.pt.json"
    assert checkpoint.exists(), result.stdout + result.stderr
    assert (tmp_path / "checkpoints" / "model-00000.pt.json").exists(), (
        result.stdout + result.stderr
    )
    metadata = json.loads(checkpoint.read_text())
    assert metadata["duels"] == 2.0
    assert metadata["max_inference_batch"] >= 2.0
    assert metadata["samples"] > 0


def test_reproducible_torch_configures_strict_cuda_determinism(monkeypatch):
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)

    configure_reproducible_torch("cuda")

    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert torch.are_deterministic_algorithms_enabled()
    assert torch.backends.cudnn.benchmark is False
    assert torch.backends.cudnn.deterministic is True
