"""`ygo_training/stacked.py`: networks of one shape answering their
decisions in one call, their weights stacked."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from ygo_training.environment import library_path
from ygo_training.model import PolicyValueNet
from ygo_training.pool import DuelPool, taught_games
from ygo_training.schedules import rollout_games
from ygo_training.stacked import Stacked, lstm

ROOT = Path(__file__).resolve().parents[1]


def _built() -> bool:
    try:
        library_path(ROOT)
    except Exception:  # noqa: BLE001 - not built
        return False
    return True


def drawn(seed: int) -> PolicyValueNet:
    torch.manual_seed(seed)
    return PolicyValueNet(board_representation="mean-pool").eval()


def test_the_lstm_written_out_is_torchs_to_its_last_digits():
    torch.manual_seed(3)
    cells = torch.nn.LSTM(20, 16, batch_first=True)
    steps = torch.randn(5, 7, 20)
    with torch.no_grad():
        theirs, _ = cells(steps)
        ours = lstm(steps, cells)
    assert ours.shape == theirs.shape
    assert torch.allclose(ours, theirs, atol=1e-6)


@pytest.mark.integration
@pytest.mark.skipif(not _built(), reason="the environment is not built: cargo build --release")
def test_each_network_stacked_answers_its_own_decisions_as_it_does():
    torch.set_num_threads(2)
    networks = {"blackwing": drawn(1), "burn": drawn(2), "heroes": drawn(3)}
    games = rollout_games(11 * 1024, 3, ["monarch", "heroes", "burn", "lightsworn"])
    with DuelPool(ROOT, threads=1, slots=1) as pool:
        trajectories, _ = taught_games(
            pool,
            {"blackwing": networks["blackwing"]},
            "cpu",
            learner="blackwing",
            games=games,
            limit=4096,
        )
        rows = [step["row"] for trajectory in trajectories for step in trajectory["steps"]][:60]
        arrays = pool.batch(rows)
    decks = np.asarray(["blackwing", "burn", "heroes"] * 20)[: len(rows)]
    stacked = Stacked(networks, "cpu").answer(arrays, decks)
    for deck, network in networks.items():
        mine = np.flatnonzero(decks == deck)
        given = {name: torch.from_numpy(array[mine]) for name, array in arrays.items()}
        with torch.no_grad():
            own = network(given)[0]
        ours = stacked[torch.from_numpy(mine)]
        assert torch.equal(torch.isfinite(ours), torch.isfinite(own))
        finite = torch.isfinite(own)
        assert torch.allclose(ours[finite], own[finite], atol=1e-4)
    with pytest.raises(ValueError, match="No network stacked plays"):
        Stacked({"burn": networks["burn"]}, "cpu").answer(arrays, decks)


@pytest.mark.integration
@pytest.mark.skipif(not _built(), reason="the environment is not built: cargo build --release")
def test_the_stacked_networks_answer_from_graphs_of_fixed_sizes():
    from ygo_training.graphs import Graphs

    torch.set_num_threads(2)
    networks = {"blackwing": drawn(1), "burn": drawn(2)}
    games = rollout_games(11 * 1024, 3, ["monarch", "heroes", "burn", "lightsworn"])
    with DuelPool(ROOT, threads=1, slots=1) as pool:
        trajectories, _ = taught_games(
            pool,
            {"blackwing": networks["blackwing"]},
            "cpu",
            learner="blackwing",
            games=games,
            limit=4096,
        )
        rows = [step["row"] for trajectory in trajectories for step in trajectory["steps"]]
        # Decisions a graph's widths hold.
        fitting = Graphs(networks["blackwing"], "cpu", largest=1)
        rows = [row for row in rows if fitting.fits(pool.batch([row]))][:10]
        arrays = pool.batch(rows)
    decks = np.asarray(["blackwing", "burn"] * 5)[: len(rows)]
    stacked = Stacked(networks, "cpu", largest=8)
    stacked.begin_update()
    ours = stacked.answer(arrays, decks)
    assert stacked.graphs is not None and stacked.graphs.told["answered"] == 1
    for deck, network in networks.items():
        mine = np.flatnonzero(decks == deck)
        given = {name: torch.from_numpy(array[mine]) for name, array in arrays.items()}
        with torch.no_grad():
            own = network(given)[0]
        finite = torch.isfinite(own)
        assert torch.equal(torch.isfinite(ours[torch.from_numpy(mine)]), finite)
        assert torch.allclose(ours[torch.from_numpy(mine)][finite], own[finite], atol=1e-4)
