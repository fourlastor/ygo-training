"""A checkpoint written for a project that seats it without PyTorch."""

from __future__ import annotations

import importlib.util
import json
import struct
import sys
from itertools import pairwise
from pathlib import Path

import numpy as np
import torch

from tests.views import (
    ACTIVATE,
    OTHER_MONSTER,
    PASS,
    batched,
    card,
    choice,
    named,
    view,
)
from ygo_training.checkpoints import save, value_fingerprint
from ygo_training.model import PolicyValueNet

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
SPEC = importlib.util.spec_from_file_location(
    "export_checkpoint", ROOT / "tools/export_checkpoint.py"
)
assert SPEC is not None and SPEC.loader is not None
tool = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(tool)


def read(raw: bytes) -> tuple[dict, dict[str, np.ndarray]]:
    """A safetensors file as its format says it is read."""
    (length,) = struct.unpack("<Q", raw[:8])
    header = json.loads(raw[8 : 8 + length])
    body = raw[8 + length :]
    types = {"F32": "<f4", "I64": "<i8", "BOOL": "?"}
    arrays = {
        name: np.frombuffer(
            body[told["data_offsets"][0] : told["data_offsets"][1]], dtype=types[told["dtype"]]
        ).reshape(told["shape"])
        for name, told in header.items()
        if name != "__metadata__"
    }
    return header, arrays


def test_weights_are_written_as_safetensors():
    tensors = {
        "score.bias": np.asarray([1.5, -2.0, 0.25], dtype=np.float32),
        "embedding.weight": np.arange(6, dtype=np.float32).reshape(2, 3),
        "counted": np.asarray([[7], [-9]], dtype=np.int64),
        "mask": np.asarray([True, False]),
    }
    raw = tool.safetensors(tensors, {"step": "3"})

    header, arrays = read(raw)
    (length,) = struct.unpack("<Q", raw[:8])
    # The header ends where the numbers can begin at a multiple of eight.
    assert length % 8 == 0 and raw[8:9] == b"{"
    assert header["__metadata__"] == {"step": "3"}
    assert header["embedding.weight"] == {"dtype": "F32", "shape": [2, 3], "data_offsets": [16, 40]}
    assert header["counted"]["dtype"] == "I64" and header["mask"]["dtype"] == "BOOL"
    # One after the other in the order of their names, nothing between them.
    places = [header[name]["data_offsets"] for name in sorted(tensors)]
    assert places[0][0] == 0 and places[-1][1] == len(raw) - 8 - length
    assert all(before[1] == after[0] for before, after in pairwise(places))
    assert arrays.keys() == tensors.keys()
    assert all(np.array_equal(arrays[name], tensors[name]) for name in tensors)


def test_a_checkpoint_is_exported_with_what_it_answers(tmp_path):
    torch.manual_seed(5)
    model = PolicyValueNet(board_representation="mean-pool")
    checkpoint = tmp_path / "model-00002.pt"
    save(checkpoint, model, torch.optim.Adam(model.parameters()), 2, {})
    shown = [
        view(),
        view(
            cards=[card(), card(OTHER_MONSTER, attacking=True)],
            choices=[
                choice(ACTIVATE, named(0)),
                choice(ACTIVATE, named(1, OTHER_MONSTER)),
                choice(PASS),
            ],
        ),
        view(choices=[choice(PASS)]),
    ]
    vectors = tmp_path / "decisions.json"
    vectors.write_text(json.dumps({"decisions": [{"view": each} for each in shown]}))

    found = tool.export(checkpoint, tmp_path / "exported", vectors)

    folder = tmp_path / "exported"
    described = json.loads((folder / "model.json").read_text())
    assert described == found["description"]
    assert described["checkpoint"] == {
        "checkpoint": "model-00002.pt",
        "step": 2,
        "format_version": 15,
        "fingerprint": value_fingerprint(model.state_dict()),
        "board_representation": "mean-pool",
        "policy_input_schema_digest": described["checkpoint"]["policy_input_schema_digest"],
    }
    assert described["parameters"] == sum(value.numel() for value in model.parameters())
    assert described["tensors"]["score.2.bias"] == [1]
    assert (
        described["revisions"].keys() == tool.CHECKOUTS.keys() and described["revisions"]["trainer"]
    )
    # The weights are the checkpoint's, under their names.
    header, weights = read((folder / "model.safetensors").read_bytes())
    state = model.state_dict()
    assert weights.keys() == state.keys()
    assert all(np.array_equal(weights[name], state[name].numpy()) for name in state)
    assert header["__metadata__"]["step"] == "2"
    # What the network answers to each decision, alone.
    outputs = json.loads((folder / "outputs.json").read_text())
    assert outputs["vectors"]["file"] == "decisions.json"
    assert outputs["vectors"]["digest"] == tool.digest(
        np.frombuffer(vectors.read_bytes(), dtype=np.uint8)
    )
    for each, answered in zip(shown, outputs["decisions"], strict=True):
        with torch.no_grad():
            logits, value = model(batched(each))
        assert answered["logits"] == [float(score) for score in logits[0]]
        assert answered["first_choice"] == int(logits[0].argmax())
        assert answered["value"] == float(value[0])
    assert [len(answered["logits"]) for answered in outputs["decisions"]] == [2, 3, 1]
    # A single answer is ahead of nothing.
    assert outputs["decisions"][2]["margin"] is None and outputs["decisions"][0]["margin"] >= 0
