"""A checkpoint for a project that seats it without PyTorch.

    uv run python tools/export_checkpoint.py \\
        runs/example/checkpoints/model-02279.pt exported/model-02279

Writes three files to the folder:

- `model.safetensors`: the network's weights in the safetensors format (a
  JSON header and the numbers as they are), under the names the trainer's
  network has them by.  Nothing of the optimizer or of the run.
- `model.json`: which checkpoint it is, how the network reads its input
  (`board_representation`), the revisions of this checkout, and every
  weight's shape.
- `outputs.json`: what the network answers to the decisions of
  `native/ygo-model-input/input/vectors/decisions.json`, one decision at a
  time: a score for each answer, the answer it takes, and the value.

A forward pass written elsewhere (`PolicyValueNet.evaluate` of
`ygo_training/model.py` is the reference) is checked with the three: the
shared crates make the arrays of each decision of that file, the pass reads
them with these weights, and what comes out is `outputs.json`.  The scores
are those of a batch of one decision, on the processor, in 32-bit floats; a
port differs from them by its arithmetic's last digits (scores reach the
hundreds, so by a few parts in a million of that), and a decision whose two
best answers lie closer than that (`margin`) is no test of the answer taken.
Answers that are copies of a card lying alike score the same to the bit, and
the first of them is the one taken.
"""

from __future__ import annotations

import argparse
import json
import struct
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from model_input_vectors import digest

from ygo_training.checkpoints import load_inference_model, value_fingerprint
from ygo_training.environment import Environment
from ygo_training.model import MONSTER_ZONE, SPELL_TRAP_ZONE

SHARED = "native/ygo-model-input"
VECTORS = f"{SHARED}/input/vectors/decisions.json"
# What safetensors calls the types a network's weights come in.
TYPES = {"float32": "F32", "float64": "F64", "int64": "I64", "int32": "I32", "bool": "BOOL"}
# The checkouts whose revisions tell what a checkpoint is shown.
CHECKOUTS = {
    "trainer": ".",
    "ygo-model-input": SHARED,
    "ygo-policies": "native/ygo-policies",
    "ocgcore": "native/ygo-policies/vendor/ocgcore",
    "CardScripts": "native/ygo-policies/vendor/CardScripts",
    "BabelCdb": "native/ygo-policies/vendor/BabelCdb",
}


def safetensors(tensors: dict[str, np.ndarray], about: dict[str, str]) -> bytes:
    """The arrays as a safetensors file: eight bytes of the header's length,
    the header (each array's type, shape and place among the bytes that
    follow, and `about` under `__metadata__`), then the arrays' own bytes,
    one after the other in the order of their names."""
    header: dict[str, Any] = {"__metadata__": about}
    parts = []
    offset = 0
    for name in sorted(tensors):
        values = np.ascontiguousarray(tensors[name])
        values = values.astype(values.dtype.newbyteorder("<"), copy=False)
        data = values.tobytes()
        header[name] = {
            "dtype": TYPES[str(tensors[name].dtype)],
            "shape": list(values.shape),
            "data_offsets": [offset, offset + len(data)],
        }
        parts.append(data)
        offset += len(data)
    told = json.dumps(header, separators=(",", ":")).encode("utf8")
    # Spaces after the header let the numbers begin at a multiple of eight.
    told += b" " * (-len(told) % 8)
    return struct.pack("<Q", len(told)) + told + b"".join(parts)


def revision(folder: Path) -> str | None:
    found = subprocess.run(
        ["git", "-C", str(folder), "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return found.stdout.strip() or None


def answers(model: torch.nn.Module, vectors: dict[str, Any]) -> list[dict[str, Any]]:
    """What the network makes of each decision of the vectors, alone, its
    input built by the library."""
    found = []
    with Environment(ROOT) as environment, torch.no_grad():
        for decision in vectors["decisions"]:
            batch = environment.batch([decision["view"]])
            logits, value = model({name: torch.from_numpy(array) for name, array in batch.items()})
            scores = [float(score) for score in logits[0]]
            best = sorted(scores, reverse=True)
            found.append(
                {
                    "logits": scores,
                    "first_choice": scores.index(best[0]),
                    # How far the answer taken is ahead of the next.
                    "margin": best[0] - best[1] if len(best) > 1 else None,
                    "value": float(value[0]),
                }
            )
    return found


def export(checkpoint: Path, output: Path, vectors: Path) -> dict[str, Any]:
    model, payload = load_inference_model(checkpoint, device="cpu")
    if model.leave_out:
        # A forward pass written elsewhere reads every part of the input.
        raise ValueError(
            f"{checkpoint} leaves out {', '.join(model.leave_out)} of its input (`train --leave-out`), "
            "which a forward pass written elsewhere does not: it is not exported"
        )
    if model.reads_descriptions or model.hides_identities:
        # Nor does one look a card up in a table (format 16).
        raise ValueError(
            f"{checkpoint} is told of a card beside its input (its description, or an identity "
            "hidden), which a forward pass written elsewhere is not: it is not exported"
        )
    state = {name: value.detach().cpu().numpy() for name, value in model.state_dict().items()}
    metadata = payload.get("metadata", {})
    about = {
        "checkpoint": checkpoint.name,
        "step": payload.get("step"),
        "format_version": payload.get("format_version"),
        "fingerprint": value_fingerprint(model.state_dict()),
        "board_representation": str(model.board_representation),
        "policy_input_schema_digest": metadata.get("policy_input_schema_digest"),
    }
    kept = json.loads(vectors.read_text(encoding="utf8"))
    description = {
        "about": (
            "A checkpoint of the trainer's PolicyValueNet, for a forward pass written "
            "elsewhere. model.safetensors holds the weights under the names below; "
            "outputs.json holds what the network answers to the decisions of the "
            "shared vectors."
        ),
        "checkpoint": about,
        "parameters": int(sum(values.size for values in state.values())),
        "reading": {
            "forward": "PolicyValueNet.evaluate, ygo_training/model.py of the trainer",
            "board_representation": (
                "individual-tokens: each answer attends to the cards on the field "
                "(board_query, board_key, board_value, board_fusion). mean-pool: it "
                "does not, and those four weights are not read."
            ),
            "field_locations": {"monster_zone": MONSTER_ZONE, "spell_trap_zone": SPELL_TRAP_ZONE},
            "card_identity": (
                "a card's code is looked up as two halves: its low 16 bits in "
                "card_identity.low, the next 16 in card_identity.high"
            ),
            "answer": (
                "the highest of the scores of a decision's answers; the first of them "
                "where several are equal, as copies of a card that lie alike are"
            ),
        },
        # Of this checkout, when the export was made: those of the
        # checkpoint's training when nothing has moved since.
        "revisions": {name: revision(ROOT / folder) for name, folder in CHECKOUTS.items()},
        "tensors": {name: list(values.shape) for name, values in sorted(state.items())},
    }
    outputs = {
        "about": (
            "What the checkpoint answers to each decision of the shared vectors, in "
            "their order, as a batch of one: a score for each answer of the view, the "
            "place of the highest, how far it is ahead of the next, and the value."
        ),
        "checkpoint": about,
        "vectors": {
            "file": str(vectors.relative_to(ROOT / SHARED))
            if vectors.is_relative_to(ROOT / SHARED)
            else vectors.name,
            "revision": description["revisions"]["ygo-model-input"],
            "digest": digest(np.frombuffer(vectors.read_bytes(), dtype=np.uint8)),
        },
        "decisions": answers(model, kept),
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "model.safetensors").write_bytes(
        safetensors(state, {key: str(value) for key, value in about.items()})
    )
    (output / "model.json").write_text(json.dumps(description, indent=2) + "\n", encoding="utf8")
    (output / "outputs.json").write_text(
        json.dumps(outputs, separators=(",", ":")) + "\n", encoding="utf8"
    )
    return {"description": description, "outputs": outputs}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path, help="the folder the three files are written to")
    parser.add_argument(
        "--vectors",
        type=Path,
        default=ROOT / VECTORS,
        help="the decisions the network answers (the shared repository's unless said)",
    )
    args = parser.parse_args()
    for path in (args.checkpoint, args.vectors):
        if not path.is_file():
            raise SystemExit(f"No file at {path}")
    found = export(args.checkpoint, args.output, args.vectors.resolve())
    about = found["description"]["checkpoint"]
    decisions = found["outputs"]["decisions"]
    close = sum(
        decision["margin"] is not None and decision["margin"] < 1e-3 for decision in decisions
    )
    size = (args.output / "model.safetensors").stat().st_size / 1e6
    print(
        f"{about['checkpoint']} after {about['step']} updates ({about['board_representation']}): "
        f"{found['description']['parameters']:,} numbers, {size:.1f} MB"
    )
    print(
        f"{len(decisions)} decisions answered, {close} of them with its two best answers "
        f"closer than 0.001"
    )
    print(f"written to {args.output}: model.safetensors, model.json, outputs.json")


if __name__ == "__main__":
    main()
