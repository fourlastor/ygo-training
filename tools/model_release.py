"""The files of a network's release, to attach to the draft release that
`.github/workflows/model-release.yml` makes from its card (`models/NAME.md`).

    uv run python tools/model_release.py all71 \\
        runs/seventy-one-decks/all71/checkpoints/model-27599.pt

Writes into `release/NAME/` (or `--output`):

- `NAME.pt`: the checkpoint for this repository's tools (`evaluate-suite`,
  `edopro`), its weights and what describes them, without the optimizer's
  state or the run's random state, which only training on needs. It loads as
  the checkpoint does for play (`checkpoints.load_inference_model`).
- `NAME.safetensors`, `NAME.json` and `NAME.outputs.json`: the network for a
  project that seats it without PyTorch, as `tools/export_checkpoint.py`
  writes it (its `model.safetensors`, `model.json` and `outputs.json`).
- `SHA256SUMS`: each file's digest, as `sha256sum --check` reads them.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import shutil
import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from export_checkpoint import ROOT, VECTORS, export

from ygo_training.checkpoints import load_inference_model

# What a checkpoint keeps for play: its weights, what describes them and
# its step.
KEPT = ("format_version", "metadata", "model", "step")
# The export's files, and the suffix each takes after the network's name.
EXPORTED = (
    ("model.safetensors", ".safetensors"),
    ("model.json", ".json"),
    ("outputs.json", ".outputs.json"),
)
# A name is a tag's last part: `model/NAME`.
NAME = re.compile(r"[a-z0-9]+(-[a-z0-9]+)*")


def stripped(checkpoint: Path, output: Path) -> None:
    """The checkpoint without what only training on needs."""
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    missing = [key for key in KEPT if key not in payload]
    if missing:
        raise SystemExit(f"{checkpoint} holds no {', '.join(missing)}")
    torch.save({key: payload[key] for key in KEPT}, output)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def release(name: str, checkpoint: Path, output: Path, vectors: Path) -> list[Path]:
    """Write the release's files into `output`, and say which they are."""
    if not NAME.fullmatch(name):
        raise SystemExit(f"A network's name is lowercase words joined by dashes, not {name!r}")
    output.mkdir(parents=True, exist_ok=True)
    played = output / f"{name}.pt"
    stripped(checkpoint, played)
    # It loads for play, or this says why not.
    load_inference_model(played)
    with tempfile.TemporaryDirectory() as folder:
        export(played, Path(folder), vectors)
        for source, suffix in EXPORTED:
            shutil.move(Path(folder) / source, output / f"{name}{suffix}")
    files = [played] + [output / f"{name}{suffix}" for _, suffix in EXPORTED]
    sums = output / "SHA256SUMS"
    sums.write_text("".join(f"{digest(path)}  {path.name}\n" for path in files), encoding="utf8")
    return [*files, sums]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("name", help="the network's name, the card's and the tag's (model/NAME)")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, help="the folder (release/NAME unless said)")
    parser.add_argument("--vectors", type=Path, default=ROOT / VECTORS)
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        raise SystemExit(f"No file at {args.checkpoint}")
    output = args.output or ROOT / "release" / args.name
    for path in release(args.name, args.checkpoint, output, args.vectors.resolve()):
        print(f"{path.stat().st_size / 1e6:8.1f} MB  {path}")
    card = ROOT / "models" / f"{args.name}.md"
    if not card.is_file():
        print(f"No card at {card}: the workflow drafts the release from it")


if __name__ == "__main__":
    main()
