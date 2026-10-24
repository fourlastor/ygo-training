"""The files of a network's release, and with `--publish` the release itself,
titled and described by its card (`models/NAME.md`).

    uv run python tools/model_release.py all71 \\
        runs/seventy-one-decks/all71/checkpoints/model-27599.pt --publish

Writes into `release/NAME/` (or `--output`):

- `NAME.pt`: the checkpoint for this repository's tools (`evaluate-suite`,
  `edopro`), its weights and what describes them, without the optimizer's
  state or the run's random state, which only training on needs. It loads as
  the checkpoint does for play (`checkpoints.load_inference_model`).
- `NAME.safetensors`, `NAME.json` and `NAME.outputs.json`: the network for a
  project that seats it without PyTorch, as `tools/export_checkpoint.py`
  writes it (its `model.safetensors`, `model.json` and `outputs.json`).
- `SHA256SUMS`: each file's digest, as `sha256sum --check` reads them.

With `--publish` it then makes the release with `gh`, as the workflow
`.github/workflows/model-release.yml` drafts it (the tag `model/NAME` on the
commit `--ref` names, `HEAD` unless said, the card's title, its notes as
`tools/release_notes.py` writes them), the files attached, and publishes
it. The card, the commit on GitHub and a tag not taken yet are checked
before anything is written. Without it, the workflow drafts the release
and the files are attached by hand.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from export_checkpoint import ROOT, VECTORS, export
from release_notes import notes

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


def git(*arguments: str) -> str:
    found = subprocess.run(
        ["git", "-C", str(ROOT), *arguments], capture_output=True, text=True, check=True
    )
    return found.stdout.strip()


def repository_of(url: str) -> str:
    """OWNER/NAME of a remote on GitHub."""
    found = re.search(r"github\.com[:/](.+?)(\.git)?$", url)
    if found is None:
        raise SystemExit(f"origin is no repository of GitHub: {url}")
    return found.group(1)


def publishing(name: str, ref: str) -> tuple[str, str, str]:
    """What a release needs before its files are written: its card's text,
    the commit on GitHub its tag is to point at, and the repository; or
    why it cannot be made."""
    card = ROOT / "models" / f"{name}.md"
    if not card.is_file():
        raise SystemExit(f"No card at {card}: a release is titled and described by its card")
    commit = git("rev-parse", f"{ref}^{{commit}}")
    if not git("branch", "--remotes", "--contains", commit):
        raise SystemExit(f"{commit[:8]} is on no branch of a remote: push it before its release")
    repository = repository_of(git("remote", "get-url", "origin"))
    taken = subprocess.run(
        ["gh", "release", "view", f"model/{name}", "--repo", repository],
        capture_output=True,
        check=False,
    )
    if taken.returncode == 0:
        raise SystemExit(f"model/{name} is released already")
    return card.read_text(encoding="utf8"), commit, repository


def command(
    name: str, files: list[Path], repository: str, commit: str, title: str, written: Path
) -> list[str]:
    """`gh`'s command for the release, published with its files."""
    told = ["--repo", repository, "--target", commit, "--title", title, "--notes-file", written]
    return ["gh", "release", "create", f"model/{name}", *map(str, files), *map(str, told)]


def publish(name: str, files: list[Path], card: str, commit: str, repository: str) -> None:
    title = card.partition("\n")[0].removeprefix("# ")
    with tempfile.TemporaryDirectory() as folder:
        written = Path(folder) / "notes.md"
        written.write_text(notes(card, repository, commit), encoding="utf8")
        subprocess.run(command(name, files, repository, commit, title, written), check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("name", help="the network's name, the card's and the tag's (model/NAME)")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, help="the folder (release/NAME unless said)")
    parser.add_argument("--vectors", type=Path, default=ROOT / VECTORS)
    parser.add_argument("--publish", action="store_true", help="make the release, published")
    parser.add_argument("--ref", default="HEAD", help="the commit the tag points at (HEAD)")
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        raise SystemExit(f"No file at {args.checkpoint}")
    ready = publishing(args.name, args.ref) if args.publish else None
    output = args.output or ROOT / "release" / args.name
    files = release(args.name, args.checkpoint, output, args.vectors.resolve())
    for path in files:
        print(f"{path.stat().st_size / 1e6:8.1f} MB  {path}")
    if ready is not None:
        publish(args.name, files, *ready)
        return
    card = ROOT / "models" / f"{args.name}.md"
    if not card.is_file():
        print(f"No card at {card}: the workflow drafts the release from it")


if __name__ == "__main__":
    main()
