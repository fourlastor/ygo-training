"""Filesystem persistence for one append-only training run."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .checkpoints import KEPT_BESIDE, beside
from .metrics import read


def _protocol_changes(previous: Any, current: Any, path: str = "") -> list[str]:
    """Compare recorded values recursively while permitting newly added keys."""
    if isinstance(previous, dict) and isinstance(current, dict):
        changes: list[str] = []
        for key in previous.keys() & current.keys():
            child = f"{path}.{key}" if path else key
            changes.extend(_protocol_changes(previous[key], current[key], child))
        return changes
    return [f"{path}: {previous!r} -> {current!r}"] if previous != current else []


class RunRepository:
    """Own paths and collision rules for a training run.

    Command handlers should decide *what* to train. This object decides where
    run metadata, checkpoints, and metrics live.
    """

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir.resolve()
        self.checkpoint_dir = self.run_dir / "checkpoints"
        self.metrics_path = self.run_dir / "metrics.jsonl"
        self.config_path = self.run_dir / "run.json"

    def latest_checkpoint(self) -> Path:
        candidates: list[tuple[int, Path]] = []
        for path in self.checkpoint_dir.glob("model-*.pt"):
            suffix = path.stem.removeprefix("model-")
            if suffix.isdigit():
                candidates.append((int(suffix), path))
        if not candidates:
            raise FileNotFoundError(f"No model-*.pt checkpoint found in {self.checkpoint_dir}")
        return max(candidates, key=lambda item: item[0])[1]

    def may_imitate_longer(self, done: int, updates: int) -> bool:
        """Whether a continuation may set its imitation-only phase to ``updates``.

        A run that has done nothing but imitate, ``done`` updates and none
        past the phase it recorded, may lengthen that phase: nothing it has
        done changes its meaning. Past the phase, or to shorten it, the
        length is the run's protocol.
        """
        if not self.config_path.exists():
            return False
        recorded = json.loads(self.config_path.read_text(encoding="utf8")).get(
            "imitation_only_updates"
        )
        return recorded is not None and done <= int(recorded) <= updates

    def schedule_origin(self) -> int:
        """The version this run began at, from which its schedule counts.

        A run begun with ``--resume`` starts its warm-up and imitation-decay
        schedule at the resumed checkpoint, not at version zero, and a
        continuation must count from there too. A run recorded before
        ``run.json`` held the origin began at its first recorded version.
        """
        if self.config_path.exists():
            origin = json.loads(self.config_path.read_text(encoding="utf8")).get("schedule_origin")
            if origin is not None:
                return int(origin)
        if self.metrics_path.exists():
            versions = [
                int(record["version"]) for record in read(self.metrics_path) if "version" in record
            ]
            if versions:
                return min(versions)
        return 0

    def checkpoint(self, version: int) -> Path:
        if version < 0:
            raise ValueError("checkpoint version must be non-negative")
        return self.checkpoint_dir / f"model-{version:05}.pt"

    def ensure_versions_available(
        self,
        versions: Iterable[int],
        *,
        allow_overwrite: bool = False,
    ) -> None:
        if allow_overwrite or not self.metrics_path.exists():
            return
        existing = {
            int(record["version"]) for record in read(self.metrics_path) if "version" in record
        }
        overlap = sorted(existing.intersection(versions))
        if overlap:
            raise FileExistsError(
                f"Run already contains version(s) "
                f"{overlap[0]}..{overlap[-1]}. Use a new --run directory to "
                "branch safely, or explicitly pass --allow-overwrite."
            )

    def ensure_checkpoint_available(
        self,
        version: int,
        *,
        allow_overwrite: bool = False,
    ) -> Path:
        checkpoint = self.checkpoint(version)
        if checkpoint.exists() and not allow_overwrite:
            raise FileExistsError(
                f"Refusing to overwrite {checkpoint}; use a new --run "
                "directory or --allow-overwrite."
            )
        return checkpoint

    def thin(self, version: int, *, every: int, first: int, keep: bool = False) -> None:
        """With ``version`` just written, remove the checkpoint before it
        unless it closes a run of ``every`` updates, or ``keep`` says the
        update played a duel the engine stopped: that checkpoint plays it
        again (``failed-duels.jsonl``).

        A long run then keeps one checkpoint in ``every`` and its latest, to
        go on from.  Nothing written before this invocation began, at
        ``first``, is touched.
        """
        previous = version - 1
        if every > 1 and previous >= first and (previous + 1) % every and not keep:
            self.checkpoint(previous).unlink(missing_ok=True)
            # What is kept beside it, where the run keeps anything.
            for kind in KEPT_BESIDE:
                beside(self.checkpoint(previous), kind).unlink(missing_ok=True)

    def initialize(
        self,
        config: dict[str, Any],
        *,
        continue_run: bool = False,
        mutable_keys: set[str] | None = None,
    ) -> None:
        """Persist protocol metadata and reject silent in-place drift.

        Historical ``run.json`` files may lack newer fields. A continuation
        validates every field they do contain, then augments them with the
        newly recorded configuration.
        """
        self.run_dir.mkdir(parents=True, exist_ok=True)
        persisted = config
        if continue_run and self.config_path.exists():
            previous = json.loads(self.config_path.read_text(encoding="utf8"))
            if "duels" not in previous and "duels_per_worker" in previous:
                previous["duels"] = int(previous["workers"]) * int(previous.pop("duels_per_worker"))
            mutable = mutable_keys or set()
            changed = sorted(
                change
                for key in previous.keys() & config.keys()
                if key not in mutable
                for change in _protocol_changes(previous[key], config[key], key)
            )
            if changed:
                raise ValueError(
                    "Refusing to continue with a changed run protocol. "
                    "Use --resume with a new --run directory to branch: " + ", ".join(changed)
                )
            persisted = {**previous, **config}
        self.config_path.write_text(
            json.dumps(persisted, indent=2, sort_keys=True), encoding="utf8"
        )


def checkpoint_version(path: Path) -> int:
    """Parse the numeric portion of a conventional model checkpoint path."""
    suffix = path.stem.removeprefix("model-")
    if not suffix.isdigit():
        raise ValueError(f"Expected a numbered model-*.pt checkpoint, got {path}")
    return int(suffix)
