from __future__ import annotations

import json
import os
import random
from collections.abc import Iterable
from hashlib import sha256
from pathlib import Path
from typing import Any

import torch

from .card_table import CardTable
from .config import left_out
from .model import PolicyValueNet
from .schema import model_policy_input_schema, schema_digest

# Format 13 is the first whose input is built from the view of a ygo-policies
# seat. Checkpoints of earlier formats read another input and do not load
# here.
# Format 14 is format 13 with the lead head. Format 15 (2026-10-16) reads no
# deck list: checkpoints of formats 13 and 14 read both and do not load here,
# and the revisions before it run them. Format 16 (2026-10-17) is format 15
# told more of a card than its input gives, by the code it carries
# (`card_table`): the cards' descriptions, or identities it hides. A model
# told nothing more is written in format 15, which revisions before 16 load
# as well; the input of the two is the same.
FORMAT_VERSION = 15
CARDS_FORMAT = 16
LOADED_FORMATS = (FORMAT_VERSION, CARDS_FORMAT)
# A checkpoint's files are named as they will be with this after it, until
# they are finished. What lists a run's checkpoints looks for `model-*.pt`
# and passes such a name by. It has no dot of its own: torch names the
# archive it writes after the file, less its last extension, and so writes
# the bytes it would write to the checkpoint's own name.
UNFINISHED = "-tmp"


def _fingerprint_value(digest, value: Any) -> None:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        digest.update(b"tensor\0")
        digest.update(str(tensor.dtype).encode())
        digest.update(b"\0")
        digest.update(json.dumps(list(tensor.shape)).encode())
        digest.update(b"\0")
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    elif isinstance(value, dict):
        digest.update(b"dict\0")
        for key in sorted(value, key=lambda item: (type(item).__name__, repr(item))):
            _fingerprint_value(digest, key)
            _fingerprint_value(digest, value[key])
    elif isinstance(value, (list, tuple)):
        digest.update(b"tuple\0" if isinstance(value, tuple) else b"list\0")
        for item in value:
            _fingerprint_value(digest, item)
    else:
        digest.update(type(value).__name__.encode())
        digest.update(b"\0")
        digest.update(repr(value).encode())
        digest.update(b"\0")


def value_fingerprint(value: Any) -> str:
    """Return a stable digest for nested scalar/tensor training data."""
    digest = sha256()
    _fingerprint_value(digest, value)
    return digest.hexdigest()


def training_state_fingerprints(
    path: Path, map_location: str | torch.device = "cpu"
) -> dict[str, str | None]:
    """Hash resumable state while excluding variable checkpoint metadata."""
    payload = torch.load(path, map_location=map_location, weights_only=False)
    result: dict[str, str | None] = {}
    for name in ("model", "optimizer", "rng_state"):
        if name not in payload:
            result[name] = None
            continue
        result[name] = value_fingerprint(payload[name])
    return result


def format_of(model: torch.nn.Module) -> int:
    """The format a model's checkpoints are written in."""
    told = getattr(model, "reads_descriptions", False) or getattr(model, "hides_identities", False)
    return CARDS_FORMAT if told else FORMAT_VERSION


def card_options(model: torch.nn.Module) -> dict[str, Any]:
    """What a model is told of a card beside its input, as its checkpoints
    say it: whether it reads the cards' descriptions, the share of the cards
    whose identities it hides in an update, and its table's fingerprint."""
    state = model.state_dict()
    table = CardTable.of_state(state)
    return {
        "card_descriptions": bool(getattr(model, "reads_descriptions", False)),
        "hide_identities": float(getattr(model, "hide_identities", 0.0)),
        "card_table": None
        if table is None
        else value_fingerprint(
            {name: value for name, value in vars(table).items() if value is not None}
        ),
    }


def saved_card_options(metadata: dict[str, Any]) -> dict[str, Any]:
    """What a checkpoint's model was told of a card beside its input: nothing
    for one of format 15."""
    return {
        "card_descriptions": bool(metadata.get("card_descriptions", False)),
        "hide_identities": float(metadata.get("hide_identities", 0.0)),
        "card_table": metadata.get("card_table"),
    }


def folded_pairing(metadata: dict[str, Any]) -> tuple[str, str] | None:
    """The one pairing of decks a checkpoint written by
    `tools/fold_deck_lists.py` plays, its seat's deck and the other seat's:
    the two deck lists a network of format 14 read are folded into its first
    layer as they are in those duels alone. None for every other checkpoint."""
    folded = metadata.get("folded_deck_lists")
    if folded is None:
        return None
    return str(folded["own"]), str(folded["opponent"])


def _check_pairings(
    path: Path, metadata: dict[str, Any], pairings: Iterable[tuple[str, str]] | None
) -> None:
    pairing = folded_pairing(metadata)
    if pairing is None:
        return
    named = None if pairings is None else set(pairings)
    if named != {pairing}:
        played = (
            ": it was not told what it plays here"
            if named is None
            else ": here it would play " + ", ".join(f"{a} against {b}" for a, b in sorted(named))
        )
        raise ValueError(
            f"{path} was folded for {pairing[0]} against {pairing[1]} "
            f"(tools/fold_deck_lists.py) and plays nothing else{played}"
        )


def _description(path: Path) -> Path:
    """Where a checkpoint's step and metadata are written for a reader."""
    return path.with_suffix(path.suffix + ".json")


def _unfinished(path: Path) -> Path:
    """What a file is called while it is written."""
    return path.with_name(path.name + UNFINISHED)


def write(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    metadata: dict,
) -> None:
    """Write a checkpoint and its description whole, under other names than
    theirs: `finish` gives them their own, and `discard` removes them.

    What has to be done before the checkpoint counts is done between the
    two: a process stopped meanwhile leaves the checkpoint before it the
    latest of its run. What a killed one leaves under the other names is
    written over when the run goes on, which writes the same checkpoint
    again.
    """
    board_representation = str(model.board_representation)
    leave_out = left_out(getattr(model, "leave_out", ()))
    metadata = dict(metadata)
    saved_board_representation = metadata.setdefault("board_representation", board_representation)
    if saved_board_representation != board_representation:
        raise ValueError(
            f"Checkpoint metadata board representation is {saved_board_representation!r}, "
            f"but the model uses {board_representation!r}"
        )
    if left_out(metadata.setdefault("leave_out", list(leave_out))) != leave_out:
        raise ValueError(
            f"Checkpoint metadata leaves out {metadata['leave_out']!r}, but the model {list(leave_out)!r}"
        )
    input_schema = model_policy_input_schema(
        board_representation=board_representation, leave_out=leave_out
    )
    supplied_schema = metadata.setdefault("policy_input_schema", input_schema)
    if supplied_schema != input_schema:
        raise ValueError("Checkpoint metadata policy input schema is not canonical")
    input_schema_digest = schema_digest(input_schema)
    supplied_digest = metadata.setdefault("policy_input_schema_digest", input_schema_digest)
    if supplied_digest != input_schema_digest:
        raise ValueError("Checkpoint metadata policy input schema digest is not canonical")
    for name, value in card_options(model).items():
        if metadata.setdefault(name, value) != value:
            raise ValueError(
                f"Checkpoint metadata {name} is {metadata[name]!r}, the model's {value!r}"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        torch.save(
            {
                "format_version": format_of(model),
                "step": step,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "rng_state": {
                    "python": random.getstate(),
                    "torch": torch.get_rng_state(),
                    "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                },
                "metadata": metadata,
            },
            _unfinished(path),
        )
        _unfinished(_description(path)).write_text(
            json.dumps({"format_version": format_of(model), "step": step, **metadata}, indent=2)
        )
    except BaseException:
        discard(path)
        raise


# What a run keeps beside each of its checkpoints, never in one, so that
# nothing that plays holds it: the critic that sees the hidden cards
# (`ygo_training/critic.py`) and the head that learns how cards work
# (`ygo_training/card_head.py`).
KEPT_BESIDE = ("critic", "card-head")


def beside(path: Path, kind: str) -> Path:
    """Where a run keeps its ``kind`` (one of `KEPT_BESIDE`) beside its
    checkpoint ``path``."""
    return path.with_name(f"{kind}-" + path.name.removeprefix("model-"))


def write_beside(
    path: Path, kind: str, module: torch.nn.Module, optimizer: torch.optim.Optimizer, step: int
) -> None:
    """Write what is kept beside the checkpoint ``path`` under another name
    than its own, as `write` writes the checkpoint: `finish` gives it its
    name with the checkpoint's, and `discard` removes it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"step": step, kind: module.state_dict(), "optimizer": optimizer.state_dict()},
        _unfinished(beside(path, kind)),
    )


def load_beside(
    path: Path,
    kind: str,
    module: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    map_location: str | torch.device = "cpu",
) -> None:
    """What was kept beside the checkpoint ``path``, and its optimizer, for a
    run that goes on from it."""
    kept = beside(path, kind)
    if not kept.exists():
        raise FileNotFoundError(f"No {kind} is kept beside {path}: {kept} is missing")
    payload = torch.load(kept, map_location=map_location, weights_only=False)
    module.load_state_dict(payload[kind])
    optimizer.load_state_dict(payload["optimizer"])


def finish(path: Path) -> None:
    """Give a written checkpoint and its description their names, the
    checkpoint last: none is there without its description, nor without what
    is kept beside it where that was written."""
    kept = [beside(path, kind) for kind in KEPT_BESIDE]
    files = [_description(path), *(file for file in kept if _unfinished(file).exists()), path]
    for file in files:
        os.replace(_unfinished(file), file)


def discard(path: Path) -> None:
    """Remove what was written of a checkpoint that is not to be finished."""
    for file in (path, _description(path), *(beside(path, kind) for kind in KEPT_BESIDE)):
        _unfinished(file).unlink(missing_ok=True)


def save(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    metadata: dict,
) -> None:
    """Write a checkpoint and give it its name: cut short, it leaves the
    checkpoint before it the latest of its run."""
    write(path, model, optimizer, step, metadata)
    try:
        finish(path)
    except BaseException:
        discard(path)
        raise


def load(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str | torch.device = "cpu",
    expected_board_representation: str | None = None,
    expected_leave_out: tuple[str, ...] | None = None,
    pairings: Iterable[tuple[str, str]] | None = None,
) -> dict:
    """Load a compatible checkpoint, optionally directly onto the training
    device.  The model is set to leave out of its input what the checkpoint's
    did (`TrainerConfig.leave_out`), which ``expected_leave_out`` refuses to
    be other than it names.  It is to be told of a card what the
    checkpoint's was (`card_options`), and is left hiding what a model hides
    at play (`PolicyValueNet.play`): an update of training hides its own.
    A checkpoint folded for one pairing of decks (`folded_pairing`) loads
    only where ``pairings``, the decks of the seat it plays and of the other
    in every duel it will play, are that pairing alone."""
    payload = torch.load(path, map_location=map_location, weights_only=False)
    saved_format = payload.get("format_version")
    if saved_format not in LOADED_FORMATS:
        raise ValueError(
            f"Checkpoint format {saved_format} is not format "
            f"{' or '.join(map(str, LOADED_FORMATS))}: "
            "it was trained on another input and cannot be loaded here"
        )
    saved_metadata = payload.get("metadata", {})
    _check_pairings(path, saved_metadata, pairings)
    saved_board_representation = saved_metadata.get("board_representation")
    saved_schema = saved_metadata.get("policy_input_schema")
    saved_schema_digest = saved_metadata.get("policy_input_schema_digest")
    if saved_schema is None or saved_schema_digest is None:
        raise ValueError("Checkpoint has no policy input schema")
    if schema_digest(saved_schema) != saved_schema_digest:
        raise ValueError("Checkpoint policy input schema digest mismatch")
    saved_leave_out = left_out(saved_metadata.get("leave_out", ()))
    expected_schema = model_policy_input_schema(
        board_representation=str(saved_board_representation), leave_out=saved_leave_out
    )
    if saved_schema != expected_schema:
        raise ValueError("Checkpoint policy input schema is incompatible with this runtime")
    if expected_leave_out is not None and saved_leave_out != left_out(expected_leave_out):
        raise ValueError(
            f"Checkpoint leaves out {', '.join(saved_leave_out) or 'nothing'} of its input, "
            f"expected {', '.join(left_out(expected_leave_out)) or 'nothing'}"
        )
    if (
        expected_board_representation is not None
        and saved_board_representation != expected_board_representation
    ):
        raise ValueError(
            f"Checkpoint board representation is {saved_board_representation!r}, "
            f"expected {expected_board_representation!r}"
        )
    saved_cards = saved_card_options(saved_metadata)
    told = {
        "reads the cards' descriptions": (
            saved_cards["card_descriptions"],
            bool(getattr(model, "reads_descriptions", False)),
        ),
        "hides identities": (
            saved_cards["hide_identities"] > 0.0,
            bool(getattr(model, "hides_identities", False)),
        ),
    }
    for what, (theirs, mine) in told.items():
        if theirs != mine:
            raise ValueError(
                f"The checkpoint's model {'' if theirs else 'never '}{what}, this one "
                f"{'does' if mine else 'does not'}: build it as the checkpoint says "
                "(`checkpoints.saved_card_options`)"
            )
    saved_optimizer = payload.get("optimizer")
    model.load_state_dict(payload["model"])
    if hasattr(model, "leave_out"):
        model.leave_out = saved_leave_out
    if isinstance(model, PolicyValueNet):
        model.play()
    if optimizer is not None:
        if saved_optimizer is None:
            raise ValueError("Checkpoint has no optimizer state and cannot resume training")
        optimizer.load_state_dict(saved_optimizer)
        rng_state = payload.get("rng_state")
        if rng_state is not None:
            random.setstate(rng_state["python"])
            torch.set_rng_state(rng_state["torch"].cpu())
            cuda_states = rng_state.get("cuda")
            if cuda_states is not None and torch.cuda.is_available():
                if len(cuda_states) != torch.cuda.device_count():
                    raise ValueError(
                        "Checkpoint CUDA RNG state does not match the available device count"
                    )
                torch.cuda.set_rng_state_all([state.cpu() for state in cuda_states])
    return payload


def load_inference_model(
    path: Path,
    *,
    device: str | torch.device = "cpu",
    pairings: Iterable[tuple[str, str]] | None = None,
) -> tuple[PolicyValueNet, dict]:
    """Construct and load the policy architecture a checkpoint describes
    (``pairings`` as for `load`)."""

    inspected = torch.load(path, map_location="cpu", weights_only=False)
    described = inspected.get("metadata", {})
    cards = saved_card_options(described)
    model = PolicyValueNet(
        board_representation=str(described.get("board_representation")),
        leave_out=left_out(described.get("leave_out", ())),
        cards=CardTable.of_state(inspected["model"]),
        descriptions=cards["card_descriptions"],
        hide_identities=cards["hide_identities"],
    ).to(device)
    payload = load(path, model, map_location=device, pairings=pairings)
    model.eval()
    return model, payload
