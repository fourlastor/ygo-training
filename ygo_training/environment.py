"""The duel environment: the Rust library that runs duels for the trainer.

``crates/ygo-env`` runs duels on the engine, cards, scripts, decks and pilots
of the ``native/ygo-policies`` checkout.  This module loads its shared
library and passes JSON both ways.  One :class:`Environment` per process: the
engine underneath is not shared between threads.

A duel is described by a setup (see ``Setup`` in ``crates/ygo-env``)::

    {"seed": 7, "decks": ["blackwing", "monarch"],
     "seats": [{"control": "caller"}, {"control": "pilot"}]}

Seat 0 goes first.  A ``pilot`` seat is played by the policy written for its
deck, a ``caller`` seat by whoever drives the episode, and a ``record`` seat
from a recorded duel.  :meth:`Episode.advance` runs the duel until the caller
has to decide or the duel ends.
"""

from __future__ import annotations

import ctypes
import json
import os
from functools import cache
from pathlib import Path
from typing import Any, Self

POLICIES_DIRECTORY = "native/ygo-policies"
CARDS_DATABASE = f"{POLICIES_DIRECTORY}/vendor/BabelCdb/cards.cdb"
CARD_SCRIPTS = f"{POLICIES_DIRECTORY}/vendor/CardScripts"
DECKS_DIRECTORY = f"{POLICIES_DIRECTORY}/decks"
LIBRARY = "target/release/libygo_env.so"
BUILD_COMMAND = "cargo build --release"


class DuelError(RuntimeError):
    """The environment refused a call; the message is its own."""


def library_path(root: str | Path) -> Path:
    """Where the checkout's build of the environment is."""
    override = os.environ.get("YGO_ENV_LIBRARY")
    path = Path(override) if override else Path(root).resolve() / LIBRARY
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing. Build the environment first: {BUILD_COMMAND}")
    return path


@cache
def _load(path: str) -> ctypes.CDLL:
    lib = ctypes.CDLL(path)
    lib.ygo_env_open.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p]
    lib.ygo_env_open.restype = ctypes.c_void_p
    lib.ygo_env_describe.argtypes = [ctypes.c_void_p]
    lib.ygo_env_describe.restype = ctypes.c_char_p
    lib.ygo_env_start.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    lib.ygo_env_start.restype = ctypes.c_void_p
    lib.ygo_env_advance.argtypes = [ctypes.c_void_p]
    lib.ygo_env_advance.restype = ctypes.c_char_p
    lib.ygo_env_choose.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    lib.ygo_env_choose.restype = ctypes.c_int32
    lib.ygo_env_finish.argtypes = [ctypes.c_void_p]
    lib.ygo_env_finish.restype = None
    lib.ygo_env_close.argtypes = [ctypes.c_void_p]
    lib.ygo_env_close.restype = None
    lib.ygo_env_last_error.argtypes = []
    lib.ygo_env_last_error.restype = ctypes.c_char_p
    lib.ygo_env_pool.argtypes = []
    lib.ygo_env_pool.restype = ctypes.c_char_p
    lib.ygo_env_batch.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    lib.ygo_env_batch.restype = ctypes.c_void_p
    lib.ygo_env_pack.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    lib.ygo_env_pack.restype = ctypes.c_void_p
    lib.ygo_env_batch_layout.argtypes = [ctypes.c_void_p]
    lib.ygo_env_batch_layout.restype = ctypes.c_char_p
    lib.ygo_env_batch_data.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    lib.ygo_env_batch_data.restype = ctypes.c_void_p
    lib.ygo_env_batch_free.argtypes = [ctypes.c_void_p]
    lib.ygo_env_batch_free.restype = None
    lib.ygo_pool_open.argtypes = [
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_uint32,
    ]
    lib.ygo_pool_open.restype = ctypes.c_void_p
    lib.ygo_pool_submit.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
    ]
    lib.ygo_pool_submit.restype = ctypes.c_int64
    lib.ygo_pool_batch.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.c_size_t,
    ]
    lib.ygo_pool_batch.restype = ctypes.c_void_p
    lib.ygo_pool_batch_whole.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.c_size_t,
    ]
    lib.ygo_pool_batch_whole.restype = ctypes.c_void_p
    lib.ygo_pool_batch_labels.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.c_size_t,
    ]
    lib.ygo_pool_batch_labels.restype = ctypes.c_void_p
    lib.ygo_pool_kept.argtypes = [ctypes.c_void_p]
    lib.ygo_pool_kept.restype = ctypes.c_int64
    lib.ygo_pool_forget.argtypes = [ctypes.c_void_p]
    lib.ygo_pool_forget.restype = ctypes.c_int32
    lib.ygo_pool_forget_rows.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.c_size_t,
    ]
    lib.ygo_pool_forget_rows.restype = ctypes.c_int32
    lib.ygo_pool_step.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32), ctypes.c_size_t]
    lib.ygo_pool_step.restype = ctypes.c_void_p
    lib.ygo_pool_begin.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_size_t,
    ]
    lib.ygo_pool_begin.restype = ctypes.c_int32
    lib.ygo_pool_finish.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    lib.ygo_pool_finish.restype = ctypes.c_void_p
    lib.ygo_pool_round_layout.argtypes = [ctypes.c_void_p]
    lib.ygo_pool_round_layout.restype = ctypes.c_char_p
    lib.ygo_pool_round_data.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    lib.ygo_pool_round_data.restype = ctypes.c_void_p
    lib.ygo_pool_round_free.argtypes = [ctypes.c_void_p]
    lib.ygo_pool_round_free.restype = None
    lib.ygo_pool_close.argtypes = [ctypes.c_void_p]
    lib.ygo_pool_close.restype = None
    return lib


def pool(root: str | Path = ".") -> dict[str, str]:
    """Policy id -> name of its deck list: what can be seated.

    Known from the library alone, without opening the engine.
    """
    entries = json.loads(_load(str(library_path(root))).ygo_env_pool())
    return {str(entry["id"]): str(entry["deck"]) for entry in entries}


class Environment:
    """The engine with its cards and scripts, and the decks of the pool."""

    def __init__(self, root: str | Path = "."):
        root = Path(root).resolve()
        self.root = root
        self._lib = _load(str(library_path(root)))
        handle = self._lib.ygo_env_open(
            str(root / CARDS_DATABASE).encode(),
            str(root / CARD_SCRIPTS).encode(),
            str(root / DECKS_DIRECTORY).encode(),
        )
        if not handle:
            raise self._error("cannot open the environment")
        self._handle: int | None = handle
        described = json.loads(self._lib.ygo_env_describe(handle))
        # Policy id -> {"id", "deck" (the list's name), "main", "extra"}.
        self.decks: dict[str, dict[str, Any]] = {
            str(entry["id"]): entry for entry in described["decks"]
        }
        # Vocabulary -> {name: number}, for the numbers of a decision.
        self.vocabulary: dict[str, dict[str, int]] = {
            name: {str(label): int(number) for label, number in pairs}
            for name, pairs in described["vocabulary"].items()
        }

    def _error(self, what: str) -> DuelError:
        message = self._lib.ygo_env_last_error()
        return DuelError(f"{what}: {message.decode() if message else 'unknown error'}")

    def decklist(self, deck: str) -> list[int]:
        """The Main Deck then the Extra Deck of ``deck``, as card codes."""
        entry = self.decks[deck]
        return [*entry["main"], *entry["extra"]]

    def start(self, setup: dict[str, Any]) -> Episode:
        """Start a duel; nothing is played until its first ``advance``."""
        if self._handle is None:
            raise DuelError("the environment is closed")
        session = self._lib.ygo_env_start(self._handle, json.dumps(setup).encode())
        if not session:
            raise self._error("cannot start the duel")
        return Episode(self, session)

    def batch(self, views: list[dict[str, Any]]) -> dict[str, Any]:
        """Decisions as the batch the model is given, built by the library.

        ``views`` are decisions as :meth:`Episode.advance` returned them.
        """
        if self._handle is None:
            raise DuelError("the environment is closed")
        request = json.dumps({"views": views})
        batch = self._lib.ygo_env_batch(self._handle, request.encode())
        if not batch:
            raise self._error("cannot build the batch")
        return self._arrays(batch)

    def pack(self, view: dict[str, Any]) -> dict[str, Any]:
        """One decision as the arrays the model reads, before a batch pads
        them, built by the library."""
        if self._handle is None:
            raise DuelError("the environment is closed")
        packed = self._lib.ygo_env_pack(self._handle, json.dumps(view).encode())
        if not packed:
            raise self._error("cannot pack the decision")
        return self._arrays(packed)

    def _arrays(self, batch: int) -> dict[str, Any]:
        """The arrays the library made, copied, and the library's given back."""
        import numpy as np

        try:
            arrays: dict[str, Any] = {}
            for index, entry in enumerate(json.loads(self._lib.ygo_env_batch_layout(batch))):
                kind = np.dtype(entry["dtype"])
                count = int(np.prod(entry["shape"], dtype=np.int64))
                data = self._lib.ygo_env_batch_data(batch, index)
                if count and not data:
                    raise self._error(f"cannot read {entry['name']}")
                # The library keeps the numbers only as long as the batch.
                flat = (
                    np.frombuffer(ctypes.string_at(data, count * kind.itemsize), dtype=kind)
                    if count
                    else np.zeros(0, dtype=kind)
                )
                arrays[entry["name"]] = flat.reshape(entry["shape"]).copy()
            return arrays
        finally:
            self._lib.ygo_env_batch_free(batch)

    def close(self) -> None:
        """Close the engine, once every episode of it is closed."""
        if self._handle is not None:
            self._lib.ygo_env_close(self._handle)
            self._handle = None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


class Episode:
    """One duel of an :class:`Environment`."""

    def __init__(self, environment: Environment, session: int):
        # The environment has to outlive its episodes.
        self._environment = environment
        self._lib = environment._lib
        self._session: int | None = session

    def advance(self) -> dict[str, Any]:
        """Run the duel until the caller has to decide, or to its end.

        ``samples`` are the decisions of observed seats answered on the way,
        ``decision`` the one to answer with :meth:`choose`, ``over`` the end.
        """
        event = self._lib.ygo_env_advance(self._session)
        if event is None:
            raise self._environment._error("the duel failed")
        return json.loads(event)

    def choose(self, index: int) -> None:
        """Answer the pending decision with one of its ``choices``."""
        if self._lib.ygo_env_choose(self._session, int(index)) != 0:
            raise self._environment._error("the answer was refused")

    def close(self) -> None:
        if self._session is not None:
            self._lib.ygo_env_finish(self._session)
            self._session = None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __del__(self) -> None:
        if getattr(self, "_session", None) is not None and self._environment._handle is not None:
            self.close()
