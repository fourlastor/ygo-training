"""Composable logging sinks for training records."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Iterable
from pathlib import Path
from typing import Any


class TrainingLogger(ABC):
    """Sink receiving one complete record per training/evaluation update."""

    @abstractmethod
    def log(self, record: dict[str, Any]) -> None:
        """Consume a record."""

    def finish(self) -> None:
        """Close resources; default sinks are stateless."""


class JsonlLogger(TrainingLogger):
    """Append records without rewriting a live run."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def log(self, record: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf8") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")


class StdoutLogger(TrainingLogger):
    """Human/pipe-friendly JSON output sink."""

    def log(self, record: dict[str, Any]) -> None:
        print(json.dumps(record, sort_keys=True), flush=True)


class CompositeLogger(TrainingLogger):
    """Fan a record out to file, stdout, or future sinks."""

    def __init__(self, sinks: Iterable[TrainingLogger]) -> None:
        self.sinks = tuple(sinks)

    def log(self, record: dict[str, Any]) -> None:
        for sink in self.sinks:
            sink.log(record)

    def finish(self) -> None:
        for sink in self.sinks:
            sink.finish()
