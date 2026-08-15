from __future__ import annotations

import fnmatch
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Protocol

from aila.contracts import FileChange
from aila.queue import ObservationQueue
from aila.workers.backends import BackendObservation
from aila.workers.base import SensorWorker
from aila.workers.config import WorkerConfig

_KIND_BY_CHANGE: dict[FileChange, str] = {
    "changed": "file.changed",
    "created": "file.created",
    "deleted": "file.deleted",
}


@dataclass(frozen=True)
class FileChangeEvent:
    path: Path
    change: FileChange
    size: int
    mtime: datetime

    def __post_init__(self) -> None:
        if self.size < 0:
            raise ValueError("file event size must be non-negative")


class FileEventAdapter(Protocol):
    def poll_events(self) -> tuple[FileChangeEvent, ...]:
        raise NotImplementedError


@dataclass(frozen=True)
class FilesystemWatchConfig:
    paths: tuple[Path, ...]
    ignore: tuple[str, ...] = ()
    debounce_ms: int = 500

    def __post_init__(self) -> None:
        if not self.paths:
            raise ValueError("filesystem watch paths must not be empty")
        if self.debounce_ms < 0:
            raise ValueError("filesystem debounce_ms must be non-negative")
        object.__setattr__(
            self,
            "paths",
            tuple(_normalize_path(path) for path in self.paths),
        )
        object.__setattr__(
            self,
            "ignore",
            tuple(_normalize_glob(pattern) for pattern in self.ignore),
        )


class FilesystemWorker(SensorWorker):
    def __init__(
        self,
        config: WorkerConfig,
        adapter: FileEventAdapter,
        queue: ObservationQueue,
        watch: FilesystemWatchConfig,
    ) -> None:
        if config.worker != "filesystem":
            raise ValueError(f"filesystem worker cannot use config for {config.worker}")
        super().__init__(config, _FilesystemBackend(adapter, watch), queue)


class _FilesystemBackend:
    def __init__(self, adapter: FileEventAdapter, watch: FilesystemWatchConfig) -> None:
        self._adapter = adapter
        self._watch = watch

    def poll(self) -> tuple[BackendObservation, ...]:
        events = (
            event
            for event in self._adapter.poll_events()
            if _is_watched(event.path, self._watch.paths)
            and not _is_ignored(event.path, self._watch.ignore)
        )
        return tuple(
            _observation_for_event(event)
            for event in _debounced(events, debounce_ms=self._watch.debounce_ms)
        )

    def handle_command(self, command: object) -> object:
        raise NotImplementedError("filesystem worker does not support commands")


def _observation_for_event(event: FileChangeEvent) -> BackendObservation:
    change = event.change
    size = 0 if change == "deleted" else event.size
    return BackendObservation(
        kind=_KIND_BY_CHANGE[change],
        ts=event.mtime,
        payload={
            "path": str(_normalize_path(event.path)),
            "change": change,
            "size": size,
            "mtime": event.mtime,
        },
    )


def _debounced(
    events: Iterable[FileChangeEvent],
    *,
    debounce_ms: int,
) -> tuple[FileChangeEvent, ...]:
    kept: list[FileChangeEvent] = []
    last_index_by_path: dict[Path, int] = {}

    for event in events:
        if not isinstance(event, FileChangeEvent):
            raise TypeError("filesystem adapter must yield FileChangeEvent instances")
        path = _normalize_path(event.path)
        event = FileChangeEvent(
            path=path,
            change=event.change,
            size=event.size,
            mtime=event.mtime,
        )
        previous_index = last_index_by_path.get(path)
        if previous_index is None:
            last_index_by_path[path] = len(kept)
            kept.append(event)
            continue

        previous = kept[previous_index]
        delta_ms = abs((event.mtime - previous.mtime).total_seconds() * 1000)
        if delta_ms <= debounce_ms:
            kept[previous_index] = event
        else:
            last_index_by_path[path] = len(kept)
            kept.append(event)

    return tuple(kept)


def _is_watched(path: Path, roots: tuple[Path, ...]) -> bool:
    normalized = _normalize_path(path)
    return any(normalized == root or normalized.is_relative_to(root) for root in roots)


def _is_ignored(path: Path, patterns: tuple[str, ...]) -> bool:
    normalized = _normalize_path(path)
    posix_path = normalized.as_posix()
    return any(
        fnmatch.fnmatchcase(posix_path, pattern)
        or fnmatch.fnmatchcase(normalized.name, pattern)
        for pattern in patterns
    )


def _normalize_path(path: Path) -> Path:
    normalized = Path(path).expanduser()
    if not normalized.is_absolute():
        raise ValueError(f"filesystem event path must be absolute: {path}")
    return normalized


def _normalize_glob(pattern: str) -> str:
    if not pattern:
        raise ValueError("filesystem ignore globs must not be empty")
    return pattern.replace("\\", "/")
