from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from aila.queue import ObservationQueue
from aila.workers.config import WorkerConfig
from aila.workers.filesystem import (
    FileChangeEvent,
    FilesystemWatchConfig,
    FilesystemWorker,
)


@dataclass
class FakeFileEventAdapter:
    events: tuple[FileChangeEvent, ...]

    def poll_events(self) -> tuple[FileChangeEvent, ...]:
        return self.events


def test_filesystem_worker_emits_all_file_change_observation_kinds(tmp_path: Path) -> None:
    root = tmp_path / "watched"
    queue = ObservationQueue(tmp_path / "queue")
    worker = FilesystemWorker(
        _filesystem_config(),
        FakeFileEventAdapter(
            (
                _event(root / "created.txt", "created", 10, 0),
                _event(root / "changed.txt", "changed", 20, 1),
                _event(root / "deleted.txt", "deleted", 30, 2),
            )
        ),
        queue,
        FilesystemWatchConfig(paths=(root,)),
    )

    observations = worker.poll_once()

    assert [observation.kind for observation in observations] == [
        "file.created",
        "file.changed",
        "file.deleted",
    ]
    assert [observation.payload.change for observation in observations] == [
        "created",
        "changed",
        "deleted",
    ]
    assert observations[0].payload.path == str(root / "created.txt")
    assert observations[0].payload.size == 10
    assert observations[2].payload.size == 0
    assert [item.observation.kind for item in queue.drain()] == [
        "file.created",
        "file.changed",
        "file.deleted",
    ]


def test_filesystem_worker_applies_watch_roots_and_ignore_globs(tmp_path: Path) -> None:
    root = tmp_path / "watched"
    outside = tmp_path / "outside"
    worker = FilesystemWorker(
        _filesystem_config(),
        FakeFileEventAdapter(
            (
                _event(root / "keep.txt", "changed", 1, 0),
                _event(root / ".git" / "config", "changed", 1, 1),
                _event(root / "__pycache__" / "module.pyc", "changed", 1, 2),
                _event(root / "scratch.tmp", "changed", 1, 3),
                _event(outside / "skip.txt", "changed", 1, 4),
            )
        ),
        ObservationQueue(tmp_path / "queue"),
        FilesystemWatchConfig(
            paths=(root,),
            ignore=("**/.git/**", "**/__pycache__/**", "*.tmp"),
        ),
    )

    observations = worker.poll_once()

    assert [observation.payload.path for observation in observations] == [
        str(root / "keep.txt")
    ]


def test_filesystem_worker_debounces_successive_events_for_same_path(tmp_path: Path) -> None:
    root = tmp_path / "watched"
    target = root / "notes.txt"
    worker = FilesystemWorker(
        _filesystem_config(),
        FakeFileEventAdapter(
            (
                _event(target, "changed", 1, 0),
                _event(target, "changed", 2, 100),
                _event(root / "other.txt", "created", 3, 150),
                _event(target, "changed", 4, 700),
            )
        ),
        ObservationQueue(tmp_path / "queue"),
        FilesystemWatchConfig(paths=(root,), debounce_ms=500),
    )

    observations = worker.poll_once()

    assert [(obs.payload.path, obs.payload.size) for obs in observations] == [
        (str(target), 2),
        (str(root / "other.txt"), 3),
        (str(target), 4),
    ]


def test_filesystem_worker_uses_adapter_metadata_without_touching_file_contents(
    tmp_path: Path,
) -> None:
    root = tmp_path / "watched"
    missing = root / "never-created.txt"
    worker = FilesystemWorker(
        _filesystem_config(),
        FakeFileEventAdapter((_event(missing, "created", 42, 0),)),
        ObservationQueue(tmp_path / "queue"),
        FilesystemWatchConfig(paths=(root,)),
    )

    observations = worker.poll_once()

    assert observations[0].payload.path == str(missing)
    assert observations[0].payload.size == 42


def _filesystem_config() -> WorkerConfig:
    return WorkerConfig.model_validate(
        {
            "worker": "filesystem",
            "role": "sensor",
            "backend": {"kind": "deterministic", "placement": "local"},
            "emits": ["file.changed", "file.created", "file.deleted"],
            "verbs": [],
        }
    )


def _event(path: Path, change: str, size: int, offset_ms: int) -> FileChangeEvent:
    return FileChangeEvent(
        path=path,
        change=change,
        size=size,
        mtime=datetime(2026, 7, 13, 12, 0, tzinfo=UTC)
        + timedelta(milliseconds=offset_ms),
    )
