from __future__ import annotations

import os
import re
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from aila.contracts import Observation

_FILENAME_RE = re.compile(r"^(?P<ts>\d{8}T\d{6}\.\d{6}Z)__(?P<obs_id>.+)\.json$")
_SAFE_OBS_ID_RE = re.compile(r"[^A-Za-z0-9_.-]+")


@dataclass(frozen=True)
class ArchiveRetention:
    max_bytes: int | None = 10 * 1024 * 1024
    max_age: timedelta | None = timedelta(days=7)

    def __post_init__(self) -> None:
        if self.max_bytes is not None and self.max_bytes < 0:
            raise ValueError("max_bytes must be non-negative")
        if self.max_age is not None and self.max_age < timedelta(0):
            raise ValueError("max_age must be non-negative")


@dataclass(frozen=True)
class QueuedObservation:
    observation: Observation
    path: Path
    archived_path: Path


class ObservationQueue:
    def __init__(
        self,
        root: Path,
        *,
        retention: ArchiveRetention | None = None,
    ) -> None:
        self.root = Path(root)
        self.pending_dir = self.root / "pending"
        self.inflight_dir = self.root / "inflight"
        self.archive_dir = self.root / "archive"
        self.retention = retention if retention is not None else ArchiveRetention()
        self._ensure_dirs()

    def append(self, observation: Observation) -> Path:
        self._ensure_dirs()
        filename = _filename_for(observation)
        target = self.pending_dir / filename
        if target.exists() or (self.inflight_dir / filename).exists() or (self.archive_dir / filename).exists():
            raise FileExistsError(f"observation already exists in queue: {filename}")

        temp = self.pending_dir / f".{filename}.{uuid.uuid4().hex}.tmp"
        try:
            temp.write_text(
                observation.model_dump_json(),
                encoding="utf-8",
            )
            _fsync_file(temp)
            temp.replace(target)
            _fsync_dir(self.pending_dir)
        finally:
            if temp.exists():
                temp.unlink()
        return target

    def drain(self, batch_size: int | None = None) -> list[QueuedObservation]:
        if batch_size is not None and batch_size < 0:
            raise ValueError("batch_size must be non-negative")

        self._ensure_dirs()
        self.reclaim_inflight()

        pending = sorted(self.pending_dir.glob("*.json"))
        if batch_size is not None:
            pending = pending[:batch_size]

        drained: list[QueuedObservation] = []
        for pending_path in pending:
            inflight_path = self.inflight_dir / pending_path.name
            pending_path.replace(inflight_path)
            _fsync_dir(self.pending_dir)
            _fsync_dir(self.inflight_dir)

            observation = Observation.model_validate_json(inflight_path.read_text(encoding="utf-8"))
            archive_path = self.archive_dir / inflight_path.name
            inflight_path.replace(archive_path)
            _fsync_dir(self.inflight_dir)
            _fsync_dir(self.archive_dir)
            drained.append(
                QueuedObservation(
                    observation=observation,
                    path=inflight_path,
                    archived_path=archive_path,
                )
            )

        self.enforce_retention()
        return drained

    def reclaim_inflight(self) -> list[Path]:
        self._ensure_dirs()
        reclaimed: list[Path] = []
        for inflight_path in sorted(self.inflight_dir.glob("*.json")):
            pending_path = self.pending_dir / inflight_path.name
            if pending_path.exists():
                raise FileExistsError(f"cannot reclaim duplicate pending observation: {pending_path.name}")
            inflight_path.replace(pending_path)
            reclaimed.append(pending_path)
        if reclaimed:
            _fsync_dir(self.inflight_dir)
            _fsync_dir(self.pending_dir)
        return reclaimed

    def enforce_retention(self, *, now: datetime | None = None) -> None:
        self._ensure_dirs()
        now_utc = _to_utc(now if now is not None else datetime.now(UTC))
        archive_files = sorted(self.archive_dir.glob("*.json"))

        if self.retention.max_age is not None:
            for path in list(archive_files):
                file_ts = _timestamp_from_filename(path.name)
                if file_ts is not None and now_utc - file_ts > self.retention.max_age:
                    path.unlink()
                    archive_files.remove(path)

        if self.retention.max_bytes is not None:
            total_bytes = sum(path.stat().st_size for path in archive_files if path.exists())
            for path in archive_files:
                if total_bytes <= self.retention.max_bytes:
                    break
                if not path.exists():
                    continue
                total_bytes -= path.stat().st_size
                path.unlink()

        _fsync_dir(self.archive_dir)

    def _ensure_dirs(self) -> None:
        self.pending_dir.mkdir(parents=True, exist_ok=True)
        self.inflight_dir.mkdir(parents=True, exist_ok=True)
        self.archive_dir.mkdir(parents=True, exist_ok=True)


def _filename_for(observation: Observation) -> str:
    timestamp = _to_utc(observation.ts).strftime("%Y%m%dT%H%M%S.%fZ")
    obs_id = _safe_obs_id(observation.obs_id)
    return f"{timestamp}__{obs_id}.json"


def _safe_obs_id(obs_id: str) -> str:
    safe = _SAFE_OBS_ID_RE.sub("_", obs_id).strip("._")
    if not safe:
        raise ValueError("obs_id does not contain any filename-safe characters")
    return safe


def _timestamp_from_filename(filename: str) -> datetime | None:
    match = _FILENAME_RE.match(filename)
    if match is None:
        return None
    return datetime.strptime(match.group("ts"), "%Y%m%dT%H%M%S.%fZ").replace(tzinfo=UTC)


def _to_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _fsync_file(path: Path) -> None:
    with path.open("r+b") as handle:
        os.fsync(handle.fileno())


def _fsync_dir(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
