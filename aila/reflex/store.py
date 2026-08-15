from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from aila.contracts import Observation
from aila.reflex.models import Event, ExpandedEvent, Priority

_SAFE_KEY_RE = re.compile(r"[^A-Za-z0-9_.-]+")
# Filenames are limited to 255 bytes on common filesystems, and writes go
# through a ``.<name>.<uuid4-hex>.tmp`` sidecar (38 extra chars) plus an
# optional ``.json`` suffix. Cap the sanitised key well below that budget so
# deep roots stay inside the Windows MAX_PATH limit as well.
_MAX_KEY_LEN = 96
_KEY_DIGEST_LEN = 16


@dataclass(frozen=True)
class EventRetention:
    """Retention policy for the reflex event store (seen events only)."""

    max_events: int | None = 5000
    max_age: timedelta | None = timedelta(days=7)

    def __post_init__(self) -> None:
        if self.max_events is not None and self.max_events < 0:
            raise ValueError("max_events must be non-negative")
        if self.max_age is not None and self.max_age < timedelta(0):
            raise ValueError("max_age must be non-negative")


class EventStore:
    """Durable file-backed store of reflex :class:`Event`s.

    Layout under ``root``::

        events/<event_id>.json     # {"event": {...}, "observation": {...}}
        dedup/<safe_dedup_key>     # text pointer -> event_id

    Each record snapshots the full originating observation so ``resolve`` keeps
    working after the source queue archive is pruned.
    """

    def __init__(self, root: Path, *, retention: EventRetention | None = None) -> None:
        self.root = Path(root)
        self.events_dir = self.root / "events"
        self.dedup_dir = self.root / "dedup"
        self.retention = retention if retention is not None else EventRetention()
        self._ensure_dirs()

    # -- writes --------------------------------------------------------------

    def find_by_dedup(self, dedup_key: str) -> Event | None:
        pointer = self.dedup_dir / _safe_key(dedup_key)
        if not pointer.is_file():
            return None
        event_id = pointer.read_text(encoding="utf-8").strip()
        record = self._read_record(event_id)
        return record[0] if record else None

    def save(self, event: Event, observation: Observation) -> Path:
        """Persist (or overwrite) an event record and its dedup pointer."""

        self._ensure_dirs()
        payload = {
            "event": json.loads(event.model_dump_json()),
            "observation": json.loads(observation.model_dump_json()),
        }
        target = self.events_dir / f"{_safe_key(event.event_id)}.json"
        _atomic_write(target, json.dumps(payload, ensure_ascii=False))
        _atomic_write(
            self.dedup_dir / _safe_key(event.dedup_key),
            event.event_id,
        )
        return target

    def mark_seen(self, event_id: str, *, when: datetime | None = None) -> bool:
        """Atomically flag an event as seen. Returns False if unknown."""

        record = self._read_record(event_id)
        if record is None:
            return False
        event, observation = record
        if event.seen:
            return True
        seen_event = event.model_copy(
            update={"seen": True, "seen_ts": _to_utc(when or datetime.now(UTC))}
        )
        self.save(seen_event, observation)
        return True

    # -- reads ---------------------------------------------------------------

    def unseen(self, priorities: Iterable[Priority] | None = None) -> list[Event]:
        """Return unseen events, optionally filtered to a set of priorities.

        Ordered by effective priority (most urgent first) then oldest-first.
        """

        wanted = set(priorities) if priorities is not None else None
        events: list[Event] = []
        for event, _ in self._iter_records():
            if event.seen:
                continue
            if wanted is not None and event.effective_priority not in wanted:
                continue
            events.append(event)
        events.sort(key=lambda e: (int(e.effective_priority), e.ts))
        return events

    def resolve(self, event_id: str) -> ExpandedEvent | None:
        record = self._read_record(event_id)
        if record is None:
            return None
        event, observation = record
        return ExpandedEvent(
            event_id=event.event_id,
            obs_id=event.obs_id,
            worker=event.worker,
            kind=event.kind,
            ts=event.ts,
            priority=event.effective_priority,
            payload=json.loads(observation.model_dump_json())["payload"],
            media_ref=observation.media_ref,
        )

    # -- maintenance ---------------------------------------------------------

    def enforce_retention(self, *, now: datetime | None = None) -> None:
        self._ensure_dirs()
        now_utc = _to_utc(now or datetime.now(UTC))
        seen: list[tuple[datetime, str, str]] = []  # (seen_ts, event_id, dedup_key)
        for event, _ in self._iter_records():
            if not event.seen or event.seen_ts is None:
                continue
            seen_ts = _to_utc(event.seen_ts)
            if (
                self.retention.max_age is not None
                and now_utc - seen_ts > self.retention.max_age
            ):
                self._delete(event)
                continue
            seen.append((seen_ts, event.event_id, event.dedup_key))

        if self.retention.max_events is not None and len(seen) > self.retention.max_events:
            seen.sort(key=lambda item: item[0])  # oldest first
            for _, event_id, dedup_key in seen[: len(seen) - self.retention.max_events]:
                self._delete_by_ids(event_id, dedup_key)

    # -- internals -----------------------------------------------------------

    def _ensure_dirs(self) -> None:
        self.events_dir.mkdir(parents=True, exist_ok=True)
        self.dedup_dir.mkdir(parents=True, exist_ok=True)

    def _iter_records(self) -> Iterable[tuple[Event, Observation]]:
        for path in sorted(self.events_dir.glob("*.json")):
            record = self._read_record_path(path)
            if record is not None:
                yield record

    def _read_record(self, event_id: str) -> tuple[Event, Observation] | None:
        return self._read_record_path(self.events_dir / f"{_safe_key(event_id)}.json")

    def _read_record_path(self, path: Path) -> tuple[Event, Observation] | None:
        if not path.is_file():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            event = Event.model_validate(data["event"])
            observation = Observation.model_validate(data["observation"])
        except (OSError, ValueError, KeyError):
            return None
        return event, observation

    def _delete(self, event: Event) -> None:
        self._delete_by_ids(event.event_id, event.dedup_key)

    def _delete_by_ids(self, event_id: str, dedup_key: str) -> None:
        (self.events_dir / f"{_safe_key(event_id)}.json").unlink(missing_ok=True)
        pointer = self.dedup_dir / _safe_key(dedup_key)
        # Only remove the pointer if it still targets this event.
        if pointer.is_file() and pointer.read_text(encoding="utf-8").strip() == event_id:
            pointer.unlink(missing_ok=True)


def _safe_key(key: str) -> str:
    safe = _SAFE_KEY_RE.sub("_", key).strip("._")
    if not safe:
        raise ValueError("key does not contain any filesystem-safe characters")
    if len(safe) > _MAX_KEY_LEN:
        digest = hashlib.blake2b(
            key.encode("utf-8"), digest_size=_KEY_DIGEST_LEN // 2
        ).hexdigest()
        head = safe[: _MAX_KEY_LEN - _KEY_DIGEST_LEN - 1].rstrip("._")
        safe = f"{head}_{digest}"
    return safe


def _atomic_write(target: Path, text: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.parent / f".{target.name}.{uuid.uuid4().hex}.tmp"
    try:
        temp.write_text(text, encoding="utf-8")
        with temp.open("r+b") as handle:
            os.fsync(handle.fileno())
        temp.replace(target)
    finally:
        if temp.exists():
            temp.unlink()


def _to_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)
