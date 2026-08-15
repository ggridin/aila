from __future__ import annotations

import json
import os
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

from aila.reflex.models import StrictModel

DEFAULT_IDLE_TIMEOUT = timedelta(minutes=5)


class PriorityState(StrictModel):
    """Single-level PRIORITY/suspend state.

    Only one PRIORITY session may be active at a time (no nesting). While it is
    active, MAIN is considered suspended (a summary is persisted to disk) and
    resumes on the next cron cadence once PRIORITY ends.
    """

    priority_active: bool = False
    priority_event_id: str | None = None
    priority_session_id: str | None = None
    started_ts: datetime | None = None
    # Idle timeout: PRIORITY terminates if now >= idle_deadline (no new activity).
    idle_deadline: datetime | None = None
    # Optional spoken/keyword termination trigger.
    termination_keyword: str | None = None
    # MAIN suspend record (summaries-only).
    main_suspended: bool = False
    main_summary: str | None = None


class PriorityStateStore:
    """Durable, atomically-written holder of the single :class:`PriorityState`."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.path = self.root / "state.json"
        self.root.mkdir(parents=True, exist_ok=True)

    def load(self) -> PriorityState:
        if not self.path.is_file():
            return PriorityState()
        try:
            return PriorityState.model_validate_json(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return PriorityState()

    def save(self, state: PriorityState) -> None:
        _atomic_write(self.path, state.model_dump_json())

    # -- transitions ---------------------------------------------------------

    def begin_priority(
        self,
        *,
        event_id: str,
        session_id: str | None = None,
        main_summary: str | None = None,
        termination_keyword: str | None = None,
        idle_timeout: timedelta = DEFAULT_IDLE_TIMEOUT,
        now: datetime | None = None,
    ) -> PriorityState:
        now = _to_utc(now or datetime.now(UTC))
        state = PriorityState(
            priority_active=True,
            priority_event_id=event_id,
            priority_session_id=session_id,
            started_ts=now,
            idle_deadline=now + idle_timeout,
            termination_keyword=termination_keyword,
            main_suspended=True,
            main_summary=main_summary,
        )
        self.save(state)
        return state

    def touch(self, *, idle_timeout: timedelta = DEFAULT_IDLE_TIMEOUT, now: datetime | None = None) -> PriorityState:
        """Extend the idle deadline on PRIORITY activity."""
        state = self.load()
        if state.priority_active:
            now = _to_utc(now or datetime.now(UTC))
            state = state.model_copy(update={"idle_deadline": now + idle_timeout})
            self.save(state)
        return state

    def end_priority(self) -> PriorityState:
        state = PriorityState()
        self.save(state)
        return state


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
