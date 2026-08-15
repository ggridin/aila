from __future__ import annotations

from datetime import datetime
from enum import Enum


class Action(str, Enum):
    """The scheduler's decision for the reflex adapter to execute."""

    none = "none"
    preempt = "preempt"      # suspend MAIN + start a PRIORITY session
    terminate = "terminate"  # end the active PRIORITY session (MAIN resumes on cadence)
    queue = "queue"          # a P2 arrived while PRIORITY active -> defer (single-level)


def decide(
    *,
    p2_pending: bool,
    priority_active: bool,
    termination_signal: bool = False,
    priority_session_ended: bool = False,
    now: datetime | None = None,
    idle_deadline: datetime | None = None,
) -> Action:
    """Pure, deterministic P2 scheduling decision (single-level, no nesting).

    - While a PRIORITY session is active: terminate it on a termination event, a
      normal session end, or idle timeout; otherwise defer any new P2 (``queue``).
    - Otherwise: a pending P2 triggers ``preempt`` (the adapter interrupts MAIN
      at a turn boundary and starts the PRIORITY session).
    """

    if priority_active:
        if termination_signal or priority_session_ended or _timed_out(now, idle_deadline):
            return Action.terminate
        if p2_pending:
            return Action.queue
        return Action.none

    if p2_pending:
        return Action.preempt
    return Action.none


def _timed_out(now: datetime | None, idle_deadline: datetime | None) -> bool:
    if now is None or idle_deadline is None:
        return False
    return now >= idle_deadline
