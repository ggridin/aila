"""Reflex P2 session-preemption controller and Hermes gateway platform shim.

Two layers:

* :class:`ReflexAdapterCore` — host-independent orchestration. Its :meth:`tick`
  runs the pure scheduler over the reflex ``EventStore`` + ``PriorityStateStore``
  and executes the decision through **injected** gateway callables
  (``interrupt_main`` / ``spawn_priority`` / ``summarize_main``). Fully testable
  with fakes; imports nothing from Hermes.

* ``build_reflex_adapter`` — a thin factory that, **inside the gateway process**,
  lazily subclasses ``BasePlatformAdapter`` and wires the core to the real
  ``deliver_wake`` / agent-interrupt primitives. Configured as AILA's home so the
  MAIN wake runs on this adapter under a deterministic, reflex-owned session.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from aila.reflex.models import Priority
from aila.reflex.output import OutputSink, route_output
from aila.reflex.priority import PrioritySeed, build_priority_seed
from aila.reflex.scheduler import Action, decide
from aila.reflex.sessions import DEFAULT_IDLE_TIMEOUT, PriorityStateStore
from aila.reflex.store import EventStore

# Injected gateway callables (host-independent signatures).
InterruptMain = Callable[[str], None]              # (reason) -> None
SpawnPriority = Callable[[PrioritySeed], str | None]  # (seed) -> priority session id
SummarizeMain = Callable[[], str]                  # () -> MAIN summary text


class ReflexAdapterCore:
    """Host-independent P2 preemption controller."""

    def __init__(
        self,
        event_store: EventStore,
        state_store: PriorityStateStore,
        *,
        interrupt_main: InterruptMain,
        spawn_priority: SpawnPriority,
        summarize_main: SummarizeMain | None = None,
        idle_timeout: timedelta = DEFAULT_IDLE_TIMEOUT,
    ) -> None:
        self._events = event_store
        self._state = state_store
        self._interrupt_main = interrupt_main
        self._spawn_priority = spawn_priority
        self._summarize_main = summarize_main or (lambda: "")
        self._idle_timeout = idle_timeout

    # -- main control loop ---------------------------------------------------

    def tick(
        self,
        *,
        now: datetime | None = None,
        priority_session_ended: bool = False,
        termination_signal: bool = False,
    ) -> Action:
        """Run one scheduling cycle; execute and return the chosen action."""

        now = _to_utc(now or datetime.now(UTC))
        state = self._state.load()
        pending = self._events.unseen({Priority.P2})
        action = decide(
            p2_pending=bool(pending),
            priority_active=state.priority_active,
            termination_signal=termination_signal,
            priority_session_ended=priority_session_ended,
            now=now,
            idle_deadline=state.idle_deadline,
        )

        if action is Action.preempt:
            self._preempt(pending[0], now=now)
        elif action is Action.terminate:
            self._state.end_priority()
        # queue / none: nothing to do (single-level; MAIN resumes on cron cadence)
        return action

    def _preempt(self, event, *, now: datetime) -> None:
        seed = build_priority_seed(event)
        summary = self._summarize_main()
        # Suspend MAIN at the turn boundary, then spawn the PRIORITY session.
        self._interrupt_main("reflex P2 preemption")
        session_id = self._spawn_priority(seed)
        self._state.begin_priority(
            event_id=event.event_id,
            session_id=session_id,
            main_summary=summary,
            idle_timeout=self._idle_timeout,
            now=now,
        )
        # Show-once: the P2 event has been actioned.
        self._events.mark_seen(event.event_id, when=now)

    # -- output routing ------------------------------------------------------

    def output_sink_for_active(self) -> OutputSink:
        """The modality sink for the currently active PRIORITY event."""

        state = self._state.load()
        if state.priority_event_id:
            expanded = self._events.resolve(state.priority_event_id)
            if expanded is not None:
                return route_output(expanded.worker, expanded.kind)
        return OutputSink.logs


# --------------------------------------------------------------------------- #
# In-gateway adapter shim (lazy Hermes import)
# --------------------------------------------------------------------------- #

REFLEX_PLATFORM = "reflex"
REFLEX_HOME_ENV = "REFLEX_HOME_CHANNEL"


def build_reflex_adapter(config: Any) -> Any:
    """Factory for the reflex ``BasePlatformAdapter`` (called inside the gateway).

    Lazily subclasses the Hermes base so the ``aila`` package stays importable
    without Hermes. The subclass captures the MAIN wake session on first inbound
    message, runs a watch loop that preempts MAIN on a P2 event (interrupt +
    spawn a separate PRIORITY session), and routes PRIORITY output by modality.
    All live-gateway calls are defensive: a glue failure logs and leaves the
    normal wake working.
    """

    import asyncio
    import dataclasses
    import logging

    from gateway.config import Platform  # type: ignore
    from gateway.platforms.base import BasePlatformAdapter, SendResult  # type: ignore

    logger = logging.getLogger("aila.reflex")

    class _ReflexAdapter(BasePlatformAdapter):
        supports_async_delivery = True

        def __init__(self, cfg: Any) -> None:
            super().__init__(cfg, Platform(REFLEX_PLATFORM))
            self._watch_task = None
            self._watch_running = False
            self._watch_interval = 1.0
            # MAIN wake session, captured from the first non-priority inbound.
            self._main_source: Any = None
            self._main_session_key: str = ""
            self._main_chat_id: str = ""
            self._events = None
            self._state = None

        # -- lifecycle -------------------------------------------------------

        async def connect(self, *, is_reconnect: bool = False) -> bool:
            from aila.reflex.sessions import PriorityStateStore
            from aila.reflex.store import EventStore

            body = _hermes_home() / "aila-body"
            self._events = EventStore(body / "reflex")
            self._state = PriorityStateStore(body / "reflex" / "sessions")
            self._watch_running = True
            self._watch_task = asyncio.ensure_future(self._watch_loop())
            logger.info("reflex platform connected; P2 watch loop started")
            return True

        async def disconnect(self) -> None:
            self._watch_running = False
            if self._watch_task is not None:
                self._watch_task.cancel()
                self._watch_task = None

        # -- capture MAIN source --------------------------------------------

        async def handle_message(self, event: Any) -> None:
            try:
                src = getattr(event, "source", None)
                chat_id = getattr(src, "chat_id", "") if src is not None else ""
                if src is not None and chat_id and not str(chat_id).endswith(":prio"):
                    if self._main_source is None:
                        self._main_source = src
                        self._main_chat_id = str(chat_id)
                        self._main_session_key = _session_key(src)
                        logger.info("reflex captured MAIN session key=%s", self._main_session_key)
            except Exception:  # pragma: no cover - never break dispatch
                logger.debug("reflex MAIN-source capture failed", exc_info=True)
            await super().handle_message(event)

        # -- watch loop ------------------------------------------------------

        async def _watch_loop(self) -> None:
            while self._watch_running:
                try:
                    await self._tick()
                except Exception:
                    logger.exception("reflex watch tick failed")
                await asyncio.sleep(self._watch_interval)

        async def _tick(self) -> None:
            from aila.reflex.models import Priority
            from aila.reflex.scheduler import Action, decide

            now = datetime.now(UTC)
            pending = self._events.unseen({Priority.P2})
            state = self._state.load()
            action = decide(
                p2_pending=bool(pending),
                priority_active=state.priority_active,
                now=now,
                idle_deadline=state.idle_deadline,
            )

            if action is Action.preempt and self._main_source is not None and pending:
                await self._preempt(pending[0], now=now)
            elif action is Action.terminate:
                self._state.end_priority()
                logger.info("reflex PRIORITY terminated (idle/timeout)")

        async def _preempt(self, event: Any, *, now: datetime) -> None:
            from gateway.wake import deliver_wake  # type: ignore

            from aila.reflex.priority import build_priority_seed

            seed = build_priority_seed(event)
            summary = await self._summarize_main()

            # 1) Suspend MAIN at its turn boundary.
            try:
                await self.interrupt_session_activity(self._main_session_key, self._main_chat_id)
            except Exception:
                logger.exception("reflex interrupt of MAIN failed")

            # 2) Spawn a separate PRIORITY session on this platform.
            prio_source = dataclasses.replace(self._main_source, chat_id=f"{self._main_chat_id}:prio")
            prio_key = _session_key(prio_source)
            try:
                await deliver_wake(self, text=seed.text, session_id=prio_key, source=prio_source)
            except Exception:
                logger.exception("reflex deliver_wake (PRIORITY) failed")

            self._state.begin_priority(
                event_id=event.event_id, session_id=prio_key, main_summary=summary, now=now
            )
            self._events.mark_seen(event.event_id, when=now)
            logger.info("reflex PREEMPT: MAIN suspended, PRIORITY started for event=%s", event.event_id)

        async def _summarize_main(self) -> str:
            # Best-effort: a short marker for the first cut. Full transcript
            # distillation via async_session_store is a refinement.
            return "MAIN was interrupted by a reflex P2 event; resume on the next wake."

        # -- output routing --------------------------------------------------

        async def send(self, chat_id: str, content: str, reply_to=None, metadata=None) -> Any:
            sink = OutputSink.logs
            try:
                state = self._state.load() if self._state else None
                if state and state.priority_event_id and self._events:
                    expanded = self._events.resolve(state.priority_event_id)
                    if expanded is not None:
                        sink = route_output(expanded.worker, expanded.kind)
            except Exception:
                logger.debug("reflex output routing failed", exc_info=True)
            logger.info("reflex send -> sink=%s (chat=%s)", sink.value, chat_id)
            return SendResult(success=True, message_id=None, raw_response={"reflex_sink": sink.value})

        async def get_chat_info(self, chat_id: str) -> dict:
            return {"name": "reflex", "type": "channel"}

    return _ReflexAdapter(config)


def _hermes_home() -> Path:
    return Path(os.environ.get("HERMES_HOME") or (Path.home() / ".hermes")).expanduser()


def _session_key(source: Any) -> str:
    """Compute the gateway session key for a source (best-effort)."""

    try:
        from gateway.session import build_session_key  # type: ignore

        return build_session_key(source)
    except Exception:
        chat_id = getattr(source, "chat_id", "")
        platform = getattr(source, "platform", "")
        return f"{platform}:{chat_id}"


async def standalone_reflex_sender(
    pconfig: Any,
    chat_id: str,
    message: str,
    *,
    thread_id: str | None = None,
    media_files: Any = None,
    force_document: bool = False,
) -> dict:
    """Deliver reflex output without a live gateway adapter (cron path).

    ``cron`` runs in a separate process from the gateway, so the in-process
    reflex adapter weakref is ``None``. This hook lets cron ``deliver=reflex``
    succeed: it routes the message to the active PRIORITY event's modality sink
    (falling back to logs) and persists it to the aila-body reflex log. The
    result dict follows the ``standalone_sender_fn`` contract.
    """

    import logging

    logger = logging.getLogger("aila.reflex")
    try:
        sink = OutputSink.logs
        body = _hermes_home() / "aila-body"
        try:
            state = PriorityStateStore(body / "reflex" / "sessions").load()
            if state.priority_event_id:
                expanded = EventStore(body / "reflex").resolve(state.priority_event_id)
                if expanded is not None:
                    sink = route_output(expanded.worker, expanded.kind)
        except Exception:
            logger.debug("reflex standalone routing failed", exc_info=True)

        log_dir = body / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        line = f"{datetime.now(UTC).isoformat()}\t{sink.value}\t{chat_id}\t{message}\n"
        with (log_dir / "reflex-output.log").open("a", encoding="utf-8") as handle:
            handle.write(line)
        logger.info("reflex standalone send -> sink=%s (chat=%s)", sink.value, chat_id)
        return {"success": True, "message_id": None, "reflex_sink": sink.value}
    except Exception as exc:  # pragma: no cover - defensive
        return {"error": f"reflex standalone send failed: {exc}"}


def _to_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)
