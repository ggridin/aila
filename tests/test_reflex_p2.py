from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from aila.reflex.models import Event, Priority
from aila.reflex.output import OutputSink, deliver, route_output
from aila.reflex.priority import build_priority_seed
from aila.reflex.scheduler import Action, decide
from aila.reflex.sessions import PriorityStateStore

TS = datetime(2026, 8, 5, 12, 0, 0, tzinfo=UTC)


def _event(priority: Priority, *, worker="mic", kind="speech.segment", detail=True) -> Event:
    return Event(
        event_id="evt-1", obs_id="obs-1", worker=worker, kind=kind, ts=TS,
        dedup_key="k", first_ts=TS, last_ts=TS,
        initial_priority=priority, effective_priority=priority,
        title="urgent thing", summary="the details", detail_available=detail,
    )


# --------------------------------------------------------------------------- #
# priority seed
# --------------------------------------------------------------------------- #

def test_build_priority_seed_from_p2() -> None:
    seed = build_priority_seed(_event(Priority.P2))
    assert seed.event_id == "evt-1"
    assert seed.worker == "mic" and seed.kind == "speech.segment"
    assert "PRIORITY INTERRUPT" in seed.text
    assert "urgent thing" in seed.text
    assert "reflex_expand" in seed.text  # detail_available


def test_priority_seed_rejects_non_p2() -> None:
    with pytest.raises(ValueError):
        build_priority_seed(_event(Priority.P3))


def test_priority_seed_omits_expand_when_no_detail() -> None:
    seed = build_priority_seed(_event(Priority.P2, detail=False))
    assert "reflex_expand" not in seed.text


# --------------------------------------------------------------------------- #
# scheduler state machine
# --------------------------------------------------------------------------- #

def test_preempt_when_p2_and_no_priority() -> None:
    assert decide(p2_pending=True, priority_active=False) is Action.preempt


def test_none_when_idle() -> None:
    assert decide(p2_pending=False, priority_active=False) is Action.none


def test_queue_new_p2_while_priority_active() -> None:
    assert decide(p2_pending=True, priority_active=True) is Action.queue


def test_terminate_on_keyword() -> None:
    assert decide(p2_pending=False, priority_active=True, termination_signal=True) is Action.terminate


def test_terminate_on_normal_session_end() -> None:
    assert decide(p2_pending=False, priority_active=True, priority_session_ended=True) is Action.terminate


def test_terminate_on_idle_timeout() -> None:
    assert decide(
        p2_pending=False, priority_active=True, now=TS + timedelta(minutes=10), idle_deadline=TS
    ) is Action.terminate


def test_no_timeout_before_deadline() -> None:
    assert decide(
        p2_pending=False, priority_active=True, now=TS, idle_deadline=TS + timedelta(minutes=5)
    ) is Action.none


# --------------------------------------------------------------------------- #
# priority state store
# --------------------------------------------------------------------------- #

def test_begin_touch_end_priority(tmp_path: Path) -> None:
    store = PriorityStateStore(tmp_path)
    assert store.load().priority_active is False

    st = store.begin_priority(event_id="evt-1", main_summary="did X, Y", now=TS, idle_timeout=timedelta(minutes=5))
    assert st.priority_active and st.main_suspended
    assert st.priority_event_id == "evt-1"
    assert st.idle_deadline == TS + timedelta(minutes=5)
    assert store.load().main_summary == "did X, Y"  # persisted

    touched = store.touch(now=TS + timedelta(minutes=2), idle_timeout=timedelta(minutes=5))
    assert touched.idle_deadline == TS + timedelta(minutes=7)

    ended = store.end_priority()
    assert ended.priority_active is False
    assert store.load().priority_active is False


# --------------------------------------------------------------------------- #
# modality output routing
# --------------------------------------------------------------------------- #

def test_route_output_by_modality() -> None:
    assert route_output("mic", "speech.segment") is OutputSink.speaker
    assert route_output("camera", "scene.caption") is OutputSink.display
    assert route_output("filesystem", "file.changed") is OutputSink.logs
    assert route_output("health", "cpu") is OutputSink.logs  # unknown -> logs


def test_deliver_dispatches_and_falls_back() -> None:
    calls: dict[str, str] = {}
    deliver(OutputSink.speaker, "hi", speaker=lambda c: calls.__setitem__("speaker", c))
    assert calls == {"speaker": "hi"}

    # Missing handler falls back to log.
    logged: list[str] = []
    deliver(OutputSink.display, "yo", log=logged.append)
    assert logged == ["yo"]


# --------------------------------------------------------------------------- #
# adapter core (P2 preemption orchestration, fake gateway)
# --------------------------------------------------------------------------- #

def _seed_p2_event(store):
    from aila.contracts import Observation
    from aila.reflex.config import default_ranking_rules
    from aila.reflex.ingest import IngestReducer

    obs = Observation(
        obs_id="obs-p2", worker="mic", kind="speech.segment", ts=TS,
        payload={"text": "stop the build now", "lang": "en", "confidence": 0.95, "start_ms": 0, "end_ms": 100},
    )
    ev = IngestReducer(store, default_ranking_rules()).reduce(obs)
    assert ev is not None and ev.effective_priority is Priority.P2
    return ev


def _core(tmp_path: Path):
    from aila.reflex.hermes_platform import ReflexAdapterCore
    from aila.reflex.store import EventStore

    store = EventStore(tmp_path / "reflex")
    state = PriorityStateStore(tmp_path / "sessions")
    calls: dict[str, object] = {"interrupt": [], "spawn": []}

    def interrupt_main(reason: str) -> None:
        calls["interrupt"].append(reason)  # type: ignore[attr-defined]

    def spawn_priority(seed) -> str:
        calls["spawn"].append(seed.event_id)  # type: ignore[attr-defined]
        return "prio-sess-1"

    core = ReflexAdapterCore(
        store, state,
        interrupt_main=interrupt_main,
        spawn_priority=spawn_priority,
        summarize_main=lambda: "MAIN did X and Y",
    )
    return core, store, state, calls


def test_core_preempts_on_p2(tmp_path: Path) -> None:
    core, store, state, calls = _core(tmp_path)
    _seed_p2_event(store)

    action = core.tick(now=TS)

    assert action is Action.preempt
    assert calls["interrupt"] == ["reflex P2 preemption"]
    assert len(calls["spawn"]) == 1  # PRIORITY spawned once
    st = state.load()
    assert st.priority_active and st.main_suspended
    assert st.main_summary == "MAIN did X and Y"
    assert store.unseen() == []  # event marked seen (show-once)


def test_core_queues_new_p2_while_active(tmp_path: Path) -> None:
    core, store, state, calls = _core(tmp_path)
    _seed_p2_event(store)
    core.tick(now=TS)  # preempt
    # A second P2 arrives while PRIORITY active.
    obs_store = store
    from aila.contracts import Observation
    from aila.reflex.config import default_ranking_rules
    from aila.reflex.ingest import IngestReducer
    IngestReducer(obs_store, default_ranking_rules()).reduce(
        Observation(obs_id="obs-p2b", worker="mic", kind="speech.segment", ts=TS,
                    payload={"text": "another urgent", "lang": "en", "confidence": 0.9, "start_ms": 0, "end_ms": 50})
    )
    action = core.tick(now=TS + timedelta(seconds=1))
    assert action is Action.queue
    assert len(calls["spawn"]) == 1  # no second spawn


def test_core_terminates_on_session_end(tmp_path: Path) -> None:
    core, store, state, _ = _core(tmp_path)
    _seed_p2_event(store)
    core.tick(now=TS)
    action = core.tick(now=TS + timedelta(seconds=1), priority_session_ended=True)
    assert action is Action.terminate
    assert state.load().priority_active is False


def test_core_output_sink_matches_modality(tmp_path: Path) -> None:
    core, store, state, _ = _core(tmp_path)
    _seed_p2_event(store)
    core.tick(now=TS)
    assert core.output_sink_for_active() is OutputSink.speaker  # mic -> speaker


def test_standalone_sender_routes_and_persists(tmp_path: Path, monkeypatch) -> None:
    import asyncio

    from aila.reflex.hermes_platform import standalone_reflex_sender
    from aila.reflex.store import EventStore

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    body = tmp_path / "aila-body"
    store = EventStore(body / "reflex")
    ev = _seed_p2_event(store)
    PriorityStateStore(body / "reflex" / "sessions").begin_priority(
        event_id=ev.event_id, session_id="prio-1", now=TS
    )

    result = asyncio.run(standalone_reflex_sender(None, "reflex-main", "wake output"))

    assert result["success"] is True
    assert result["reflex_sink"] == OutputSink.speaker.value  # mic -> speaker
    log_text = (body / "logs" / "reflex-output.log").read_text(encoding="utf-8")
    assert "wake output" in log_text
    assert "speaker" in log_text


