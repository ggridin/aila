from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from aila.contracts import Observation
from aila.queue import ObservationQueue
from aila.device_services import camera_input_config
from aila.reflex.config import (
    default_ranking_rules,
    parse_dedup_config,
    parse_ingest_filter,
    parse_ranking_rules,
)
from aila.reflex.ingest import DedupConfig, IngestFilterConfig, IngestReducer, dedup_key_for
from aila.reflex.store import EventStore
from aila.workers.camera import CameraFrame, CameraWorker
from aila.workers.config import WorkerConfig

TS = datetime(2026, 7, 13, 12, 0, 0, tzinfo=UTC)


def _caption(obs_id: str, ts: datetime, *, caption: str, labels: tuple[str, ...]) -> Observation:
    return Observation(
        obs_id=obs_id,
        worker="camera",
        kind="scene.caption",
        ts=ts,
        payload={"caption": caption, "labels": list(labels), "boxes": []},
    )


def _motion(obs_id: str, ts: datetime, *, level: float, region: str = "frame") -> Observation:
    return Observation(
        obs_id=obs_id,
        worker="camera",
        kind="scene.motion",
        ts=ts,
        payload={"level": level, "region": region},
    )


# --------------------------------------------------------------------------- #
# dedup_key_for
# --------------------------------------------------------------------------- #

def test_caption_dedup_key_defaults_to_caption_text() -> None:
    a = _caption("a", TS, caption="objects: sink; motion: 0.08", labels=("sink",))
    b = _caption("b", TS, caption="objects: sink; motion: 0.09", labels=("sink",))
    # Default mode keys on caption text -> jittering motion makes distinct keys.
    assert dedup_key_for(a) != dedup_key_for(b)


def test_caption_dedup_key_labels_mode_ignores_caption_jitter() -> None:
    cfg = DedupConfig(caption_key="labels")
    a = _caption("a", TS, caption="objects: sink; motion: 0.08", labels=("sink", "chair"))
    b = _caption("b", TS, caption="objects: sink; motion: 0.42", labels=("chair", "sink"))
    # Same label set (order-insensitive) -> identical key despite caption text.
    assert dedup_key_for(a, cfg) == dedup_key_for(b, cfg)
    assert dedup_key_for(a, cfg) == "camera:scene.caption:chair,sink"


# --------------------------------------------------------------------------- #
# filter + reducer merge
# --------------------------------------------------------------------------- #

def test_min_motion_level_filter_drops_low_motion(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "reflex")
    reducer = IngestReducer(
        store,
        default_ranking_rules(),
        filter_config=IngestFilterConfig(min_motion_level=0.2),
    )
    assert reducer.reduce(_motion("m1", TS, level=0.05)) is None  # below threshold -> dropped
    assert reducer.reduce(_motion("m2", TS, level=0.5)) is not None  # above threshold -> kept


def test_labels_mode_merges_caption_repeats(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "reflex")
    reducer = IngestReducer(
        store,
        default_ranking_rules(),
        dedup_config=DedupConfig(caption_key="labels"),
    )
    first = reducer.reduce(_caption("c1", TS, caption="objects: sink; motion: 0.08", labels=("sink",)))
    second = reducer.reduce(
        _caption("c2", TS + timedelta(seconds=5), caption="objects: sink; motion: 0.11", labels=("sink",))
    )
    assert first is not None and second is not None
    assert second.event_id == first.event_id  # merged into the same unseen event
    assert second.count == 2
    assert second.first_ts == TS
    assert second.last_ts == TS + timedelta(seconds=5)


# --------------------------------------------------------------------------- #
# config parsing
# --------------------------------------------------------------------------- #

def test_parse_sections_from_combined_yaml_document() -> None:
    data = {
        "default_priority": "P5",
        "rules": [{"worker": "mic", "kind": "speech.segment", "priority": "P2"}],
        "filter": {"min_motion_level": 0.15},
        "dedup": {"caption_key": "labels"},
    }
    rules = parse_ranking_rules(data)  # must ignore filter/dedup without raising
    assert rules.rules[0].worker == "mic"
    assert parse_ingest_filter(data).min_motion_level == 0.15
    assert parse_dedup_config(data).caption_key == "labels"


# --------------------------------------------------------------------------- #
# camera worker emit-on-change + keepalive
# --------------------------------------------------------------------------- #

@dataclass
class _FakeSource:
    frames: tuple[CameraFrame, ...]

    def poll_frames(self) -> tuple[CameraFrame, ...]:
        return self.frames

    def capture_snapshot(self) -> CameraFrame:
        return self.frames[0]


def _frame(obs_id: str, labels: tuple[str, ...], motion: float) -> CameraFrame:
    caption = ("objects: " + ", ".join(labels) + f"; motion: {motion:.2f}") if labels else f"motion: {motion:.2f}"
    return CameraFrame(
        obs_id=obs_id,
        ts=TS,
        caption=caption,
        labels=labels,
        motion_level=motion,
        motion_region="frame",
    )


def _camera_config(change_detection: dict | None = None) -> WorkerConfig:
    sampling: dict = {"frame_interval_seconds": 5}
    if change_detection is not None:
        sampling["change_detection"] = change_detection
    return WorkerConfig.model_validate(
        {
            "worker": "camera",
            "role": "sensor",
            "device_service": "camera-input",
            "backend": {"kind": "deterministic", "placement": "local"},
            "sampling": sampling,
            "emits": ["scene.caption", "scene.motion"],
            "verbs": ["snapshot"],
        }
    )


def _build_worker(source: _FakeSource, tmp_path: Path, change_detection: dict | None) -> CameraWorker:
    return CameraWorker(
        _camera_config(change_detection),
        camera_input_config(device="/dev/video0", capture={"fps": 1}),
        source,
        ObservationQueue(tmp_path / "queue"),
    )


def test_emit_on_change_suppresses_identical_frames(tmp_path: Path) -> None:
    # Two near-identical frames (same labels, motion within one bucket).
    source = _FakeSource(frames=(_frame("f1", ("sink",), 0.08), _frame("f2", ("sink",), 0.09)))
    worker = _build_worker(source, tmp_path, {"emit_on_change": True, "keepalive_seconds": 999, "motion_bucket": 0.05})
    obs = worker.poll_once()
    # Only the first frame emits (caption + motion); the second is suppressed.
    assert [o.obs_id for o in obs] == ["f1-caption", "f1-motion"]


def test_emit_on_change_emits_when_labels_change(tmp_path: Path) -> None:
    source = _FakeSource(frames=(_frame("f1", ("sink",), 0.08), _frame("f2", ("sink", "person"), 0.08)))
    worker = _build_worker(source, tmp_path, {"emit_on_change": True, "keepalive_seconds": 999, "motion_bucket": 0.05})
    obs = worker.poll_once()
    assert [o.obs_id for o in obs] == ["f1-caption", "f1-motion", "f2-caption", "f2-motion"]


def test_disabled_change_detection_emits_every_frame(tmp_path: Path) -> None:
    source = _FakeSource(frames=(_frame("f1", ("sink",), 0.08), _frame("f2", ("sink",), 0.08)))
    worker = _build_worker(source, tmp_path, None)  # default: emit_on_change off
    obs = worker.poll_once()
    assert [o.obs_id for o in obs] == ["f1-caption", "f1-motion", "f2-caption", "f2-motion"]
