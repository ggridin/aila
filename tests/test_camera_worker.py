from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from aila.contracts import Command
from aila.device_services import audio_input_config, camera_input_config
from aila.queue import ObservationQueue
from aila.workers.camera import CameraFrame, CameraWorker
from aila.workers.camera_local import LocalCameraFrameSource, build_camera_frame_source
from aila.workers.config import WorkerConfig


@dataclass
class FakeCameraFrameSource:
    frames: tuple[CameraFrame, ...]
    snapshot: CameraFrame

    def poll_frames(self) -> tuple[CameraFrame, ...]:
        return self.frames

    def capture_snapshot(self) -> CameraFrame:
        return self.snapshot


def test_camera_worker_emits_periodic_caption_and_motion_observations(tmp_path: Path) -> None:
    queue = ObservationQueue(tmp_path / "queue")
    worker = CameraWorker(
        _camera_config(),
        camera_input_config(device="/dev/video0", capture={"fps": 1}),
        FakeCameraFrameSource(
            frames=(
                _frame("frame-1", "desk with a laptop", ("laptop", "desk"), 0.1, 0),
                _frame("frame-2", "person entered the room", ("person",), 0.8, 1000),
            ),
            snapshot=_frame("snapshot-1", "current desk", ("desk",), 0.0, 2000),
        ),
        queue,
    )

    observations = worker.poll_once()

    assert worker.device_service.service == "camera-input"
    assert worker.device_service.capture == {"fps": 1}
    assert [observation.obs_id for observation in observations] == [
        "frame-1-caption",
        "frame-1-motion",
        "frame-2-caption",
        "frame-2-motion",
    ]
    assert [observation.kind for observation in observations] == [
        "scene.caption",
        "scene.motion",
        "scene.caption",
        "scene.motion",
    ]
    assert observations[0].payload.caption == "desk with a laptop"
    assert observations[0].payload.labels == ["laptop", "desk"]
    assert observations[1].payload.level == 0.1
    assert observations[3].payload.region == "frame"
    assert [item.observation.obs_id for item in queue.drain()] == [
        "frame-1-caption",
        "frame-1-motion",
        "frame-2-caption",
        "frame-2-motion",
    ]


def test_local_camera_source_builds_from_local_model_config() -> None:
    config = WorkerConfig.model_validate(
        {
            "worker": "camera",
            "role": "sensor",
            "device_service": "camera-input",
            "backend": {"kind": "model", "placement": "local", "model": "camera-vlm"},
            "sampling": {
                "source": {"device": "/dev/video0"},
                "pipeline": {"motion": {"enabled": False}},
                "vision": {"endpoint": "http://127.0.0.1:8081/v1", "model": "Qwen2.5-VL-3B"},
            },
            "emits": ["scene.caption", "scene.motion"],
            "verbs": ["snapshot"],
        }
    )

    source = build_camera_frame_source(config, camera_input_config(device="/dev/video0"))

    assert isinstance(source, LocalCameraFrameSource)
    assert source.status.vision_endpoint == "http://127.0.0.1:8081/v1"
    assert source.vision.model == "Qwen2.5-VL-3B"


def test_camera_worker_snapshot_pull_verb_returns_current_caption(tmp_path: Path) -> None:
    source = FakeCameraFrameSource(
        frames=(),
        snapshot=_frame("snapshot-1", "whiteboard notes", ("whiteboard", "notes"), 0.0, 0),
    )
    worker = CameraWorker(
        _camera_config(),
        camera_input_config(device="/dev/video0"),
        source,
        ObservationQueue(tmp_path / "queue"),
    )
    command = Command(id="cmd-1", worker="camera", verb="snapshot", args={})

    result = worker.handle_command(command)

    assert result.ok is True
    assert result.data is not None
    assert result.data.scene_caption.caption == "whiteboard notes"
    assert result.data.scene_caption.labels == ["whiteboard", "notes"]


def test_camera_worker_persists_derived_captions_and_labels_without_raw_frames(
    tmp_path: Path,
) -> None:
    queue = ObservationQueue(tmp_path / "queue")
    worker = CameraWorker(
        _camera_config(),
        camera_input_config(device="/dev/video0"),
        FakeCameraFrameSource(
            frames=(
                CameraFrame(
                    obs_id="private-frame",
                    ts=_timestamp(0),
                    caption="derived caption only",
                    labels=("plant", "window"),
                    motion_level=0.2,
                    motion_region="frame",
                    raw_frame=b"RAW_FRAME_BYTES_MUST_NOT_BE_PERSISTED",
                ),
            ),
            snapshot=_frame("snapshot-1", "current scene", ("scene",), 0.0, 1000),
        ),
        queue,
    )

    observations = worker.poll_once()
    persisted = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted((tmp_path / "queue" / "pending").glob("*.json"))
    )

    assert observations[0].payload.caption == "derived caption only"
    assert observations[0].payload.labels == ["plant", "window"]
    assert all(observation.media_ref is None for observation in observations)
    assert "derived caption only" in persisted
    assert "plant" in persisted
    assert "raw_frame" not in persisted
    assert "RAW_FRAME_BYTES_MUST_NOT_BE_PERSISTED" not in persisted


def test_camera_worker_requires_camera_input_device_service(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="camera-input"):
        CameraWorker(
            _camera_config(),
            audio_input_config(device="default"),
            FakeCameraFrameSource(
                frames=(),
                snapshot=_frame("snapshot-1", "current scene", ("scene",), 0.0, 0),
            ),
            ObservationQueue(tmp_path / "queue"),
        )


def _camera_config() -> WorkerConfig:
    return WorkerConfig.model_validate(
        {
            "worker": "camera",
            "role": "sensor",
            "device_service": "camera-input",
            "backend": {"kind": "deterministic", "placement": "local"},
            "sampling": {"interval_seconds": 5},
            "emits": ["scene.caption", "scene.motion"],
            "verbs": ["snapshot"],
        }
    )


def _frame(
    obs_id: str,
    caption: str,
    labels: tuple[str, ...],
    motion_level: float,
    offset_ms: int,
) -> CameraFrame:
    return CameraFrame(
        obs_id=obs_id,
        ts=_timestamp(offset_ms),
        caption=caption,
        labels=labels,
        motion_level=motion_level,
        motion_region="frame",
        raw_frame=f"raw-frame-{obs_id}".encode("utf-8"),
    )


def _timestamp(offset_ms: int) -> datetime:
    return datetime(2026, 7, 13, 12, 0, tzinfo=UTC) + timedelta(milliseconds=offset_ms)
