from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol

from aila.contracts import Command, Error, Result, SceneBox
from aila.device_services import DeviceServiceConfig
from aila.queue import ObservationQueue
from aila.workers.backends import BackendError, BackendObservation
from aila.workers.base import SensorWorker
from aila.workers.config import WorkerConfig


@dataclass(frozen=True)
class CameraFrame:
    caption: str
    labels: tuple[str, ...] = ()
    boxes: tuple[SceneBox | dict[str, Any], ...] = ()
    motion_level: float = 0.0
    motion_region: str = "frame"
    obs_id: str | None = None
    ts: datetime | None = None
    raw_frame: bytes | None = None

    def __post_init__(self) -> None:
        if not self.caption.strip():
            raise ValueError("camera frame caption must not be empty")
        if any(not label for label in self.labels):
            raise ValueError("camera frame labels must not be empty")
        if not 0.0 <= self.motion_level <= 1.0:
            raise ValueError("camera frame motion_level must be between 0 and 1")
        if not self.motion_region:
            raise ValueError("camera frame motion_region must not be empty")


class CameraFrameSource(Protocol):
    def poll_frames(self) -> tuple[CameraFrame, ...]:
        raise NotImplementedError

    def capture_snapshot(self) -> CameraFrame:
        raise NotImplementedError


@dataclass(frozen=True)
class ChangeDetectionConfig:
    """Worker-side suppression of near-identical consecutive frames.

    When ``emit_on_change`` is true, a frame is only emitted if its signature
    (sorted object labels + a coarsely bucketed motion level) differs from the
    last emitted frame, OR if ``keepalive_seconds`` has elapsed since the last
    emission. The keepalive guarantees liveness for a motionless scene, while
    downstream dedup records duration via the merged event's ``count`` and
    ``first_ts``/``last_ts``.
    """

    emit_on_change: bool = False
    keepalive_seconds: float = 60.0
    motion_bucket: float = 0.05

    @classmethod
    def from_sampling(cls, sampling: dict[str, Any]) -> "ChangeDetectionConfig":
        section = sampling.get("change_detection") if isinstance(sampling, dict) else None
        if not section:
            return cls()
        if not isinstance(section, dict):
            raise ValueError("camera sampling.change_detection must be a mapping")
        defaults = cls()
        keepalive = float(section.get("keepalive_seconds", defaults.keepalive_seconds))
        motion_bucket = float(section.get("motion_bucket", defaults.motion_bucket))
        if keepalive < 0:
            raise ValueError("change_detection.keepalive_seconds must be >= 0")
        if motion_bucket < 0:
            raise ValueError("change_detection.motion_bucket must be >= 0")
        return cls(
            emit_on_change=bool(section.get("emit_on_change", defaults.emit_on_change)),
            keepalive_seconds=keepalive,
            motion_bucket=motion_bucket,
        )


def _frame_signature(frame: CameraFrame, motion_bucket: float) -> tuple[tuple[str, ...], int]:
    labels = tuple(sorted({str(label).strip().lower() for label in frame.labels if str(label).strip()}))
    if motion_bucket > 0:
        motion_key = int(frame.motion_level / motion_bucket)
    else:
        motion_key = int(round(frame.motion_level * 1000))
    return labels, motion_key


class CameraWorker(SensorWorker):
    def __init__(
        self,
        config: WorkerConfig,
        device_service: DeviceServiceConfig,
        source: CameraFrameSource,
        queue: ObservationQueue,
    ) -> None:
        if config.worker != "camera":
            raise ValueError(f"camera worker cannot use config for {config.worker}")
        if device_service.service != "camera-input":
            raise ValueError(f"camera worker requires camera-input, got {device_service.service}")
        if device_service.consumer != "camera":
            raise ValueError("camera-input device service must be assigned to camera")
        if device_service.kind != "camera":
            raise ValueError("camera-input device service must use camera kind")

        self.device_service = device_service
        change_config = ChangeDetectionConfig.from_sampling(config.sampling)
        super().__init__(config, _CameraBackend(source, change_config), queue)

    def handle_command(self, command: Command) -> Result:
        if command.worker != self.worker:
            return Result(
                id=command.id,
                ok=False,
                error=Error(
                    code="WRONG_WORKER",
                    message=f"command targets {command.worker}, not {self.worker}",
                ),
            )
        if command.verb not in self.config.verbs:
            return Result(
                id=command.id,
                ok=False,
                error=Error(
                    code="UNSUPPORTED_VERB",
                    message=f"worker {self.worker} does not support verb {command.verb}",
                ),
            )

        try:
            data = self.backend.handle_command(command)
        except BackendError as exc:
            return Result(
                id=command.id,
                ok=False,
                error=Error(code="BACKEND_ERROR", message=str(exc), retryable=True),
            )
        return Result(id=command.id, ok=True, data=data)


class _CameraBackend:
    def __init__(
        self,
        source: CameraFrameSource,
        change_config: ChangeDetectionConfig | None = None,
        *,
        time_source: Callable[[], float] = time.monotonic,
    ) -> None:
        self._source = source
        self._change = change_config or ChangeDetectionConfig()
        self._time = time_source
        self._last_signature: tuple[tuple[str, ...], int] | None = None
        self._last_emit_at: float | None = None

    def poll(self) -> tuple[BackendObservation, ...]:
        observations: list[BackendObservation] = []
        for frame in self._source.poll_frames():
            if not isinstance(frame, CameraFrame):
                raise TypeError("camera source must yield CameraFrame instances")
            if not self._should_emit(frame):
                continue
            observations.append(
                BackendObservation(
                    kind="scene.caption",
                    obs_id=_obs_id(frame, "caption"),
                    ts=frame.ts,
                    payload=_caption_payload(frame),
                )
            )
            observations.append(
                BackendObservation(
                    kind="scene.motion",
                    obs_id=_obs_id(frame, "motion"),
                    ts=frame.ts,
                    payload={"level": frame.motion_level, "region": frame.motion_region},
                )
            )
        return tuple(observations)

    def _should_emit(self, frame: CameraFrame) -> bool:
        if not self._change.emit_on_change:
            return True
        signature = _frame_signature(frame, self._change.motion_bucket)
        now = self._time()
        changed = signature != self._last_signature
        keepalive_due = (
            self._last_emit_at is None
            or (now - self._last_emit_at) >= self._change.keepalive_seconds
        )
        if changed or keepalive_due:
            self._last_signature = signature
            self._last_emit_at = now
            return True
        return False

    def handle_command(self, command: Command) -> object:
        if command.verb != "snapshot":
            raise BackendError(f"camera worker does not support verb {command.verb}")
        frame = self._source.capture_snapshot()
        if not isinstance(frame, CameraFrame):
            raise TypeError("camera source must return a CameraFrame snapshot")
        return {"scene.caption": _caption_payload(frame)}


def _caption_payload(frame: CameraFrame) -> dict[str, object]:
    return {
        "caption": frame.caption.strip(),
        "labels": list(frame.labels),
        "boxes": [_box_payload(box) for box in frame.boxes],
    }


def _box_payload(box: SceneBox | dict[str, Any]) -> dict[str, Any]:
    if isinstance(box, SceneBox):
        return box.model_dump()
    return dict(box)


def _obs_id(frame: CameraFrame, suffix: str) -> str | None:
    if frame.obs_id is None:
        return None
    return f"{frame.obs_id}-{suffix}"
