from __future__ import annotations

import base64
import json
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel, ConfigDict

from aila._paths import expand_path
from aila.device_services import DeviceServiceConfig
from aila.workers.camera import CameraFrame, CameraFrameSource
from aila.workers.config import WorkerConfig
from aila.workers.model_client import post_for_json


DEFAULT_VISION_PROMPT = (
    "You are the eyes of an AI living on a laptop. In one concise, factual "
    "sentence, describe what you see: the setting, notable objects, and any "
    "people and what they are doing. Do not speculate beyond the image."
)

# Emitted (and published) when the VLM backend cannot be reached or fails, so
# the agent can tell it is temporarily blind rather than seeing an empty room.
VISION_UNAVAILABLE_CAPTION = "vision unavailable (could not reach the scene-description model)"


class VisionConfig(BaseModel):
    """Configuration for the vision-language captioning backend."""

    model_config = ConfigDict(extra="ignore", frozen=True, protected_namespaces=())

    endpoint: str = "http://127.0.0.1:8081/v1"
    model: str = "Qwen2.5-VL-3B"
    prompt: str = DEFAULT_VISION_PROMPT
    max_tokens: int = 64
    timeout_seconds: float = 30.0
    downscale_width: int = 640
    # Trigger policy (keeps VLM calls to roughly one per min_interval_seconds).
    min_interval_seconds: float = 12.0
    keepalive_seconds: float = 120.0
    motion_trigger_level: float = 0.12
    dark_mean_min: float = 20.0
    # Perceptual pre-dedup: after the trigger fires, compare the frame against
    # the last frame we actually described. If fewer than this fraction of
    # pixels changed, the scene is effectively unchanged -- reuse the previous
    # caption and skip the VLM call entirely.
    scene_dedup_level: float = 0.02
    # Deliberate "look": the worker publishes its latest caption here and honors
    # an on-demand request file (dropped by the body dispatcher) by forcing a
    # fresh description on the next poll.
    state_path: str = "~/.hermes/aila-body/state/camera-latest.json"
    request_path: str = "~/.hermes/aila-body/state/camera-look-request"

    @classmethod
    def from_sampling(cls, sampling: dict[str, Any]) -> "VisionConfig":
        section = sampling.get("vision") if isinstance(sampling, dict) else None
        if not section:
            return cls()
        if not isinstance(section, dict):
            raise ValueError("camera sampling.vision must be a mapping")
        # Pydantic applies defaults and coercion; unknown keys are ignored.
        return cls.model_validate(section)


def write_latest_state(path: Path, *, caption: str, motion: float, ts: str) -> None:
    """Atomically publish the most recent caption for the deliberate look."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps({"caption": caption, "motion": motion, "ts": ts}),
        encoding="utf-8",
    )
    tmp.replace(path)


def read_latest_state(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def consume_look_request(path: Path) -> bool:
    """Return True (and clear it) if an on-demand look was requested."""
    try:
        if path.exists():
            path.unlink()
            return True
    except Exception:
        pass
    return False


def should_look(
    *,
    now: float,
    last_look_at: float | None,
    motion_level: float,
    mean_brightness: float,
    config: VisionConfig,
) -> bool:
    """Decide whether to invoke the VLM this cycle.

    Event-driven with a throttle floor: look on the first frame, then only when
    something moved (motion >= trigger) or a keepalive interval elapsed, and
    never more often than ``min_interval_seconds``. Dark frames are skipped.
    """

    if mean_brightness < config.dark_mean_min:
        return False
    if last_look_at is None:
        return True
    elapsed = now - last_look_at
    if elapsed < config.min_interval_seconds:
        return False
    if motion_level >= config.motion_trigger_level:
        return True
    if elapsed >= config.keepalive_seconds:
        return True
    return False


def frame_change_ratio(previous_gray: Any, current_gray: Any, cv2: Any, *, threshold: int = 25) -> float:
    """Fraction of pixels that differ between two blurred grayscale frames.

    Returns 1.0 (treat as fully changed) when there is no baseline yet or the
    frames are not comparable, so the caller never suppresses on missing data.
    """
    if previous_gray is None or current_gray is None:
        return 1.0
    if getattr(previous_gray, "shape", None) != getattr(current_gray, "shape", None):
        return 1.0
    diff = cv2.absdiff(previous_gray, current_gray)
    _, binary = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    changed = float(cv2.countNonZero(binary))
    total = float(binary.shape[0] * binary.shape[1])
    if total <= 0:
        return 1.0
    return max(0.0, min(1.0, changed / total))


def build_vlm_payload(
    *,
    model: str,
    prompt: str,
    max_tokens: int,
    image_b64: str,
    mime: str = "image/jpeg",
) -> dict[str, Any]:
    """Build an OpenAI-compatible multimodal chat request for the VLM."""

    return {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": 0.2,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{image_b64}"}},
                ],
            }
        ],
    }


@dataclass(frozen=True)
class CameraPipelineStatus:
    opencv: bool
    vision_endpoint: str


class LocalCameraFrameSource(CameraFrameSource):
    def __init__(
        self,
        config: WorkerConfig,
        device_service: DeviceServiceConfig,
        *,
        time_source: Callable[[], float] = time.monotonic,
    ) -> None:
        self.config = config
        self.device_service = device_service
        self.sampling = config.sampling
        self.pipeline = self._mapping(self.sampling.get("pipeline", {}), "sampling.pipeline")
        self.source = self._mapping(self.sampling.get("source", {}), "sampling.source")
        self.vision = VisionConfig.from_sampling(self.sampling)
        self._previous_gray: Any | None = None
        self._last_described_gray: Any | None = None
        self._last_caption: str = ""
        self._cv2: Any | None = self._import_cv2()
        self._capture: Any | None = None
        self._time = time_source
        self._last_look_at: float | None = None
        self.status = CameraPipelineStatus(
            opencv=self._cv2 is not None,
            vision_endpoint=self.vision.endpoint,
        )

    def poll_frames(self) -> tuple[CameraFrame, ...]:
        image = self._capture_frame()
        if image is None:
            return ()
        motion = self._motion_level(image)
        mean_brightness = float(image.mean()) if self._cv2 is not None else 0.0
        now = self._time()
        forced = consume_look_request(expand_path(self.vision.request_path))
        if not forced and not should_look(
            now=now,
            last_look_at=self._last_look_at,
            motion_level=motion,
            mean_brightness=mean_brightness,
            config=self.vision,
        ):
            return ()
        # Perceptual pre-dedup: if an ambient trigger fired but the scene is
        # effectively identical to the last frame we described, reuse that
        # caption and skip the VLM. A forced (deliberate) look always runs.
        if not forced and self._scene_unchanged():
            self._last_look_at = now
            caption = self._last_caption or "scene captured (no description available)"
            self._publish_latest(caption, motion)
            return (self._frame(image, caption=caption, motion=motion, obs_id=f"camera-{_timestamp_id()}"),)
        self._last_look_at = now
        description = self._describe(image)
        if description is None:
            # VLM outage: surface a distinct caption and do NOT remember this
            # frame, so we retry (rather than reusing a placeholder) once the
            # backend recovers.
            caption = VISION_UNAVAILABLE_CAPTION
            self._publish_latest(caption, motion)
            return (self._frame(image, caption=caption, motion=motion, obs_id=f"camera-{_timestamp_id()}"),)
        caption = description or "scene captured (no description available)"
        self._remember_described(caption)
        self._publish_latest(caption, motion)
        return (self._frame(image, caption=caption, motion=motion, obs_id=f"camera-{_timestamp_id()}"),)

    def capture_snapshot(self) -> CameraFrame:
        # Deliberate "look": always describe, ignoring the throttle.
        image = self._capture_frame()
        if image is None:
            return CameraFrame(caption="no camera frame available", labels=(), motion_level=0.0)
        motion = self._motion_level(image)
        self._last_look_at = self._time()
        description = self._describe(image)
        if description is None:
            caption = VISION_UNAVAILABLE_CAPTION
            self._publish_latest(caption, motion)
            return self._frame(image, caption=caption, motion=motion, obs_id=f"snapshot-{_timestamp_id()}")
        caption = description or "scene captured (no description available)"
        self._remember_described(caption)
        self._publish_latest(caption, motion)
        return self._frame(image, caption=caption, motion=motion, obs_id=f"snapshot-{_timestamp_id()}")

    def _scene_unchanged(self) -> bool:
        """True when the current frame ~matches the last frame we described."""
        if self._cv2 is None or self._last_described_gray is None:
            return False
        ratio = frame_change_ratio(self._last_described_gray, self._previous_gray, self._cv2)
        return ratio < self.vision.scene_dedup_level

    def _remember_described(self, caption: str) -> None:
        """Snapshot the frame/caption that was just sent to the VLM."""
        self._last_caption = caption
        if self._cv2 is not None and self._previous_gray is not None:
            self._last_described_gray = self._previous_gray.copy()

    def _publish_latest(self, caption: str, motion: float) -> None:
        try:
            write_latest_state(
                expand_path(self.vision.state_path),
                caption=caption,
                motion=motion,
                ts=datetime.now(UTC).isoformat(),
            )
        except Exception:
            pass

    def _frame(self, image: Any, *, caption: str, motion: float, obs_id: str) -> CameraFrame:
        region = str(self._mapping(self.pipeline.get("motion", {}), "motion").get("region", "frame"))
        return CameraFrame(
            obs_id=obs_id,
            ts=datetime.now(UTC),
            caption=caption,
            labels=(),
            boxes=(),
            motion_level=motion,
            motion_region=region,
        )

    def _describe(self, image: Any) -> str | None:
        """Ask the VLM to caption the frame.

        Returns the caption text on success, or ``None`` when the VLM backend
        is unavailable / errored so callers can distinguish an outage from a
        genuine empty description.
        """
        if self._cv2 is None:
            return None
        try:
            image_b64 = self._encode_jpeg(image)
        except Exception:
            return None
        payload = build_vlm_payload(
            model=self.vision.model,
            prompt=self.vision.prompt,
            max_tokens=self.vision.max_tokens,
            image_b64=image_b64,
        )
        body = post_for_json(
            f"{self.vision.endpoint.rstrip('/')}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            content_type="application/json",
            timeout=self.vision.timeout_seconds,
        )
        if body is None:
            return None
        try:
            return str(body["choices"][0]["message"]["content"]).strip()
        except (KeyError, IndexError, TypeError):
            return None

    def _encode_jpeg(self, image: Any) -> str:
        img = image
        width = self.vision.downscale_width
        if width and image.shape[1] > width:
            scale = width / float(image.shape[1])
            new_size = (width, max(1, int(round(image.shape[0] * scale))))
            img = self._cv2.resize(image, new_size)
        ok, buffer = self._cv2.imencode(".jpg", img, [self._cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            raise RuntimeError("failed to JPEG-encode camera frame")
        return base64.b64encode(buffer.tobytes()).decode("ascii")

    def _capture_frame(self) -> Any | None:
        if self._cv2 is None:
            return None
        capture = self._get_capture()
        if capture is None:
            return None
        ok, frame = capture.read()
        if not ok:
            # The persistent handle went bad (e.g. device hiccup); drop it so
            # the next poll reopens cleanly instead of spinning on a dead fd.
            self._release_capture()
            return None
        return frame

    def _get_capture(self) -> Any | None:
        """Open the V4L2 capture once and reuse it across polls."""
        if self._capture is not None:
            return self._capture
        device = str(self.source.get("device", self.device_service.device))
        capture = self._cv2.VideoCapture(device)
        if not capture.isOpened():
            capture.release()
            return None
        width = self.source.get("width", self.device_service.capture.get("width"))
        height = self.source.get("height", self.device_service.capture.get("height"))
        fps = self.source.get("fps", self.device_service.capture.get("fps"))
        if width is not None:
            capture.set(self._cv2.CAP_PROP_FRAME_WIDTH, float(width))
        if height is not None:
            capture.set(self._cv2.CAP_PROP_FRAME_HEIGHT, float(height))
        if fps is not None:
            capture.set(self._cv2.CAP_PROP_FPS, float(fps))
        self._capture = capture
        return capture

    def _release_capture(self) -> None:
        if self._capture is not None:
            try:
                self._capture.release()
            except Exception:
                pass
            self._capture = None

    def close(self) -> None:
        """Release the persistent capture handle (idempotent)."""
        self._release_capture()

    def _motion_level(self, image: Any) -> float:
        if self._cv2 is None:
            return 0.0
        motion = self._mapping(self.pipeline.get("motion", {}), "motion")
        if not bool(motion.get("enabled", True)):
            return 0.0
        gray = self._cv2.cvtColor(image, self._cv2.COLOR_BGR2GRAY)
        gray = self._cv2.GaussianBlur(gray, (21, 21), 0)
        if self._previous_gray is None:
            self._previous_gray = gray
            return 0.0
        diff = self._cv2.absdiff(self._previous_gray, gray)
        self._previous_gray = gray
        threshold = int(motion.get("threshold", 25))
        _, binary = self._cv2.threshold(diff, threshold, 255, self._cv2.THRESH_BINARY)
        changed = float(self._cv2.countNonZero(binary))
        total = float(binary.shape[0] * binary.shape[1])
        return max(0.0, min(1.0, changed / total if total else 0.0))

    @staticmethod
    def _import_cv2() -> Any | None:
        try:
            import cv2

            return cv2
        except Exception:
            return None

    @staticmethod
    def _mapping(value: Any, name: str) -> dict[str, Any]:
        if not isinstance(value, dict):
            raise ValueError(f"{name} must be a mapping")
        return value


def build_camera_frame_source(config: WorkerConfig, device_service: DeviceServiceConfig) -> CameraFrameSource:
    if config.backend.kind == "model" and config.backend.placement == "local":
        return LocalCameraFrameSource(config, device_service)
    return _NoFrameSource()


@dataclass(frozen=True)
class _NoFrameSource(CameraFrameSource):
    snapshot: CameraFrame = CameraFrame(caption="no camera frame available")

    def poll_frames(self) -> tuple[CameraFrame, ...]:
        return ()

    def capture_snapshot(self) -> CameraFrame:
        return self.snapshot


def _timestamp_id() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
