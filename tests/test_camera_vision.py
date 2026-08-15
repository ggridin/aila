from __future__ import annotations

from aila.workers.camera_local import (
    DEFAULT_VISION_PROMPT,
    VISION_UNAVAILABLE_CAPTION,
    LocalCameraFrameSource,
    VisionConfig,
    build_vlm_payload,
    consume_look_request,
    frame_change_ratio,
    read_latest_state,
    should_look,
    write_latest_state,
)
from aila.device_services import camera_input_config
from aila.workers.config import WorkerConfig


def test_vision_config_defaults_and_from_sampling() -> None:
    assert VisionConfig().endpoint.endswith("/v1")
    cfg = VisionConfig.from_sampling(
        {
            "vision": {
                "endpoint": "http://lan-gpu:9000/v1",
                "model": "Qwen2.5-VL-7B",
                "min_interval_seconds": 20,
                "motion_trigger_level": 0.2,
            }
        }
    )
    assert cfg.endpoint == "http://lan-gpu:9000/v1"
    assert cfg.model == "Qwen2.5-VL-7B"
    assert cfg.min_interval_seconds == 20.0
    assert cfg.motion_trigger_level == 0.2
    # unspecified fields keep defaults
    assert cfg.keepalive_seconds == VisionConfig().keepalive_seconds


def test_vision_config_missing_section_uses_defaults() -> None:
    assert VisionConfig.from_sampling({}) == VisionConfig()
    assert VisionConfig.from_sampling({"vision": None}) == VisionConfig()


def test_should_look_first_frame_then_throttled() -> None:
    cfg = VisionConfig(min_interval_seconds=10, keepalive_seconds=60, motion_trigger_level=0.1, dark_mean_min=20)
    # First look always happens (last_look_at is None), given enough light.
    assert should_look(now=100.0, last_look_at=None, motion_level=0.0, mean_brightness=100.0, config=cfg) is True
    # Within the throttle floor: suppressed even with motion.
    assert should_look(now=105.0, last_look_at=100.0, motion_level=0.9, mean_brightness=100.0, config=cfg) is False
    # After the floor, motion triggers a look.
    assert should_look(now=111.0, last_look_at=100.0, motion_level=0.5, mean_brightness=100.0, config=cfg) is True
    # After the floor, no motion, before keepalive: suppressed.
    assert should_look(now=130.0, last_look_at=100.0, motion_level=0.0, mean_brightness=100.0, config=cfg) is False
    # Keepalive elapsed: look even without motion.
    assert should_look(now=161.0, last_look_at=100.0, motion_level=0.0, mean_brightness=100.0, config=cfg) is True


def test_should_look_skips_dark_frames() -> None:
    cfg = VisionConfig(dark_mean_min=20)
    assert should_look(now=1.0, last_look_at=None, motion_level=1.0, mean_brightness=5.0, config=cfg) is False


def test_build_vlm_payload_shape() -> None:
    payload = build_vlm_payload(model="m", prompt="p", max_tokens=32, image_b64="AAAA")
    assert payload["model"] == "m"
    assert payload["max_tokens"] == 32
    content = payload["messages"][0]["content"]
    assert content[0] == {"type": "text", "text": "p"}
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"] == "data:image/jpeg;base64,AAAA"


def test_default_prompt_is_nonempty() -> None:
    assert isinstance(DEFAULT_VISION_PROMPT, str) and DEFAULT_VISION_PROMPT.strip()


def test_vision_config_look_paths_configurable() -> None:
    cfg = VisionConfig.from_sampling(
        {
            "vision": {
                "state_path": "/tmp/cam-latest.json",
                "request_path": "/tmp/cam-look",
                "scene_dedup_level": 0.05,
            }
        }
    )
    assert cfg.state_path == "/tmp/cam-latest.json"
    assert cfg.request_path == "/tmp/cam-look"
    assert cfg.scene_dedup_level == 0.05


def test_latest_state_roundtrip(tmp_path) -> None:
    path = tmp_path / "state" / "camera-latest.json"
    write_latest_state(path, caption="a cat on a mat", motion=0.3, ts="2026-01-01T00:00:00+00:00")
    data = read_latest_state(path)
    assert data == {"caption": "a cat on a mat", "motion": 0.3, "ts": "2026-01-01T00:00:00+00:00"}


def test_read_latest_state_missing_returns_none(tmp_path) -> None:
    assert read_latest_state(tmp_path / "nope.json") is None


def test_consume_look_request(tmp_path) -> None:
    req = tmp_path / "camera-look-request"
    assert consume_look_request(req) is False
    req.write_text("now", encoding="utf-8")
    assert consume_look_request(req) is True
    assert not req.exists()
    assert consume_look_request(req) is False


class _FakeGray:
    """Minimal stand-in for a 2D grayscale frame (list-of-rows)."""

    def __init__(self, rows: list[list[int]]) -> None:
        self.rows = rows
        self.shape = (len(rows), len(rows[0]) if rows else 0)


class _FakeCv2:
    THRESH_BINARY = 0

    def absdiff(self, a: _FakeGray, b: _FakeGray) -> _FakeGray:
        return _FakeGray(
            [[abs(x - y) for x, y in zip(ra, rb)] for ra, rb in zip(a.rows, b.rows)]
        )

    def threshold(self, img: _FakeGray, thresh: int, maxval: int, _type: int):
        binary = _FakeGray([[maxval if v > thresh else 0 for v in row] for row in img.rows])
        return None, binary

    def countNonZero(self, img: _FakeGray) -> int:
        return sum(1 for row in img.rows for v in row if v)


def test_frame_change_ratio_identical_is_zero() -> None:
    cv2 = _FakeCv2()
    frame = _FakeGray([[10, 10], [10, 10]])
    assert frame_change_ratio(frame, _FakeGray([[10, 10], [10, 10]]), cv2) == 0.0


def test_frame_change_ratio_all_changed_is_one() -> None:
    cv2 = _FakeCv2()
    a = _FakeGray([[0, 0], [0, 0]])
    b = _FakeGray([[255, 255], [255, 255]])
    assert frame_change_ratio(a, b, cv2) == 1.0


def test_frame_change_ratio_no_baseline_returns_one() -> None:
    cv2 = _FakeCv2()
    assert frame_change_ratio(None, _FakeGray([[1]]), cv2) == 1.0


def test_frame_change_ratio_shape_mismatch_returns_one() -> None:
    cv2 = _FakeCv2()
    a = _FakeGray([[1, 2, 3]])
    b = _FakeGray([[1, 2]])
    assert frame_change_ratio(a, b, cv2) == 1.0


class _FakeImage:
    """Stand-in for a captured BGR frame; motion is disabled in tests."""

    shape = (2, 2, 3)

    def mean(self) -> float:
        return 100.0


def _make_source(tmp_path, describe_returns) -> LocalCameraFrameSource:
    config = WorkerConfig.model_validate(
        {
            "worker": "camera",
            "role": "sensor",
            "device_service": "camera-input",
            "backend": {"kind": "model", "placement": "local", "model": "camera-vlm"},
            "sampling": {
                "source": {"device": "/dev/video0"},
                "pipeline": {"motion": {"enabled": False}},
                "vision": {
                    "state_path": str(tmp_path / "camera-latest.json"),
                    "request_path": str(tmp_path / "camera-look-request"),
                },
            },
            "emits": ["scene.caption", "scene.motion"],
            "verbs": ["snapshot"],
        }
    )
    source = LocalCameraFrameSource(config, camera_input_config(device="/dev/video0"))
    # Force a usable pipeline without a real camera / cv2.
    source._cv2 = object()
    source._capture_frame = lambda: _FakeImage()  # type: ignore[assignment]
    source._motion_level = lambda image: 0.0  # type: ignore[assignment]
    calls = {"n": 0}

    def _describe(image):
        result = describe_returns[min(calls["n"], len(describe_returns) - 1)]
        calls["n"] += 1
        return result

    source._describe = _describe  # type: ignore[assignment]
    source._calls = calls  # type: ignore[attr-defined]
    return source


def test_poll_emits_unavailable_caption_when_vlm_down(tmp_path) -> None:
    source = _make_source(tmp_path, describe_returns=[None])
    frames = source.poll_frames()
    assert len(frames) == 1
    assert frames[0].caption == VISION_UNAVAILABLE_CAPTION
    # Outage must not poison the pre-dedup memory.
    assert source._last_described_gray is None
    assert source._last_caption == ""


def test_poll_retries_after_vlm_recovers(tmp_path) -> None:
    # First poll: VLM down. Second poll: VLM back with a real caption.
    source = _make_source(tmp_path, describe_returns=[None, "a desk with a laptop"])
    clock = {"t": 0.0}
    source._time = lambda: clock["t"]  # type: ignore[assignment]
    first = source.poll_frames()
    assert first[0].caption == VISION_UNAVAILABLE_CAPTION
    clock["t"] = 1000.0  # advance well past min_interval_seconds
    second = source.poll_frames()
    assert second[0].caption == "a desk with a laptop"
    assert source._last_caption == "a desk with a laptop"


def test_snapshot_returns_unavailable_caption_when_vlm_down(tmp_path) -> None:
    source = _make_source(tmp_path, describe_returns=[None])
    frame = source.capture_snapshot()
    assert frame.caption == VISION_UNAVAILABLE_CAPTION
    assert source._last_caption == ""



