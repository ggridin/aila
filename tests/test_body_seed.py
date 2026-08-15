from __future__ import annotations

from pathlib import Path

import yaml

from aila.contracts.payloads import OBSERVATION_KINDS_BY_WORKER, VERBS_BY_WORKER
from aila.device_services import load_device_service_config
from aila.registry import load_registry_config, validate_registry_files
from aila.subscriptions import load_subscriptions
from aila.workers import load_worker_config

SEED_ROOT = Path(__file__).resolve().parents[1] / "workspace-seed"
BODY_ROOT = SEED_ROOT / "aila-body"

WORKERS = ("mic", "camera", "filesystem", "speaker", "display")
DEVICE_SERVICES = ("audio-input", "camera-input")

EXPECTED_BODY_FILES = {
    "subscriptions.yaml",
    "device-services/audio-input/config.yaml",
    "device-services/camera-input/config.yaml",
    "workers/mic/config.yaml",
    "workers/mic/.env.example",
    "workers/camera/config.yaml",
    "workers/camera/.env.example",
    "workers/filesystem/config.yaml",
    "workers/filesystem/.env.example",
    "workers/speaker/config.yaml",
    "workers/speaker/.env.example",
    "workers/display/config.yaml",
    "workers/display/.env.example",
    "queue/pending/.gitkeep",
    "queue/inflight/.gitkeep",
    "queue/archive/.gitkeep",
    "logs/.gitkeep",
    "systemd/aila-device-audio-input.service",
    "systemd/aila-device-camera-input.service",
    "systemd/aila-mic.service",
    "systemd/aila-camera.service",
    "systemd/aila-filesystem.service",
    "systemd/aila-speaker.service",
    "systemd/aila-display.service",
    "reflex-ranking.yaml",
    "systemd/aila-reflex-ingest.service",
    "systemd/aila-hindsight.service",
}


def test_body_seed_contains_step_18_files() -> None:
    files = {
        path.relative_to(BODY_ROOT).as_posix()
        for path in BODY_ROOT.rglob("*")
        if path.is_file()
    }

    assert files == EXPECTED_BODY_FILES


def test_worker_seed_configs_validate_against_fixed_contract_catalog() -> None:
    configs = {
        worker: load_worker_config(BODY_ROOT / "workers" / worker / "config.yaml")
        for worker in WORKERS
    }

    assert configs["mic"].role == "sensor"
    assert configs["mic"].device_service == "audio-input"
    assert configs["mic"].backend.placement == "local"
    assert configs["mic"].backend.model == "whisper-large-v3-turbo-q5"
    assert configs["mic"].sampling["vad"] is True
    assert configs["mic"].sampling["max_segment_seconds"] == 15
    assert configs["mic"].sampling["source"] == {
        "kind": "sounddevice",
        "device": "default",
        "sample_rate_hz": 16000,
        "channels": 1,
        "dtype": "int16",
    }
    mic_pipeline = configs["mic"].sampling["pipeline"]
    assert mic_pipeline["aec"] == {
        "enabled": False,
        "status": "hook-only",
        "mode": "playback-reference",
        "reference_state_path": "~/.hermes/aila-body/state/speaker-playback.json",
    }
    assert mic_pipeline["vad"] == {
        "model": "silero-vad-v6-onnx",
        "enabled": True,
        "status": "python-package",
        "runtime": "silero-vad",
        "source": "https://github.com/snakers4/silero-vad",
    }
    assert mic_pipeline["denoise"] == {
        "model": "rnnoise",
        "enabled": False,
        "status": "source-only",
        "source": "https://github.com/GregorR/rnnoise-models",
    }
    assert mic_pipeline["stt"]["status"] == "configured"
    assert mic_pipeline["stt"]["model_path"].endswith("whisper-large-v3-turbo-q5_k.gguf")
    assert mic_pipeline["audio_events"] == {
        "model": "yamnet",
        "enabled": False,
        "status": "source-only",
        "source": "https://www.kaggle.com/models/google/yamnet/tfLite/classification-tflite/1",
    }
    assert mic_pipeline["speaker_embedding"]["runtime"] == "speechbrain"
    assert mic_pipeline["diarization"]["runtime"] == "pyannote.audio"
    assert configs["mic"].sampling["echo_filter"] == {
        "enabled": True,
        "speaker_state_path": "~/.hermes/aila-body/state/speaker-playback.json",
        "mode": "transcript-similarity",
        "similarity_threshold": 0.82,
        "keep_barge_in": True,
        "playback_tail_ms": 750,
    }

    assert configs["camera"].role == "sensor"
    assert configs["camera"].device_service == "camera-input"
    assert configs["camera"].backend.placement == "local"
    assert configs["camera"].backend.model == "camera-vlm"
    assert configs["camera"].sampling["frame_interval_seconds"] == 5
    assert configs["camera"].sampling["vision"]["model"] == "Qwen2.5-VL-3B"
    assert configs["camera"].sampling["source"] == {
        "kind": "opencv-v4l2",
        "device": "/dev/video0",
        "width": 640,
        "height": 480,
        "fps": 5,
    }
    assert configs["camera"].sampling["pipeline"]["motion"] == {
        "method": "opencv-frame-differencing",
        "enabled": True,
        "threshold": 25,
        "region": "frame",
    }
    vision = configs["camera"].sampling["vision"]
    assert vision["endpoint"].endswith(":8081/v1")
    assert vision["model"] == "Qwen2.5-VL-3B"
    assert vision["min_interval_seconds"] == 12
    assert configs["camera"].requires.python == ("opencv-python-headless",)

    assert configs["filesystem"].backend.kind == "deterministic"
    assert configs["filesystem"].sampling["paths"] == [
        "~/projects",
        "~/.hermes/aila-home",
    ]
    assert configs["filesystem"].requires.python == ("watchdog",)

    assert configs["speaker"].role == "effector"
    assert configs["speaker"].backend.placement == "local"
    assert configs["speaker"].backend.model == "piper-en"
    assert configs["speaker"].sampling["backend"] == "piper-cli"
    assert configs["speaker"].sampling["model_path"].endswith("en_US-lessac-medium.onnx")
    assert configs["speaker"].sampling["config_path"].endswith("en_US-lessac-medium.onnx.json")
    assert configs["speaker"].sampling["temp_dir"] == "~/.hermes/aila-body/tmp/speaker"
    assert configs["speaker"].sampling["playback"] == {"backend": "aplay", "device": "default"}
    assert configs["speaker"].sampling["feedback_reference"] == {
        "enabled": True,
        "state_path": "~/.hermes/aila-body/state/speaker-playback.json",
        "tail_ms": 750,
        "include_text": True,
    }
    assert configs["speaker"].requires.os == ("libsndfile1", "alsa-utils")
    assert configs["speaker"].requires.python == ("piper-tts", "onnxruntime")

    assert configs["display"].role == "effector"
    assert configs["display"].backend.kind == "deterministic"
    assert configs["display"].sampling == {"target": "framebuffer", "default_region": "full"}

    for worker, config in configs.items():
        # sensor.status is an optional cross-cutting health kind, not declared
        # in seed emits.
        expected = OBSERVATION_KINDS_BY_WORKER[worker] - {"sensor.status"}
        assert set(config.emits) == expected
        assert set(config.verbs) == VERBS_BY_WORKER[worker]


def test_device_service_seed_configs_validate_and_are_not_brain_facing() -> None:
    configs = {
        service: load_device_service_config(
            BODY_ROOT / "device-services" / service / "config.yaml"
        )
        for service in DEVICE_SERVICES
    }

    assert configs["audio-input"].kind == "audio"
    assert configs["audio-input"].consumer == "mic"
    assert configs["audio-input"].capture["sample_rate_hz"] == 16000
    assert configs["camera-input"].kind == "camera"
    assert configs["camera-input"].consumer == "camera"
    assert configs["camera-input"].capture["fps"] == 5

    registry = validate_registry_files(
        load_registry_config(SEED_ROOT / "config.yaml"),
        workers_dir=BODY_ROOT / "workers",
        device_services_dir=BODY_ROOT / "device-services",
    )

    assert registry.enabled_workers == WORKERS
    assert registry.required_device_services == DEVICE_SERVICES


def test_subscriptions_seed_loads_queue_only_sensor_events() -> None:
    subscriptions = load_subscriptions(BODY_ROOT / "subscriptions.yaml")

    assert [subscription.on_match for subscription in subscriptions] == [
        "queue",
        "queue",
        "queue",
    ]
    assert [(subscription.worker, subscription.kind) for subscription in subscriptions] == [
        ("mic", "speech.segment"),
        ("camera", "scene.caption"),
        ("filesystem", "file.changed"),
    ]


def test_secret_templates_are_placeholders_only() -> None:
    for worker in WORKERS:
        worker_dir = BODY_ROOT / "workers" / worker
        text = (worker_dir / ".env.example").read_text(encoding="utf-8")

        assert not (worker_dir / ".env").exists()
        assert "replace-with-" in text or "No " in text
        assert "sk-" not in text


def test_queue_and_logs_seed_empty_tracked_directories() -> None:
    for relative_path in (
        "queue/pending/.gitkeep",
        "queue/inflight/.gitkeep",
        "queue/archive/.gitkeep",
        "logs/.gitkeep",
    ):
        assert (BODY_ROOT / relative_path).read_text(encoding="utf-8") == ""


def test_systemd_user_units_reference_seeded_configs_and_order_device_services() -> None:
    for worker in WORKERS:
        unit = _unit(f"aila-{worker}.service")

        assert "[Unit]" in unit
        assert "[Service]" in unit
        assert "[Install]" in unit
        assert f"aila-worker {worker}" in unit
        assert f"%h/.hermes/aila-body/workers/{worker}/config.yaml" in unit
        assert "%h/.hermes/config.yaml" in unit
        assert "%h/.hermes/aila-body/queue" in unit
        assert "Restart=on-failure" in unit
        assert f"logs/{worker}.log" in unit

    assert "Requires=aila-device-audio-input.service" in _unit("aila-mic.service")
    assert "After=aila-device-audio-input.service" in _unit("aila-mic.service")
    assert "Requires=aila-device-camera-input.service" in _unit("aila-camera.service")
    assert "After=aila-device-camera-input.service" in _unit("aila-camera.service")

    for service in DEVICE_SERVICES:
        unit = _unit(f"aila-device-{service}.service")

        assert "load_device_service_config" in unit
        assert f"%h/.hermes/aila-body/device-services/{service}/config.yaml" in unit
        assert "Restart=on-failure" in unit
        assert f"logs/device-{service}.log" in unit


def test_seed_yaml_files_are_plain_mappings() -> None:
    yaml_paths = [
        BODY_ROOT / "subscriptions.yaml",
        *(BODY_ROOT / "workers" / worker / "config.yaml" for worker in WORKERS),
        *(
            BODY_ROOT / "device-services" / service / "config.yaml"
            for service in DEVICE_SERVICES
        ),
    ]

    for path in yaml_paths:
        parsed = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert isinstance(parsed, dict)


def _unit(name: str) -> str:
    return (BODY_ROOT / "systemd" / name).read_text(encoding="utf-8")
