from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from aila.device_services import audio_input_config, camera_input_config
from aila.registry import RegistryConfig, load_registry_config, validate_registry
from aila.workers import WorkerConfig


def _worker_config(worker: str) -> WorkerConfig:
    data = {
        "mic": {
            "worker": "mic",
            "role": "sensor",
            "device_service": "audio-input",
            "backend": {"kind": "model", "placement": "local", "model": "whisper-base"},
            "emits": ["speech.segment"],
            "verbs": [],
        },
        "camera": {
            "worker": "camera",
            "role": "sensor",
            "device_service": "camera-input",
            "backend": {"kind": "model", "placement": "local", "model": "captioner-small"},
            "emits": ["scene.caption", "scene.motion"],
            "verbs": ["snapshot"],
        },
        "filesystem": {
            "worker": "filesystem",
            "role": "sensor",
            "backend": {"kind": "deterministic", "placement": "local"},
            "emits": ["file.changed", "file.created", "file.deleted"],
            "verbs": [],
        },
    }[worker]
    return WorkerConfig.model_validate(data)


def test_load_registry_config_validates_workers_enabled(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("workers:\n  enabled: [mic, camera, filesystem]\n", encoding="utf-8")

    config = load_registry_config(path)

    assert config.workers.enabled == ("mic", "camera", "filesystem")


@pytest.mark.parametrize(
    "enabled",
    [
        ["mic", "mic"],
        ["audio-input"],
        ["unknown"],
    ],
)
def test_registry_config_rejects_duplicates_device_services_and_unknown_workers(
    enabled: list[str],
) -> None:
    with pytest.raises(ValidationError):
        RegistryConfig.model_validate({"workers": {"enabled": enabled}})


def test_validate_registry_returns_required_device_services() -> None:
    registry = validate_registry(
        {"workers": {"enabled": ["mic", "camera", "filesystem"]}},
        worker_configs={
            "mic": _worker_config("mic"),
            "camera": _worker_config("camera"),
            "filesystem": _worker_config("filesystem"),
        },
        device_service_configs={
            "audio-input": audio_input_config(),
            "camera-input": camera_input_config(),
        },
    )

    assert registry.enabled_workers == ("mic", "camera", "filesystem")
    assert registry.required_device_services == ("audio-input", "camera-input")


def test_validate_registry_fails_fast_when_enabled_worker_is_missing() -> None:
    with pytest.raises(ValueError, match="missing enabled worker config: camera"):
        validate_registry(
            {"workers": {"enabled": ["mic", "camera"]}},
            available_workers=["mic"],
            available_device_services=["audio-input", "camera-input"],
        )


def test_validate_registry_fails_fast_when_required_device_service_is_missing() -> None:
    with pytest.raises(ValueError, match="missing required device-service config: camera-input"):
        validate_registry(
            {"workers": {"enabled": ["mic", "camera"]}},
            available_workers=["mic", "camera"],
            available_device_services=["audio-input"],
        )
