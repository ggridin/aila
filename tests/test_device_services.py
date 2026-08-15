from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from aila.device_services import (
    DeviceServiceConfig,
    audio_input_config,
    camera_input_config,
    load_device_service_config,
    required_device_services_for_workers,
)


def test_load_device_service_config_substitutes_environment(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        """
service: audio-input
consumer: mic
device: ${MIC_DEVICE}
capture:
  sample_rate_hz: 16000
requires:
  os: [portaudio19-dev, libsndfile1]
""",
        encoding="utf-8",
    )

    config = load_device_service_config(path, environ={"MIC_DEVICE": "default"})

    assert config.service == "audio-input"
    assert config.kind == "audio"
    assert config.consumer == "mic"
    assert config.device == "default"
    assert config.capture == {"sample_rate_hz": 16000}
    assert config.requires.os == ("portaudio19-dev", "libsndfile1")


def test_device_service_helpers_build_fixed_v1_services() -> None:
    audio = audio_input_config(device="default")
    camera = camera_input_config(device="/dev/video0")

    assert audio.service == "audio-input"
    assert audio.consumer == "mic"
    assert audio.kind == "audio"
    assert camera.service == "camera-input"
    assert camera.consumer == "camera"
    assert camera.kind == "camera"


def test_required_device_services_are_derived_from_workers() -> None:
    assert required_device_services_for_workers(["mic", "filesystem", "camera"]) == (
        "audio-input",
        "camera-input",
    )


@pytest.mark.parametrize(
    "data",
    [
        {"service": "mic", "consumer": "mic"},
        {"service": "audio-input", "consumer": "camera"},
        {"service": "camera-input", "kind": "audio", "consumer": "camera"},
        {"service": "unknown-input", "consumer": "mic"},
    ],
)
def test_device_service_config_rejects_invalid_service_consumer_and_kind(
    data: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        DeviceServiceConfig.model_validate(data)
