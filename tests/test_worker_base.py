from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from aila.contracts import Command, RenderResult
from aila.queue import ObservationQueue
from aila.workers import (
    BackendObservation,
    DeterministicFakeBackend,
    EffectorWorker,
    SensorWorker,
    WorkerConfig,
    load_worker_config,
)


def test_load_worker_config_substitutes_environment_and_validates_contract(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        """
worker: mic
role: sensor
device_service: audio-input
backend:
  kind: model
  placement: lan
  endpoint: ${MIC_ENDPOINT}/v1
  endpoint_token: ${MIC_TOKEN}
  model: whisper-base
sampling:
  vad: true
  max_segment_seconds: 15
emits: [speech.segment]
verbs: []
requires:
  os: [portaudio19-dev, libsndfile1]
  python: [pywhispercpp]
  models: [whisper-base]
""",
        encoding="utf-8",
    )

    config = load_worker_config(
        path,
        environ={"MIC_ENDPOINT": "http://lan-models.local:9000", "MIC_TOKEN": "token"},
    )

    assert config.worker == "mic"
    assert config.role == "sensor"
    assert config.device_service == "audio-input"
    assert config.backend.endpoint == "http://lan-models.local:9000/v1"
    assert config.backend.endpoint_token == "token"
    assert config.sampling == {"vad": True, "max_segment_seconds": 15}
    assert config.emits == ("speech.segment",)
    assert config.verbs == ()
    assert config.requires.os == ("portaudio19-dev", "libsndfile1")


def test_load_worker_config_fails_fast_on_missing_environment(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        """
worker: speaker
role: effector
backend:
  kind: model
  placement: lan
  endpoint: ${SPEAKER_ENDPOINT}
  model: piper-en
emits: []
verbs: [speak]
""",
        encoding="utf-8",
    )

    with pytest.raises(KeyError, match="SPEAKER_ENDPOINT"):
        load_worker_config(path, environ={})


@pytest.mark.parametrize(
    "data",
    [
        {
            "worker": "speaker",
            "role": "sensor",
            "backend": {"kind": "model", "placement": "local", "model": "piper-en"},
            "emits": [],
            "verbs": ["speak"],
        },
        {
            "worker": "display",
            "role": "effector",
            "backend": {"kind": "deterministic", "placement": "lan"},
            "emits": [],
            "verbs": ["render", "clear"],
        },
        {
            "worker": "camera",
            "role": "sensor",
            "device_service": "audio-input",
            "backend": {"kind": "model", "placement": "local", "model": "captioner-small"},
            "emits": ["scene.caption", "scene.motion"],
            "verbs": ["snapshot"],
        },
        {
            "worker": "filesystem",
            "role": "sensor",
            "backend": {"kind": "deterministic", "placement": "local"},
            "emits": ["file.changed"],
            "verbs": [],
        },
    ],
)
def test_worker_config_rejects_invalid_role_backend_device_and_contract(data: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        WorkerConfig.model_validate(data)


def test_sensor_worker_polls_fake_backend_into_observation_queue(tmp_path: Path) -> None:
    config = WorkerConfig.model_validate(
        {
            "worker": "mic",
            "role": "sensor",
            "device_service": "audio-input",
            "backend": {"kind": "model", "placement": "local", "model": "whisper-base"},
            "emits": ["speech.segment"],
            "verbs": [],
        }
    )
    backend = DeterministicFakeBackend(
        observations=(
            BackendObservation(
                kind="speech.segment",
                obs_id="speech-1",
                ts=datetime(2026, 7, 13, 12, 0, tzinfo=UTC),
                payload={
                    "text": "hello",
                    "lang": "en",
                    "confidence": 0.9,
                    "start_ms": 0,
                    "end_ms": 10,
                },
            ),
        )
    )
    queue = ObservationQueue(tmp_path / "queue")

    observations = SensorWorker(config, backend, queue).poll_once()

    assert observations[0].worker == "mic"
    assert observations[0].kind == "speech.segment"
    assert [item.observation.obs_id for item in queue.drain()] == ["speech-1"]


def test_effector_worker_handles_commands_with_fake_backend() -> None:
    config = WorkerConfig.model_validate(
        {
            "worker": "display",
            "role": "effector",
            "backend": {"kind": "deterministic", "placement": "local"},
            "emits": [],
            "verbs": ["render", "clear"],
        }
    )
    backend = DeterministicFakeBackend(command_results={"render": {"rendered": True}})
    worker = EffectorWorker(config, backend)
    command = Command(
        id="cmd-1",
        worker="display",
        verb="render",
        args={"kind": "text", "content": "hi"},
    )

    result = worker.handle_command(command)

    assert result.ok is True
    assert isinstance(result.data, RenderResult)
    assert backend.commands == [command]


def test_effector_worker_returns_structured_backend_errors() -> None:
    config = WorkerConfig.model_validate(
        {
            "worker": "display",
            "role": "effector",
            "backend": {"kind": "deterministic", "placement": "local"},
            "emits": [],
            "verbs": ["render", "clear"],
        }
    )
    command = Command(
        id="cmd-1",
        worker="display",
        verb="clear",
        args={},
    )

    result = EffectorWorker(config, DeterministicFakeBackend()).handle_command(command)

    assert result.ok is False
    assert result.error is not None
    assert result.error.code == "BACKEND_ERROR"
