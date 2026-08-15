from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from aila.device_services import audio_input_config, camera_input_config
from aila.queue import ObservationQueue
from aila.workers.feedback import write_playback_reference
from aila.workers.config import WorkerConfig
from aila.workers.mic import MicWorker, SpeechSegment
from aila.workers.mic_local import LocalSpeechSegmentSource, build_speech_segment_source


@dataclass
class FakeSpeechSegmentSource:
    segments: tuple[SpeechSegment, ...]

    def poll_segments(self) -> tuple[SpeechSegment, ...]:
        return self.segments


@dataclass
class FakeStatusSource:
    segments: tuple[SpeechSegment, ...]
    statuses: tuple[dict, ...]

    def poll_segments(self) -> tuple[SpeechSegment, ...]:
        return self.segments

    def poll_status(self) -> tuple[dict, ...]:
        return self.statuses


def test_mic_worker_surfaces_sensor_status_observation(tmp_path: Path) -> None:
    queue = ObservationQueue(tmp_path / "queue")
    worker = MicWorker(
        _mic_config(vad=True),
        audio_input_config(device="default"),
        FakeStatusSource(
            segments=(),
            statuses=(
                {"component": "transcriber", "state": "unavailable", "detail": "whisper down"},
            ),
        ),
        queue,
    )

    observations = worker.poll_once()

    assert [o.kind for o in observations] == ["sensor.status"]
    assert observations[0].worker == "mic"
    assert observations[0].payload.component == "transcriber"
    assert observations[0].payload.state == "unavailable"
    assert observations[0].severity.value == "warning"


def test_mic_worker_emits_vad_gated_speech_segments(tmp_path: Path) -> None:
    queue = ObservationQueue(tmp_path / "queue")
    worker = MicWorker(
        _mic_config(vad=True),
        audio_input_config(device="default", capture={"sample_rate_hz": 16000}),
        FakeSpeechSegmentSource(
            (
                _segment("speech-1", "hello aila", 0, vad_active=True),
                _segment("speech-2", "background noise", 100, vad_active=False),
                _segment("speech-3", "second transcript", 200, vad_active=True),
            )
        ),
        queue,
    )

    observations = worker.poll_once()

    assert worker.device_service.service == "audio-input"
    assert worker.device_service.capture == {"sample_rate_hz": 16000}
    assert [observation.obs_id for observation in observations] == ["speech-1", "speech-3"]
    assert [observation.kind for observation in observations] == [
        "speech.segment",
        "speech.segment",
    ]
    assert [observation.payload.text for observation in observations] == [
        "hello aila",
        "second transcript",
    ]
    assert [item.observation.obs_id for item in queue.drain()] == ["speech-1", "speech-3"]


def test_local_mic_source_builds_from_local_model_config() -> None:
    config = WorkerConfig.model_validate(
        {
            "worker": "mic",
            "role": "sensor",
            "device_service": "audio-input",
            "backend": {"kind": "model", "placement": "local", "model": "whisper-large-v3-turbo-q5"},
            "sampling": {
                "source": {"device": "default"},
                "pipeline": {
                    "vad": {"enabled": False},
                    "denoise": {"enabled": False},
                    "stt": {"enabled": False},
                    "audio_events": {"enabled": False},
                    "speaker_embedding": {"enabled": False},
                    "diarization": {"enabled": False},
                },
            },
            "emits": ["speech.segment"],
            "verbs": [],
        }
    )

    source = build_speech_segment_source(config, audio_input_config(device="default"))

    assert isinstance(source, LocalSpeechSegmentSource)
    assert source.status.silero_vad is False
    assert source.status.whisper is False
    assert source.status.rnnoise is False
    assert source.status.yamnet is False


def test_mic_worker_persists_transcripts_without_raw_audio(tmp_path: Path) -> None:
    queue = ObservationQueue(tmp_path / "queue")
    worker = MicWorker(
        _mic_config(vad=True),
        audio_input_config(device="default"),
        FakeSpeechSegmentSource(
            (
                SpeechSegment(
                    obs_id="private-audio",
                    ts=_timestamp(0),
                    text="derived transcript only",
                    lang="en",
                    confidence=0.98,
                    start_ms=0,
                    end_ms=1500,
                    vad_active=True,
                    raw_audio=b"RAW_AUDIO_BYTES_MUST_NOT_BE_PERSISTED",
                ),
            )
        ),
        queue,
    )

    observations = worker.poll_once()
    persisted = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted((tmp_path / "queue" / "pending").glob("*.json"))
    )

    assert observations[0].payload.text == "derived transcript only"
    assert observations[0].media_ref is None
    assert "derived transcript only" in persisted
    assert "raw_audio" not in persisted
    assert "RAW_AUDIO_BYTES_MUST_NOT_BE_PERSISTED" not in persisted


def test_mic_worker_suppresses_speaker_echo_but_keeps_barge_in(tmp_path: Path) -> None:
    state_path = tmp_path / "state" / "speaker-playback.json"
    write_playback_reference(
        state_path,
        text="The weather is sunny today.",
        duration_ms=2000,
        tail_ms=750,
        now=datetime.now(UTC),
    )
    worker = MicWorker(
        _mic_config(vad=True, echo_state_path=state_path),
        audio_input_config(device="default"),
        FakeSpeechSegmentSource(
            (
                _segment("echo", "the weather is sunny today", 0, vad_active=True),
                _segment("barge", "wait stop talking", 100, vad_active=True),
            )
        ),
        ObservationQueue(tmp_path / "queue"),
    )

    observations = worker.poll_once()

    assert [observation.obs_id for observation in observations] == ["barge"]
    assert observations[0].payload.text == "wait stop talking"


def test_mic_worker_requires_audio_input_device_service(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="audio-input"):
        MicWorker(
            _mic_config(vad=True),
            camera_input_config(device="/dev/video0"),
            FakeSpeechSegmentSource(()),
            ObservationQueue(tmp_path / "queue"),
        )


def _mic_config(*, vad: bool, echo_state_path: Path | None = None) -> WorkerConfig:
    sampling: dict[str, object] = {"vad": vad, "max_segment_seconds": 15}
    if echo_state_path is not None:
        sampling["echo_filter"] = {
            "enabled": True,
            "speaker_state_path": str(echo_state_path),
            "mode": "transcript-similarity",
            "similarity_threshold": 0.82,
            "keep_barge_in": True,
        }
    return WorkerConfig.model_validate(
        {
            "worker": "mic",
            "role": "sensor",
            "device_service": "audio-input",
            "backend": {"kind": "deterministic", "placement": "local"},
            "sampling": sampling,
            "emits": ["speech.segment"],
            "verbs": [],
        }
    )


def _segment(
    obs_id: str,
    text: str,
    offset_ms: int,
    *,
    vad_active: bool,
) -> SpeechSegment:
    return SpeechSegment(
        obs_id=obs_id,
        ts=_timestamp(offset_ms),
        text=text,
        lang="en",
        confidence=0.95,
        start_ms=offset_ms,
        end_ms=offset_ms + 500,
        vad_active=vad_active,
        raw_audio=f"raw-audio-{obs_id}".encode("utf-8"),
    )


def _timestamp(offset_ms: int) -> datetime:
    return datetime(2026, 7, 13, 12, 0, tzinfo=UTC) + timedelta(milliseconds=offset_ms)
