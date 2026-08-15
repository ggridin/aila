from __future__ import annotations

import io
import wave

from aila.device_services import audio_input_config
from aila.workers.config import WorkerConfig
from aila.workers.mic import SpeechSegment
from aila.workers.mic_local import (
    TRANSCRIPTION_UNAVAILABLE_DETAIL,
    LocalSpeechSegmentSource,
    VadSegmenter,
    WhisperConfig,
    _extract_transcript,
    _multipart_wav,
    encode_wav,
)


def _config(stt: dict | None = None) -> WorkerConfig:
    pipeline = {
        "frame_ms": 20,
        "vad": {"enabled": False},
        "denoise": {"enabled": False},
        "stt": stt if stt is not None else {"enabled": True, "endpoint": "http://x/inference"},
        "audio_events": {"enabled": False},
        "speaker_embedding": {"enabled": False},
        "diarization": {"enabled": False},
    }
    return WorkerConfig.model_validate(
        {
            "worker": "mic",
            "role": "sensor",
            "device_service": "audio-input",
            "backend": {"kind": "model", "placement": "local", "model": "whisper-large-v3-turbo-q5"},
            "sampling": {
                "max_segment_seconds": 15,
                "source": {"device": "default", "sample_rate_hz": 16000, "channels": 1},
                "pipeline": pipeline,
            },
            "emits": ["speech.segment"],
            "verbs": [],
        }
    )


def _source(stt: dict | None = None) -> LocalSpeechSegmentSource:
    return LocalSpeechSegmentSource(_config(stt), audio_input_config(device="default"))


# -- WhisperConfig -------------------------------------------------------


def test_whisper_config_from_pipeline_reads_endpoint_and_source() -> None:
    cfg = WhisperConfig.from_pipeline(
        {"stt": {"endpoint": "http://lan:9000/inference", "language": "en", "timeout_seconds": 5}},
        {"sample_rate_hz": 8000, "channels": 2},
    )
    assert cfg.endpoint == "http://lan:9000/inference"
    assert cfg.language == "en"
    assert cfg.timeout_seconds == 5.0
    assert cfg.sample_rate_hz == 8000
    assert cfg.channels == 2


def test_whisper_config_defaults_when_missing() -> None:
    cfg = WhisperConfig.from_pipeline({}, {})
    assert cfg.enabled is True
    assert cfg.endpoint.endswith("/inference")
    assert cfg.sample_rate_hz == 16000


# -- WAV encoding --------------------------------------------------------


def test_encode_wav_roundtrips_pcm() -> None:
    pcm = b"\x01\x00\x02\x00\x03\x00"
    wav = encode_wav(pcm, sample_rate_hz=16000, channels=1)
    with wave.open(io.BytesIO(wav), "rb") as reader:
        assert reader.getframerate() == 16000
        assert reader.getnchannels() == 1
        assert reader.getsampwidth() == 2
        assert reader.readframes(reader.getnframes()) == pcm


def test_multipart_wav_has_boundary_and_file_field() -> None:
    body, content_type = _multipart_wav(b"WAVDATA", language="auto")
    assert "multipart/form-data; boundary=" in content_type
    assert b'name="file"' in body
    assert b"WAVDATA" in body


def test_extract_transcript_handles_shapes() -> None:
    assert _extract_transcript({"text": "  hi there "}) == "hi there"
    assert _extract_transcript({"nope": 1}) is None
    assert _extract_transcript(["x"]) is None


# -- VAD segmenter -------------------------------------------------------


def test_vad_segmenter_emits_span_after_hangover() -> None:
    seg = VadSegmenter(frame_ms=10, sample_rate_hz=16000, threshold=0.5, hangover_frames=2, min_span_ms=10)
    frame = b"\x00\x01"
    spans: list = []
    # 3 speech frames, then 2 silence frames -> one span closes.
    for prob in (0.9, 0.9, 0.9):
        spans += seg.push(frame, prob)
    assert spans == []
    spans += seg.push(frame, 0.0)
    spans += seg.push(frame, 0.0)
    assert len(spans) == 1
    assert spans[0].start_ms == 0
    assert spans[0].vad_confidence > 0.0


def test_vad_segmenter_closes_at_max_span() -> None:
    seg = VadSegmenter(frame_ms=10, sample_rate_hz=16000, threshold=0.5, hangover_frames=99, max_span_ms=30, min_span_ms=10)
    frame = b"\x00\x01"
    spans: list = []
    for _ in range(5):
        spans += seg.push(frame, 0.9)
    assert len(spans) == 1
    assert spans[0].end_ms - spans[0].start_ms >= 30


def test_vad_segmenter_drops_too_short_span() -> None:
    seg = VadSegmenter(frame_ms=10, sample_rate_hz=16000, threshold=0.5, hangover_frames=1, min_span_ms=50)
    frame = b"\x00\x01"
    spans = seg.push(frame, 0.9)
    spans += seg.push(frame, 0.0)
    assert spans == []


# -- transcription success / outage --------------------------------------


def test_poll_segments_transcribes_span(monkeypatch) -> None:
    source = _source()
    frame = b"\x00\x01" * 160  # 20ms @16k mono int16
    monkeypatch.setattr(source, "_read_frames", lambda: [frame] * 3 + [b"\x00\x00" * 160] * 12)
    monkeypatch.setattr(source, "_speech_probability", lambda f: 0.9 if f == frame else 0.0)
    monkeypatch.setattr(source, "_transcribe", lambda pcm: "hello world")

    segments = source.poll_segments()

    assert len(segments) == 1
    assert isinstance(segments[0], SpeechSegment)
    assert segments[0].text == "hello world"
    assert source.poll_status() == ()  # healthy: no degraded signal


def test_poll_segments_flags_degraded_on_outage(monkeypatch) -> None:
    source = _source()
    frame = b"\x00\x01" * 160
    monkeypatch.setattr(source, "_read_frames", lambda: [frame] * 3 + [b"\x00\x00" * 160] * 12)
    monkeypatch.setattr(source, "_speech_probability", lambda f: 0.9 if f == frame else 0.0)
    monkeypatch.setattr(source, "_transcribe", lambda pcm: None)  # whisper down

    segments = source.poll_segments()

    assert segments == ()  # no fabricated transcript
    status = source.poll_status()
    assert len(status) == 1
    assert status[0]["state"] == "unavailable"
    assert status[0]["detail"] == TRANSCRIPTION_UNAVAILABLE_DETAIL
    # Consumed once.
    assert source.poll_status() == ()


def test_poll_segments_disabled_returns_empty() -> None:
    source = _source(stt={"enabled": False})
    assert source.poll_segments() == ()
