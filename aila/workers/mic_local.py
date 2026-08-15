from __future__ import annotations

import io
import sys
import wave
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from aila._paths import expand_path
from aila.device_services import DeviceServiceConfig
from aila.workers.config import WorkerConfig
from aila.workers.health import BackendHealth
from aila.workers.mic import SpeechSegment, SpeechSegmentSource
from aila.workers.model_client import post_for_json

# Emitted as a sensor.status observation when whisper-server cannot be reached
# or errors, so the agent can tell its ears are offline rather than assuming
# silence. Mirrors the camera's "vision unavailable" contract.
TRANSCRIPTION_UNAVAILABLE_DETAIL = "could not reach the speech-to-text model (whisper-server)"


@dataclass(frozen=True)
class MicPipelineStatus:
    capture: bool
    silero_vad: bool
    whisper: bool
    rnnoise: bool
    yamnet: bool
    speechbrain: bool
    pyannote: bool


class WhisperConfig(BaseModel):
    """Configuration for the whisper.cpp transcription backend."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    enabled: bool = True
    endpoint: str = "http://127.0.0.1:8082/inference"
    language: str = "auto"
    timeout_seconds: float = 30.0
    sample_rate_hz: int = 16000
    channels: int = 1
    # Minimum characters in a transcript to accept it (drops empty/garbage).
    min_chars: int = 1

    @classmethod
    def from_pipeline(cls, pipeline: dict[str, Any], source: dict[str, Any]) -> "WhisperConfig":
        stt = pipeline.get("stt") if isinstance(pipeline, dict) else None
        stt = stt if isinstance(stt, dict) else {}
        src = source if isinstance(source, dict) else {}
        # stt keys map straight onto the model; sample rate / channels come from
        # the shared source section. Pydantic applies defaults and coercion.
        data: dict[str, Any] = {
            key: stt[key]
            for key in ("enabled", "endpoint", "language", "timeout_seconds", "min_chars")
            if key in stt
        }
        for key in ("sample_rate_hz", "channels"):
            if key in src:
                data[key] = src[key]
        return cls.model_validate(data)


def encode_wav(pcm: bytes, *, sample_rate_hz: int, channels: int, sample_width: int = 2) -> bytes:
    """Wrap raw little-endian PCM samples in a WAV container (stdlib only)."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(sample_width)
        wav.setframerate(sample_rate_hz)
        wav.writeframes(pcm)
    return buffer.getvalue()


@dataclass
class SpeechSpan:
    """A detected span of speech (raw PCM) awaiting transcription."""

    pcm: bytes
    start_ms: int
    end_ms: int
    vad_confidence: float = 1.0


class VadSegmenter:
    """Turn a stream of fixed-size audio frames into speech spans.

    Pure and dependency-free: the caller supplies a per-frame speech
    probability (from Silero VAD or any detector). A span opens when a frame
    is speech and closes after ``hangover_frames`` of trailing non-speech or
    when it reaches ``max_span_ms``.
    """

    def __init__(
        self,
        *,
        frame_ms: int,
        sample_rate_hz: int,
        threshold: float = 0.5,
        hangover_frames: int = 8,
        max_span_ms: int = 15000,
        min_span_ms: int = 200,
    ) -> None:
        self.frame_ms = frame_ms
        self.sample_rate_hz = sample_rate_hz
        self.threshold = threshold
        self.hangover_frames = hangover_frames
        self.max_span_ms = max_span_ms
        self.min_span_ms = min_span_ms
        self._active = False
        self._silence = 0
        self._buffer = bytearray()
        self._start_ms = 0
        self._elapsed_ms = 0
        self._prob_sum = 0.0
        self._prob_count = 0

    def push(self, frame: bytes, speech_prob: float) -> list[SpeechSpan]:
        """Feed one frame; return any spans that completed on this frame."""
        spans: list[SpeechSpan] = []
        is_speech = speech_prob >= self.threshold
        if is_speech:
            if not self._active:
                self._active = True
                self._start_ms = self._elapsed_ms
                self._buffer = bytearray()
                self._silence = 0
                self._prob_sum = 0.0
                self._prob_count = 0
            self._buffer.extend(frame)
            self._silence = 0
            self._prob_sum += speech_prob
            self._prob_count += 1
        elif self._active:
            self._buffer.extend(frame)
            self._silence += 1

        self._elapsed_ms += self.frame_ms

        if self._active:
            span_ms = self._elapsed_ms - self._start_ms
            if self._silence >= self.hangover_frames or span_ms >= self.max_span_ms:
                span = self._close()
                if span is not None:
                    spans.append(span)
        return spans

    def flush(self) -> list[SpeechSpan]:
        span = self._close()
        return [span] if span is not None else []

    def _close(self) -> SpeechSpan | None:
        if not self._active:
            return None
        end_ms = self._elapsed_ms
        pcm = bytes(self._buffer)
        confidence = self._prob_sum / self._prob_count if self._prob_count else 0.0
        start_ms = self._start_ms
        self._active = False
        self._buffer = bytearray()
        self._silence = 0
        if (end_ms - start_ms) < self.min_span_ms or not pcm:
            return None
        return SpeechSpan(
            pcm=pcm,
            start_ms=start_ms,
            end_ms=end_ms,
            vad_confidence=max(0.0, min(1.0, confidence)),
        )


class LocalSpeechSegmentSource(SpeechSegmentSource):
    def __init__(self, config: WorkerConfig, device_service: DeviceServiceConfig) -> None:
        self.config = config
        self.device_service = device_service
        self.sampling = config.sampling
        self.source = self._mapping(self.sampling.get("source", {}), "sampling.source")
        self.pipeline = self._mapping(self.sampling.get("pipeline", {}), "sampling.pipeline")
        self.whisper = WhisperConfig.from_pipeline(self.pipeline, self.source)
        self._frame_ms = int(self.pipeline.get("frame_ms", 32))
        self._max_segment_ms = int(float(self.sampling.get("max_segment_seconds", 15)) * 1000)
        self._add_sensory_site_packages()
        self._capture: Any | None = None
        self._vad: Any | None = None
        self._segmenter = VadSegmenter(
            frame_ms=self._frame_ms,
            sample_rate_hz=self.whisper.sample_rate_hz,
            max_span_ms=self._max_segment_ms,
        )
        # Track whisper reachability so we only emit a status observation on a
        # healthy->unavailable transition, not once per failed poll.
        self._whisper_health = BackendHealth()
        self.status = self._initialize_pipeline()

    # -- capture + segmentation ------------------------------------------

    def poll_segments(self) -> tuple[SpeechSegment, ...]:
        if not self.whisper.enabled:
            return ()
        spans: list[SpeechSpan] = []
        for frame in self._read_frames():
            prob = self._speech_probability(frame)
            spans.extend(self._segmenter.push(frame, prob))
        segments: list[SpeechSegment] = []
        for span in spans:
            segment = self._transcribe_span(span)
            if segment is not None:
                segments.append(segment)
        return tuple(segments)

    def poll_status(self) -> tuple[dict[str, Any], ...]:
        """Return pending health/status signals (e.g. whisper outage)."""
        if not self._whisper_health.take_degraded():
            return ()
        return (
            {
                "component": "transcriber",
                "state": "unavailable",
                "detail": TRANSCRIPTION_UNAVAILABLE_DETAIL,
            },
        )

    def _transcribe_span(self, span: SpeechSpan) -> SpeechSegment | None:
        text = self._transcribe(span.pcm)
        if text is None:
            # Outage: flag a degraded signal (once per transition) and do not
            # fabricate a transcript. We retry on the next span/poll.
            self._whisper_health.record(False)
            return None
        self._whisper_health.record(True)
        text = text.strip()
        if len(text) < self.whisper.min_chars:
            return None
        return SpeechSegment(
            text=text,
            lang=self.whisper.language if self.whisper.language != "auto" else "und",
            confidence=span.vad_confidence,
            start_ms=span.start_ms,
            end_ms=span.end_ms,
            vad_active=True,
            obs_id=f"mic-{_timestamp_id()}",
            ts=datetime.now(UTC),
        )

    def _transcribe(self, pcm: bytes) -> str | None:
        """POST the WAV to whisper-server. Return text, or None on any failure."""
        wav = encode_wav(
            pcm,
            sample_rate_hz=self.whisper.sample_rate_hz,
            channels=self.whisper.channels,
        )
        body, content_type = _multipart_wav(wav, language=self.whisper.language)
        parsed = post_for_json(
            self.whisper.endpoint,
            data=body,
            content_type=content_type,
            timeout=self.whisper.timeout_seconds,
        )
        if parsed is None:
            return None
        return _extract_transcript(parsed)

    def _read_frames(self) -> list[bytes]:
        """Drain available fixed-size PCM frames from the capture device."""
        capture = self._get_capture()
        if capture is None:
            return []
        try:
            return list(capture.read_frames())
        except Exception:
            self._release_capture()
            return []

    def _speech_probability(self, frame: bytes) -> float:
        vad = self._get_vad()
        if vad is None:
            # Without a VAD model, treat every frame as speech so audio still
            # flows to whisper (whisper has its own silence handling).
            return 1.0
        try:
            return float(vad.speech_probability(frame, self.whisper.sample_rate_hz))
        except Exception:
            return 1.0

    def _get_capture(self) -> Any | None:
        if self._capture is not None:
            return self._capture
        self._capture = self._open_capture()
        return self._capture

    def _open_capture(self) -> Any | None:
        if not self.status.capture:
            return None
        try:
            from aila.workers.audio_capture import SoundDeviceCapture

            return SoundDeviceCapture(
                device=str(self.source.get("device", self.device_service.device)),
                sample_rate_hz=self.whisper.sample_rate_hz,
                channels=self.whisper.channels,
                frame_ms=self._frame_ms,
            )
        except Exception:
            return None

    def _get_vad(self) -> Any | None:
        if self._vad is not None:
            return self._vad
        if not self.status.silero_vad:
            return None
        try:
            from aila.workers.audio_capture import SileroVad

            self._vad = SileroVad(threshold=self._segmenter.threshold)
            return self._vad
        except Exception:
            return None

    def _release_capture(self) -> None:
        if self._capture is not None:
            try:
                self._capture.close()
            except Exception:
                pass
            self._capture = None

    def close(self) -> None:
        self._release_capture()

    # -- readiness probes (startup) --------------------------------------

    def _initialize_pipeline(self) -> MicPipelineStatus:
        return MicPipelineStatus(
            capture=self._import_available("sounddevice"),
            silero_vad=self._silero_ready(),
            whisper=self._whisper_ready(),
            rnnoise=self._enabled_stage_ready("denoise", import_name=None),
            yamnet=self._enabled_stage_ready("audio_events", import_name="tensorflow_hub"),
            speechbrain=self._enabled_stage_ready("speaker_embedding", import_name="speechbrain"),
            pyannote=self._enabled_stage_ready("diarization", import_name="pyannote.audio"),
        )

    def _silero_ready(self) -> bool:
        vad = self._mapping(self.pipeline.get("vad", {}), "vad")
        if not bool(vad.get("enabled", True)):
            return False
        return self._import_available("silero_vad") or self._import_available("torch")

    def _whisper_ready(self) -> bool:
        stt = self._mapping(self.pipeline.get("stt", {}), "stt")
        if not bool(stt.get("enabled", True)):
            return False
        # An HTTP endpoint means readiness is runtime (reachability), not a file.
        if stt.get("endpoint"):
            return True
        model_path = self._expand_path(stt.get("model_path"))
        return model_path is not None and model_path.is_file()

    def _enabled_stage_ready(self, key: str, *, import_name: str | None) -> bool:
        stage = self._mapping(self.pipeline.get(key, {}), key)
        if not bool(stage.get("enabled", False)):
            return False
        if import_name is None:
            return False
        return self._import_available(import_name)

    @staticmethod
    def _import_available(module_name: str) -> bool:
        try:
            __import__(module_name)
            return True
        except Exception:
            return False

    @staticmethod
    def _add_sensory_site_packages() -> None:
        sensory_venv = Path.home() / ".hermes" / "aila-body" / "models" / "sensory" / "venv"
        site_root = sensory_venv / "lib"
        if not site_root.is_dir():
            return
        for site_packages in site_root.glob("python*/site-packages"):
            site_path = str(site_packages)
            if site_path not in sys.path:
                sys.path.append(site_path)

    def _expand_path(self, value: Any) -> Path | None:
        if not isinstance(value, str) or not value:
            return None
        return expand_path(value)

    @staticmethod
    def _mapping(value: Any, name: str) -> dict[str, Any]:
        if not isinstance(value, dict):
            raise ValueError(f"{name} must be a mapping")
        return value


def _multipart_wav(wav: bytes, *, language: str) -> tuple[bytes, str]:
    """Build a multipart/form-data body for whisper.cpp /inference."""
    boundary = "----aila-whisper-boundary"
    crlf = b"\r\n"
    parts: list[bytes] = []
    parts.append(f"--{boundary}".encode("ascii"))
    parts.append(b'Content-Disposition: form-data; name="file"; filename="audio.wav"')
    parts.append(b"Content-Type: audio/wav")
    parts.append(b"")
    parts.append(wav)
    for field, value in (("temperature", "0.0"), ("response_format", "json"), ("language", language)):
        parts.append(f"--{boundary}".encode("ascii"))
        parts.append(f'Content-Disposition: form-data; name="{field}"'.encode("ascii"))
        parts.append(b"")
        parts.append(value.encode("ascii"))
    parts.append(f"--{boundary}--".encode("ascii"))
    parts.append(b"")
    body = crlf.join(parts)
    return body, f"multipart/form-data; boundary={boundary}"


def _extract_transcript(parsed: Any) -> str | None:
    """Pull the transcript text from a whisper.cpp /inference response."""
    if isinstance(parsed, dict):
        text = parsed.get("text")
        if isinstance(text, str):
            return text.strip()
    return None


def build_speech_segment_source(config: WorkerConfig, device_service: DeviceServiceConfig) -> SpeechSegmentSource:
    if config.backend.kind == "model" and config.backend.placement == "local":
        return LocalSpeechSegmentSource(config, device_service)
    return _NoSpeechSegmentSource()


@dataclass(frozen=True)
class _NoSpeechSegmentSource(SpeechSegmentSource):
    def poll_segments(self) -> tuple[SpeechSegment, ...]:
        return ()


def _timestamp_id() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
