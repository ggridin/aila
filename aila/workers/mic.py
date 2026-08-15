from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Protocol

from aila.contracts import Command, Severity
from aila.device_services import DeviceServiceConfig
from aila.queue import ObservationQueue
from aila.workers.backends import BackendObservation
from aila.workers.base import SensorWorker
from aila.workers.config import WorkerConfig
from aila.workers.feedback import read_playback_reference, should_suppress_echo


@dataclass(frozen=True)
class SpeechSegment:
    text: str
    lang: str
    confidence: float
    start_ms: int
    end_ms: int
    vad_active: bool = True
    obs_id: str | None = None
    ts: datetime | None = None
    raw_audio: bytes | None = None

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError("speech segment text must not be empty")
        if not self.lang:
            raise ValueError("speech segment lang must not be empty")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("speech segment confidence must be between 0 and 1")
        if self.start_ms < 0 or self.end_ms < self.start_ms:
            raise ValueError("speech segment timestamps must be non-negative and ordered")


class SpeechSegmentSource(Protocol):
    def poll_segments(self) -> tuple[SpeechSegment, ...]:
        raise NotImplementedError


class MicWorker(SensorWorker):
    def __init__(
        self,
        config: WorkerConfig,
        device_service: DeviceServiceConfig,
        source: SpeechSegmentSource,
        queue: ObservationQueue,
    ) -> None:
        if config.worker != "mic":
            raise ValueError(f"mic worker cannot use config for {config.worker}")
        if device_service.service != "audio-input":
            raise ValueError(f"mic worker requires audio-input, got {device_service.service}")
        if device_service.consumer != "mic":
            raise ValueError("audio-input device service must be assigned to mic")
        if device_service.kind != "audio":
            raise ValueError("audio-input device service must use audio kind")

        self.device_service = device_service
        super().__init__(
            config,
            _MicBackend(
                source,
                vad_enabled=bool(config.sampling.get("vad", True)),
                echo_filter=_echo_filter_config(config),
            ),
            queue,
        )


class _MicBackend:
    def __init__(
        self,
        source: SpeechSegmentSource,
        *,
        vad_enabled: bool,
        echo_filter: dict[str, object],
    ) -> None:
        self._source = source
        self._vad_enabled = vad_enabled
        self._echo_filter = echo_filter

    def poll(self) -> tuple[BackendObservation, ...]:
        observations: list[BackendObservation] = []
        for segment in self._source.poll_segments():
            if not isinstance(segment, SpeechSegment):
                raise TypeError("mic source must yield SpeechSegment instances")
            if self._vad_enabled and not segment.vad_active:
                continue
            if self._is_speaker_echo(segment):
                continue
            observations.append(_observation_for_segment(segment))
        observations.extend(self._poll_status_observations())
        return tuple(observations)

    def _poll_status_observations(self) -> list[BackendObservation]:
        poll_status = getattr(self._source, "poll_status", None)
        if not callable(poll_status):
            return []
        results: list[BackendObservation] = []
        for status in poll_status():
            results.append(_observation_for_status(status))
        return results

    def handle_command(self, command: Command) -> object:
        raise NotImplementedError("mic worker does not support commands")

    def _is_speaker_echo(self, segment: SpeechSegment) -> bool:
        if not bool(self._echo_filter.get("enabled", False)):
            return False
        state_path = self._echo_filter.get("speaker_state_path")
        if not isinstance(state_path, str) or not state_path:
            return False
        reference = read_playback_reference(Path(state_path).expanduser())
        threshold = float(self._echo_filter.get("similarity_threshold", 0.82))
        return should_suppress_echo(segment.text, reference, threshold=threshold)


def _observation_for_segment(segment: SpeechSegment) -> BackendObservation:
    return BackendObservation(
        kind="speech.segment",
        obs_id=segment.obs_id,
        ts=segment.ts,
        payload={
            "text": segment.text.strip(),
            "lang": segment.lang,
            "confidence": segment.confidence,
            "start_ms": segment.start_ms,
            "end_ms": segment.end_ms,
        },
    )


def _observation_for_status(status: object) -> BackendObservation:
    if not isinstance(status, dict):
        raise TypeError("mic source poll_status must yield mappings")
    state = str(status.get("state", "unavailable"))
    severity = Severity.warning if state != "ok" else Severity.info
    return BackendObservation(
        kind="sensor.status",
        severity=severity,
        payload={
            "component": str(status.get("component", "transcriber")),
            "state": state,
            "detail": str(status.get("detail", "")),
        },
    )


def _echo_filter_config(config: WorkerConfig) -> dict[str, object]:
    raw = config.sampling.get("echo_filter", {})
    if not isinstance(raw, dict):
        raise ValueError("mic sampling.echo_filter must be a mapping")
    return raw
