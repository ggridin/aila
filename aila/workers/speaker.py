from __future__ import annotations

from typing import Protocol

from aila.contracts import Command, SpeakArgs
from aila.workers.backends import BackendError, BackendObservation
from aila.workers.base import EffectorWorker
from aila.workers.config import WorkerConfig


class TextToSpeechBackend(Protocol):
    def speak(self, args: SpeakArgs) -> int:
        raise NotImplementedError


class SpeakerWorker(EffectorWorker):
    def __init__(self, config: WorkerConfig, tts: TextToSpeechBackend) -> None:
        if config.worker != "speaker":
            raise ValueError(f"speaker worker cannot use config for {config.worker}")
        super().__init__(config, _SpeakerBackend(tts))


class _SpeakerBackend:
    def __init__(self, tts: TextToSpeechBackend) -> None:
        self._tts = tts

    def poll(self) -> tuple[BackendObservation, ...]:
        return ()

    def handle_command(self, command: Command) -> object:
        if command.verb != "speak":
            raise BackendError(f"speaker worker does not support verb {command.verb}")
        duration_ms = self._tts.speak(SpeakArgs.model_validate(command.args))
        return {"duration_ms": duration_ms}
