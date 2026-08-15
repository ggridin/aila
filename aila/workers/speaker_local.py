from __future__ import annotations

import subprocess
import sys
import wave
from dataclasses import dataclass
from pathlib import Path
from time import time_ns
from typing import Callable, Sequence

from aila._paths import expand_path
from aila.contracts import SpeakArgs
from aila.workers.config import WorkerConfig
from aila.workers.feedback import write_playback_reference
from aila.workers.speaker import TextToSpeechBackend

CommandRunner = Callable[[Sequence[str]], subprocess.CompletedProcess[str]]


@dataclass(frozen=True)
class SpeakerPipelineStatus:
    piper: bool
    aplay: bool
    model: bool
    config: bool
    temp_dir: bool


class PiperTextToSpeechBackend(TextToSpeechBackend):
    def __init__(
        self,
        config: WorkerConfig,
        *,
        runner: CommandRunner | None = None,
        require_commands: bool = True,
    ) -> None:
        self.config = config
        self.sampling = config.sampling
        self.runner = runner or _run_command
        self.require_commands = require_commands
        self.model_path = self._path(self.sampling.get("model_path"))
        self.config_path = self._path(self.sampling.get("config_path"))
        self.temp_dir = self._path(self.sampling.get("temp_dir"))
        self.playback = self._mapping(self.sampling.get("playback", {}), "sampling.playback")
        self.feedback_reference = self._mapping(
            self.sampling.get("feedback_reference", {}), "sampling.feedback_reference"
        )
        self.piper_command = _command_path("piper")
        self.aplay_command = _command_path("aplay")
        self.status = self._status()

    def speak(self, args: SpeakArgs) -> int:
        self._ensure_ready()
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        text_path = self.temp_dir / f"speak-{time_ns()}.txt"
        wav_path = self.temp_dir / f"speak-{time_ns()}.wav"
        try:
            text_path.write_text(args.text, encoding="utf-8")
            self.runner(self._piper_command(text_path, wav_path, args))
            duration_ms = _wav_duration_ms(wav_path)
            self._write_feedback_reference(args, duration_ms)
            self.runner(self._playback_command(wav_path))
            return duration_ms
        finally:
            if bool(self.sampling.get("cleanup_audio", True)):
                text_path.unlink(missing_ok=True)
                wav_path.unlink(missing_ok=True)

    def _piper_command(self, text_path: Path, wav_path: Path, args: SpeakArgs) -> list[str]:
        command = [
            self.piper_command or "piper",
            "--model",
            str(self.model_path),
            "--config",
            str(self.config_path),
            "--input-file",
            str(text_path),
            "--output-file",
            str(wav_path),
            "--speaker",
            str(self.sampling.get("speaker_id", 0)),
            "--length-scale",
            str(self._length_scale(args)),
            "--noise-scale",
            str(self.sampling.get("noise_scale", 0.667)),
            "--noise-w-scale",
            str(self.sampling.get("noise_w_scale", 0.8)),
            "--sentence-silence",
            str(self.sampling.get("sentence_silence", 0.2)),
            "--volume",
            str(self.sampling.get("volume", 1.0)),
        ]
        return command

    def _playback_command(self, wav_path: Path) -> list[str]:
        backend = self.playback.get("backend", "aplay")
        if backend != "aplay":
            raise ValueError(f"unsupported speaker playback backend: {backend}")
        device = str(self.playback.get("device", "default"))
        return [self.aplay_command or "aplay", "-q", "-D", device, str(wav_path)]

    def _length_scale(self, args: SpeakArgs) -> float:
        configured = float(self.sampling.get("length_scale", 1.0))
        return configured / float(args.rate or 1.0)

    def _write_feedback_reference(self, args: SpeakArgs, duration_ms: int) -> None:
        if not bool(self.feedback_reference.get("enabled", False)):
            return
        state_path = self.feedback_reference.get("state_path")
        if not isinstance(state_path, str) or not state_path:
            return
        include_text = bool(self.feedback_reference.get("include_text", True))
        write_playback_reference(
            Path(state_path).expanduser(),
            text=args.text if include_text else "",
            duration_ms=duration_ms,
            tail_ms=int(self.feedback_reference.get("tail_ms", 750)),
            backend=str(self.sampling.get("backend", "piper-cli")),
            model=str(self.config.backend.model or "piper-en"),
        )

    def _ensure_ready(self) -> None:
        status = self._status()
        if not status.piper:
            if self.require_commands:
                raise RuntimeError("piper executable not found")
        if not status.aplay:
            if self.require_commands:
                raise RuntimeError("aplay executable not found")
        if not status.model:
            raise RuntimeError(f"Piper model not found: {self.model_path}")
        if not status.config:
            raise RuntimeError(f"Piper config not found: {self.config_path}")
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    def _status(self) -> SpeakerPipelineStatus:
        temp_ready = self.temp_dir.exists() or self.temp_dir.parent.exists()
        return SpeakerPipelineStatus(
            piper=self.piper_command is not None,
            aplay=self.aplay_command is not None,
            model=self.model_path.is_file(),
            config=self.config_path.is_file(),
            temp_dir=temp_ready,
        )

    @staticmethod
    def _path(value: object) -> Path:
        if not isinstance(value, str) or not value:
            raise ValueError("speaker path setting must be a non-empty string")
        return expand_path(value)

    @staticmethod
    def _mapping(value: object, name: str) -> dict[str, object]:
        if not isinstance(value, dict):
            raise ValueError(f"{name} must be a mapping")
        return value


def build_text_to_speech_backend(config: WorkerConfig) -> TextToSpeechBackend:
    if config.backend.kind == "model" and config.backend.placement == "local":
        return PiperTextToSpeechBackend(config)
    return _NoopTextToSpeechBackend()


@dataclass
class _NoopTextToSpeechBackend(TextToSpeechBackend):
    def speak(self, args: SpeakArgs) -> int:
        return 0


def _run_command(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(list(command), check=True, text=True, capture_output=True)


def _command_path(command: str) -> str | None:
    from shutil import which

    resolved = which(command)
    if resolved is not None:
        return resolved
    for venv_candidate in (
        Path(sys.executable).parent / command,
        Path(sys.executable).resolve().parent / command,
        Path.home() / ".hermes" / "venv" / "bin" / command,
    ):
        if venv_candidate.is_file():
            return str(venv_candidate)
    return None


def _wav_duration_ms(path: Path) -> int:
    try:
        with wave.open(str(path), "rb") as wav_file:
            frames = wav_file.getnframes()
            rate = wav_file.getframerate()
            if rate <= 0:
                return 0
            return int((frames / float(rate)) * 1000)
    except (FileNotFoundError, wave.Error):
        return 0
