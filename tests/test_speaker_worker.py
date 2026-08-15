from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pytest

from aila.contracts import Command, SpeakArgs, SpeakResult
from aila.workers.feedback import read_playback_reference
from aila.workers.config import WorkerConfig
from aila.workers.speaker import SpeakerWorker
from aila.workers.speaker_local import PiperTextToSpeechBackend


@dataclass
class FakeTextToSpeechBackend:
    duration_ms: int
    synthesized_audio: bytes = b"SYNTHESIZED_AUDIO_MUST_NOT_BE_PERSISTED"
    spoken: list[SpeakArgs] = field(default_factory=list)

    def speak(self, args: SpeakArgs) -> int:
        self.spoken.append(args)
        return self.duration_ms


def test_speaker_worker_speaks_with_fake_tts_backend() -> None:
    tts = FakeTextToSpeechBackend(duration_ms=1250)
    worker = SpeakerWorker(_speaker_config(), tts)
    command = Command(
        id="cmd-speak",
        worker="speaker",
        verb="speak",
        args={"text": "Hello AILA", "voice": "test-voice", "rate": 1.1},
    )

    result = worker.handle_command(command)

    assert result.ok is True
    assert isinstance(result.data, SpeakResult)
    assert result.data.duration_ms == 1250
    assert tts.spoken == [SpeakArgs(text="Hello AILA", voice="test-voice", rate=1.1)]


def test_speaker_worker_does_not_persist_synthesized_audio(tmp_path: Path) -> None:
    tts = FakeTextToSpeechBackend(duration_ms=500)
    worker = SpeakerWorker(_speaker_config(), tts)
    command = Command(
        id="cmd-private",
        worker="speaker",
        verb="speak",
        args={"text": "Derived command text only"},
    )

    result = worker.handle_command(command)
    persisted = "\n".join(
        path.read_text(encoding="utf-8") for path in sorted(tmp_path.rglob("*")) if path.is_file()
    )

    assert result.ok is True
    assert isinstance(result.data, SpeakResult)
    assert result.data.duration_ms == 500
    assert tts.synthesized_audio == b"SYNTHESIZED_AUDIO_MUST_NOT_BE_PERSISTED"
    assert "SYNTHESIZED_AUDIO_MUST_NOT_BE_PERSISTED" not in persisted


def test_speaker_worker_is_output_only_and_emits_no_observations() -> None:
    worker = SpeakerWorker(_speaker_config(), FakeTextToSpeechBackend(duration_ms=1))

    assert worker.backend.poll() == ()


def test_piper_backend_runs_synthesis_and_playback_commands(tmp_path: Path) -> None:
    model_path = tmp_path / "voice.onnx"
    config_path = tmp_path / "voice.onnx.json"
    model_path.write_bytes(b"model")
    config_path.write_text("{}", encoding="utf-8")
    commands: list[list[str]] = []

    def runner(command: list[str]):
        commands.append(command)
        if command[0] == "piper":
            output_path = Path(command[command.index("--output-file") + 1])
            _write_wav(output_path)
        return _Completed()

    backend = PiperTextToSpeechBackend(
        _speaker_config(
            sampling={
                "backend": "piper-cli",
                "model_path": str(model_path),
                "config_path": str(config_path),
                "temp_dir": str(tmp_path / "tmp"),
                "cleanup_audio": True,
                "playback": {"backend": "aplay", "device": "default"},
                "feedback_reference": {
                    "enabled": True,
                    "state_path": str(tmp_path / "state" / "speaker-playback.json"),
                    "tail_ms": 750,
                    "include_text": True,
                },
            }
        ),
        runner=runner,
        require_commands=False,
    )

    duration_ms = backend.speak(SpeakArgs(text="hello", rate=1.0))

    assert duration_ms == 1000
    assert commands[0][0] == "piper"
    assert commands[1][:4] == ["aplay", "-q", "-D", "default"]
    assert not list((tmp_path / "tmp").glob("*.wav"))
    reference = read_playback_reference(tmp_path / "state" / "speaker-playback.json")
    assert reference is not None
    assert reference.text == "hello"
    assert reference.duration_ms == 1000


def test_speaker_worker_rejects_non_speaker_config() -> None:
    config = WorkerConfig.model_validate(
        {
            "worker": "display",
            "role": "effector",
            "backend": {"kind": "deterministic", "placement": "local"},
            "emits": [],
            "verbs": ["render", "clear"],
        }
    )

    with pytest.raises(ValueError, match="speaker worker cannot use config for display"):
        SpeakerWorker(config, FakeTextToSpeechBackend(duration_ms=1))


def _speaker_config(*, sampling: dict[str, object] | None = None) -> WorkerConfig:
    return WorkerConfig.model_validate(
        {
            "worker": "speaker",
            "role": "effector",
            "backend": {"kind": "model", "placement": "local", "model": "piper-en"},
            "sampling": sampling or {},
            "emits": [],
            "verbs": ["speak"],
        }
    )


class _Completed:
    returncode = 0


def _write_wav(path: Path) -> None:
    import wave

    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        wav_file.writeframes(b"\0\0" * 16000)
