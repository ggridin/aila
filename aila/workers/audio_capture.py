from __future__ import annotations

from typing import Any


class SoundDeviceCapture:
    """Non-blocking microphone capture that yields fixed-size PCM frames.

    Thin adapter over ``sounddevice`` (imported lazily so the module stays
    importable without the sensory venv). Audio is buffered by an input stream
    callback; ``read_frames`` drains whatever whole frames are available.
    """

    def __init__(
        self,
        *,
        device: str,
        sample_rate_hz: int,
        channels: int,
        frame_ms: int,
    ) -> None:
        import queue

        import sounddevice as sd

        self._sd = sd
        self._channels = channels
        self._frame_bytes = int(sample_rate_hz * (frame_ms / 1000.0)) * channels * 2
        self._queue: "queue.Queue[bytes]" = queue.Queue()
        self._residual = bytearray()

        def _callback(indata, _frames, _time, _status) -> None:  # pragma: no cover - hw callback
            self._queue.put(bytes(indata))

        self._stream = sd.RawInputStream(
            samplerate=sample_rate_hz,
            channels=channels,
            dtype="int16",
            device=None if device in ("", "default") else device,
            callback=_callback,
        )
        self._stream.start()

    def read_frames(self) -> list[bytes]:
        import queue

        frames: list[bytes] = []
        while True:
            try:
                self._residual.extend(self._queue.get_nowait())
            except queue.Empty:
                break
        while len(self._residual) >= self._frame_bytes:
            frames.append(bytes(self._residual[: self._frame_bytes]))
            del self._residual[: self._frame_bytes]
        return frames

    def close(self) -> None:  # pragma: no cover - hw teardown
        try:
            self._stream.stop()
            self._stream.close()
        except Exception:
            pass


class SileroVad:
    """Per-frame speech probability using the Silero VAD ONNX/torch model."""

    def __init__(self, *, threshold: float = 0.5) -> None:
        self.threshold = threshold
        self._model: Any | None = None
        self._load()

    def _load(self) -> None:  # pragma: no cover - model load is environment dependent
        try:
            from silero_vad import load_silero_vad

            self._model = load_silero_vad(onnx=True)
        except Exception:
            self._model = None

    def speech_probability(self, frame: bytes, sample_rate_hz: int) -> float:  # pragma: no cover - model inference
        if self._model is None:
            return 1.0
        import numpy as np
        import torch

        samples = np.frombuffer(frame, dtype=np.int16).astype("float32") / 32768.0
        tensor = torch.from_numpy(samples)
        with torch.no_grad():
            return float(self._model(tensor, sample_rate_hz).item())
