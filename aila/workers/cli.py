from __future__ import annotations

import argparse
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from aila.contracts import RenderArgs, SpeakArgs
from aila.device_services import (
    DeviceServiceConfig,
    audio_input_config,
    camera_input_config,
    load_device_service_config,
)
from aila.queue import ObservationQueue
from aila.registry import RegistryConfig, load_registry_config
from aila.workers.base import SensorWorker, WorkerRuntime
from aila.workers.camera import CameraFrame, CameraWorker
from aila.workers.camera_local import build_camera_frame_source
from aila.workers.config import WorkerConfig, load_worker_config
from aila.workers.display import DisplayWorker
from aila.workers.filesystem import FileChangeEvent, FilesystemWatchConfig, FilesystemWorker
from aila.workers.mic import MicWorker, SpeechSegment
from aila.workers.mic_local import build_speech_segment_source
from aila.workers.speaker import SpeakerWorker
from aila.workers.speaker_local import build_text_to_speech_backend


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    worker = load_worker_process(
        worker_id=args.worker_id,
        config_path=args.config,
        registry_path=args.registry,
        queue_dir=args.queue_dir,
        device_service_config_path=args.device_service_config,
    )
    if args.once:
        run_worker_once(worker)
        return 0
    try:
        run_worker_forever(worker)
    except KeyboardInterrupt:
        return 0
    return 0


def load_worker_process(
    *,
    worker_id: str,
    config_path: Path,
    registry_path: Path,
    queue_dir: Path,
    device_service_config_path: Path | None = None,
) -> WorkerRuntime:
    config = load_worker_config(config_path)
    registry = load_registry_config(registry_path)
    validate_worker_id(worker_id, config=config, registry=registry)
    device_service = _load_device_service_config(config, device_service_config_path)
    return build_worker(config, queue_dir=queue_dir, device_service=device_service)


def run_worker_once(worker: WorkerRuntime) -> None:
    if isinstance(worker, SensorWorker):
        worker.poll_once()


def run_worker_forever(
    worker: WorkerRuntime,
    *,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    interval_seconds = _poll_interval_seconds(worker.config)
    while True:
        run_worker_once(worker)
        sleep(interval_seconds)


def validate_worker_id(
    worker_id: str,
    *,
    config: WorkerConfig,
    registry: RegistryConfig,
) -> None:
    if worker_id != config.worker:
        raise ValueError(f"worker id {worker_id} does not match config worker {config.worker}")
    if worker_id not in registry.workers.enabled:
        raise ValueError(f"worker {worker_id} is not enabled in registry")


def build_worker(
    config: WorkerConfig,
    *,
    queue_dir: Path,
    device_service: DeviceServiceConfig | None = None,
) -> WorkerRuntime:
    queue = ObservationQueue(queue_dir)
    if config.worker == "mic":
        mic_device_service = _required_device_service(config, device_service)
        return MicWorker(
            config,
            mic_device_service,
            build_speech_segment_source(config, mic_device_service),
            queue,
        )
    if config.worker == "camera":
        camera_device_service = _required_device_service(config, device_service)
        return CameraWorker(
            config,
            camera_device_service,
            build_camera_frame_source(config, camera_device_service),
            queue,
        )
    if config.worker == "filesystem":
        return FilesystemWorker(
            config,
            _EmptyFileEventAdapter(),
            queue,
            _filesystem_watch_config(config),
        )
    if config.worker == "speaker":
        return SpeakerWorker(config, build_text_to_speech_backend(config))
    if config.worker == "display":
        return DisplayWorker(config, _NoopDisplayRenderer())
    raise ValueError(f"unsupported worker: {config.worker}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="aila-worker")
    parser.add_argument("worker_id")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--registry", required=True, type=Path)
    parser.add_argument("--queue-dir", required=True, type=Path)
    parser.add_argument("--device-service-config", type=Path)
    parser.add_argument(
        "--once",
        action="store_true",
        help="load the worker, run one sensor poll if applicable, then exit",
    )
    return parser


def _poll_interval_seconds(config: WorkerConfig) -> float:
    raw = config.sampling.get(
        "poll_interval_seconds",
        config.sampling.get("frame_interval_seconds", 1.0),
    )
    if isinstance(raw, bool) or not isinstance(raw, int | float):
        raise ValueError("worker poll interval must be a number")
    if raw <= 0:
        raise ValueError("worker poll interval must be greater than zero")
    return float(raw)


def _load_device_service_config(
    config: WorkerConfig,
    path: Path | None,
) -> DeviceServiceConfig | None:
    if config.device_service is None:
        return None
    if path is not None:
        return load_device_service_config(path)
    if config.device_service == "audio-input":
        return audio_input_config()
    if config.device_service == "camera-input":
        return camera_input_config()
    raise ValueError(f"unsupported device service: {config.device_service}")


def _required_device_service(
    config: WorkerConfig,
    device_service: DeviceServiceConfig | None,
) -> DeviceServiceConfig:
    if device_service is None:
        raise ValueError(f"worker {config.worker} requires device service {config.device_service}")
    return device_service


def _filesystem_watch_config(config: WorkerConfig) -> FilesystemWatchConfig:
    paths = config.sampling.get("paths", [str(Path.cwd())])
    if not isinstance(paths, list):
        raise ValueError("filesystem sampling.paths must be a list")
    ignore = config.sampling.get("ignore", [])
    if not isinstance(ignore, list):
        raise ValueError("filesystem sampling.ignore must be a list")
    debounce_ms = config.sampling.get("debounce_ms", 500)
    if not isinstance(debounce_ms, int):
        raise ValueError("filesystem sampling.debounce_ms must be an integer")
    return FilesystemWatchConfig(
        paths=tuple(Path(path) for path in paths),
        ignore=tuple(str(pattern) for pattern in ignore),
        debounce_ms=debounce_ms,
    )


@dataclass(frozen=True)
class _EmptySpeechSegmentSource:
    def poll_segments(self) -> tuple[SpeechSegment, ...]:
        return ()


@dataclass(frozen=True)
class _EmptyCameraFrameSource:
    snapshot: CameraFrame = field(
        default_factory=lambda: CameraFrame(caption="no camera frame available")
    )

    def poll_frames(self) -> tuple[CameraFrame, ...]:
        return ()

    def capture_snapshot(self) -> CameraFrame:
        return self.snapshot


@dataclass(frozen=True)
class _EmptyFileEventAdapter:
    def poll_events(self) -> tuple[FileChangeEvent, ...]:
        return ()


@dataclass
class _NoopTextToSpeechBackend:
    def speak(self, args: SpeakArgs) -> int:
        return 0


@dataclass
class _NoopDisplayRenderer:
    def render(self, args: RenderArgs) -> None:
        return None

    def clear(self) -> None:
        return None
