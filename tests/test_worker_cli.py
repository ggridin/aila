from __future__ import annotations

from pathlib import Path

import pytest

from aila.workers import cli as worker_cli
from aila.workers.camera import CameraWorker
from aila.workers.cli import load_worker_process, main, run_worker_forever
from aila.workers.display import DisplayWorker
from aila.workers.filesystem import FilesystemWorker
from aila.workers.mic import MicWorker
from aila.workers.speaker import SpeakerWorker


def test_load_worker_process_dispatches_to_fixed_worker_implementation(tmp_path: Path) -> None:
    registry_path = _registry(tmp_path, "mic", "camera", "filesystem", "speaker", "display")

    cases = {
        "mic": MicWorker,
        "camera": CameraWorker,
        "filesystem": FilesystemWorker,
        "speaker": SpeakerWorker,
        "display": DisplayWorker,
    }
    for worker_id, expected_type in cases.items():
        worker = load_worker_process(
            worker_id=worker_id,
            config_path=_worker_config(tmp_path, worker_id),
            registry_path=registry_path,
            queue_dir=tmp_path / "queue" / worker_id,
        )

        assert isinstance(worker, expected_type)
        assert worker.worker == worker_id


def test_load_worker_process_rejects_worker_not_enabled_in_registry(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not enabled"):
        load_worker_process(
            worker_id="speaker",
            config_path=_worker_config(tmp_path, "speaker"),
            registry_path=_registry(tmp_path, "display"),
            queue_dir=tmp_path / "queue",
        )


def test_load_worker_process_rejects_cli_id_that_does_not_match_config(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="does not match config"):
        load_worker_process(
            worker_id="display",
            config_path=_worker_config(tmp_path, "speaker"),
            registry_path=_registry(tmp_path, "speaker", "display"),
            queue_dir=tmp_path / "queue",
        )


def test_main_exposes_console_entrypoint_arguments_for_one_shot_validation(
    tmp_path: Path,
) -> None:
    config_path = _worker_config(tmp_path, "display")
    registry_path = _registry(tmp_path, "display")

    status = main(
        [
            "display",
            "--config",
            str(config_path),
            "--registry",
            str(registry_path),
            "--queue-dir",
            str(tmp_path / "queue"),
            "--once",
        ]
    )

    assert status == 0


def test_main_runs_worker_until_stopped_by_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = _worker_config(tmp_path, "display")
    registry_path = _registry(tmp_path, "display")
    started: list[str] = []

    def run_until_stopped(worker: object) -> None:
        started.append(worker.worker)  # type: ignore[attr-defined]

    monkeypatch.setattr(worker_cli, "run_worker_forever", run_until_stopped)

    status = worker_cli.main(
        [
            "display",
            "--config",
            str(config_path),
            "--registry",
            str(registry_path),
            "--queue-dir",
            str(tmp_path / "queue"),
        ]
    )

    assert status == 0
    assert started == ["display"]


def test_run_worker_forever_polls_sensors_before_sleeping(tmp_path: Path) -> None:
    worker = load_worker_process(
        worker_id="mic",
        config_path=_worker_config(tmp_path, "mic"),
        registry_path=_registry(tmp_path, "mic"),
        queue_dir=tmp_path / "queue",
    )
    polls = 0

    def poll_once() -> tuple[object, ...]:
        nonlocal polls
        polls += 1
        return ()

    class StopLoop(Exception):
        pass

    def stop_after_first_sleep(_: float) -> None:
        raise StopLoop

    worker.poll_once = poll_once  # type: ignore[method-assign]

    with pytest.raises(StopLoop):
        run_worker_forever(worker, sleep=stop_after_first_sleep)

    assert polls == 1


def _registry(tmp_path: Path, *enabled: str) -> Path:
    path = tmp_path / "registry.yaml"
    path.write_text(
        "workers:\n  enabled: [" + ", ".join(enabled) + "]\n",
        encoding="utf-8",
    )
    return path


def _worker_config(tmp_path: Path, worker_id: str) -> Path:
    path = tmp_path / f"{worker_id}.yaml"
    path.write_text(_worker_config_yaml(worker_id), encoding="utf-8")
    return path


def _worker_config_yaml(worker_id: str) -> str:
    configs = {
        "mic": """
worker: mic
role: sensor
device_service: audio-input
backend: {kind: deterministic, placement: local}
sampling: {vad: true}
emits: [speech.segment]
verbs: []
""",
        "camera": """
worker: camera
role: sensor
device_service: camera-input
backend: {kind: deterministic, placement: local}
emits: [scene.caption, scene.motion]
verbs: [snapshot]
""",
        "filesystem": f"""
worker: filesystem
role: sensor
backend: {{kind: deterministic, placement: local}}
sampling:
  paths:
    - '{Path.cwd()}'
emits: [file.changed, file.created, file.deleted]
verbs: []
""",
        "speaker": """
worker: speaker
role: effector
backend: {kind: deterministic, placement: local}
emits: []
verbs: [speak]
""",
        "display": """
worker: display
role: effector
backend: {kind: deterministic, placement: local}
emits: []
verbs: [render, clear]
""",
    }
    return configs[worker_id]
