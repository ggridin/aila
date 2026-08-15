from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from aila.contracts import Subscription
from aila.device_services import load_device_service_config
from aila.installer import generate_body_contract_manifest, seed_tree_if_absent
from aila.queue import ArchiveRetention, ObservationQueue
from aila.registry import load_registry_config, validate_registry_files
from aila.subscriptions import load_subscriptions, matching_subscriptions
from aila.wake import build_sensory_digest
from aila.workers import load_worker_config
from aila.workers.camera import CameraFrame, CameraWorker
from aila.workers.filesystem import (
    FileChangeEvent,
    FilesystemWatchConfig,
    FilesystemWorker,
)
from aila.workers.mic import MicWorker, SpeechSegment

SEED_ROOT = Path(__file__).resolve().parents[1] / "workspace-seed"
WORKERS = ("mic", "camera", "filesystem", "speaker", "display")
DEVICE_SERVICES = ("audio-input", "camera-input")


@dataclass(frozen=True)
class FakeSpeechSegmentSource:
    segments: tuple[SpeechSegment, ...]

    def poll_segments(self) -> tuple[SpeechSegment, ...]:
        return self.segments


@dataclass(frozen=True)
class FakeCameraFrameSource:
    frames: tuple[CameraFrame, ...]
    snapshot: CameraFrame

    def poll_frames(self) -> tuple[CameraFrame, ...]:
        return self.frames

    def capture_snapshot(self) -> CameraFrame:
        return self.snapshot


@dataclass(frozen=True)
class FakeFileEventAdapter:
    events: tuple[FileChangeEvent, ...]

    def poll_events(self) -> tuple[FileChangeEvent, ...]:
        return self.events


def test_host_independent_core_flow_from_seeds_to_digest(tmp_path: Path) -> None:
    runtime_root = tmp_path / "runtime-home"
    body_root = runtime_root / "aila-body"
    existing_soul = runtime_root / "SOUL.md"
    existing_soul.parent.mkdir(parents=True)
    existing_soul.write_text("operator-authored soul\n", encoding="utf-8")

    first_seed = seed_tree_if_absent(SEED_ROOT, runtime_root)
    second_seed = seed_tree_if_absent(SEED_ROOT, runtime_root)

    assert existing_soul.read_text(encoding="utf-8") == "operator-authored soul\n"
    assert "SOUL.md" in _relative_paths(first_seed.skipped, runtime_root)
    assert "config.yaml" in _relative_paths(first_seed.created, runtime_root)
    assert second_seed.created == ()
    assert "aila-body/workers/mic/config.yaml" in _relative_paths(
        second_seed.skipped,
        runtime_root,
    )

    manifest = generate_body_contract_manifest(body_root)
    assert manifest["fleet"] == list(WORKERS)
    assert (body_root / "contracts" / "manifest.json").is_file()
    assert (body_root / "contracts" / "observation.speech.segment.payload.schema.json").is_file()

    registry = validate_registry_files(
        load_registry_config(runtime_root / "config.yaml"),
        workers_dir=body_root / "workers",
        device_services_dir=body_root / "device-services",
    )
    assert registry.enabled_workers == WORKERS
    assert registry.required_device_services == DEVICE_SERVICES

    worker_configs = {
        worker: load_worker_config(body_root / "workers" / worker / "config.yaml")
        for worker in WORKERS
    }
    device_services = {
        service: load_device_service_config(
            body_root / "device-services" / service / "config.yaml"
        )
        for service in DEVICE_SERVICES
    }
    seed_subscriptions = load_subscriptions(body_root / "subscriptions.yaml")
    assert all(subscription.on_match == "queue" for subscription in seed_subscriptions)

    queue = ObservationQueue(body_root / "queue", retention=ArchiveRetention(max_age=None))
    watched_root = tmp_path / "watched"
    base_ts = datetime(2026, 7, 13, 12, 0, tzinfo=UTC)

    mic = MicWorker(
        worker_configs["mic"],
        device_services["audio-input"],
        FakeSpeechSegmentSource(
            (
                _speech_segment("speech-1", "hello aila", base_ts, vad_active=True),
                _speech_segment(
                    "speech-muted",
                    "background noise",
                    base_ts + timedelta(milliseconds=100),
                    vad_active=False,
                ),
            )
        ),
        queue,
    )
    camera = CameraWorker(
        worker_configs["camera"],
        device_services["camera-input"],
        FakeCameraFrameSource(
            frames=(
                CameraFrame(
                    obs_id="frame-1",
                    ts=base_ts + timedelta(seconds=1),
                    caption="person at desk",
                    labels=("person",),
                    motion_level=0.4,
                ),
            ),
            snapshot=CameraFrame(caption="current scene", labels=("desk",)),
        ),
        queue,
    )
    filesystem = FilesystemWorker(
        worker_configs["filesystem"],
        FakeFileEventAdapter(
            (
                FileChangeEvent(
                    path=watched_root / "notes.txt",
                    change="changed",
                    size=42,
                    mtime=base_ts + timedelta(seconds=2),
                ),
            )
        ),
        queue,
        FilesystemWatchConfig(
            paths=(watched_root,),
            ignore=tuple(str(pattern) for pattern in worker_configs["filesystem"].sampling["ignore"]),
            debounce_ms=int(worker_configs["filesystem"].sampling["debounce_ms"]),
        ),
    )

    observations = (
        *mic.poll_once(),
        *camera.poll_once(),
        *filesystem.poll_once(),
    )

    assert [observation.kind for observation in observations] == [
        "speech.segment",
        "scene.caption",
        "scene.motion",
        "file.changed",
    ]
    assert [path.name for path in (body_root / "queue" / "pending").glob("*.json")]

    runtime_subscriptions = (
        *seed_subscriptions,
        Subscription(
            worker="filesystem",
            kind="file.changed",
            predicate={"path~": str(watched_root / "**")},
        ),
    )
    matches = {
        observation.kind: matching_subscriptions(runtime_subscriptions, observation)
        for observation in observations
    }
    assert [(match.worker, match.kind) for match in matches["speech.segment"]] == [
        ("mic", "speech.segment")
    ]
    assert [(match.worker, match.kind) for match in matches["scene.caption"]] == [
        ("camera", "scene.caption")
    ]
    assert [(match.worker, match.kind) for match in matches["file.changed"]] == [
        ("filesystem", "file.changed")
    ]

    digest = build_sensory_digest(queue)

    assert digest.total_observations == 4
    assert digest.by_worker["mic"][0].payload["text"] == "hello aila"
    assert {item.kind for item in digest.by_worker["camera"]} == {
        "scene.caption",
        "scene.motion",
    }
    assert digest.by_worker["filesystem"][0].payload["size"] == 42
    assert not list((body_root / "queue" / "pending").glob("*.json"))
    assert not list((body_root / "queue" / "inflight").glob("*.json"))
    assert len(list((body_root / "queue" / "archive").glob("*.json"))) == 4


def _speech_segment(
    obs_id: str,
    text: str,
    ts: datetime,
    *,
    vad_active: bool,
) -> SpeechSegment:
    return SpeechSegment(
        obs_id=obs_id,
        ts=ts,
        text=text,
        lang="en",
        confidence=0.95,
        start_ms=0,
        end_ms=500,
        vad_active=vad_active,
        raw_audio=f"raw-audio-{obs_id}".encode("utf-8"),
    )


def _relative_paths(paths: tuple[Path, ...], root: Path) -> set[str]:
    return {path.relative_to(root).as_posix() for path in paths}
