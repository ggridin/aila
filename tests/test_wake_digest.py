from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from aila.contracts import Observation
from aila.queue import ArchiveRetention, ObservationQueue
from aila.wake import build_sensory_digest


def test_build_sensory_digest_groups_observations_newest_first_and_archives(
    tmp_path: Path,
) -> None:
    # Disable age-based pruning so archived fixtures survive regardless of the
    # wall clock (build_sensory_digest drains, which enforces retention).
    queue = ObservationQueue(tmp_path, retention=ArchiveRetention(max_age=None))
    older_speech = _speech_observation(
        "speech-1",
        datetime(2026, 7, 13, 12, 0, 1, tzinfo=UTC),
        "open the pod bay doors",
    )
    newer_speech = _speech_observation(
        "speech-2",
        datetime(2026, 7, 13, 12, 0, 3, tzinfo=UTC),
        "never mind",
    )
    caption = _caption_observation(
        "caption-1",
        datetime(2026, 7, 13, 12, 0, 2, tzinfo=UTC),
        "person at desk",
    )

    for observation in (older_speech, caption, newer_speech):
        queue.append(observation)

    digest = build_sensory_digest(queue)

    assert digest.total_observations == 3
    assert list(digest.by_worker) == ["mic", "camera"]
    assert [item.obs_id for item in digest.by_worker["mic"]] == ["speech-2", "speech-1"]
    assert digest.by_worker["mic"][0].payload == {
        "text": "never mind",
        "lang": "en",
        "confidence": 0.9,
        "start_ms": 0,
        "end_ms": 10,
    }
    assert [item.obs_id for item in digest.by_worker["camera"]] == ["caption-1"]
    assert not list((tmp_path / "pending").glob("*.json"))
    assert not list((tmp_path / "inflight").glob("*.json"))
    assert sorted(path.name for path in (tmp_path / "archive").glob("*.json")) == [
        "20260713T120001.000000Z__speech-1.json",
        "20260713T120002.000000Z__caption-1.json",
        "20260713T120003.000000Z__speech-2.json",
    ]


def test_build_sensory_digest_omits_media_references_and_preserves_derived_records(
    tmp_path: Path,
) -> None:
    queue = ObservationQueue(tmp_path)
    queue.append(
        _speech_observation(
            "speech-with-media",
            datetime(2026, 7, 13, 12, 0, 1, tzinfo=UTC),
            "derived transcript only",
            media_ref="file:///tmp/raw-audio.wav",
        )
    )
    queue.append(
        _file_observation(
            "file-1",
            datetime(2026, 7, 13, 12, 0, 2, tzinfo=UTC),
        )
    )

    digest = build_sensory_digest(queue)
    serialized = digest.model_dump_json()

    assert "media_ref" not in serialized
    assert "raw-audio.wav" not in serialized
    assert digest.by_worker["filesystem"][0].payload == {
        "path": "C:\\Users\\Grego\\notes.txt",
        "change": "changed",
        "size": 12,
        "mtime": "2026-07-13T12:00:00Z",
    }


def test_build_sensory_digest_honors_batch_size(tmp_path: Path) -> None:
    queue = ObservationQueue(tmp_path)
    queue.append(_speech_observation("speech-1", datetime(2026, 7, 13, 12, 0, 1, tzinfo=UTC), "one"))
    queue.append(_speech_observation("speech-2", datetime(2026, 7, 13, 12, 0, 2, tzinfo=UTC), "two"))

    digest = build_sensory_digest(queue, batch_size=1)

    assert digest.total_observations == 1
    assert [item.obs_id for item in digest.by_worker["mic"]] == ["speech-1"]
    assert [path.name for path in (tmp_path / "pending").glob("*.json")] == [
        "20260713T120002.000000Z__speech-2.json"
    ]


def _speech_observation(
    obs_id: str,
    ts: datetime,
    text: str,
    *,
    media_ref: str | None = None,
) -> Observation:
    return Observation(
        obs_id=obs_id,
        worker="mic",
        kind="speech.segment",
        ts=ts,
        payload={
            "text": text,
            "lang": "en",
            "confidence": 0.9,
            "start_ms": 0,
            "end_ms": 10,
        },
        media_ref=media_ref,
    )


def _caption_observation(obs_id: str, ts: datetime, caption: str) -> Observation:
    return Observation(
        obs_id=obs_id,
        worker="camera",
        kind="scene.caption",
        ts=ts,
        payload={"caption": caption, "labels": ["person", "desk"], "boxes": []},
    )


def _file_observation(obs_id: str, ts: datetime) -> Observation:
    return Observation(
        obs_id=obs_id,
        worker="filesystem",
        kind="file.changed",
        ts=ts,
        payload={
            "path": "C:\\Users\\Grego\\notes.txt",
            "change": "changed",
            "size": 12,
            "mtime": "2026-07-13T12:00:00Z",
        },
    )
