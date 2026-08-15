from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from aila.contracts import Observation
from aila.queue import ArchiveRetention, ObservationQueue


def test_append_writes_pending_file_atomically_named_by_timestamp_and_obs_id(tmp_path: Path) -> None:
    queue = ObservationQueue(tmp_path)
    observation = _speech_observation("obs-1", datetime(2026, 7, 13, 12, 0, 1, 234, tzinfo=UTC))

    path = queue.append(observation)

    assert path.parent == tmp_path / "pending"
    assert path.name == "20260713T120001.000234Z__obs-1.json"
    assert not list((tmp_path / "pending").glob("*.tmp"))
    assert Observation.model_validate_json(path.read_text(encoding="utf-8")) == observation


def test_drain_reclaims_inflight_and_archives_batch_in_chronological_order(tmp_path: Path) -> None:
    # Disable age-based pruning so archived fixtures survive regardless of the
    # wall clock (drain() runs enforce_retention() with the real "now").
    queue = ObservationQueue(tmp_path, retention=ArchiveRetention(max_age=None))
    newest = _speech_observation("obs-3", datetime(2026, 7, 13, 12, 0, 3, tzinfo=UTC))
    oldest = _speech_observation("obs-1", datetime(2026, 7, 13, 12, 0, 1, tzinfo=UTC))
    stuck = _speech_observation("obs-2", datetime(2026, 7, 13, 12, 0, 2, tzinfo=UTC))

    queue.append(newest)
    queue.append(oldest)
    stuck_path = queue.append(stuck)
    stuck_path.replace(tmp_path / "inflight" / stuck_path.name)

    drained = queue.drain(batch_size=2)

    assert [item.observation.obs_id for item in drained] == ["obs-1", "obs-2"]
    assert not list((tmp_path / "inflight").glob("*.json"))
    assert sorted(path.name for path in (tmp_path / "archive").glob("*.json")) == [
        "20260713T120001.000000Z__obs-1.json",
        "20260713T120002.000000Z__obs-2.json",
    ]
    assert [path.name for path in (tmp_path / "pending").glob("*.json")] == [
        "20260713T120003.000000Z__obs-3.json"
    ]


def test_archive_retention_prunes_by_age_and_size_budget(tmp_path: Path) -> None:
    queue = ObservationQueue(
        tmp_path,
        retention=ArchiveRetention(max_bytes=230, max_age=timedelta(days=2)),
    )
    # Anchor timestamps to "now" so age-based pruning is independent of the
    # machine clock (drain() applies retention using the real current time).
    now_ref = datetime.now(UTC).replace(microsecond=0)
    old = _speech_observation("old", now_ref - timedelta(days=3))
    first = _speech_observation("first", now_ref - timedelta(days=1))
    second = _speech_observation("second", now_ref)

    for observation in (old, first, second):
        queue.append(observation)
    queue.drain()
    queue.enforce_retention(now=now_ref)

    archived_ids = sorted(path.name.split("__", 1)[1] for path in (tmp_path / "archive").glob("*.json"))
    assert archived_ids == ["second.json"]


def _speech_observation(obs_id: str, ts: datetime) -> Observation:
    return Observation(
        obs_id=obs_id,
        worker="mic",
        kind="speech.segment",
        ts=ts,
        payload={
            "text": f"hello {obs_id}",
            "lang": "en",
            "confidence": 0.9,
            "start_ms": 0,
            "end_ms": 10,
        },
    )
