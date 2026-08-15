from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from aila.workers.feedback import (
    normalize_transcript,
    read_playback_reference,
    should_suppress_echo,
    transcript_similarity,
    write_playback_reference,
)


def test_playback_reference_round_trips_and_expires(tmp_path: Path) -> None:
    now = datetime(2026, 1, 1, tzinfo=UTC)
    path = tmp_path / "speaker-playback.json"

    written = write_playback_reference(
        path,
        text="Hello, AILA!",
        duration_ms=1000,
        tail_ms=750,
        backend="piper-cli",
        model="piper-en",
        now=now,
    )
    read = read_playback_reference(path)

    assert read == written
    assert read is not None
    assert read.is_active(now=now + timedelta(milliseconds=1749)) is True
    assert read.is_active(now=now + timedelta(milliseconds=1751)) is False


def test_similarity_suppresses_echo_but_keeps_barge_in(tmp_path: Path) -> None:
    path = tmp_path / "speaker-playback.json"
    now = datetime(2026, 1, 1, tzinfo=UTC)
    reference = write_playback_reference(
        path,
        text="The weather is sunny today.",
        duration_ms=2000,
        tail_ms=750,
        now=now,
    )

    assert should_suppress_echo("weather is sunny today", reference, threshold=0.82, now=now) is True
    assert should_suppress_echo("wait stop talking", reference, threshold=0.82, now=now) is False


def test_missing_or_corrupt_reference_does_not_suppress(tmp_path: Path) -> None:
    missing = tmp_path / "missing.json"
    corrupt = tmp_path / "corrupt.json"
    corrupt.write_text("not-json", encoding="utf-8")

    assert read_playback_reference(missing) is None
    assert read_playback_reference(corrupt) is None
    assert should_suppress_echo("anything", read_playback_reference(corrupt), threshold=0.82) is False


def test_transcript_normalization_and_similarity() -> None:
    assert normalize_transcript("Hello, AILA!!!") == "hello aila"
    assert transcript_similarity("Hello AILA", "hello, aila") == 1.0
