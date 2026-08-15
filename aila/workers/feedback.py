from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any


_WORD_RE = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class PlaybackReference:
    text: str
    normalized_text: str
    started_at: datetime
    expected_until: datetime
    tail_until: datetime
    duration_ms: int
    command_id: str | None = None
    backend: str | None = None
    model: str | None = None

    def is_active(self, *, now: datetime | None = None) -> bool:
        current = now or datetime.now(UTC)
        return current <= self.tail_until


def normalize_transcript(text: str) -> str:
    return " ".join(_WORD_RE.findall(text.casefold()))


def transcript_similarity(left: str, right: str) -> float:
    normalized_left = normalize_transcript(left)
    normalized_right = normalize_transcript(right)
    if not normalized_left or not normalized_right:
        return 0.0
    if normalized_left in normalized_right or normalized_right in normalized_left:
        return 1.0
    return SequenceMatcher(None, normalized_left, normalized_right).ratio()


def should_suppress_echo(
    transcript: str,
    reference: PlaybackReference | None,
    *,
    threshold: float,
    now: datetime | None = None,
) -> bool:
    if reference is None or not reference.is_active(now=now):
        return False
    return transcript_similarity(transcript, reference.normalized_text) >= threshold


def write_playback_reference(
    path: Path,
    *,
    text: str,
    duration_ms: int,
    tail_ms: int,
    command_id: str | None = None,
    backend: str | None = None,
    model: str | None = None,
    now: datetime | None = None,
) -> PlaybackReference:
    started_at = now or datetime.now(UTC)
    expected_until = started_at + timedelta(milliseconds=max(0, duration_ms))
    tail_until = expected_until + timedelta(milliseconds=max(0, tail_ms))
    reference = PlaybackReference(
        text=text,
        normalized_text=normalize_transcript(text),
        started_at=started_at,
        expected_until=expected_until,
        tail_until=tail_until,
        duration_ms=max(0, duration_ms),
        command_id=command_id,
        backend=backend,
        model=model,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_reference_to_json(reference), indent=2) + "\n", encoding="utf-8")
    return reference


def read_playback_reference(path: Path) -> PlaybackReference | None:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    if not isinstance(raw, dict):
        return None
    try:
        text = str(raw.get("text") or "")
        normalized_text = str(raw.get("normalized_text") or normalize_transcript(text))
        return PlaybackReference(
            text=text,
            normalized_text=normalized_text,
            started_at=_parse_datetime(raw["started_at"]),
            expected_until=_parse_datetime(raw["expected_until"]),
            tail_until=_parse_datetime(raw["tail_until"]),
            duration_ms=int(raw.get("duration_ms", 0)),
            command_id=_optional_str(raw.get("command_id")),
            backend=_optional_str(raw.get("backend")),
            model=_optional_str(raw.get("model")),
        )
    except (KeyError, TypeError, ValueError):
        return None


def playback_reference_active(path: Path, *, now: datetime | None = None) -> bool:
    reference = read_playback_reference(path)
    return reference is not None and reference.is_active(now=now)


def _reference_to_json(reference: PlaybackReference) -> dict[str, Any]:
    return {
        "text": reference.text,
        "normalized_text": reference.normalized_text,
        "started_at": _format_datetime(reference.started_at),
        "expected_until": _format_datetime(reference.expected_until),
        "tail_until": _format_datetime(reference.tail_until),
        "duration_ms": reference.duration_ms,
        "command_id": reference.command_id,
        "backend": reference.backend,
        "model": reference.model,
    }


def _format_datetime(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _parse_datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed.astimezone(UTC)


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    return str(value)
