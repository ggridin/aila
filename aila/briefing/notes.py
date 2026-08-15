"""Daily-note mirror of retained episodes, and the parser that reads it back.

Daily notes under ``aila-home/memory/YYYY-MM-DD.md`` are the agent's own
narrative and the *primary* recency source for the wake briefing (verified on
the host: Hindsight has never retained anything, while daily notes exist).

Two shapes are parsed:

* **Structured** -- ``## Session HH:MM-HH:MM UTC (episode_id)`` sections written
  by :func:`render_note_entry`. These round-trip losslessly apart from
  ``session_id``.
* **Free-form** -- notes the agent wrote by hand, which have no session
  sections at all. These degrade to a single whole-file episode per day, marked
  with :data:`FREEFORM_PREFIX` so they can never be mistaken for a real one.

Appends are performed via write-temp-fsync-replace so a crash mid-write cannot
truncate an existing note.
"""

from __future__ import annotations

import os
import re
import uuid
from datetime import UTC, datetime, time
from pathlib import Path

from aila.briefing.models import MAX_SUMMARY_CHARS, Episode

# Episode ids synthesized from unstructured notes carry this prefix.
FREEFORM_PREFIX = "note:"
# session_id recorded for episodes reconstructed from disk (never written out).
PARSED_SESSION_ID = "from-note"

_SECTION_RE = re.compile(
    r"^##\s+Session\s+(?P<start>\d{2}:\d{2})-(?P<end>\d{2}:\d{2})\s+UTC\s+\((?P<id>[^)]+)\)\s*$"
)
_DATE_RE = re.compile(r"^(?P<date>\d{4}-\d{2}-\d{2})\.md$")


def daily_note_path(memory_dir: Path, episode: Episode) -> Path:
    """Return the daily note a given episode belongs to (UTC date of its end)."""

    return Path(memory_dir) / f"{episode.ended_ts.date().isoformat()}.md"


def render_note_entry(episode: Episode) -> str:
    """Render an episode as a markdown section for the daily note."""

    started = episode.started_ts.strftime("%H:%M")
    ended = episode.ended_ts.strftime("%H:%M")
    lines = [f"## Session {started}-{ended} UTC ({episode.episode_id})", ""]

    if episode.summary.strip():
        lines += [episode.summary.strip(), ""]

    if episode.decisions:
        lines.append("**Decisions**")
        lines += [f"- {item}" for item in episode.decisions]
        lines.append("")

    if episode.open_loops:
        lines.append("**Open loops**")
        lines += [f"- {item}" for item in episode.open_loops]
        lines.append("")

    if episode.entities:
        # Written so the note round-trips losslessly back into an Episode;
        # entities seed the next wake's semantic recall query.
        lines.append(f"**Entities:** {', '.join(episode.entities)}")
        lines.append("")

    return "\n".join(lines)


def append_episode_note(memory_dir: Path, episode: Episode) -> Path:
    """Append ``episode`` to its daily note and return the note path."""

    target = daily_note_path(memory_dir, episode)
    target.parent.mkdir(parents=True, exist_ok=True)

    existing = target.read_text(encoding="utf-8") if target.is_file() else ""
    if existing and not existing.endswith("\n"):
        existing += "\n"
    _atomic_write(target, existing + render_note_entry(episode))
    return target


def _atomic_write(target: Path, text: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.parent / f".{target.name}.{uuid.uuid4().hex}.tmp"
    try:
        temp.write_text(text, encoding="utf-8")
        with temp.open("r+b") as handle:
            os.fsync(handle.fileno())
        temp.replace(target)
    finally:
        if temp.exists():
            temp.unlink()


# -- parsing -----------------------------------------------------------------


def note_date(path: Path) -> datetime | None:
    """Return the UTC date encoded in a ``YYYY-MM-DD.md`` filename."""

    match = _DATE_RE.match(Path(path).name)
    if match is None:
        return None
    try:
        return datetime.fromisoformat(match.group("date")).replace(tzinfo=UTC)
    except ValueError:
        return None


def parse_note(path: Path) -> tuple[Episode, ...]:
    """Parse a daily note into episodes, newest section last.

    Returns ``()`` for files that are not dated notes or cannot be read.
    """

    day = note_date(path)
    if day is None:
        return ()
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError:
        return ()
    if not text.strip():
        return ()

    episodes = _parse_sections(text, day)
    if episodes:
        return episodes
    return _freeform_episode(text, day)


def _parse_sections(text: str, day: datetime) -> tuple[Episode, ...]:
    """Extract ``## Session ...`` sections written by :func:`render_note_entry`."""

    lines = text.splitlines()
    starts = [i for i, line in enumerate(lines) if _SECTION_RE.match(line)]
    if not starts:
        return ()

    bounds = starts + [len(lines)]
    episodes: list[Episode] = []
    for index, start in enumerate(starts):
        header = _SECTION_RE.match(lines[start])
        assert header is not None  # guaranteed by the scan above
        body = lines[start + 1 : bounds[index + 1]]
        episode = _episode_from_section(header, body, day)
        if episode is not None:
            episodes.append(episode)
    return tuple(episodes)


def _episode_from_section(header: re.Match[str], body: list[str], day: datetime) -> Episode | None:
    summary_lines: list[str] = []
    decisions: list[str] = []
    open_loops: list[str] = []
    entities: list[str] = []
    bucket: str | None = None

    for raw in body:
        line = raw.strip()
        if not line:
            continue
        if line == "**Decisions**":
            bucket = "decisions"
            continue
        if line == "**Open loops**":
            bucket = "open_loops"
            continue
        if line.startswith("**Entities:**"):
            entities = [item.strip() for item in line[len("**Entities:**") :].split(",")]
            bucket = None
            continue
        if line.startswith("- ") and bucket == "decisions":
            decisions.append(line[2:])
            continue
        if line.startswith("- ") and bucket == "open_loops":
            open_loops.append(line[2:])
            continue
        if bucket is None:
            summary_lines.append(line)

    try:
        return Episode(
            episode_id=header.group("id"),
            session_id=PARSED_SESSION_ID,
            started_ts=_at(day, header.group("start")),
            ended_ts=_at(day, header.group("end")),
            summary=" ".join(summary_lines)[:MAX_SUMMARY_CHARS],
            decisions=tuple(decisions),
            open_loops=tuple(open_loops),
            entities=tuple(entities),
        )
    except ValueError:
        return None


def _freeform_episode(text: str, day: datetime) -> tuple[Episode, ...]:
    """Degrade a hand-written note into one low-fidelity episode for the day.

    No episode id, decisions or open loops can be recovered, so the summary is
    the leading prose and the id is marked with :data:`FREEFORM_PREFIX`.
    """

    summary = " ".join(
        line.strip().lstrip("#").strip()
        for line in text.splitlines()
        if line.strip() and not line.strip().startswith("**")
    ).strip()
    if not summary:
        return ()

    date_key = day.date().isoformat()
    try:
        return (
            Episode(
                episode_id=f"{FREEFORM_PREFIX}{date_key}",
                session_id=PARSED_SESSION_ID,
                started_ts=day,
                # End of day so a free-form note sorts after timed sections.
                ended_ts=datetime.combine(day.date(), time(23, 59), tzinfo=UTC),
                summary=summary[:MAX_SUMMARY_CHARS],
            ),
        )
    except ValueError:
        return ()


def _at(day: datetime, hhmm: str) -> datetime:
    hour, minute = (int(part) for part in hhmm.split(":"))
    return datetime.combine(day.date(), time(hour, minute), tzinfo=UTC)
