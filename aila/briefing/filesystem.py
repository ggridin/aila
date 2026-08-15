"""Filesystem-backed recency source for the wake briefing.

Verified on the host: Hindsight has never retained anything (no datastore, zero
log lines), while ``aila-home/memory/YYYY-MM-DD.md`` daily notes exist and are
actively written by the agent. Reading those notes therefore gives deterministic
"what happened recently" with **no external dependency** -- no service, no
embeddings, no ranking.

This is what makes the briefing answer a *recency* question properly: files are
days, so a `days` window is the natural bound, and ISO filenames sort
chronologically.

Semantic search is deliberately NOT implemented here; that belongs to Hindsight
via :mod:`aila.briefing.composite`.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path

from aila.briefing.models import Episode, MemoryFact
from aila.briefing.notes import note_date, parse_note

logger = logging.getLogger(__name__)

DEFAULT_DAYS = 2


class FilesystemMemoryProvider:
    """Recency from daily notes on disk.

    ``days`` bounds how many recent note *files* are considered (default 2,
    matching the two-day window this replaces); ``recent_episodes(limit=...)``
    then bounds how many episodes survive into the briefing.
    """

    def __init__(self, memory_dir: Path, *, days: int = DEFAULT_DAYS) -> None:
        self._memory_dir = Path(memory_dir)
        self._days = max(1, days)

    def recent_episodes(self, *, limit: int) -> tuple[Episode, ...]:
        episodes: list[Episode] = []
        for path in self._recent_notes():
            episodes.extend(parse_note(path))
        episodes.sort(key=lambda item: item.ended_ts, reverse=True)
        return tuple(episodes[:limit])

    # -- MemoryProvider: unsupported halves ---------------------------------

    def retain_episode(self, episode: Episode) -> None:
        """No-op: the daily-note mirror is written by the plugin already."""
        return None

    def retain_text(
        self,
        text: str,
        *,
        document_id: str | None = None,
        timestamp: datetime | None = None,
    ) -> None:
        """No-op: raw prose has no home on disk; it is for the semantic store."""
        return None

    def recall_facts(self, query: str, *, limit: int) -> tuple[MemoryFact, ...]:
        """No-op: semantic recall is Hindsight's job, not the filesystem's."""
        return ()

    def flush(self) -> None:
        return None

    # -- internals -----------------------------------------------------------

    def _recent_notes(self) -> list[Path]:
        """Return dated notes within the window, newest first."""

        try:
            candidates = list(self._memory_dir.glob("*.md"))
        except OSError:
            logger.warning("could not list daily notes in %s", self._memory_dir, exc_info=True)
            return []

        cutoff = datetime.now(UTC) - timedelta(days=self._days)
        dated: list[tuple[datetime, Path]] = []
        for path in candidates:
            day = note_date(path)
            # A note dated today has day == midnight, so compare on date.
            if day is None or day.date() < cutoff.date():
                continue
            dated.append((day, path))

        dated.sort(key=lambda item: item[0], reverse=True)
        return [path for _, path in dated]
