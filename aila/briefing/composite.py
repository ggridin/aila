"""Composite provider: recency from disk, semantics from Hindsight.

This is the arrangement the host verification argued for. Each half is given
the job it can actually do:

* **Recency** -> the filesystem. Deterministic, dependency-free, and correct
  even though Hindsight has never run.
* **Semantics** -> Hindsight. Associative recall over older material, which is
  what a semantic store is genuinely good at.
* **Retention** -> Hindsight (the daily-note mirror is written separately by
  the plugin, so an episode survives even when Hindsight is inert).

:mod:`aila.briefing.compose` is unchanged: it still sees one
:class:`~aila.briefing.provider.MemoryProvider`.
"""

from __future__ import annotations

import logging

from aila.briefing.models import Episode, MemoryFact
from aila.briefing.provider import MemoryProvider

logger = logging.getLogger(__name__)


class CompositeMemoryProvider:
    """Route each half of the protocol to the provider that can serve it."""

    def __init__(self, *, recency: MemoryProvider, semantic: MemoryProvider) -> None:
        self._recency = recency
        self._semantic = semantic

    def retain_episode(self, episode: Episode) -> None:
        self._semantic.retain_episode(episode)

    def retain_text(
        self,
        text: str,
        *,
        document_id: str | None = None,
        timestamp: datetime | None = None,
    ) -> None:
        self._semantic.retain_text(text, document_id=document_id, timestamp=timestamp)

    def recent_episodes(self, *, limit: int) -> tuple[Episode, ...]:
        return self._recency.recent_episodes(limit=limit)

    def recall_facts(self, query: str, *, limit: int) -> tuple[MemoryFact, ...]:
        return self._semantic.recall_facts(query, limit=limit)

    def flush(self) -> None:
        self._semantic.flush()
