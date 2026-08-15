"""The narrow memory-provider seam used by the wake briefing.

Hindsight is the external memory provider (it owns Hermes' single
memory-provider slot), but its client library is not importable in this
repository. Defining the seam as a :class:`Protocol` keeps every piece of
briefing logic pure and testable against a fake, and confines the
Hindsight-specific details to :mod:`aila.briefing.hindsight`.

Design note -- why the two halves are different shapes:

* :meth:`MemoryProvider.recent_episodes` is **deterministic and recency
  ordered**. "What did I do last session and what was I in the middle of?" is
  a recency question, not a similarity question; semantic ranking cannot answer
  it reliably. This is the actual previous-session briefing, and it returns
  exact :class:`Episode` records.
* :meth:`MemoryProvider.recall_facts` is the associative channel. It returns
  :class:`MemoryFact` prose, **not** episodes: a semantic store like Hindsight
  consumes what you give it and returns its own synthesized knowledge, so the
  original records cannot be recovered. Verified on the host -- retained JSON
  came back as LLM-written sentences.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from aila.briefing.models import Episode, MemoryFact


@runtime_checkable
class MemoryProvider(Protocol):
    """Cross-session episodic storage and retrieval."""

    def retain_episode(self, episode: Episode) -> None:
        """Durably store ``episode`` for recall by later wakes."""

    def retain_text(
        self,
        text: str,
        *,
        document_id: str | None = None,
        timestamp: datetime | None = None,
    ) -> None:
        """Durably store free prose for recall by later wakes.

        Separate from :meth:`retain_episode` because the two carry different
        things: an :class:`Episode` is a *curated* record the agent authored,
        while ``text`` is raw material (a wake transcript) handed to the store
        for its own extraction.

        ``document_id`` makes the write **idempotent**: retaining twice under
        one id replaces the stored document rather than adding a second copy.
        The session-end hook can fire more than once per wake, so without it a
        wake is recorded twice.

        ``timestamp`` is when the wake actually happened, defaulting to now. It
        matters when backfilling history: stamping past wakes with the present
        would collapse them into one instant and destroy the temporal ordering
        the store builds its links from.
        """

    def recent_episodes(self, *, limit: int) -> tuple[Episode, ...]:
        """Return up to ``limit`` episodes, most recent first.

        Ordering is by ``ended_ts`` and must not be influenced by semantic
        relevance.
        """

    def recall_facts(self, query: str, *, limit: int) -> tuple[MemoryFact, ...]:
        """Return up to ``limit`` synthesized facts related to ``query``."""

    def flush(self) -> None:
        """Release resources and make pending writes durable.

        A cron wake can terminate immediately after the session ends, so any
        buffered write that has not been flushed is silently lost -- which is
        exactly the memory we most wanted to keep.
        """


class NullMemoryProvider:
    """A provider that stores nothing and recalls nothing.

    Used when no memory provider is configured, so the briefing degrades to the
    filesystem recency channel instead of raising during a wake.
    """

    def retain_episode(self, episode: Episode) -> None:
        return None

    def retain_text(
        self,
        text: str,
        *,
        document_id: str | None = None,
        timestamp: datetime | None = None,
    ) -> None:
        return None

    def recent_episodes(self, *, limit: int) -> tuple[Episode, ...]:
        return ()

    def recall_facts(self, query: str, *, limit: int) -> tuple[MemoryFact, ...]:
        return ()

    def flush(self) -> None:
        return None
