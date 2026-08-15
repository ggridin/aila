"""AILA wake-briefing plugin for the Hermes agent runtime.

This is a *general* Hermes plugin: it consumes the memory provider through the
narrow :class:`~aila.briefing.provider.MemoryProvider` seam rather than
registering as one, so Hermes' single external memory-provider slot stays with
Hindsight.

* ``pre_llm_call`` hook -> injects the fenced session briefing on the first turn
  of a wake (inject-once), before the reflex digest.
* ``on_session_end`` hook -> builds the episode for the finished wake, retains
  it through the provider, mirrors it to the daily note, and flushes.

Layering note: LCM (``context.engine: lcm``) is a *within-session* context
compressor and cannot carry anything across wakes. It is downstream of this
plugin -- it compresses the injected briefing as the session grows.

The Hermes plugin API is accessed purely by duck typing so this module imports
cleanly in a host-independent test environment.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from aila.briefing.compose import build_briefing
from aila.briefing.models import MAX_SUMMARY_CHARS, Episode
from aila.briefing.notes import append_episode_note
from aila.briefing.provider import MemoryProvider, NullMemoryProvider
from aila.briefing.session_log import session_messages
from aila.briefing.transcript import render_transcript

logger = logging.getLogger(__name__)

BRIEFING_TOOLSET = "briefing"
RECORD_EPISODE_TOOL = "record_episode"
RECALL_MEMORY_TOOL = "recall_memory"

# Plugin-registered tools do not appear in the *static* tool list on this
# Hermes build: the `briefing` toolset registers and resolves
# (`resolve_toolset("briefing") == [...]`, and `registry.get_definitions(...)`
# returns the schema), yet `get_tool_definitions()` omits it -- even when
# `enabled_toolsets=["briefing"]` is passed explicitly.
#
# They ARE reachable, however, through Hermes' dynamic `tool_search` ->
# `tool_call` indirection. Verified in the host logs: the agent discovered and
# invoked `record_episode` that way (2026-08-12 22:13:01, "tool record_episode
# completed"), after two failed attempts that omitted the required `summary`.
#
# Two consequences shape the schemas below:
#
# * Keep `required` minimal -- every required argument is one more thing the
#   model can omit when calling blind through `tool_call`.
# * Make `description` self-contained, since it is what `tool_search` matches
#   on and often the only thing the model sees before calling.

_RECORD_SCHEMA: dict[str, Any] = {
    "name": RECORD_EPISODE_TOOL,
    "description": (
        "Record what this wake accomplished so your next wake inherits it. "
        "Call this once, before you sleep. 'open_loops' is the most valuable "
        "field: whatever you list there is what your next self will be told to "
        "pick up. Calling it again replaces this wake's record."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "summary": {
                "type": "string",
                "description": "What you did this wake, in a few sentences.",
            },
            "decisions": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Choices you made that a later wake should not re-litigate.",
            },
            "open_loops": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "What you started but did not finish, and what you would "
                    "want to pick up next. These carry your intent forward."
                ),
            },
            "entities": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Topics, files or components involved, for later recall.",
            },
        },
        "required": ["summary"],
    },
}

# How many facts a single recall returns when the model does not say.
DEFAULT_RECALL_LIMIT = 5

# Upper bound on `limit`. Recall results are prose facts that land straight in
# the context window, and the wake already carries a briefing plus the reflex
# digest; an unbounded limit would let one tool call crowd out the session.
MAX_RECALL_LIMIT = 10

_RECALL_SCHEMA: dict[str, Any] = {
    "name": RECALL_MEMORY_TOOL,
    "description": (
        "Search your own long-term memory of past wakes. Use this when you "
        "need something a previous self knew and the session briefing does "
        "not cover it -- an earlier decision, an attempt already made, or why "
        "something is the way it is. Ask in plain language, the way you would "
        "ask a colleague; matching is semantic, not keyword."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "What you want to remember, in plain language. "
                    "E.g. 'what did I decide about the camera worker?'"
                ),
            },
            "limit": {
                "type": "integer",
                "description": (
                    f"How many facts to return (1-{MAX_RECALL_LIMIT}, "
                    f"default {DEFAULT_RECALL_LIMIT})."
                ),
            },
        },
        "required": ["query"],
    },
}


def _hermes_home() -> Path:
    return Path(os.environ.get("HERMES_HOME") or (Path.home() / ".hermes")).expanduser()


def default_memory_dir() -> Path:
    return _hermes_home() / "aila-home" / "memory"


def default_provider(memory_dir: Path | None = None) -> MemoryProvider:
    """Build the composite provider: recency from disk, semantics from Hindsight.

    Recency deliberately does not depend on Hindsight -- on the host it has
    never retained anything, while daily notes exist and are actively written.
    """

    from aila.briefing.composite import CompositeMemoryProvider
    from aila.briefing.filesystem import FilesystemMemoryProvider
    from aila.briefing.hindsight import HindsightMemoryProvider

    recency = FilesystemMemoryProvider(memory_dir or default_memory_dir())

    hindsight = HindsightMemoryProvider()
    semantic: MemoryProvider = hindsight if hindsight.available else NullMemoryProvider()

    return CompositeMemoryProvider(recency=recency, semantic=semantic)


def _as_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        stripped = value.strip()
        return (stripped,) if stripped else ()
    try:
        return tuple(str(item) for item in value)
    except TypeError:
        return ()


def _as_limit(value: Any) -> int:
    """Coerce a model-supplied ``limit`` into the allowed range.

    Never raises: the model reaches this tool through ``tool_call``, where it
    routinely supplies the wrong type (or a string), and a rejected call would
    cost it a turn for no reason. Junk falls back to the default.
    """

    try:
        limit = int(value)
    except (TypeError, ValueError):
        return DEFAULT_RECALL_LIMIT
    return max(1, min(limit, MAX_RECALL_LIMIT))


def _as_datetime(value: Any, *, default: datetime) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return default
    return default


class BriefingPlugin:
    """Adapter binding the wake briefing to Hermes plugin callbacks."""

    def __init__(
        self,
        provider: MemoryProvider | None = None,
        *,
        memory_dir: Path | None = None,
    ) -> None:
        self._memory_dir = Path(memory_dir) if memory_dir is not None else default_memory_dir()
        # The provider must read the same directory the mirror writes to.
        self._provider = provider if provider is not None else default_provider(self._memory_dir)
        # Sessions whose episode the agent already recorded via the tool, so
        # on_session_end does not write a second, poorer one for the same wake.
        self._recorded_sessions: set[str] = set()

    # -- tool: record_episode ------------------------------------------------

    def record_episode(self, args: dict[str, Any], **kwargs: Any) -> str:
        """Record this wake's episode at the agent's own initiative."""

        args = args or {}
        session_id = str(kwargs.get("session_id") or args.get("session_id") or "unknown")
        episode = self.build_episode(
            session_id=session_id,
            summary=args.get("summary"),
            decisions=args.get("decisions"),
            open_loops=args.get("open_loops"),
            entities=args.get("entities"),
        )
        if episode is None:
            return json.dumps(
                {"ok": False, "error": "summary, decisions or open_loops must be non-empty"}
            )

        self._persist(episode)
        self._recorded_sessions.add(session_id)
        return json.dumps(
            {
                "ok": True,
                "episode_id": episode.episode_id,
                "open_loops": list(episode.open_loops),
            },
            ensure_ascii=False,
        )

    # -- tool: recall_memory -------------------------------------------------

    def recall_memory(self, args: dict[str, Any], **kwargs: Any) -> str:
        """Search long-term memory on the agent's own initiative.

        Read-only: recall never writes, so this adds no second retention path
        alongside :meth:`record_episode` / :meth:`on_session_end`.

        The provider applies the same tag and type filters the session briefing
        uses (see :mod:`aila.briefing.hindsight`), so the model searches the
        same memory it was briefed from rather than a wider, noisier set.
        """

        args = args or {}
        query = str(args.get("query") or "").strip()
        if not query:
            return json.dumps({"ok": False, "error": "query must be non-empty"})

        limit = _as_limit(args.get("limit"))

        try:
            facts = self._provider.recall_facts(query, limit=limit)
        except Exception:  # noqa: BLE001 - a memory failure must not break a wake
            logger.warning("recall_memory failed for query %r", query, exc_info=True)
            return json.dumps({"ok": False, "error": "memory search is unavailable"})

        logger.info("recall_memory: %d fact(s) for %r", len(facts), query)
        return json.dumps(
            {
                "ok": True,
                "query": query,
                # An empty list is a normal answer, not an error: it means
                # nothing was retained about this yet.
                "facts": [
                    {
                        "text": fact.text,
                        "type": fact.fact_type,
                        "when": fact.ts.isoformat() if fact.ts is not None else None,
                    }
                    for fact in facts
                ],
            },
            ensure_ascii=False,
        )

    # -- hook: pre_llm_call --------------------------------------------------

    def pre_llm_call(self, **kwargs: Any) -> dict[str, str] | None:
        """Return the session briefing for the first turn of a wake only.

        Note this hook cannot capture the wake's work: it fires once per
        ``run_conversation``, and its ``conversation_history`` is the state
        *before* the turn runs. The transcript is read from Hermes' session
        database at session end instead (see :mod:`aila.briefing.session_log`).
        """

        if not kwargs.get("is_first_turn", False):
            return None
        try:
            result = build_briefing(self._provider)
        except Exception:  # noqa: BLE001 - a briefing failure must not block the wake
            logger.warning("session briefing failed to build", exc_info=True)
            return None
        if result.is_empty:
            return None
        # Hermes injects hook context into the user message for the LLM call
        # only -- it is never persisted to the transcript, so this log line is
        # the only evidence a briefing actually reached the model.
        logger.info(
            "session briefing injected: %d chars, episodes=%s",
            len(result.block),
            result.episode_ids,
        )
        return {"context": result.block}

    # -- hook: on_session_end ------------------------------------------------

    def on_session_end(self, **kwargs: Any) -> None:
        """Persist the finished wake.

        Two paths, in order of preference:

        1. An :class:`Episode` built from the hook payload. Hermes' kwargs are
           host-defined, so every field is read defensively.
        2. Failing that, the wake **transcript**, retained as prose for the
           semantic store to extract from. This is the path that normally runs:
           Hermes' ``on_session_end`` payload carries no ``summary`` /
           ``decisions`` / ``open_loops``, so :meth:`build_episode` returns
           ``None`` for every wake unless the agent called ``record_episode``.
           Without the fallback nothing is ever retained.

        Skipped entirely when the agent already called ``record_episode`` for
        this session: a self-authored episode is better than anything derived
        here, and two records for one wake would double-count in recall.
        """

        session_id = str(kwargs.get("session_id") or "unknown")
        if session_id in self._recorded_sessions:
            logger.info("session %s already recorded via record_episode; skipping", session_id)
            return

        episode = self.build_episode(**kwargs)
        if episode is not None:
            self._persist(episode)
            return

        self._retain_transcript(session_id, interrupted=bool(kwargs.get("interrupted")))

    def _retain_transcript(self, session_id: str, *, interrupted: bool) -> None:
        """Retain the wake conversation so the store has something to extract.

        Interrupted wakes are skipped: the transcript is cut mid-thought and
        retaining it would teach the store a conclusion that was never reached.
        """

        if interrupted:
            logger.info("session %s was interrupted; not retaining its transcript", session_id)
            return

        text = render_transcript(session_messages(session_id))
        if not text:
            logger.info("session %s produced no transcript prose to retain", session_id)
            return

        try:
            # Keyed on the session so a second on_session_end for the same wake
            # replaces the document instead of duplicating it.
            self._provider.retain_text(text, document_id=f"aila-wake-{session_id}")
        except Exception:  # noqa: BLE001 - a memory failure must not break a wake
            logger.warning("failed to retain transcript for session %s", session_id, exc_info=True)
            return

        logger.info("retained transcript for session %s: %d chars", session_id, len(text))

    def _persist(self, episode: Episode) -> None:
        """Retain an episode and mirror it to the daily note.

        Each half is guarded independently so a provider outage still leaves
        the note on disk, which is what the filesystem recency channel reads.
        """

        try:
            self._provider.retain_episode(episode)
            # Flush before returning: a cron wake can exit immediately after
            # this, and an unflushed write is silently lost.
            self._provider.flush()
        except Exception:  # noqa: BLE001
            logger.warning("failed to retain episode %s", episode.episode_id, exc_info=True)

        try:
            append_episode_note(self._memory_dir, episode)
        except OSError:
            logger.warning("failed to mirror episode %s to daily note", episode.episode_id, exc_info=True)

    def build_episode(self, **kwargs: Any) -> Episode | None:
        """Assemble an :class:`Episode` from the session-end payload."""

        now = datetime.now(UTC)
        summary = str(kwargs.get("summary") or "").strip()[:MAX_SUMMARY_CHARS]
        decisions = _as_tuple(kwargs.get("decisions"))
        open_loops = _as_tuple(kwargs.get("open_loops"))

        if not (summary or decisions or open_loops):
            # Nothing worth carrying forward; do not pollute the store.
            return None

        session_id = str(kwargs.get("session_id") or "unknown")
        started_ts = _as_datetime(kwargs.get("started_ts"), default=now)
        ended_ts = _as_datetime(kwargs.get("ended_ts"), default=now)

        try:
            return Episode(
                episode_id=str(kwargs.get("episode_id") or uuid.uuid4().hex),
                session_id=session_id,
                started_ts=started_ts,
                ended_ts=ended_ts,
                summary=summary,
                decisions=decisions,
                open_loops=open_loops,
                entities=_as_tuple(kwargs.get("entities")),
            )
        except ValueError:
            logger.warning("could not build episode for session %s", session_id, exc_info=True)
            return None

    # -- registration --------------------------------------------------------

    def register(self, ctx: Any) -> None:
        ctx.register_tool(
            name=RECORD_EPISODE_TOOL,
            toolset=BRIEFING_TOOLSET,
            schema=_RECORD_SCHEMA,
            handler=self.record_episode,
            description=_RECORD_SCHEMA["description"],
        )
        ctx.register_tool(
            name=RECALL_MEMORY_TOOL,
            toolset=BRIEFING_TOOLSET,
            schema=_RECALL_SCHEMA,
            handler=self.recall_memory,
            description=_RECALL_SCHEMA["description"],
        )
        ctx.register_hook("pre_llm_call", self.pre_llm_call)
        ctx.register_hook("on_session_end", self.on_session_end)


def register(ctx: Any) -> None:
    """Entry point Hermes calls to load the briefing plugin."""

    BriefingPlugin().register(ctx)
