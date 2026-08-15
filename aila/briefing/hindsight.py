"""Hindsight-backed :class:`~aila.briefing.provider.MemoryProvider` adapter.

Talks to the daemon's HTTP API directly (see :class:`_RestClient`) rather than
through the ``hindsight-client`` SDK. The SDK's synchronous facade is unusable
here: it caches one ``aiohttp.ClientSession``, created lazily on the first
request, and drives every later call through ``asyncio.get_event_loop()``. The
session stays bound to the loop that created it, so a call from any *other*
thread raises ``RuntimeError: Timeout context manager should be used inside a
task``. Hermes builds this provider once per process and runs each wake on a
different worker thread, which made every recall after the first one fail.
``httpx.Client`` needs no event loop, so the failure mode cannot recur.

Endpoints used, verified against the daemon's OpenAPI schema (API 0.8.6)::

    POST /v1/default/banks/{bank_id}/memories
         {"items": [{"content", "timestamp", "metadata", "tags"}], "async"}
    POST /v1/default/banks/{bank_id}/memories/recall
         {"query", "types", "budget", "tags"}

**Hindsight is a knowledge-extraction store, not a key-value store.** This was
established by live testing: retained JSON came back as LLM-written sentences
*about* the JSON. It consumes text and emits its own synthesized facts, so the
original :class:`~aila.briefing.models.Episode` records are unrecoverable.
Three consequences shape this module:

* Retain sends **prose** (:func:`render_episode_text`), because extraction
  quality depends entirely on the text it is given.
* Recall returns :class:`~aila.briefing.models.MemoryFact` prose, never
  episodes -- hence :meth:`HindsightMemoryProvider.recent_episodes` always
  returns ``()``. Recency comes from the filesystem provider (see
  :mod:`aila.briefing.composite`).
* Record shapes differ by endpoint: ``list_memories`` returns plain dicts with
  ``fact_type``; ``recall`` returns ``RecallResult`` objects with ``type``.
  :func:`fact_from_record` handles both.

Retention is asynchronous (see :data:`RETAIN_ASYNC`): the daemon owns the work
once the request is accepted, so a wake exiting immediately afterwards cannot
lose it. Nothing is buffered client-side and no session is held open, so
:meth:`HindsightMemoryProvider.flush` has nothing to do.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from aila.briefing.models import MAX_SUMMARY_CHARS, Episode, MemoryFact

logger = logging.getLogger(__name__)

# Tag applied to every retained episode; the API supports tag filtering, which
# is a far better discriminator than sniffing the payload.
EPISODE_TAG = "aila-episode"

# Hindsight's own record type. Must stay aligned with `recall_types` in
# ~/.hermes/hindsight/config.json, or episodes are stored but never recalled.
EPISODE_TYPE = "observation"

# Marks retained wake transcripts in metadata, so they can be told apart from
# episodes the agent authored deliberately.
TRANSCRIPT_TAG = "aila-transcript"

# Retain asynchronously: the daemon persists the operation and its own worker
# completes the extraction, so a cron wake exiting immediately afterwards
# cannot lose the write. Verified on the host: an async retain returned in 15ms
# and its facts landed after the client process had already exited. Extraction
# takes ~20s per chunk, so a synchronous retain would block every wake for as
# long as it takes the LLM to run.
RETAIN_ASYNC = True

DEFAULT_BANK_ID = "hermes"
DEFAULT_PROFILE = "hermes"

# The API namespaces every bank route; the embedded daemon serves exactly one.
NAMESPACE = "default"

# Recall runs an LLM-backed pipeline, so it is far slower than a plain HTTP
# call. Matches the ``timeout`` in ~/.hermes/hindsight/config.json.
DEFAULT_TIMEOUT = 120.0

# AILA-namespaced so it cannot collide with Hindsight's own HINDSIGHT_* vars.
BASE_URL_ENV = "AILA_HINDSIGHT_BASE_URL"


def _hermes_home() -> Path:
    return Path(os.environ.get("HERMES_HOME") or (Path.home() / ".hermes")).expanduser()


def hindsight_config_path() -> Path:
    return _hermes_home() / "hindsight" / "config.json"


def profile_env_path(profile: str = DEFAULT_PROFILE) -> Path:
    """Path of the daemon's profile env file, which pins the API port."""

    return Path.home() / ".hindsight" / "profiles" / f"{profile}.env"


def port_from_profile_env(path: Path) -> int | None:
    """Extract ``HINDSIGHT_API_PORT`` from a profile env file."""

    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError:
        return None
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        if key.strip() != "HINDSIGHT_API_PORT":
            continue
        try:
            return int(value.strip())
        except ValueError:
            return None
    return None


def resolve_base_url(config: dict[str, Any] | None = None) -> str | None:
    """Resolve the daemon endpoint, or ``None`` when it cannot be determined.

    Priority:

    1. ``AILA_HINDSIGHT_BASE_URL`` -- explicit override.
    2. ``base_url`` in ``hindsight/config.json``.
    3. ``HINDSIGHT_API_PORT`` from the daemon's profile env file. Hermes runs
       the embedded daemon on a per-profile port that is *pinned* in that file,
       so this is the reliable local discovery path.

    Returning ``None`` is a normal outcome: the briefing then runs on
    filesystem recency alone.
    """

    override = os.environ.get(BASE_URL_ENV)
    if override:
        return override.strip()

    cfg = config if config is not None else load_hindsight_config()
    configured = cfg.get("base_url")
    if configured:
        return str(configured).strip()

    profile = str(cfg.get("profile") or DEFAULT_PROFILE)
    port = port_from_profile_env(profile_env_path(profile))
    if port:
        return f"http://127.0.0.1:{port}"
    return None


def load_hindsight_config(path: Path | None = None) -> dict[str, Any]:
    """Read ``~/.hermes/hindsight/config.json`` written by ``setup-hermes.sh``."""

    target = path if path is not None else hindsight_config_path()
    try:
        loaded = json.loads(Path(target).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def render_episode_text(episode: Episode) -> str:
    """Render an episode as prose for Hindsight to extract knowledge from.

    Hindsight consumes text and synthesizes its own facts, so extraction
    quality depends entirely on what it is given. JSON was a mistake: it was
    parsed into sentences *about* the JSON. Plain declarative prose produces
    usable observations.
    """

    started = episode.started_ts.isoformat()
    ended = episode.ended_ts.isoformat()
    parts = [f"AILA wake session from {started} to {ended}."]

    if episode.summary.strip():
        parts.append(episode.summary.strip())
    if episode.decisions:
        parts.append("Decisions made: " + "; ".join(episode.decisions) + ".")
    if episode.open_loops:
        parts.append("Unfinished work to pick up next: " + "; ".join(episode.open_loops) + ".")
    if episode.entities:
        parts.append("Related topics: " + ", ".join(episode.entities) + ".")

    return " ".join(parts)


def episode_metadata(episode: Episode) -> dict[str, str]:
    """Metadata for a retained episode (Hindsight requires string values)."""

    return {
        "kind": EPISODE_TAG,
        "episode_id": episode.episode_id,
        "session_id": episode.session_id,
        "ended_ts": episode.ended_ts.isoformat(),
    }


def _field(record: Any, name: str) -> Any:
    """Read ``name`` from a record that may be a dict or a model object.

    ``list_memories`` returns plain dicts; ``recall`` returns ``RecallResult``
    objects. Both are handled.
    """

    if isinstance(record, dict):
        return record.get(name)
    return getattr(record, name, None)


def fact_from_record(record: Any) -> MemoryFact | None:
    """Build a :class:`MemoryFact` from a Hindsight record, or ``None``."""

    text = _field(record, "text")
    if not isinstance(text, str) or not text.strip():
        return None

    ts = _field(record, "mentioned_at") or _field(record, "date")
    if isinstance(ts, str):
        try:
            ts = datetime.fromisoformat(ts)
        except ValueError:
            ts = None
    elif not isinstance(ts, datetime):
        ts = None

    proof = _field(record, "proof_count")
    try:
        proof_count = max(int(proof), 0)
    except (TypeError, ValueError):
        proof_count = 0

    try:
        return MemoryFact(
            fact_id=str(_field(record, "id") or ""),
            text=text.strip()[:MAX_SUMMARY_CHARS],
            # Shape differs by endpoint: ``list_memories`` dicts carry
            # ``fact_type`` while ``RecallResult`` objects carry ``type``.
            # Only the *request* params use ``type``/``types``.
            fact_type=str(_field(record, "fact_type") or _field(record, "type") or ""),
            ts=ts,
            proof_count=proof_count,
        )
    except ValueError:
        return None


class _RestClient:
    """Minimal transport for the two endpoints this module needs.

    Method signatures deliberately mirror the ``hindsight-client`` SDK's, so
    the provider stays agnostic about which one it holds and tests can inject
    a fake in either shape.

    A fresh ``httpx.Client`` per call keeps the object stateless: there is no
    connection pool bound to a thread or an event loop, which is precisely the
    trap the SDK fell into (see the module docstring). On loopback the setup
    cost is negligible next to Hindsight's own LLM work.
    """

    def __init__(self, base_url: str, httpx_module: Any, *, timeout: float = DEFAULT_TIMEOUT) -> None:
        self._base_url = base_url.rstrip("/")
        self._httpx = httpx_module
        self._timeout = timeout

    def _post(self, path: str, payload: dict[str, Any]) -> Any:
        url = f"{self._base_url}/v1/{NAMESPACE}/banks/{path}"
        with self._httpx.Client(timeout=self._timeout) as client:
            response = client.post(url, json=payload)
            response.raise_for_status()
            return response.json()

    def retain(
        self,
        *,
        bank_id: str,
        content: str,
        timestamp: datetime | None = None,
        metadata: dict[str, str] | None = None,
        tags: list[str] | None = None,
        retain_async: bool = False,
        document_id: str | None = None,
        update_mode: str | None = None,
    ) -> Any:
        item: dict[str, Any] = {"content": content}
        if timestamp is not None:
            item["timestamp"] = timestamp.isoformat()
        if metadata:
            item["metadata"] = metadata
        if tags:
            item["tags"] = tags
        if document_id:
            item["document_id"] = document_id
        if update_mode:
            item["update_mode"] = update_mode
        return self._post(f"{bank_id}/memories", {"items": [item], "async": retain_async})

    def recall(
        self,
        *,
        bank_id: str,
        query: str,
        types: list[str] | None = None,
        budget: str = "mid",
        tags: list[str] | None = None,
    ) -> Any:
        payload: dict[str, Any] = {"query": query, "budget": budget}
        if types:
            payload["types"] = types
        if tags:
            payload["tags"] = tags
        return self._post(f"{bank_id}/memories/recall", payload)


class HindsightMemoryProvider:
    """Adapter binding :class:`Episode` storage onto the Hindsight client."""

    def __init__(
        self,
        client: Any | None = None,
        *,
        bank_id: str | None = None,
        recall_budget: str | None = None,
    ) -> None:
        config = load_hindsight_config()
        self._bank_id = bank_id or str(config.get("bank_id") or DEFAULT_BANK_ID)
        self._recall_budget = recall_budget or str(config.get("recall_budget") or "mid")
        self._client = client if client is not None else _build_client()

    @property
    def available(self) -> bool:
        return self._client is not None

    # -- MemoryProvider ------------------------------------------------------

    def retain_episode(self, episode: Episode) -> None:
        if self._client is None:
            return
        try:
            self._client.retain(
                bank_id=self._bank_id,
                content=render_episode_text(episode),
                timestamp=episode.ended_ts,
                metadata=episode_metadata(episode),
                tags=[EPISODE_TAG],
                retain_async=RETAIN_ASYNC,
            )
        except Exception:  # noqa: BLE001 - never break a wake on memory failure
            logger.warning("hindsight retain failed for episode %s", episode.episode_id, exc_info=True)

    def retain_text(
        self,
        text: str,
        *,
        document_id: str | None = None,
        timestamp: datetime | None = None,
    ) -> None:
        """Retain free prose (a wake transcript) for Hindsight to extract from.

        Tagged like an episode so recall -- which filters on
        :data:`EPISODE_TAG` -- sees the facts extracted from it.

        With ``document_id`` the write is idempotent: ``update_mode="replace"``
        makes a second retain under the same id *replace* the stored document
        instead of adding a duplicate. Hermes fires ``on_session_end`` once per
        ``run_conversation``, which can happen more than once per wake, and the
        later call carries the fuller transcript -- so replacing keeps the best
        version and never double-counts a wake in recall.

        ``timestamp`` defaults to now, which is right for a live wake; a
        backfill must pass the wake's real end time or the store loses the
        temporal ordering it links facts by.
        """

        if self._client is None or not text.strip():
            return
        try:
            self._client.retain(
                bank_id=self._bank_id,
                content=text,
                timestamp=timestamp or datetime.now(UTC),
                metadata={"kind": TRANSCRIPT_TAG},
                tags=[EPISODE_TAG],
                retain_async=RETAIN_ASYNC,
                document_id=document_id,
                update_mode="replace" if document_id else None,
            )
        except Exception:  # noqa: BLE001
            logger.warning("hindsight transcript retain failed", exc_info=True)

    def recent_episodes(self, *, limit: int) -> tuple[Episode, ...]:
        """Always empty: Hindsight cannot return stored episodes.

        It consumes retained text and returns its own synthesized facts, so the
        original :class:`Episode` records are unrecoverable. Recency comes from
        the filesystem provider (see :mod:`aila.briefing.composite`).
        """
        return ()

    def recall_facts(self, query: str, *, limit: int) -> tuple[MemoryFact, ...]:
        if self._client is None or not query.strip():
            return ()
        try:
            response = self._client.recall(
                bank_id=self._bank_id,
                query=query,
                types=[EPISODE_TYPE],
                budget=self._recall_budget,
                tags=[EPISODE_TAG],
            )
        except Exception:  # noqa: BLE001
            logger.warning("hindsight recall failed", exc_info=True)
            return ()

        # ``_field`` reads either shape: the REST endpoint returns a JSON
        # dict, an injected SDK-style fake returns a model object.
        records = _field(response, "results")
        if records is None:
            records = _field(response, "items")

        facts: list[MemoryFact] = []
        for record in records or ():
            fact = fact_from_record(record)
            if fact is not None:
                facts.append(fact)
        # Better-corroborated facts first; the budget drops from the tail.
        # ``recall`` results carry no ``proof_count`` (that is a
        # ``list_memories`` field), so they all tie at 0 -- and because the
        # sort is stable, Hindsight's own relevance ordering is preserved.
        facts.sort(key=lambda item: item.proof_count, reverse=True)
        return tuple(facts[:limit])

    def flush(self) -> None:
        """No-op: retention is synchronous and no connection is held open.

        The provider must *not* drop its client here. It is built once per
        process and reused by every later wake, so releasing it on the first
        retain would silently disable recall for the rest of the process.
        """

        return


def _build_client() -> Any | None:
    """Instantiate the REST client, or ``None`` when unreachable.

    The endpoint is discovered directly (see :func:`resolve_base_url`) because
    Hermes' own provider never constructs one on this host -- the briefing
    talks to the embedded daemon independently of Hermes.
    """

    base_url = resolve_base_url()
    if not base_url:
        logger.info(
            "no hindsight endpoint discovered; briefing uses filesystem recency only",
        )
        return None

    try:
        import httpx
    except ImportError:
        logger.info("httpx unavailable; briefing uses filesystem recency only")
        return None

    logger.info("hindsight semantic channel enabled at %s", base_url)
    return _RestClient(base_url, httpx)
