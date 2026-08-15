"""AILA reflex plugin for the Hermes agent runtime.

This is a *general* Hermes plugin (not a memory provider, so it does not consume
the single external memory-provider slot reserved for Hindsight). It wires the
reflex event pipeline into the wake loop through supported extension points
only -- no edits to Hermes core:

* ``pre_llm_call`` hook -> injects the fenced reflex digest into the user
  message on the first turn of a wake (inject-once); empty afterwards.
* ``register_tool("reflex_expand")`` -> on-demand retrieval of an event's full
  context.
* ``on_session_end`` hook -> reserved for v3 episode/self-reflection.

The Hermes plugin API is accessed purely by duck typing so this module imports
cleanly in a host-independent test environment.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from aila.reflex.digest import build_digest
from aila.reflex.store import EventStore

REFLEX_TOOLSET = "reflex"
REFLEX_EXPAND_TOOL = "reflex_expand"

_EXPAND_SCHEMA: dict[str, Any] = {
    "name": REFLEX_EXPAND_TOOL,
    "description": (
        "Retrieve the full context of a reflex event that was shown title-only. "
        "Call this only when the event's detail_available flag is true and you "
        "need more than the title/summary."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "event_id": {
                "type": "string",
                "description": "The event_id from a reflex-events block entry.",
            }
        },
        "required": ["event_id"],
    },
}


def _hermes_home() -> Path:
    return Path(os.environ.get("HERMES_HOME") or (Path.home() / ".hermes")).expanduser()


def default_store_dir() -> Path:
    return _hermes_home() / "aila-body" / "reflex"


class ReflexPlugin:
    """Adapter binding the reflex pipeline to Hermes plugin callbacks."""

    def __init__(self, store: EventStore | None = None) -> None:
        self._store = store if store is not None else EventStore(default_store_dir())

    # -- hook: pre_llm_call --------------------------------------------------

    def pre_llm_call(self, **kwargs: Any) -> dict[str, str] | None:
        """Return the reflex digest for the first turn of a wake only."""

        if not kwargs.get("is_first_turn", False):
            return None
        result = build_digest(self._store)
        if result.is_empty:
            return None
        now = datetime.now(UTC)
        for event_id in result.event_ids:
            self._store.mark_seen(event_id, when=now)
        return {"context": result.block}

    # -- hook: on_session_end (v3 seam) -------------------------------------

    def on_session_end(self, **kwargs: Any) -> None:  # noqa: D401 - reserved
        """Reserved for v3 episode creation / self-reflection. No-op in v2."""
        return None

    # -- tool: reflex_expand -------------------------------------------------

    def reflex_expand(self, args: dict[str, Any], **kwargs: Any) -> str:
        event_id = str((args or {}).get("event_id", "")).strip()
        if not event_id:
            return json.dumps({"ok": False, "error": "event_id is required"})
        expanded = self._store.resolve(event_id)
        if expanded is None:
            return json.dumps({"ok": False, "error": f"unknown event_id: {event_id}"})
        return json.dumps(
            {"ok": True, "event": json.loads(expanded.model_dump_json())},
            ensure_ascii=False,
        )

    # -- registration --------------------------------------------------------

    def register(self, ctx: Any) -> None:
        ctx.register_tool(
            name=REFLEX_EXPAND_TOOL,
            toolset=REFLEX_TOOLSET,
            schema=_EXPAND_SCHEMA,
            handler=self.reflex_expand,
            description=_EXPAND_SCHEMA["description"],
        )
        ctx.register_hook("pre_llm_call", self.pre_llm_call)
        ctx.register_hook("on_session_end", self.on_session_end)


def register(ctx: Any) -> None:
    """Entry point Hermes calls to load the reflex plugin."""

    ReflexPlugin().register(ctx)
    _maybe_register_priority_platform(ctx)


def _maybe_register_priority_platform(ctx: Any) -> None:
    """Register the reflex gateway platform for P2 preemption (opt-in).

    Gated behind ``AILA_REFLEX_P2`` so enabling P2 session preemption (which makes
    reflex load-bearing for the main wake loop) is an explicit operator choice.
    Guarded so a Hermes build without ``register_platform`` is a no-op.
    """

    import os

    if os.environ.get("AILA_REFLEX_P2", "").strip().lower() not in {"1", "true", "yes", "on"}:
        return
    register_platform = getattr(ctx, "register_platform", None)
    if register_platform is None:
        return
    try:
        from aila.reflex.hermes_platform import (
            REFLEX_HOME_ENV,
            REFLEX_PLATFORM,
            build_reflex_adapter,
            standalone_reflex_sender,
        )

        register_platform(
            name=REFLEX_PLATFORM,
            label="Reflex",
            adapter_factory=build_reflex_adapter,
            check_fn=lambda: True,
            cron_deliver_env_var=REFLEX_HOME_ENV,
            standalone_sender_fn=standalone_reflex_sender,
        )
    except Exception:  # pragma: no cover - defensive: never break plugin load
        pass
