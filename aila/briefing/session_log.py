"""Read a wake's transcript from Hermes' session database.

This is the **only** source of a wake's own account of itself. The plugin hooks
cannot supply it:

* ``pre_llm_call`` fires once per ``run_conversation``, with
  ``conversation_history=list(messages)`` captured *before* the turn runs -- so
  it holds the wake's starting state, never its work.
* ``on_session_end`` carries no messages at all.
* ``final_response`` exists only on ``pre_verify``, a verification-stop hook
  that is not guaranteed to fire.

So this reads ``$HERMES_HOME/state.db`` directly. That is a deliberate coupling
to a Hermes internal, and it is why **every** failure here is swallowed: a
missing file, a renamed table, a locked database or a schema change must degrade
to "no transcript" and leave the wake untouched.
"""

from __future__ import annotations

import logging
import os
import sqlite3
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Bounds a runaway session; the renderer applies the real budget. Well above the
# ~66 messages a long wake produces.
MAX_MESSAGES = 400

# The gateway writes to this database continuously; never wait long for a lock.
CONNECT_TIMEOUT_SECONDS = 2.0


def state_db_path() -> Path:
    hermes_home = os.environ.get("HERMES_HOME") or (Path.home() / ".hermes")
    return Path(hermes_home).expanduser() / "state.db"


def session_messages(
    session_id: str,
    *,
    limit: int = MAX_MESSAGES,
    path: Path | None = None,
) -> list[dict[str, Any]]:
    """Return every stored message of ``session_id``, oldest first.

    Returns ``[]`` for any failure at all -- retention is best-effort and must
    never be a reason for a wake to fail.
    """

    if not session_id:
        return []

    target = path if path is not None else state_db_path()
    if not Path(target).exists():
        return []

    try:
        # Read-only: this process must never write to Hermes' database.
        uri = f"file:{Path(target).as_posix()}?mode=ro"
        with sqlite3.connect(uri, uri=True, timeout=CONNECT_TIMEOUT_SECONDS) as con:
            rows = con.execute(
                "SELECT role, content FROM messages WHERE session_id = ? "
                "ORDER BY timestamp ASC, id ASC LIMIT ?",
                (session_id, max(1, limit)),
            ).fetchall()
    except Exception:  # noqa: BLE001 - see module docstring
        logger.debug("could not read session %s", session_id, exc_info=True)
        return []

    return [
        {"role": role, "content": content}
        for role, content in rows
        if isinstance(content, str) and content.strip()
    ]
