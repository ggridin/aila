from __future__ import annotations

from pathlib import Path


def expand_path(value: str | Path) -> Path:
    """Return ``value`` as a ``Path`` with ``~`` and ``~user`` expanded."""
    return Path(str(value)).expanduser()
