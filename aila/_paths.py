from __future__ import annotations

from pathlib import Path


def expand_path(value: str | Path) -> Path:
    """Return ``value`` as a ``Path`` with ``~`` and ``~user`` expanded.

    Centralizes the ``Path(str(value)).expanduser()`` idiom used across the
    installer and workers so path handling is consistent in one place.
    """
    return Path(str(value)).expanduser()
