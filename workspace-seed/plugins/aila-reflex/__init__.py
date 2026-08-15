"""AILA reflex plugin entrypoint for Hermes.

Hermes loads directory plugins from ``~/.hermes/plugins/<name>/`` and calls the
module-level ``register(ctx)``. The actual implementation lives in the installed
``aila`` package so it stays testable and host-independent.
"""

from __future__ import annotations

from aila.reflex.hermes_plugin import register

__all__ = ["register"]
