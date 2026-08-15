from __future__ import annotations


class BackendHealth:
    """Track a model backend's reachability and flag one status transition.

    Model-backed sensor workers (e.g. mic/whisper) want to emit a single
    ``sensor.status`` observation when a backend goes down, not one per failed
    poll. Feed each attempt's outcome to :meth:`record`; :meth:`take_degraded`
    returns ``True`` exactly once per healthy->unavailable transition.
    """

    def __init__(self) -> None:
        # None until the first attempt; True/False track reachability so we only
        # surface a status change when it actually flips.
        self._ok: bool | None = None
        self._degraded_pending = False

    def record(self, ok: bool) -> None:
        """Record the outcome of one backend attempt."""
        if not ok and self._ok is not False:
            self._degraded_pending = True
        self._ok = ok

    def take_degraded(self) -> bool:
        """Return True once after a transition into the unavailable state."""
        if not self._degraded_pending:
            return False
        self._degraded_pending = False
        return True
