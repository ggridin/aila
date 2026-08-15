from __future__ import annotations

from collections.abc import Callable
from enum import Enum


class OutputSink(str, Enum):
    """Where a PRIORITY session's output should be delivered."""

    speaker = "speaker"
    display = "display"
    logs = "logs"


# Modality routing by the triggering event's worker. Audio-origin events answer
# through the speaker; visual-origin through the display; everything else
# (filesystem, system/laptop-health, unknown) goes to tooling/logs.
_WORKER_SINKS: dict[str, OutputSink] = {
    "mic": OutputSink.speaker,
    "camera": OutputSink.display,
    "filesystem": OutputSink.logs,
}


def route_output(worker: str, kind: str | None = None) -> OutputSink:
    """Map the triggering event's source to an output sink.

    ``worker``/``kind`` are treated as opaque, so future sources
    (e.g. ``health``) fall through to ``logs`` without code changes.
    """

    return _WORKER_SINKS.get(worker, OutputSink.logs)


def deliver(
    sink: OutputSink,
    content: str,
    *,
    speaker: Callable[[str], None] | None = None,
    display: Callable[[str], None] | None = None,
    log: Callable[[str], None] | None = None,
) -> OutputSink:
    """Dispatch ``content`` to the chosen sink via injected handlers.

    Handlers are injected (the host wires them to speaker.speak / display.render /
    a log writer) so this stays host-independent and unit-testable. A missing
    handler falls back to ``log`` and finally to a no-op.
    """

    handlers: dict[OutputSink, Callable[[str], None] | None] = {
        OutputSink.speaker: speaker,
        OutputSink.display: display,
        OutputSink.logs: log,
    }
    handler = handlers.get(sink) or log
    if handler is not None:
        handler(content)
    return sink
