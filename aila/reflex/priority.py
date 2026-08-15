from __future__ import annotations

from aila.reflex.models import Event, Priority, StrictModel


class PrioritySeed(StrictModel):
    """The content used to seed a PRIORITY session from a P2 event.

    ``worker``/``kind`` are carried through so the PRIORITY session's output can
    be routed by modality (see :mod:`aila.reflex.output`). ``event_id`` lets the
    brain pull full detail via the ``reflex_expand`` tool.
    """

    event_id: str
    worker: str
    kind: str
    text: str


_HEADER = (
    "PRIORITY INTERRUPT (reflex P2). A high-priority event requires your "
    "attention now. Address it, then you will return to your prior work on the "
    "next wake."
)


def build_priority_seed(event: Event) -> PrioritySeed:
    """Turn a P2 ``Event`` into the seed message for a PRIORITY session.

    Only P2 events are valid seeds. The seed is a short directive + the event's
    title/summary + a pointer to ``reflex_expand`` for full detail.
    """

    if event.effective_priority != Priority.P2:
        raise ValueError(
            f"priority seed requires a P2 event, got {event.effective_priority.name}"
        )

    lines = [
        _HEADER,
        "",
        f"Source: {event.worker}/{event.kind} (event_id={event.event_id})",
        f"Title: {event.title}",
    ]
    if event.summary:
        lines.append(f"Summary: {event.summary}")
    if event.detail_available:
        lines.append(
            f"Full detail available: call reflex_expand(event_id=\"{event.event_id}\")."
        )

    return PrioritySeed(
        event_id=event.event_id,
        worker=event.worker,
        kind=event.kind,
        text="\n".join(lines),
    )
