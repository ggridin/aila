from __future__ import annotations

import re
from typing import Any

from aila.contracts import Observation
from aila.reflex.models import StrictModel

# ChatML / delimiter control tokens that must never survive into an injected
# block, so untrusted payload text cannot forge turn boundaries or break the
# reflex-events fence.
_CHATML_TOKEN_RE = re.compile(r"<\|[^|>]*\|>")
_FENCE_TOKEN_RE = re.compile(r"<<<[^>\n]*>>>")
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")

TITLE_MAX_LEN = 80
SUMMARY_MAX_LEN = 200


class Summary(StrictModel):
    title: str
    summary: str
    detail_available: bool


def sanitize(text: str) -> str:
    """Strip ChatML control tokens / control chars and collapse whitespace.

    Applied to all untrusted payload-derived text before it enters a title or
    summary.
    """

    text = _CHATML_TOKEN_RE.sub("", text)
    text = _FENCE_TOKEN_RE.sub("", text)
    text = _CONTROL_CHARS_RE.sub(" ", text)
    return " ".join(text.split()).strip()


def _truncate(text: str, limit: int) -> tuple[str, bool]:
    """Return ``(text_or_truncation, was_truncated)``."""

    if len(text) <= limit:
        return text, False
    return text[: max(0, limit - 1)].rstrip() + "\u2026", True


def _field(payload: Any, name: str, default: Any = None) -> Any:
    if isinstance(payload, dict):
        return payload.get(name, default)
    return getattr(payload, name, default)


def summarize(observation: Observation) -> Summary:
    """Produce a compact title/summary and whether more detail exists.

    ``detail_available`` is ``False`` only when the summary is effectively
    lossless (nothing meaningful would be gained from ``reflex_expand``).
    """

    payload = observation.payload
    kind = observation.kind

    if kind == "speech.segment":
        text = sanitize(str(_field(payload, "text", "")))
        lang = str(_field(payload, "lang", "") or "")
        title, _ = _truncate(text or "(speech)", TITLE_MAX_LEN)
        summary, _ = _truncate(text, SUMMARY_MAX_LEN)
        prefix = f"Speech [{lang}]: " if lang else "Speech: "
        # Full transcript + timing/confidence are always richer than the line.
        return Summary(
            title=(f"{prefix}{title}").strip(),
            summary=summary,
            detail_available=True,
        )

    if kind == "scene.caption":
        caption = sanitize(str(_field(payload, "caption", "")))
        labels = [sanitize(str(x)) for x in (_field(payload, "labels", []) or [])]
        boxes = _field(payload, "boxes", []) or []
        title, _ = _truncate(caption or "(scene)", TITLE_MAX_LEN)
        summary_text = caption
        if labels:
            summary_text = f"{caption} [{', '.join(labels)}]"
        summary, _ = _truncate(summary_text, SUMMARY_MAX_LEN)
        return Summary(
            title=f"Scene: {title}",
            summary=summary,
            detail_available=bool(boxes),
        )

    if kind == "scene.motion":
        region = sanitize(str(_field(payload, "region", "")))
        level = _field(payload, "level", 0.0)
        line = f"Motion in {region} (level {level:.2f})"
        return Summary(title=line, summary=line, detail_available=False)

    if kind == "sensor.status":
        component = sanitize(str(_field(payload, "component", "sensor")))
        state = sanitize(str(_field(payload, "state", "")))
        detail = sanitize(str(_field(payload, "detail", "")))
        head = f"Sensor {component}: {state}" if state else f"Sensor {component}"
        summary_text = f"{head} - {detail}" if detail else head
        title, _ = _truncate(head, TITLE_MAX_LEN)
        summary, _ = _truncate(summary_text, SUMMARY_MAX_LEN)
        return Summary(title=title, summary=summary, detail_available=False)

    if kind in {"file.changed", "file.created", "file.deleted"}:
        path = sanitize(str(_field(payload, "path", "")))
        change = str(_field(payload, "change", kind.split(".")[-1]))
        base = path.rsplit("/", 1)[-1].rsplit("\\", 1)[-1] or path
        title, _ = _truncate(f"File {change}: {base}", TITLE_MAX_LEN)
        summary, _ = _truncate(f"File {change}: {path}", SUMMARY_MAX_LEN)
        # size/mtime are extra detail retrievable via expand.
        return Summary(title=title, summary=summary, detail_available=True)

    # Source-agnostic fallback for unknown/future kinds.
    text = sanitize(str(_field(payload, "text", "") or _field(payload, "caption", "") or ""))
    title, trunc = _truncate(text or f"{observation.worker}/{kind}", TITLE_MAX_LEN)
    summary, strunc = _truncate(text, SUMMARY_MAX_LEN)
    return Summary(
        title=f"{observation.worker}/{kind}: {title}" if text else title,
        summary=summary,
        detail_available=bool(trunc or strunc),
    )
