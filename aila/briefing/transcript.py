"""Render a wake transcript as prose for the semantic store to extract from.

Hindsight is a knowledge-extraction store: it consumes text and synthesizes its
own facts with its own LLM (see :mod:`aila.briefing.hindsight`). That makes the
raw wake conversation usable as retention input directly -- no local
summarization step is needed, and the extraction model is a better writer than
the local wake model.

What it is given still decides the quality, so this module is aggressive about
what it drops. Measured on the host, a wake transcript is ~55-184 KB, of which
**~97% is tool payloads** -- file dumps, terminal output, directory listings.
That bulk is not prose *about* the work, it *is* the work's raw material, and it
is worthless to extraction.

Only **AILA's own messages** survive. The wake has no human in it: its single
user-role message is the cron trigger, byte-identical every wake -- 520 chars of
delivery directives (``[IMPORTANT: You are running as a scheduled cron job...]``)
plus a fixed prompt. Retaining that taught the store nothing about the wake while
making up ~23% of every retained document, so the whole role is dropped rather
than pattern-stripped.

The most valuable message is the last one: a deliberate wake report (measured
500-1600 chars) carrying system status, what was done, and open loops. It must
never be truncated -- see :data:`MAX_MESSAGE_CHARS`.
"""

from __future__ import annotations

import re
from typing import Any

# Only AILA's own prose. Tool results are payloads; the sole user-role message
# is the invariant cron trigger (see module docstring).
CONVERSATION_ROLES = ("assistant",)

# Speaker label. Named rather than "I" because the extraction LLM reads it to
# decide *who acted*: with an ambiguous label it defaults to the assistant/user
# framing of an ordinary chat and writes facts like "the user decided X" for
# things AILA itself decided -- handing AILA's own agency to the human every
# time such a fact is recalled. Kept aligned with the bank's ``retain_mission``.
AGENT_LABEL = "AILA"

# Whole-transcript budget; above this the head is dropped.
MAX_TRANSCRIPT_CHARS = 6000

# Per-message cap. Generous on purpose: the closing wake report runs 500-1600
# chars and is the single most valuable thing a wake produces. An earlier
# 1500-char cap amputated it mid-sentence, cutting the open-loops section --
# exactly the content the next wake needs most.
MAX_MESSAGE_CHARS = 3000

# Matches any injected block, e.g. ``<<<SESSION_BRIEFING ...>>> ... <<<END_SESSION_BRIEFING>>>``.
# The opening marker carries a trailing description, so only the leading tag is
# captured and matched against the closing marker.
_FENCED_BLOCK = re.compile(
    r"<<<(?P<tag>[A-Z_]+)\b[^>]*>>>.*?<<<END_(?P=tag)>>>",
    re.DOTALL,
)


def strip_injected_blocks(text: str) -> str:
    """Remove fenced blocks that were injected *into* the wake.

    These are recalled memory, not new material. Retaining them would create a
    feedback loop in which the store re-ingests its own recall output. They
    arrive in the user message, which is now dropped wholesale -- this stays as
    defence in depth, for when AILA quotes its own briefing back.
    """

    return _FENCED_BLOCK.sub(" ", text)


def _message_text(message: Any) -> str:
    """Extract prose from a message, or ``""`` when it carries none."""

    if not isinstance(message, dict):
        return ""
    if message.get("role") not in CONVERSATION_ROLES:
        return ""

    content = message.get("content")
    if not isinstance(content, str):
        # Structured content (tool-call assistant turns, multimodal parts)
        # carries no prose worth extracting.
        return ""

    text = strip_injected_blocks(content).strip()
    if not text:
        return ""
    return text[:MAX_MESSAGE_CHARS]


def render_transcript(
    messages: Any,
    *,
    max_chars: int = MAX_TRANSCRIPT_CHARS,
) -> str:
    """Render ``messages`` as AILA's account of the wake, or ``""`` when empty.

    Keeps the **tail** when over budget: a wake's conclusions -- what was
    settled and what was left open -- land at the end, and that is what the next
    wake needs.
    """

    if not messages:
        return ""

    lines: list[str] = []
    for message in messages:
        text = _message_text(message)
        if text:
            lines.append(f"{AGENT_LABEL}: {text}")

    if not lines:
        return ""

    # Drop from the head until the budget is met, so the tail survives.
    while lines and sum(len(line) + 1 for line in lines) > max_chars:
        lines.pop(0)

    return "\n".join(lines)
