from __future__ import annotations

from aila.briefing.composite import CompositeMemoryProvider
from aila.briefing.compose import BriefingResult, build_briefing, build_semantic_query
from aila.briefing.filesystem import FilesystemMemoryProvider
from aila.briefing.models import BriefingEntry, Episode, MemoryFact
from aila.briefing.provider import MemoryProvider, NullMemoryProvider

__all__ = [
    "BriefingEntry",
    "BriefingResult",
    "CompositeMemoryProvider",
    "Episode",
    "FilesystemMemoryProvider",
    "MemoryFact",
    "MemoryProvider",
    "NullMemoryProvider",
    "build_briefing",
    "build_semantic_query",
]
