from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from aila.reflex.ingest import DedupConfig, IngestFilterConfig
from aila.reflex.models import Priority
from aila.reflex.ranker import RankingRule, RankingRules


def default_ranking_rules() -> RankingRules:
    """Return the built-in v2 ranking defaults.

    v2 only actively routes P2/P3/P5. P0/P1/P4 are valid values but are left
    for v3 immediacy/session-start handling, so the defaults never assign them.
    Unknown workers/kinds fall through to ``default_priority`` (P5).
    """

    return RankingRules(
        rules=(
            # Severity-driven escalation (checked first, source-agnostic).
            RankingRule(severity="alert", priority=Priority.P2),
            RankingRule(severity="warning", priority=Priority.P3),
            # Spoken input to the agent is worth injecting.
            RankingRule(worker="mic", kind="speech.segment", priority=Priority.P2),
            # Scene perception is a soft recommendation.
            RankingRule(worker="camera", kind="scene.caption", priority=Priority.P3),
            RankingRule(worker="camera", kind="scene.motion", priority=Priority.P3),
            # Filesystem activity: notable but optional.
            RankingRule(worker="filesystem", kind="file.created", priority=Priority.P3),
            RankingRule(worker="filesystem", kind="file.changed", priority=Priority.P3),
            RankingRule(worker="filesystem", kind="file.deleted", priority=Priority.P3),
        ),
        default_priority=Priority.P5,
    )


def parse_ranking_rules(data: Any) -> RankingRules:
    """Build :class:`RankingRules` from an already-parsed mapping.

    Accepts an empty/None document as "use defaults". Only the ``rules`` and
    ``default_priority`` keys are consumed; sibling sections such as ``filter``
    and ``dedup`` are ignored here (see :func:`parse_ingest_filter` /
    :func:`parse_dedup_config`).
    """

    if not data:
        return default_ranking_rules()
    if not isinstance(data, dict):
        raise ValueError("reflex ranking config must be a mapping")
    ranking_only = {key: data[key] for key in ("rules", "default_priority") if key in data}
    if not ranking_only:
        return default_ranking_rules()
    return RankingRules.model_validate(ranking_only)


def parse_ingest_filter(data: Any) -> IngestFilterConfig:
    """Build :class:`IngestFilterConfig` from a parsed mapping's ``filter``.

    Missing/empty documents yield the permissive defaults.
    """

    if not data or not isinstance(data, dict):
        return IngestFilterConfig()
    section = data.get("filter")
    if not section:
        return IngestFilterConfig()
    if not isinstance(section, dict):
        raise ValueError("reflex 'filter' config must be a mapping")
    defaults = IngestFilterConfig()
    return IngestFilterConfig(
        min_speech_confidence=float(section.get("min_speech_confidence", defaults.min_speech_confidence)),
        min_motion_level=float(section.get("min_motion_level", defaults.min_motion_level)),
        drop_empty_text=bool(section.get("drop_empty_text", defaults.drop_empty_text)),
    )


def parse_dedup_config(data: Any) -> DedupConfig:
    """Build :class:`DedupConfig` from a parsed mapping's ``dedup`` section."""

    if not data or not isinstance(data, dict):
        return DedupConfig()
    section = data.get("dedup")
    if not section:
        return DedupConfig()
    if not isinstance(section, dict):
        raise ValueError("reflex 'dedup' config must be a mapping")
    caption_key = str(section.get("caption_key", DedupConfig().caption_key)).strip().lower()
    if caption_key not in ("caption", "labels"):
        raise ValueError(f"invalid dedup.caption_key: {caption_key!r} (expected 'caption' or 'labels')")
    return DedupConfig(caption_key=caption_key)  # type: ignore[arg-type]


def load_ranking_rules(path: Path) -> RankingRules:
    """Load ranking rules from a YAML file, falling back to defaults.

    If the file does not exist, the built-in defaults are returned so the
    pipeline is usable out of the box.
    """

    path = Path(path)
    if not path.is_file():
        return default_ranking_rules()
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return parse_ranking_rules(data)


def load_ingest_filter(path: Path) -> IngestFilterConfig:
    """Load the ingest filter thresholds from the reflex YAML (or defaults)."""

    path = Path(path)
    if not path.is_file():
        return IngestFilterConfig()
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return parse_ingest_filter(data)


def load_dedup_config(path: Path) -> DedupConfig:
    """Load the dedup config from the reflex YAML (or defaults)."""

    path = Path(path)
    if not path.is_file():
        return DedupConfig()
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return parse_dedup_config(data)
