from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal
from uuid import uuid4

from aila.contracts import Observation
from aila.queue import ObservationQueue
from aila.reflex.models import Event
from aila.reflex.ranker import RankingRules, initial_rank
from aila.reflex.store import EventStore
from aila.reflex.summarize import summarize


@dataclass(frozen=True)
class IngestFilterConfig:
    """Thresholds for dropping low-value observations before ranking."""

    min_speech_confidence: float = 0.0
    min_motion_level: float = 0.0
    drop_empty_text: bool = True


CaptionKeyMode = Literal["caption", "labels"]


@dataclass(frozen=True)
class DedupConfig:
    """Controls how observations collapse into a single event.

    ``caption_key`` selects the ``scene.caption`` dedup granularity:

    - ``"caption"`` (default): key on the full caption text. Any change in the
      caption string (including a jittering motion figure) starts a new event.
    - ``"labels"``: key on the sorted set of detected object labels only, so a
      static scene collapses into one event regardless of motion jitter. This
      is the aggressive setting for reducing near-identical camera shots.
    """

    caption_key: CaptionKeyMode = "caption"


def _caption_dedup_value(payload: object, mode: CaptionKeyMode) -> str:
    if mode == "labels":
        labels = _get(payload, "labels", [])
        if isinstance(labels, (list, tuple)):
            normalized = sorted({str(label).strip().lower() for label in labels if str(label).strip()})
            return ",".join(normalized)
        return ""
    return str(_get(payload, "caption", ""))


def dedup_key_for(observation: Observation, config: DedupConfig | None = None) -> str:
    """Compute the collapse key for an observation.

    Repeated perceptions that share a key merge into one event (with a count).
    Distinct utterances/captions stay separate by keying on ``obs_id``.
    """

    config = config or DedupConfig()
    worker, kind, payload = observation.worker, observation.kind, observation.payload
    if kind in {"file.changed", "file.created", "file.deleted"}:
        path = _get(payload, "path", "")
        return f"{worker}:{kind}:{path}"
    if kind == "scene.motion":
        return f"{worker}:{kind}:{_get(payload, 'region', '')}"
    if kind == "scene.caption":
        return f"{worker}:{kind}:{_caption_dedup_value(payload, config.caption_key)}"
    # speech.segment and unknown kinds: keep each occurrence distinct.
    return f"{worker}:{kind}:{observation.obs_id}"


def should_keep(observation: Observation, config: IngestFilterConfig) -> bool:
    """FILTER stage: drop below-threshold noise."""

    kind, payload = observation.kind, observation.payload
    if kind == "speech.segment":
        if config.drop_empty_text and not str(_get(payload, "text", "")).strip():
            return False
        if float(_get(payload, "confidence", 1.0)) < config.min_speech_confidence:
            return False
    if kind == "scene.motion":
        if float(_get(payload, "level", 1.0)) < config.min_motion_level:
            return False
    return True


class IngestReducer:
    """Reflex-arc reducer: raw observations -> ranked, deduped events."""

    def __init__(
        self,
        store: EventStore,
        rules: RankingRules,
        *,
        filter_config: IngestFilterConfig | None = None,
        dedup_config: DedupConfig | None = None,
    ) -> None:
        self._store = store
        self._rules = rules
        self._filter = filter_config or IngestFilterConfig()
        self._dedup = dedup_config or DedupConfig()

    @property
    def store(self) -> EventStore:
        return self._store

    def reduce(self, observation: Observation, *, now: datetime | None = None) -> Event | None:
        """Reduce a single observation into a persisted event (or drop it)."""

        if not should_keep(observation, self._filter):
            return None

        now = now or datetime.now(UTC)
        dedup_key = dedup_key_for(observation, self._dedup)
        summary = summarize(observation)

        existing = self._store.find_by_dedup(dedup_key)
        if existing is not None and not existing.seen:
            # Merge a repeat into the still-unseen event.
            merged = existing.model_copy(
                update={
                    "obs_id": observation.obs_id,
                    "ts": observation.ts,
                    "count": existing.count + 1,
                    "last_ts": _max_ts(existing.last_ts, observation.ts),
                    "title": summary.title,
                    "summary": summary.summary,
                    "detail_available": summary.detail_available,
                }
            )
            self._store.save(merged, observation)
            return merged

        priority = initial_rank(observation, self._rules)
        event = Event(
            event_id=f"evt-{uuid4().hex}",
            obs_id=observation.obs_id,
            worker=observation.worker,
            kind=observation.kind,
            ts=observation.ts,
            dedup_key=dedup_key,
            count=1,
            first_ts=observation.ts,
            last_ts=observation.ts,
            initial_priority=priority,
            effective_priority=priority,
            title=summary.title,
            summary=summary.summary,
            detail_available=summary.detail_available,
        )
        self._store.save(event, observation)
        return event

    def drain_queue(self, queue: ObservationQueue, *, batch_size: int | None = None) -> list[Event]:
        """Consume pending observations from the queue into events."""

        events: list[Event] = []
        for queued in queue.drain(batch_size=batch_size):
            event = self.reduce(queued.observation)
            if event is not None:
                events.append(event)
        return events


def _get(payload: object, name: str, default: object = "") -> object:
    if isinstance(payload, dict):
        return payload.get(name, default)
    return getattr(payload, name, default)


def _max_ts(a: datetime, b: datetime) -> datetime:
    return a if a >= b else b
