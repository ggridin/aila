from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from aila.contracts import CONTRACT_VERSION
from aila.queue import ObservationQueue, QueuedObservation

ContractVersion = Literal["1.0"]


class DigestObservation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    v: ContractVersion = CONTRACT_VERSION
    obs_id: str = Field(min_length=1)
    kind: str = Field(min_length=1)
    ts: datetime
    severity: str = Field(min_length=1)
    payload: dict[str, Any]


class SensoryDigest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    v: ContractVersion = CONTRACT_VERSION
    by_worker: dict[str, tuple[DigestObservation, ...]] = Field(default_factory=dict)
    total_observations: int = Field(ge=0)


def build_sensory_digest(
    queue: ObservationQueue,
    *,
    batch_size: int | None = None,
) -> SensoryDigest:
    drained = queue.drain(batch_size=batch_size)
    groups: dict[str, list[DigestObservation]] = defaultdict(list)

    for item in sorted(drained, key=_presentation_key, reverse=True):
        observation = item.observation
        groups[observation.worker].append(
            DigestObservation(
                obs_id=observation.obs_id,
                kind=observation.kind,
                ts=observation.ts,
                severity=observation.severity.value,
                payload=observation.payload.model_dump(mode="json"),
            )
        )

    return SensoryDigest(
        by_worker={worker: tuple(observations) for worker, observations in groups.items()},
        total_observations=len(drained),
    )


def _presentation_key(item: QueuedObservation) -> tuple[datetime, str]:
    observation = item.observation
    return (observation.ts, observation.obs_id)
