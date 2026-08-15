from __future__ import annotations

from collections.abc import Iterable, Mapping
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from aila.contracts import CONTRACT_VERSION, Observation, Subscription

ContractVersion = Literal["1.0"]


class SubscriptionsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: ContractVersion = CONTRACT_VERSION
    subscriptions: tuple[Subscription, ...] = Field(default_factory=tuple)

    @model_validator(mode="after")
    def validate_predicates(self) -> SubscriptionsConfig:
        for subscription in self.subscriptions:
            _validate_predicate(subscription.predicate)
        return self


def load_subscriptions(path: str | Path) -> tuple[Subscription, ...]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if raw is None:
        raw = {}
    if not isinstance(raw, Mapping):
        raise ValueError("subscriptions file must contain a mapping")

    return SubscriptionsConfig.model_validate(raw).subscriptions


def matching_subscriptions(
    subscriptions: Iterable[Subscription],
    observation: Observation,
) -> tuple[Subscription, ...]:
    return tuple(
        subscription
        for subscription in subscriptions
        if subscription_matches(subscription, observation)
    )


def subscription_matches(subscription: Subscription, observation: Observation) -> bool:
    if subscription.worker != "*" and subscription.worker != observation.worker:
        return False
    if subscription.kind != "*" and subscription.kind != observation.kind:
        return False

    payload = _payload_mapping(observation.payload)
    for field, expected in subscription.predicate.items():
        if field == "path~":
            actual = payload.get("path")
            if not isinstance(expected, str) or not isinstance(actual, str):
                return False
            if not fnmatchcase(actual, expected):
                return False
            continue

        if field not in payload or payload[field] != expected:
            return False

    return True


matches_subscription = subscription_matches


def _payload_mapping(payload: Any) -> Mapping[str, Any]:
    if hasattr(payload, "model_dump"):
        dumped = payload.model_dump(mode="json")
        if isinstance(dumped, Mapping):
            return dumped
    if isinstance(payload, Mapping):
        return payload
    raise TypeError("observation payload must be a mapping or Pydantic model")


def _validate_predicate(predicate: Mapping[str, Any]) -> None:
    for field, expected in predicate.items():
        if field == "path~":
            if not isinstance(expected, str):
                raise ValueError("path~ predicate value must be a glob string")
            continue
        if field.endswith("~"):
            raise ValueError(f"unsupported predicate operator: {field}")
