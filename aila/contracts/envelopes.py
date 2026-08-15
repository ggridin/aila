from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from aila.contracts.payloads import (
    COMMAND_ARG_MODELS,
    CONTRACT_VERSION,
    OBSERVATION_KINDS_BY_WORKER,
    OBSERVATION_PAYLOAD_MODELS,
    RESULT_DATA_MODELS,
    VALID_OBSERVATION_KINDS,
    VALID_WORKERS,
    VERBS_BY_WORKER,
    Severity,
)

ContractVersion = Literal["1.0"]


class Error(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str = Field(min_length=1)
    message: str = Field(min_length=1)
    retryable: bool = False


class Command(BaseModel):
    model_config = ConfigDict(extra="forbid")

    v: ContractVersion = CONTRACT_VERSION
    id: str = Field(min_length=1)
    worker: str = Field(min_length=1)
    verb: str = Field(min_length=1)
    args: Any = Field(default_factory=dict)
    deadline_ms: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def validate_worker_verb_and_args(self) -> Command:
        if self.worker not in VALID_WORKERS:
            raise ValueError(f"invalid worker: {self.worker}")
        if self.verb not in VERBS_BY_WORKER[self.worker]:
            raise ValueError(f"invalid verb for worker {self.worker}: {self.verb}")

        model = COMMAND_ARG_MODELS[(self.worker, self.verb)]
        self.args = model.model_validate(self.args)
        return self


class Result(BaseModel):
    model_config = ConfigDict(extra="forbid")

    v: ContractVersion = CONTRACT_VERSION
    id: str = Field(min_length=1)
    ok: bool
    data: Any | None = None
    error: Error | None = None

    @model_validator(mode="after")
    def validate_result_shape(self) -> Result:
        if self.ok:
            if self.error is not None:
                raise ValueError("successful results must not include error")
            if self.data is not None:
                self.data = _validate_result_data(self.data)
            return self

        if self.error is None:
            raise ValueError("failed results must include error")
        if self.data is not None:
            raise ValueError("failed results must not include data")
        return self


class Observation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    v: ContractVersion = CONTRACT_VERSION
    obs_id: str = Field(min_length=1)
    worker: str = Field(min_length=1)
    kind: str = Field(min_length=1)
    ts: datetime
    severity: Severity = Severity.info
    payload: Any
    media_ref: str | None = None

    @model_validator(mode="after")
    def validate_worker_kind_and_payload(self) -> Observation:
        if self.worker not in VALID_WORKERS:
            raise ValueError(f"invalid worker: {self.worker}")
        if self.kind not in OBSERVATION_KINDS_BY_WORKER[self.worker]:
            raise ValueError(f"invalid observation kind for worker {self.worker}: {self.kind}")

        model = OBSERVATION_PAYLOAD_MODELS[self.kind]
        self.payload = model.model_validate(self.payload)
        return self


class Subscription(BaseModel):
    model_config = ConfigDict(extra="forbid")

    worker: str | Literal["*"]
    kind: str | Literal["*"]
    predicate: dict[str, Any] = Field(default_factory=dict)
    on_match: Literal["queue"] = "queue"

    @model_validator(mode="after")
    def validate_subscription_target(self) -> Subscription:
        if self.worker != "*" and self.worker not in VALID_WORKERS:
            raise ValueError(f"invalid worker: {self.worker}")
        if self.kind != "*" and self.kind not in VALID_OBSERVATION_KINDS:
            raise ValueError(f"invalid observation kind: {self.kind}")
        if (
            self.worker != "*"
            and self.kind == "*"
            and not OBSERVATION_KINDS_BY_WORKER[self.worker]
        ):
            raise ValueError(f"worker {self.worker} does not emit observations")
        if (
            self.worker != "*"
            and self.kind != "*"
            and self.kind not in OBSERVATION_KINDS_BY_WORKER[self.worker]
        ):
            raise ValueError(f"invalid observation kind for worker {self.worker}: {self.kind}")
        return self


def _validate_result_data(data: Any) -> BaseModel:
    errors: list[str] = []
    for model in RESULT_DATA_MODELS:
        try:
            return model.model_validate(data)
        except ValueError as exc:
            errors.append(f"{model.__name__}: {exc}")
    raise ValueError("data does not match any known result payload: " + "; ".join(errors))
