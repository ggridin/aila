from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Literal, Mapping

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from aila.contracts.payloads import OBSERVATION_KINDS_BY_WORKER, VALID_WORKERS, VERBS_BY_WORKER

Role = Literal["sensor", "effector"]
BackendKind = Literal["model", "deterministic"]
BackendPlacement = Literal["local", "lan"]

_ENV_REF_RE = re.compile(r"\$\{(?P<name>[A-Za-z_][A-Za-z0-9_]*)\}")
_SENSOR_WORKERS = frozenset({"mic", "camera", "filesystem"})
_EFFECTOR_WORKERS = frozenset({"speaker", "display"})
_DEVICE_SERVICES_BY_WORKER = {
    "mic": "audio-input",
    "camera": "camera-input",
    "filesystem": None,
    "speaker": None,
    "display": None,
}


class RequiresConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    os: tuple[str, ...] = ()
    python: tuple[str, ...] = ()
    models: tuple[str, ...] = ()


class BackendConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: BackendKind
    placement: BackendPlacement
    endpoint: str | None = None
    endpoint_token: str | None = None
    model: str | None = None

    @model_validator(mode="after")
    def validate_backend_shape(self) -> BackendConfig:
        if self.kind == "deterministic":
            if self.placement != "local":
                raise ValueError("deterministic backends must use local placement")
            if self.endpoint is not None:
                raise ValueError("deterministic backends must not configure endpoint")
            if self.endpoint_token is not None:
                raise ValueError("deterministic backends must not configure endpoint_token")
            if self.model is not None:
                raise ValueError("deterministic backends must not configure model")
            return self

        if self.model is None:
            raise ValueError("model backends must configure model")
        if self.placement == "lan" and self.endpoint is None:
            raise ValueError("lan model backends must configure endpoint")
        if self.placement == "local" and self.endpoint_token is not None:
            raise ValueError("local model backends must not configure endpoint_token")
        return self


class WorkerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    worker: str = Field(min_length=1)
    role: Role
    device_service: str | None = None
    backend: BackendConfig
    sampling: dict[str, Any] = Field(default_factory=dict)
    emits: tuple[str, ...] = ()
    verbs: tuple[str, ...] = ()
    requires: RequiresConfig = Field(default_factory=RequiresConfig)

    @model_validator(mode="after")
    def validate_worker_contract(self) -> WorkerConfig:
        if self.worker not in VALID_WORKERS:
            raise ValueError(f"invalid worker: {self.worker}")

        expected_role = "sensor" if self.worker in _SENSOR_WORKERS else "effector"
        if self.role != expected_role:
            raise ValueError(f"worker {self.worker} must use role {expected_role}")

        expected_device_service = _DEVICE_SERVICES_BY_WORKER[self.worker]
        if self.device_service != expected_device_service:
            raise ValueError(
                f"worker {self.worker} must configure device_service {expected_device_service!r}"
            )

        self._validate_no_duplicates("emits", self.emits)
        self._validate_no_duplicates("verbs", self.verbs)

        expected_emits = OBSERVATION_KINDS_BY_WORKER[self.worker]
        # sensor.status is a cross-cutting health signal any sensor may emit; it
        # is not required to be declared in `emits`.
        health_kind = frozenset({"sensor.status"})
        if (frozenset(self.emits) - health_kind) != (expected_emits - health_kind):
            raise ValueError(
                f"worker {self.worker} emits must match {sorted(expected_emits - health_kind)}"
            )

        expected_verbs = VERBS_BY_WORKER[self.worker]
        if frozenset(self.verbs) != expected_verbs:
            raise ValueError(f"worker {self.worker} verbs must match {sorted(expected_verbs)}")

        return self

    @staticmethod
    def _validate_no_duplicates(field_name: str, values: tuple[str, ...]) -> None:
        if len(values) != len(set(values)):
            raise ValueError(f"{field_name} must not contain duplicates")


def load_worker_config(
    path: Path,
    *,
    environ: Mapping[str, str] | None = None,
) -> WorkerConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("worker config must be a YAML mapping")
    substituted = substitute_env(raw, os.environ if environ is None else environ)
    return WorkerConfig.model_validate(substituted)


def substitute_env(value: Any, environ: Mapping[str, str]) -> Any:
    if isinstance(value, str):
        return _substitute_string(value, environ)
    if isinstance(value, list):
        return [substitute_env(item, environ) for item in value]
    if isinstance(value, dict):
        return {key: substitute_env(item, environ) for key, item in value.items()}
    return value


def _substitute_string(value: str, environ: Mapping[str, str]) -> str:
    def replace(match: re.Match[str]) -> str:
        name = match.group("name")
        if name not in environ:
            raise KeyError(f"missing environment variable for worker config: {name}")
        return environ[name]

    return _ENV_REF_RE.sub(replace, value)
