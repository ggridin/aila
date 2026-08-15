from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal, Mapping

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from aila.contracts.payloads import VALID_WORKERS
from aila.workers.config import RequiresConfig, substitute_env

DeviceServiceKind = Literal["audio", "camera"]

DEVICE_SERVICE_BY_WORKER: dict[str, str] = {
    "mic": "audio-input",
    "camera": "camera-input",
}
DEVICE_SERVICE_CONSUMER: dict[str, str] = {
    service: worker for worker, service in DEVICE_SERVICE_BY_WORKER.items()
}
DEVICE_SERVICE_KIND: dict[str, DeviceServiceKind] = {
    "audio-input": "audio",
    "camera-input": "camera",
}
VALID_DEVICE_SERVICES: frozenset[str] = frozenset(DEVICE_SERVICE_CONSUMER)


class DeviceServiceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    service: str = Field(min_length=1)
    kind: DeviceServiceKind | None = None
    consumer: str = Field(min_length=1)
    device: str | None = None
    capture: dict[str, Any] = Field(default_factory=dict)
    requires: RequiresConfig = Field(default_factory=RequiresConfig)

    @model_validator(mode="after")
    def validate_device_service_contract(self) -> DeviceServiceConfig:
        if self.service in VALID_WORKERS:
            raise ValueError(f"device service {self.service} conflicts with a worker id")
        if self.service not in VALID_DEVICE_SERVICES:
            raise ValueError(f"invalid device service: {self.service}")

        expected_consumer = DEVICE_SERVICE_CONSUMER[self.service]
        if self.consumer != expected_consumer:
            raise ValueError(
                f"device service {self.service} must configure consumer {expected_consumer!r}"
            )

        expected_kind = DEVICE_SERVICE_KIND[self.service]
        if self.kind is None:
            self.kind = expected_kind
        elif self.kind != expected_kind:
            raise ValueError(
                f"device service {self.service} must configure kind {expected_kind!r}"
            )

        return self


def audio_input_config(**overrides: Any) -> DeviceServiceConfig:
    return _device_service_config("audio-input", **overrides)


def camera_input_config(**overrides: Any) -> DeviceServiceConfig:
    return _device_service_config("camera-input", **overrides)


def load_device_service_config(
    path: Path,
    *,
    environ: Mapping[str, str] | None = None,
) -> DeviceServiceConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("device-service config must be a YAML mapping")
    substituted = substitute_env(raw, os.environ if environ is None else environ)
    return DeviceServiceConfig.model_validate(substituted)


def required_device_services_for_workers(workers: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    required: list[str] = []
    for worker in workers:
        service = DEVICE_SERVICE_BY_WORKER.get(worker)
        if service is not None and service not in required:
            required.append(service)
    return tuple(required)


def _device_service_config(service: str, **overrides: Any) -> DeviceServiceConfig:
    data: dict[str, Any] = {
        "service": service,
        "consumer": DEVICE_SERVICE_CONSUMER[service],
        "kind": DEVICE_SERVICE_KIND[service],
    }
    data.update(overrides)
    return DeviceServiceConfig.model_validate(data)
