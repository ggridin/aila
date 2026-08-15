from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from aila.contracts.payloads import VALID_WORKERS
from aila.device_services import (
    VALID_DEVICE_SERVICES,
    DeviceServiceConfig,
    required_device_services_for_workers,
)
from aila.device_services.config import DEVICE_SERVICE_CONSUMER
from aila.workers import WorkerConfig, load_worker_config
from aila.workers.config import substitute_env


class WorkersRegistryConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: tuple[str, ...] = Field(default_factory=tuple)

    @model_validator(mode="after")
    def validate_enabled_workers(self) -> WorkersRegistryConfig:
        if len(self.enabled) != len(set(self.enabled)):
            raise ValueError("workers.enabled must not contain duplicates")

        for worker in self.enabled:
            if worker in VALID_DEVICE_SERVICES:
                raise ValueError(f"device service {worker} must not appear in workers.enabled")
            if worker not in VALID_WORKERS:
                raise ValueError(f"invalid worker in workers.enabled: {worker}")
        return self


class RegistryConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    workers: WorkersRegistryConfig


@dataclass(frozen=True)
class Registry:
    enabled_workers: tuple[str, ...]
    required_device_services: tuple[str, ...]


def load_registry_config(
    path: Path,
    *,
    environ: Mapping[str, str] | None = None,
) -> RegistryConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("registry config must be a YAML mapping")
    substituted = substitute_env(raw, os.environ if environ is None else environ)
    return RegistryConfig.model_validate(substituted)


def validate_registry(
    config: RegistryConfig | Mapping[str, object],
    *,
    worker_configs: Mapping[str, WorkerConfig] | Sequence[str] | None = None,
    device_service_configs: Mapping[str, DeviceServiceConfig] | Sequence[str] | None = None,
    available_workers: Sequence[str] | None = None,
    available_device_services: Sequence[str] | None = None,
) -> Registry:
    registry_config = _coerce_registry_config(config)
    enabled_workers = registry_config.workers.enabled

    worker_ids = _available_worker_ids(worker_configs, available_workers)
    if worker_ids is not None:
        missing_workers = tuple(worker for worker in enabled_workers if worker not in worker_ids)
        if missing_workers:
            raise ValueError(f"missing enabled worker config: {', '.join(missing_workers)}")

    required_services = required_device_services_for_workers(list(enabled_workers))
    service_ids = _available_service_ids(device_service_configs, available_device_services)
    if service_ids is not None:
        missing_services = tuple(
            service for service in required_services if service not in service_ids
        )
        if missing_services:
            raise ValueError(
                f"missing required device-service config: {', '.join(missing_services)}"
            )

    _validate_config_id_consistency(worker_configs, device_service_configs, required_services)
    return Registry(
        enabled_workers=enabled_workers,
        required_device_services=required_services,
    )


def validate_registry_files(
    config: RegistryConfig | Mapping[str, object],
    *,
    workers_dir: Path,
    device_services_dir: Path,
) -> Registry:
    registry_config = _coerce_registry_config(config)
    worker_configs: dict[str, WorkerConfig] = {}
    for worker in registry_config.workers.enabled:
        config_path = Path(workers_dir) / worker / "config.yaml"
        if not config_path.is_file():
            raise ValueError(f"missing enabled worker config: {worker}")
        worker_configs[worker] = load_worker_config(config_path)

    device_service_configs: dict[str, DeviceServiceConfig] = {}
    for service in required_device_services_for_workers(list(registry_config.workers.enabled)):
        config_path = Path(device_services_dir) / service / "config.yaml"
        if not config_path.is_file():
            raise ValueError(f"missing required device-service config: {service}")
        from aila.device_services import load_device_service_config

        device_service_configs[service] = load_device_service_config(config_path)

    return validate_registry(
        registry_config,
        worker_configs=worker_configs,
        device_service_configs=device_service_configs,
    )


def _coerce_registry_config(config: RegistryConfig | Mapping[str, object]) -> RegistryConfig:
    if isinstance(config, RegistryConfig):
        return config
    return RegistryConfig.model_validate(dict(config))


def _available_worker_ids(
    worker_configs: Mapping[str, WorkerConfig] | Sequence[str] | None,
    available_workers: Sequence[str] | None,
) -> frozenset[str] | None:
    if worker_configs is not None:
        if isinstance(worker_configs, Mapping):
            return frozenset(worker_configs)
        return frozenset(worker_configs)
    if available_workers is not None:
        return frozenset(available_workers)
    return None


def _available_service_ids(
    device_service_configs: Mapping[str, DeviceServiceConfig] | Sequence[str] | None,
    available_device_services: Sequence[str] | None,
) -> frozenset[str] | None:
    if device_service_configs is not None:
        if isinstance(device_service_configs, Mapping):
            return frozenset(device_service_configs)
        return frozenset(device_service_configs)
    if available_device_services is not None:
        return frozenset(available_device_services)
    return None


def _validate_config_id_consistency(
    worker_configs: Mapping[str, WorkerConfig] | Sequence[str] | None,
    device_service_configs: Mapping[str, DeviceServiceConfig] | Sequence[str] | None,
    required_services: tuple[str, ...],
) -> None:
    if isinstance(worker_configs, Mapping):
        for worker, config in worker_configs.items():
            if config.worker != worker:
                raise ValueError(f"worker config key {worker} does not match {config.worker}")

    if isinstance(device_service_configs, Mapping):
        for service, config in device_service_configs.items():
            if config.service != service:
                raise ValueError(
                    f"device-service config key {service} does not match {config.service}"
                )
        for service in required_services:
            config = device_service_configs.get(service)
            if config is None:
                continue
            expected_consumer = DEVICE_SERVICE_CONSUMER[service]
            if config.consumer != expected_consumer:
                raise ValueError(
                    f"device service {service} must configure consumer {expected_consumer!r}"
                )
