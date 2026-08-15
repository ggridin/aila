from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from aila.device_services import load_device_service_config
from aila.registry import RegistryConfig, load_registry_config
from aila.workers import WorkerConfig, load_worker_config


@dataclass(frozen=True)
class DependencyPlan:
    workers: tuple[str, ...]
    os: tuple[str, ...]
    python: tuple[str, ...]
    models: tuple[str, ...]


def plan_local_dependencies(
    registry_config: RegistryConfig | Mapping[str, object] | str | Path,
    *,
    workers_dir: str | Path,
    device_services_dir: str | Path | None = None,
) -> DependencyPlan:
    """Collect requires for enabled workers with local, non-deterministic backends."""
    registry = _load_registry(registry_config)
    selected_workers: list[str] = []
    os_packages: list[str] = []
    python_packages: list[str] = []
    models: list[str] = []

    for worker in registry.workers.enabled:
        config = load_worker_config(Path(workers_dir) / worker / "config.yaml")
        if not _should_provision(config):
            continue

        selected_workers.append(worker)
        _append_unique(os_packages, config.requires.os)
        _append_unique(python_packages, config.requires.python)
        _append_unique(models, config.requires.models)
        if device_services_dir is not None and config.device_service is not None:
            service_config = load_device_service_config(
                Path(device_services_dir) / config.device_service / "config.yaml"
            )
            _append_unique(os_packages, service_config.requires.os)
            _append_unique(python_packages, service_config.requires.python)
            _append_unique(models, service_config.requires.models)

    return DependencyPlan(
        workers=tuple(selected_workers),
        os=tuple(os_packages),
        python=tuple(python_packages),
        models=tuple(models),
    )


plan_enabled_local_dependencies = plan_local_dependencies


def _load_registry(
    config: RegistryConfig | Mapping[str, object] | str | Path,
) -> RegistryConfig:
    if isinstance(config, RegistryConfig):
        return config
    if isinstance(config, Mapping):
        return RegistryConfig.model_validate(dict(config))
    return load_registry_config(Path(config))


def _should_provision(config: WorkerConfig) -> bool:
    return config.backend.placement == "local" and config.backend.kind != "deterministic"


def _append_unique(target: list[str], values: tuple[str, ...]) -> None:
    for value in values:
        if value not in target:
            target.append(value)
