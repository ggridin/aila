from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol

from aila.contracts import Command, Result, Severity


class BackendError(RuntimeError):
    pass


@dataclass(frozen=True)
class BackendObservation:
    kind: str
    payload: Any
    severity: Severity = Severity.info
    obs_id: str | None = None
    ts: datetime | None = None
    media_ref: str | None = None


class WorkerBackend(Protocol):
    def poll(self) -> tuple[BackendObservation, ...]:
        raise NotImplementedError

    def handle_command(self, command: Command) -> Any:
        raise NotImplementedError


class DeterministicFakeBackend:
    def __init__(
        self,
        *,
        observations: tuple[BackendObservation, ...] = (),
        command_results: dict[str, Any] | None = None,
    ) -> None:
        self._observations: deque[BackendObservation] = deque(observations)
        self._command_results = dict(command_results or {})
        self.commands: list[Command] = []

    def queue_observation(self, observation: BackendObservation) -> None:
        self._observations.append(observation)

    def set_command_result(self, verb: str, result: Any) -> None:
        self._command_results[verb] = result

    def poll(self) -> tuple[BackendObservation, ...]:
        observations = tuple(self._observations)
        self._observations.clear()
        return observations

    def handle_command(self, command: Command) -> Any:
        self.commands.append(command)
        if command.verb not in self._command_results:
            raise BackendError(f"no deterministic result configured for verb: {command.verb}")
        return self._command_results[command.verb]
