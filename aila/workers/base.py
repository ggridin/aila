from __future__ import annotations

import uuid
from datetime import UTC, datetime

from aila.contracts import Command, Error, Observation, Result, Severity
from aila.queue import ObservationQueue
from aila.workers.backends import BackendError, BackendObservation, WorkerBackend
from aila.workers.config import WorkerConfig


class WorkerRuntime:
    def __init__(self, config: WorkerConfig, backend: WorkerBackend) -> None:
        self.config = config
        self.backend = backend

    @property
    def worker(self) -> str:
        return self.config.worker


class SensorWorker(WorkerRuntime):
    def __init__(
        self,
        config: WorkerConfig,
        backend: WorkerBackend,
        queue: ObservationQueue,
    ) -> None:
        if config.role != "sensor":
            raise ValueError(f"worker {config.worker} is not a sensor")
        super().__init__(config, backend)
        self.queue = queue

    def poll_once(self) -> tuple[Observation, ...]:
        observations = tuple(self._build_observation(item) for item in self.backend.poll())
        for observation in observations:
            self.queue.append(observation)
        return observations

    def emit(
        self,
        *,
        kind: str,
        payload: object,
        severity: Severity = Severity.info,
        obs_id: str | None = None,
        ts: datetime | None = None,
        media_ref: str | None = None,
    ) -> Observation:
        observation = self._build_observation(
            BackendObservation(
                kind=kind,
                payload=payload,
                severity=severity,
                obs_id=obs_id,
                ts=ts,
                media_ref=media_ref,
            )
        )
        self.queue.append(observation)
        return observation

    def _build_observation(self, item: BackendObservation) -> Observation:
        return Observation(
            obs_id=item.obs_id or f"{self.worker}-{uuid.uuid4().hex}",
            worker=self.worker,
            kind=item.kind,
            ts=item.ts or datetime.now(UTC),
            severity=item.severity,
            payload=item.payload,
            media_ref=item.media_ref,
        )


class EffectorWorker(WorkerRuntime):
    def __init__(self, config: WorkerConfig, backend: WorkerBackend) -> None:
        if config.role != "effector":
            raise ValueError(f"worker {config.worker} is not an effector")
        super().__init__(config, backend)

    def handle_command(self, command: Command) -> Result:
        if command.worker != self.worker:
            return Result(
                id=command.id,
                ok=False,
                error=Error(
                    code="WRONG_WORKER",
                    message=f"command targets {command.worker}, not {self.worker}",
                ),
            )
        if command.verb not in self.config.verbs:
            return Result(
                id=command.id,
                ok=False,
                error=Error(
                    code="UNSUPPORTED_VERB",
                    message=f"worker {self.worker} does not support verb {command.verb}",
                ),
            )

        try:
            data = self.backend.handle_command(command)
        except BackendError as exc:
            return Result(
                id=command.id,
                ok=False,
                error=Error(code="BACKEND_ERROR", message=str(exc), retryable=True),
            )
        return Result(id=command.id, ok=True, data=data)
