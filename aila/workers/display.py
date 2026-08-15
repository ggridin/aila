from __future__ import annotations

from typing import Protocol

from aila.contracts import ClearArgs, Command, RenderArgs
from aila.workers.backends import BackendObservation
from aila.workers.base import EffectorWorker
from aila.workers.config import WorkerConfig


class DisplayRenderer(Protocol):
    def render(self, args: RenderArgs) -> None:
        raise NotImplementedError

    def clear(self) -> None:
        raise NotImplementedError


class DisplayWorker(EffectorWorker):
    def __init__(self, config: WorkerConfig, renderer: DisplayRenderer) -> None:
        if config.worker != "display":
            raise ValueError(f"display worker cannot use config for {config.worker}")
        super().__init__(config, _DisplayBackend(renderer))


class _DisplayBackend:
    def __init__(self, renderer: DisplayRenderer) -> None:
        self._renderer = renderer

    def poll(self) -> tuple[BackendObservation, ...]:
        return ()

    def handle_command(self, command: Command) -> object:
        if command.verb == "render":
            self._renderer.render(RenderArgs.model_validate(command.args))
            return {"rendered": True}
        if command.verb == "clear":
            ClearArgs.model_validate(command.args)
            self._renderer.clear()
            return {"cleared": True}
        raise NotImplementedError(f"display worker does not support verb {command.verb}")
