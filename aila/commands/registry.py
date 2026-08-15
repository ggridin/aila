from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from pydantic import BaseModel, ValidationError

from aila.contracts import Command, Error, Result

CommandHandler = Callable[[Command], Any]
CommandInput = Command | Mapping[str, Any]

INVALID_COMMAND_ID = "<invalid-command>"


class BackendTimeout(TimeoutError):
    """Raised by command handlers when a backend exceeds the command deadline."""


class CommandDispatcher:
    def __init__(self, handlers: Mapping[tuple[str, str], CommandHandler]) -> None:
        self._handlers = dict(handlers)

    def dispatch(self, command: CommandInput) -> Result:
        try:
            validated = _validate_command(command)
        except (TypeError, ValueError, ValidationError) as exc:
            return _failure(
                _command_id(command),
                "BAD_ARGS",
                f"invalid command: {exc}",
            )

        handler = self._handlers.get((validated.worker, validated.verb))
        if handler is None:
            return _failure(
                validated.id,
                "UNSUPPORTED_VERB",
                f"worker {validated.worker} does not support verb {validated.verb}",
            )

        try:
            return _coerce_handler_result(validated, handler(validated))
        except BackendTimeout as exc:
            return _failure(
                validated.id,
                "BACKEND_TIMEOUT",
                str(exc) or "backend timed out while handling command",
                retryable=True,
            )
        except TimeoutError as exc:
            return _failure(
                validated.id,
                "BACKEND_TIMEOUT",
                str(exc) or "backend timed out while handling command",
                retryable=True,
            )
        except Exception as exc:
            return _failure(
                validated.id,
                "BACKEND_ERROR",
                str(exc) or exc.__class__.__name__,
                retryable=True,
            )


def dispatch_command(
    command: CommandInput,
    handlers: Mapping[tuple[str, str], CommandHandler],
) -> Result:
    return CommandDispatcher(handlers).dispatch(command)


def _validate_command(command: CommandInput) -> Command:
    if isinstance(command, Command):
        return Command.model_validate(command.model_dump(mode="python"))
    if not isinstance(command, Mapping):
        raise TypeError("command must be a Command or mapping")
    return Command.model_validate(dict(command))


def _coerce_handler_result(command: Command, value: Any) -> Result:
    if isinstance(value, Result):
        return Result.model_validate(value.model_dump(mode="python"))
    if isinstance(value, Error):
        return Result(id=command.id, ok=False, error=value)
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="python", by_alias=True)
    return Result(id=command.id, ok=True, data=value)


def _failure(
    command_id: str,
    code: str,
    message: str,
    *,
    retryable: bool = False,
) -> Result:
    return Result(
        id=command_id,
        ok=False,
        error=Error(code=code, message=message, retryable=retryable),
    )


def _command_id(command: CommandInput) -> str:
    if isinstance(command, Command):
        return command.id or INVALID_COMMAND_ID
    if isinstance(command, Mapping):
        raw_id = command.get("id")
        if isinstance(raw_id, str) and raw_id:
            return raw_id
    return INVALID_COMMAND_ID
