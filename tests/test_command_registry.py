from __future__ import annotations

from typing import Any

import pytest

from aila.commands import BackendTimeout, dispatch_command
from aila.contracts import Command, RenderArgs, RenderResult, Result, SpeakArgs


def test_dispatch_validates_args_before_calling_handler() -> None:
    calls: list[Command] = []

    def handler(command: Command) -> dict[str, object]:
        calls.append(command)
        return {"rendered": True}

    result = dispatch_command(
        {
            "id": "cmd-1",
            "worker": "display",
            "verb": "render",
            "args": {"kind": "text", "content": "hello"},
        },
        {("display", "render"): handler},
    )

    assert result.ok is True
    assert isinstance(calls[0].args, RenderArgs)
    assert isinstance(result.data, RenderResult)


@pytest.mark.parametrize(
    "command",
    [
        {
            "id": "cmd-1",
            "worker": "display",
            "verb": "render",
            "args": {"kind": "movie", "content": "hello"},
        },
        {
            "id": "cmd-2",
            "worker": "audio-input",
            "verb": "speak",
            "args": {"text": "hello"},
        },
        {
            "worker": "speaker",
            "verb": "speak",
            "args": {"text": "hello"},
        },
    ],
)
def test_dispatch_returns_bad_args_for_invalid_commands(command: dict[str, Any]) -> None:
    result = dispatch_command(command, {("display", "render"): lambda _: {"rendered": True}})

    assert result.ok is False
    assert result.error is not None
    assert result.error.code == "BAD_ARGS"


def test_dispatch_revalidates_constructed_command_args() -> None:
    command = Command.model_construct(
        id="cmd-1",
        worker="speaker",
        verb="speak",
        args={"rate": -1},
    )

    result = dispatch_command(command, {("speaker", "speak"): lambda _: {"duration_ms": 1}})

    assert result.ok is False
    assert result.error is not None
    assert result.error.code == "BAD_ARGS"


def test_dispatch_returns_unsupported_verb_when_no_handler_is_registered() -> None:
    result = dispatch_command(
        {"id": "cmd-1", "worker": "speaker", "verb": "speak", "args": {"text": "hello"}},
        {},
    )

    assert result.ok is False
    assert result.error is not None
    assert result.error.code == "UNSUPPORTED_VERB"


def test_dispatch_accepts_structured_result_from_handler() -> None:
    def handler(command: Command) -> Result:
        assert isinstance(command.args, SpeakArgs)
        return Result(id=command.id, ok=True, data={"duration_ms": 123})

    result = dispatch_command(
        {"id": "cmd-1", "worker": "speaker", "verb": "speak", "args": {"text": "hello"}},
        {("speaker", "speak"): handler},
    )

    assert result.ok is True
    assert result.data is not None


def test_dispatch_returns_backend_timeout_error() -> None:
    def handler(_: Command) -> object:
        raise BackendTimeout("deadline exceeded")

    result = dispatch_command(
        {"id": "cmd-1", "worker": "speaker", "verb": "speak", "args": {"text": "hello"}},
        {("speaker", "speak"): handler},
    )

    assert result.ok is False
    assert result.error is not None
    assert result.error.code == "BACKEND_TIMEOUT"
    assert result.error.retryable is True


def test_dispatch_returns_backend_error_for_handler_exceptions() -> None:
    def handler(_: Command) -> object:
        raise RuntimeError("backend failed")

    result = dispatch_command(
        {"id": "cmd-1", "worker": "speaker", "verb": "speak", "args": {"text": "hello"}},
        {("speaker", "speak"): handler},
    )

    assert result.ok is False
    assert result.error is not None
    assert result.error.code == "BACKEND_ERROR"
    assert result.error.retryable is True
