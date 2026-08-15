from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from aila.contracts import ClearResult, Command, RenderArgs, RenderResult
from aila.workers.config import WorkerConfig
from aila.workers.display import DisplayWorker


@dataclass
class FakeRenderer:
    rendered: list[RenderArgs] = field(default_factory=list)
    clear_count: int = 0

    def render(self, args: RenderArgs) -> None:
        self.rendered.append(args)

    def clear(self) -> None:
        self.clear_count += 1


def test_display_worker_renders_with_fake_renderer() -> None:
    renderer = FakeRenderer()
    worker = DisplayWorker(_display_config(), renderer)
    command = Command(
        id="cmd-render",
        worker="display",
        verb="render",
        args={"kind": "markdown", "content": "# Hello", "region": "main"},
    )

    result = worker.handle_command(command)

    assert result.ok is True
    assert isinstance(result.data, RenderResult)
    assert result.data.rendered is True
    assert renderer.rendered == [
        RenderArgs(kind="markdown", content="# Hello", region="main")
    ]
    assert renderer.clear_count == 0


def test_display_worker_clears_with_fake_renderer() -> None:
    renderer = FakeRenderer()
    worker = DisplayWorker(_display_config(), renderer)
    command = Command(id="cmd-clear", worker="display", verb="clear", args={})

    result = worker.handle_command(command)

    assert result.ok is True
    assert isinstance(result.data, ClearResult)
    assert result.data.cleared is True
    assert renderer.rendered == []
    assert renderer.clear_count == 1


def test_display_worker_is_output_only_and_emits_no_observations() -> None:
    worker = DisplayWorker(_display_config(), FakeRenderer())

    assert worker.backend.poll() == ()


def test_display_worker_rejects_non_display_config() -> None:
    config = WorkerConfig.model_validate(
        {
            "worker": "speaker",
            "role": "effector",
            "backend": {"kind": "deterministic", "placement": "local"},
            "emits": [],
            "verbs": ["speak"],
        }
    )

    with pytest.raises(ValueError, match="display worker cannot use config for speaker"):
        DisplayWorker(config, FakeRenderer())


def _display_config() -> WorkerConfig:
    return WorkerConfig.model_validate(
        {
            "worker": "display",
            "role": "effector",
            "backend": {"kind": "deterministic", "placement": "local"},
            "emits": [],
            "verbs": ["render", "clear"],
        }
    )
