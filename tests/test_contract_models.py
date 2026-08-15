from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from aila.contracts import (
    CONTRACT_VERSION,
    ClearArgs,
    ClearResult,
    Command,
    Error,
    FileChangedPayload,
    Observation,
    RenderArgs,
    RenderResult,
    Result,
    SceneCaptionPayload,
    SnapshotArgs,
    SnapshotResult,
    SpeakArgs,
    SpeakResult,
    SpeechSegmentPayload,
    Subscription,
)


def test_command_validates_worker_verb_and_args() -> None:
    command = Command(
        id="cmd-1",
        worker="speaker",
        verb="speak",
        args={"text": "hello", "rate": 1.25},
    )

    assert command.v == CONTRACT_VERSION
    assert isinstance(command.args, SpeakArgs)


@pytest.mark.parametrize(
    ("worker", "verb", "args"),
    [
        ("audio-input", "speak", {"text": "hello"}),
        ("mic", "speak", {"text": "hello"}),
        ("speaker", "whisper", {"text": "hello"}),
        ("speaker", "speak", {"rate": 1.0}),
        ("display", "render", {"kind": "movie", "content": "x"}),
        ("display", "clear", {"unexpected": True}),
    ],
)
def test_command_rejects_invalid_worker_verbs_and_args(
    worker: str,
    verb: str,
    args: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        Command(id="cmd-1", worker=worker, verb=verb, args=args)


def test_observation_validates_kind_and_payload() -> None:
    observation = Observation(
        obs_id="obs-1",
        worker="mic",
        kind="speech.segment",
        ts=datetime.now(timezone.utc),
        payload={
            "text": "hello",
            "lang": "en",
            "confidence": 0.9,
            "start_ms": 10,
            "end_ms": 20,
        },
    )

    assert observation.v == CONTRACT_VERSION
    assert isinstance(observation.payload, SpeechSegmentPayload)


@pytest.mark.parametrize(
    ("worker", "kind", "payload"),
    [
        ("camera-input", "scene.caption", {"caption": "desk"}),
        ("mic", "scene.caption", {"caption": "desk", "labels": [], "boxes": []}),
        ("camera", "scene.caption", {"labels": [], "boxes": []}),
        (
            "filesystem",
            "file.changed",
            {
                "path": "relative.txt",
                "change": "changed",
                "size": 1,
                "mtime": "2026-07-13T12:00:00Z",
            },
        ),
        (
            "filesystem",
            "file.created",
            {
                "path": "/tmp/a.txt",
                "change": "deleted",
                "size": 0,
                "mtime": "2026-07-13T12:00:00Z",
            },
        ),
    ],
)
def test_observation_rejects_invalid_worker_kind_and_payload(
    worker: str,
    kind: str,
    payload: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        Observation(
            obs_id="obs-1",
            worker=worker,
            kind=kind,
            ts=datetime.now(timezone.utc),
            payload=payload,
        )


def test_all_payload_and_result_types_are_available_and_strict() -> None:
    caption = SceneCaptionPayload(
        caption="person at desk",
        labels=["person"],
        boxes=[{"label": "person", "x": 1, "y": 2, "w": 3, "h": 4, "score": 0.5}],
    )

    assert SnapshotArgs().model_dump() == {}
    assert SnapshotResult(**{"scene.caption": caption}).scene_caption == caption
    assert isinstance(FileChangedPayload(
        path="/tmp/a.txt",
        change="changed",
        size=3,
        mtime="2026-07-13T12:00:00Z",
    ), FileChangedPayload)
    assert isinstance(RenderArgs(kind="markdown", content="# hi"), RenderArgs)
    assert isinstance(RenderResult(rendered=True), RenderResult)
    assert isinstance(ClearArgs(), ClearArgs)
    assert isinstance(ClearResult(cleared=True), ClearResult)
    assert isinstance(SpeakResult(duration_ms=100), SpeakResult)

    with pytest.raises(ValidationError):
        SceneCaptionPayload(caption="x", labels=[], boxes=[], raw_frame="/tmp/raw.jpg")


def test_result_validates_success_and_failure_shapes() -> None:
    success = Result(id="cmd-1", ok=True, data={"duration_ms": 123})
    failure = Result(
        id="cmd-2",
        ok=False,
        error=Error(code="BAD_ARGS", message="bad command"),
    )

    assert success.v == CONTRACT_VERSION
    assert isinstance(success.data, SpeakResult)
    assert failure.error is not None

    with pytest.raises(ValidationError):
        Result(id="cmd-3", ok=True, error=Error(code="NOPE", message="bad"))
    with pytest.raises(ValidationError):
        Result(id="cmd-4", ok=False)
    with pytest.raises(ValidationError):
        Result(id="cmd-5", ok=True, data={"duration_ms": -1})


def test_subscription_validates_worker_kind_pairs_and_queue_only() -> None:
    assert Subscription(worker="camera", kind="scene.caption").on_match == "queue"
    assert Subscription(worker="*", kind="speech.segment").predicate == {}

    with pytest.raises(ValidationError):
        Subscription(worker="camera", kind="speech.segment")
    with pytest.raises(ValidationError):
        Subscription(worker="display", kind="*")
    with pytest.raises(ValidationError):
        Subscription(worker="mic", kind="speech.segment", on_match="wake")
