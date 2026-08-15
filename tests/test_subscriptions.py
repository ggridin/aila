from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from aila.contracts import Observation, Subscription
from aila.subscriptions import (
    load_subscriptions,
    matching_subscriptions,
    subscription_matches,
)


def test_load_subscriptions_parses_yaml_and_validates_queue_only(tmp_path: Path) -> None:
    path = tmp_path / "subscriptions.yaml"
    path.write_text(
        """
version: "1.0"
subscriptions:
  - worker: camera
    kind: scene.caption
    predicate: { labels: ["person"] }
    on_match: queue
  - worker: filesystem
    kind: file.changed
    predicate: { path~: "C:\\\\Users\\\\Grego\\\\projects\\\\**" }
    on_match: queue
""",
        encoding="utf-8",
    )

    subscriptions = load_subscriptions(path)

    assert subscriptions == (
        Subscription(
            worker="camera",
            kind="scene.caption",
            predicate={"labels": ["person"]},
            on_match="queue",
        ),
        Subscription(
            worker="filesystem",
            kind="file.changed",
            predicate={"path~": "C:\\Users\\Grego\\projects\\**"},
            on_match="queue",
        ),
    )


def test_load_subscriptions_rejects_non_queue_on_match(tmp_path: Path) -> None:
    path = tmp_path / "subscriptions.yaml"
    path.write_text(
        """
version: "1.0"
subscriptions:
  - worker: mic
    kind: speech.segment
    predicate: {}
    on_match: wake
""",
        encoding="utf-8",
    )

    with pytest.raises(ValidationError):
        load_subscriptions(path)


def test_load_subscriptions_rejects_unknown_predicate_operator(tmp_path: Path) -> None:
    path = tmp_path / "subscriptions.yaml"
    path.write_text(
        """
version: "1.0"
subscriptions:
  - worker: filesystem
    kind: file.changed
    predicate: { name~: "*.py" }
    on_match: queue
""",
        encoding="utf-8",
    )

    with pytest.raises(ValidationError):
        load_subscriptions(path)


def test_subscription_matches_worker_kind_and_payload_equality() -> None:
    observation = _caption_observation(labels=["person"])

    assert subscription_matches(
        Subscription(
            worker="camera",
            kind="scene.caption",
            predicate={"labels": ["person"], "caption": "person at desk"},
        ),
        observation,
    )
    assert not subscription_matches(
        Subscription(worker="camera", kind="scene.caption", predicate={"labels": ["cat"]}),
        observation,
    )
    assert not subscription_matches(
        Subscription(worker="mic", kind="speech.segment", predicate={}),
        observation,
    )
    assert not subscription_matches(
        Subscription(worker="camera", kind="scene.caption", predicate={"missing": None}),
        observation,
    )


def test_subscription_matches_path_glob_predicate() -> None:
    observation = _file_observation("C:\\Users\\Grego\\projects\\aila\\notes.txt")

    assert subscription_matches(
        Subscription(
            worker="filesystem",
            kind="file.changed",
            predicate={"path~": "C:\\Users\\Grego\\projects\\**"},
        ),
        observation,
    )
    assert not subscription_matches(
        Subscription(
            worker="filesystem",
            kind="file.changed",
            predicate={"path~": "C:\\Users\\Grego\\Downloads\\**"},
        ),
        observation,
    )


def test_matching_subscriptions_returns_only_matches_in_input_order() -> None:
    observation = _speech_observation("hello")
    subscriptions = (
        Subscription(worker="camera", kind="scene.caption", predicate={}),
        Subscription(worker="*", kind="speech.segment", predicate={"lang": "en"}),
        Subscription(worker="mic", kind="speech.segment", predicate={"text": "nope"}),
        Subscription(worker="mic", kind="*", predicate={}),
    )

    assert matching_subscriptions(subscriptions, observation) == (
        subscriptions[1],
        subscriptions[3],
    )


def _speech_observation(text: str) -> Observation:
    return Observation(
        obs_id="speech-1",
        worker="mic",
        kind="speech.segment",
        ts=datetime(2026, 7, 13, 12, 0, tzinfo=UTC),
        payload={
            "text": text,
            "lang": "en",
            "confidence": 0.9,
            "start_ms": 0,
            "end_ms": 10,
        },
    )


def _caption_observation(*, labels: list[str]) -> Observation:
    return Observation(
        obs_id="caption-1",
        worker="camera",
        kind="scene.caption",
        ts=datetime(2026, 7, 13, 12, 0, tzinfo=UTC),
        payload={"caption": "person at desk", "labels": labels, "boxes": []},
    )


def _file_observation(path: str) -> Observation:
    return Observation(
        obs_id="file-1",
        worker="filesystem",
        kind="file.changed",
        ts=datetime(2026, 7, 13, 12, 0, tzinfo=UTC),
        payload={
            "path": path,
            "change": "changed",
            "size": 12,
            "mtime": "2026-07-13T12:00:00Z",
        },
    )
