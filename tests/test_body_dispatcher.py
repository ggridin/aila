from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import yaml

from aila.body.dispatcher import digest_preview, dispatch_command, load_manifest, queue_peek, queue_status
from aila.contracts import Observation
from aila.queue import ObservationQueue


REPO_ROOT = Path(__file__).resolve().parents[1]
PLUGIN_ROOT = REPO_ROOT / "plugins" / "aila_body"


def test_dispatcher_loads_or_generates_manifest(tmp_path: Path) -> None:
    manifest = load_manifest(tmp_path / "contracts")

    assert "camera.snapshot" in manifest["verbs"]
    assert "speaker.speak" in manifest["verbs"]
    assert "display.render" in manifest["verbs"]
    assert "display.clear" in manifest["verbs"]


def test_dispatcher_queue_read_only_tools_do_not_drain(tmp_path: Path) -> None:
    queue_dir = tmp_path / "queue"
    queue = ObservationQueue(queue_dir)
    queue.append(
        Observation(
            obs_id="obs-1",
            worker="camera",
            kind="scene.caption",
            ts=datetime(2026, 1, 1, tzinfo=UTC),
            payload={"caption": "desk", "labels": ["desk"], "boxes": []},
        )
    )

    status = queue_status(queue_dir)
    peek = queue_peek(queue_dir, limit=10)
    digest = digest_preview(queue_dir, limit=10)

    assert status["pending"] == 1
    assert peek["mutated"] is False
    assert digest["mutated"] is False
    assert queue_status(queue_dir)["pending"] == 1


def test_dispatcher_validates_display_command() -> None:
    result = dispatch_command(
        worker="display",
        verb="render",
        command_args={"kind": "text", "content": "hello"},
        hermes_home=Path.home() / ".hermes",
    )

    assert result["ok"] is True
    assert result["data"] == {"rendered": True}


def test_dispatcher_rejects_bad_command_args() -> None:
    result = dispatch_command(
        worker="display",
        verb="render",
        command_args={"kind": "invalid", "content": "hello"},
        hermes_home=Path.home() / ".hermes",
    )

    assert result["ok"] is False
    assert result["error"]["code"] == "BAD_ARGS"


def test_dispatcher_camera_snapshot_returns_fresh_caption(tmp_path: Path) -> None:
    state_dir = tmp_path / "aila-body" / "state"
    state_dir.mkdir(parents=True)
    # Pre-publish a caption dated in the future so the poll sees it as "fresh".
    future = datetime(2999, 1, 1, tzinfo=UTC).isoformat()
    (state_dir / "camera-latest.json").write_text(
        json.dumps({"caption": "a tidy desk with a lamp", "motion": 0.2, "ts": future}),
        encoding="utf-8",
    )

    result = dispatch_command(
        worker="camera",
        verb="snapshot",
        command_args={},
        hermes_home=tmp_path,
    )

    assert result["ok"] is True
    assert result["data"]["scene.caption"]["caption"] == "a tidy desk with a lamp"
    assert (state_dir / "camera-look-request").exists()


def test_aila_body_plugin_metadata_lists_registered_tools() -> None:
    manifest = yaml.safe_load((PLUGIN_ROOT / "plugin.yaml").read_text(encoding="utf-8"))
    source = (PLUGIN_ROOT / "__init__.py").read_text(encoding="utf-8")

    expected_tools = {
        "aila_body_manifest",
        "aila_body_queue_status",
        "aila_body_queue_peek",
        "aila_body_digest_preview",
        "aila_camera_snapshot",
        "aila_speaker_speak",
        "aila_display_render",
        "aila_display_clear",
    }
    assert set(manifest["provides_tools"]) == expected_tools
    for tool_name in expected_tools:
        assert tool_name in source
