from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from tools.registry import tool_error, tool_result

TOOLSET = "aila_body"


def _run_aila_body(*args: str) -> dict[str, Any]:
    completed = subprocess.run(
        [_aila_body_command(), *args],
        text=True,
        capture_output=True,
        check=False,
    )
    output = (completed.stdout or "").strip()
    if completed.returncode != 0:
        detail = output or (completed.stderr or "aila-body command failed").strip()
        raise RuntimeError(detail)
    try:
        loaded = json.loads(output)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"aila-body returned invalid JSON: {exc}: {output[:500]}") from exc
    if not isinstance(loaded, dict):
        raise RuntimeError("aila-body returned non-object JSON")
    return loaded


def _aila_body_command() -> str:
    for candidate in (
        Path.home() / ".hermes" / "venv" / "bin" / "aila-body",
        Path.home() / ".local" / "bin" / "aila-body",
    ):
        if candidate.is_file():
            return str(candidate)
    return "aila-body"


def _handle_manifest(args: dict, **kwargs) -> str:
    return tool_result(_run_aila_body("manifest"))


def _handle_queue_status(args: dict, **kwargs) -> str:
    return tool_result(_run_aila_body("queue-status"))


def _handle_queue_peek(args: dict, **kwargs) -> str:
    limit = int(args.get("limit", 10))
    return tool_result(_run_aila_body("queue-peek", "--limit", str(limit)))


def _handle_digest_preview(args: dict, **kwargs) -> str:
    limit = int(args.get("limit", 20))
    return tool_result(_run_aila_body("digest-preview", "--limit", str(limit)))


def _handle_command(worker: str, verb: str, command_args: dict[str, Any]) -> str:
    try:
        return tool_result(
            _run_aila_body(
                "command",
                worker,
                verb,
                "--args",
                json.dumps(command_args, ensure_ascii=False),
            )
        )
    except Exception as exc:
        return tool_error(str(exc))


def _handle_camera_snapshot(args: dict, **kwargs) -> str:
    return _handle_command("camera", "snapshot", {})


def _handle_speaker_speak(args: dict, **kwargs) -> str:
    command_args = {"text": args["text"]}
    if args.get("voice") is not None:
        command_args["voice"] = args["voice"]
    if args.get("rate") is not None:
        command_args["rate"] = args["rate"]
    return _handle_command("speaker", "speak", command_args)


def _handle_display_render(args: dict, **kwargs) -> str:
    command_args = {"kind": args["kind"], "content": args["content"]}
    if args.get("region") is not None:
        command_args["region"] = args["region"]
    return _handle_command("display", "render", command_args)


def _handle_display_clear(args: dict, **kwargs) -> str:
    return _handle_command("display", "clear", {})


MANIFEST_SCHEMA = {
    "name": "aila_body_manifest",
    "description": "Read the AILA body contract manifest that defines workers, verbs, and observation kinds.",
    "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
}

QUEUE_STATUS_SCHEMA = {
    "name": "aila_body_queue_status",
    "description": "Read non-mutating counts for AILA body observation queue pending/inflight/archive files.",
    "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
}

QUEUE_PEEK_SCHEMA = {
    "name": "aila_body_queue_peek",
    "description": "Read pending AILA body observations without draining or mutating the queue.",
    "parameters": {
        "type": "object",
        "properties": {"limit": {"type": "integer", "minimum": 1, "maximum": 100, "default": 10}},
        "additionalProperties": False,
    },
}

DIGEST_PREVIEW_SCHEMA = {
    "name": "aila_body_digest_preview",
    "description": "Build a non-mutating preview grouped like the AILA sensory digest from pending observations.",
    "parameters": {
        "type": "object",
        "properties": {"limit": {"type": "integer", "minimum": 1, "maximum": 100, "default": 20}},
        "additionalProperties": False,
    },
}

CAMERA_SNAPSHOT_SCHEMA = {
    "name": "aila_camera_snapshot",
    "description": "Ask the AILA camera worker for a snapshot/caption result.",
    "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
}

SPEAKER_SPEAK_SCHEMA = {
    "name": "aila_speaker_speak",
    "description": "Ask the AILA speaker worker to speak text using the local Piper voice.",
    "parameters": {
        "type": "object",
        "properties": {
            "text": {"type": "string", "minLength": 1, "description": "Text to speak."},
            "voice": {"type": ["string", "null"], "description": "Optional voice name."},
            "rate": {"type": "number", "exclusiveMinimum": 0, "default": 1.0},
        },
        "required": ["text"],
        "additionalProperties": False,
    },
}

DISPLAY_RENDER_SCHEMA = {
    "name": "aila_display_render",
    "description": "Ask the AILA display worker to render text, markdown, or an image reference.",
    "parameters": {
        "type": "object",
        "properties": {
            "kind": {"type": "string", "enum": ["text", "markdown", "image"]},
            "content": {"type": "string", "minLength": 1},
            "region": {"type": ["string", "null"]},
        },
        "required": ["kind", "content"],
        "additionalProperties": False,
    },
}

DISPLAY_CLEAR_SCHEMA = {
    "name": "aila_display_clear",
    "description": "Ask the AILA display worker to clear its output.",
    "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
}


def _check_available() -> bool:
    try:
        subprocess.run(["aila-body", "queue-status"], text=True, capture_output=True, check=True)
        return True
    except Exception:
        return False


def register(ctx) -> None:
    tools = (
        ("aila_body_manifest", MANIFEST_SCHEMA, _handle_manifest, "📜"),
        ("aila_body_queue_status", QUEUE_STATUS_SCHEMA, _handle_queue_status, "📊"),
        ("aila_body_queue_peek", QUEUE_PEEK_SCHEMA, _handle_queue_peek, "👁️"),
        ("aila_body_digest_preview", DIGEST_PREVIEW_SCHEMA, _handle_digest_preview, "🧠"),
        ("aila_camera_snapshot", CAMERA_SNAPSHOT_SCHEMA, _handle_camera_snapshot, "📷"),
        ("aila_speaker_speak", SPEAKER_SPEAK_SCHEMA, _handle_speaker_speak, "🔊"),
        ("aila_display_render", DISPLAY_RENDER_SCHEMA, _handle_display_render, "🖥️"),
        ("aila_display_clear", DISPLAY_CLEAR_SCHEMA, _handle_display_clear, "🧹"),
    )
    for name, schema, handler, emoji in tools:
        ctx.register_tool(
            name=name,
            toolset=TOOLSET,
            schema=schema,
            handler=handler,
            check_fn=_check_available,
            emoji=emoji,
        )
