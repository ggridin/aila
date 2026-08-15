from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import ValidationError

from aila.contracts import Command, Error, Observation, Result, Severity
from aila.contracts.manifest import generate_contract_manifest
from aila.contracts.payloads import ClearResult, RenderResult, SpeakResult
from aila.queue import ObservationQueue

DEFAULT_HERMES_HOME = Path.home() / ".hermes"


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.action == "manifest":
            payload = load_manifest(_contracts_dir(args.hermes_home))
        elif args.action == "queue-status":
            payload = queue_status(_queue_dir(args.hermes_home))
        elif args.action == "queue-peek":
            payload = queue_peek(_queue_dir(args.hermes_home), limit=args.limit)
        elif args.action == "digest-preview":
            payload = digest_preview(_queue_dir(args.hermes_home), limit=args.limit)
        elif args.action == "reflex-digest":
            payload = reflex_digest_preview(_reflex_dir(args.hermes_home))
        elif args.action == "reflex-expand":
            payload = reflex_expand_preview(_reflex_dir(args.hermes_home), event_id=args.event_id)
        elif args.action == "command":
            payload = dispatch_command(
                worker=args.worker,
                verb=args.verb,
                command_args=_json_arg(args.args),
                hermes_home=args.hermes_home,
            )
        else:
            raise ValueError(f"unsupported action: {args.action}")
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False))
        return 1

    print(json.dumps(payload, ensure_ascii=False, default=str))
    return 0


def load_manifest(contracts_dir: Path) -> dict[str, Any]:
    manifest_path = contracts_dir / "manifest.json"
    if not manifest_path.is_file():
        generate_contract_manifest(contracts_dir)
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def queue_status(queue_dir: Path) -> dict[str, Any]:
    queue = ObservationQueue(queue_dir)
    return {
        "ok": True,
        "queue_dir": str(queue.root),
        "pending": _count_json(queue.pending_dir),
        "inflight": _count_json(queue.inflight_dir),
        "archive": _count_json(queue.archive_dir),
    }


def queue_peek(queue_dir: Path, *, limit: int) -> dict[str, Any]:
    queue = ObservationQueue(queue_dir)
    items = []
    for path in sorted(queue.pending_dir.glob("*.json"))[:limit]:
        try:
            observation = Observation.model_validate_json(path.read_text(encoding="utf-8"))
            items.append(_observation_summary(observation, path=path))
        except (OSError, ValidationError, ValueError) as exc:
            items.append({"path": str(path), "error": str(exc)})
    return {"ok": True, "pending": items, "count": len(items), "mutated": False}


def digest_preview(queue_dir: Path, *, limit: int) -> dict[str, Any]:
    queue = ObservationQueue(queue_dir)
    by_worker: dict[str, list[dict[str, Any]]] = {}
    total = 0
    for path in sorted(queue.pending_dir.glob("*.json"))[:limit]:
        try:
            observation = Observation.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValidationError, ValueError):
            continue
        by_worker.setdefault(observation.worker, []).append(_observation_summary(observation, path=path))
        total += 1
    return {"ok": True, "mutated": False, "total_observations": total, "by_worker": by_worker}


def dispatch_command(
    *,
    worker: str,
    verb: str,
    command_args: dict[str, Any],
    hermes_home: Path,
) -> dict[str, Any]:
    command_id = f"cmd-{uuid4().hex}"
    try:
        command = Command(id=command_id, worker=worker, verb=verb, args=command_args)
    except (TypeError, ValueError, ValidationError) as exc:
        return Result(
            id=command_id,
            ok=False,
            error=Error(code="BAD_ARGS", message=f"invalid command: {exc}"),
        ).model_dump(mode="json")
    if (worker, verb) == ("speaker", "speak"):
        return _speak(command, hermes_home)
    if (worker, verb) == ("display", "render"):
        return _success(command, RenderResult(rendered=True).model_dump(mode="json"))
    if (worker, verb) == ("display", "clear"):
        return _success(command, ClearResult(cleared=True).model_dump(mode="json"))
    if (worker, verb) == ("camera", "snapshot"):
        return _camera_snapshot(command, hermes_home)
    return Result(
        id=command.id,
        ok=False,
        error=Error(code="UNSUPPORTED_VERB", message=f"unsupported body command: {worker}.{verb}"),
    ).model_dump(mode="json")


def _speak(command: Command, hermes_home: Path) -> dict[str, Any]:
    result = _run_worker_command("speaker", command, hermes_home)
    return result.model_dump(mode="json", by_alias=True)


def _camera_snapshot(command: Command, hermes_home: Path) -> dict[str, Any]:
    # Deliberate "look": ask the running camera worker (which owns /dev/video0)
    # to capture and describe a fresh frame via a file-based request, then wait
    # for it to publish an updated caption to its state file.
    state_dir = hermes_home / "aila-body" / "state"
    state_path = state_dir / "camera-latest.json"
    request_path = state_dir / "camera-look-request"
    wait_seconds = 12.0
    poll_interval = 0.25

    requested_at = datetime.now(UTC)
    try:
        state_dir.mkdir(parents=True, exist_ok=True)
        request_path.write_text(requested_at.isoformat(), encoding="utf-8")
    except OSError as exc:
        return Result(
            id=command.id,
            ok=False,
            error=Error(code="BACKEND_ERROR", message=f"could not request look: {exc}", retryable=True),
        ).model_dump(mode="json", by_alias=True)

    deadline = time.monotonic() + wait_seconds
    fresh = None
    while time.monotonic() < deadline:
        latest = _read_latest_camera_state(state_path)
        if latest is not None:
            ts = _parse_iso(latest.get("ts"))
            if ts is not None and ts >= requested_at:
                fresh = latest
                break
        time.sleep(poll_interval)

    if fresh is not None:
        caption = str(fresh.get("caption", "")).strip() or "scene captured (no description available)"
        return _success(
            command,
            {"scene.caption": {"caption": caption, "labels": [], "boxes": []}},
        )

    # No fresh frame in time — fall back to the most recent published caption.
    latest = _read_latest_camera_state(state_path)
    if latest is not None and str(latest.get("caption", "")).strip():
        ts = _parse_iso(latest.get("ts"))
        age = ""
        if ts is not None:
            age = f" (observed {int((datetime.now(UTC) - ts).total_seconds())}s ago)"
        caption = f"{str(latest['caption']).strip()}{age}"
        return _success(
            command,
            {"scene.caption": {"caption": caption, "labels": [], "boxes": []}},
        )

    return _success(
        command,
        {"scene.caption": {"caption": "camera did not respond in time", "labels": [], "boxes": []}},
    )


def _read_latest_camera_state(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _parse_iso(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _run_worker_command(worker: str, command: Command, hermes_home: Path) -> Result:
    payload = json.dumps(command.model_dump(mode="json", by_alias=True), ensure_ascii=False)
    script = (
        "import json, sys\n"
        "from pathlib import Path\n"
        "from aila.contracts import Command\n"
        "from aila.workers.config import load_worker_config\n"
        "from aila.workers.speaker import SpeakerWorker\n"
        "from aila.workers.speaker_local import build_text_to_speech_backend\n"
        "cfg=load_worker_config(Path.home()/'.hermes/aila-body/workers/speaker/config.yaml')\n"
        "worker=SpeakerWorker(cfg, build_text_to_speech_backend(cfg))\n"
        "result=worker.handle_command(Command.model_validate(json.loads(sys.stdin.read())))\n"
        "print(result.model_dump_json(by_alias=True))\n"
    )
    completed = subprocess.run(
        [str(hermes_home / "venv" / "bin" / "python"), "-c", script],
        input=payload,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        return Result(
            id=command.id,
            ok=False,
            error=Error(
                code="BACKEND_ERROR",
                message=(completed.stderr or completed.stdout or "speaker command failed").strip(),
                retryable=True,
            ),
        )
    return Result.model_validate_json(completed.stdout)


def _success(command: Command, data: dict[str, Any]) -> dict[str, Any]:
    return Result(id=command.id, ok=True, data=data).model_dump(mode="json", by_alias=True)


def _observation_summary(observation: Observation, *, path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "obs_id": observation.obs_id,
        "worker": observation.worker,
        "kind": observation.kind,
        "ts": observation.ts.isoformat(),
        "severity": observation.severity.value if isinstance(observation.severity, Severity) else str(observation.severity),
        "payload": observation.payload.model_dump(mode="json"),
    }


def _contracts_dir(hermes_home: Path) -> Path:
    return hermes_home / "aila-body" / "contracts"


def _queue_dir(hermes_home: Path) -> Path:
    return hermes_home / "aila-body" / "queue"


def _reflex_dir(hermes_home: Path) -> Path:
    return hermes_home / "aila-body" / "reflex"


def reflex_digest_preview(reflex_dir: Path) -> dict[str, Any]:
    """Preview the reflex digest block WITHOUT marking events seen."""
    from aila.reflex.digest import build_digest
    from aila.reflex.store import EventStore

    result = build_digest(EventStore(reflex_dir))
    return {
        "ok": True,
        "mutated": False,
        "count": len(result.event_ids),
        "event_ids": result.event_ids,
        "block": result.block,
    }


def reflex_expand_preview(reflex_dir: Path, *, event_id: str) -> dict[str, Any]:
    """Resolve an event's full context by id (read-only)."""
    from aila.reflex.store import EventStore

    expanded = EventStore(reflex_dir).resolve(event_id)
    if expanded is None:
        return {"ok": False, "error": f"unknown event_id: {event_id}"}
    return {"ok": True, "mutated": False, "event": expanded.model_dump(mode="json")}



def _count_json(path: Path) -> int:
    return len(list(path.glob("*.json"))) if path.is_dir() else 0


def _json_arg(value: str) -> dict[str, Any]:
    loaded = json.loads(value)
    if not isinstance(loaded, dict):
        raise ValueError("args must be a JSON object")
    return loaded


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="aila-body")
    parser.add_argument("--hermes-home", type=Path, default=DEFAULT_HERMES_HOME)
    subparsers = parser.add_subparsers(dest="action", required=True)
    subparsers.add_parser("manifest")
    subparsers.add_parser("queue-status")
    peek = subparsers.add_parser("queue-peek")
    peek.add_argument("--limit", type=int, default=10)
    digest = subparsers.add_parser("digest-preview")
    digest.add_argument("--limit", type=int, default=20)
    subparsers.add_parser("reflex-digest")
    reflex_expand = subparsers.add_parser("reflex-expand")
    reflex_expand.add_argument("--event-id", required=True)
    command = subparsers.add_parser("command")
    command.add_argument("worker")
    command.add_argument("verb")
    command.add_argument("--args", default="{}")
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
