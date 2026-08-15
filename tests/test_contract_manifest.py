from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from aila.contracts.manifest import generate_contract_manifest


def test_manifest_generator_writes_manifest_and_referenced_schemas(tmp_path: Path) -> None:
    manifest = generate_contract_manifest(tmp_path)

    assert json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8")) == manifest
    assert manifest["schema_version"] == 1
    assert manifest["contract_version"] == "1.0"
    assert manifest["fleet"] == ["mic", "camera", "filesystem", "speaker", "display"]

    assert set(manifest["envelopes"]) == {
        "command",
        "result",
        "error",
        "observation",
        "subscription",
    }
    assert set(manifest["verbs"]) == {
        "camera.snapshot",
        "speaker.speak",
        "display.clear",
        "display.render",
    }
    assert set(manifest["observation_kinds"]) == {
        "speech.segment",
        "scene.caption",
        "scene.motion",
        "file.changed",
        "file.created",
        "file.deleted",
        "sensor.status",
    }

    schema_refs = _schema_refs(manifest)
    assert len(schema_refs) == 20
    for schema_ref in schema_refs:
        schema = json.loads((tmp_path / schema_ref).read_text(encoding="utf-8"))
        assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
        assert schema["$id"] == schema_ref
        assert schema["type"] == "object"


def test_manifest_indexes_each_worker_verbs_and_observation_kinds(tmp_path: Path) -> None:
    manifest = generate_contract_manifest(tmp_path)

    assert manifest["workers"]["mic"] == {
        "verbs": {},
        "observation_kinds": {
            "sensor.status": {
                "payload_schema_ref": "observation.sensor.status.payload.schema.json"
            },
            "speech.segment": {
                "payload_schema_ref": "observation.speech.segment.payload.schema.json"
            },
        },
    }
    assert manifest["workers"]["camera"]["verbs"]["snapshot"] == {
        "args_schema_ref": "verb.camera.snapshot.args.schema.json",
        "result_schema_ref": "verb.camera.snapshot.result.schema.json",
    }
    assert set(manifest["workers"]["filesystem"]["observation_kinds"]) == {
        "file.changed",
        "file.created",
        "file.deleted",
    }
    assert manifest["workers"]["speaker"]["verbs"]["speak"] == {
        "args_schema_ref": "verb.speaker.speak.args.schema.json",
        "result_schema_ref": "verb.speaker.speak.result.schema.json",
    }
    assert set(manifest["workers"]["display"]["verbs"]) == {"clear", "render"}


def test_manifest_schema_refs_point_to_expected_payload_shapes(tmp_path: Path) -> None:
    generate_contract_manifest(tmp_path)

    speak_args = json.loads(
        (tmp_path / "verb.speaker.speak.args.schema.json").read_text(encoding="utf-8")
    )
    assert set(speak_args["properties"]) == {"text", "voice", "rate"}
    assert speak_args["required"] == ["text"]
    assert speak_args["additionalProperties"] is False

    speech_segment = json.loads(
        (tmp_path / "observation.speech.segment.payload.schema.json").read_text(
            encoding="utf-8"
        )
    )
    assert set(speech_segment["required"]) == {
        "text",
        "lang",
        "confidence",
        "start_ms",
        "end_ms",
    }
    assert speech_segment["properties"]["confidence"]["maximum"] == 1.0
    assert speech_segment["additionalProperties"] is False

    command = json.loads((tmp_path / "envelope.command.schema.json").read_text(encoding="utf-8"))
    assert set(command["required"]) == {"id", "worker", "verb"}
    assert command["additionalProperties"] is False


def test_contracts_console_module_entrypoint_writes_target_dir(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "aila.contracts.manifest", str(tmp_path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert (tmp_path / "manifest.json").is_file()
    assert (tmp_path / "verb.display.render.args.schema.json").is_file()


def _schema_refs(manifest: dict[str, object]) -> set[str]:
    refs: set[str] = set()
    envelopes = manifest["envelopes"]
    assert isinstance(envelopes, dict)
    for envelope in envelopes.values():
        assert isinstance(envelope, dict)
        refs.add(str(envelope["schema_ref"]))

    verbs = manifest["verbs"]
    assert isinstance(verbs, dict)
    for verb in verbs.values():
        assert isinstance(verb, dict)
        refs.add(str(verb["args_schema_ref"]))
        refs.add(str(verb["result_schema_ref"]))

    observations = manifest["observation_kinds"]
    assert isinstance(observations, dict)
    for observation in observations.values():
        assert isinstance(observation, dict)
        refs.add(str(observation["payload_schema_ref"]))

    return refs
