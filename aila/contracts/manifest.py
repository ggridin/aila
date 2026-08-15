from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from aila.contracts.envelopes import Command, Error, Observation, Result, Subscription
from aila.contracts.payloads import (
    COMMAND_ARG_MODELS,
    CONTRACT_VERSION,
    OBSERVATION_KINDS_BY_WORKER,
    OBSERVATION_PAYLOAD_MODELS,
    VERBS_BY_WORKER,
    ClearResult,
    RenderResult,
    SnapshotResult,
    SpeakResult,
)

JSON_SCHEMA_DIALECT = "https://json-schema.org/draft/2020-12/schema"
MANIFEST_SCHEMA_VERSION = 1
WORKER_ORDER: tuple[str, ...] = ("mic", "camera", "filesystem", "speaker", "display")

ENVELOPE_MODELS: dict[str, type[BaseModel]] = {
    "command": Command,
    "result": Result,
    "error": Error,
    "observation": Observation,
    "subscription": Subscription,
}

RESULT_DATA_MODELS_BY_VERB: dict[tuple[str, str], type[BaseModel]] = {
    ("camera", "snapshot"): SnapshotResult,
    ("speaker", "speak"): SpeakResult,
    ("display", "render"): RenderResult,
    ("display", "clear"): ClearResult,
}


def generate_contract_manifest(target_dir: str | Path) -> dict[str, Any]:
    """Write the static v1 contract manifest and all referenced JSON Schemas."""
    target = Path(target_dir)
    target.mkdir(parents=True, exist_ok=True)

    schema_refs: dict[str, str] = {}
    manifest = _build_manifest(schema_refs)

    for name, schema_ref in schema_refs.items():
        model = _model_for_schema_name(name)
        _write_json(target / schema_ref, _schema_for_model(model, schema_ref))

    _write_json(target / "manifest.json", manifest)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="aila-contracts",
        description="Generate AILA v1 static contract manifest and JSON Schemas.",
    )
    parser.add_argument(
        "target_dir",
        type=Path,
        help="Directory to receive manifest.json and referenced *.schema.json files.",
    )
    args = parser.parse_args(argv)

    generate_contract_manifest(args.target_dir)
    return 0


def _build_manifest(schema_refs: dict[str, str]) -> dict[str, Any]:
    _validate_catalog()

    envelopes = {
        name: {"schema_ref": _register_schema_ref(schema_refs, f"envelope.{name}")}
        for name in ENVELOPE_MODELS
    }
    workers: dict[str, Any] = {}
    verbs: dict[str, Any] = {}
    observation_kinds: dict[str, Any] = {}

    for worker in WORKER_ORDER:
        worker_verbs: dict[str, Any] = {}
        for verb in sorted(VERBS_BY_WORKER[worker]):
            args_ref = _register_schema_ref(schema_refs, f"verb.{worker}.{verb}.args")
            result_ref = _register_schema_ref(schema_refs, f"verb.{worker}.{verb}.result")
            verb_entry = {
                "worker": worker,
                "verb": verb,
                "args_schema_ref": args_ref,
                "result_schema_ref": result_ref,
            }
            worker_verbs[verb] = {
                "args_schema_ref": args_ref,
                "result_schema_ref": result_ref,
            }
            verbs[f"{worker}.{verb}"] = verb_entry

        worker_observations: dict[str, Any] = {}
        for kind in sorted(OBSERVATION_KINDS_BY_WORKER[worker]):
            payload_ref = _register_schema_ref(
                schema_refs,
                f"observation.{kind}.payload",
            )
            kind_entry = {
                "worker": worker,
                "kind": kind,
                "payload_schema_ref": payload_ref,
            }
            worker_observations[kind] = {"payload_schema_ref": payload_ref}
            observation_kinds[kind] = kind_entry

        workers[worker] = {
            "verbs": worker_verbs,
            "observation_kinds": worker_observations,
        }

    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "contract_version": CONTRACT_VERSION,
        "fleet": list(WORKER_ORDER),
        "envelopes": envelopes,
        "workers": workers,
        "verbs": verbs,
        "observation_kinds": observation_kinds,
    }


def _validate_catalog() -> None:
    workers = set(WORKER_ORDER)
    if set(VERBS_BY_WORKER) != workers:
        raise ValueError("verb catalog does not match the fixed worker fleet")
    if set(OBSERVATION_KINDS_BY_WORKER) != workers:
        raise ValueError("observation catalog does not match the fixed worker fleet")
    if set(COMMAND_ARG_MODELS) != set(RESULT_DATA_MODELS_BY_VERB):
        raise ValueError("verb arg/result schemas are not paired")


def _model_for_schema_name(name: str) -> type[BaseModel]:
    if name.startswith("envelope."):
        return ENVELOPE_MODELS[name.removeprefix("envelope.")]
    if name.startswith("observation.") and name.endswith(".payload"):
        kind = name.removeprefix("observation.").removesuffix(".payload")
        return OBSERVATION_PAYLOAD_MODELS[kind]
    if name.startswith("verb.") and name.endswith(".args"):
        worker, verb = name.removeprefix("verb.").removesuffix(".args").split(".", 1)
        return COMMAND_ARG_MODELS[(worker, verb)]
    if name.startswith("verb.") and name.endswith(".result"):
        worker, verb = name.removeprefix("verb.").removesuffix(".result").split(".", 1)
        return RESULT_DATA_MODELS_BY_VERB[(worker, verb)]
    raise KeyError(f"unknown schema name: {name}")


def _register_schema_ref(schema_refs: dict[str, str], name: str) -> str:
    schema_ref = f"{name}.schema.json"
    schema_refs[name] = schema_ref
    return schema_ref


def _schema_for_model(model: type[BaseModel], schema_ref: str) -> dict[str, Any]:
    schema = model.model_json_schema(
        mode="validation",
        ref_template="#/$defs/{model}",
    )
    return {
        "$schema": JSON_SCHEMA_DIALECT,
        "$id": schema_ref,
        **schema,
    }


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
