from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml

from aila.contracts.manifest import generate_contract_manifest
from aila.installer.seeding import write_text_if_absent


def render_config(
    template: Mapping[str, Any],
    *,
    overrides: Mapping[str, Any] | None = None,
) -> str:
    """Render a YAML config mapping with optional deep overrides."""
    data = _deep_merge(dict(template), overrides or {})
    return yaml.safe_dump(data, sort_keys=False)


def render_config_file_if_absent(
    template_path: str | Path,
    target_path: str | Path,
    *,
    overrides: Mapping[str, Any] | None = None,
    mode: int | None = None,
) -> bool:
    """Render a YAML config template to target only when target is absent."""
    source = Path(template_path)
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"config template must be a YAML mapping: {source}")
    return write_text_if_absent(
        target_path,
        render_config(raw, overrides=overrides),
        mode=mode,
    )


render_config_if_absent = render_config_file_if_absent


def generate_body_contract_manifest(body_dir: str | Path) -> dict[str, Any]:
    """Generate the static contract manifest under a runtime aila-body directory."""
    return generate_contract_manifest(Path(body_dir) / "contracts")


generate_contract_manifest_into_body = generate_body_contract_manifest


def _deep_merge(
    base: dict[str, Any],
    overrides: Mapping[str, Any],
) -> dict[str, Any]:
    for key, value in overrides.items():
        existing = base.get(key)
        if isinstance(existing, dict) and isinstance(value, Mapping):
            base[key] = _deep_merge(dict(existing), value)
        else:
            base[key] = value
    return base
