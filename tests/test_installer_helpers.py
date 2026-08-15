from __future__ import annotations

import os
import stat
from pathlib import Path

import yaml

from aila.installer import (
    MODE_600,
    generate_body_contract_manifest,
    plan_local_dependencies,
    render_config,
    render_config_file_if_absent,
    seed_tree_if_absent,
    write_text_if_absent,
)

SEED_ROOT = Path(__file__).resolve().parents[1] / "workspace-seed"
BODY_ROOT = SEED_ROOT / "aila-body"


def test_write_text_if_absent_sets_mode_600_and_never_clobbers(tmp_path: Path) -> None:
    secret_path = tmp_path / ".env"

    assert write_text_if_absent(secret_path, "OPENROUTER_API_KEY=first\n", mode=MODE_600)
    assert not write_text_if_absent(secret_path, "OPENROUTER_API_KEY=second\n", mode=MODE_600)

    assert secret_path.read_text(encoding="utf-8") == "OPENROUTER_API_KEY=first\n"
    if os.name != "nt":
        assert stat.S_IMODE(secret_path.stat().st_mode) == MODE_600


def test_seed_tree_if_absent_creates_missing_files_and_skips_existing(
    tmp_path: Path,
) -> None:
    source = tmp_path / "seed"
    target = tmp_path / "runtime"
    (source / "skills" / "starter").mkdir(parents=True)
    (source / "SOUL.md").write_text("seed soul\n", encoding="utf-8")
    (source / "skills" / "starter" / "SKILL.md").write_text("seed skill\n", encoding="utf-8")
    (target / "SOUL.md").parent.mkdir(parents=True)
    (target / "SOUL.md").write_text("operator edit\n", encoding="utf-8")

    report = seed_tree_if_absent(source, target)

    assert [path.relative_to(target).as_posix() for path in report.created] == [
        "skills/starter/SKILL.md"
    ]
    assert [path.relative_to(target).as_posix() for path in report.skipped] == ["SOUL.md"]
    assert (target / "SOUL.md").read_text(encoding="utf-8") == "operator edit\n"
    assert (target / "skills" / "starter" / "SKILL.md").read_text(
        encoding="utf-8"
    ) == "seed skill\n"


def test_render_config_deep_merges_overrides_without_sorting_keys() -> None:
    rendered = render_config(
        {
            "model": "openrouter/default",
            "workers": {"enabled": ["mic", "camera"]},
            "cron": {"enabled": True},
        },
        overrides={
            "model": "openrouter/custom",
            "workers": {"enabled": ["speaker"]},
        },
    )

    assert yaml.safe_load(rendered) == {
        "model": "openrouter/custom",
        "workers": {"enabled": ["speaker"]},
        "cron": {"enabled": True},
    }
    assert rendered.splitlines()[0] == "model: openrouter/custom"


def test_render_config_file_if_absent_respects_existing_config(tmp_path: Path) -> None:
    target = tmp_path / "config.yaml"

    assert render_config_file_if_absent(
        SEED_ROOT / "config.yaml",
        target,
        overrides={"model": "openrouter/custom", "workers": {"enabled": ["mic"]}},
    )
    assert not render_config_file_if_absent(
        SEED_ROOT / "config.yaml",
        target,
        overrides={"model": "openrouter/ignored"},
    )

    assert yaml.safe_load(target.read_text(encoding="utf-8"))["model"] == "openrouter/custom"


def test_generate_body_contract_manifest_writes_under_contracts_dir(tmp_path: Path) -> None:
    manifest = generate_body_contract_manifest(tmp_path / "aila-body")

    contracts_dir = tmp_path / "aila-body" / "contracts"
    assert (contracts_dir / "manifest.json").is_file()
    assert (contracts_dir / "envelope.command.schema.json").is_file()
    assert manifest["fleet"] == ["mic", "camera", "filesystem", "speaker", "display"]


def test_plan_local_dependencies_uses_only_enabled_local_model_workers() -> None:
    plan = plan_local_dependencies(
        {"workers": {"enabled": ["mic", "camera", "filesystem", "speaker", "display"]}},
        workers_dir=BODY_ROOT / "workers",
    )

    # mic, camera and speaker are local (non-deterministic) model workers;
    # filesystem and display are deterministic and contribute nothing.
    assert plan.workers == ("mic", "camera", "speaker")
    assert plan.os == ("portaudio19-dev", "libsndfile1", "v4l-utils", "libgl1", "alsa-utils")
    assert plan.python == (
        "sounddevice",
        "torch",
        "torchaudio",
        "onnxruntime",
        "silero-vad",
        "speechbrain",
        "pyannote.audio",
        "opencv-python-headless",
        "piper-tts",
    )
    assert plan.models == (
        "whisper-large-v3-turbo-q5",
        "silero-vad-v6-onnx",
        "rnnoise",
        "yamnet",
        "speechbrain-ecapa-tdnn",
        "pyannote-community-1",
        "qwen2.5-vl-3b",
        "piper-en",
    )
