from __future__ import annotations

from pathlib import Path

import yaml


SEED_ROOT = Path(__file__).resolve().parents[1] / "workspace-seed"

EXPECTED_FILES = {
    # Mind + home seeds (STEP-017)
    "SOUL.md",
    "config.yaml",
    ".env.example",
    "memories/MEMORY.md",
    "memories/USER.md",
    "skills/explore-system/SKILL.md",
    "skills/web-research/SKILL.md",
    "skills/build-tool/SKILL.md",
    "aila-home/AGENTS.md",
    "aila-home/MESSAGES.md",
    "aila-home/IDENTITY.md",
    "aila-home/GOALS.md",
    "aila-home/memory/.gitkeep",
    # Body seeds (STEP-018)
    "aila-body/subscriptions.yaml",
    "aila-body/device-services/audio-input/config.yaml",
    "aila-body/device-services/camera-input/config.yaml",
    "aila-body/workers/mic/config.yaml",
    "aila-body/workers/mic/.env.example",
    "aila-body/workers/camera/config.yaml",
    "aila-body/workers/camera/.env.example",
    "aila-body/workers/filesystem/config.yaml",
    "aila-body/workers/filesystem/.env.example",
    "aila-body/workers/speaker/config.yaml",
    "aila-body/workers/speaker/.env.example",
    "aila-body/workers/display/config.yaml",
    "aila-body/workers/display/.env.example",
    "aila-body/queue/pending/.gitkeep",
    "aila-body/queue/inflight/.gitkeep",
    "aila-body/queue/archive/.gitkeep",
    "aila-body/logs/.gitkeep",
    "aila-body/systemd/aila-device-audio-input.service",
    "aila-body/systemd/aila-device-camera-input.service",
    "aila-body/systemd/aila-mic.service",
    "aila-body/systemd/aila-camera.service",
    "aila-body/systemd/aila-filesystem.service",
    "aila-body/systemd/aila-speaker.service",
    "aila-body/systemd/aila-display.service",
    # Reflex event pipeline seeds (v2)
    "aila-body/reflex-ranking.yaml",
    "aila-body/systemd/aila-reflex-ingest.service",
    "aila-body/systemd/aila-hindsight.service",
    "plugins/aila-reflex/plugin.yaml",
    "plugins/aila-reflex/__init__.py",
    "plugins/aila-reflex/PRIORITY.md",
    "skills/reflex-events/SKILL.md",
    # Wake session briefing seeds
    "plugins/aila-briefing/plugin.yaml",
    "plugins/aila-briefing/__init__.py",
}


def test_workspace_seed_contains_only_expected_v1_files() -> None:
    files = {
        path.relative_to(SEED_ROOT).as_posix()
        for path in SEED_ROOT.rglob("*")
        if path.is_file()
    }

    assert files == EXPECTED_FILES


def test_mind_config_yaml_parses_and_declares_v1_defaults() -> None:
    config = yaml.safe_load((SEED_ROOT / "config.yaml").read_text(encoding="utf-8"))

    assert config["model"].startswith("openrouter/")
    assert config["terminal"] == {"backend": "local"}
    assert config["approvals"] == {"mode": "off"}
    assert config["cron"] == {"enabled": True}
    assert config["workers"]["enabled"] == [
        "mic",
        "camera",
        "filesystem",
        "speaker",
        "display",
    ]
    assert config["memory"] == {
        "distillation": True,
        "session_search": True,
        "semantic_knowledge": True,
        "provider": "hindsight",
    }
    assert config["plugins"]["enabled"] == ["aila-reflex", "aila-briefing"]


def test_mind_secret_template_is_placeholder_only() -> None:
    env_example = (SEED_ROOT / ".env.example").read_text(encoding="utf-8")

    assert "OPENROUTER_API_KEY=" in env_example
    assert "HINDSIGHT_LLM_API_KEY=" in env_example
    assert "LCM_CONTEXT_THRESHOLD=0.35" in env_example
    assert "LCM_ENABLE_SLASH_COMMAND=false" in env_example
    assert "replace-with-openrouter-api-key" in env_example
    assert "sk-or-" not in env_example
    assert not (SEED_ROOT / ".env").exists()


def test_memory_seeds_are_minimal_and_fresh() -> None:
    memory = (SEED_ROOT / "memories" / "MEMORY.md").read_text(encoding="utf-8")
    user = (SEED_ROOT / "memories" / "USER.md").read_text(encoding="utf-8")
    combined = "\n".join(
        path.read_text(encoding="utf-8")
        for path in SEED_ROOT.rglob("*")
        if path.is_file() and path.name != ".gitkeep"
    )

    assert 0 < len(memory) <= 2200
    assert 0 < len(user) <= 1375
    assert "Fresh start" in memory
    assert "OpenClaw" not in combined
    assert "PoC" not in combined


def test_starter_skills_have_valid_frontmatter() -> None:
    expected = {
        "explore-system": "Map a new corner of the laptop and save the finding.",
        "web-research": "Chase a question online and preserve a sourced synthesis.",
        "build-tool": "Create a small runnable script and leave notes to extend it.",
    }

    for skill_name, description in expected.items():
        text = (SEED_ROOT / "skills" / skill_name / "SKILL.md").read_text(
            encoding="utf-8"
        )
        frontmatter, body = _split_frontmatter(text)

        assert frontmatter == {"name": skill_name, "description": description}
        assert body.lstrip().startswith(f"# {skill_name.replace('-', ' ').title()}")


def test_aila_home_seeds_start_create_if_absent_friendly() -> None:
    agents = (SEED_ROOT / "aila-home" / "AGENTS.md").read_text(encoding="utf-8")
    messages = (SEED_ROOT / "aila-home" / "MESSAGES.md").read_text(encoding="utf-8")
    identity = (SEED_ROOT / "aila-home" / "IDENTITY.md").read_text(encoding="utf-8")

    assert "fresh session" in agents
    assert messages.startswith("# Messages\n")
    assert "\n## " in messages
    assert "Name: AILA" in identity
    assert (SEED_ROOT / "aila-home" / "memory" / ".gitkeep").read_text(
        encoding="utf-8"
    ) == ""


def _split_frontmatter(text: str) -> tuple[dict[str, str], str]:
    assert text.startswith("---\n")
    _, frontmatter_text, body = text.split("---\n", 2)
    frontmatter = yaml.safe_load(frontmatter_text)

    assert isinstance(frontmatter, dict)
    return frontmatter, body
