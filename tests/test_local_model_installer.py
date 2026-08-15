from __future__ import annotations

from pathlib import Path

import tomllib


REPO_ROOT = Path(__file__).resolve().parents[1]
INSTALLER = REPO_ROOT / "aila-models" / "install-local-models.sh"
SETUP = REPO_ROOT / "setup-hermes.sh"
CONFIG = REPO_ROOT / "aila-models" / "catalog" / "local-models.toml"
PYPROJECT = REPO_ROOT / "aila-models" / "pyproject.toml"
PYTHON_INSTALLER = REPO_ROOT / "aila-models" / "aila_models" / "local_models.py"


def test_local_model_installer_is_llama_cpp_only_and_configurable() -> None:
    text = INSTALLER.read_text(encoding="utf-8")
    config_text = CONFIG.read_text(encoding="utf-8")
    python_text = PYTHON_INSTALLER.read_text(encoding="utf-8")

    assert "aila-local-models --config" in text
    assert "python3 -m aila_models.local_models" in text
    assert "ggml-org/llama.cpp.git" in config_text
    assert "ollama" not in config_text.lower()
    assert "unsloth/Qwen3.5-9B-GGUF" in config_text
    assert 'quant = "Q5_K_M"' in config_text
    assert "Q6_K" not in python_text
    assert "ggml-org/Nomic-Embed-Text-V2-GGUF" in config_text
    assert "llama-server" in config_text
    assert "127.0.0.1" in config_text
    assert "port = 8080" in config_text
    for expected in ("systemctl", "--user", "enable", "--now"):
        assert expected in python_text
    for expected in (
        "Silero VAD v6 ONNX",
        "RNNoise",
        "Whisper large-v3-turbo Q5",
        "YAMNet",
        "SpeechBrain ECAPA-TDNN",
        "pyannote Community-1",
        "install_python",
    ):
        assert expected in config_text or expected in python_text


def test_local_model_toml_is_the_single_settings_source() -> None:
    with CONFIG.open("rb") as handle:
        config = tomllib.load(handle)

    assert config["llama_cpp"]["repo"] == "https://github.com/ggml-org/llama.cpp.git"
    assert config["models"]["chat"] == {
        "repo": "unsloth/Qwen3.5-9B-GGUF",
        "file": "",
        "quant": "Q5_K_M",
        "alias": "Qwen3.5-9B",
    }
    assert config["models"]["embeddings"]["repo"] == "ggml-org/Nomic-Embed-Text-V2-GGUF"
    assert config["service"]["chat"]["install"] is True
    assert config["sensory"]["install_python"] is True
    assert "git+https://github.com/snakers4/silero-vad.git" in config["sensory"][
        "python_packages"
    ]
    assert "pyannote.audio" in config["sensory"]["python_packages"]
    # Vision now uses the co-located VLM service, not the old object stack.
    assert config["models"]["vision"]["repo"] == "ggml-org/Qwen2.5-VL-3B-Instruct-GGUF"
    assert config["models"]["vision"]["mmproj"].startswith("mmproj-")
    assert config["service"]["vision"]["install"] is True
    assert "commands" not in config["sensory"]

    assets = config["sensory"]["assets"]
    assert assets["whisper_large_v3_turbo_q5"]["url"].endswith(
        "/whisper.cpp/whisper-large-v3-turbo-q5_k.gguf?download=true"
    )
    # Object-stack assets were removed in the VLM migration.
    assert "yolo_openvino_small" not in assets
    assert "mobileclip2_s0" not in assets

    installer_text = PYTHON_INSTALLER.read_text(encoding="utf-8")
    assert "tomllib" in installer_text
    assert "os.environ" not in installer_text
    assert "_run_sensory_commands" in installer_text
    assert "_download_hf_file_set" in installer_text


def test_package_declares_local_model_entrypoint() -> None:
    pyproject = PYPROJECT.read_text(encoding="utf-8")

    assert 'aila-local-models = "aila_models.local_models:main"' in pyproject


def test_setup_hermes_allows_non_openrouter_seed_model_without_openrouter_key() -> None:
    text = SETUP.read_text(encoding="utf-8")

    assert "OPENROUTER_API_KEY must be set when SEED_MODEL uses OpenRouter." in text
    assert "openrouter/*)" in text
    assert "OPENAI_API_BASE" in text
    assert "OPENAI_BASE_URL" in text
    assert "OPENAI_API_KEY" in text
    assert "Ensuring AILA config keys in config.yaml" in text
    assert "config[\"workers\"] = {\"enabled\": workers}" in text
    assert "memory\", {})[\"provider\"] = \"hindsight\"" in text
    assert "hermes-lcm" in text
    assert "context\", {})[\"engine\"] = \"lcm\"" in text
    assert "hindsight-client>=0.6.1" in text
    assert "hindsight-all>=0.6.1" in text
    assert "aila-briefing" in text
    assert 'config["retain_async"] = False' in text
    assert "materialize_hindsight_embedded_env" in text
    assert "_materialize_embedded_profile_env" in text
    assert "HINDSIGHT_LLM_MODEL" in text
    assert "OPENROUTER_API_KEY must be set in the environment before running setup-hermes.sh" not in text
