from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tomllib
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aila_models._paths import expand_path


@dataclass(frozen=True)
class Paths:
    repo_root: Path
    config_path: Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Install AILA local model assets.")
    parser.add_argument(
        "--config",
        default=None,
        help="Path to local model TOML config. Defaults to catalog/local-models.toml beside the aila_models package.",
    )
    args = parser.parse_args(argv)

    paths = _paths(args.config)
    config = _load_toml(paths.config_path)
    installer = LocalModelInstaller(config=config, repo_root=paths.repo_root)
    installer.run()
    return 0


class LocalModelInstaller:
    def __init__(self, *, config: dict[str, Any], repo_root: Path) -> None:
        self.config = config
        self.repo_root = repo_root
        self.home = Path.home()
        self.llama_cpp = config["llama_cpp"]
        self.models = config["models"]
        self.chat_model = self.models["chat"]
        self.embed_model = self.models["embeddings"]
        self.vision_model = self.models.get("vision")
        self.chat_service = config["service"]["chat"]
        self.vision_service = config["service"].get("vision")
        self.sensory = config["sensory"]
        self.whisper_cpp = config.get("whisper_cpp")
        self.stt_service = config["service"].get("stt")

    def run(self) -> None:
        self._require_tools("bash", "cmake", "curl", "git", "python3")
        self._sync_llama_cpp()
        self._build_llama_server()
        self._install_models()
        self._install_sensory_stack()
        self._install_chat_service()
        self._install_vision()
        self._install_whisper()
        self._log(
            "Local llama.cpp models are installed. "
            f"Chat endpoint: http://{self.chat_service['host']}:{self.chat_service['port']}/v1"
        )

    def _sync_llama_cpp(self) -> None:
        self._sync_repo(
            self._path(self.llama_cpp["root"]), self.llama_cpp["repo"], "llama.cpp"
        )

    def _build_llama_server(self) -> None:
        self._build_cmake_server(
            spec=self.llama_cpp,
            target="llama-server",
            extra_cmake_args=["-DLLAMA_CURL=ON"],
        )

    def _sync_repo(self, root: Path, repo: str, label: str) -> None:
        """Clone (or fast-forward) a git repo at ``root``."""
        root.parent.mkdir(parents=True, exist_ok=True)
        if (root / ".git").is_dir():
            self._log(f"Updating {label} at {root}.")
            self._run(["git", "-C", str(root), "pull", "--ff-only"])
        else:
            self._log(f"Cloning {label} into {root}.")
            self._run(["git", "clone", "--depth", "1", repo, str(root)])

    def _build_cmake_server(
        self, *, spec: dict[str, Any], target: str, extra_cmake_args: list[str]
    ) -> Path:
        """Configure/build ``target`` with cmake and stage its binary.

        Shared by llama.cpp and whisper.cpp: both are cmake projects that build
        a ``*-server`` target with an optional CUDA toggle, then copy the result
        into ``bin_dir``. Returns the staged server binary path.
        """
        root = self._path(spec["root"])
        build_dir = self._path(spec["build_dir"])
        server_bin = self._path(spec["server_bin"])
        bin_dir = self._path(spec["bin_dir"])

        self._log(f"Building {target}.")
        configure = [
            "cmake",
            "-S",
            str(root),
            "-B",
            str(build_dir),
            "-DCMAKE_BUILD_TYPE=Release",
            *extra_cmake_args,
        ]
        if bool(spec.get("cuda", False)):
            configure.append("-DGGML_CUDA=ON")
            cuda_arch = str(spec.get("cuda_arch", "")).strip()
            if cuda_arch:
                configure.append(f"-DCMAKE_CUDA_ARCHITECTURES={cuda_arch}")
        self._run(configure)
        self._run(
            [
                "cmake",
                "--build",
                str(build_dir),
                "--config",
                "Release",
                "--target",
                target,
                f"-j{os.cpu_count() or 1}",
            ]
        )

        built_server = build_dir / "bin" / target
        if not built_server.is_file():
            raise RuntimeError(f"{target} was not built at {built_server}.")

        bin_dir.mkdir(parents=True, exist_ok=True)
        temp_server = server_bin.with_name(server_bin.name + ".new")
        shutil.copy2(built_server, temp_server)
        temp_server.replace(server_bin)
        server_bin.chmod(0o755)
        return server_bin

    def _install_models(self) -> None:
        model_root = self._path(self.models["root"])
        chat_file = self._resolve_hf_file(
            repo=self.chat_model["repo"],
            explicit_file=self.chat_model.get("file", ""),
            quant=self.chat_model.get("quant", ""),
            purpose="chat model",
        )
        embed_file = self._resolve_hf_file(
            repo=self.embed_model["repo"],
            explicit_file=self.embed_model.get("file", ""),
            quant=self.embed_model.get("quant", ""),
            purpose="embedding model",
        )

        chat_path = self._download_hf_file(self.chat_model["repo"], chat_file, model_root / "chat")
        embed_path = self._download_hf_file(
            self.embed_model["repo"], embed_file, model_root / "embeddings"
        )

        (model_root / "chat").mkdir(parents=True, exist_ok=True)
        (model_root / "embeddings").mkdir(parents=True, exist_ok=True)
        (model_root / "chat" / "current-model.path").write_text(
            f"{chat_path}\n", encoding="utf-8"
        )
        (model_root / "embeddings" / "current-model.path").write_text(
            f"{embed_path}\n", encoding="utf-8"
        )

    def _install_chat_service(self) -> None:
        if not bool(self.chat_service["install"]):
            self._log("Skipping chat service installation because service.chat.install=false.")
            return

        model_root = self._path(self.models["root"])
        chat_path = (model_root / "chat" / "current-model.path").read_text(
            encoding="utf-8"
        ).strip()
        server_bin = self._path(self.llama_cpp["server_bin"])
        systemd_user_dir = self._path(self.chat_service["systemd_user_dir"])
        service_name = self.chat_service["name"]
        service_path = systemd_user_dir / service_name

        command = [
            str(server_bin),
            "--host",
            str(self.chat_service["host"]),
            "--port",
            str(self.chat_service["port"]),
            "--model",
            chat_path,
            "--alias",
            str(self.chat_model["alias"]),
            "--ctx-size",
            str(self.chat_service["ctx_size"]),
        ]
        threads = str(self.chat_service.get("threads", ""))
        if threads:
            command.extend(["--threads", threads])
        gpu_layers = str(self.chat_service.get("gpu_layers", "")).strip()
        if gpu_layers:
            command.extend(["--n-gpu-layers", gpu_layers])
        command.extend(str(arg) for arg in self.chat_service.get("extra_args", []))

        self._write_systemd_service(
            service_path=service_path,
            description="AILA llama.cpp chat model server",
            command=command,
        )
        self._enable_service(service_name)

    def _install_vision(self) -> None:
        if not self.vision_model or not self.vision_service:
            self._log("Skipping vision model: no [models.vision]/[service.vision] configured.")
            return
        if not bool(self.vision_service.get("install", True)):
            self._log("Skipping vision model because service.vision.install=false.")
            return

        model_root = self._path(self.models["root"])
        vision_dir = model_root / "vision-llm"
        vision_dir.mkdir(parents=True, exist_ok=True)

        model_file = self._resolve_hf_file(
            repo=self.vision_model["repo"],
            explicit_file=self.vision_model.get("file", ""),
            quant=self.vision_model.get("quant", ""),
            purpose="vision model",
        )
        model_path = self._download_hf_file(self.vision_model["repo"], model_file, vision_dir)

        mmproj_file = str(self.vision_model.get("mmproj", "")).strip()
        if not mmproj_file:
            raise RuntimeError("set models.vision.mmproj in catalog/local-models.toml (multimodal projector file).")
        mmproj_path = self._download_hf_file(self.vision_model["repo"], mmproj_file, vision_dir)

        (vision_dir / "current-model.path").write_text(f"{model_path}\n", encoding="utf-8")
        (vision_dir / "current-mmproj.path").write_text(f"{mmproj_path}\n", encoding="utf-8")

        self._install_vision_service(model_path, mmproj_path)

    def _install_vision_service(self, model_path: Path, mmproj_path: Path) -> None:
        server_bin = self._path(self.llama_cpp["server_bin"])
        systemd_user_dir = self._path(self.vision_service["systemd_user_dir"])
        service_name = self.vision_service["name"]
        service_path = systemd_user_dir / service_name

        command = [
            str(server_bin),
            "--host",
            str(self.vision_service["host"]),
            "--port",
            str(self.vision_service["port"]),
            "--model",
            str(model_path),
            "--mmproj",
            str(mmproj_path),
            "--alias",
            str(self.vision_model["alias"]),
            "--ctx-size",
            str(self.vision_service["ctx_size"]),
        ]
        threads = str(self.vision_service.get("threads", ""))
        if threads:
            command.extend(["--threads", threads])
        gpu_layers = str(self.vision_service.get("gpu_layers", "")).strip()
        if gpu_layers:
            command.extend(["--n-gpu-layers", gpu_layers])
        command.extend(str(arg) for arg in self.vision_service.get("extra_args", []))

        self._write_systemd_service(
            service_path=service_path,
            description="AILA llama.cpp vision (VLM) server",
            command=command,
        )
        self._enable_service(service_name)

    def _install_whisper(self) -> None:
        if not self.whisper_cpp or not self.stt_service:
            self._log("Skipping whisper: no [whisper_cpp]/[service.stt] configured.")
            return
        if not bool(self.stt_service.get("install", True)):
            self._log("Skipping whisper because service.stt.install=false.")
            return

        server_bin = self._build_whisper_server()
        model_path = self._path(str(self.stt_service["model"]))
        if not model_path.is_file():
            raise RuntimeError(
                f"whisper model not found at {model_path}; "
                "ensure [sensory.assets.whisper_large_v3_turbo_q5] downloaded it."
            )
        self._install_whisper_service(server_bin, model_path)

    def _build_whisper_server(self) -> Path:
        self._sync_repo(
            self._path(self.whisper_cpp["root"]), self.whisper_cpp["repo"], "whisper.cpp"
        )
        return self._build_cmake_server(
            spec=self.whisper_cpp,
            target="whisper-server",
            extra_cmake_args=["-DWHISPER_BUILD_SERVER=ON"],
        )

    def _install_whisper_service(self, server_bin: Path, model_path: Path) -> None:
        systemd_user_dir = self._path(self.stt_service["systemd_user_dir"])
        service_name = self.stt_service["name"]
        service_path = systemd_user_dir / service_name

        command = [
            str(server_bin),
            "--host",
            str(self.stt_service["host"]),
            "--port",
            str(self.stt_service["port"]),
            "--model",
            str(model_path),
        ]
        threads = str(self.stt_service.get("threads", "")).strip()
        if threads:
            command.extend(["--threads", threads])
        command.extend(str(arg) for arg in self.stt_service.get("extra_args", []))

        self._write_systemd_service(
            service_path=service_path,
            description="AILA whisper.cpp speech-to-text server",
            command=command,
        )
        self._enable_service(service_name)

    def _install_sensory_stack(self) -> None:
        if not bool(self.sensory["install"]):
            self._log("Skipping sensory stack because sensory.install=false.")
            return
        sensory_root = self._path(self.sensory["model_root"])
        for child in ("audio", "vision", "multimodal"):
            (sensory_root / child).mkdir(parents=True, exist_ok=True)
        self._install_sensory_python_dependencies()
        self._run_sensory_commands(sensory_root)

        assets = self.sensory.get("assets", {})
        for asset in assets.values():
            self._install_sensory_asset(sensory_root, asset)
        self._write_sensory_manifest(sensory_root, assets)

    def _install_sensory_python_dependencies(self) -> None:
        if not bool(self.sensory["install_python"]):
            self._log("Skipping sensory Python dependency install because sensory.install_python=false.")
            return

        venv = self._path(self.sensory["venv"])
        python = venv / "bin" / "python"
        pip = venv / "bin" / "pip"
        if not python.is_file():
            self._run(["python3", "-m", "venv", str(venv)])
        self._run([str(python), "-m", "pip", "install", "--upgrade", "pip"])
        self._run([str(pip), "install", *self.sensory["python_packages"]])

    def _run_sensory_commands(self, sensory_root: Path) -> None:
        if not bool(self.sensory["install_python"]):
            self._log("Skipping sensory commands because sensory.install_python=false.")
            return

        venv = self._path(self.sensory["venv"])
        bin_dir = venv / "bin"
        for command_config in self.sensory.get("commands", []):
            name = command_config["name"]
            creates = sensory_root / command_config["creates"]
            if creates.exists():
                self._log(f"Keeping existing {name} output at {creates}.")
                continue
            creates.parent.mkdir(parents=True, exist_ok=True)
            argv = [str(part) for part in command_config["argv"]]
            executable = bin_dir / argv[0]
            if executable.exists():
                argv[0] = str(executable)
            self._log(f"Running sensory command: {name}.")
            self._run(argv, cwd=creates.parent)

    def _write_sensory_manifest(self, sensory_root: Path, assets: dict[str, Any]) -> None:
        lines = ["# AILA sensory model stack", ""]
        descriptions = {
            "Silero VAD v6 ONNX": "audio voice activity detection",
            "RNNoise": "audio denoising",
            "Whisper large-v3-turbo Q5": "speech transcription",
            "YAMNet": "audio event classification",
            "SpeechBrain ECAPA-TDNN": "speaker embedding",
            "pyannote Community-1": "diarization pipeline",
            "small YOLO OpenVINO": "object detection",
            "MobileCLIP2-S0": "image-text embeddings",
            "MobileCLIP2-S2": "image-text embeddings",
        }
        for asset in assets.values():
            name = asset["name"]
            if asset.get("url"):
                note = f"configured target `{asset['target']}`"
            elif asset.get("repo") and asset.get("files"):
                note = f"configured Hugging Face file set under `{asset['target']}`"
            elif asset.get("install") == "python-package":
                note = "installed through sensory.python_packages"
            elif asset.get("install") == "command":
                note = "generated by a sensory command"
            else:
                note = f"set URL in catalog/local-models.toml to download `{asset['target']}`"
            lines.append(f"- {name}: {descriptions.get(name, 'model asset')}. {note}.")
        lines.extend(
            [
                "- OpenCV frame differencing/background subtraction: algorithmic dependency, no model file required.",
                "- MediaPipe Tasks: vision/audio task runtime dependency.",
                "",
                f"sensory.install_python={str(self.sensory['install_python']).lower()}",
                f"sensory.venv={self.sensory['venv']}",
                "",
            ]
        )
        (sensory_root / "MANIFEST.md").write_text("\n".join(lines), encoding="utf-8")

    def _resolve_hf_file(
        self, *, repo: str, explicit_file: str, quant: str, purpose: str
    ) -> str:
        if explicit_file:
            return explicit_file

        url = f"https://huggingface.co/api/models/{repo}"
        with urllib.request.urlopen(url, timeout=30) as response:
            data = json.load(response)
        files = [
            sibling.get("rfilename", "")
            for sibling in data.get("siblings", [])
            if sibling.get("rfilename", "").lower().endswith(".gguf")
        ]
        if quant:
            exact_suffix = f"-{quant}.gguf".lower()
            exact_files = [path for path in files if path.lower().endswith(exact_suffix)]
            files = exact_files or [path for path in files if quant.lower() in path.lower()]

        if len(files) == 1:
            return files[0]
        if not files:
            qualifier = f" matching quantization '{quant}'" if quant else ""
            raise RuntimeError(f"No GGUF files{qualifier} found for {purpose} in {repo}.")
        formatted = "\n".join(f"  - {path}" for path in files[:25])
        raise RuntimeError(
            f"Multiple GGUF files found for {purpose} in {repo}; "
            f"set file or quant in catalog/local-models.toml.\n{formatted}"
        )

    def _download_hf_file(self, repo: str, file_name: str, target_dir: Path) -> Path:
        target_path = target_dir / file_name
        url = f"https://huggingface.co/{repo}/resolve/main/{file_name}?download=true"
        self._download_url(url, target_path, f"{repo}/{file_name}")
        return target_path

    def _download_optional_asset(self, *, name: str, url: str, target_path: Path) -> None:
        if not url:
            self._log(f"No URL configured for {name}; recorded in sensory manifest only.")
            return
        self._download_url(url, target_path, f"{name} asset")

    def _install_sensory_asset(self, sensory_root: Path, asset: dict[str, Any]) -> None:
        target_path = sensory_root / asset["target"]
        if asset.get("repo") and asset.get("files"):
            self._download_hf_file_set(
                repo=asset["repo"],
                files=[str(file_name) for file_name in asset["files"]],
                target_dir=target_path,
                label=asset["name"],
            )
            return
        if asset.get("install") in {"python-package", "command"}:
            self._log(f"{asset['name']} is installed via {asset['install']}.")
            return
        self._download_optional_asset(
            name=asset["name"],
            url=asset.get("url", ""),
            target_path=target_path,
        )

    def _download_hf_file_set(
        self, *, repo: str, files: list[str], target_dir: Path, label: str
    ) -> None:
        target_dir.mkdir(parents=True, exist_ok=True)
        for file_name in files:
            url = f"https://huggingface.co/{repo}/resolve/main/{file_name}?download=true"
            self._download_url(url, target_dir / file_name, f"{label} {file_name}")

    def _download_url(self, url: str, target_path: Path, label: str) -> None:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        if target_path.is_file() and target_path.stat().st_size > 0:
            self._log(f"Keeping existing model file {target_path}.")
            return
        self._log(f"Downloading {label}.")
        self._run(["curl", "-fL", "--retry", "3", "--retry-delay", "5", "-o", str(target_path), url])

    def _require_tools(self, *commands: str) -> None:
        for command in commands:
            if shutil.which(command) is None:
                raise RuntimeError(f"missing required command: {command}. Run install-prereqs.sh first.")

    def _run(self, command: list[str], *, cwd: Path | None = None) -> None:
        subprocess.run(command, check=True, cwd=cwd)

    def _write_systemd_service(
        self, *, service_path: Path, description: str, command: list[str]
    ) -> None:
        """Write a --user systemd unit for a long-running model server."""
        service_path.parent.mkdir(parents=True, exist_ok=True)
        service_path.write_text(
            "\n".join(
                [
                    "[Unit]",
                    f"Description={description}",
                    "After=network-online.target",
                    "",
                    "[Service]",
                    "Type=simple",
                    "ExecStart=" + " ".join(_shell_quote(part) for part in command),
                    "Restart=on-failure",
                    "RestartSec=5",
                    "",
                    "[Install]",
                    "WantedBy=default.target",
                    "",
                ]
            ),
            encoding="utf-8",
        )

    def _enable_service(self, service_name: str) -> None:
        self._run(["systemctl", "--user", "daemon-reload"])
        self._run(["systemctl", "--user", "enable", "--now", service_name])

    def _path(self, value: str) -> Path:
        return expand_path(value)

    def _log(self, message: str) -> None:
        print(f"[install-local-models] {message}", file=sys.stderr)


def _paths(config_path: str | None) -> Paths:
    package_root = Path(__file__).resolve().parents[1]
    resolved_config = (
        expand_path(config_path)
        if config_path
        else package_root / "catalog" / "local-models.toml"
    )
    return Paths(repo_root=package_root, config_path=resolved_config)


def _load_toml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"local model config not found: {path}")
    with path.open("rb") as handle:
        return tomllib.load(handle)


def _shell_quote(value: str) -> str:
    import shlex

    return shlex.quote(value)


if __name__ == "__main__":
    raise SystemExit(main())
