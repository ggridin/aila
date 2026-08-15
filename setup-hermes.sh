#!/usr/bin/env bash
set -euo pipefail

WAKE_CRON_SCHEDULE="${WAKE_CRON_SCHEDULE:-*/30 * * * *}"
WAKE_CRON_NAME="${WAKE_CRON_NAME:-wake}"
HERMES_HOME="${HERMES_HOME:-$HOME/.hermes}"
AILA_HOME="${AILA_HOME:-$HERMES_HOME/aila-home}"
AILA_BODY="${AILA_BODY:-$HERMES_HOME/aila-body}"
SEED_MODEL="${SEED_MODEL:-Qwen3.5-9B}"
WORKERS_ENABLED="${WORKERS_ENABLED:-mic camera filesystem speaker display}"
MODEL_PLACEMENT_DEFAULT="${MODEL_PLACEMENT_DEFAULT:-lan}"
LAN_MODEL_ENDPOINT="${LAN_MODEL_ENDPOINT:-http://lan-models.local:9000/v1}"
OPENAI_BASE_URL="${OPENAI_BASE_URL:-${OPENAI_API_BASE:-}}"
HERMES_LCM_REPO="${HERMES_LCM_REPO:-https://github.com/stephenschoettler/hermes-lcm}"
HERMES_LCM_DIR="${HERMES_LCM_DIR:-$HERMES_HOME/plugins/hermes-lcm}"
HINDSIGHT_MODE="${HINDSIGHT_MODE:-local_embedded}"
HINDSIGHT_LLM_PROVIDER="${HINDSIGHT_LLM_PROVIDER:-openrouter}"
HINDSIGHT_LLM_MODEL="${HINDSIGHT_LLM_MODEL:-qwen/qwen3.7-flash}"
# Hindsight keeps its own endpoint: the agent brain runs on the local
# llama-server, but memory extraction stays on OpenRouter. Without an explicit
# base URL the embedded daemon maps provider 'openrouter' to OpenAI wire format
# and would send the sk-or-v1 key to api.openai.com.
HINDSIGHT_LLM_BASE_URL="${HINDSIGHT_LLM_BASE_URL:-https://openrouter.ai/api/v1}"
HINDSIGHT_MEMORY_MODE="${HINDSIGHT_MEMORY_MODE:-hybrid}"
HINDSIGHT_RECALL_BUDGET="${HINDSIGHT_RECALL_BUDGET:-mid}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SEED_ROOT="$SCRIPT_DIR/workspace-seed"
SYSTEMD_USER_DIR="$HOME/.config/systemd/user"
AILA_VENV="$HERMES_HOME/venv"
AILA_PYTHON="$AILA_VENV/bin/python"
AILA_PIP="$AILA_VENV/bin/pip"

PATH="$HERMES_HOME/bin:$HOME/.local/bin:$PATH"
export PATH

IFS=' ' read -r -a WORKERS <<<"$WORKERS_ENABLED"

log() {
  printf '[setup-hermes] %s\n' "$*"
}

fail() {
  printf '[setup-hermes] ERROR: %s\n' "$*" >&2
  exit 1
}

require_command() {
  local command_name="$1"
  if ! command -v "$command_name" >/dev/null 2>&1; then
    fail "missing required command: $command_name. Run install-prereqs.sh first."
  fi
}

require_debian_package() {
  local package_name="$1"
  if ! dpkg-query -W -f='${Status}' "$package_name" 2>/dev/null | grep -qx 'install ok installed'; then
    fail "missing required OS package: $package_name. Run install-prereqs.sh first."
  fi
}

validate_input() {
  if [ "$(id -u)" -eq 0 ]; then
    fail "setup-hermes.sh must run as the target user, not as root."
  fi

  OPENROUTER_API_KEY="${OPENROUTER_API_KEY:-}"
  OPENAI_API_KEY="${OPENAI_API_KEY:-}"
  case "$SEED_MODEL" in
    openrouter/*)
      if [ -z "$OPENROUTER_API_KEY" ]; then
        fail "OPENROUTER_API_KEY must be set when SEED_MODEL uses OpenRouter."
      fi
      case "$OPENROUTER_API_KEY" in
        sk-or-v1-*) ;;
        *) fail "OPENROUTER_API_KEY must look like an OpenRouter key (sk-or-v1-...)." ;;
      esac
      ;;
    *)
      if [ -n "$OPENROUTER_API_KEY" ]; then
        case "$OPENROUTER_API_KEY" in
          sk-or-v1-*) ;;
          *) fail "OPENROUTER_API_KEY must look like an OpenRouter key (sk-or-v1-...)." ;;
        esac
      fi
      ;;
  esac

  if [ ! -d "$SEED_ROOT" ]; then
    fail "workspace seed directory not found: $SEED_ROOT"
  fi
  if [ ! -f "$SCRIPT_DIR/pyproject.toml" ]; then
    fail "setup-hermes.sh must run from a synced AILA repository checkout."
  fi
}

verify_prerequisites() {
  require_command bash
  require_command curl
  require_command git
  require_command jq
  require_command rg
  require_command dpkg-query
  require_command systemctl

  require_debian_package libsndfile1
}

install_hermes() {
  if command -v hermes >/dev/null 2>&1; then
    log "Hermes already installed."
    return
  fi

  log "Installing Hermes in user space."
  curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash
  if ! command -v hermes >/dev/null 2>&1; then
    fail "Hermes installer completed but hermes is not on PATH."
  fi
}

ensure_aila_venv() {
  mkdir -p "$HERMES_HOME"
  if [ ! -x "$AILA_PYTHON" ]; then
    if command -v uv >/dev/null 2>&1; then
      uv venv --seed "$AILA_VENV" --python 3.11
    elif command -v python3 >/dev/null 2>&1; then
      python3 -m venv "$AILA_VENV"
    else
      fail "missing uv or python3 to create $AILA_VENV."
    fi
  fi

  "$AILA_PYTHON" -m pip install --upgrade pip
  "$AILA_PIP" install -e "$SCRIPT_DIR"
}

write_main_env() {
  mkdir -p "$HERMES_HOME"
  umask 077
  "$AILA_PYTHON" - "$HERMES_HOME/.env" \
    "OPENROUTER_API_KEY=$OPENROUTER_API_KEY" \
    "OPENAI_API_KEY=$OPENAI_API_KEY" \
    "OPENAI_API_BASE=$OPENAI_BASE_URL" \
    "OPENAI_BASE_URL=$OPENAI_BASE_URL" \
    "LCM_CONTEXT_THRESHOLD=${LCM_CONTEXT_THRESHOLD:-0.35}" \
    "LCM_FRESH_TAIL_COUNT=${LCM_FRESH_TAIL_COUNT:-32}" \
    "LCM_INCREMENTAL_MAX_DEPTH=${LCM_INCREMENTAL_MAX_DEPTH:-3}" \
    "LCM_LEAF_CHUNK_TOKENS=${LCM_LEAF_CHUNK_TOKENS:-20000}" \
    "LCM_DATABASE_PATH=${LCM_DATABASE_PATH:-$HERMES_HOME/lcm.db}" \
    "LCM_ENABLE_SLASH_COMMAND=${LCM_ENABLE_SLASH_COMMAND:-false}" <<'PY'
from __future__ import annotations

import sys
from pathlib import Path

env_path = Path(sys.argv[1])
updates: dict[str, str] = {}
for item in sys.argv[2:]:
    key, value = item.split("=", 1)
    if value:
        updates[key] = value

existing_lines = env_path.read_text(encoding="utf-8-sig").splitlines() if env_path.exists() else []
existing_values: dict[str, str] = {}
for line in existing_lines:
    if line and not line.lstrip().startswith("#") and "=" in line:
        key, value = line.split("=", 1)
        existing_values[key.strip()] = value

if "HINDSIGHT_LLM_API_KEY" not in existing_values and "OPENROUTER_API_KEY" in existing_values:
    updates["HINDSIGHT_LLM_API_KEY"] = existing_values["OPENROUTER_API_KEY"]

written: set[str] = set()
new_lines: list[str] = []
for line in existing_lines:
    if line and not line.lstrip().startswith("#") and "=" in line:
        key = line.split("=", 1)[0].strip()
        if key in updates:
            new_lines.append(f"{key}={updates[key]}")
            written.add(key)
        else:
            new_lines.append(line)
    else:
        new_lines.append(line)

for key, value in updates.items():
    if key not in written:
        new_lines.append(f"{key}={value}")

env_path.write_text("\n".join(new_lines).rstrip() + "\n", encoding="utf-8")
PY
  chmod 600 "$HERMES_HOME/.env"
}

seed_config_if_absent() {
  local config_path="$HERMES_HOME/config.yaml"
  mkdir -p "$HERMES_HOME"
  log "Ensuring AILA config keys in config.yaml."
  "$AILA_PYTHON" - "$config_path" "$SEED_MODEL" "${WORKERS[*]}" "$OPENAI_BASE_URL" "$OPENAI_API_KEY" <<'PY'
from __future__ import annotations

import sys
from pathlib import Path

import yaml

config_path = Path(sys.argv[1])
seed_model = sys.argv[2]
workers = sys.argv[3].split()
openai_base_url = sys.argv[4]
openai_api_key = sys.argv[5]

if config_path.exists():
    loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config = loaded if isinstance(loaded, dict) else {}
else:
    config = {}

if openai_base_url and not seed_model.startswith("openrouter/"):
    model_name = seed_model.split("/", 1)[1] if "/" in seed_model else seed_model
    config["model"] = {
        "provider": "custom",
        "default": model_name,
        "base_url": openai_base_url,
        "api_key": openai_api_key or "local",
    }
else:
    config["model"] = seed_model
config.setdefault("terminal", {})["backend"] = "local"
config.setdefault("approvals", {})["mode"] = "off"
config.setdefault("cron", {})["enabled"] = True
config["workers"] = {"enabled": workers}
config.setdefault("memory", {})["distillation"] = True
config.setdefault("memory", {})["session_search"] = True
config.setdefault("memory", {})["semantic_knowledge"] = True
config.setdefault("memory", {})["provider"] = "hindsight"
config.setdefault("plugins", {})["enabled"] = sorted(
    set(config.setdefault("plugins", {}).get("enabled") or [])
    | {"hermes-lcm", "aila-body", "aila-reflex", "aila-briefing"}
)
config.setdefault("context", {})["engine"] = "lcm"
config.setdefault("compression", {})["enabled"] = True

config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
PY
}

install_aila_body_plugin() {
  local source_dir="$SCRIPT_DIR/plugins/aila_body"
  local target_dir="$HERMES_HOME/plugins/aila_body"

  if [ ! -d "$source_dir" ]; then
    fail "AILA body plugin source not found: $source_dir"
  fi
  mkdir -p "$(dirname -- "$target_dir")"
  rm -rf "$target_dir"
  cp -R "$source_dir" "$target_dir"
}

provision_memory_dependencies() {
  local hermes_python

  if [ "$HINDSIGHT_MODE" = "local_embedded" ]; then
    hermes_python="$(python3 - <<'PY'
from __future__ import annotations

import shutil
from pathlib import Path

for candidate in (
    Path.home() / ".hermes" / "hermes-agent" / "venv" / "bin" / "python",
    Path.home() / ".hermes" / "hermes-agent" / ".venv" / "bin" / "python",
    Path.home() / ".hermes" / "venv" / "bin" / "python",
):
    if candidate.exists():
        print(candidate)
        raise SystemExit(0)

hermes = shutil.which("hermes")
if hermes:
    launcher = Path(hermes)
    try:
        first_line = launcher.read_text(encoding="utf-8", errors="ignore").splitlines()[0]
    except Exception:
        first_line = ""
    if first_line.startswith("#!"):
        parts = first_line[2:].strip().split()
        if parts and Path(parts[0]).name != "env" and Path(parts[0]).exists():
            print(parts[0])
            raise SystemExit(0)

raise SystemExit(1)
PY
)"
    if [ -z "$hermes_python" ] || [ ! -x "$hermes_python" ]; then
      fail "Hermes Python not found; install Hermes before configuring Hindsight."
    fi
    log "Installing Hindsight local embedded dependencies."
    "$hermes_python" -m pip install "hindsight-client>=0.6.1" "hindsight-all>=0.6.1"
  fi
}

materialize_hindsight_embedded_env() {
  local hermes_python

  if [ "$HINDSIGHT_MODE" != "local_embedded" ]; then
    return
  fi

  hermes_python="$(python3 - <<'PY'
from pathlib import Path
for candidate in (
    Path.home() / ".hermes" / "hermes-agent" / "venv" / "bin" / "python",
    Path.home() / ".hermes" / "hermes-agent" / ".venv" / "bin" / "python",
    Path.home() / ".hermes" / "venv" / "bin" / "python",
):
    if candidate.exists():
        print(candidate)
        raise SystemExit(0)
raise SystemExit(1)
PY
)"
  if [ -z "$hermes_python" ] || [ ! -x "$hermes_python" ]; then
    fail "Hermes Python not found; cannot materialize Hindsight embedded env."
  fi

  log "Materializing Hindsight local embedded profile environment."
  "$hermes_python" - "$HERMES_HOME/hindsight/config.json" "$HERMES_HOME/.env" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

hermes_agent = Path.home() / ".hermes" / "hermes-agent"
sys.path.insert(0, str(hermes_agent))

from plugins.memory.hindsight import _materialize_embedded_profile_env


def load_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8-sig", errors="replace").splitlines():
        if not line or line.lstrip().startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    return values


config_path = Path(sys.argv[1])
env_path = Path(sys.argv[2])
config = json.loads(config_path.read_text(encoding="utf-8"))
env = load_env(env_path)
llm_api_key = env.get("HINDSIGHT_LLM_API_KEY") or env.get("OPENROUTER_API_KEY")
profile_env = _materialize_embedded_profile_env(config, llm_api_key=llm_api_key)
print(profile_env)
PY
}

install_hermes_lcm() {
  mkdir -p "$(dirname -- "$HERMES_LCM_DIR")"
  if [ -d "$HERMES_LCM_DIR/.git" ]; then
    log "Updating Hermes-LCM plugin at $HERMES_LCM_DIR."
    git -C "$HERMES_LCM_DIR" pull --ff-only
  elif [ -e "$HERMES_LCM_DIR" ]; then
    fail "Hermes-LCM target exists but is not a git checkout: $HERMES_LCM_DIR"
  else
    log "Cloning Hermes-LCM plugin into $HERMES_LCM_DIR."
    git clone --depth 1 "$HERMES_LCM_REPO" "$HERMES_LCM_DIR"
  fi
}

configure_hindsight() {
  local hindsight_dir="$HERMES_HOME/hindsight"
  local config_path="$hindsight_dir/config.json"

  mkdir -p "$hindsight_dir"
  log "Ensuring Hindsight memory provider config."
  "$AILA_PYTHON" - "$config_path" "$HINDSIGHT_MODE" "$HINDSIGHT_LLM_PROVIDER" "$HINDSIGHT_LLM_MODEL" "$HINDSIGHT_MEMORY_MODE" "$HINDSIGHT_RECALL_BUDGET" "$HINDSIGHT_LLM_BASE_URL" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

config_path = Path(sys.argv[1])
mode, llm_provider, llm_model, memory_mode, recall_budget, llm_base_url = sys.argv[2:8]

if config_path.exists():
    try:
        loaded = json.loads(config_path.read_text(encoding="utf-8"))
        config = loaded if isinstance(loaded, dict) else {}
    except json.JSONDecodeError:
        config = {}
else:
    config = {}

config["mode"] = mode
config["llm_provider"] = llm_provider
config["llm_model"] = llm_model
# Pin the memory endpoint explicitly. The embedded daemon maps provider
# 'openrouter' onto OpenAI wire format, so without this it would send the
# OpenRouter key to api.openai.com once the agent brain moved off OpenRouter.
if llm_base_url:
    config["llm_base_url"] = llm_base_url
config.setdefault("bank_id", "hermes")
config["recall_budget"] = recall_budget
config["memory_mode"] = memory_mode
# Wake-briefing episodes are retained as "observation" records, so this must
# include that type or episodes are stored but never recalled.
config.setdefault("recall_types", "observation")
config.setdefault("auto_recall", True)
config.setdefault("auto_retain", True)
# A cron wake can exit as soon as the session ends; an async retain that has
# not flushed loses exactly the episode we wanted to keep.
config["retain_async"] = False

config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
}

seed_file_if_absent() {
  local source_path="$1"
  local target_path="$2"
  local mode="${3:-}"

  if [ -e "$target_path" ]; then
    return
  fi

  mkdir -p "$(dirname -- "$target_path")"
  cp "$source_path" "$target_path"
  if [ -n "$mode" ]; then
    chmod "$mode" "$target_path"
  fi
}

seed_tree_if_absent() {
  local source_path
  local relative_path
  local target_path

  while IFS= read -r -d '' source_path; do
    relative_path="${source_path#"$SEED_ROOT"/}"
    case "$relative_path" in
      .env.example|config.yaml)
        continue
        ;;
    esac

    target_path="$HERMES_HOME/$relative_path"
    if [ -d "$source_path" ]; then
      mkdir -p "$target_path"
    else
      seed_file_if_absent "$source_path" "$target_path"
    fi
  done < <(find "$SEED_ROOT" -mindepth 1 -print0)
}

seed_worker_envs() {
  local worker
  local example_path
  local env_path

  for worker in "${WORKERS[@]}"; do
    example_path="$SEED_ROOT/aila-body/workers/$worker/.env.example"
    env_path="$AILA_BODY/workers/$worker/.env"
    if [ -f "$example_path" ]; then
      seed_file_if_absent "$example_path" "$env_path" 600
    else
      mkdir -p "$(dirname -- "$env_path")"
      if [ ! -e "$env_path" ]; then
        : >"$env_path"
      fi
      chmod 600 "$env_path"
    fi
  done
}

generate_contract_manifest() {
  mkdir -p "$AILA_BODY/contracts"
  "$AILA_VENV/bin/aila-contracts" "$AILA_BODY/contracts"
}

local_dependency_values() {
  local field="$1"
  "$AILA_PYTHON" - "$HERMES_HOME/config.yaml" "$AILA_BODY/workers" "$AILA_BODY/device-services" "$field" <<'PY'
from __future__ import annotations

import sys
from pathlib import Path

from aila.installer import plan_local_dependencies

registry_path = Path(sys.argv[1])
workers_dir = Path(sys.argv[2])
device_services_dir = Path(sys.argv[3])
field = sys.argv[4]
plan = plan_local_dependencies(
    registry_path,
    workers_dir=workers_dir,
    device_services_dir=device_services_dir,
)
for value in getattr(plan, field):
    print(value)
PY
}

provision_python_dependencies() {
  local deps=("$@")
  if [ "${#deps[@]}" -eq 0 ]; then
    log "No local worker Python dependencies to install."
    return
  fi

  "$AILA_PIP" install "${deps[@]}"
}

provision_model() {
  local model_name="$1"
  local model_dir="$AILA_BODY/models/$model_name"

  case "$model_name" in
    piper-en)
      mkdir -p "$model_dir"
      if [ ! -f "$model_dir/en_US-lessac-medium.onnx" ]; then
        curl -fL \
          -o "$model_dir/en_US-lessac-medium.onnx" \
          "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx"
      fi
      if [ ! -f "$model_dir/en_US-lessac-medium.onnx.json" ]; then
        curl -fL \
          -o "$model_dir/en_US-lessac-medium.onnx.json" \
          "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json"
      fi
      ;;
    whisper-large-v3-turbo-q5|silero-vad-v6-onnx|rnnoise|yamnet|speechbrain-ecapa-tdnn|pyannote-community-1|qwen2.5-vl-3b|opencv-frame-differencing|yolo-openvino-small|mediapipe-tasks|mobileclip2-s0|mobileclip2-s2|camera-local-sensory-stack)
      log "Model '$model_name' is managed by aila-models/install-local-models.sh; skipping setup-hermes provisioning."
      ;;
    *)
      fail "no user-space provisioning recipe for local model: $model_name"
      ;;
  esac
}

provision_local_dependencies() {
  local python_deps=()
  local models=()
  local model
  local python_deps_file
  local models_file

  python_deps_file="$(mktemp)"
  models_file="$(mktemp)"
  trap 'rm -f "$python_deps_file" "$models_file"' RETURN

  local_dependency_values python >"$python_deps_file"
  local_dependency_values models >"$models_file"
  mapfile -t python_deps <"$python_deps_file"
  mapfile -t models <"$models_file"

  provision_python_dependencies "${python_deps[@]}"
  for model in "${models[@]}"; do
    provision_model "$model"
  done

  rm -f "$python_deps_file" "$models_file"
  trap - RETURN
}

install_gateway() {
  hermes gateway install
}

worker_device_service() {
  local worker="$1"
  case "$worker" in
    mic) printf 'audio-input\n' ;;
    camera) printf 'camera-input\n' ;;
    *) return 1 ;;
  esac
}

install_unit() {
  local unit_name="$1"
  local source_path="$AILA_BODY/systemd/$unit_name"
  local target_path="$SYSTEMD_USER_DIR/$unit_name"

  if [ ! -f "$source_path" ]; then
    fail "missing systemd unit template: $source_path"
  fi

  mkdir -p "$SYSTEMD_USER_DIR"
  cp "$source_path" "$target_path"
}

install_and_start_units() {
  local worker
  local service
  local device_services=()
  local seen_services=" "

  for worker in "${WORKERS[@]}"; do
    if [ ! -d "$AILA_BODY/workers/$worker" ]; then
      fail "enabled worker is missing its runtime directory: $worker"
    fi
    if service="$(worker_device_service "$worker")"; then
      if [[ "$seen_services" != *" $service "* ]]; then
        device_services+=("$service")
        seen_services="$seen_services$service "
      fi
    fi
  done

  for service in "${device_services[@]}"; do
    if [ ! -d "$AILA_BODY/device-services/$service" ]; then
      fail "required device service is missing its runtime directory: $service"
    fi
    install_unit "aila-device-$service.service"
  done

  for worker in "${WORKERS[@]}"; do
    install_unit "aila-$worker.service"
  done

  install_unit "aila-reflex-ingest.service"

  systemctl --user daemon-reload

  for service in "${device_services[@]}"; do
    systemctl --user enable --now "aila-device-$service.service"
  done

  for worker in "${WORKERS[@]}"; do
    systemctl --user enable --now "aila-$worker.service"
  done

  systemctl --user enable --now "aila-reflex-ingest.service"
}

register_wake_job() {
  local wake_prompt="You just woke up. Follow the rhythm in AGENTS.md. You are a fresh session: you only know SOUL.md, your memories, and what you choose to look up. Check MESSAGES.md. Decide what to do with this time, then write down anything worth keeping before you sleep."

  mkdir -p "$AILA_HOME"
  if hermes cron list | grep -Eq "(^|[[:space:]])${WAKE_CRON_NAME}($|[[:space:]])"; then
    log "Hermes cron job '$WAKE_CRON_NAME' already exists."
    return
  fi

  hermes cron create "$WAKE_CRON_SCHEDULE" "$wake_prompt" --name "$WAKE_CRON_NAME" --workdir "$AILA_HOME"
}

verify_installation() {
  local worker
  local service

  hermes doctor
  hermes cron list
  hermes gateway status

  test -s "$AILA_BODY/contracts/manifest.json"
  mkdir -p "$AILA_BODY/queue/pending" "$AILA_BODY/queue/inflight" "$AILA_BODY/queue/archive"
  test -w "$AILA_BODY/queue/pending"

  for worker in "${WORKERS[@]}"; do
    systemctl --user status "aila-$worker.service" >/dev/null
    if service="$(worker_device_service "$worker")"; then
      systemctl --user status "aila-device-$service.service" >/dev/null
    fi
  done
}

main() {
  validate_input
  verify_prerequisites
  install_hermes
  install_hermes_lcm
  install_aila_body_plugin
  ensure_aila_venv
  write_main_env
  seed_config_if_absent
  configure_hindsight
  provision_memory_dependencies
  materialize_hindsight_embedded_env
  seed_tree_if_absent
  seed_worker_envs
  generate_contract_manifest
  provision_local_dependencies
  install_gateway
  install_and_start_units
  register_wake_job
  verify_installation
  log "AILA Hermes setup complete."
}

main "$@"
