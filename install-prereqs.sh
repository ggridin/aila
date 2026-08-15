#!/usr/bin/env bash
set -euo pipefail

if ! command -v apt-get >/dev/null 2>&1; then
  echo "install-prereqs.sh requires apt-get on an Ubuntu/Debian host." >&2
  exit 1
fi

if ! command -v sudo >/dev/null 2>&1 && [ "$(id -u)" -ne 0 ]; then
  echo "install-prereqs.sh must run as root or as a sudo-capable user." >&2
  exit 1
fi

run_apt_get() {
  if [ "$(id -u)" -eq 0 ]; then
    apt-get "$@"
  else
    sudo apt-get "$@"
  fi
}

run_as_root() {
  if [ "$(id -u)" -eq 0 ]; then
    "$@"
  else
    sudo "$@"
  fi
}

# Install uv system-wide so the AILA user can build a Python 3.11 virtualenv even
# when the host's default python3 is older. uv provisions a managed 3.11
# interpreter on demand, which keeps this step host-independent of the distro's
# Python version. Idempotent: skips when uv is already on PATH.
install_uv() {
  if command -v uv >/dev/null 2>&1; then
    echo "install-prereqs.sh: uv already installed at $(command -v uv)."
    return
  fi

  curl -LsSf https://astral.sh/uv/install.sh \
    | run_as_root env UV_INSTALL_DIR=/usr/local/bin INSTALLER_NO_MODIFY_PATH=1 sh

  if ! command -v uv >/dev/null 2>&1; then
    echo "install-prereqs.sh: uv installation completed but uv is not on PATH." >&2
    exit 1
  fi
}

packages=(
  # Core operator tools.
  git
  curl
  jq
  ripgrep

  # Audio input/output libraries for audio-input, mic, and speaker backends.
  portaudio19-dev
  libsndfile1
  libasound2-dev
  alsa-utils

  # Camera capture/runtime libraries for camera-input and camera plumbing.
  v4l-utils
  libgl1

  # Headless display backend support.
  xvfb

  # Native build helpers for local model backends such as whisper.cpp/Piper.
  build-essential
  cmake
)

export DEBIAN_FRONTEND=noninteractive

# 'apt-get update' can fail on a real workstation because of unrelated
# pre-existing third-party repositories (e.g. Google Chrome, VS Code). That
# should not block AILA's base-repo packages, so treat an update failure as a
# warning and proceed with the existing package lists. A genuinely missing
# package will still fail loudly at the install step below.
if ! run_apt_get update; then
  echo "install-prereqs.sh: 'apt-get update' failed (often a pre-existing third-party repo); continuing with existing package lists." >&2
fi
run_apt_get install -y --no-install-recommends "${packages[@]}"

install_uv

# Device nodes such as /dev/video0 and /dev/snd/* are owned by root with group
# access only (crw-rw---- root video). Without membership in those groups the
# camera and audio workers fail every capture with "Permission denied". Adding
# the AILA user here keeps the fix alongside the rest of host provisioning.
# Idempotent: usermod is a no-op when the user is already a member.
grant_device_group_access() {
  local target_user="${SUDO_USER:-$(id -un)}"
  local group

  for group in video audio; do
    if ! getent group "$group" >/dev/null 2>&1; then
      echo "install-prereqs.sh: group '$group' not present; skipping." >&2
      continue
    fi
    if id -nG "$target_user" | tr ' ' '\n' | grep -qx "$group"; then
      echo "install-prereqs.sh: user '$target_user' already in group '$group'."
      continue
    fi
    echo "install-prereqs.sh: adding user '$target_user' to group '$group'."
    run_as_root usermod -aG "$group" "$target_user"
    echo "install-prereqs.sh: log out/in (or reboot) so '$group' applies to existing sessions." >&2
  done
}

grant_device_group_access
