#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if command -v aila-local-models >/dev/null 2>&1; then
  exec aila-local-models --config "$SCRIPT_DIR/catalog/local-models.toml" "$@"
fi

exec python3 -m aila_models.local_models --config "$SCRIPT_DIR/catalog/local-models.toml" "$@"
