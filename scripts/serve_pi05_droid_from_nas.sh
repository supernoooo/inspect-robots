#!/usr/bin/env bash

set -euo pipefail

usage() {
    printf 'Usage: %s OPENPI_ROOT CHECKPOINT_DIR [PORT]\n' "$0" >&2
    printf 'Example: %s /path/to/openpi /mnt/nas/openpi/pi05_droid 8000\n' "$0" >&2
}

die() {
    printf 'error: %s\n' "$1" >&2
    exit 1
}

if [[ $# -lt 2 || $# -gt 3 ]]; then
    usage
    exit 2
fi

openpi_root="$(realpath "$1")"
checkpoint_dir="$(realpath "$2")"
port="${3:-8000}"

[[ -f "$openpi_root/scripts/serve_policy.py" ]] || die "invalid OpenPI root: $openpi_root"
[[ -d "$checkpoint_dir/params" ]] || die "checkpoint has no params directory"
[[ -d "$checkpoint_dir/assets" ]] || die "checkpoint has no assets directory"
[[ "$port" =~ ^[0-9]+$ ]] || die "port must be an integer"

cd "$openpi_root"
exec uv run scripts/serve_policy.py \
    policy:checkpoint \
    --policy.config=pi05_droid \
    --policy.dir="$checkpoint_dir" \
    --port="$port"
