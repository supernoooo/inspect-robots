#!/usr/bin/env bash

set -euo pipefail

usage() {
    printf 'Usage: %s SOURCE_CHECKPOINT TARGET_CHECKPOINT\n' "$0" >&2
    printf '       %s --calvin-only\n' "$0" >&2
    printf 'Example: %s ~/.cache/openpi/openpi-assets/checkpoints/pi05_droid /mnt/nas/openpi/pi05_droid\n' "$0" >&2
    printf '\nThe two-argument form also migrates /mnt/yuantao_nas/datasets/calvin\n' >&2
    printf 'to /mnt/yuantao_nas/data/calvin after the checkpoint is verified.\n' >&2
}

die() {
    printf 'error: %s\n' "$1" >&2
    exit 1
}

calvin_source="/mnt/yuantao_nas/datasets/calvin"
calvin_target="/mnt/yuantao_nas/data/calvin"

ensure_writable_directory() {
    local directory="$1"
    local existing="$directory"
    local mount_options

    while [[ ! -e "$existing" ]]; do
        existing="$(dirname "$existing")"
    done
    mount_options="$(findmnt -n -o OPTIONS -T "$existing" 2>/dev/null || true)"
    if [[ ",$mount_options," == *,ro,* ]]; then
        die "target filesystem is mounted read-only: $existing"
    fi
    mkdir -p "$directory"
    [[ -w "$directory" ]] || die "target directory is not writable: $directory"
}

directory_metadata_fingerprint() {
    local directory="$1"

    find "$directory" -type f -printf '%P\0%s\0%T@\0' \
        | LC_ALL=C sort -z \
        | sha256sum \
        | awk '{print $1}'
}

migrate_calvin() {
    local calvin_parent
    local source_device
    local target_device
    local existing_entry
    local fingerprint_before
    local fingerprint_after
    local backup_target=""

    calvin_parent="$(dirname "$calvin_target")"
    ensure_writable_directory "$calvin_parent"
    if [[ ! -e "$calvin_source" ]]; then
        if [[ -d "$calvin_target" ]]; then
            printf 'CALVIN is already at %s; source is absent.\n' "$calvin_target"
            return
        fi
        die "neither CALVIN source nor target exists"
    fi
    [[ -d "$calvin_source" ]] || die "CALVIN source is not a directory: $calvin_source"

    source_device="$(stat -c '%d' "$calvin_source")"
    target_device="$(stat -c '%d' "$calvin_parent")"
    if [[ "$source_device" != "$target_device" ]]; then
        die "CALVIN source and target are not on the same filesystem"
    fi

    if [[ -e "$calvin_target" ]]; then
        [[ -d "$calvin_target" ]] || die "CALVIN target exists and is not a directory"
        existing_entry="$(find "$calvin_target" -mindepth 1 -maxdepth 1 -print -quit)"
        if [[ -n "$existing_entry" ]]; then
            fingerprint_before="$(directory_metadata_fingerprint "$calvin_target")"
            printf 'Checking the existing CALVIN target for active writes for 30 seconds\n'
            sleep 30
            fingerprint_after="$(directory_metadata_fingerprint "$calvin_target")"
            if [[ "$fingerprint_before" != "$fingerprint_after" ]]; then
                die "CALVIN target changed during the safety window; another transfer is active"
            fi
            backup_target="${calvin_target}.preexisting.$(date -u +%Y%m%dT%H%M%SZ).$$"
            printf 'Preserving the existing CALVIN target at %s\n' "$backup_target"
            mv -- "$calvin_target" "$backup_target"
        else
            rmdir -- "$calvin_target"
        fi
    fi

    printf 'Moving %s to %s on the same filesystem\n' "$calvin_source" "$calvin_target"
    if ! mv -- "$calvin_source" "$calvin_target"; then
        if [[ -n "$backup_target" && -d "$backup_target" && ! -e "$calvin_target" ]]; then
            mv -- "$backup_target" "$calvin_target" || true
        fi
        die "CALVIN directory move failed"
    fi
    printf 'CALVIN migration complete: %s\n' "$calvin_target"
    if [[ -n "$backup_target" ]]; then
        printf 'The pre-existing partial target was retained at: %s\n' "$backup_target"
    fi
}

if [[ $# -eq 1 && "$1" == "--calvin-only" ]]; then
    migrate_calvin
    exit 0
fi

if [[ $# -ne 2 ]]; then
    usage
    exit 2
fi

command -v rsync >/dev/null 2>&1 || die "rsync is required"

source_checkpoint="$(realpath "$1")"
target_checkpoint="$(realpath -m "$2")"
target_parent="$(dirname "$target_checkpoint")"
partial_checkpoint="${target_checkpoint}.partial"

[[ -d "$source_checkpoint/params" ]] || die "source checkpoint has no params directory"
[[ -d "$source_checkpoint/assets" ]] || die "source checkpoint has no assets directory"
[[ "$source_checkpoint" != "$target_checkpoint" ]] || die "source and target are identical"
[[ ! -e "$target_checkpoint" ]] || die "target already exists: $target_checkpoint"

ensure_writable_directory "$target_parent"

source_bytes="$(du -sb "$source_checkpoint" | awk '{print $1}')"
available_bytes="$(df -B1 --output=avail "$target_parent" | tail -n 1 | tr -d ' ')"
if (( available_bytes < source_bytes )); then
    die "target filesystem does not have enough free space"
fi

mkdir -p "$partial_checkpoint"
printf 'Copying %s to %s\n' "$source_checkpoint" "$partial_checkpoint"
rsync -a --partial --info=progress2 "$source_checkpoint/" "$partial_checkpoint/"

verification_output="$(mktemp)"
trap 'rm -f "$verification_output"' EXIT
rsync -a --checksum --dry-run --itemize-changes \
    "$source_checkpoint/" "$partial_checkpoint/" >"$verification_output"
if [[ -s "$verification_output" ]]; then
    printf 'Verification found differences:\n' >&2
    sed -n '1,40p' "$verification_output" >&2
    die "checkpoint verification failed; the partial copy was retained"
fi

mv -- "$partial_checkpoint" "$target_checkpoint"
printf 'Checkpoint copy verified: %s\n' "$target_checkpoint"
printf 'The source was retained for rollback: %s\n' "$source_checkpoint"
printf 'Start the policy server from the target once before deleting the source.\n'
migrate_calvin
