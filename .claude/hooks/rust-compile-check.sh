#!/usr/bin/env bash
#
# rust-compile-check.sh - Claude Code PostToolUse Hook
#
# Runs cargo check and cargo clippy after Rust files are modified.
# Only triggers when *.rs, Cargo.toml, or Cargo.lock files are changed.
#
# Environment variables (provided by Claude Code):
#   CLAUDE_PROJECT_DIR  - Project root directory
#   CLAUDE_TOOL_NAME    - Tool that was used (Edit, Write, MultiEdit)
#   CLAUDE_FILE_PATHS   - Space-separated list of affected file paths
#

set -euo pipefail

# --- Configuration ---
RUST_PATTERNS=('\.rs$' 'Cargo\.toml$' 'Cargo\.lock$')

# --- Helper Functions ---
is_rust_file() {
    local file="$1"
    for pattern in "${RUST_PATTERNS[@]}"; do
        if [[ "$file" =~ $pattern ]]; then
            return 0
        fi
    done
    return 1
}

# --- Main Logic ---

# Exit silently if no file paths provided
if [[ -z "${CLAUDE_FILE_PATHS:-}" ]]; then
    exit 0
fi

# Check if any changed files are Rust-related
rust_files_changed=false
# Handle space-separated paths (note: won't work perfectly with spaces in filenames)
for file in $CLAUDE_FILE_PATHS; do
    if is_rust_file "$file"; then
        rust_files_changed=true
        break
    fi
done

# Exit silently if no Rust files were changed
if [[ "$rust_files_changed" != "true" ]]; then
    exit 0
fi

# --- Run Rust Checks ---

# Change to project directory if specified
if [[ -n "${CLAUDE_PROJECT_DIR:-}" ]]; then
    cd "$CLAUDE_PROJECT_DIR"
fi

# Find the nearest Cargo.toml (could be in icn-chain subdirectory)
cargo_dir=""
if [[ -f "Cargo.toml" ]]; then
    cargo_dir="."
elif [[ -f "icn-chain/Cargo.toml" ]]; then
    cargo_dir="icn-chain"
else
    # Search for Cargo.toml in changed file paths
    for file in $CLAUDE_FILE_PATHS; do
        dir=$(dirname "$file")
        while [[ "$dir" != "." && "$dir" != "/" ]]; do
            if [[ -f "$dir/Cargo.toml" ]]; then
                cargo_dir="$dir"
                break 2
            fi
            dir=$(dirname "$dir")
        done
    done
fi

# Exit if no Cargo.toml found
if [[ -z "$cargo_dir" ]]; then
    echo "::warning:: No Cargo.toml found, skipping Rust checks"
    exit 0
fi

cd "$cargo_dir"

echo "::group::Rust Compilation Check"
echo "Running cargo check and clippy in $(pwd)..."

# Run cargo check (quiet first, verbose on failure)
if ! cargo check --quiet 2>/dev/null; then
    echo ""
    echo "::error::cargo check failed! Full output:"
    echo "----------------------------------------"
    cargo check 2>&1 || true
    echo "----------------------------------------"
    echo ""
    echo "::endgroup::"
    exit 1
fi

# Run cargo clippy with warnings as errors (quiet first, verbose on failure)
if ! cargo clippy --quiet -- -D warnings 2>/dev/null; then
    echo ""
    echo "::error::cargo clippy failed! Full output:"
    echo "----------------------------------------"
    cargo clippy -- -D warnings 2>&1 || true
    echo "----------------------------------------"
    echo ""
    echo "::endgroup::"
    exit 1
fi

echo "Rust checks passed."
echo "::endgroup::"
exit 0
