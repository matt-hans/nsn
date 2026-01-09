#!/bin/bash
# =============================================================================
# activate-venv.sh - Universal Virtual Environment Activation Hook
# =============================================================================
# Purpose: Automatically discover and activate Python virtual environments
#          before Claude Code executes Bash/shell commands.
#
# Features:
# - Project-agnostic: Works with any project structure
# - Multi-location search: Checks common venv directories and project root
# - Idempotent: Won't re-activate if already in a venv
# - Robust: Handles edge cases and provides clear error messages
# - Portable: Works on macOS, Linux, and WSL
#
# Usage in .claude/settings.json:
#   "hooks": {
#     "PreToolUse": [{
#       "matcher": "Bash",
#       "hooks": [{
#         "type": "command",
#         "command": "source .claude/hooks/activate-venv.sh"
#       }]
#     }]
#   }
# =============================================================================

set -e  # Exit on error (will be caught by Claude Code)

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

# Common virtual environment directory names (in order of preference)
VENV_NAMES=("venv" ".venv" "env" ".env" "virtualenv")

# Common subdirectories where venv might live (relative to project root)
SEARCH_SUBDIRS=("." "backend" "server" "api" "src" "app")

# Maximum depth to search upward for project root
MAX_DEPTH=5

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------

log_info() {
    echo "[venv-hook] $1"
}

log_error() {
    echo "[venv-hook] ERROR: $1" >&2
}

# Find project root by looking for common markers
find_project_root() {
    local dir="$1"
    local depth=0
    
    while [ "$depth" -lt "$MAX_DEPTH" ] && [ "$dir" != "/" ]; do
        # Check for common project root markers
        if [ -f "$dir/.git/config" ] || \
           [ -f "$dir/pyproject.toml" ] || \
           [ -f "$dir/setup.py" ] || \
           [ -f "$dir/requirements.txt" ] || \
           [ -f "$dir/Pipfile" ] || \
           [ -f "$dir/poetry.lock" ] || \
           [ -d "$dir/.claude" ]; then
            echo "$dir"
            return 0
        fi
        dir="$(dirname "$dir")"
        depth=$((depth + 1))
    done
    
    # Fallback to current directory
    echo "$PWD"
    return 0
}

# Check if a directory is a valid virtual environment
is_valid_venv() {
    local venv_path="$1"
    
    # Check for Unix-style venv
    if [ -f "$venv_path/bin/activate" ] && [ -f "$venv_path/bin/python" ]; then
        return 0
    fi
    
    # Check for Windows-style venv (in WSL or Git Bash)
    if [ -f "$venv_path/Scripts/activate" ] && [ -f "$venv_path/Scripts/python.exe" ]; then
        return 0
    fi
    
    return 1
}

# Get the activate script path for a venv
get_activate_script() {
    local venv_path="$1"
    
    if [ -f "$venv_path/bin/activate" ]; then
        echo "$venv_path/bin/activate"
    elif [ -f "$venv_path/Scripts/activate" ]; then
        echo "$venv_path/Scripts/activate"
    else
        return 1
    fi
}

# Search for virtual environment in a directory
find_venv_in_dir() {
    local search_dir="$1"
    
    for venv_name in "${VENV_NAMES[@]}"; do
        local candidate="$search_dir/$venv_name"
        if [ -d "$candidate" ] && is_valid_venv "$candidate"; then
            echo "$candidate"
            return 0
        fi
    done
    
    return 1
}

# Search for virtual environment across project structure
find_venv() {
    local project_root="$1"
    
    # Search in each subdirectory
    for subdir in "${SEARCH_SUBDIRS[@]}"; do
        local search_path
        if [ "$subdir" = "." ]; then
            search_path="$project_root"
        else
            search_path="$project_root/$subdir"
        fi
        
        if [ -d "$search_path" ]; then
            local found_venv
            found_venv=$(find_venv_in_dir "$search_path") && {
                echo "$found_venv"
                return 0
            }
        fi
    done
    
    return 1
}

# -----------------------------------------------------------------------------
# Main Logic
# -----------------------------------------------------------------------------

main() {
    # Check if already in a virtual environment
    if [ -n "$VIRTUAL_ENV" ]; then
        # Verify the venv still exists and is valid
        if is_valid_venv "$VIRTUAL_ENV"; then
            log_info "Already active: $VIRTUAL_ENV"
            return 0
        else
            log_info "Previous venv invalid, searching for new one..."
            unset VIRTUAL_ENV
        fi
    fi
    
    # Find the project root
    local project_root
    project_root=$(find_project_root "$PWD")
    
    # Search for virtual environment
    local venv_path
    venv_path=$(find_venv "$project_root") || {
        # No venv found - this might be intentional (not all projects need venv)
        # Only error if we detect this is a Python project
        if [ -f "$project_root/requirements.txt" ] || \
           [ -f "$project_root/pyproject.toml" ] || \
           [ -f "$project_root/setup.py" ] || \
           [ -f "$project_root/Pipfile" ]; then
            log_error "Python project detected but no virtual environment found."
            log_error "Searched in: $project_root"
            log_error "Expected venv locations: ${VENV_NAMES[*]}"
            log_error "Searched subdirs: ${SEARCH_SUBDIRS[*]}"
            log_error ""
            log_error "To create a venv: python3 -m venv venv"
            return 1
        else
            # Not a Python project, silently skip
            return 0
        fi
    }
    
    # Get the activate script
    local activate_script
    activate_script=$(get_activate_script "$venv_path") || {
        log_error "Found venv at $venv_path but couldn't locate activate script"
        return 1
    }
    
    # Activate the virtual environment
    # shellcheck source=/dev/null
    source "$activate_script" || {
        log_error "Failed to activate virtual environment at $venv_path"
        return 1
    }
    
    log_info "Activated: $venv_path"
    log_info "Python: $(which python)"
    
    return 0
}

# Run main function
main "$@"

