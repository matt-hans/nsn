# Claude Code Hooks - Virtual Environment Management

This directory contains hooks for Claude Code to automatically manage Python virtual environments.

## Overview

When Claude Code executes shell commands (via the `Bash` tool), these hooks automatically detect and activate the appropriate Python virtual environment, ensuring:

- **Consistency**: Python commands use the project's isolated dependencies
- **Safety**: No accidental installations to system Python
- **Portability**: Works across different projects without modification

## Files

### `activate-venv.sh`
A standalone shell script for venv activation. Can be sourced directly:

```bash
source .claude/hooks/activate-venv.sh
```

**Features:**
- Project root auto-discovery (looks for `.git`, `pyproject.toml`, etc.)
- Multi-location search (`venv`, `.venv`, `backend/venv`, etc.)
- Idempotent (won't re-activate if already active)
- Clear error messages for debugging

### `venv-discovery.py`
A Python utility for advanced venv discovery and validation:

```bash
# Find venv and print path
python .claude/hooks/venv-discovery.py

# Get JSON details about the venv
python .claude/hooks/venv-discovery.py --format json

# Get shell export commands
python .claude/hooks/venv-discovery.py --format shell

# Validate a specific venv
python .claude/hooks/venv-discovery.py --validate /path/to/venv

# Create venv if not found
python .claude/hooks/venv-discovery.py --create
```

### `virtual-env.py` (Legacy)
The original Python hook - kept for reference but **not recommended** for use.

**Why it doesn't work:**
- Running `subprocess.run('source activate')` creates a child process
- The activation only affects that subprocess
- When it exits, the activation is lost
- Parent process remains unaffected

## Configuration

### Option 1: Inline Hook (Recommended)

The `../.claude/settings.json` contains an inline shell script that handles venv activation. This is the most portable option as it doesn't require external files.

### Option 2: Source External Script

If you prefer to maintain the hook logic separately, modify `settings.json`:

```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Bash",
        "hooks": [
          {
            "type": "command",
            "command": "source \"$(git rev-parse --show-toplevel 2>/dev/null || echo .)/.claude/hooks/activate-venv.sh\""
          }
        ]
      }
    ]
  }
}
```

### Option 3: Project-Specific Path

For monorepos or complex structures, use absolute paths:

```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Bash",
        "hooks": [
          {
            "type": "command",
            "command": "if [ -z \"$VIRTUAL_ENV\" ]; then\n  source /path/to/project/backend/venv/bin/activate\nfi"
          }
        ]
      }
    ]
  }
}
```

## How It Works

1. **Pre-Tool Hook Triggers**: Before Claude Code runs any `Bash` command
2. **Check Current State**: If `$VIRTUAL_ENV` is set and valid, skip activation
3. **Find Project Root**: Search upward for markers (`.git`, `pyproject.toml`, etc.)
4. **Search for Venv**: Check common locations and names
5. **Activate**: Source the `activate` script if found
6. **Log Result**: Print activation status for debugging

## Supported Venv Locations

The hook searches these directory names (in order):
- `venv`
- `.venv`
- `env`
- `.env`
- `virtualenv`

In these subdirectories:
- `.` (project root)
- `backend/`
- `server/`
- `api/`
- `src/`
- `app/`

## Troubleshooting

### "No virtual environment found"

1. Create a venv: `python3 -m venv venv`
2. Verify structure: Check that `venv/bin/activate` exists
3. Check location: Ensure venv is in a searched location

### Hook not running

1. Verify `settings.json` syntax is valid JSON
2. Check that the `matcher` is set to `"Bash"`
3. Ensure hooks are enabled in Claude Code settings

### Wrong Python being used

1. Check `which python` after activation
2. Verify the venv contains the expected Python version
3. Check for conflicting environment variables

### Permission Issues

```bash
chmod +x .claude/hooks/activate-venv.sh
chmod +x .claude/hooks/venv-discovery.py
```

## Customization

### Adding Search Paths

Edit the `SEARCH_SUBDIRS` variable in `activate-venv.sh` or `venv-discovery.py`:

```bash
SEARCH_SUBDIRS=". backend server api src app services"
```

### Adding Venv Names

Edit the `VENV_NAMES` variable:

```bash
VENV_NAMES="venv .venv env .env virtualenv my-venv"
```

### Changing Project Root Markers

Edit the marker checks in `find_project_root()` or `PROJECT_MARKERS` list.

## Windows Support

The hooks include Windows support for Git Bash and WSL:
- Detects `Scripts/activate` (Windows) vs `bin/activate` (Unix)
- Handles `.bat` activation scripts

For native Windows CMD/PowerShell, additional hooks would be needed.

## Best Practices

1. **Always use virtual environments** for Python projects
2. **Include venv in `.gitignore`** - don't commit the venv directory
3. **Maintain `requirements.txt`** or `pyproject.toml` for reproducibility
4. **Use consistent naming** - `venv` is the most common convention
5. **Document setup steps** in your project README

