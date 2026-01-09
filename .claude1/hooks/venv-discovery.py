#!/usr/bin/env python3
"""
venv-discovery.py - Virtual Environment Discovery and Validation Utility
==========================================================================

A robust Python utility for discovering and validating virtual environments.
This can be called from shell scripts or used standalone for complex venv
management scenarios.

Features:
- Discovers venvs in project structures (monorepos, microservices, etc.)
- Validates venv integrity (checks for python, pip, activate scripts)
- Returns structured output for shell script consumption
- Supports multiple output formats (path, json, shell-export)

Usage:
    python venv-discovery.py                    # Find and print venv path
    python venv-discovery.py --format json      # Output as JSON
    python venv-discovery.py --format shell     # Output as shell export commands
    python venv-discovery.py --validate /path   # Validate specific venv
    python venv-discovery.py --create           # Create venv if not found

Exit Codes:
    0 - Success (venv found/valid)
    1 - No venv found
    2 - Venv found but invalid
    3 - Error during execution
"""

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, List, Dict, Any


# =============================================================================
# Configuration
# =============================================================================

# Common virtual environment directory names (in order of preference)
VENV_NAMES = ["venv", ".venv", "env", ".env", "virtualenv"]

# Common subdirectories where venv might live (relative to project root)
SEARCH_SUBDIRS = [".", "backend", "server", "api", "src", "app"]

# Project root markers
PROJECT_MARKERS = [
    ".git",
    "pyproject.toml",
    "setup.py",
    "requirements.txt",
    "Pipfile",
    "poetry.lock",
    ".claude",
]

# Maximum depth to search upward for project root
MAX_DEPTH = 5


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class VenvInfo:
    """Information about a discovered virtual environment."""
    path: str
    python_path: str
    activate_script: str
    python_version: Optional[str] = None
    is_valid: bool = True
    packages_count: Optional[int] = None


@dataclass
class DiscoveryResult:
    """Result of venv discovery operation."""
    success: bool
    venv: Optional[VenvInfo] = None
    project_root: Optional[str] = None
    searched_locations: Optional[List[str]] = None
    error: Optional[str] = None


# =============================================================================
# Helper Functions
# =============================================================================

def find_project_root(start_dir: Path) -> Path:
    """
    Find the project root by looking for common markers.
    
    Args:
        start_dir: Directory to start searching from
        
    Returns:
        Path to project root, or start_dir if not found
    """
    current = start_dir.resolve()
    depth = 0
    
    while depth < MAX_DEPTH and current != current.parent:
        for marker in PROJECT_MARKERS:
            marker_path = current / marker
            if marker_path.exists():
                return current
        current = current.parent
        depth += 1
    
    # Fallback to start directory
    return start_dir.resolve()


def is_valid_venv(venv_path: Path) -> bool:
    """
    Check if a directory is a valid virtual environment.
    
    Args:
        venv_path: Path to potential venv directory
        
    Returns:
        True if valid venv, False otherwise
    """
    if not venv_path.is_dir():
        return False
    
    # Check for Unix-style venv
    unix_activate = venv_path / "bin" / "activate"
    unix_python = venv_path / "bin" / "python"
    if unix_activate.exists() and unix_python.exists():
        return True
    
    # Check for Windows-style venv
    win_activate = venv_path / "Scripts" / "activate"
    win_python = venv_path / "Scripts" / "python.exe"
    if win_activate.exists() and win_python.exists():
        return True
    
    return False


def get_venv_info(venv_path: Path) -> Optional[VenvInfo]:
    """
    Get detailed information about a virtual environment.
    
    Args:
        venv_path: Path to venv directory
        
    Returns:
        VenvInfo object or None if invalid
    """
    if not is_valid_venv(venv_path):
        return None
    
    # Determine paths based on OS style
    if (venv_path / "bin" / "python").exists():
        python_path = venv_path / "bin" / "python"
        activate_script = venv_path / "bin" / "activate"
    else:
        python_path = venv_path / "Scripts" / "python.exe"
        activate_script = venv_path / "Scripts" / "activate"
    
    info = VenvInfo(
        path=str(venv_path.resolve()),
        python_path=str(python_path.resolve()),
        activate_script=str(activate_script.resolve()),
        is_valid=True
    )
    
    # Try to get Python version
    try:
        result = subprocess.run(
            [str(python_path), "--version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            info.python_version = result.stdout.strip() or result.stderr.strip()
    except (subprocess.TimeoutExpired, OSError):
        pass
    
    # Try to count installed packages
    try:
        result = subprocess.run(
            [str(python_path), "-m", "pip", "list", "--format=json"],
            capture_output=True,
            text=True,
            timeout=30
        )
        if result.returncode == 0:
            packages = json.loads(result.stdout)
            info.packages_count = len(packages)
    except (subprocess.TimeoutExpired, OSError, json.JSONDecodeError):
        pass
    
    return info


def find_venv_in_dir(search_dir: Path) -> Optional[Path]:
    """
    Search for virtual environment in a specific directory.
    
    Args:
        search_dir: Directory to search in
        
    Returns:
        Path to venv if found, None otherwise
    """
    for venv_name in VENV_NAMES:
        candidate = search_dir / venv_name
        if candidate.is_dir() and is_valid_venv(candidate):
            return candidate
    return None


def discover_venv(start_dir: Optional[Path] = None) -> DiscoveryResult:
    """
    Discover virtual environment in project structure.
    
    Args:
        start_dir: Directory to start search from (defaults to cwd)
        
    Returns:
        DiscoveryResult with venv info or error details
    """
    if start_dir is None:
        start_dir = Path.cwd()
    
    project_root = find_project_root(start_dir)
    searched = []
    
    # Search in each subdirectory
    for subdir in SEARCH_SUBDIRS:
        if subdir == ".":
            search_path = project_root
        else:
            search_path = project_root / subdir
        
        if search_path.is_dir():
            searched.append(str(search_path))
            venv_path = find_venv_in_dir(search_path)
            if venv_path:
                venv_info = get_venv_info(venv_path)
                if venv_info:
                    return DiscoveryResult(
                        success=True,
                        venv=venv_info,
                        project_root=str(project_root),
                        searched_locations=searched
                    )
    
    return DiscoveryResult(
        success=False,
        project_root=str(project_root),
        searched_locations=searched,
        error="No virtual environment found"
    )


def create_venv(target_dir: Path, venv_name: str = "venv") -> DiscoveryResult:
    """
    Create a new virtual environment.
    
    Args:
        target_dir: Directory to create venv in
        venv_name: Name for the venv directory
        
    Returns:
        DiscoveryResult with new venv info
    """
    venv_path = target_dir / venv_name
    
    if venv_path.exists():
        return DiscoveryResult(
            success=False,
            error=f"Directory already exists: {venv_path}"
        )
    
    try:
        subprocess.run(
            [sys.executable, "-m", "venv", str(venv_path)],
            check=True,
            capture_output=True,
            text=True
        )
    except subprocess.CalledProcessError as e:
        return DiscoveryResult(
            success=False,
            error=f"Failed to create venv: {e.stderr}"
        )
    
    venv_info = get_venv_info(venv_path)
    if venv_info:
        return DiscoveryResult(
            success=True,
            venv=venv_info,
            project_root=str(target_dir)
        )
    
    return DiscoveryResult(
        success=False,
        error="Created venv but failed to validate it"
    )


# =============================================================================
# Output Formatters
# =============================================================================

def format_path(result: DiscoveryResult) -> str:
    """Output just the venv path."""
    if result.success and result.venv:
        return result.venv.path
    return ""


def format_json(result: DiscoveryResult) -> str:
    """Output as JSON."""
    data: Dict[str, Any] = {
        "success": result.success,
        "project_root": result.project_root,
        "searched_locations": result.searched_locations,
    }
    
    if result.venv:
        data["venv"] = asdict(result.venv)
    
    if result.error:
        data["error"] = result.error
    
    return json.dumps(data, indent=2)


def format_shell(result: DiscoveryResult) -> str:
    """Output as shell export commands."""
    lines = []
    
    if result.success and result.venv:
        lines.append(f'export VIRTUAL_ENV="{result.venv.path}"')
        lines.append(f'export VENV_PYTHON="{result.venv.python_path}"')
        lines.append(f'export VENV_ACTIVATE="{result.venv.activate_script}"')
        if result.venv.python_version:
            lines.append(f'export VENV_PYTHON_VERSION="{result.venv.python_version}"')
    else:
        lines.append("# No virtual environment found")
        if result.error:
            lines.append(f"# Error: {result.error}")
    
    return "\n".join(lines)


# =============================================================================
# CLI Interface
# =============================================================================

def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Discover and validate Python virtual environments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "-d", "--directory",
        type=Path,
        default=Path.cwd(),
        help="Directory to start search from (default: current directory)"
    )
    
    parser.add_argument(
        "-f", "--format",
        choices=["path", "json", "shell"],
        default="path",
        help="Output format (default: path)"
    )
    
    parser.add_argument(
        "-v", "--validate",
        type=Path,
        metavar="VENV_PATH",
        help="Validate a specific venv path instead of discovering"
    )
    
    parser.add_argument(
        "-c", "--create",
        action="store_true",
        help="Create venv if not found"
    )
    
    parser.add_argument(
        "--venv-name",
        default="venv",
        help="Name for venv directory when creating (default: venv)"
    )
    
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress error messages"
    )
    
    args = parser.parse_args()
    
    # Determine output formatter
    formatters = {
        "path": format_path,
        "json": format_json,
        "shell": format_shell,
    }
    formatter = formatters[args.format]
    
    try:
        # Validate specific path if requested
        if args.validate:
            venv_info = get_venv_info(args.validate)
            if venv_info:
                result = DiscoveryResult(success=True, venv=venv_info)
            else:
                result = DiscoveryResult(
                    success=False,
                    error=f"Invalid venv: {args.validate}"
                )
        else:
            # Discover venv
            result = discover_venv(args.directory)
            
            # Create if requested and not found
            if not result.success and args.create:
                project_root = Path(result.project_root) if result.project_root else args.directory
                result = create_venv(project_root, args.venv_name)
        
        # Output result
        output = formatter(result)
        if output:
            print(output)
        
        # Return appropriate exit code
        if result.success:
            return 0
        elif args.quiet:
            return 1
        else:
            if args.format == "path" and result.error:
                print(f"Error: {result.error}", file=sys.stderr)
            return 1
            
    except Exception as e:
        if not args.quiet:
            print(f"Error: {e}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    sys.exit(main())

