"""Mutmut configuration for mutation testing.

Mutation testing verifies that tests actually catch bugs by introducing
small code mutations and checking if tests fail.

Usage:
    pip install mutmut
    mutmut run
    mutmut results
    mutmut html

Configuration targets core CLIP ensemble logic, avoiding:
- Logging statements
- Type annotations
- Docstrings
- Configuration constants
"""


def pre_mutation(context):
    """Filter mutations before they are applied."""
    # Skip mutations in test files
    if "test_" in context.filename:
        context.skip = True

    # Skip mutations in __init__.py
    if "__init__.py" in context.filename:
        context.skip = True

    # Skip mutations in logging statements
    if "logger." in context.current_source_line:
        context.skip = True

    # Skip mutations in type annotations
    if context.current_source_line.strip().startswith("def") and "->" in context.current_source_line:
        context.skip = True


# Paths to include in mutation testing
paths_to_mutate = [
    "src/vortex/models/clip_ensemble.py",
]

# Test command
test_command = "pytest tests/unit/test_clip_ensemble.py -x -v"

# Runner (pytest)
runner = "pytest"
