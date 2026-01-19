#!/bin/bash
# Wrapper script to run Vortex plugin runner with the correct Python environment
cd /home/matt/nsn/vortex
source .venv/bin/activate

# Run plugin with suppressed warnings, only output JSON
export PYTHONWARNINGS=ignore
export TF_CPP_MIN_LOG_LEVEL=3  # Suppress TensorFlow warnings

# Filter out any lines that don't start with { (the JSON output)
python -m vortex.plugins.runner "$@" 2>/dev/null | grep '^{'
