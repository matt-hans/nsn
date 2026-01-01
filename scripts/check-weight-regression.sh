#!/bin/bash
# Weight Regression Check Script
# Compares current runtime weights against baseline to detect >10% regressions
#
# Requirements:
#   - jq (for JSON parsing)
#   - baseline.json file with expected weights
#
# Usage:
#   ./check-weight-regression.sh <baseline_path> <current_weights_path>
#   For CI: ./check-weight-regression.sh ../benchmarks/baseline.json ./current-weights.json

set -euo pipefail

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Constants
REGRESSION_THRESHOLD=10  # Percentage threshold for weight regression

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Usage information
usage() {
    cat <<EOF
Weight Regression Check Script

Usage:
    $0 <baseline_path> [current_weights_path]

Arguments:
    baseline_path         Path to baseline.json with expected weights
    current_weights_path  Path to current weights JSON (optional)
                         If not provided, only validates baseline format

Options:
    -h, --help           Show this help message

Exit Codes:
    0 - No regressions detected
    1 - Regressions detected or validation failed
    2 - Invalid arguments or missing dependencies

Examples:
    # Validate baseline format only
    $0 ../benchmarks/baseline.json

    # Compare against current weights
    $0 ../benchmarks/baseline.json ./current-weights.json

EOF
}

# Parse arguments
if [[ $# -lt 1 ]]; then
    log_error "Missing required argument: baseline_path"
    usage
    exit 2
fi

if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    usage
    exit 0
fi

BASELINE_PATH="$1"
CURRENT_WEIGHTS_PATH="${2:-}"

# Validate jq is installed
if ! command -v jq &> /dev/null; then
    log_error "jq not found. Install with: apt-get install jq (Ubuntu) or brew install jq (macOS)"
    exit 2
fi

# Validate baseline file exists
if [[ ! -f "$BASELINE_PATH" ]]; then
    log_error "Baseline file not found: $BASELINE_PATH"
    exit 2
fi

# Validate baseline JSON format
log_info "Validating baseline format..."
if ! jq empty "$BASELINE_PATH" 2>/dev/null; then
    log_error "Invalid JSON format in baseline file"
    exit 1
fi

# Check baseline has required structure
PALLET_COUNT=$(jq -r 'to_entries | map(select(.key | startswith("pallet-"))) | length' "$BASELINE_PATH")
if [[ $PALLET_COUNT -eq 0 ]]; then
    log_error "No pallets found in baseline (expected keys like 'pallet-nsn-stake')"
    exit 1
fi

log_info "✅ Baseline validation passed ($PALLET_COUNT pallets found)"

# If no current weights provided, just validate baseline and exit
if [[ -z "$CURRENT_WEIGHTS_PATH" ]]; then
    log_info "No current weights provided - baseline validation only"
    exit 0
fi

# Validate current weights file exists
if [[ ! -f "$CURRENT_WEIGHTS_PATH" ]]; then
    log_error "Current weights file not found: $CURRENT_WEIGHTS_PATH"
    exit 2
fi

# Validate current weights JSON format
log_info "Validating current weights format..."
if ! jq empty "$CURRENT_WEIGHTS_PATH" 2>/dev/null; then
    log_error "Invalid JSON format in current weights file"
    exit 1
fi

# Compare weights
log_info "Comparing weights against baseline (threshold: ${REGRESSION_THRESHOLD}%)..."

REGRESSIONS_FOUND=false
REGRESSION_DETAILS=""

# Iterate through all pallets in baseline
for pallet in $(jq -r 'to_entries | map(select(.key | startswith("pallet-"))) | .[].key' "$BASELINE_PATH"); do
    # Check if pallet exists in current weights
    if ! jq -e ".[\"$pallet\"]" "$CURRENT_WEIGHTS_PATH" &>/dev/null; then
        log_warn "Pallet $pallet not found in current weights - skipping"
        continue
    fi

    # Iterate through all extrinsics in the pallet
    for extrinsic in $(jq -r ".[\"$pallet\"] | keys[]" "$BASELINE_PATH" 2>/dev/null); do
        # Get baseline weights
        baseline_ref_time=$(jq -r ".[\"$pallet\"][\"$extrinsic\"].ref_time" "$BASELINE_PATH" 2>/dev/null)
        baseline_proof_size=$(jq -r ".[\"$pallet\"][\"$extrinsic\"].proof_size" "$BASELINE_PATH" 2>/dev/null)

        # Skip if baseline values are null or invalid
        if [[ "$baseline_ref_time" == "null" || "$baseline_proof_size" == "null" ]]; then
            continue
        fi

        # Get current weights
        current_ref_time=$(jq -r ".[\"$pallet\"][\"$extrinsic\"].ref_time // \"null\"" "$CURRENT_WEIGHTS_PATH" 2>/dev/null)
        current_proof_size=$(jq -r ".[\"$pallet\"][\"$extrinsic\"].proof_size // \"null\"" "$CURRENT_WEIGHTS_PATH" 2>/dev/null)

        # Skip if extrinsic not in current weights
        if [[ "$current_ref_time" == "null" || "$current_proof_size" == "null" ]]; then
            log_warn "$pallet::$extrinsic not found in current weights"
            continue
        fi

        # Calculate percentage changes
        ref_time_change=$(awk "BEGIN {printf \"%.2f\", (($current_ref_time - $baseline_ref_time) / $baseline_ref_time) * 100}")
        proof_size_change=$(awk "BEGIN {printf \"%.2f\", (($current_proof_size - $baseline_proof_size) / $baseline_proof_size) * 100}")

        # Check for regressions
        ref_time_regression=false
        proof_size_regression=false

        if (( $(awk "BEGIN {print ($ref_time_change > $REGRESSION_THRESHOLD)}") )); then
            ref_time_regression=true
        fi

        if (( $(awk "BEGIN {print ($proof_size_change > $REGRESSION_THRESHOLD)}") )); then
            proof_size_regression=true
        fi

        # Report regressions
        if [[ "$ref_time_regression" == true || "$proof_size_regression" == true ]]; then
            REGRESSIONS_FOUND=true
            log_error "REGRESSION: $pallet::$extrinsic"

            if [[ "$ref_time_regression" == true ]]; then
                log_error "  ref_time: $baseline_ref_time → $current_ref_time (+${ref_time_change}%)"
                REGRESSION_DETAILS+="$pallet::$extrinsic ref_time: +${ref_time_change}%\n"
            fi

            if [[ "$proof_size_regression" == true ]]; then
                log_error "  proof_size: $baseline_proof_size → $current_proof_size (+${proof_size_change}%)"
                REGRESSION_DETAILS+="$pallet::$extrinsic proof_size: +${proof_size_change}%\n"
            fi
        else
            # Log improvements and acceptable changes
            if (( $(awk "BEGIN {print ($ref_time_change < -5)}") )); then
                log_info "IMPROVEMENT: $pallet::$extrinsic ref_time: ${ref_time_change}%"
            fi
        fi
    done
done

# Summary
if [[ "$REGRESSIONS_FOUND" == true ]]; then
    log_error "❌ Weight regressions detected (>${REGRESSION_THRESHOLD}% increase)"
    echo -e "\n${RED}Regression Summary:${NC}"
    echo -e "$REGRESSION_DETAILS"
    exit 1
else
    log_info "✅ No weight regressions detected"
    exit 0
fi
