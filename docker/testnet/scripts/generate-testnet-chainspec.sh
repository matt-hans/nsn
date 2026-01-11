#!/bin/bash
# =============================================================================
# Generate NSN Testnet Chain Specification
# =============================================================================
# Generates the testnet chain spec from the nsn_testnet_chain_spec() preset.
#
# Prerequisites:
#   - NSN node binary built: cargo build --release -p nsn-node
#
# Usage:
#   ./scripts/generate-testnet-chainspec.sh [--raw]
#
# Options:
#   --raw    Generate raw chain spec (required for production)
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
OUTPUT_DIR="$SCRIPT_DIR/../chain-spec"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if NSN node binary exists
NSN_NODE=""
if [[ -f "$PROJECT_ROOT/target/release/nsn-node" ]]; then
    NSN_NODE="$PROJECT_ROOT/target/release/nsn-node"
elif [[ -f "$PROJECT_ROOT/nsn-chain/target/release/nsn-node" ]]; then
    NSN_NODE="$PROJECT_ROOT/nsn-chain/target/release/nsn-node"
else
    log_error "NSN node binary not found!"
    log_error "Please build it first:"
    log_error "  cd nsn-chain && cargo build --release -p nsn-node"
    exit 1
fi

log_info "Using NSN node binary: $NSN_NODE"

# Parse arguments
RAW_FLAG=""
if [[ "${1:-}" == "--raw" ]]; then
    RAW_FLAG="--raw"
    log_info "Generating raw chain spec"
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Generate chain spec
log_info "Generating chain spec from nsn-testnet preset..."

if [[ -n "$RAW_FLAG" ]]; then
    # Generate raw chain spec (for production)
    "$NSN_NODE" build-spec --chain nsn-testnet --disable-default-bootnode $RAW_FLAG > "$OUTPUT_DIR/testnet.json"
else
    # Generate human-readable chain spec (for customization)
    "$NSN_NODE" build-spec --chain nsn-testnet --disable-default-bootnode > "$OUTPUT_DIR/testnet-readable.json"
    # Also generate raw version
    "$NSN_NODE" build-spec --chain nsn-testnet --disable-default-bootnode --raw > "$OUTPUT_DIR/testnet.json"
fi

log_info "Chain spec generated successfully!"
log_info "Output: $OUTPUT_DIR/testnet.json"

# Display chain spec summary
if command -v jq &> /dev/null; then
    echo ""
    log_info "Chain Spec Summary:"
    jq '{name: .name, id: .id, chainType: .chainType, protocolId: .protocolId, properties: .properties}' "$OUTPUT_DIR/testnet.json" 2>/dev/null || true
fi

# Reminder for bootnode configuration
echo ""
log_warn "Remember to update bootnode addresses in testnet.json"
log_warn "See scripts/generate-bootnode-addr.sh to generate bootnode multiaddrs"
