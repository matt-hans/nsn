#!/bin/bash
# =============================================================================
# Setup Bootnode Identity
# =============================================================================
# Generates persistent identity for NSN bootnode.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SECRETS_DIR="$SCRIPT_DIR/secrets"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if subkey is available
if ! command -v subkey &> /dev/null; then
    log_error "subkey command not found!"
    echo ""
    echo "Install subkey:"
    echo "  cargo install subkey --git https://github.com/paritytech/polkadot-sdk"
    echo ""
    echo "Or use Docker:"
    echo "  docker run --rm -v \$(pwd)/secrets:/secrets parity/subkey:latest generate-node-key --file /secrets/bootnode-key"
    exit 1
fi

# Create secrets directory
mkdir -p "$SECRETS_DIR"

# Generate bootnode key
if [[ -f "$SECRETS_DIR/bootnode-key" ]]; then
    log_warn "Bootnode key already exists!"
    echo ""
    read -p "Regenerate? This will change the peer ID! (y/N) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log_info "Keeping existing key"
        echo ""
        echo "Current Peer ID:"
        subkey inspect-node-key --file "$SECRETS_DIR/bootnode-key"
        exit 0
    fi
fi

log_info "Generating bootnode key..."
subkey generate-node-key --file "$SECRETS_DIR/bootnode-key"
chmod 600 "$SECRETS_DIR/bootnode-key"

echo ""
log_info "Bootnode key generated successfully!"
echo ""
echo "========================================"
echo "         BOOTNODE PEER ID"
echo "========================================"
PEER_ID=$(subkey inspect-node-key --file "$SECRETS_DIR/bootnode-key")
echo "$PEER_ID"
echo "========================================"
echo ""

log_info "Next steps:"
echo "1. Configure DNS to point to your server"
echo "2. Update .env with your BOOTNODE_DNS"
echo "3. Start with: docker compose up -d"
echo ""
echo "Generate multiaddr for chain spec:"
echo "  ../scripts/generate-bootnode-addr.sh <your-dns> secrets/bootnode-key"
