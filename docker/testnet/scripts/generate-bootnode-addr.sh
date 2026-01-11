#!/bin/bash
# =============================================================================
# Generate Bootnode Multiaddr
# =============================================================================
# Generates the bootnode multiaddr for use in chain spec configuration.
#
# Usage:
#   ./scripts/generate-bootnode-addr.sh <dns-name> <node-key-file>
#
# Example:
#   ./scripts/generate-bootnode-addr.sh bootnode.nsn.network secrets/bootnode-key
#
# Output:
#   /dns/bootnode.nsn.network/tcp/30333/p2p/12D3KooW...
# =============================================================================

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check arguments
if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <dns-name> <node-key-file>"
    echo ""
    echo "Example:"
    echo "  $0 bootnode.nsn.network secrets/bootnode-key"
    exit 1
fi

DNS_NAME="$1"
NODE_KEY_FILE="$2"

# Check if node key file exists
if [[ ! -f "$NODE_KEY_FILE" ]]; then
    log_error "Node key file not found: $NODE_KEY_FILE"
    exit 1
fi

# Check if subkey is available
if ! command -v subkey &> /dev/null; then
    log_error "subkey command not found!"
    echo ""
    echo "Install subkey:"
    echo "  cargo install subkey --git https://github.com/paritytech/polkadot-sdk"
    exit 1
fi

# Get peer ID from node key
log_info "Reading peer ID from $NODE_KEY_FILE..."
PEER_ID=$(subkey inspect-node-key --file "$NODE_KEY_FILE" 2>/dev/null)

if [[ -z "$PEER_ID" ]]; then
    log_error "Failed to get peer ID from node key"
    exit 1
fi

# Generate multiaddr
MULTIADDR="/dns/$DNS_NAME/tcp/30333/p2p/$PEER_ID"

echo ""
echo "=== Bootnode Configuration ==="
echo ""
echo "DNS Name:  $DNS_NAME"
echo "Peer ID:   $PEER_ID"
echo ""
echo "Multiaddr (for chain spec):"
echo -e "${GREEN}$MULTIADDR${NC}"
echo ""

# Also output IP-based variants for reference
echo "IP-based variants (if needed):"
echo "  IPv4: /ip4/<IP_ADDRESS>/tcp/30333/p2p/$PEER_ID"
echo "  IPv6: /ip6/<IP_ADDRESS>/tcp/30333/p2p/$PEER_ID"
echo ""

# JSON format for chain spec
echo "JSON array format (for testnet.json bootNodes):"
echo "\"bootNodes\": ["
echo "  \"$MULTIADDR\""
echo "]"
