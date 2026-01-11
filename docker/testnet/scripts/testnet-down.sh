#!/bin/bash
# =============================================================================
# Stop NSN Testnet
# =============================================================================
# Stops all testnet services. Data volumes are preserved.
# Use -v flag to also remove volumes (clean slate).
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TESTNET_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

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

cd "$TESTNET_DIR"

REMOVE_VOLUMES=false
if [[ "${1:-}" == "-v" ]] || [[ "${1:-}" == "--volumes" ]]; then
    REMOVE_VOLUMES=true
    log_warn "Volume removal requested - all chain data will be deleted!"
    read -p "Are you sure? (y/N) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log_info "Aborted"
        exit 0
    fi
fi

log_info "Stopping NSN Testnet..."

if [[ "$REMOVE_VOLUMES" == "true" ]]; then
    docker compose down -v
    log_info "Services stopped and volumes removed"
else
    docker compose down
    log_info "Services stopped (volumes preserved)"
fi

echo ""
log_info "Done!"
