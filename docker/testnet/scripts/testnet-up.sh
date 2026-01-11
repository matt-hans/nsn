#!/bin/bash
# =============================================================================
# Start NSN Testnet
# =============================================================================
# Starts all testnet services in the correct order.
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

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

cd "$TESTNET_DIR"

# Check for .env file
if [[ ! -f ".env" ]]; then
    log_error ".env file not found!"
    echo "Please copy .env.example to .env and configure it:"
    echo "  cp .env.example .env"
    exit 1
fi

# Check for required passwords
source .env
if [[ "${GRAFANA_ADMIN_PASSWORD:-}" == "CHANGE_ME_IN_PRODUCTION" ]] || [[ -z "${GRAFANA_ADMIN_PASSWORD:-}" ]]; then
    log_warn "GRAFANA_ADMIN_PASSWORD is not set or using default!"
    echo "Please set a secure password in .env"
fi

if [[ "${TURN_PASSWORD:-}" == "CHANGE_ME_IN_PRODUCTION" ]] || [[ -z "${TURN_PASSWORD:-}" ]]; then
    log_warn "TURN_PASSWORD is not set or using default!"
    echo "Please set a secure password in .env"
fi

log_info "Starting NSN Testnet..."

# Start services
docker compose up -d

log_info "Waiting for services to become healthy..."

# Wait for validators to start
sleep 10

# Check status
log_info "Checking service status..."
docker compose ps

echo ""
log_info "NSN Testnet started!"
echo ""
echo "Services:"
echo "  Alice RPC:    http://localhost:9944"
echo "  Bob RPC:      http://localhost:9945"
echo "  Charlie RPC:  http://localhost:9946"
echo "  Grafana:      http://localhost:3000"
echo "  Prometheus:   http://localhost:9090"
echo "  Signaling:    http://localhost:8080"
echo ""
echo "View logs with: ./scripts/testnet-logs.sh"
echo "Check status with: ./scripts/testnet-status.sh"
