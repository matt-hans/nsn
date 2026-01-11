#!/bin/bash
# =============================================================================
# View NSN Testnet Logs
# =============================================================================
# Aggregates and follows logs from all testnet services.
# Pass service name to view specific service logs.
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TESTNET_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$TESTNET_DIR"

if [[ $# -gt 0 ]]; then
    # View specific service logs
    docker compose logs -f --tail=100 "$@"
else
    # View all logs
    docker compose logs -f --tail=50
fi
