#!/bin/bash
# =============================================================================
# NSN Testnet Status Check
# =============================================================================
# Displays health status of all testnet services and chain state.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TESTNET_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

cd "$TESTNET_DIR"

echo ""
echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}                    NSN Testnet Status                          ${NC}"
echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
echo ""

# Service status
echo -e "${YELLOW}Service Status:${NC}"
echo "───────────────────────────────────────────────────────────────"
docker compose ps --format "table {{.Name}}\t{{.Status}}\t{{.Ports}}" 2>/dev/null || docker compose ps
echo ""

# Check if validators are responding
echo -e "${YELLOW}Validator Health:${NC}"
echo "───────────────────────────────────────────────────────────────"

check_rpc() {
    local name=$1
    local port=$2
    local response

    response=$(curl -s --connect-timeout 2 "http://localhost:$port/health" 2>/dev/null)
    if [[ $? -eq 0 ]]; then
        echo -e "  $name (port $port): ${GREEN}healthy${NC}"
        return 0
    else
        echo -e "  $name (port $port): ${RED}unreachable${NC}"
        return 1
    fi
}

check_rpc "Alice" 9944 || true
check_rpc "Bob" 9945 || true
check_rpc "Charlie" 9946 || true
echo ""

# Block height
echo -e "${YELLOW}Chain State:${NC}"
echo "───────────────────────────────────────────────────────────────"

get_block_height() {
    local port=$1
    local result

    result=$(curl -s --connect-timeout 2 "http://localhost:$port" \
        -H "Content-Type: application/json" \
        -d '{"id":1,"jsonrpc":"2.0","method":"chain_getHeader"}' 2>/dev/null)

    if [[ $? -eq 0 ]] && [[ -n "$result" ]]; then
        local height
        height=$(echo "$result" | jq -r '.result.number // empty' 2>/dev/null)
        if [[ -n "$height" ]]; then
            # Convert hex to decimal
            printf "%d" "$height" 2>/dev/null || echo "$height"
        else
            echo "N/A"
        fi
    else
        echo "N/A"
    fi
}

ALICE_HEIGHT=$(get_block_height 9944)
BOB_HEIGHT=$(get_block_height 9945)
CHARLIE_HEIGHT=$(get_block_height 9946)

echo "  Alice block height:   $ALICE_HEIGHT"
echo "  Bob block height:     $BOB_HEIGHT"
echo "  Charlie block height: $CHARLIE_HEIGHT"
echo ""

# Peer count
echo -e "${YELLOW}Network Peers:${NC}"
echo "───────────────────────────────────────────────────────────────"

get_peer_count() {
    local port=$1
    local result

    result=$(curl -s --connect-timeout 2 "http://localhost:$port" \
        -H "Content-Type: application/json" \
        -d '{"id":1,"jsonrpc":"2.0","method":"system_peers"}' 2>/dev/null)

    if [[ $? -eq 0 ]] && [[ -n "$result" ]]; then
        echo "$result" | jq '.result | length' 2>/dev/null || echo "N/A"
    else
        echo "N/A"
    fi
}

echo "  Alice peers:   $(get_peer_count 9944)"
echo "  Bob peers:     $(get_peer_count 9945)"
echo "  Charlie peers: $(get_peer_count 9946)"
echo ""

# Off-chain services
echo -e "${YELLOW}Off-Chain Services:${NC}"
echo "───────────────────────────────────────────────────────────────"

check_metrics() {
    local name=$1
    local port=$2

    if curl -s --connect-timeout 2 "http://localhost:$port/metrics" >/dev/null 2>&1; then
        echo -e "  $name metrics (port $port): ${GREEN}available${NC}"
    else
        echo -e "  $name metrics (port $port): ${RED}unavailable${NC}"
    fi
}

check_metrics "Director 1" 9100 || true
check_metrics "Director 2" 9101 || true
check_metrics "Vortex" 9102 || true

# Signaling server
if curl -s --connect-timeout 2 "http://localhost:8080/health" | jq -e '.status == "ok"' >/dev/null 2>&1; then
    PEERS=$(curl -s "http://localhost:8080/health" | jq -r '.peers // 0')
    echo -e "  Signaling (port 8080): ${GREEN}healthy${NC} ($PEERS connected peers)"
else
    echo -e "  Signaling (port 8080): ${RED}unhealthy${NC}"
fi
echo ""

# Observability
echo -e "${YELLOW}Observability:${NC}"
echo "───────────────────────────────────────────────────────────────"

if curl -s --connect-timeout 2 "http://localhost:9090/-/healthy" >/dev/null 2>&1; then
    echo -e "  Prometheus: ${GREEN}healthy${NC} - http://localhost:9090"
else
    echo -e "  Prometheus: ${RED}unhealthy${NC}"
fi

if curl -s --connect-timeout 2 "http://localhost:3000/api/health" >/dev/null 2>&1; then
    echo -e "  Grafana: ${GREEN}healthy${NC} - http://localhost:3000"
else
    echo -e "  Grafana: ${RED}unhealthy${NC}"
fi

echo ""
echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
