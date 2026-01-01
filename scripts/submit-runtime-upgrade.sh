#!/bin/bash
# NSN Runtime Upgrade Submission Script
# Submits runtime WASM to NSN Testnet or Mainnet via sudo_unchecked_weight
#
# Requirements:
#   - @polkadot/api-cli installed globally (npm install -g @polkadot/api-cli)
#   - Valid sudo account seed phrase
#   - Network WebSocket endpoint access
#
# Usage:
#   ./submit-runtime-upgrade.sh --network nsn-testnet --wasm ./nsn_runtime.wasm --sudo-seed "..." --ws-url "wss://..."

set -euo pipefail

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Script variables
NETWORK=""
WASM_PATH=""
SUDO_SEED=""
WS_URL=""
DRY_RUN=false

# Logging function
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
NSN Runtime Upgrade Submission Script

Usage:
    $0 --network <NETWORK> --wasm <WASM_PATH> --sudo-seed <SEED> [--ws-url <WS_URL>] [--dry-run]

Options:
    --network       Network name (nsn-testnet or nsn-mainnet)
    --wasm          Path to runtime WASM file
    --sudo-seed     Sudo account seed phrase (keep secure!)
    --ws-url        WebSocket URL (optional, uses default for network)
    --dry-run       Validate parameters without submitting
    -h, --help      Show this help message

Examples:
    # Submit to testnet
    $0 --network nsn-testnet --wasm ./nsn_runtime.wasm --sudo-seed "your seed phrase"

    # Submit to mainnet with custom WS URL
    $0 --network nsn-mainnet --wasm ./nsn_runtime.wasm --sudo-seed "..." --ws-url "wss://mainnet.nsn.network"

    # Dry run validation
    $0 --network nsn-testnet --wasm ./nsn_runtime.wasm --sudo-seed "test" --dry-run

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --network)
            NETWORK="$2"
            shift 2
            ;;
        --wasm)
            WASM_PATH="$2"
            shift 2
            ;;
        --sudo-seed)
            SUDO_SEED="$2"
            shift 2
            ;;
        --ws-url)
            WS_URL="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Validate required parameters
if [[ -z "$NETWORK" ]]; then
    log_error "Missing required parameter: --network"
    usage
    exit 1
fi

if [[ -z "$WASM_PATH" ]]; then
    log_error "Missing required parameter: --wasm"
    usage
    exit 1
fi

if [[ -z "$SUDO_SEED" ]]; then
    log_error "Missing required parameter: --sudo-seed"
    usage
    exit 1
fi

# Set default WS URL if not provided
if [[ -z "$WS_URL" ]]; then
    case "$NETWORK" in
        nsn-testnet)
            WS_URL="wss://testnet.nsn.network"
            log_info "Using default testnet WS URL: $WS_URL"
            ;;
        nsn-mainnet)
            WS_URL="wss://mainnet.nsn.network"
            log_warn "Using default mainnet WS URL: $WS_URL"
            log_warn "⚠️  MAINNET DEPLOYMENT - Double check all parameters!"
            ;;
        *)
            log_error "Invalid network: $NETWORK (must be nsn-testnet or nsn-mainnet)"
            exit 1
            ;;
    esac
fi

# Validate WASM file exists
if [[ ! -f "$WASM_PATH" ]]; then
    log_error "WASM file not found: $WASM_PATH"
    exit 1
fi

# Check WASM file size
WASM_SIZE=$(stat -f%z "$WASM_PATH" 2>/dev/null || stat -c%s "$WASM_PATH" 2>/dev/null)
WASM_SIZE_MB=$(( WASM_SIZE / 1024 / 1024 ))
log_info "WASM file size: ${WASM_SIZE_MB} MB"

if [[ $WASM_SIZE -gt 10485760 ]]; then # 10MB
    log_warn "⚠️  WASM file is larger than 10MB - this may fail on-chain"
fi

# Check if node is installed
if ! command -v node &> /dev/null; then
    log_error "node not found. Install Node.js from: https://nodejs.org/"
    exit 1
fi

# Check if polkadot-js-api is installed
if ! command -v polkadot-js-api &> /dev/null; then
    log_error "polkadot-js-api not found. Install with: npm install -g @polkadot/api-cli"
    exit 1
fi

# Summary
log_info "Runtime Upgrade Summary:"
log_info "  Network: $NETWORK"
log_info "  WS URL: $WS_URL"
log_info "  WASM: $WASM_PATH (${WASM_SIZE_MB} MB)"
log_info "  Dry Run: $DRY_RUN"

# Dry run mode - exit after validation
if [[ "$DRY_RUN" == true ]]; then
    log_info "✅ Dry run validation passed - parameters are valid"
    exit 0
fi

# Confirmation prompt for mainnet
if [[ "$NETWORK" == "nsn-mainnet" ]]; then
    log_warn "⚠️  You are about to submit a runtime upgrade to MAINNET"
    read -p "Type 'CONFIRM MAINNET UPGRADE' to proceed: " confirmation
    if [[ "$confirmation" != "CONFIRM MAINNET UPGRADE" ]]; then
        log_error "Mainnet upgrade cancelled"
        exit 1
    fi
fi

# Convert WASM to hex
log_info "Converting WASM to hex..."
WASM_HEX="0x$(hexdump -ve '1/1 "%.2x"' "$WASM_PATH")"
WASM_HEX_SIZE=${#WASM_HEX}
log_info "Hex encoded size: $WASM_HEX_SIZE characters"

# Create temporary script for polkadot-js-api
TEMP_SCRIPT=$(mktemp)
# Secure the temporary file (only owner can read/write)
chmod 600 "$TEMP_SCRIPT"
# Ensure cleanup on exit
trap 'rm -f "$TEMP_SCRIPT"' EXIT
cat > "$TEMP_SCRIPT" <<EOF
// Submit runtime upgrade via sudo_unchecked_weight
const { ApiPromise, WsProvider } = require('@polkadot/api');
const { Keyring } = require('@polkadot/keyring');

async function main() {
    const provider = new WsProvider('$WS_URL');
    const api = await ApiPromise.create({ provider });

    const keyring = new Keyring({ type: 'sr25519' });
    const sudo = keyring.addFromUri('$SUDO_SEED');

    console.log('[INFO] Connected to', await api.rpc.system.chain());
    console.log('[INFO] Using sudo account:', sudo.address);

    // Create setCode call
    const code = '$WASM_HEX';
    const setCode = api.tx.system.setCode(code);

    // Wrap in sudo_unchecked_weight
    const sudoTx = api.tx.sudo.sudoUncheckedWeight(setCode, { refTime: 1000000000, proofSize: 1000000 });

    // Submit and watch
    console.log('[INFO] Submitting runtime upgrade...');
    const unsub = await sudoTx.signAndSend(sudo, ({ status, events }) => {
        if (status.isInBlock) {
            console.log('[INFO] Included in block:', status.asInBlock.toHex());
        } else if (status.isFinalized) {
            console.log('[SUCCESS] Finalized in block:', status.asFinalized.toHex());

            events.forEach(({ event }) => {
                if (api.events.system.ExtrinsicSuccess.is(event)) {
                    console.log('[SUCCESS] ✅ Runtime upgrade successful!');
                } else if (api.events.system.ExtrinsicFailed.is(event)) {
                    const [dispatchError] = event.data;
                    let errorInfo = dispatchError.toString();

                    if (dispatchError.isModule) {
                        const decoded = api.registry.findMetaError(dispatchError.asModule);
                        errorInfo = \`\${decoded.section}.\${decoded.name}: \${decoded.docs}\`;
                    }

                    console.error('[ERROR] ❌ Extrinsic failed:', errorInfo);
                    process.exit(1);
                }
            });

            unsub();
            process.exit(0);
        }
    });
}

main().catch((error) => {
    console.error('[ERROR] ❌ Runtime upgrade failed:', error);
    process.exit(1);
});
EOF

# Execute the upgrade
log_info "Submitting runtime upgrade to $NETWORK..."
log_info "This may take several minutes..."

if node "$TEMP_SCRIPT" 2>&1 | tee deployment.log; then
    log_info "✅ Runtime upgrade completed successfully!"
    exit 0
else
    log_error "❌ Runtime upgrade failed - check deployment.log for details"
    exit 1
fi
# Note: TEMP_SCRIPT cleanup is handled by trap on EXIT
