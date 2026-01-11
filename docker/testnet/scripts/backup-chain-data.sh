#!/bin/bash
# =============================================================================
# Backup NSN Testnet Chain Data
# =============================================================================
# Creates timestamped backup of validator chain data volumes.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TESTNET_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BACKUP_DIR="${BACKUP_DIR:-$TESTNET_DIR/backups}"

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

TIMESTAMP=$(date +%Y%m%d-%H%M%S)
BACKUP_PATH="$BACKUP_DIR/$TIMESTAMP"

log_info "Creating backup at: $BACKUP_PATH"

# Create backup directory
mkdir -p "$BACKUP_PATH"

# Check if services are running
RUNNING=$(docker compose ps -q 2>/dev/null | wc -l)
if [[ $RUNNING -gt 0 ]]; then
    log_warn "Services are running. Stopping validators for consistent backup..."
    docker compose stop validator-alice validator-bob validator-charlie
    STOPPED=true
else
    STOPPED=false
fi

# Backup validator data
log_info "Backing up validator data..."

for validator in alice bob charlie; do
    VOLUME_NAME="testnet_${validator}-data"

    log_info "  Backing up $validator..."

    # Create backup using docker cp from a temporary container
    docker run --rm \
        -v "${VOLUME_NAME}:/source:ro" \
        -v "$BACKUP_PATH:/backup" \
        alpine:3.19 \
        tar czf "/backup/${validator}-data.tar.gz" -C /source .

    if [[ $? -eq 0 ]]; then
        SIZE=$(du -h "$BACKUP_PATH/${validator}-data.tar.gz" | cut -f1)
        log_info "    $validator: $SIZE"
    else
        log_error "    Failed to backup $validator"
    fi
done

# Restart validators if they were stopped
if [[ "$STOPPED" == "true" ]]; then
    log_info "Restarting validators..."
    docker compose start validator-alice validator-bob validator-charlie
fi

# Create backup manifest
cat > "$BACKUP_PATH/manifest.json" << EOF
{
    "timestamp": "$TIMESTAMP",
    "date": "$(date -Iseconds)",
    "validators": ["alice", "bob", "charlie"],
    "files": [
        "alice-data.tar.gz",
        "bob-data.tar.gz",
        "charlie-data.tar.gz"
    ]
}
EOF

log_info "Backup complete!"
echo ""
echo "Backup location: $BACKUP_PATH"
echo ""
echo "Files:"
ls -lh "$BACKUP_PATH"
echo ""
echo "To restore:"
echo "  1. Stop validators: docker compose stop validator-alice validator-bob validator-charlie"
echo "  2. Restore data:"
echo "     docker run --rm -v testnet_alice-data:/target -v $BACKUP_PATH:/backup alpine tar xzf /backup/alice-data.tar.gz -C /target"
echo "  3. Restart: docker compose start validator-alice validator-bob validator-charlie"
