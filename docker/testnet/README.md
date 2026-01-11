# NSN Testnet Deployment

Production-ready Docker Compose configuration for NSN testnet deployment.

## Overview

This deployment includes:

- **3 Validator Nodes** (Alice, Bob, Charlie) - BFT consensus
- **2 Director Nodes** - Off-chain Lane 0 orchestration
- **1 Vortex GPU Node** - AI video generation
- **1 Signaling Server** - WebRTC peer discovery
- **1 IPFS Node** - Content storage
- **Observability Stack** - Prometheus, Grafana, Jaeger (optional)
- **NAT Traversal** - STUN/TURN servers

## Prerequisites

- Docker Engine 24.0+
- Docker Compose 2.20+
- NVIDIA Container Toolkit (for Vortex GPU node)
- At least 16GB RAM
- NVIDIA GPU with 12GB+ VRAM (for Lane 0)

### Check Docker Installation

```bash
docker --version      # Should be 24.0+
docker compose version # Should be 2.20+
nvidia-smi           # Should show GPU
```

## Quick Start

### 1. Configure Environment

```bash
cd docker/testnet

# Copy and edit environment file
cp .env.example .env

# REQUIRED: Set secure passwords
# Edit .env and change:
# - GRAFANA_ADMIN_PASSWORD
# - TURN_PASSWORD
```

### 2. Generate Node Keys

```bash
# Install subkey if not available
cargo install subkey --git https://github.com/paritytech/polkadot-sdk

# Generate validator node keys
cd secrets
for name in alice bob charlie; do
    subkey generate-node-key --file ${name}-node-key
    chmod 600 ${name}-node-key
done

# Get peer IDs and update .env
for name in alice bob charlie; do
    echo "${name} peer ID:"
    subkey inspect-node-key --file ${name}-node-key
done
```

### 3. Generate Chain Spec

```bash
# Build NSN node first (if not done)
cd ../../../nsn-chain
cargo build --release

# Generate testnet chain spec
cd ../docker/testnet
./scripts/generate-testnet-chainspec.sh --raw
```

### 4. Build Docker Images

```bash
# From project root
cd ../..

# Build all images
docker build -f docker/Dockerfile.substrate-local -t nsn-node:latest .
docker build -f docker/Dockerfile.nsn-offchain -t nsn-offchain:latest .
docker build -f docker/Dockerfile.signaling -t nsn-signaling:latest .
docker build -f docker/Dockerfile.vortex -t nsn-vortex:latest .
```

### 5. Start the Testnet

```bash
cd docker/testnet

# Start all services
docker compose up -d

# View logs
docker compose logs -f

# Check status
docker compose ps
```

### 6. Verify Deployment

```bash
# Check all services are healthy
./scripts/testnet-status.sh

# Verify block production
curl -s http://localhost:9944 -H "Content-Type: application/json" \
  -d '{"id":1,"jsonrpc":"2.0","method":"chain_getHeader"}' | jq .result.number
```

## Configuration Reference

### Environment Variables

See `.env.example` for all available options. Key variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `NSN_VERSION` | Docker image version | `latest` |
| `GRAFANA_ADMIN_PASSWORD` | Grafana admin password | **Required** |
| `TURN_PASSWORD` | TURN server password | **Required** |
| `PROMETHEUS_RETENTION_DAYS` | Metrics retention | `15` |
| `VALIDATOR_MEMORY_LIMIT` | Validator RAM limit | `4G` |
| `VORTEX_MAX_VRAM_GB` | GPU VRAM limit | `11.8` |

### Port Mappings

| Service | Port | Protocol | Description |
|---------|------|----------|-------------|
| Alice Validator | 9944 | HTTP/WS | RPC endpoint |
| Alice Validator | 30333 | TCP | P2P networking |
| Alice Validator | 9615 | HTTP | Prometheus metrics |
| Bob Validator | 9945 | HTTP/WS | RPC endpoint |
| Charlie Validator | 9946 | HTTP/WS | RPC endpoint |
| Director 1 | 9100 | HTTP | Prometheus metrics |
| Director 2 | 9101 | HTTP | Prometheus metrics |
| Vortex | 50051 | gRPC | AI engine API |
| Signaling | 8080 | HTTP/WS | WebRTC signaling |
| IPFS | 5001 | HTTP | IPFS API |
| Prometheus | 9090 | HTTP | Metrics UI |
| Grafana | 3000 | HTTP | Dashboards UI |
| Jaeger | 16686 | HTTP | Tracing UI |
| STUN/TURN | 3478 | UDP/TCP | NAT traversal |

## Operations

### Start/Stop

```bash
# Start all services
./scripts/testnet-up.sh

# Stop all services (preserves data)
./scripts/testnet-down.sh

# Stop and remove volumes (clean slate)
docker compose down -v
```

### Logs

```bash
# All services
./scripts/testnet-logs.sh

# Specific service
docker compose logs -f validator-alice
docker compose logs -f director-1

# Last 100 lines
docker compose logs --tail=100 validator-alice
```

### Health Check

```bash
./scripts/testnet-status.sh
```

### Backup Chain Data

```bash
./scripts/backup-chain-data.sh

# Restore from backup
# See backup directory for instructions
```

## Monitoring

### Grafana Dashboards

Access Grafana at http://localhost:3000

Default credentials: `admin` / `<GRAFANA_ADMIN_PASSWORD>`

Available dashboards:
- **NSN Testnet Overview** - Validator status, block production, resources

### Prometheus Alerts

Configured alerts:
- `ValidatorDown` - Validator unreachable
- `BlockProductionStalled` - No new blocks
- `ConsensusStalled` - GRANDPA not progressing
- `ValidatorPeersLow` - Few P2P connections
- `DirectorDown` - Off-chain node unreachable
- `VortexDown` - GPU node unreachable

### Manual Queries

```bash
# Block height
curl -s http://localhost:9944 \
  -H "Content-Type: application/json" \
  -d '{"id":1,"jsonrpc":"2.0","method":"chain_getHeader"}' | jq .result.number

# Peer count
curl -s http://localhost:9944 \
  -H "Content-Type: application/json" \
  -d '{"id":1,"jsonrpc":"2.0","method":"system_peers"}' | jq '.result | length'

# Node info
curl -s http://localhost:9944 \
  -H "Content-Type: application/json" \
  -d '{"id":1,"jsonrpc":"2.0","method":"system_nodeRoles"}' | jq
```

## Upgrade Procedure

### Rolling Validator Upgrade

1. Stop one validator at a time
2. Pull new image
3. Start validator
4. Wait for sync and peer connections
5. Repeat for next validator

```bash
# Example: Upgrade Alice
docker compose stop validator-alice
docker compose pull validator-alice
docker compose up -d validator-alice

# Wait for healthy status
docker compose ps validator-alice

# Check sync
curl -s http://localhost:9944 \
  -d '{"id":1,"jsonrpc":"2.0","method":"system_syncState"}' | jq
```

### Runtime Upgrade

Runtime upgrades are submitted as on-chain extrinsics:

1. Build new runtime WASM
2. Submit `system.setCode` extrinsic via sudo
3. Validators will automatically upgrade at next block

## Troubleshooting

### Validator Won't Start

```bash
# Check logs
docker compose logs validator-alice

# Common issues:
# - Invalid chain spec: Regenerate with generate-testnet-chainspec.sh
# - Node key permission: chmod 600 secrets/*-node-key
# - Port conflict: Check ports aren't in use
```

### No Peers Connecting

```bash
# Check firewall
sudo ufw status
# Ensure 30333-30335 are open

# Check bootnode address
# Verify ALICE_PEER_ID matches actual peer ID
subkey inspect-node-key --file secrets/alice-node-key
```

### Block Production Stopped

```bash
# Check GRANDPA status
curl -s http://localhost:9944 \
  -d '{"id":1,"jsonrpc":"2.0","method":"grandpa_roundState"}' | jq

# Check validator connections
docker compose exec validator-alice curl localhost:9933 \
  -d '{"id":1,"jsonrpc":"2.0","method":"system_peers"}' | jq '.result | length'
```

### GPU Not Available

```bash
# Verify NVIDIA runtime
docker info | grep -i nvidia

# Check GPU visibility
docker compose exec vortex nvidia-smi

# Check CUDA_VISIBLE_DEVICES in .env
```

## Security Considerations

### Production Checklist

- [ ] Changed `GRAFANA_ADMIN_PASSWORD`
- [ ] Changed `TURN_PASSWORD`
- [ ] Generated unique node keys (not dev keys)
- [ ] Configured firewall (only expose necessary ports)
- [ ] Set `NSN_ALLOWED_ORIGINS` to specific domains
- [ ] Disabled anonymous Grafana access (`GRAFANA_ANONYMOUS_ENABLED=false`)
- [ ] Configured TLS termination for public endpoints
- [ ] Set up log aggregation
- [ ] Configured alerting (Alertmanager)

### Network Security

| Port | Public Access | Notes |
|------|---------------|-------|
| 30333-30335 | Yes | P2P required for consensus |
| 9944-9946 | Limited | RPC - use reverse proxy with auth |
| 8080 | Yes | Signaling for viewers |
| 3000, 9090 | No | Monitoring - internal only |

## Directory Structure

```
docker/testnet/
├── docker-compose.yml      # Main compose file
├── .env.example           # Environment template
├── prometheus.yml         # Prometheus config
├── alerts/
│   └── nsn-alerts.yml     # Alerting rules
├── chain-spec/
│   └── testnet.json       # Chain specification
├── config/
│   ├── director.toml      # Director config
│   └── validator.toml     # Validator reference
├── secrets/
│   └── README.md          # Key generation guide
├── grafana/
│   ├── dashboards/        # Dashboard JSON
│   └── provisioning/      # Auto-provisioning
├── bootstrap/
│   ├── docker-compose.yml # Bootstrap node
│   └── README.md          # Setup guide
└── scripts/
    ├── testnet-up.sh      # Start testnet
    ├── testnet-down.sh    # Stop testnet
    ├── testnet-logs.sh    # View logs
    ├── testnet-status.sh  # Health check
    └── backup-chain-data.sh # Backup data
```

## Support

- Documentation: https://docs.nsn.network
- Issues: https://github.com/interdim-cable/nsn/issues
- Telemetry: https://telemetry.nsn.network
