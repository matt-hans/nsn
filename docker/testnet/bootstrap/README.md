# NSN Bootstrap Node

Dedicated bootstrap node configuration for NSN testnet network discovery.

## Overview

Bootstrap nodes serve as initial entry points for new nodes joining the network. They:

- Provide peer discovery for new nodes
- Maintain stable, publicly-known peer IDs
- Do not participate in validation or block production
- Should be highly available with good network connectivity

## Quick Start

```bash
# 1. Navigate to bootstrap directory
cd docker/testnet/bootstrap

# 2. Create and populate secrets
mkdir -p secrets
./setup-bootnode.sh

# 3. Configure environment
cp ../.env.example .env
# Edit .env: set BOOTNODE_DNS to your server's DNS name

# 4. Start the bootstrap node
docker compose up -d

# 5. Verify it's running
docker compose logs -f
```

## Setup Script

The `setup-bootnode.sh` script generates persistent identity:

```bash
#!/bin/bash
# setup-bootnode.sh - Generate bootnode identity

set -e

SECRETS_DIR="./secrets"
mkdir -p "$SECRETS_DIR"

if [ ! -f "$SECRETS_DIR/bootnode-key" ]; then
    echo "Generating bootnode key..."
    subkey generate-node-key --file "$SECRETS_DIR/bootnode-key"
    chmod 600 "$SECRETS_DIR/bootnode-key"

    echo ""
    echo "Bootnode Peer ID:"
    subkey inspect-node-key --file "$SECRETS_DIR/bootnode-key"
else
    echo "Bootnode key already exists"
    echo ""
    echo "Bootnode Peer ID:"
    subkey inspect-node-key --file "$SECRETS_DIR/bootnode-key"
fi
```

## DNS Configuration

For the bootnode to be useful, it needs a stable DNS name. Configure:

1. **A Record:** Point DNS to your server's public IP
   ```
   bootnode.nsn.network  A  203.0.113.50
   ```

2. **AAAA Record (optional):** For IPv6 support
   ```
   bootnode.nsn.network  AAAA  2001:db8::1
   ```

3. **Update Environment:**
   ```bash
   BOOTNODE_DNS=bootnode.nsn.network
   ```

## Firewall Rules

The bootnode requires public access on port 30333:

```bash
# UFW
sudo ufw allow 30333/tcp

# iptables
sudo iptables -A INPUT -p tcp --dport 30333 -j ACCEPT

# firewalld
sudo firewall-cmd --permanent --add-port=30333/tcp
sudo firewall-cmd --reload
```

## Generating Bootnode Multiaddr

After the bootnode is running, generate the multiaddr for use in chain spec:

```bash
# Get peer ID from key
PEER_ID=$(subkey inspect-node-key --file secrets/bootnode-key)

# Construct multiaddr
echo "/dns/bootnode.nsn.network/tcp/30333/p2p/$PEER_ID"
```

Example output:
```
/dns/bootnode.nsn.network/tcp/30333/p2p/12D3KooWDpJ7As7BWAwRMfu1VU2WCqNjvq387JEYKDBj4kx6nXTN
```

## Monitoring

### Health Check

```bash
# Check if node is responsive
curl http://localhost:9944/health
```

### Connected Peers

```bash
# Query connected peers via RPC
curl -H "Content-Type: application/json" \
  -d '{"id":1,"jsonrpc":"2.0","method":"system_peers","params":[]}' \
  http://localhost:9944 | jq '.result | length'
```

### Prometheus Metrics

```bash
# Check node metrics
curl http://localhost:9615/metrics | grep substrate_sub_libp2p_peers_count
```

## High Availability

For production, consider running multiple bootnodes:

1. **Geographic Distribution:** Different regions/datacenters
2. **DNS Round-Robin:** Multiple A records for same DNS name
3. **Health-Based Routing:** Use cloud DNS with health checks

### Multiple Bootnode Example

```yaml
# Chain spec bootNodes array
"bootNodes": [
  "/dns/bootnode-us.nsn.network/tcp/30333/p2p/12D3KooWUS...",
  "/dns/bootnode-eu.nsn.network/tcp/30333/p2p/12D3KooWEU...",
  "/dns/bootnode-asia.nsn.network/tcp/30333/p2p/12D3KooWAS..."
]
```

## Backup and Recovery

### Backup Peer Identity

```bash
# The only critical file is the node key
cp secrets/bootnode-key /secure-backup/bootnode-key-$(date +%Y%m%d)
```

### Restore

```bash
# Restore key to new server
mkdir -p secrets
cp /secure-backup/bootnode-key-YYYYMMDD secrets/bootnode-key
chmod 600 secrets/bootnode-key
```

Chain data can be resynced, but the peer identity (node key) must be preserved
to maintain the same peer ID that's configured in chain specs.

## Troubleshooting

### Node won't start

```bash
# Check logs
docker compose logs bootnode

# Verify chain spec
docker compose run --rm bootnode --chain=/chain-spec/testnet.json --version
```

### Peers can't connect

1. Check firewall allows port 30333
2. Verify DNS resolves correctly: `nslookup bootnode.nsn.network`
3. Check `--public-addr` matches actual DNS
4. Ensure node key is readable: `ls -la secrets/bootnode-key`

### Slow sync

Bootstrap nodes sync the full chain. Initial sync may take time:

```bash
# Check sync progress
curl -H "Content-Type: application/json" \
  -d '{"id":1,"jsonrpc":"2.0","method":"system_syncState","params":[]}' \
  http://localhost:9944 | jq
```
