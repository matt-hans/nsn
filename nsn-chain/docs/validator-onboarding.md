# NSN Validator Onboarding Guide

## Overview

This guide walks you through the process of setting up and running an NSN Chain validator node. NSN Chain uses Aura (Authority Round) for block production and GRANDPA for finalization.

## Prerequisites

### Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU | 4 cores | 8+ cores |
| RAM | 16 GB | 32 GB |
| Storage | 500 GB SSD | 1 TB NVMe SSD |
| Network | 100 Mbps | 1 Gbps |

### Software Requirements

- **Operating System**: Ubuntu 22.04 LTS or later
- **Rust**: 1.75.0 or later
- **Build Tools**: `build-essential`, `clang`, `libssl-dev`, `pkg-config`

## Step 1: Build NSN Node

### Clone the Repository

```bash
git clone https://github.com/neural-sovereign-network/nsn-chain.git
cd nsn-chain
```

### Install Rust (if not already installed)

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env
rustup default stable
rustup update
rustup target add wasm32-unknown-unknown
```

### Build the Node

```bash
# Development build (faster compile, slower runtime)
cargo build --release

# Production build (slower compile, optimized runtime)
cargo build --release --features on-chain-release-build
```

The compiled binary will be at `./target/release/nsn-node`.

## Step 2: Generate Session Keys

Session keys are used for block production and finalization. Each validator needs unique keys.

### Option A: Generate Keys via RPC (Recommended)

1. Start your node in development mode:

```bash
./target/release/nsn-node --dev --tmp
```

2. In a new terminal, generate keys:

```bash
curl -H "Content-Type: application/json" \
  -d '{"id":1, "jsonrpc":"2.0", "method": "author_rotateKeys", "params":[]}' \
  http://localhost:9944
```

3. The response will contain your session keys (hex string):

```json
{
  "jsonrpc":"2.0",
  "result":"0x1234567890abcdef...",
  "id":1
}
```

**IMPORTANT**: Save these keys securely. You'll need them for your validator account.

### Option B: Generate Keys Manually

```bash
# Generate Aura key
./target/release/nsn-node key generate --scheme Sr25519 --password-interactive

# Output example:
# Secret phrase: `your twelve word mnemonic phrase here is very important`
# Network ID: substrate
# Secret seed: 0x...
# Public key (hex): 0x...
# Account ID: 0x...
# Public key (SS58): 5...
# SS58 Address: 5...
```

**IMPORTANT**: Store the secret phrase and seeds in a secure location (e.g., encrypted vault, HSM).

## Step 3: Set Up Validator Account

### Using Polkadot.js Apps

1. Navigate to [Polkadot.js Apps](https://polkadot.js.org/apps/?rpc=ws://localhost:9944)
2. Go to **Accounts** → **Add Account**
3. Import your validator account using the secret phrase from Step 2
4. Fund the account with NSN tokens (minimum 100 NSN for Director role)

### Set Session Keys On-Chain

1. Go to **Developer** → **Extrinsics**
2. Select your validator account
3. Call `session.setKeys(keys, proof)`
   - `keys`: Paste the session keys from Step 2
   - `proof`: `0x00`
4. Sign and submit the transaction

## Step 4: Configure Your Node

### Create systemd Service (Recommended for Production)

```bash
sudo nano /etc/systemd/system/nsn-validator.service
```

Add the following configuration:

```ini
[Unit]
Description=NSN Chain Validator
After=network.target

[Service]
Type=simple
User=nsn
WorkingDirectory=/home/nsn/nsn-chain
ExecStart=/home/nsn/nsn-chain/target/release/nsn-node \
  --chain=nsn-testnet \
  --validator \
  --name "Your-Validator-Name" \
  --base-path /var/lib/nsn-data \
  --port 30333 \
  --rpc-port 9944 \
  --prometheus-port 9615 \
  --rpc-cors all \
  --rpc-methods Safe \
  --telemetry-url "wss://telemetry.polkadot.io/submit/ 0"

Restart=always
RestartSec=120

[Install]
WantedBy=multi-user.target
```

### Enable and Start the Service

```bash
sudo systemctl daemon-reload
sudo systemctl enable nsn-validator.service
sudo systemctl start nsn-validator.service
```

### Check Service Status

```bash
sudo systemctl status nsn-validator.service
sudo journalctl -u nsn-validator.service -f
```

## Step 5: Stake and Register as Validator

### Stake NSN Tokens

1. Go to **Developer** → **Extrinsics**
2. Select your validator account
3. Call `nsnStake.depositStake(amount, lockBlocks, region)`
   - `amount`: Minimum 100 NSN (in base units: 100 × 10^18)
   - `lockBlocks`: Lock duration (e.g., 100800 blocks ≈ 7 days)
   - `region`: Your geographic region (0-6)
4. Sign and submit

### Verify Validator Status

```bash
# Check if your node is validating
curl -H "Content-Type: application/json" \
  -d '{"id":1, "jsonrpc":"2.0", "method":"author_hasSessionKeys", "params":["YOUR_SESSION_KEYS"]}' \
  http://localhost:9944
```

Expected response:

```json
{
  "jsonrpc":"2.0",
  "result":true,
  "id":1
}
```

## Step 6: Monitor Your Validator

### Check Block Production

```bash
# View logs
sudo journalctl -u nsn-validator.service -f | grep "Prepared block"
```

### Prometheus Metrics

Access metrics at `http://localhost:9615/metrics`

Key metrics to monitor:
- `substrate_block_height{status="best"}` - Current best block
- `substrate_block_height{status="finalized"}` - Finalized block
- `substrate_ready_transactions_number` - Pending transactions
- `substrate_sub_libp2p_peers_count` - Peer count

### Grafana Dashboard (Optional)

Import NSN Chain dashboard for comprehensive monitoring.

## Step 7: Chain Spec Generation (For New Networks)

### Generate Chain Spec (Human-Readable)

```bash
./target/release/nsn-node build-spec --chain=nsn-testnet > nsn-testnet.json
```

### Generate Raw Chain Spec (For Distribution)

```bash
./target/release/nsn-node build-spec --chain=nsn-testnet --raw > nsn-testnet-raw.json
```

### Start Node with Custom Chain Spec

```bash
./target/release/nsn-node \
  --chain=./nsn-testnet-raw.json \
  --validator \
  --name "Validator-1" \
  --base-path /var/lib/nsn-data
```

## Troubleshooting

### Node Not Syncing

1. Check peer count:

```bash
curl -H "Content-Type: application/json" \
  -d '{"id":1, "jsonrpc":"2.0", "method":"system_health"}' \
  http://localhost:9944
```

2. Add bootnodes manually:

```bash
./target/release/nsn-node \
  --chain=nsn-testnet \
  --validator \
  --bootnodes /dns/boot1.nsn.network/tcp/30333/p2p/12D3KooW...
```

### Session Keys Not Working

1. Verify keys are set on-chain:

```bash
curl -H "Content-Type: application/json" \
  -d '{"id":1, "jsonrpc":"2.0", "method":"author_hasSessionKeys", "params":["YOUR_KEYS"]}' \
  http://localhost:9944
```

2. Regenerate keys if necessary (see Step 2)

### Database Corruption

```bash
# Purge chain database and resync
./target/release/nsn-node purge-chain --chain=nsn-testnet --base-path /var/lib/nsn-data
```

### Insufficient Stake

Ensure your account has at least 100 NSN staked:

```bash
# Query stake via Polkadot.js Apps
# Developer → Chain State → nsnStake → stakes(AccountId)
```

## Security Best Practices

### Key Management

- **NEVER** share your secret phrase or seed
- Store keys in encrypted vaults or HSMs
- Use unique keys for testnet and mainnet
- Rotate session keys periodically

### Firewall Configuration

```bash
# Allow P2P traffic
sudo ufw allow 30333/tcp

# Allow RPC (only from trusted IPs)
sudo ufw allow from TRUSTED_IP to any port 9944

# Allow Prometheus (internal network only)
sudo ufw allow from 10.0.0.0/8 to any port 9615

# Enable firewall
sudo ufw enable
```

### SSH Hardening

```bash
# Disable root login
sudo nano /etc/ssh/sshd_config
# Set: PermitRootLogin no
# Set: PasswordAuthentication no

sudo systemctl restart sshd
```

### System Updates

```bash
# Enable automatic security updates
sudo apt install unattended-upgrades
sudo dpkg-reconfigure --priority=low unattended-upgrades
```

## Support and Resources

- **Documentation**: https://docs.nsn.network
- **GitHub**: https://github.com/neural-sovereign-network/nsn-chain
- **Discord**: https://discord.gg/nsn
- **Telemetry**: https://telemetry.polkadot.io

## Appendix: Genesis Accounts (Testnet Only)

For NSN Testnet, the following well-known accounts are pre-funded:

| Account | Address | Balance |
|---------|---------|---------|
| Alice | 5GrwvaEF... | 1,000,000 NSN |
| Bob | 5FHneW46... | 1,000,000 NSN |
| Charlie | 5FLSigC9... | 1,000,000 NSN |
| Dave | 5DAAnrj7... | 1,000,000 NSN |
| Eve | 5HGjWAe... | 1,000,000 NSN |
| Ferdie | 5CiPPse... | 1,000,000 NSN |

**WARNING**: These accounts are for testnet ONLY. Never use these keys in production.

## Appendix: Chain Specifications

### NSN Testnet

- **Chain ID**: `nsn-testnet`
- **Token Symbol**: NSN
- **Decimals**: 18
- **SS58 Format**: 42
- **Block Time**: 6 seconds
- **Epoch Duration**: 100 blocks (~10 minutes)

### NSN Mainnet (Template)

- **Chain ID**: `nsn-mainnet`
- **Token Symbol**: NSN
- **Decimals**: 18
- **SS58 Format**: 42
- **Total Supply**: 1,000,000,000 NSN
- **Allocations**:
  - Treasury: 40% (400M NSN)
  - Development: 20% (200M NSN)
  - Ecosystem: 15% (150M NSN)
  - Team: 15% (150M NSN)
  - Liquidity: 10% (100M NSN)

---

**Last Updated**: 2025-12-31
**Version**: 1.0.0
