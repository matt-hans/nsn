# NSN Chain Specification Guide

## Overview

This guide explains how to generate and manage NSN Chain specifications for different network environments.

## Available Chain Specs

### Development

- **ID**: `dev`
- **Purpose**: Local single-node development
- **Validators**: Alice, Bob (well-known test keys)
- **Token Allocations**: All well-known accounts funded equally

### Local Testnet

- **ID**: `local`
- **Purpose**: Multi-node local testing
- **Validators**: Alice, Bob (well-known test keys)
- **Token Allocations**: All well-known accounts funded equally

### NSN Testnet

- **ID**: `nsn-testnet`
- **Purpose**: Public testnet for integration testing
- **Validators**: Alice, Bob, Charlie (replace for production testnet)
- **Token Allocations**: Generous allocations for testing

### NSN Mainnet (Template)

- **ID**: `nsn-mainnet`
- **Purpose**: Production mainnet (TEMPLATE ONLY)
- **Validators**: TBD (replace with actual production keys)
- **Token Allocations**: Production distribution (1B total supply)

## Generating Chain Specs

### 1. Human-Readable Chain Spec

```bash
# Development
./target/release/nsn-node build-spec --chain=dev > chain-specs/nsn-dev.json

# Local testnet
./target/release/nsn-node build-spec --chain=local > chain-specs/nsn-local.json

# NSN Testnet
./target/release/nsn-node build-spec --chain=nsn-testnet > chain-specs/nsn-testnet.json

# NSN Mainnet (template)
./target/release/nsn-node build-spec --chain=nsn-mainnet > chain-specs/nsn-mainnet.json
```

### 2. Raw Chain Spec (For Distribution)

Raw chain specs are optimized for node startup and should be used in production.

```bash
# NSN Testnet (raw)
./target/release/nsn-node build-spec \
  --chain=nsn-testnet \
  --raw > chain-specs/nsn-testnet-raw.json

# NSN Mainnet (raw)
./target/release/nsn-node build-spec \
  --chain=nsn-mainnet \
  --raw > chain-specs/nsn-mainnet-raw.json
```

## Using Chain Specs

### Start Node with Chain Spec ID

```bash
# Development
./target/release/nsn-node --chain=dev --alice --tmp

# NSN Testnet
./target/release/nsn-node --chain=nsn-testnet --validator --name "My-Validator"
```

### Start Node with Chain Spec File

```bash
./target/release/nsn-node \
  --chain=./chain-specs/nsn-testnet-raw.json \
  --validator \
  --name "My-Validator" \
  --base-path /var/lib/nsn-data
```

## Chain Properties

All NSN chain specs share these properties:

```json
{
  "tokenSymbol": "NSN",
  "tokenDecimals": 18,
  "ss58Format": 42
}
```

### Token Units

| Unit | Value | Base Units |
|------|-------|------------|
| 1 NSN | 1.0 | 1,000,000,000,000,000,000 (10^18) |
| 1 mNSN | 0.001 | 1,000,000,000,000,000 (10^15) |
| 1 µNSN | 0.000001 | 1,000,000,000,000 (10^12) |

### Constants

- **Existential Deposit**: 0.001 NSN (1 mNSN)
- **Block Time**: 6 seconds
- **Epoch Duration**: 100 blocks (~10 minutes)

## Customizing Chain Specs

### Modify Genesis Accounts (Mainnet Only)

Before mainnet launch, update the genesis configuration in `/nsn-chain/runtime/src/genesis_config_presets.rs`:

```rust
fn nsn_mainnet_genesis_template() -> Value {
    // Replace with actual production accounts
    let treasury_account = AccountId::from_ss58check("5GrwvaEF...").unwrap();
    let dev_fund_account = AccountId::from_ss58check("5FHneW46...").unwrap();
    // ...
}
```

### Add Bootnodes

Edit the chain spec functions in `/nsn-chain/node/src/chain_spec.rs`:

```rust
pub fn nsn_testnet_chain_spec() -> ChainSpec {
    ChainSpec::builder(...)
        .with_boot_nodes(vec![
            "/dns/boot1.nsn.network/tcp/30333/p2p/12D3KooW...".parse().unwrap(),
            "/dns/boot2.nsn.network/tcp/30333/p2p/12D3KooW...".parse().unwrap(),
        ])
        .build()
}
```

## Verifying Chain Specs

### Inspect Chain Spec

```bash
cat chain-specs/nsn-testnet.json | jq '.properties'
```

Expected output:

```json
{
  "ss58Format": 42,
  "tokenDecimals": 18,
  "tokenSymbol": "NSN"
}
```

### Validate Genesis Balances

```bash
cat chain-specs/nsn-mainnet.json | jq '.genesis.runtimeGenesis.patch.balances.balances'
```

### Check Validator Session Keys

```bash
cat chain-specs/nsn-testnet.json | jq '.genesis.runtimeGenesis.patch.session.keys'
```

## Production Mainnet Checklist

Before generating the final mainnet chain spec:

- [ ] Replace all test accounts with production accounts
- [ ] Replace validator session keys with actual production keys
- [ ] Update bootnode addresses with production infrastructure
- [ ] Verify total token supply matches tokenomics (1B NSN)
- [ ] Verify allocation percentages:
  - [ ] Treasury: 40% (400M NSN)
  - [ ] Development: 20% (200M NSN)
  - [ ] Ecosystem: 15% (150M NSN)
  - [ ] Team: 15% (150M NSN)
  - [ ] Liquidity: 10% (100M NSN)
- [ ] Security audit completed
- [ ] Runtime benchmarks executed
- [ ] Migration plan documented
- [ ] Generate raw chain spec for distribution
- [ ] Test genesis on private network first

## Available Presets

Query available genesis presets:

```bash
./target/release/nsn-node build-spec --list-presets
```

Expected output:

```
dev
local-testnet
nsn-testnet
nsn-mainnet
```

## Common Commands

### Start Development Node

```bash
./target/release/nsn-node --dev --tmp
```

### Start Local Multi-Node Network

```bash
# Terminal 1 (Alice)
./target/release/nsn-node --chain=local --alice --tmp --port 30333

# Terminal 2 (Bob)
./target/release/nsn-node --chain=local --bob --tmp --port 30334 \
  --bootnodes /ip4/127.0.0.1/tcp/30333/p2p/ALICE_PEER_ID
```

### Purge Chain Database

```bash
./target/release/nsn-node purge-chain --chain=nsn-testnet --base-path /var/lib/nsn-data
```

## Troubleshooting

### Error: "WASM binary not available"

```bash
# Rebuild runtime
cargo build --release

# Or with optimization
cargo build --release --features on-chain-release-build
```

### Error: "Invalid chain spec"

Ensure you're using the raw chain spec for production:

```bash
./target/release/nsn-node build-spec --chain=nsn-testnet --raw > nsn-testnet-raw.json
./target/release/nsn-node --chain=./nsn-testnet-raw.json
```

### Chain Spec Doesn't Include Custom Genesis

Verify the preset is registered in `genesis_config_presets.rs`:

```rust
pub fn get_preset(id: &PresetId) -> Option<vec::Vec<u8>> {
    let patch = match id.as_ref() {
        NSN_TESTNET_PRESET => nsn_testnet_genesis(),
        NSN_MAINNET_PRESET => nsn_mainnet_genesis_template(),
        // ...
    };
    // ...
}
```

## References

- [Polkadot SDK Chain Spec Documentation](https://paritytech.github.io/polkadot-sdk/master/polkadot_sdk_docs/reference_docs/chain_spec_genesis/index.html)
- [Substrate Genesis Configuration](https://docs.substrate.io/build/genesis-configuration/)
- [NSN Validator Onboarding](./validator-onboarding.md)

---

**Last Updated**: 2025-12-31
**Version**: 1.0.0
