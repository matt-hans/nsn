# NSN Chain

NSN Chain is the on-chain layer of the Neural Sovereign Network (NSN), built using the Polkadot SDK.

## Overview

NSN Chain is a Polkadot SDK-based blockchain that provides the foundation for the decentralized AI-powered video streaming platform. It implements custom FRAME pallets for staking, reputation management, BFT consensus, content storage, and treasury operations.

## Architecture

```
nsn-chain/
├── node/              # NSN node implementation
├── runtime/           # NSN runtime (WASM + native)
├── pallets/           # NSN custom pallets
│   ├── nsn-stake/     # Token staking, slashing, role eligibility
│   ├── nsn-reputation/# Reputation scoring with Merkle proofs
│   ├── nsn-director/  # VRF-based director election, BFT coordination
│   ├── nsn-bft/       # BFT consensus storage and finalization
│   ├── nsn-storage/   # Erasure coding deals and audits
│   └── nsn-treasury/  # Reward distribution and emissions
├── Cargo.toml         # Workspace configuration
├── rust-toolchain.toml # Rust toolchain specification
└── .cargo/config.toml  # Cargo build configuration
```

## Prerequisites

- Rust toolchain: `stable-2024-09-05` (specified in `rust-toolchain.toml`)
- WASM target: `wasm32-unknown-unknown`
- OS: Linux (Ubuntu 22.04+) or macOS
- RAM: 16GB minimum
- Disk: 500GB SSD

## Setup

### 1. Install Rust Toolchain

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env
```

### 2. Install WASM Target

The `rust-toolchain.toml` file automatically configures the correct Rust version and WASM target. Simply navigate to the `nsn-chain` directory:

```bash
cd nsn-chain
# Rust toolchain and wasm32-unknown-unknown target will be installed automatically
```

### 3. Build NSN Chain

```bash
# Development build (faster, with debug symbols)
cargo build

# Release build (optimized, production-ready)
cargo build --release
```

The build process will:
1. Compile the runtime to WASM
2. Compile the runtime to native
3. Build the node binary

## Running NSN Chain

### Development Node (Single Node)

```bash
./target/release/nsn-node --dev
```

This starts a single-node development chain with:
- Instant block production (no consensus delay)
- Alice as sudo account
- Fresh chain state on each run
- RPC server on `ws://127.0.0.1:9944`

### Multi-Node Local Testnet

**Alice (validator node):**

```bash
./target/release/nsn-node \
  --chain=local \
  --alice \
  --port 30333 \
  --rpc-port 9944 \
  --node-key=0000000000000000000000000000000000000000000000000000000000000001
```

**Bob (validator node):**

```bash
./target/release/nsn-node \
  --chain=local \
  --bob \
  --port 30334 \
  --rpc-port 9945 \
  --bootnodes /ip4/127.0.0.1/tcp/30333/p2p/12D3KooWEyoppNCUx8Yx66oV9fJnriXwCcXwDDUA2kj6vnc6iDEp
```

## NSN Custom Pallets

### pallet-nsn-stake

Token staking with role-based requirements:

| Role | Min Stake | Max Stake |
|------|-----------|-----------|
| Director | 100 NSN | 1,000 NSN |
| SuperNode | 50 NSN | 500 NSN |
| Validator | 10 NSN | 100 NSN |
| Relay | 5 NSN | 50 NSN |

**Anti-Centralization:**
- Per-node cap: 1,000 NSN maximum
- Per-region cap: 20% of total stake
- Delegation cap: 5× validator's own stake

### pallet-nsn-reputation

Verifiable reputation events with Merkle proofs:

| Event | Delta |
|-------|-------|
| DirectorSlotAccepted | +100 |
| DirectorSlotRejected | -200 |
| ValidatorVoteCorrect | +5 |
| PinningAuditPassed | +10 |

**Weighted Score:** 50% director + 30% validator + 20% seeder

### pallet-nsn-director

Multi-director election and BFT coordination:
- 5 directors per slot (3-of-5 BFT threshold)
- VRF-based cryptographically secure selection
- 50-block challenge period
- Max 2 directors per region

### pallet-nsn-bft

BFT consensus result storage and finalization

### pallet-nsn-storage

Erasure-coded content storage with audit mechanism:
- Reed-Solomon (10+4) erasure coding
- 5× replication across regions
- Stake-weighted audit probability

### pallet-nsn-treasury

Reward distribution and emissions management

## Testing

```bash
# Run all tests
cargo test --all

# Run specific pallet tests
cargo test -p pallet-nsn-stake
cargo test -p pallet-nsn-reputation

# Run with detailed output
cargo test -- --nocapture
```

## Linting & Formatting

```bash
# Check formatting
cargo fmt --all -- --check

# Apply formatting
cargo fmt --all

# Run clippy
cargo clippy --all-targets --all-features -- -D warnings
```

## Chain Specifications

### Development

- Chain ID: `nsn-dev`
- Block time: 6 seconds
- Finality: Instant (dev mode)
- Initial validators: Alice

### Local Testnet

- Chain ID: `nsn-local`
- Block time: 6 seconds
- Finality: GRANDPA
- Initial validators: Alice, Bob

## Deployment Phases

### Phase A: NSN Solochain (Current)

- Controlled validator set (3-5 trusted operators)
- Fast iteration on chain logic
- Low operational overhead
- Target: MVP deployment

### Phase B: NSN Mainnet

- Public validator onboarding
- Security audit completed
- NSN token launch
- Production deployment

### Phase C: Parachain Migration (Future)

- Cumulus integration
- Polkadot shared security
- XCM cross-chain messaging
- Coretime for scaling

## Configuration Parameters

Key runtime parameters (defined in `runtime/src/configs/mod.rs`):

```rust
// Token unit (12 decimals)
pub const UNIT: Balance = 1_000_000_000_000;

// Staking minimums (from PRD)
pub const MinStakeDirector: Balance = 100 * UNIT;
pub const MinStakeSuperNode: Balance = 50 * UNIT;
pub const MinStakeValidator: Balance = 10 * UNIT;
pub const MinStakeRelay: Balance = 5 * UNIT;

// Anti-centralization
pub const MaxStakePerNode: Balance = 1_000 * UNIT;
pub const MaxRegionPercentage: u32 = 20;
```

## RPC Methods

Connect via `ws://127.0.0.1:9944` (WebSocket) or `http://127.0.0.1:9933` (HTTP).

**NSN Stake:**
- `nsnStake.stakes(AccountId)`: Get stake info for account
- `nsnStake.totalStaked()`: Get total staked in network
- `nsnStake.regionStakes(Region)`: Get stake per region

**NSN Reputation:**
- `nsnReputation.reputationScores(AccountId)`: Get reputation scores

**NSN Director:**
- `nsnDirector.currentSlot()`: Get current slot number
- `nsnDirector.electedDirectors()`: Get elected directors for current slot

## Troubleshooting

### Build fails with "error: failed to run custom build command"

Ensure `wasm32-unknown-unknown` target is installed:

```bash
rustup target add wasm32-unknown-unknown
```

### Node fails to start with "Database version mismatch"

Purge the chain database:

```bash
./target/release/nsn-node purge-chain --dev
```

### RPC connection refused

Ensure the node is running and RPC is enabled:

```bash
./target/release/nsn-node --dev --rpc-external --rpc-cors all
```

## Resources

- [Polkadot SDK Documentation](https://docs.substrate.io/)
- [FRAME Pallets](https://docs.substrate.io/reference/frame-pallets/)
- [NSN Architecture Document](../.claude/rules/architecture.md)
- [NSN PRD](../.claude/rules/prd.md)

## License

GPL-3.0

## Authors

Neural Sovereign Network <dev@nsn.network>
