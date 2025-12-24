# ICN Chain

ICN Chain is the on-chain layer of the Interdimensional Cable Network (ICN), built using the Polkadot SDK.

## Overview

ICN Chain is a Polkadot SDK-based blockchain that provides the foundation for the decentralized AI-powered video streaming platform. It implements custom FRAME pallets for staking, reputation management, BFT consensus, content pinning, and treasury operations.

## Architecture

```
icn-chain/
├── node/              # ICN node implementation
├── runtime/           # ICN runtime (WASM + native)
├── pallets/           # ICN custom pallets
│   ├── icn-stake/     # Token staking, slashing, role eligibility
│   ├── icn-reputation/# Reputation scoring with Merkle proofs
│   ├── icn-director/  # VRF-based director election, BFT coordination
│   ├── icn-bft/       # BFT consensus storage and finalization
│   ├── icn-pinning/   # Erasure coding deals and audits
│   └── icn-treasury/  # Reward distribution and emissions
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

The `rust-toolchain.toml` file automatically configures the correct Rust version and WASM target. Simply navigate to the `icn-chain` directory:

```bash
cd icn-chain
# Rust toolchain and wasm32-unknown-unknown target will be installed automatically
```

### 3. Build ICN Chain

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

## Running ICN Chain

### Development Node (Single Node)

```bash
./target/release/icn-node --dev
```

This starts a single-node development chain with:
- Instant block production (no consensus delay)
- Alice as sudo account
- Fresh chain state on each run
- RPC server on `ws://127.0.0.1:9944`

### Multi-Node Local Testnet

**Alice (validator node):**

```bash
./target/release/icn-node \
  --chain=local \
  --alice \
  --port 30333 \
  --rpc-port 9944 \
  --node-key=0000000000000000000000000000000000000000000000000000000000000001
```

**Bob (validator node):**

```bash
./target/release/icn-node \
  --chain=local \
  --bob \
  --port 30334 \
  --rpc-port 9945 \
  --bootnodes /ip4/127.0.0.1/tcp/30333/p2p/12D3KooWEyoppNCUx8Yx66oV9fJnriXwCcXwDDUA2kj6vnc6iDEp
```

## ICN Custom Pallets

### pallet-icn-stake

Token staking with role-based requirements:

| Role | Min Stake | Max Stake |
|------|-----------|-----------|
| Director | 100 ICN | 1,000 ICN |
| SuperNode | 50 ICN | 500 ICN |
| Validator | 10 ICN | 100 ICN |
| Relay | 5 ICN | 50 ICN |

**Anti-Centralization:**
- Per-node cap: 1,000 ICN maximum
- Per-region cap: 20% of total stake
- Delegation cap: 5× validator's own stake

### pallet-icn-reputation

Verifiable reputation events with Merkle proofs:

| Event | Delta |
|-------|-------|
| DirectorSlotAccepted | +100 |
| DirectorSlotRejected | -200 |
| ValidatorVoteCorrect | +5 |
| PinningAuditPassed | +10 |

**Weighted Score:** 50% director + 30% validator + 20% seeder

### pallet-icn-director

Multi-director election and BFT coordination:
- 5 directors per slot (3-of-5 BFT threshold)
- VRF-based cryptographically secure selection
- 50-block challenge period
- Max 2 directors per region

### pallet-icn-bft

BFT consensus result storage and finalization

### pallet-icn-pinning

Erasure-coded content pinning with audit mechanism:
- Reed-Solomon (10+4) erasure coding
- 5× replication across regions
- Stake-weighted audit probability

### pallet-icn-treasury

Reward distribution and emissions management

## Testing

```bash
# Run all tests
cargo test --all

# Run specific pallet tests
cargo test -p pallet-icn-stake
cargo test -p pallet-icn-reputation

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

- Chain ID: `icn-dev`
- Block time: 6 seconds
- Finality: Instant (dev mode)
- Initial validators: Alice

### Local Testnet

- Chain ID: `icn-local`
- Block time: 6 seconds
- Finality: GRANDPA
- Initial validators: Alice, Bob

## Deployment Phases

### Phase A: ICN Solochain (Current)

- Controlled validator set (3-5 trusted operators)
- Fast iteration on chain logic
- Low operational overhead
- Target: MVP deployment

### Phase B: ICN Mainnet

- Public validator onboarding
- Security audit completed
- ICN token launch
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

**ICN Stake:**
- `icnStake.stakes(AccountId)`: Get stake info for account
- `icnStake.totalStaked()`: Get total staked in network
- `icnStake.regionStakes(Region)`: Get stake per region

**ICN Reputation:**
- `icnReputation.reputationScores(AccountId)`: Get reputation scores

**ICN Director:**
- `icnDirector.currentSlot()`: Get current slot number
- `icnDirector.electedDirectors()`: Get elected directors for current slot

## Troubleshooting

### Build fails with "error: failed to run custom build command"

Ensure `wasm32-unknown-unknown` target is installed:

```bash
rustup target add wasm32-unknown-unknown
```

### Node fails to start with "Database version mismatch"

Purge the chain database:

```bash
./target/release/icn-node purge-chain --dev
```

### RPC connection refused

Ensure the node is running and RPC is enabled:

```bash
./target/release/icn-node --dev --rpc-external --rpc-cors all
```

## Resources

- [Polkadot SDK Documentation](https://docs.substrate.io/)
- [FRAME Pallets](https://docs.substrate.io/reference/frame-pallets/)
- [ICN Architecture Document](../.claude/rules/architecture.md)
- [ICN PRD](../.claude/rules/prd.md)

## License

GPL-3.0

## Authors

Interdimensional Cable Network <dev@icn.network>
