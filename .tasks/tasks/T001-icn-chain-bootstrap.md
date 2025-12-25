# Task T001: ICN Polkadot SDK Chain Bootstrap and Development Environment

## Metadata
```yaml
id: T001
title: ICN Polkadot SDK Chain Bootstrap and Development Environment
status: completed
priority: P1
tags: [foundation, infrastructure, substrate, polkadot-sdk, chain]
estimated_tokens: 8000
actual_tokens: 0
dependencies: []
created_at: 2025-12-24
updated_at: 2025-12-24
```

## Description

Bootstrap ICN as its own Polkadot SDK chain using the official parachain template, set up the local development environment, and configure the workspace for ICN custom pallet development. This establishes the foundation for all Substrate pallet work and enables full sovereignty over runtime upgrades.

## Business Context

**Why this matters**: ICN requires its own blockchain to enable trustless coordination of AI video generation without external governance dependencies. Building as a Polkadot SDK chain provides full control over runtime upgrades, deployment timeline, and chain parameters.

**Value delivered**: 
- Enables all subsequent pallet development tasks (T002-T007)
- Eliminates Moonbeam governance dependency
- Provides staged deployment path: Solochain → Parachain → Coretime
- Supports "deployable by anyone" - open-source chain artifacts

## Acceptance Criteria

1. ICN Chain project bootstrapped from Polkadot SDK parachain template
2. Project renamed and branded for ICN (icn-node, icn-runtime, etc.)
3. Rust toolchain matches Polkadot SDK requirements (polkadot-stable2409)
4. `wasm32-unknown-unknown` target installed
5. Repository builds successfully: `cargo build --release`
6. Local ICN dev node runs: `./target/release/icn-node --dev`
7. `pallets/` directory structure created for ICN custom pallets
8. `.cargo/config.toml` configured for optimal build settings
9. CI/CD workflow skeleton committed (`.github/workflows/icn-chain.yml`)
10. Development branch created: `feature/icn-chain-bootstrap`

## Test Scenarios

### Scenario 1: Fresh Project Clone and Build
```gherkin
GIVEN a clean development machine with Rust installed
  AND Polkadot SDK parachain template cloned as base
WHEN developer runs:
  git clone <repo-url> icn-chain
  cd icn-chain
  cargo build --release
THEN build completes without errors
  AND target/release/icn-node binary exists
  AND binary size >50MB
```

### Scenario 2: Successful Local Dev Node Launch
```gherkin
GIVEN compiled icn-node binary exists
WHEN developer runs: ./target/release/icn-node --dev
THEN node starts and produces blocks
  AND RPC server listens on ws://127.0.0.1:9944
  AND node logs show block production
  AND Ctrl+C cleanly shuts down node
```

### Scenario 3: Multi-Node Local Testnet
```gherkin
GIVEN compiled icn-node binary exists
WHEN developer runs:
  ./target/release/icn-node --chain=local --alice --port 30333 --rpc-port 9944
  ./target/release/icn-node --chain=local --bob --port 30334 --rpc-port 9945 --bootnodes <alice-multiaddr>
THEN both nodes start successfully
  AND nodes discover each other (peer count > 0)
  AND blocks are produced via Aura consensus
  AND blocks are finalized via GRANDPA
```

### Scenario 4: Custom Pallet Directory Creation
```gherkin
GIVEN icn-chain repository root
WHEN directory structure created:
  pallets/
  ├── icn-stake/
  ├── icn-reputation/
  ├── icn-director/
  ├── icn-bft/
  ├── icn-pinning/
  └── icn-treasury/
THEN each directory contains: Cargo.toml, src/lib.rs, README.md
  AND Cargo.toml includes correct Polkadot SDK dependencies
```

### Scenario 5: Rust Toolchain Verification
```gherkin
GIVEN rust-toolchain.toml in repository root
WHEN developer runs: rustup show
THEN active toolchain matches: stable-x86_64-<platform> or nightly as required
  AND wasm32-unknown-unknown target is installed
  AND cargo --version shows compatible version
```

### Scenario 6: Runtime WASM Compilation
```gherkin
GIVEN successful native binary build
WHEN developer runs:
  cargo build --release --target wasm32-unknown-unknown -p icn-runtime
THEN WASM binary created at: target/wasm32-unknown-unknown/release/icn_runtime.wasm
  AND WASM binary size ~1-2MB (compressed)
  AND no clippy warnings for icn-runtime package
```

### Scenario 7: CI Pipeline Execution
```gherkin
GIVEN .github/workflows/icn-chain.yml workflow file
  AND workflow includes: build, test, clippy, fmt jobs
WHEN developer pushes to feature branch
THEN GitHub Actions triggers workflow
  AND all jobs complete successfully
  AND workflow duration <15 minutes
```

## Technical Implementation

### Step 1: Bootstrap from Polkadot SDK Template
```bash
# Clone the Polkadot SDK parachain template
# Note: Using polkadot-sdk-parachain-template or polkadot-sdk-solochain-template
git clone https://github.com/paritytech/polkadot-sdk-parachain-template.git icn-chain
cd icn-chain

# Or use substrate-node-template for pure solochain
# git clone https://github.com/substrate-developer-hub/substrate-node-template.git icn-chain

# Rename crates for ICN branding
# - parachain-template-node → icn-node
# - parachain-template-runtime → icn-runtime

# Update all Cargo.toml files with ICN naming
find . -name "Cargo.toml" -exec sed -i 's/parachain-template/icn/g' {} \;
find . -name "Cargo.toml" -exec sed -i 's/template-node/icn-node/g' {} \;
find . -name "Cargo.toml" -exec sed -i 's/template-runtime/icn-runtime/g' {} \;

# Verify Polkadot SDK version
grep 'polkadot-stable' Cargo.lock | head -1
# Expected: polkadot-stable2409 or later
```

### Step 2: Install Toolchain
```bash
# Install required toolchain (check rust-toolchain.toml in template)
rustup show

# Ensure WASM target is installed
rustup target add wasm32-unknown-unknown

# Install dev tools
rustup component add rustfmt clippy
```

### Step 3: Create Pallet Directory Structure
```bash
# Create ICN pallets workspace
mkdir -p pallets/{icn-stake,icn-reputation,icn-director,icn-bft,icn-pinning,icn-treasury}

# Template Cargo.toml for each pallet (using workspace dependencies)
for pallet in icn-stake icn-reputation icn-director icn-bft icn-pinning icn-treasury; do
cat > pallets/$pallet/Cargo.toml <<EOF
[package]
name = "pallet-$pallet"
version = "0.1.0"
edition = "2021"
license = "MIT"
repository = "https://github.com/your-org/icn-chain"

[dependencies]
codec = { package = "parity-scale-codec", version = "3.6", default-features = false, features = ["derive"] }
scale-info = { version = "2.10", default-features = false, features = ["derive"] }
frame-support = { workspace = true }
frame-system = { workspace = true }
sp-std = { workspace = true }
sp-runtime = { workspace = true }

[dev-dependencies]
sp-core = { workspace = true }
sp-io = { workspace = true }

[features]
default = ["std"]
std = [
    "codec/std",
    "scale-info/std",
    "frame-support/std",
    "frame-system/std",
    "sp-std/std",
    "sp-runtime/std",
]
runtime-benchmarks = [
    "frame-support/runtime-benchmarks",
    "frame-system/runtime-benchmarks",
]
try-runtime = [
    "frame-support/try-runtime",
    "frame-system/try-runtime",
]
EOF

# Template lib.rs skeleton
mkdir -p pallets/$pallet/src
cat > pallets/$pallet/src/lib.rs <<'EOF'
//! # ICN ${PALLET_NAME} Pallet
//!
//! Stub pallet for ICN Chain.

#![cfg_attr(not(feature = "std"), no_std)]

pub use pallet::*;

#[frame_support::pallet]
pub mod pallet {
    use frame_support::pallet_prelude::*;
    use frame_system::pallet_prelude::*;

    #[pallet::config]
    pub trait Config: frame_system::Config {
        /// The overarching event type.
        type RuntimeEvent: From<Event<Self>> + IsType<<Self as frame_system::Config>::RuntimeEvent>;
    }

    #[pallet::pallet]
    pub struct Pallet<T>(_);

    #[pallet::storage]
    #[pallet::getter(fn something)]
    pub type Something<T> = StorageValue<_, u32>;

    #[pallet::event]
    #[pallet::generate_deposit(pub(super) fn deposit_event)]
    pub enum Event<T: Config> {
        /// Event emitted when something is stored.
        SomethingStored { something: u32, who: T::AccountId },
    }

    #[pallet::error]
    pub enum Error<T> {
        /// No value was previously stored.
        NoneValue,
        /// Value exceeded maximum.
        StorageOverflow,
    }

    #[pallet::call]
    impl<T: Config> Pallet<T> {
        /// Example extrinsic (stub).
        #[pallet::call_index(0)]
        #[pallet::weight(Weight::from_parts(10_000, 0))]
        pub fn do_something(origin: OriginFor<T>, something: u32) -> DispatchResult {
            let who = ensure_signed(origin)?;
            Something::<T>::put(something);
            Self::deposit_event(Event::SomethingStored { something, who });
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    // Unit tests will be added in subsequent tasks
}
EOF

# Create README
cat > pallets/$pallet/README.md <<EOF
# pallet-$pallet

ICN ${pallet^^} pallet - see PRD v9.0 for specifications.

## Overview

Stub pallet to be implemented in task T00X.

## Build

\`\`\`bash
cargo build --release -p pallet-$pallet
cargo test -p pallet-$pallet
\`\`\`
EOF

done
```

### Step 4: Configure Workspace Cargo.toml
```toml
# Add to root Cargo.toml [workspace] members
[workspace]
members = [
    "node",
    "runtime",
    # ICN Custom Pallets
    "pallets/icn-stake",
    "pallets/icn-reputation",
    "pallets/icn-director",
    "pallets/icn-bft",
    "pallets/icn-pinning",
    "pallets/icn-treasury",
]

[workspace.dependencies]
# Ensure all pallets use consistent versions
frame-support = { git = "https://github.com/paritytech/polkadot-sdk", branch = "release-polkadot-stable2409", default-features = false }
frame-system = { git = "https://github.com/paritytech/polkadot-sdk", branch = "release-polkadot-stable2409", default-features = false }
sp-std = { git = "https://github.com/paritytech/polkadot-sdk", branch = "release-polkadot-stable2409", default-features = false }
sp-runtime = { git = "https://github.com/paritytech/polkadot-sdk", branch = "release-polkadot-stable2409", default-features = false }
sp-core = { git = "https://github.com/paritytech/polkadot-sdk", branch = "release-polkadot-stable2409" }
sp-io = { git = "https://github.com/paritytech/polkadot-sdk", branch = "release-polkadot-stable2409" }
```

### Step 5: Create CI/CD Workflow
```yaml
# .github/workflows/icn-chain.yml
name: ICN Chain CI

on:
  push:
    branches: [main, develop, feature/*]
  pull_request:
    branches: [main]

env:
  CARGO_TERM_COLOR: always

jobs:
  build-and-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Setup Rust
        uses: dtolnay/rust-toolchain@stable
        with:
          targets: wasm32-unknown-unknown
          components: rustfmt, clippy

      - name: Cache Cargo
        uses: actions/cache@v4
        with:
          path: |
            ~/.cargo/registry
            ~/.cargo/git
            target
          key: ${{ runner.os }}-cargo-${{ hashFiles('**/Cargo.lock') }}
          restore-keys: |
            ${{ runner.os }}-cargo-

      - name: Check formatting
        run: cargo fmt --all -- --check

      - name: Clippy
        run: cargo clippy --all-targets --all-features -- -D warnings

      - name: Build
        run: cargo build --release

      - name: Build WASM Runtime
        run: cargo build --release -p icn-runtime

      - name: Unit Tests
        run: cargo test --all-features

  pallet-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Setup Rust
        uses: dtolnay/rust-toolchain@stable
        with:
          targets: wasm32-unknown-unknown

      - name: Cache Cargo
        uses: actions/cache@v4
        with:
          path: |
            ~/.cargo/registry
            ~/.cargo/git
            target
          key: ${{ runner.os }}-cargo-pallets-${{ hashFiles('**/Cargo.lock') }}

      - name: Test ICN Pallets
        run: |
          cargo test -p pallet-icn-stake
          cargo test -p pallet-icn-reputation
          cargo test -p pallet-icn-director
          cargo test -p pallet-icn-bft
          cargo test -p pallet-icn-pinning
          cargo test -p pallet-icn-treasury
```

### Step 6: Configure Build Settings
```toml
# .cargo/config.toml
[build]
rustflags = ["-C", "link-arg=-fuse-ld=lld"]

[target.x86_64-unknown-linux-gnu]
linker = "clang"
rustflags = ["-C", "link-arg=-fuse-ld=lld"]

[net]
git-fetch-with-cli = true

[alias]
icn-node = "run --release -p icn-node --"
```

## Dependencies

None (this is the foundational task).

## Design Decisions

1. **Polkadot SDK Template over Moonbeam Fork**: Using official Polkadot SDK templates provides clean foundation without Moonbeam governance dependency. Full control over runtime upgrades.

2. **Solochain-First Approach**: Starting as solochain enables fast MVP iteration. Cumulus integration added in Phase C (T039) for optional parachain migration.

3. **Workspace Dependencies**: Using `workspace = true` in Cargo.toml ensures all pallets use consistent Polkadot SDK versions.

4. **Toolchain Management**: Using `rust-toolchain.toml` ensures all developers use identical compiler versions, preventing "works on my machine" issues.

5. **CI Early Setup**: Establishing CI from day 1 catches integration issues immediately rather than during final deployment.

## Risks & Mitigations

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Polkadot SDK version incompatibility | High | Low | Pin exact branch/tag, test on upgrade |
| Build failures on different OS | Medium | Medium | Test on Linux/macOS/Windows in CI matrix |
| WASM compilation issues | High | Low | Verify wasm32 target early, include in CI checks |
| Template structure changes | Medium | Low | Pin template version, document customizations |

## Progress Log

- 2025-12-24: Task rewritten for ICN Polkadot SDK Chain strategy (previously Moonbeam fork)

## Completion Checklist

- [ ] Polkadot SDK template cloned and customized for ICN
- [ ] All 10 acceptance criteria verified
- [ ] All 7 test scenarios pass
- [ ] Pallet directory structure created with stub implementations
- [ ] CI workflow passing
- [ ] Documentation updated (repository README)
- [ ] icn-node binary runs in --dev mode
- [ ] Multi-node local testnet works
- [ ] Code committed to feature branch

