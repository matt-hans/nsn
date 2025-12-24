# Task T001: Moonbeam Repository Fork and Development Environment Setup

## Metadata
```yaml
id: T001
title: Moonbeam Repository Fork and Development Environment Setup
status: pending
priority: P1
tags: [foundation, infrastructure, substrate, moonbeam]
estimated_tokens: 8000
actual_tokens: 0
dependencies: []
created_at: 2025-12-24
updated_at: 2025-12-24
```

## Description

Fork the Moonbeam repository, set up the local development environment, and configure the workspace for ICN custom pallet development. This establishes the foundation for all Substrate pallet work.

## Business Context

**Why this matters**: All ICN on-chain logic depends on integrating custom FRAME pallets into Moonbeam's runtime. Without a properly configured development environment matching Moonbeam's exact Substrate version (polkadot-v1.0.0), pallet compilation will fail and runtime upgrades will be incompatible.

**Value delivered**: Enables all subsequent pallet development tasks (T002-T007) by providing the correct build environment and dependency versions.

## Acceptance Criteria

1. Moonbeam repository forked to ICN organization/account
2. Local clone exists with all submodules initialized
3. Rust toolchain matches Moonbeam requirements (nightly-2024-01-01)
4. `wasm32-unknown-unknown` target installed
5. Repository builds successfully: `cargo build --release`
6. Local Moonriver test node runs: `./target/release/moonbeam --dev`
7. `pallets/` directory structure created for ICN custom pallets
8. `.cargo/config.toml` configured for optimal build settings
9. CI/CD workflow skeleton committed (`.github/workflows/pallets.yml`)
10. Development branch created: `feature/icn-pallets-integration`

## Test Scenarios

### Scenario 1: Fresh Repository Clone and Build
```gherkin
GIVEN a clean development machine with Rust installed
  AND Moonbeam repository URL: https://github.com/moonbeam-foundation/moonbeam.git
WHEN developer runs:
  git clone --recursive <forked-url> icn-moonbeam
  cd icn-moonbeam
  cargo build --release
THEN build completes without errors
  AND target/release/moonbeam binary exists
  AND binary size >50MB
```

### Scenario 2: Successful Local Test Node Launch
```gherkin
GIVEN compiled moonbeam binary exists
WHEN developer runs: ./target/release/moonbeam --dev --tmp
THEN node starts and produces blocks
  AND RPC server listens on ws://127.0.0.1:9944
  AND node logs show "Idle (0 peers)" initially
  AND Ctrl+C cleanly shuts down node
```

### Scenario 3: Custom Pallet Directory Creation
```gherkin
GIVEN icn-moonbeam repository root
WHEN directory structure created:
  pallets/
  ├── icn-stake/
  ├── icn-reputation/
  ├── icn-director/
  ├── icn-bft/
  ├── icn-pinning/
  └── icn-treasury/
THEN each directory contains: Cargo.toml, src/lib.rs, README.md
  AND Cargo.toml includes correct FRAME dependencies
```

### Scenario 4: Rust Toolchain Verification
```gherkin
GIVEN rust-toolchain.toml in repository root
WHEN developer runs: rustup show
THEN active toolchain matches: nightly-2024-01-01-x86_64-<platform>
  AND wasm32-unknown-unknown target is installed
  AND cargo --version shows version matching toolchain
```

### Scenario 5: Runtime WASM Compilation
```gherkin
GIVEN successful native binary build
WHEN developer runs:
  cargo build --release --target wasm32-unknown-unknown -p moonbeam-runtime
THEN WASM binary created at: target/wasm32-unknown-unknown/release/moonbeam_runtime.wasm
  AND WASM binary size ~1-2MB (compressed)
  AND no clippy warnings for moonbeam-runtime package
```

### Scenario 6: CI Pipeline Skeleton Execution
```gherkin
GIVEN .github/workflows/pallets.yml workflow file
  AND workflow includes: build, test, clippy, fmt jobs
WHEN developer pushes to feature branch
THEN GitHub Actions triggers workflow
  AND all jobs complete successfully
  AND workflow duration <15 minutes
```

## Technical Implementation

### Step 1: Repository Setup
```bash
# Fork on GitHub UI, then clone
git clone --recursive https://github.com/<org>/moonbeam.git icn-moonbeam
cd icn-moonbeam

# Verify Substrate version
grep 'polkadot-v' Cargo.lock | head -1
# Expected: polkadot-v1.0.0

# Install correct toolchain
rustup install nightly-2024-01-01
rustup target add wasm32-unknown-unknown --toolchain nightly-2024-01-01
rustup component add rustfmt clippy --toolchain nightly-2024-01-01

# Set project toolchain
echo 'nightly-2024-01-01' > rust-toolchain.toml
```

### Step 2: Create Pallet Directory Structure
```bash
# Create ICN pallets workspace
mkdir -p pallets/{icn-stake,icn-reputation,icn-director,icn-bft,icn-pinning,icn-treasury}

# Template Cargo.toml for each pallet
for pallet in icn-stake icn-reputation icn-director icn-bft icn-pinning icn-treasury; do
cat > pallets/$pallet/Cargo.toml <<EOF
[package]
name = "pallet-$pallet"
version = "0.1.0"
edition = "2021"

[dependencies]
codec = { package = "parity-scale-codec", version = "3.0.0", default-features = false }
scale-info = { version = "2.0.0", default-features = false, features = ["derive"] }
frame-support = { git = "https://github.com/paritytech/substrate", branch = "polkadot-v1.0.0", default-features = false }
frame-system = { git = "https://github.com/paritytech/substrate", branch = "polkadot-v1.0.0", default-features = false }
sp-std = { git = "https://github.com/paritytech/substrate", branch = "polkadot-v1.0.0", default-features = false }
sp-runtime = { git = "https://github.com/paritytech/substrate", branch = "polkadot-v1.0.0", default-features = false }

[dev-dependencies]
sp-core = { git = "https://github.com/paritytech/substrate", branch = "polkadot-v1.0.0" }
sp-io = { git = "https://github.com/paritytech/substrate", branch = "polkadot-v1.0.0" }

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
EOF

# Template lib.rs skeleton
mkdir -p pallets/$pallet/src
cat > pallets/$pallet/src/lib.rs <<'EOF'
#![cfg_attr(not(feature = "std"), no_std)]

pub use pallet::*;

#[frame_support::pallet]
pub mod pallet {
    use frame_support::pallet_prelude::*;
    use frame_system::pallet_prelude::*;

    #[pallet::config]
    pub trait Config: frame_system::Config {
        type RuntimeEvent: From<Event<Self>> + IsType<<Self as frame_system::Config>::RuntimeEvent>;
    }

    #[pallet::pallet]
    pub struct Pallet<T>(_);

    #[pallet::storage]
    pub type Something<T> = StorageValue<_, u32>;

    #[pallet::event]
    #[pallet::generate_deposit(pub(super) fn deposit_event)]
    pub enum Event<T: Config> {
        SomethingStored { something: u32, who: T::AccountId },
    }

    #[pallet::error]
    pub enum Error<T> {
        NoneValue,
        StorageOverflow,
    }

    #[pallet::call]
    impl<T: Config> Pallet<T> {
        #[pallet::weight(10_000)]
        pub fn do_something(origin: OriginFor<T>, something: u32) -> DispatchResult {
            let who = ensure_signed(origin)?;
            Something::<T>::put(something);
            Self::deposit_event(Event::SomethingStored { something, who });
            Ok(())
        }
    }
}
EOF
done
```

### Step 3: Configure Workspace Cargo.toml
```toml
# Add to root Cargo.toml [workspace] members
[workspace]
members = [
    # ... existing members ...
    "pallets/icn-stake",
    "pallets/icn-reputation",
    "pallets/icn-director",
    "pallets/icn-bft",
    "pallets/icn-pinning",
    "pallets/icn-treasury",
]
```

### Step 4: Create CI/CD Workflow
```yaml
# .github/workflows/pallets.yml
name: ICN Pallets CI

on:
  push:
    branches: [main, develop, feature/icn-pallets-integration]
  pull_request:
    branches: [main]

jobs:
  build-and-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          submodules: recursive

      - name: Setup Rust
        uses: dtolnay/rust-toolchain@stable
        with:
          toolchain: nightly-2024-01-01
          targets: wasm32-unknown-unknown
          components: rustfmt, clippy

      - name: Cache Cargo
        uses: actions/cache@v3
        with:
          path: |
            ~/.cargo/registry
            ~/.cargo/git
            target
          key: ${{ runner.os }}-cargo-${{ hashFiles('**/Cargo.lock') }}

      - name: Build Pallets
        run: |
          cargo build --release --all-features -p pallet-icn-stake
          cargo build --release --all-features -p pallet-icn-reputation

      - name: Unit Tests
        run: cargo test --all-features -p pallet-icn-stake -p pallet-icn-reputation

      - name: Clippy
        run: cargo clippy --all-features -- -D warnings

      - name: Format Check
        run: cargo fmt -- --check
```

## Dependencies

None (this is the foundational task).

## Design Decisions

1. **Moonbeam fork vs fresh Substrate**: Forking Moonbeam provides EVM compatibility, existing collator infrastructure, and governance mechanisms. Fresh Substrate would require 9-18 months to build equivalent features.

2. **Workspace structure**: Placing ICN pallets in `pallets/` keeps them separate from Moonbeam core pallets, simplifying future maintenance and governance proposals.

3. **Toolchain pinning**: Using `rust-toolchain.toml` ensures all developers use identical compiler versions, preventing "works on my machine" issues.

4. **CI early setup**: Establishing CI from day 1 catches integration issues immediately rather than during final deployment.

## Risks & Mitigations

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Moonbeam version drift | High | Medium | Pin exact commit hash in fork, monitor Moonbeam releases |
| Build failures on different OS | Medium | Medium | Test on Linux/macOS/Windows in CI matrix |
| Submodule initialization failures | Low | Medium | Use `--recursive` flag consistently, document in README |
| WASM compilation issues | High | Low | Verify wasm32 target early, include in CI checks |

## Progress Log

- 2025-12-24: Task created from PRD Phase 1 requirements

## Completion Checklist

- [ ] Repository forked and cloned
- [ ] All 10 acceptance criteria verified
- [ ] All 6 test scenarios pass
- [ ] Pallet directory structure created
- [ ] CI workflow passing
- [ ] Documentation updated (repository README)
- [ ] Code committed to feature branch
- [ ] No regressions in Moonbeam core functionality
