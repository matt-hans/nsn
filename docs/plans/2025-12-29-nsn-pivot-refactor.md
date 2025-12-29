# Neural Sovereign Network (NSN) Pivot & Refactor Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Transform ICN (video-only network) into NSN (hybrid video + general AI compute marketplace) with dual-lane architecture preserving the 45-second video latency guarantee.

**Architecture:** Two operational lanes: Lane 0 (Prime Directive) for time-triggered video generation with pinned VRAM, Lane 1 (Sovereign Market) for demand-triggered general AI compute with dynamic model swapping. Epoch-based director elections (1-hour shifts) with On-Deck protocol for graceful lane transitions. Off-chain task discovery via P2P, on-chain settlement. Rust sidecar supervises Python containers.

**Tech Stack:** Polkadot SDK (FRAME pallets), Rust (Tokio, libp2p, tonic), Python (PyTorch, Vortex), gRPC, Podman/Docker, NVML

---

## Phase 1: Foundation (Rename & Restructure)

### Task 1.1: Global Rename ICN → NSN (Pallets)

**Files:**
- Modify: `icn-chain/pallets/icn-stake/Cargo.toml`
- Modify: `icn-chain/pallets/icn-stake/src/lib.rs`
- Modify: `icn-chain/pallets/icn-reputation/Cargo.toml`
- Modify: `icn-chain/pallets/icn-reputation/src/lib.rs`
- Modify: `icn-chain/pallets/icn-director/Cargo.toml`
- Modify: `icn-chain/pallets/icn-director/src/lib.rs`
- Modify: `icn-chain/pallets/icn-bft/Cargo.toml`
- Modify: `icn-chain/pallets/icn-bft/src/lib.rs`
- Modify: `icn-chain/pallets/icn-pinning/Cargo.toml`
- Modify: `icn-chain/pallets/icn-pinning/src/lib.rs`
- Modify: `icn-chain/pallets/icn-treasury/Cargo.toml`
- Modify: `icn-chain/pallets/icn-treasury/src/lib.rs`

**Step 1: Rename pallet directories**

```bash
cd /Users/matthewhans/Desktop/Programming/interdim-cable/icn-chain/pallets
mv icn-stake nsn-stake
mv icn-reputation nsn-reputation
mv icn-director nsn-director
mv icn-bft nsn-bft
mv icn-pinning nsn-storage
mv icn-treasury nsn-treasury
```

**Step 2: Update Cargo.toml names in each pallet**

For each pallet, update the `[package]` name. Example for nsn-stake:

```toml
[package]
name = "pallet-nsn-stake"
version = "0.1.0"
edition = "2021"
```

**Step 3: Global find/replace in source files**

```bash
# In icn-chain directory
find . -name "*.rs" -exec sed -i '' 's/icn_stake/nsn_stake/g' {} \;
find . -name "*.rs" -exec sed -i '' 's/icn_reputation/nsn_reputation/g' {} \;
find . -name "*.rs" -exec sed -i '' 's/icn_director/nsn_director/g' {} \;
find . -name "*.rs" -exec sed -i '' 's/icn_bft/nsn_bft/g' {} \;
find . -name "*.rs" -exec sed -i '' 's/icn_pinning/nsn_storage/g' {} \;
find . -name "*.rs" -exec sed -i '' 's/icn_treasury/nsn_treasury/g' {} \;
find . -name "*.rs" -exec sed -i '' 's/pallet-icn/pallet-nsn/g' {} \;
find . -name "*.toml" -exec sed -i '' 's/pallet-icn/pallet-nsn/g' {} \;
```

**Step 4: Update runtime Cargo.toml dependencies**

Modify `icn-chain/runtime/Cargo.toml`:

```toml
[dependencies]
pallet-nsn-stake = { path = "../pallets/nsn-stake", default-features = false }
pallet-nsn-reputation = { path = "../pallets/nsn-reputation", default-features = false }
pallet-nsn-director = { path = "../pallets/nsn-director", default-features = false }
pallet-nsn-bft = { path = "../pallets/nsn-bft", default-features = false }
pallet-nsn-storage = { path = "../pallets/nsn-storage", default-features = false }
pallet-nsn-treasury = { path = "../pallets/nsn-treasury", default-features = false }
```

**Step 5: Verify compilation**

```bash
cd /Users/matthewhans/Desktop/Programming/interdim-cable/icn-chain
cargo check --workspace
```

Expected: Compilation succeeds with warnings only

**Step 6: Commit**

```bash
git add -A
git commit -m "refactor: rename ICN pallets to NSN namespace"
```

---

### Task 1.2: Rename Chain Directory & Node Binary

**Files:**
- Rename: `icn-chain/` → `nsn-chain/`
- Modify: `nsn-chain/node/Cargo.toml`
- Modify: `nsn-chain/node/src/main.rs`

**Step 1: Rename chain directory**

```bash
cd /Users/matthewhans/Desktop/Programming/interdim-cable
mv icn-chain nsn-chain
```

**Step 2: Update node binary name**

Modify `nsn-chain/node/Cargo.toml`:

```toml
[[bin]]
name = "nsn-node"
path = "src/main.rs"
```

**Step 3: Update references in main.rs**

```rust
// nsn-chain/node/src/main.rs
fn main() -> sc_cli::Result<()> {
    command::run()
}
```

**Step 4: Update workspace Cargo.toml**

Modify `nsn-chain/Cargo.toml`:

```toml
[workspace]
members = [
    "node",
    "runtime",
    "pallets/nsn-stake",
    "pallets/nsn-reputation",
    "pallets/nsn-director",
    "pallets/nsn-bft",
    "pallets/nsn-storage",
    "pallets/nsn-treasury",
]
```

**Step 5: Verify build**

```bash
cd /Users/matthewhans/Desktop/Programming/interdim-cable/nsn-chain
cargo build --release -p nsn-node
```

Expected: Binary compiles successfully

**Step 6: Commit**

```bash
git add -A
git commit -m "refactor: rename icn-chain to nsn-chain, node binary to nsn-node"
```

---

### Task 1.3: Create Node-Core Workspace Structure

**Files:**
- Create: `node-core/Cargo.toml`
- Create: `node-core/bin/nsn-node/Cargo.toml`
- Create: `node-core/bin/nsn-node/src/main.rs`
- Create: `node-core/crates/types/Cargo.toml`
- Create: `node-core/crates/types/src/lib.rs`

**Step 1: Create workspace structure**

```bash
cd /Users/matthewhans/Desktop/Programming/interdim-cable
mkdir -p node-core/bin/nsn-node/src
mkdir -p node-core/crates/types/src
mkdir -p node-core/crates/chain-client/src
mkdir -p node-core/crates/p2p/src
mkdir -p node-core/crates/scheduler/src
mkdir -p node-core/crates/lane0/src
mkdir -p node-core/crates/lane1/src
mkdir -p node-core/crates/storage/src
mkdir -p node-core/crates/validator/src
mkdir -p node-core/sidecar/src
mkdir -p node-core/sidecar/proto
```

**Step 2: Create workspace Cargo.toml**

Create `node-core/Cargo.toml`:

```toml
[workspace]
resolver = "2"
members = [
    "bin/nsn-node",
    "crates/types",
    "crates/chain-client",
    "crates/p2p",
    "crates/scheduler",
    "crates/lane0",
    "crates/lane1",
    "crates/storage",
    "crates/validator",
    "sidecar",
]

[workspace.package]
version = "0.1.0"
edition = "2021"
license = "Apache-2.0"
repository = "https://github.com/interdim-cable/nsn"

[workspace.dependencies]
# Async runtime
tokio = { version = "1.35", features = ["full"] }
futures = "0.3"

# Serialization
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
prost = "0.12"
tonic = "0.10"

# Substrate client
subxt = "0.34"

# P2P
libp2p = { version = "0.53", features = ["tokio", "gossipsub", "kad", "identify", "noise", "tcp", "quic", "yamux", "request-response"] }

# Logging
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["env-filter"] }

# Error handling
thiserror = "1.0"
anyhow = "1.0"

# Config
config = "0.14"
clap = { version = "4.4", features = ["derive"] }

# Internal crates
nsn-types = { path = "crates/types" }
nsn-chain-client = { path = "crates/chain-client" }
nsn-p2p = { path = "crates/p2p" }
nsn-scheduler = { path = "crates/scheduler" }
nsn-lane0 = { path = "crates/lane0" }
nsn-lane1 = { path = "crates/lane1" }
nsn-storage = { path = "crates/storage" }
nsn-validator = { path = "crates/validator" }
nsn-sidecar = { path = "sidecar" }
```

**Step 3: Create types crate**

Create `node-core/crates/types/Cargo.toml`:

```toml
[package]
name = "nsn-types"
version.workspace = true
edition.workspace = true

[dependencies]
serde = { workspace = true }
codec = { package = "parity-scale-codec", version = "3.6", features = ["derive"] }
```

Create `node-core/crates/types/src/lib.rs`:

```rust
//! Shared types for the NSN node-core workspace.

use codec::{Decode, Encode};
use serde::{Deserialize, Serialize};

/// Unique identifier for a model in the registry
pub type ModelId = String;

/// Unique identifier for a task
pub type TaskId = u64;

/// Unique identifier for an epoch
pub type EpochId = u64;

/// Container identifier (Docker/Podman)
pub type ContainerId = String;

/// IPFS Content Identifier
pub type Cid = String;

/// Node operational mode
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Encode, Decode)]
pub enum NodeMode {
    /// Full mode: Lane 0 + Lane 1 capable (GPU required)
    Full,
    /// Compute only: Lane 1 only (GPU required, no video)
    Compute,
    /// Storage tier: No GPU, erasure coding and relay
    Storage,
    /// Validator only: CPU sufficient, CLIP verification
    Validator,
}

/// Node state in the scheduling state machine
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum NodeState {
    /// Accepting and executing Lane 1 tasks
    Lane1Active,
    /// Elected for upcoming epoch, finishing current tasks
    Draining {
        epoch_start_block: u64,
        hard_stop_deadline_ms: u64,
    },
    /// Active video director for current epoch
    Lane0Active {
        epoch_id: EpochId,
        epoch_end_block: u64,
    },
    /// Storage-only mode (no GPU)
    StorageOnly,
    /// Maintenance mode (refuses all work)
    Maintenance,
    /// Shutting down
    Offline,
}

/// Container priority for VRAM allocation
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ContainerPriority {
    /// Normal Lane 1 task
    Lane1Normal = 0,
    /// High priority Lane 1 task
    Lane1High = 1,
    /// Lane 0 video generation (cannot be preempted)
    Lane0Video = 2,
}

/// Model loading state for capability advertisement
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelState {
    /// Model loaded in VRAM, ready for immediate inference
    Hot,
    /// Model cached on disk, ~5-10s startup
    Warm,
    /// Model not present, must download (~minutes)
    Cold,
    /// Node does not support this model
    Unsupported,
}

/// Task status in the lifecycle
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TaskStatus {
    /// Task created, waiting for assignment
    Open,
    /// Task assigned to a node
    Assigned {
        executor: String,
        assigned_at_block: u64,
    },
    /// Task completed successfully
    Completed {
        output_cid: Cid,
        completed_at_block: u64,
    },
    /// Task failed
    Failed {
        reason: FailureReason,
    },
    /// Task expired (deadline passed)
    Expired,
}

/// Reasons for task failure
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum FailureReason {
    /// Executor timeout
    Timeout,
    /// Preempted for Lane 0 video generation
    Preempted,
    /// Executor reported error
    ExecutionError(String),
    /// Invalid input
    InvalidInput,
}

/// VRAM allocation tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VramAllocation {
    pub container_id: ContainerId,
    pub model_id: ModelId,
    pub allocated_mb: u32,
    pub priority: ContainerPriority,
    pub allocated_at_ms: u64,
}

/// Epoch information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpochInfo {
    pub id: EpochId,
    pub start_block: u64,
    pub end_block: u64,
    pub directors: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_node_state_transitions() {
        let state = NodeState::Lane1Active;
        assert_eq!(state, NodeState::Lane1Active);
    }

    #[test]
    fn test_container_priority_ordering() {
        assert!(ContainerPriority::Lane0Video > ContainerPriority::Lane1High);
        assert!(ContainerPriority::Lane1High > ContainerPriority::Lane1Normal);
    }
}
```

**Step 4: Create minimal main binary**

Create `node-core/bin/nsn-node/Cargo.toml`:

```toml
[package]
name = "nsn-node-bin"
version.workspace = true
edition.workspace = true

[[bin]]
name = "nsn-node"
path = "src/main.rs"

[dependencies]
nsn-types = { workspace = true }
tokio = { workspace = true }
tracing = { workspace = true }
tracing-subscriber = { workspace = true }
clap = { workspace = true }
anyhow = { workspace = true }
```

Create `node-core/bin/nsn-node/src/main.rs`:

```rust
//! NSN Node - Unified dual-mode node for the Neural Sovereign Network.

use clap::Parser;
use nsn_types::NodeMode;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

#[derive(Parser, Debug)]
#[command(name = "nsn-node")]
#[command(about = "Neural Sovereign Network Node")]
struct Cli {
    /// Node operational mode
    #[arg(long, value_enum, default_value = "full")]
    mode: CliNodeMode,

    /// Chain RPC endpoint
    #[arg(long, default_value = "ws://127.0.0.1:9944")]
    chain_rpc: String,

    /// P2P listen address
    #[arg(long, default_value = "/ip4/0.0.0.0/tcp/30333")]
    p2p_listen: String,

    /// Sidecar gRPC address
    #[arg(long, default_value = "http://127.0.0.1:50051")]
    sidecar_addr: String,
}

#[derive(clap::ValueEnum, Clone, Debug)]
enum CliNodeMode {
    /// Lane 0 + Lane 1 capable (GPU required)
    Full,
    /// Lane 1 only (GPU required, no video)
    Compute,
    /// Storage tier (no GPU)
    Storage,
    /// Validator only (CPU sufficient)
    Validator,
}

impl From<CliNodeMode> for NodeMode {
    fn from(mode: CliNodeMode) -> Self {
        match mode {
            CliNodeMode::Full => NodeMode::Full,
            CliNodeMode::Compute => NodeMode::Compute,
            CliNodeMode::Storage => NodeMode::Storage,
            CliNodeMode::Validator => NodeMode::Validator,
        }
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize tracing
    tracing_subscriber::registry()
        .with(tracing_subscriber::EnvFilter::new(
            std::env::var("RUST_LOG").unwrap_or_else(|_| "info".into()),
        ))
        .with(tracing_subscriber::fmt::layer())
        .init();

    let cli = Cli::parse();
    let mode: NodeMode = cli.mode.into();

    tracing::info!("Starting NSN Node in {:?} mode", mode);
    tracing::info!("Chain RPC: {}", cli.chain_rpc);
    tracing::info!("P2P Listen: {}", cli.p2p_listen);
    tracing::info!("Sidecar: {}", cli.sidecar_addr);

    // TODO: Initialize components based on mode
    // - chain-client
    // - p2p
    // - scheduler
    // - lane0 (if Full mode)
    // - lane1 (if Full or Compute mode)
    // - storage (if Storage mode)
    // - validator (if Validator mode)

    tracing::info!("NSN Node started successfully");

    // Keep running
    tokio::signal::ctrl_c().await?;
    tracing::info!("Shutting down...");

    Ok(())
}
```

**Step 5: Verify workspace compiles**

```bash
cd /Users/matthewhans/Desktop/Programming/interdim-cable/node-core
cargo check --workspace
```

Expected: Compilation succeeds

**Step 6: Commit**

```bash
git add -A
git commit -m "feat: create node-core workspace structure with types crate and main binary"
```

---

### Task 1.4: Deprecate Relay Node (Merge into Storage)

**Files:**
- Remove: `icn-nodes/relay/` (after extracting useful code)
- Modify: `icn-nodes/super-node/` → migrate to `node-core/crates/storage/`

**Step 1: Remove relay directory**

```bash
cd /Users/matthewhans/Desktop/Programming/interdim-cable
rm -rf icn-nodes/relay
```

**Step 2: Rename icn-nodes to legacy-nodes (preserve for reference)**

```bash
mv icn-nodes legacy-nodes
```

**Step 3: Commit**

```bash
git add -A
git commit -m "refactor: deprecate relay node, rename icn-nodes to legacy-nodes for reference"
```

---

## Phase 2: Pallet Layer - Core Extensions

### Task 2.1: Add NodeMode to pallet-nsn-stake

**Files:**
- Modify: `nsn-chain/pallets/nsn-stake/src/types.rs`
- Modify: `nsn-chain/pallets/nsn-stake/src/lib.rs`
- Test: `nsn-chain/pallets/nsn-stake/src/tests.rs`

**Step 1: Write failing test for NodeMode**

Add to `nsn-chain/pallets/nsn-stake/src/tests.rs`:

```rust
#[test]
fn test_node_mode_transitions() {
    new_test_ext().execute_with(|| {
        let account = 1u64;

        // Setup: stake as director
        assert_ok!(NsnStake::deposit_stake(
            RuntimeOrigin::signed(account),
            1000,
            100,
            Region::NorthAmerica,
        ));

        // Initially should be Lane1Active (default for staked nodes)
        assert_eq!(
            NsnStake::node_modes(account),
            NodeMode::Lane1Active
        );

        // Transition to Draining (called by director pallet during election)
        assert_ok!(NsnStake::set_node_mode(
            RuntimeOrigin::root(),
            account,
            NodeMode::Draining { epoch_start: 100 }
        ));

        assert_eq!(
            NsnStake::node_modes(account),
            NodeMode::Draining { epoch_start: 100 }
        );
    });
}
```

**Step 2: Run test to verify it fails**

```bash
cd /Users/matthewhans/Desktop/Programming/interdim-cable/nsn-chain
cargo test -p pallet-nsn-stake test_node_mode_transitions
```

Expected: FAIL - `NodeMode` type not found

**Step 3: Add NodeMode type**

Add to `nsn-chain/pallets/nsn-stake/src/types.rs`:

```rust
/// Node operational mode for dual-lane architecture
#[derive(
    Clone, Encode, Decode, Eq, PartialEq, RuntimeDebug, TypeInfo, MaxEncodedLen, Default,
)]
pub enum NodeMode {
    /// Ready for Lane 1 tasks, not elected for video
    #[default]
    Lane1Active,
    /// Elected for upcoming epoch, draining Lane 1 tasks
    Draining {
        epoch_start: BlockNumber,
    },
    /// Active video director for current epoch
    Lane0Active {
        epoch_end: BlockNumber,
    },
    /// Offline or maintenance
    Offline,
}

/// Block number type (re-export for convenience)
pub type BlockNumber = u32;
```

**Step 4: Add NodeModes storage and extrinsic**

Add to `nsn-chain/pallets/nsn-stake/src/lib.rs`:

```rust
/// Node mode tracking
#[pallet::storage]
pub type NodeModes<T: Config> = StorageMap<
    _,
    Blake2_128Concat,
    T::AccountId,
    NodeMode,
    ValueQuery,
>;

#[pallet::call]
impl<T: Config> Pallet<T> {
    /// Set node mode (root only, called by director pallet)
    #[pallet::call_index(10)]
    #[pallet::weight(T::WeightInfo::set_node_mode())]
    pub fn set_node_mode(
        origin: OriginFor<T>,
        account: T::AccountId,
        mode: NodeMode,
    ) -> DispatchResult {
        ensure_root(origin)?;

        // Verify account has stake
        ensure!(Stakes::<T>::contains_key(&account), Error::<T>::NotStaked);

        NodeModes::<T>::insert(&account, mode.clone());

        Self::deposit_event(Event::NodeModeChanged {
            account,
            new_mode: mode
        });

        Ok(())
    }
}

#[pallet::event]
#[pallet::generate_deposit(pub(super) fn deposit_event)]
pub enum Event<T: Config> {
    // ... existing events ...

    /// Node operational mode changed
    NodeModeChanged {
        account: T::AccountId,
        new_mode: NodeMode,
    },
}
```

**Step 5: Run test to verify it passes**

```bash
cd /Users/matthewhans/Desktop/Programming/interdim-cable/nsn-chain
cargo test -p pallet-nsn-stake test_node_mode_transitions
```

Expected: PASS

**Step 6: Commit**

```bash
git add -A
git commit -m "feat(pallet-nsn-stake): add NodeMode state machine for dual-lane architecture"
```

---

### Task 2.2: Implement Epoch-Based Elections in pallet-nsn-director

**Files:**
- Modify: `nsn-chain/pallets/nsn-director/src/types.rs`
- Modify: `nsn-chain/pallets/nsn-director/src/lib.rs`
- Test: `nsn-chain/pallets/nsn-director/src/tests.rs`

**Step 1: Write failing test for epoch-based election**

Add to `nsn-chain/pallets/nsn-director/src/tests.rs`:

```rust
#[test]
fn test_epoch_based_election() {
    new_test_ext().execute_with(|| {
        // Setup: Multiple staked directors
        for i in 1..=10u64 {
            setup_director(i, 1000);
        }

        // Get current epoch
        let epoch = NsnDirector::current_epoch();
        assert_eq!(epoch.id, 0);

        // Run to the lookahead block (epoch_end - LOOKAHEAD)
        let lookahead_block = epoch.end_block.saturating_sub(EpochLookahead::get());
        run_to_block(lookahead_block);

        // Should have elected next epoch directors
        let next_directors = NsnDirector::next_epoch_directors();
        assert_eq!(next_directors.len(), 5); // 5 directors per epoch

        // Event should be emitted
        System::assert_has_event(Event::OnDeckElection {
            epoch_id: 1,
            directors: next_directors.clone(),
            start_block: epoch.end_block,
        }.into());
    });
}
```

**Step 2: Run test to verify it fails**

```bash
cd /Users/matthewhans/Desktop/Programming/interdim-cable/nsn-chain
cargo test -p pallet-nsn-director test_epoch_based_election
```

Expected: FAIL - `current_epoch`, `next_epoch_directors` not found

**Step 3: Add epoch types**

Add to `nsn-chain/pallets/nsn-director/src/types.rs`:

```rust
/// Epoch identifier
pub type EpochId = u64;

/// Epoch information
#[derive(Clone, Encode, Decode, Eq, PartialEq, RuntimeDebug, TypeInfo, MaxEncodedLen)]
pub struct Epoch<BlockNumber, AccountId, MaxDirectors: Get<u32>> {
    pub id: EpochId,
    pub start_block: BlockNumber,
    pub end_block: BlockNumber,
    pub directors: BoundedVec<AccountId, MaxDirectors>,
    pub status: EpochStatus,
}

/// Epoch status
#[derive(Clone, Encode, Decode, Eq, PartialEq, RuntimeDebug, TypeInfo, MaxEncodedLen, Default)]
pub enum EpochStatus {
    /// Election complete, waiting to start
    #[default]
    Scheduled,
    /// Currently generating video
    Active,
    /// Finished, pending finalization
    Completed,
}

impl<BlockNumber: Default, AccountId, MaxDirectors: Get<u32>> Default
    for Epoch<BlockNumber, AccountId, MaxDirectors>
{
    fn default() -> Self {
        Self {
            id: 0,
            start_block: BlockNumber::default(),
            end_block: BlockNumber::default(),
            directors: BoundedVec::default(),
            status: EpochStatus::default(),
        }
    }
}
```

**Step 4: Add epoch storage and election logic**

Add to `nsn-chain/pallets/nsn-director/src/lib.rs`:

```rust
#[pallet::config]
pub trait Config: frame_system::Config + pallet_nsn_stake::Config {
    // ... existing config ...

    /// Epoch duration in blocks (1 hour at 6s blocks = 600 blocks)
    #[pallet::constant]
    type EpochDuration: Get<BlockNumberFor<Self>>;

    /// Lookahead blocks for On-Deck notification (2 min = 20 blocks)
    #[pallet::constant]
    type EpochLookahead: Get<BlockNumberFor<Self>>;

    /// Maximum directors per epoch
    #[pallet::constant]
    type MaxDirectorsPerEpoch: Get<u32>;
}

/// Current epoch information
#[pallet::storage]
pub type CurrentEpoch<T: Config> = StorageValue<
    _,
    Epoch<BlockNumberFor<T>, T::AccountId, T::MaxDirectorsPerEpoch>,
    ValueQuery,
>;

/// Directors elected for next epoch (lookahead)
#[pallet::storage]
pub type NextEpochDirectors<T: Config> = StorageValue<
    _,
    BoundedVec<T::AccountId, T::MaxDirectorsPerEpoch>,
    ValueQuery,
>;

#[pallet::hooks]
impl<T: Config> Hooks<BlockNumberFor<T>> for Pallet<T> {
    fn on_initialize(now: BlockNumberFor<T>) -> Weight {
        let mut weight = Weight::zero();

        let current_epoch = CurrentEpoch::<T>::get();
        let lookahead = T::EpochLookahead::get();

        // Check 1: Is it time to elect next epoch's directors?
        let election_trigger = current_epoch.end_block.saturating_sub(lookahead);
        if now == election_trigger && NextEpochDirectors::<T>::get().is_empty() {
            weight = weight.saturating_add(Self::run_epoch_election(now));
        }

        // Check 2: Is current epoch ending?
        if now >= current_epoch.end_block {
            weight = weight.saturating_add(Self::transition_epoch(now));
        }

        weight
    }
}

impl<T: Config> Pallet<T> {
    /// Run election for next epoch
    fn run_epoch_election(now: BlockNumberFor<T>) -> Weight {
        let current_epoch = CurrentEpoch::<T>::get();

        // Get eligible candidates (simplified - should use VRF in production)
        let candidates = Self::get_eligible_candidates();

        // Select top N by reputation (bounded by MaxDirectorsPerEpoch)
        let max_directors = T::MaxDirectorsPerEpoch::get() as usize;
        let elected: BoundedVec<T::AccountId, T::MaxDirectorsPerEpoch> = candidates
            .into_iter()
            .take(max_directors)
            .collect::<Vec<_>>()
            .try_into()
            .unwrap_or_default();

        // Store elected directors
        NextEpochDirectors::<T>::put(elected.clone());

        // Emit On-Deck event
        Self::deposit_event(Event::OnDeckElection {
            epoch_id: current_epoch.id + 1,
            directors: elected.into_inner(),
            start_block: current_epoch.end_block,
        });

        // Update node modes to Draining via stake pallet
        // (Implementation depends on cross-pallet coupling)

        T::DbWeight::get().reads_writes(10, 5)
    }

    /// Transition to next epoch
    fn transition_epoch(now: BlockNumberFor<T>) -> Weight {
        let current_epoch = CurrentEpoch::<T>::get();
        let next_directors = NextEpochDirectors::<T>::take();

        let new_epoch = Epoch {
            id: current_epoch.id + 1,
            start_block: now,
            end_block: now + T::EpochDuration::get(),
            directors: next_directors,
            status: EpochStatus::Active,
        };

        CurrentEpoch::<T>::put(new_epoch.clone());

        Self::deposit_event(Event::EpochStarted {
            epoch_id: new_epoch.id,
            directors: new_epoch.directors.into_inner(),
            end_block: new_epoch.end_block,
        });

        T::DbWeight::get().reads_writes(2, 2)
    }

    /// Get eligible director candidates (simplified)
    fn get_eligible_candidates() -> Vec<T::AccountId> {
        // In production: query stake pallet, filter by role, sort by reputation
        // For now: return empty (will be implemented with cross-pallet integration)
        Vec::new()
    }
}

#[pallet::event]
#[pallet::generate_deposit(pub(super) fn deposit_event)]
pub enum Event<T: Config> {
    /// Directors elected for upcoming epoch (On-Deck notification)
    OnDeckElection {
        epoch_id: EpochId,
        directors: Vec<T::AccountId>,
        start_block: BlockNumberFor<T>,
    },
    /// New epoch started
    EpochStarted {
        epoch_id: EpochId,
        directors: Vec<T::AccountId>,
        end_block: BlockNumberFor<T>,
    },
    /// Epoch ended
    EpochEnded {
        epoch_id: EpochId,
    },
}
```

**Step 5: Run test to verify it passes**

```bash
cd /Users/matthewhans/Desktop/Programming/interdim-cable/nsn-chain
cargo test -p pallet-nsn-director test_epoch_based_election
```

Expected: PASS (or partial pass - may need mock adjustments)

**Step 6: Commit**

```bash
git add -A
git commit -m "feat(pallet-nsn-director): implement epoch-based elections with On-Deck lookahead"
```

---

### Task 2.3: Create pallet-task-market

**Files:**
- Create: `nsn-chain/pallets/task-market/Cargo.toml`
- Create: `nsn-chain/pallets/task-market/src/lib.rs`
- Create: `nsn-chain/pallets/task-market/src/types.rs`
- Create: `nsn-chain/pallets/task-market/src/tests.rs`
- Create: `nsn-chain/pallets/task-market/src/mock.rs`

**Step 1: Create pallet directory structure**

```bash
cd /Users/matthewhans/Desktop/Programming/interdim-cable/nsn-chain/pallets
mkdir -p task-market/src
```

**Step 2: Create Cargo.toml**

Create `nsn-chain/pallets/task-market/Cargo.toml`:

```toml
[package]
name = "pallet-task-market"
version = "0.1.0"
edition = "2021"
license = "Apache-2.0"
description = "Lane 1 task marketplace for the Neural Sovereign Network"

[package.metadata.docs.rs]
targets = ["x86_64-unknown-linux-gnu"]

[dependencies]
codec = { package = "parity-scale-codec", version = "3.6", default-features = false, features = ["derive"] }
scale-info = { version = "2.10", default-features = false, features = ["derive"] }
frame-support = { git = "https://github.com/paritytech/polkadot-sdk", tag = "polkadot-stable2409", default-features = false }
frame-system = { git = "https://github.com/paritytech/polkadot-sdk", tag = "polkadot-stable2409", default-features = false }
sp-runtime = { git = "https://github.com/paritytech/polkadot-sdk", tag = "polkadot-stable2409", default-features = false }
sp-std = { git = "https://github.com/paritytech/polkadot-sdk", tag = "polkadot-stable2409", default-features = false }

# Local dependencies
pallet-nsn-stake = { path = "../nsn-stake", default-features = false }

[dev-dependencies]
sp-core = { git = "https://github.com/paritytech/polkadot-sdk", tag = "polkadot-stable2409" }
sp-io = { git = "https://github.com/paritytech/polkadot-sdk", tag = "polkadot-stable2409" }
pallet-balances = { git = "https://github.com/paritytech/polkadot-sdk", tag = "polkadot-stable2409" }

[features]
default = ["std"]
std = [
    "codec/std",
    "scale-info/std",
    "frame-support/std",
    "frame-system/std",
    "sp-runtime/std",
    "sp-std/std",
    "pallet-nsn-stake/std",
]
runtime-benchmarks = [
    "frame-support/runtime-benchmarks",
    "frame-system/runtime-benchmarks",
]
try-runtime = [
    "frame-support/try-runtime",
    "frame-system/try-runtime",
]
```

**Step 3: Create types.rs**

Create `nsn-chain/pallets/task-market/src/types.rs`:

```rust
//! Types for the task market pallet.

use codec::{Decode, Encode, MaxEncodedLen};
use frame_support::BoundedVec;
use scale_info::TypeInfo;
use sp_runtime::RuntimeDebug;

/// Task identifier
pub type TaskId = u64;

/// Model identifier (bounded string)
pub type ModelId<MaxLen> = BoundedVec<u8, MaxLen>;

/// IPFS CID (bounded string)
pub type Cid<MaxLen> = BoundedVec<u8, MaxLen>;

/// Task in the marketplace
#[derive(Clone, Encode, Decode, Eq, PartialEq, RuntimeDebug, TypeInfo, MaxEncodedLen)]
#[scale_info(skip_type_params(MaxModelIdLen, MaxCidLen))]
pub struct Task<AccountId, Balance, BlockNumber, MaxModelIdLen, MaxCidLen>
where
    MaxModelIdLen: frame_support::traits::Get<u32>,
    MaxCidLen: frame_support::traits::Get<u32>,
{
    pub id: TaskId,
    pub requester: AccountId,
    pub model_id: ModelId<MaxModelIdLen>,
    pub input_cid: Cid<MaxCidLen>,
    pub max_compute_units: u32,
    pub deadline: BlockNumber,
    pub escrowed: Balance,
    pub status: TaskStatus<AccountId, BlockNumber, MaxCidLen>,
    pub created_at: BlockNumber,
}

/// Task status in the lifecycle
#[derive(Clone, Encode, Decode, Eq, PartialEq, RuntimeDebug, TypeInfo, MaxEncodedLen, Default)]
#[scale_info(skip_type_params(MaxCidLen))]
pub enum TaskStatus<AccountId, BlockNumber, MaxCidLen>
where
    MaxCidLen: frame_support::traits::Get<u32>,
{
    /// Task created, waiting for assignment
    #[default]
    Open,
    /// Task assigned to a node
    Assigned {
        executor: AccountId,
        assigned_at: BlockNumber,
    },
    /// Task completed successfully
    Completed {
        output_cid: Cid<MaxCidLen>,
        completed_at: BlockNumber,
    },
    /// Task failed
    Failed {
        reason: FailureReason,
    },
    /// Task expired (deadline passed)
    Expired,
}

/// Reasons for task failure
#[derive(Clone, Encode, Decode, Eq, PartialEq, RuntimeDebug, TypeInfo, MaxEncodedLen)]
pub enum FailureReason {
    /// Executor timeout
    Timeout,
    /// Preempted for Lane 0 video generation
    Preempted,
    /// Executor reported error
    ExecutionError,
    /// Invalid input
    InvalidInput,
}
```

**Step 4: Create main pallet lib.rs**

Create `nsn-chain/pallets/task-market/src/lib.rs`:

```rust
//! # Task Market Pallet
//!
//! Lane 1 task marketplace for the Neural Sovereign Network.
//! Handles task submission, assignment, completion, and payment settlement.

#![cfg_attr(not(feature = "std"), no_std)]

pub use pallet::*;

mod types;
pub use types::*;

#[cfg(test)]
mod mock;

#[cfg(test)]
mod tests;

#[frame_support::pallet]
pub mod pallet {
    use super::*;
    use frame_support::{
        pallet_prelude::*,
        traits::{Currency, ReservableCurrency},
    };
    use frame_system::pallet_prelude::*;
    use sp_runtime::traits::Zero;

    pub type BalanceOf<T> =
        <<T as Config>::Currency as Currency<<T as frame_system::Config>::AccountId>>::Balance;

    #[pallet::pallet]
    pub struct Pallet<T>(_);

    #[pallet::config]
    pub trait Config: frame_system::Config {
        /// The overarching event type.
        type RuntimeEvent: From<Event<Self>> + IsType<<Self as frame_system::Config>::RuntimeEvent>;

        /// Currency for payments and escrow.
        type Currency: ReservableCurrency<Self::AccountId>;

        /// Maximum length for model IDs.
        #[pallet::constant]
        type MaxModelIdLen: Get<u32>;

        /// Maximum length for CIDs.
        #[pallet::constant]
        type MaxCidLen: Get<u32>;

        /// Maximum pending tasks.
        #[pallet::constant]
        type MaxPendingTasks: Get<u32>;

        /// Minimum escrow amount.
        #[pallet::constant]
        type MinEscrow: Get<BalanceOf<Self>>;
    }

    /// Task storage
    #[pallet::storage]
    pub type Tasks<T: Config> = StorageMap<
        _,
        Blake2_128Concat,
        TaskId,
        Task<
            T::AccountId,
            BalanceOf<T>,
            BlockNumberFor<T>,
            T::MaxModelIdLen,
            T::MaxCidLen,
        >,
        OptionQuery,
    >;

    /// Next task ID
    #[pallet::storage]
    pub type NextTaskId<T: Config> = StorageValue<_, TaskId, ValueQuery>;

    /// Open tasks queue
    #[pallet::storage]
    pub type OpenTasks<T: Config> = StorageValue<
        _,
        BoundedVec<TaskId, T::MaxPendingTasks>,
        ValueQuery,
    >;

    #[pallet::event]
    #[pallet::generate_deposit(pub(super) fn deposit_event)]
    pub enum Event<T: Config> {
        /// Task created and escrowed
        TaskCreated {
            task_id: TaskId,
            requester: T::AccountId,
            model_id: BoundedVec<u8, T::MaxModelIdLen>,
            escrowed: BalanceOf<T>,
        },
        /// Task assigned to executor
        TaskAssigned {
            task_id: TaskId,
            executor: T::AccountId,
        },
        /// Task completed successfully
        TaskCompleted {
            task_id: TaskId,
            executor: T::AccountId,
            output_cid: BoundedVec<u8, T::MaxCidLen>,
        },
        /// Task failed
        TaskFailed {
            task_id: TaskId,
            reason: FailureReason,
        },
    }

    #[pallet::error]
    pub enum Error<T> {
        /// Task not found
        TaskNotFound,
        /// Task not in expected status
        InvalidTaskStatus,
        /// Insufficient escrow
        InsufficientEscrow,
        /// Not authorized
        NotAuthorized,
        /// Too many pending tasks
        TooManyPendingTasks,
        /// Task already assigned
        TaskAlreadyAssigned,
        /// Invalid signature
        InvalidSignature,
    }

    #[pallet::call]
    impl<T: Config> Pallet<T> {
        /// Create a task intent (locks funds in escrow)
        #[pallet::call_index(0)]
        #[pallet::weight(Weight::from_parts(10_000, 0))]
        pub fn create_task_intent(
            origin: OriginFor<T>,
            model_id: BoundedVec<u8, T::MaxModelIdLen>,
            input_cid: BoundedVec<u8, T::MaxCidLen>,
            max_compute_units: u32,
            deadline_blocks: BlockNumberFor<T>,
            escrow_amount: BalanceOf<T>,
        ) -> DispatchResult {
            let requester = ensure_signed(origin)?;

            // Verify minimum escrow
            ensure!(
                escrow_amount >= T::MinEscrow::get(),
                Error::<T>::InsufficientEscrow
            );

            // Reserve escrow
            T::Currency::reserve(&requester, escrow_amount)?;

            // Create task
            let task_id = NextTaskId::<T>::get();
            NextTaskId::<T>::put(task_id + 1);

            let now = frame_system::Pallet::<T>::block_number();
            let deadline = now + deadline_blocks;

            let task = Task {
                id: task_id,
                requester: requester.clone(),
                model_id: model_id.clone(),
                input_cid,
                max_compute_units,
                deadline,
                escrowed: escrow_amount,
                status: TaskStatus::Open,
                created_at: now,
            };

            Tasks::<T>::insert(task_id, task);

            // Add to open tasks queue
            OpenTasks::<T>::try_mutate(|tasks| {
                tasks.try_push(task_id).map_err(|_| Error::<T>::TooManyPendingTasks)
            })?;

            Self::deposit_event(Event::TaskCreated {
                task_id,
                requester,
                model_id,
                escrowed: escrow_amount,
            });

            Ok(())
        }

        /// Accept a task assignment (with requester's signature)
        #[pallet::call_index(1)]
        #[pallet::weight(Weight::from_parts(10_000, 0))]
        pub fn accept_assignment(
            origin: OriginFor<T>,
            task_id: TaskId,
            // In production: include requester signature verification
            // For MVP: simplified direct assignment
        ) -> DispatchResult {
            let executor = ensure_signed(origin)?;

            Tasks::<T>::try_mutate(task_id, |maybe_task| {
                let task = maybe_task.as_mut().ok_or(Error::<T>::TaskNotFound)?;

                ensure!(
                    matches!(task.status, TaskStatus::Open),
                    Error::<T>::TaskAlreadyAssigned
                );

                let now = frame_system::Pallet::<T>::block_number();

                task.status = TaskStatus::Assigned {
                    executor: executor.clone(),
                    assigned_at: now,
                };

                // Remove from open tasks
                OpenTasks::<T>::try_mutate(|tasks| {
                    if let Some(pos) = tasks.iter().position(|&id| id == task_id) {
                        tasks.remove(pos);
                    }
                    Ok::<(), Error<T>>(())
                })?;

                Self::deposit_event(Event::TaskAssigned {
                    task_id,
                    executor,
                });

                Ok(())
            })
        }

        /// Complete a task with output
        #[pallet::call_index(2)]
        #[pallet::weight(Weight::from_parts(10_000, 0))]
        pub fn complete_task(
            origin: OriginFor<T>,
            task_id: TaskId,
            output_cid: BoundedVec<u8, T::MaxCidLen>,
        ) -> DispatchResult {
            let executor = ensure_signed(origin)?;

            Tasks::<T>::try_mutate(task_id, |maybe_task| {
                let task = maybe_task.as_mut().ok_or(Error::<T>::TaskNotFound)?;

                // Verify executor
                match &task.status {
                    TaskStatus::Assigned { executor: assigned, .. } => {
                        ensure!(assigned == &executor, Error::<T>::NotAuthorized);
                    }
                    _ => return Err(Error::<T>::InvalidTaskStatus.into()),
                }

                let now = frame_system::Pallet::<T>::block_number();

                // Release escrow to executor
                T::Currency::unreserve(&task.requester, task.escrowed);
                T::Currency::transfer(
                    &task.requester,
                    &executor,
                    task.escrowed,
                    frame_support::traits::ExistenceRequirement::KeepAlive,
                )?;

                task.status = TaskStatus::Completed {
                    output_cid: output_cid.clone(),
                    completed_at: now,
                };

                Self::deposit_event(Event::TaskCompleted {
                    task_id,
                    executor,
                    output_cid,
                });

                Ok(())
            })
        }

        /// Report task failure
        #[pallet::call_index(3)]
        #[pallet::weight(Weight::from_parts(10_000, 0))]
        pub fn fail_task(
            origin: OriginFor<T>,
            task_id: TaskId,
            reason: FailureReason,
        ) -> DispatchResult {
            let caller = ensure_signed(origin)?;

            Tasks::<T>::try_mutate(task_id, |maybe_task| {
                let task = maybe_task.as_mut().ok_or(Error::<T>::TaskNotFound)?;

                // Verify caller is executor or requester
                let is_authorized = match &task.status {
                    TaskStatus::Assigned { executor, .. } => {
                        &caller == executor || caller == task.requester
                    }
                    TaskStatus::Open => caller == task.requester,
                    _ => false,
                };
                ensure!(is_authorized, Error::<T>::NotAuthorized);

                // Refund escrow to requester
                T::Currency::unreserve(&task.requester, task.escrowed);

                task.status = TaskStatus::Failed { reason: reason.clone() };

                Self::deposit_event(Event::TaskFailed { task_id, reason });

                Ok(())
            })
        }
    }
}
```

**Step 5: Create mock.rs**

Create `nsn-chain/pallets/task-market/src/mock.rs`:

```rust
//! Mock runtime for testing task-market pallet.

use crate as pallet_task_market;
use frame_support::{
    parameter_types,
    traits::{ConstU32, ConstU64},
};
use sp_core::H256;
use sp_runtime::{
    traits::{BlakeTwo256, IdentityLookup},
    BuildStorage,
};

type Block = frame_system::mocking::MockBlock<Test>;

frame_support::construct_runtime!(
    pub enum Test {
        System: frame_system,
        Balances: pallet_balances,
        TaskMarket: pallet_task_market,
    }
);

impl frame_system::Config for Test {
    type BaseCallFilter = frame_support::traits::Everything;
    type BlockWeights = ();
    type BlockLength = ();
    type DbWeight = ();
    type RuntimeOrigin = RuntimeOrigin;
    type RuntimeCall = RuntimeCall;
    type Nonce = u64;
    type Hash = H256;
    type Hashing = BlakeTwo256;
    type AccountId = u64;
    type Lookup = IdentityLookup<Self::AccountId>;
    type Block = Block;
    type RuntimeEvent = RuntimeEvent;
    type BlockHashCount = ConstU64<250>;
    type Version = ();
    type PalletInfo = PalletInfo;
    type AccountData = pallet_balances::AccountData<u64>;
    type OnNewAccount = ();
    type OnKilledAccount = ();
    type SystemWeightInfo = ();
    type SS58Prefix = ();
    type OnSetCode = ();
    type MaxConsumers = ConstU32<16>;
    type RuntimeTask = ();
    type SingleBlockMigrations = ();
    type MultiBlockMigrator = ();
    type PreInherents = ();
    type PostInherents = ();
    type PostTransactions = ();
}

impl pallet_balances::Config for Test {
    type MaxLocks = ConstU32<50>;
    type MaxReserves = ConstU32<50>;
    type ReserveIdentifier = [u8; 8];
    type Balance = u64;
    type RuntimeEvent = RuntimeEvent;
    type DustRemoval = ();
    type ExistentialDeposit = ConstU64<1>;
    type AccountStore = System;
    type WeightInfo = ();
    type FreezeIdentifier = ();
    type MaxFreezes = ConstU32<0>;
    type RuntimeHoldReason = ();
    type RuntimeFreezeReason = ();
}

parameter_types! {
    pub const MinEscrow: u64 = 10;
}

impl pallet_task_market::Config for Test {
    type RuntimeEvent = RuntimeEvent;
    type Currency = Balances;
    type MaxModelIdLen = ConstU32<64>;
    type MaxCidLen = ConstU32<64>;
    type MaxPendingTasks = ConstU32<1000>;
    type MinEscrow = MinEscrow;
}

pub fn new_test_ext() -> sp_io::TestExternalities {
    let mut t = frame_system::GenesisConfig::<Test>::default()
        .build_storage()
        .unwrap();

    pallet_balances::GenesisConfig::<Test> {
        balances: vec![(1, 10000), (2, 10000), (3, 10000)],
    }
    .assimilate_storage(&mut t)
    .unwrap();

    t.into()
}
```

**Step 6: Create tests.rs**

Create `nsn-chain/pallets/task-market/src/tests.rs`:

```rust
//! Tests for task-market pallet.

use crate::{mock::*, Error, Event, FailureReason, TaskStatus};
use frame_support::{assert_noop, assert_ok, BoundedVec};

#[test]
fn test_create_task_intent() {
    new_test_ext().execute_with(|| {
        let requester = 1u64;
        let model_id: BoundedVec<u8, _> = b"llama-3-70b".to_vec().try_into().unwrap();
        let input_cid: BoundedVec<u8, _> = b"QmTest123".to_vec().try_into().unwrap();

        assert_ok!(TaskMarket::create_task_intent(
            RuntimeOrigin::signed(requester),
            model_id.clone(),
            input_cid,
            100,  // compute units
            100,  // deadline blocks
            100,  // escrow
        ));

        // Verify task created
        let task = TaskMarket::tasks(0).expect("Task should exist");
        assert_eq!(task.requester, requester);
        assert!(matches!(task.status, TaskStatus::Open));

        // Verify escrow reserved
        assert_eq!(Balances::reserved_balance(requester), 100);

        // Verify event
        System::assert_has_event(Event::TaskCreated {
            task_id: 0,
            requester,
            model_id,
            escrowed: 100,
        }.into());
    });
}

#[test]
fn test_accept_assignment() {
    new_test_ext().execute_with(|| {
        let requester = 1u64;
        let executor = 2u64;

        // Create task
        let model_id: BoundedVec<u8, _> = b"llama-3-70b".to_vec().try_into().unwrap();
        let input_cid: BoundedVec<u8, _> = b"QmTest123".to_vec().try_into().unwrap();

        assert_ok!(TaskMarket::create_task_intent(
            RuntimeOrigin::signed(requester),
            model_id,
            input_cid,
            100,
            100,
            100,
        ));

        // Accept assignment
        assert_ok!(TaskMarket::accept_assignment(
            RuntimeOrigin::signed(executor),
            0,
        ));

        // Verify status changed
        let task = TaskMarket::tasks(0).expect("Task should exist");
        assert!(matches!(task.status, TaskStatus::Assigned { .. }));
    });
}

#[test]
fn test_complete_task() {
    new_test_ext().execute_with(|| {
        let requester = 1u64;
        let executor = 2u64;

        // Create and assign task
        let model_id: BoundedVec<u8, _> = b"llama-3-70b".to_vec().try_into().unwrap();
        let input_cid: BoundedVec<u8, _> = b"QmTest123".to_vec().try_into().unwrap();

        assert_ok!(TaskMarket::create_task_intent(
            RuntimeOrigin::signed(requester),
            model_id,
            input_cid,
            100,
            100,
            100,
        ));

        assert_ok!(TaskMarket::accept_assignment(
            RuntimeOrigin::signed(executor),
            0,
        ));

        // Complete task
        let output_cid: BoundedVec<u8, _> = b"QmOutput456".to_vec().try_into().unwrap();

        let executor_balance_before = Balances::free_balance(executor);

        assert_ok!(TaskMarket::complete_task(
            RuntimeOrigin::signed(executor),
            0,
            output_cid.clone(),
        ));

        // Verify payment transferred
        assert_eq!(Balances::free_balance(executor), executor_balance_before + 100);
        assert_eq!(Balances::reserved_balance(requester), 0);
    });
}

#[test]
fn test_fail_task_preemption() {
    new_test_ext().execute_with(|| {
        let requester = 1u64;
        let executor = 2u64;

        // Create and assign task
        let model_id: BoundedVec<u8, _> = b"llama-3-70b".to_vec().try_into().unwrap();
        let input_cid: BoundedVec<u8, _> = b"QmTest123".to_vec().try_into().unwrap();

        assert_ok!(TaskMarket::create_task_intent(
            RuntimeOrigin::signed(requester),
            model_id,
            input_cid,
            100,
            100,
            100,
        ));

        assert_ok!(TaskMarket::accept_assignment(
            RuntimeOrigin::signed(executor),
            0,
        ));

        let requester_balance_before = Balances::free_balance(requester);

        // Fail task due to preemption
        assert_ok!(TaskMarket::fail_task(
            RuntimeOrigin::signed(executor),
            0,
            FailureReason::Preempted,
        ));

        // Verify refund to requester
        assert_eq!(Balances::free_balance(requester), requester_balance_before + 100);

        // Verify event
        System::assert_has_event(Event::TaskFailed {
            task_id: 0,
            reason: FailureReason::Preempted,
        }.into());
    });
}

#[test]
fn test_insufficient_escrow() {
    new_test_ext().execute_with(|| {
        let model_id: BoundedVec<u8, _> = b"llama-3-70b".to_vec().try_into().unwrap();
        let input_cid: BoundedVec<u8, _> = b"QmTest123".to_vec().try_into().unwrap();

        assert_noop!(
            TaskMarket::create_task_intent(
                RuntimeOrigin::signed(1),
                model_id,
                input_cid,
                100,
                100,
                5,  // Below MinEscrow of 10
            ),
            Error::<Test>::InsufficientEscrow
        );
    });
}
```

**Step 7: Add to workspace**

Update `nsn-chain/Cargo.toml`:

```toml
members = [
    "node",
    "runtime",
    "pallets/nsn-stake",
    "pallets/nsn-reputation",
    "pallets/nsn-director",
    "pallets/nsn-bft",
    "pallets/nsn-storage",
    "pallets/nsn-treasury",
    "pallets/task-market",  # NEW
]
```

**Step 8: Run tests**

```bash
cd /Users/matthewhans/Desktop/Programming/interdim-cable/nsn-chain
cargo test -p pallet-task-market
```

Expected: All tests pass

**Step 9: Commit**

```bash
git add -A
git commit -m "feat: add pallet-task-market for Lane 1 compute marketplace"
```

---

### Task 2.4: Create pallet-model-registry

**Files:**
- Create: `nsn-chain/pallets/model-registry/Cargo.toml`
- Create: `nsn-chain/pallets/model-registry/src/lib.rs`
- Create: `nsn-chain/pallets/model-registry/src/types.rs`
- Create: `nsn-chain/pallets/model-registry/src/tests.rs`
- Create: `nsn-chain/pallets/model-registry/src/mock.rs`

**Step 1: Create pallet directory**

```bash
cd /Users/matthewhans/Desktop/Programming/interdim-cable/nsn-chain/pallets
mkdir -p model-registry/src
```

**Step 2: Create Cargo.toml**

Create `nsn-chain/pallets/model-registry/Cargo.toml`:

```toml
[package]
name = "pallet-model-registry"
version = "0.1.0"
edition = "2021"
license = "Apache-2.0"
description = "Model catalog and node capability registry for NSN"

[dependencies]
codec = { package = "parity-scale-codec", version = "3.6", default-features = false, features = ["derive"] }
scale-info = { version = "2.10", default-features = false, features = ["derive"] }
frame-support = { git = "https://github.com/paritytech/polkadot-sdk", tag = "polkadot-stable2409", default-features = false }
frame-system = { git = "https://github.com/paritytech/polkadot-sdk", tag = "polkadot-stable2409", default-features = false }
sp-runtime = { git = "https://github.com/paritytech/polkadot-sdk", tag = "polkadot-stable2409", default-features = false }
sp-std = { git = "https://github.com/paritytech/polkadot-sdk", tag = "polkadot-stable2409", default-features = false }

[dev-dependencies]
sp-core = { git = "https://github.com/paritytech/polkadot-sdk", tag = "polkadot-stable2409" }
sp-io = { git = "https://github.com/paritytech/polkadot-sdk", tag = "polkadot-stable2409" }

[features]
default = ["std"]
std = [
    "codec/std",
    "scale-info/std",
    "frame-support/std",
    "frame-system/std",
    "sp-runtime/std",
    "sp-std/std",
]
```

**Step 3: Create types.rs**

Create `nsn-chain/pallets/model-registry/src/types.rs`:

```rust
//! Types for model registry pallet.

use codec::{Decode, Encode, MaxEncodedLen};
use frame_support::BoundedVec;
use scale_info::TypeInfo;
use sp_runtime::RuntimeDebug;

/// Model identifier
pub type ModelId<MaxLen> = BoundedVec<u8, MaxLen>;

/// IPFS CID for container image
pub type ContainerCid<MaxLen> = BoundedVec<u8, MaxLen>;

/// Model capabilities bitfield
#[derive(Clone, Encode, Decode, Eq, PartialEq, RuntimeDebug, TypeInfo, MaxEncodedLen, Default)]
pub struct ModelCapabilities {
    pub text_generation: bool,
    pub image_generation: bool,
    pub code_generation: bool,
    pub embedding: bool,
    pub speech_synthesis: bool,
    pub video_generation: bool,
}

/// Model metadata in the catalog
#[derive(Clone, Encode, Decode, Eq, PartialEq, RuntimeDebug, TypeInfo, MaxEncodedLen)]
#[scale_info(skip_type_params(MaxCidLen))]
pub struct ModelMetadata<AccountId, BlockNumber, MaxCidLen>
where
    MaxCidLen: frame_support::traits::Get<u32>,
{
    pub container_cid: ContainerCid<MaxCidLen>,
    pub vram_required_mb: u32,
    pub capabilities: ModelCapabilities,
    pub registered_by: AccountId,
    pub registered_at: BlockNumber,
}

/// Model loading state
#[derive(Clone, Encode, Decode, Eq, PartialEq, RuntimeDebug, TypeInfo, MaxEncodedLen, Default)]
pub enum ModelState {
    /// Model loaded in VRAM
    Hot,
    /// Model cached on disk
    Warm,
    /// Model not present
    #[default]
    Cold,
}

/// Node capability advertisement
#[derive(Clone, Encode, Decode, Eq, PartialEq, RuntimeDebug, TypeInfo, MaxEncodedLen)]
#[scale_info(skip_type_params(MaxModelIdLen, MaxHotModels, MaxWarmModels))]
pub struct NodeCapabilityAd<BlockNumber, MaxModelIdLen, MaxHotModels, MaxWarmModels>
where
    MaxModelIdLen: frame_support::traits::Get<u32>,
    MaxHotModels: frame_support::traits::Get<u32>,
    MaxWarmModels: frame_support::traits::Get<u32>,
{
    pub available_vram_mb: u32,
    pub hot_models: BoundedVec<ModelId<MaxModelIdLen>, MaxHotModels>,
    pub warm_models: BoundedVec<ModelId<MaxModelIdLen>, MaxWarmModels>,
    pub last_updated: BlockNumber,
}
```

**Step 4: Create main pallet lib.rs**

Create `nsn-chain/pallets/model-registry/src/lib.rs`:

```rust
//! # Model Registry Pallet
//!
//! Maintains the model catalog and node capability advertisements.

#![cfg_attr(not(feature = "std"), no_std)]

pub use pallet::*;

mod types;
pub use types::*;

#[cfg(test)]
mod mock;

#[cfg(test)]
mod tests;

#[frame_support::pallet]
pub mod pallet {
    use super::*;
    use frame_support::pallet_prelude::*;
    use frame_system::pallet_prelude::*;

    #[pallet::pallet]
    pub struct Pallet<T>(_);

    #[pallet::config]
    pub trait Config: frame_system::Config {
        type RuntimeEvent: From<Event<Self>> + IsType<<Self as frame_system::Config>::RuntimeEvent>;

        #[pallet::constant]
        type MaxModelIdLen: Get<u32>;

        #[pallet::constant]
        type MaxCidLen: Get<u32>;

        #[pallet::constant]
        type MaxHotModels: Get<u32>;

        #[pallet::constant]
        type MaxWarmModels: Get<u32>;
    }

    /// Model catalog
    #[pallet::storage]
    pub type ModelCatalog<T: Config> = StorageMap<
        _,
        Blake2_128Concat,
        ModelId<T::MaxModelIdLen>,
        ModelMetadata<T::AccountId, BlockNumberFor<T>, T::MaxCidLen>,
        OptionQuery,
    >;

    /// Node capability advertisements
    #[pallet::storage]
    pub type NodeCapabilities<T: Config> = StorageMap<
        _,
        Blake2_128Concat,
        T::AccountId,
        NodeCapabilityAd<BlockNumberFor<T>, T::MaxModelIdLen, T::MaxHotModels, T::MaxWarmModels>,
        OptionQuery,
    >;

    #[pallet::event]
    #[pallet::generate_deposit(pub(super) fn deposit_event)]
    pub enum Event<T: Config> {
        /// Model registered in catalog
        ModelRegistered {
            model_id: BoundedVec<u8, T::MaxModelIdLen>,
            container_cid: BoundedVec<u8, T::MaxCidLen>,
            vram_required_mb: u32,
        },
        /// Node capability updated
        NodeCapabilityUpdated {
            node: T::AccountId,
            available_vram_mb: u32,
            hot_model_count: u32,
            warm_model_count: u32,
        },
    }

    #[pallet::error]
    pub enum Error<T> {
        /// Model already registered
        ModelAlreadyRegistered,
        /// Model not found
        ModelNotFound,
        /// Invalid container CID
        InvalidContainerCid,
    }

    #[pallet::call]
    impl<T: Config> Pallet<T> {
        /// Register a new model in the catalog
        #[pallet::call_index(0)]
        #[pallet::weight(Weight::from_parts(10_000, 0))]
        pub fn register_model(
            origin: OriginFor<T>,
            model_id: BoundedVec<u8, T::MaxModelIdLen>,
            container_cid: BoundedVec<u8, T::MaxCidLen>,
            vram_required_mb: u32,
            capabilities: ModelCapabilities,
        ) -> DispatchResult {
            let who = ensure_signed(origin)?;

            ensure!(
                !ModelCatalog::<T>::contains_key(&model_id),
                Error::<T>::ModelAlreadyRegistered
            );

            let metadata = ModelMetadata {
                container_cid: container_cid.clone(),
                vram_required_mb,
                capabilities,
                registered_by: who,
                registered_at: frame_system::Pallet::<T>::block_number(),
            };

            ModelCatalog::<T>::insert(&model_id, metadata);

            Self::deposit_event(Event::ModelRegistered {
                model_id,
                container_cid,
                vram_required_mb,
            });

            Ok(())
        }

        /// Update node capability advertisement
        #[pallet::call_index(1)]
        #[pallet::weight(Weight::from_parts(10_000, 0))]
        pub fn update_capabilities(
            origin: OriginFor<T>,
            available_vram_mb: u32,
            hot_models: BoundedVec<BoundedVec<u8, T::MaxModelIdLen>, T::MaxHotModels>,
            warm_models: BoundedVec<BoundedVec<u8, T::MaxModelIdLen>, T::MaxWarmModels>,
        ) -> DispatchResult {
            let who = ensure_signed(origin)?;

            let ad = NodeCapabilityAd {
                available_vram_mb,
                hot_models,
                warm_models: warm_models.clone(),
                last_updated: frame_system::Pallet::<T>::block_number(),
            };

            let hot_count = ad.hot_models.len() as u32;
            let warm_count = ad.warm_models.len() as u32;

            NodeCapabilities::<T>::insert(&who, ad);

            Self::deposit_event(Event::NodeCapabilityUpdated {
                node: who,
                available_vram_mb,
                hot_model_count: hot_count,
                warm_model_count: warm_count,
            });

            Ok(())
        }
    }
}
```

**Step 5: Create mock.rs and tests.rs**

(Similar structure to task-market - abbreviated for space)

**Step 6: Add to workspace and verify**

```bash
# Update nsn-chain/Cargo.toml to include model-registry
cd /Users/matthewhans/Desktop/Programming/interdim-cable/nsn-chain
cargo test -p pallet-model-registry
```

**Step 7: Commit**

```bash
git add -A
git commit -m "feat: add pallet-model-registry for model catalog and node capability ads"
```

---

## Phase 3: Node Core - Scheduler & Sidecar

### Task 3.1: Implement Scheduler State Machine

**Files:**
- Create: `node-core/crates/scheduler/Cargo.toml`
- Create: `node-core/crates/scheduler/src/lib.rs`
- Create: `node-core/crates/scheduler/src/state_machine.rs`
- Test: `node-core/crates/scheduler/src/tests.rs`

(Detailed implementation following the architecture from brainstorming session)

### Task 3.2: Implement Sidecar gRPC Service

**Files:**
- Create: `node-core/sidecar/Cargo.toml`
- Create: `node-core/sidecar/proto/sidecar.proto`
- Create: `node-core/sidecar/src/main.rs`
- Create: `node-core/sidecar/src/container/lifecycle.rs`
- Create: `node-core/sidecar/src/vram/budget.rs`

(Detailed implementation following the architecture from brainstorming session)

### Task 3.3: Implement VRAM Budget Manager

**Files:**
- Create: `node-core/sidecar/src/vram/budget.rs`
- Create: `node-core/sidecar/src/vram/nvidia.rs`

(Detailed implementation with NVML integration)

### Task 3.4: Implement Preemption Logic

**Files:**
- Create: `node-core/sidecar/src/container/preempt.rs`

(Detailed implementation with Double-Tap strategy)

---

## Phase 4: Integration & Testing

### Task 4.1: Wire Up Runtime with New Pallets

**Files:**
- Modify: `nsn-chain/runtime/src/lib.rs`

### Task 4.2: Integration Tests

**Files:**
- Create: `nsn-chain/tests/integration/task_lifecycle.rs`
- Create: `nsn-chain/tests/integration/epoch_transition.rs`

### Task 4.3: Update Documentation

**Files:**
- Modify: `CLAUDE.md`
- Modify: `.claude/rules/architecture.md`
- Modify: `.claude/rules/prd.md`

---

## Summary: Implementation Order

| Phase | Tasks | Estimated Time |
|-------|-------|----------------|
| **Phase 1: Foundation** | 1.1-1.4 | 1-2 days |
| **Phase 2: Pallets** | 2.1-2.4 | 3-4 days |
| **Phase 3: Node Core** | 3.1-3.4 | 4-5 days |
| **Phase 4: Integration** | 4.1-4.3 | 2-3 days |
| **Total** | | ~2 weeks |

---

## Validation Checklist

After completing all tasks, verify:

- [ ] `cargo check --workspace` passes in `nsn-chain/`
- [ ] `cargo check --workspace` passes in `node-core/`
- [ ] All pallet tests pass: `cargo test -p pallet-nsn-*`
- [ ] All node-core tests pass: `cargo test --workspace` in `node-core/`
- [ ] Node binary starts: `./target/release/nsn-node --mode full --help`
- [ ] Sidecar starts: `./target/release/nsn-sidecar --help`
- [ ] Documentation updated with NSN terminology
