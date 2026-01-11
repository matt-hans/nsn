# NSN Simulation Crate

Deterministic network simulation harness for multi-node E2E testing of NSN's distributed components.

## Features

- **In-memory network simulation** - No real sockets or ports required
- **Deterministic timing** - Uses `tokio::time::pause()` for reproducible tests
- **Byzantine fault injection** - Configure nodes to exhibit Byzantine behavior
- **Network partition simulation** - Test consensus under network splits
- **Reusable scenarios** - Pre-built test scenarios for common patterns

## Quick Start

```rust
use nsn_simulation::{TestHarness, Scenario};

#[tokio::test]
async fn test_consensus() {
    let mut harness = TestHarness::new();

    // Add 5 directors
    for _ in 0..5 {
        harness.add_director();
    }

    // Run baseline consensus scenario
    let result = Scenario::BaselineConsensus.run(&mut harness).await;
    Scenario::BaselineConsensus.verify(&result).unwrap();
}
```

## Architecture

### Components

- **`SimulatedNetwork`** - In-memory message routing with configurable latency
- **`TestHarness`** - High-level orchestrator for multi-node scenarios
- **`mocks`** - Reusable mock implementations for NSN services
- **`Scenario`** - Pre-defined test scenarios with verification

### Mock Types

| Mock | Purpose |
|------|---------|
| `MockVortexClient` | Video generation without actual sidecar |
| `MockBftParticipant` | BFT consensus with configurable behavior |
| `MockChunkPublisher` | Video chunk P2P publishing |
| `MockChainClient` | On-chain event simulation |

## Scenarios

| Scenario | Description |
|----------|-------------|
| `BaselineConsensus` | 5 directors, 1 slot, successful consensus |
| `ByzantineDirector` | 5 directors, 1 Byzantine with divergent embeddings |
| `HighLatencyConsensus` | 5 directors, variable latency (10-100ms) |
| `NetworkPartition` | 5 directors, 2|3 partition |
| `DirectorFailure` | Director goes offline mid-slot |
| `FullEpochLifecycle` | Complete epoch: stake → election → BFT → publish |
| `TaskLifecycle` | Lane 1 task: created → assigned → executed → verified |
| `LaneSwitching` | Lane 0 active, Lane 1 draining, switch |

## Usage

### Basic Harness

```rust
use nsn_simulation::{TestHarness, ByzantineBehavior};
use std::time::Duration;

#[tokio::test]
async fn test_byzantine_fault() {
    let mut harness = TestHarness::new();

    // Add directors
    let d1 = harness.add_director();
    let d2 = harness.add_director();
    let d3 = harness.add_director();
    let d4 = harness.add_director();
    let d5 = harness.add_director();

    // Configure success slots
    harness.configure_slot_success(&[1]);

    // Make d1 Byzantine
    harness.set_byzantine(d1, ByzantineBehavior::DropMessages).unwrap();

    // Activate directors
    harness.emit_epoch_started(&[d1, d2, d3, d4, d5]);

    // Run slot
    let successful = harness.run_slot(1).await.unwrap();

    // 4 honest directors should succeed
    assert_eq!(successful, 4);
}
```

### Network Partitions

```rust
use nsn_simulation::TestHarness;

#[tokio::test]
async fn test_partition() {
    let mut harness = TestHarness::new();

    let directors: Vec<_> = (0..5).map(|_| harness.add_director()).collect();
    harness.configure_slot_success(&[1]);

    // Create partition: [3] | [2]
    harness.inject_partition(vec![
        vec![directors[0], directors[1], directors[2]],
        vec![directors[3], directors[4]],
    ]);

    // Only larger partition active
    harness.emit_epoch_started(&[directors[0], directors[1], directors[2]]);

    let successful = harness.run_slot(1).await.unwrap();
    assert_eq!(successful, 3);

    // Heal partition
    harness.heal_partition();
}
```

### Time Control

```rust
use nsn_simulation::TestHarness;
use std::time::Duration;

#[tokio::test]
async fn test_with_time() {
    let mut harness = TestHarness::new();

    // Pause time for deterministic simulation
    tokio::time::pause();

    harness.add_director();
    harness.configure_slot_success(&[1]);

    // Advance simulation time
    harness.advance_time(Duration::from_millis(100)).await;
}
```

## Adding New Scenarios

1. Add variant to `Scenario` enum in `scenarios.rs`
2. Implement `configure()` to return `ScenarioConfig`
3. Implement scenario runner function
4. Add `run()` match arm
5. Implement `verify()` validation logic

Example:

```rust
// In scenarios.rs
pub enum Scenario {
    // ... existing variants ...
    CustomScenario,
}

impl Scenario {
    pub fn configure(&self) -> ScenarioConfig {
        match self {
            Scenario::CustomScenario => ScenarioConfig {
                num_directors: 5,
                num_executors: 2,
                num_byzantine: 1,
                ..Default::default()
            },
            // ...
        }
    }

    pub async fn run(&self, harness: &mut TestHarness) -> ScenarioResult {
        match self {
            Scenario::CustomScenario => run_custom_scenario(harness).await,
            // ...
        }
    }
}

async fn run_custom_scenario(harness: &mut TestHarness) -> ScenarioResult {
    // Implementation
}
```

## Debugging Tips

1. **Enable tracing** - Set `RUST_LOG=nsn_simulation=debug` for detailed logs
2. **Use `run_until`** - Run simulation until condition met with timeout
3. **Check metrics** - `harness.metrics()` provides execution statistics
4. **Inspect state** - `harness.get_director(&peer)` for node state

## Running Tests

```bash
# Run all simulation tests
cargo test -p nsn-simulation

# Run specific scenario tests
cargo test -p nsn-simulation scenarios::tests

# Run with parallel execution
cargo test -p nsn-simulation -- --test-threads=8
```

## Performance

| Metric | Target |
|--------|--------|
| Single slot consensus | <10ms |
| Full epoch (5 slots) | <50ms |
| 100 node network setup | <100ms |
| Test suite (24 tests) | <1s |
