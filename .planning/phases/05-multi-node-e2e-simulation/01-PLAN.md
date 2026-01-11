# Phase 5: Multi-Node E2E Simulation

## Plan 01: Simulation Test Harness

**Created:** 2026-01-09
**Estimated Scope:** Medium (6-8 tasks)
**Prerequisites:** Phases 2, 3 (Lane 0 and Lane 1 pipelines)

---

## Objective

Create a deterministic network simulation harness for multi-node E2E testing of NSN's distributed components. This enables testing of epoch elections, BFT consensus, P2P message propagation, and fault tolerance without spawning actual processes.

---

## Execution Context

```
@node-core/crates/lane0/tests/director_integration.rs  # Existing mock patterns
@node-core/crates/p2p/src/test_helpers.rs              # P2P service test helpers
@node-core/crates/scheduler/src/state_machine.rs      # State machine implementation
@.planning/codebase/TESTING.md                         # Testing patterns reference
```

---

## Context

### Current Test Infrastructure

**Existing Coverage (362 tests across codebase):**
- Lane 0: Mock implementations for VortexClient, BftParticipant, ChunkPublisher
- Lane 1: TaskExecutorService lifecycle tests with mock chain events
- P2P: Kademlia DHT, NAT traversal, rate limiting (uses real ports, requires `serial_test`)
- Scheduler: State transition validation, task queue ordering

**Key Gaps:**
- No multi-node network simulation (tests use real ports, serialized execution)
- No deterministic timing control for consensus scenarios
- No Byzantine fault injection capability
- No network partition testing

### Research Findings

**Recommended Tools:**
1. **Turmoil** (tokio-rs/turmoil) - Deterministic simulation testing for async Rust
   - Single-threaded execution with simulated time
   - Network topology control (latency, partitions, message drops)
   - Used by Tokio team for distributed systems testing

2. **libp2p-swarm-test** - Official libp2p testing framework
   - MemoryTransport for in-memory P2P without real sockets
   - Eliminates port conflicts, fully deterministic
   - Can run 100+ nodes in single test process

3. **Existing Patterns to Leverage:**
   - MockVortexClient with HashSet<u64> for success/timeout slots
   - MockBftParticipant with HashSet<u64> for consensus slots
   - test_helpers.rs for service creation patterns

### Architecture Decision

**Approach:** Create `/node-core/crates/simulation/` crate containing:
- In-memory network simulation (MemoryTransport)
- Turmoil-based scenario runner
- Reusable mock components extracted from existing tests
- Deterministic time control via tokio::time::pause()

**Rationale:**
- Removes need for `serial_test` barriers
- Enables parallel test execution
- Provides reproducible failure scenarios
- Faster execution (~10ms vs ~100ms per test)

---

## Tasks

### Task 1: Create simulation crate structure

**Goal:** Initialize new simulation crate with proper dependencies

**Actions:**
1. Create `node-core/crates/simulation/` directory structure:
   ```
   simulation/
   ├── Cargo.toml
   ├── src/
   │   ├── lib.rs
   │   ├── network.rs      # SimulatedNetwork implementation
   │   ├── harness.rs      # TestHarness coordinator
   │   ├── mocks/
   │   │   ├── mod.rs
   │   │   ├── vortex.rs   # MockVortexClient
   │   │   ├── bft.rs      # MockBftParticipant
   │   │   ├── chain.rs    # MockChainClient
   │   │   └── p2p.rs      # MockP2pService
   │   └── scenarios.rs    # Scenario definitions
   └── tests/
       └── integration.rs
   ```

2. Configure Cargo.toml with dependencies:
   ```toml
   [package]
   name = "nsn-simulation"
   version = "0.1.0"
   edition = "2021"

   [dependencies]
   tokio = { version = "1.35", features = ["full", "test-util"] }
   async-trait = "0.1"
   thiserror = "1.0"
   tracing = "0.1"

   # Internal dependencies
   nsn-lane0 = { path = "../lane0" }
   nsn-lane1 = { path = "../lane1" }
   nsn-scheduler = { path = "../scheduler" }

   [dev-dependencies]
   turmoil = "0.6"
   libp2p-swarm-test = "0.44"
   proptest = "1.4"
   tracing-subscriber = "0.3"
   ```

3. Add to workspace in `node-core/Cargo.toml`

4. Create lib.rs with module structure and public exports

**Verification:**
```bash
cd node-core && cargo check -p nsn-simulation
```

**Checkpoint:** Crate compiles with all dependencies resolved

---

### Task 2: Extract reusable mocks from existing tests

**Goal:** Consolidate mock implementations into simulation crate

**Actions:**
1. Extract MockVortexClient from `lane0/tests/director_integration.rs`:
   - Move to `simulation/src/mocks/vortex.rs`
   - Add latency injection capability
   - Add failure mode configuration

2. Extract MockBftParticipant from `lane0/tests/director_integration.rs`:
   - Move to `simulation/src/mocks/bft.rs`
   - Add Byzantine behavior configuration (divergent embeddings)
   - Add message delay simulation

3. Extract MockChunkPublisher from `lane0/tests/director_integration.rs`:
   - Move to `simulation/src/mocks/p2p.rs`
   - Track all published messages for assertions

4. Create MockChainClient for chain event simulation:
   - Location: `simulation/src/mocks/chain.rs`
   - Emit TaskEvent variants (Created, AssignedToMe, Verified, Failed)
   - Emit EpochEvent variants (OnDeck, EpochStarted, EpochEnded)
   - Track submitted extrinsics for verification

5. Update lane0 tests to use shared mocks from simulation crate

**Reference:** `node-core/crates/lane0/tests/director_integration.rs` lines 22-169

**Verification:**
```bash
cd node-core && cargo test -p nsn-simulation
cd node-core && cargo test -p nsn-lane0  # Ensure no regression
```

**Checkpoint:** Mocks extracted, lane0 tests still pass

---

### Task 3: Implement SimulatedNetwork

**Goal:** Create in-memory network layer for multi-node testing

**Actions:**
1. Create `simulation/src/network.rs` with:
   ```rust
   pub struct SimulatedNetwork {
       nodes: HashMap<PeerId, SimulatedNode>,
       message_queue: VecDeque<PendingMessage>,
       latency_profile: LatencyProfile,
       partitions: Vec<HashSet<PeerId>>,
   }

   pub struct SimulatedNode {
       peer_id: PeerId,
       role: NodeRole,
       state: NodeState,
       inbox: VecDeque<Message>,
   }

   pub struct PendingMessage {
       from: PeerId,
       to: Option<PeerId>,  // None = broadcast
       topic: TopicCategory,
       payload: Vec<u8>,
       deliver_at: Instant,
   }

   pub enum LatencyProfile {
       Instant,
       Uniform(Duration),
       Variable { min: Duration, max: Duration },
       Custom(Box<dyn Fn(&PeerId, &PeerId) -> Duration>),
   }
   ```

2. Implement network operations:
   - `add_node(role: NodeRole) -> PeerId`
   - `remove_node(peer: PeerId)`
   - `send_message(from, to, topic, payload)`
   - `broadcast(from, topic, payload)`
   - `deliver_pending(until: Instant)`
   - `inject_partition(groups: Vec<HashSet<PeerId>>)`
   - `heal_partition()`

3. Implement message filtering for partitions:
   - Messages only delivered between nodes in same partition
   - Partition changes take effect immediately

4. Add deterministic time control:
   - Use `tokio::time::pause()` in test setup
   - Advance time with `tokio::time::advance(duration)`

**Verification:**
```bash
cd node-core && cargo test -p nsn-simulation network::tests
```

**Checkpoint:** SimulatedNetwork can route messages between nodes with configurable latency

---

### Task 4: Implement TestHarness coordinator

**Goal:** Create high-level test harness for multi-node scenarios

**Actions:**
1. Create `simulation/src/harness.rs` with:
   ```rust
   pub struct TestHarness {
       network: SimulatedNetwork,
       directors: HashMap<PeerId, DirectorService>,
       executors: HashMap<PeerId, TaskExecutorService>,
       chain_state: MockChainState,
       time: Instant,
   }

   impl TestHarness {
       pub fn new() -> Self;

       // Node management
       pub fn add_director(&mut self) -> PeerId;
       pub fn add_executor(&mut self) -> PeerId;
       pub fn set_byzantine(&mut self, peer: PeerId, behavior: ByzantineBehavior);

       // Event injection
       pub async fn emit_epoch_event(&mut self, event: EpochEvent);
       pub async fn broadcast_recipe(&mut self, recipe: Recipe);
       pub async fn submit_task(&mut self, task: Task);

       // Time control
       pub async fn advance_time(&mut self, duration: Duration);
       pub async fn run_until(&mut self, condition: impl Fn(&Self) -> bool);

       // Assertions
       pub fn assert_consensus_reached(&self, slot: u64);
       pub fn assert_chunk_published(&self, slot: u64);
       pub fn assert_task_completed(&self, task_id: u64);
       pub fn get_director_state(&self, peer: PeerId) -> DirectorState;
   }

   pub enum ByzantineBehavior {
       DropMessages,
       DelayMessages(Duration),
       DivergentEmbeddings,
       InvalidSignatures,
   }
   ```

2. Implement lifecycle management:
   - Initialize directors with mock components
   - Wire up message routing through SimulatedNetwork
   - Track all state transitions for verification

3. Implement epoch simulation:
   - Emit OnDeck events to selected directors
   - Track which directors become Active
   - Emit EpochEnded to trigger state transitions

4. Implement consensus simulation:
   - Distribute recipe to Active directors
   - Collect BFT votes through network
   - Verify 3-of-5 threshold reached

**Verification:**
```bash
cd node-core && cargo test -p nsn-simulation harness::tests
```

**Checkpoint:** TestHarness can orchestrate multi-node scenarios

---

### Task 5: Create scenario definitions

**Goal:** Define reusable test scenarios for common patterns

**Actions:**
1. Create `simulation/src/scenarios.rs` with:
   ```rust
   pub enum Scenario {
       /// 5 directors, 1 slot, successful consensus
       BaselineConsensus,

       /// 5 directors, 1 Byzantine with divergent embeddings
       ByzantineDirector,

       /// 5 directors, variable latency (10-100ms)
       HighLatencyConsensus,

       /// 5 directors, 2|3 partition, verify 3-of-5 succeeds in larger partition
       NetworkPartition,

       /// Director goes offline mid-slot, verify recovery
       DirectorFailure,

       /// Full workflow: stake → election → BFT → chunk publish
       FullEpochLifecycle,

       /// Lane 1 task: created → assigned → executed → verified
       TaskLifecycle,

       /// Lane 0 active, Lane 1 draining, epoch ends, Lane 1 resumes
       LaneSwitching,
   }

   impl Scenario {
       pub fn configure(&self) -> ScenarioConfig;
       pub async fn run(&self, harness: &mut TestHarness) -> ScenarioResult;
       pub fn verify(&self, result: &ScenarioResult) -> Result<(), ScenarioFailure>;
   }
   ```

2. Implement each scenario with:
   - Setup phase (node creation, configuration)
   - Execution phase (event injection, time advancement)
   - Verification phase (state assertions)

3. Add metrics collection per scenario:
   - Consensus rounds required
   - Message count
   - Time to completion
   - Failure modes encountered

**Verification:**
```bash
cd node-core && cargo test -p nsn-simulation scenarios::tests
```

**Checkpoint:** All 8 scenarios defined and runnable

---

### Task 6: Implement integration tests

**Goal:** Create comprehensive integration test suite using harness

**Actions:**
1. Create `simulation/tests/integration.rs` with test cases:

   ```rust
   #[tokio::test]
   async fn test_baseline_five_director_consensus() {
       // Setup: 5 directors, all honest
       // Action: Broadcast recipe, run consensus
       // Verify: 3-of-5 threshold reached, chunk published
   }

   #[tokio::test]
   async fn test_byzantine_director_divergent_embedding() {
       // Setup: 5 directors, 1 returns different CLIP embedding
       // Action: Run consensus
       // Verify: Byzantine director's vote rejected, 3 honest votes succeed
   }

   #[tokio::test]
   async fn test_network_partition_recovery() {
       // Setup: 5 directors, partition into [3] | [2]
       // Action: Run consensus in partition
       // Verify: Larger partition (3) reaches consensus
       // Action: Heal partition
       // Verify: Smaller partition accepts result
   }

   #[tokio::test]
   async fn test_epoch_transition_lane_switching() {
       // Setup: 5 directors, 3 executors
       // Action: Lane 1 task in progress, epoch starts
       // Verify: Lane 1 drains, Lane 0 activates
       // Action: Epoch ends
       // Verify: Lane 1 resumes, task continues
   }

   #[tokio::test]
   async fn test_high_latency_consensus_timeout() {
       // Setup: 5 directors, 200ms latency
       // Action: Run consensus with 5000ms timeout
       // Verify: Consensus completes within timeout
   }

   #[tokio::test]
   async fn test_director_failure_mid_slot() {
       // Setup: 5 directors
       // Action: Remove 1 director mid-consensus
       // Verify: Remaining 4 complete consensus (3-of-4 threshold)
   }

   #[tokio::test]
   async fn test_lane1_task_lifecycle() {
       // Setup: 3 executors
       // Action: Submit task, assign to executor
       // Verify: Task created → assigned → executed → verified
   }

   #[tokio::test]
   async fn test_cascading_director_failures() {
       // Setup: 5 directors
       // Action: Remove directors one by one
       // Verify: System fails gracefully when <3 remain
   }
   ```

2. Add property-based tests using proptest:
   ```rust
   proptest! {
       #[test]
       fn consensus_with_random_latency(latency_ms in 1u64..500) {
           // Verify consensus succeeds regardless of latency within bounds
       }

       #[test]
       fn partition_tolerance(partition_size in 1usize..5) {
           // Verify behavior across all partition configurations
       }
   }
   ```

3. Add benchmark scenarios:
   ```rust
   #[tokio::test]
   async fn bench_consensus_latency() {
       // Measure: Time from recipe to chunk publish
       // Target: <100ms simulated time
   }
   ```

**Verification:**
```bash
cd node-core && cargo test -p nsn-simulation --test integration
```

**Checkpoint:** All integration tests pass

---

### Task 7: Update existing tests to use simulation crate

**Goal:** Migrate lane0/lane1 tests to use shared simulation infrastructure

**Actions:**
1. Update `lane0/tests/director_integration.rs`:
   - Import mocks from `nsn_simulation::mocks`
   - Remove duplicated mock definitions
   - Use TestHarness for multi-component tests

2. Update `lane1/tests/integration.rs`:
   - Import MockChainClient from simulation crate
   - Use shared test fixtures

3. Update `p2p/tests/` to use MemoryTransport where applicable:
   - Replace real port bindings with in-memory transport
   - Remove `serial_test` annotations where possible

4. Verify no test regressions:
   ```bash
   cd node-core && cargo test --workspace
   ```

**Verification:**
```bash
cd node-core && cargo test --workspace -- --test-threads=4
```

**Checkpoint:** All workspace tests pass, parallel execution enabled

---

### Task 8: Documentation and CI integration

**Goal:** Document simulation harness and integrate with CI

**Actions:**
1. Add rustdoc to all public APIs in simulation crate:
   - Module-level documentation explaining purpose
   - Examples in doc comments for key types
   - Usage patterns for TestHarness

2. Create `simulation/README.md` with:
   - Quick start guide
   - Scenario catalog
   - Adding new scenarios
   - Debugging tips

3. Update CI workflow (if exists) to run simulation tests:
   ```yaml
   - name: Run simulation tests
     run: cargo test -p nsn-simulation --release
   ```

4. Add test coverage tracking for simulation crate

**Verification:**
```bash
cargo doc -p nsn-simulation --no-deps --open
```

**Checkpoint:** Documentation complete, CI updated

---

## Verification

### Per-Task Verification

Each task includes specific verification commands in its section.

### Phase-Level Verification

After all tasks complete:

```bash
# 1. All simulation tests pass
cd node-core && cargo test -p nsn-simulation

# 2. All workspace tests pass (no regressions)
cd node-core && cargo test --workspace

# 3. Parallel execution works (no serial_test required for simulation)
cd node-core && cargo test -p nsn-simulation -- --test-threads=8

# 4. Documentation builds
cargo doc -p nsn-simulation --no-deps
```

### Scenario Validation Matrix

| Scenario | Directors | Byzantine | Latency | Expected Result |
|----------|-----------|-----------|---------|-----------------|
| Baseline | 5 | 0 | 0ms | Consensus in 1 round |
| Byzantine | 5 | 1 | 0ms | Consensus in 1 round (4 honest) |
| High Latency | 5 | 0 | 100ms | Consensus within timeout |
| Partition | 5 | 0 | 0ms | Larger partition succeeds |
| Failure | 5→4 | 0 | 0ms | Consensus with 4 nodes |
| Cascade | 5→2 | 0 | 0ms | System halts gracefully |

---

## Success Criteria

1. **Simulation crate created** with documented public API
2. **8+ reusable scenarios** covering consensus, faults, and lifecycle
3. **15+ integration tests** using TestHarness
4. **Zero regressions** in existing lane0/lane1/p2p tests
5. **Parallel test execution** enabled (no serial_test barriers in simulation)
6. **Deterministic execution** - tests produce same results on every run
7. **Test execution time** <10 seconds for full simulation suite

---

## Output

Upon completion:
- `node-core/crates/simulation/` - New simulation crate
- Updated tests in lane0, lane1, p2p using shared mocks
- CI integration for simulation tests
- Documentation for extending scenarios

---

## Notes

### Design Decisions

1. **In-process simulation over multi-process**: Enables deterministic testing without port conflicts and faster execution

2. **Turmoil for network simulation**: Provides tokio-native deterministic simulation with network fault injection

3. **Extracted mocks as library**: Prevents code duplication and ensures consistent mock behavior across tests

4. **Scenario-based testing**: Makes it easy to add new test cases and understand test intent

### Dependencies Added

```toml
turmoil = "0.6"              # Deterministic network simulation
libp2p-swarm-test = "0.44"   # In-memory P2P testing
proptest = "1.4"             # Property-based testing
```

### Future Extensions

- Add chaos testing (random fault injection)
- Performance benchmarking with criterion
- Visual test output (network topology graphs)
- Record/replay for debugging
