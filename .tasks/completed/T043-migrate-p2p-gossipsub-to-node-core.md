---
id: T043
title: Migrate GossipSub, Reputation Oracle, and P2P Metrics to node-core
status: in_progress
priority: 1
agent: backend
dependencies: [T042, T003]
blocked_by: []
created: 2025-12-30T08:00:00Z
updated: 2025-12-30T08:00:00Z

context_refs:
  - context/project.md
  - context/architecture.md
  - context/acceptance-templates.md

docs_refs:
  - .claude/rules/architecture.md
  - .claude/rules/prd.md

est_tokens: 13000
actual_tokens: null
---

## Description

Migrate GossipSub configuration, topic management, peer scoring, reputation oracle, and Prometheus metrics from `legacy-nodes/common/src/p2p/` to `node-core/crates/p2p/src/`. This completes the P2P migration started in T042 by adding the messaging, reputation, and observability layers.

**Context**: T022 implemented a complete GossipSub system with 6 topics (5 Lane 0 + 1 Lane 1), reputation-integrated peer scoring, and on-chain reputation oracle. This code currently lives in `legacy-nodes/common/src/p2p/` but needs to migrate to `node-core/crates/p2p/` to support the dual-lane architecture and enable legacy-nodes deprecation.

**Technical Approach**:
- Migrate GossipSub configuration with NSN-specific parameters (mesh_n=6, 16MB max transmit)
- Migrate topic definitions for 6 NSN topics (recipes, video, bft, attestations, challenges, tasks)
- Migrate peer scoring with reputation-integrated thresholds (gossip, publish, graylist)
- Migrate reputation oracle for on-chain score syncing (subxt integration)
- Migrate Prometheus metrics for P2P observability
- Integrate all components with T042's P2pService

**Integration Points**:
- **T042 P2pService**: Use migrated GossipSub in service.rs Swarm
- **pallet-icn-reputation (T003)**: Reputation oracle subscribes to on-chain events via subxt
- **Prometheus**: Expose P2P metrics on configured port (default 9100)
- **All NSN Topics**: Support both Lane 0 (video generation) and Lane 1 (task marketplace)

## Business Context

**User Story**: As a node operator, I need reputation-integrated GossipSub messaging with comprehensive metrics so that I can participate in NSN's dual-lane network with confidence that malicious peers are deprioritized and network health is observable.

**Why This Matters**:
- **Network Security**: Reputation-based peer scoring prevents Sybil attacks and spam
- **Message Reliability**: GossipSub ensures 99.5%+ message delivery for BFT and video distribution
- **Observability**: Prometheus metrics enable monitoring of P2P health (peer count, message throughput, scoring)
- **Dual-Lane Support**: Topics cover both Lane 0 (video) and Lane 1 (task marketplace)
- **Operational Confidence**: Operators can monitor and troubleshoot P2P networking

**What It Unblocks**:
- T044: Complete legacy-nodes deprecation (all P2P code now in node-core)
- Future Director/Validator/Super-Node implementations using node-core
- Production observability stack (T033)
- Mainnet readiness with proven GossipSub implementation

**Priority Justification**: Priority 1 (Critical Path) because:
- Completes P2P migration started in T042
- Required before legacy-nodes can be removed (T044)
- GossipSub is core to NSN off-chain communication
- Metrics are essential for production operations

## Acceptance Criteria

- [ ] **GossipSub Config Migrated**: `gossipsub.rs` with `build_gossipsub_config()`, `create_gossipsub_behaviour()` functions
- [ ] **GossipSub Mesh Parameters**: MESH_N=6, MESH_N_LOW=4, MESH_N_HIGH=12 configured correctly
- [ ] **Max Transmit Size**: 16MB limit for video chunks configured
- [ ] **Flood Publishing**: Enabled for low-latency BFT signals
- [ ] **Topic Definitions**: `topics.rs` with all 6 NSN topics (recipes, video, bft, attestations, challenges, tasks)
- [ ] **Topic Parsing**: `parse_topic()` function correctly identifies Lane 0 vs Lane 1 topics
- [ ] **Peer Scoring**: `scoring.rs` with reputation-integrated scoring parameters (GOSSIP_THRESHOLD=-10, PUBLISH_THRESHOLD=-50, GRAYLIST_THRESHOLD=-100)
- [ ] **Reputation Oracle**: `reputation_oracle.rs` with on-chain score syncing via subxt, 60s sync interval
- [ ] **Metrics**: `metrics.rs` with Prometheus metrics (peer_count, connection_count, message_throughput, scoring metrics)
- [ ] **Metrics Endpoint**: Prometheus metrics exposed on configured port (default 9100)
- [ ] **Integration**: GossipSub integrated into T042's P2pService::new() and event loop
- [ ] **Subscription**: `subscribe_to_all_topics()` and `subscribe_to_categories()` functions work
- [ ] **Publishing**: `publish_message()` function successfully broadcasts to topics
- [ ] **Compilation**: `cargo build --release -p nsn-p2p` succeeds without warnings
- [ ] **Tests**: All migrated GossipSub tests pass (from T022)

## Test Scenarios

**Test Case 1: GossipSub Initialization**
- **Given**: P2pService with keypair and reputation oracle
- **When**: `create_gossipsub_behaviour(keypair, oracle)` is called
- **Then**: GossipSub behavior is created with correct mesh parameters, validation mode STRICT, and peer scoring enabled

**Test Case 2: Topic Subscription**
- **Given**: GossipSub behavior initialized
- **When**: `subscribe_to_all_topics(&mut gossipsub)` is called
- **Then**: Service subscribes to all 6 topics (recipes, video, bft, attestations, challenges, tasks) successfully

**Test Case 3: Message Publishing**
- **Given**: Service subscribed to `/nsn/bft/1.0.0` topic
- **When**: `publish_message(&mut gossipsub, TopicCategory::Bft, b"embedding_hash")` is called
- **Then**: Message is published with Ed25519 signature, MessageId is returned

**Test Case 4: Message Reception**
- **Given**: Two connected peers both subscribed to `/nsn/recipes/1.0.0`
- **When**: Peer A publishes a recipe message
- **Then**: Peer B receives the message via GossipSub, message is validated and delivered

**Test Case 5: Reputation Oracle Sync**
- **Given**: Reputation oracle connected to NSN Chain RPC (ws://localhost:9944)
- **When**: Oracle syncs after 60s interval
- **Then**: Latest reputation scores fetched from pallet-icn-reputation, cached locally

**Test Case 6: Peer Scoring**
- **Given**: Peer with on-chain reputation score of 800
- **When**: `compute_app_specific_score(peer_id, &oracle)` is called
- **Then**: Score contribution is calculated correctly (normalized 0-50 boost)

**Test Case 7: Graylist Enforcement**
- **Given**: Peer with total score below GRAYLIST_THRESHOLD (-100)
- **When**: Peer attempts to send message
- **Then**: GossipSub rejects message, peer is graylisted

**Test Case 8: Prometheus Metrics**
- **Given**: P2pService running with metrics enabled
- **When**: HTTP GET to `http://localhost:9100/metrics`
- **Then**: Metrics endpoint returns Prometheus-formatted metrics (nsn_p2p_peer_count, nsn_p2p_message_throughput, etc.)

**Test Case 9: Lane 0 vs Lane 1 Topics**
- **Given**: Topic parsing function
- **When**: `parse_topic("/nsn/bft/1.0.0")` is called
- **Then**: Returns `TopicCategory::Bft` with lane=0
- **When**: `parse_topic("/nsn/tasks/1.0.0")` is called
- **Then**: Returns `TopicCategory::Tasks` with lane=1

**Test Case 10: Error Handling**
- **Given**: Oracle unable to connect to chain RPC
- **When**: Reputation sync is attempted
- **Then**: `OracleError::RpcConnectionFailed` is logged, service continues with cached scores

## Technical Implementation

**Required Components**:

1. **node-core/crates/p2p/src/gossipsub.rs**
   - Migrate `build_gossipsub_config()` with NSN parameters
   - Migrate `create_gossipsub_behaviour()` with reputation integration
   - Migrate `handle_gossipsub_event()` for event processing
   - Migrate `publish_message()` for topic publishing
   - Migrate `subscribe_to_all_topics()`, `subscribe_to_categories()`
   - Migrate constants: MESH_N, MESH_N_LOW, MESH_N_HIGH, MAX_TRANSMIT_SIZE, HEARTBEAT_INTERVAL
   - Migrate `GossipsubError` enum

2. **node-core/crates/p2p/src/topics.rs**
   - Migrate `TopicCategory` enum: Recipes, Video, Bft, Attestations, Challenges, Tasks
   - Migrate topic string constants: `/nsn/recipes/1.0.0`, `/nsn/video/1.0.0`, etc.
   - Migrate `all_topics()` function returning all 6 topics
   - Migrate `lane_0_topics()` (5 topics) and `lane_1_topics()` (1 topic)
   - Migrate `parse_topic(topic_str)` for topic parsing

3. **node-core/crates/p2p/src/scoring.rs**
   - Migrate `build_peer_score_params()` with reputation integration
   - Migrate `compute_app_specific_score(peer_id, oracle)` using on-chain reputation
   - Migrate threshold constants: GOSSIP_THRESHOLD, PUBLISH_THRESHOLD, GRAYLIST_THRESHOLD
   - Migrate penalty constants: INVALID_MESSAGE_PENALTY, BFT_INVALID_MESSAGE_PENALTY
   - Migrate topic weights: BFT (3.0) > Video (2.0) > Recipes (1.0)

4. **node-core/crates/p2p/src/reputation_oracle.rs**
   - Migrate `ReputationOracle` struct with subxt client
   - Migrate `new(rpc_url)` for initialization
   - Migrate `sync_reputation_scores()` for periodic on-chain sync (60s interval)
   - Migrate `get_score(account_id)` for cached score lookup
   - Migrate `DEFAULT_REPUTATION` constant (500)
   - Migrate `SYNC_INTERVAL` constant (60s)
   - Migrate `OracleError` enum

5. **node-core/crates/p2p/src/metrics.rs**
   - Migrate `P2pMetrics` struct with Prometheus gauges/counters/histograms
   - Migrate metrics: `peer_count`, `connection_count`, `message_throughput`, `gossipsub_mesh_size`
   - Migrate `new(registry)` for initialization
   - Migrate metric update methods: `set_peer_count()`, `inc_message_sent()`, etc.
   - Migrate `MetricsError` enum
   - Integrate with Prometheus registry

6. **node-core/crates/p2p/src/service.rs** (Update from T042)
   - Integrate GossipSub behavior into Swarm
   - Add reputation oracle initialization in `P2pService::new()`
   - Add metrics initialization
   - Add GossipSub event handling in event loop
   - Add `ServiceCommand::Subscribe` and `ServiceCommand::Publish` handling

7. **node-core/crates/p2p/src/mod.rs** (Update)
   - Re-export GossipSub types: `build_gossipsub_config`, `create_gossipsub_behaviour`, `publish_message`
   - Re-export topics: `TopicCategory`, `all_topics`, `lane_0_topics`, `lane_1_topics`
   - Re-export scoring: `build_peer_score_params`, `compute_app_specific_score`
   - Re-export oracle: `ReputationOracle`, `DEFAULT_REPUTATION`, `SYNC_INTERVAL`
   - Re-export metrics: `P2pMetrics`

8. **node-core/crates/p2p/Cargo.toml** (Update)
   - Add dependency: `subxt` (0.37, for reputation oracle chain client)
   - Add dependency: `prometheus` (0.13, for metrics)
   - Verify existing: `libp2p` (with gossipsub feature), `tokio`, `futures`, `thiserror`

**Validation Commands**:

```bash
# Build P2P crate with new components
cargo build --release -p nsn-p2p

# Run all tests including GossipSub tests
cargo test -p nsn-p2p

# Run Clippy
cargo clippy -p nsn-p2p -- -D warnings

# Test metrics endpoint (requires running service)
curl http://localhost:9100/metrics | grep nsn_p2p

# Verify topic parsing
cargo test -p nsn-p2p test_topic_parsing

# Verify reputation oracle
cargo test -p nsn-p2p test_reputation_oracle_sync
```

**Code Patterns**:

From `legacy-nodes/common/src/p2p/gossipsub.rs`:
```rust
pub fn create_gossipsub_behaviour(
    keypair: &Keypair,
    reputation_oracle: Arc<ReputationOracle>,
) -> Result<GossipsubBehaviour, GossipsubError> {
    let config = build_gossipsub_config()?;
    let message_id_fn = |message: &Message| {
        let mut hasher = DefaultHasher::new();
        message.data.hash(&mut hasher);
        MessageId::from(hasher.finish().to_string())
    };

    let peer_score_params = build_peer_score_params(reputation_oracle);

    GossipsubBehaviour::new(
        MessageAuthenticity::Signed(keypair.clone()),
        config,
    )
    .map_err(|e| GossipsubError::BehaviourCreation(e.to_string()))?
    .with_peer_score(peer_score_params, peer_score_thresholds())
    .map_err(|e| GossipsubError::BehaviourCreation(e.to_string()))
}
```

From `legacy-nodes/common/src/p2p/topics.rs`:
```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TopicCategory {
    Recipes,      // Lane 0
    Video,        // Lane 0
    Bft,          // Lane 0
    Attestations, // Lane 0
    Challenges,   // Lane 0
    Tasks,        // Lane 1
}

pub fn all_topics() -> Vec<(&'static str, TopicCategory)> {
    vec![
        ("/nsn/recipes/1.0.0", TopicCategory::Recipes),
        ("/nsn/video/1.0.0", TopicCategory::Video),
        ("/nsn/bft/1.0.0", TopicCategory::Bft),
        ("/nsn/attestations/1.0.0", TopicCategory::Attestations),
        ("/nsn/challenges/1.0.0", TopicCategory::Challenges),
        ("/nsn/tasks/1.0.0", TopicCategory::Tasks),
    ]
}
```

## Dependencies

**Hard Dependencies** (must be complete first):
- [T042] Migrate P2P Core Implementation - **PENDING** 🟡70 [REPORTED] - Provides P2pService foundation to integrate GossipSub
- [T003] pallet-icn-reputation - **COMPLETED** 🟢95 [CONFIRMED] - Reputation oracle subscribes to reputation events

**Soft Dependencies** (nice to have):
- None

**External Dependencies**:
- subxt 0.37 - Chain client for reputation oracle
- prometheus 0.13 - Metrics instrumentation
- libp2p 0.53 (gossipsub feature) - GossipSub protocol implementation

## Design Decisions

**Decision 1: Preserve Exact GossipSub Config from T022**
- **Rationale**: T022 config is battle-tested and optimized for NSN (mesh_n=6, 16MB video chunks, flood publish for BFT)
- **Alternatives**: Redesign config (high risk of regressions), use libp2p defaults (not optimized for NSN)
- **Trade-offs**:
  - (+) Proven to work from T022 implementation
  - (+) No behavior changes during migration
  - (+) Easier verification (compare output with legacy-nodes)
  - (-) Inherits any suboptimal decisions from T022
  - (-) Future optimizations require separate refactor

**Decision 2: Keep 6 Topics (5 Lane 0 + 1 Lane 1)**
- **Rationale**: Topic structure supports dual-lane architecture with clear Lane 0 (video) and Lane 1 (task marketplace) separation
- **Alternatives**: Merge topics (loses flexibility), add more topics (increases overhead)
- **Trade-offs**:
  - (+) Clear separation of Lane 0 and Lane 1 concerns
  - (+) Flood publishing can target BFT topic specifically
  - (+) Topic-specific scoring weights (BFT > Video > Recipes)
  - (-) 6 topics = 6 mesh maintenance cycles per heartbeat
  - (-) Overhead for nodes only participating in one lane

**Decision 3: Reputation Oracle with 60s Sync Interval**
- **Rationale**: 60s strikes balance between on-chain score freshness and RPC load
- **Alternatives**: Real-time event subscription (high RPC load), longer interval (stale scores)
- **Trade-offs**:
  - (+) Low RPC overhead (1 query per minute)
  - (+) Recent enough for peer scoring (reputation changes slowly)
  - (+) Cached scores reduce latency
  - (-) Up to 60s delay in score updates
  - (-) Malicious peer could exploit stale scores for ~60s

**Decision 4: Integrate Metrics into P2pService**
- **Rationale**: Centralized metrics in P2pService makes them easy to access and consistent across all P2P operations
- **Alternatives**: Distributed metrics (each module owns metrics), no metrics (blind operation)
- **Trade-offs**:
  - (+) Single source of truth for P2P metrics
  - (+) Easy to expose Prometheus endpoint
  - (+) Simplified testing (mock metrics in one place)
  - (-) P2pService has more responsibilities
  - (-) Tight coupling between service and metrics

## Risks & Mitigations

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| GossipSub config mismatch breaks message delivery | High | Low | Copy exact config from T022, run integration tests with message publishing |
| Reputation oracle RPC failures cause scoring to fail | Medium | Medium | Cache scores locally, use DEFAULT_REPUTATION (500) on fetch failure, log errors |
| Topic parsing errors misroute messages | High | Low | Comprehensive unit tests for all 6 topics, validate against legacy-nodes behavior |
| Metrics endpoint conflicts with other services | Low | Medium | Make metrics port configurable, default to 9100, document in config |
| Peer scoring thresholds too aggressive | Medium | Low | Use exact values from T022 (-10, -50, -100), monitor graylist rate in production |
| Integration with T042 breaks service initialization | High | Low | Test P2pService::new() with GossipSub, verify Swarm creation succeeds |
| Message size exceeds MAX_TRANSMIT_SIZE (16MB) | Medium | Low | Validate message sizes before publishing, return error if too large |
| Ed25519 signature verification fails | High | Low | Test message authenticity with signed messages, verify keypair compatibility |

## Progress Log

### [2025-12-30T08:00:00Z] - Task Created

**Created By**: task-creator agent
**Reason**: Migrate GossipSub, reputation oracle, and metrics from legacy-nodes to node-core to complete P2P migration
**Dependencies**: T042 (P2P Core), T003 (pallet-icn-reputation)
**Estimated Complexity**: Standard (13,000 tokens)

## Completion Checklist

**Code Quality**:
- [ ] All migrated files compile (`cargo build -p nsn-p2p`)
- [ ] Clippy passes (`cargo clippy -p nsn-p2p -- -D warnings`)
- [ ] Code formatted (`cargo fmt -p nsn-p2p`)
- [ ] No unused imports or dead code

**Testing**:
- [ ] Unit tests pass (`cargo test -p nsn-p2p`)
- [ ] GossipSub initialization test passes
- [ ] Topic subscription tests pass
- [ ] Message publishing/receiving tests pass
- [ ] Reputation oracle sync test passes
- [ ] Peer scoring tests pass
- [ ] Metrics endpoint test passes

**Documentation**:
- [ ] All public functions have rustdoc comments
- [ ] GossipSub config parameters documented
- [ ] Topic categories explained
- [ ] Scoring thresholds justified
- [ ] Metrics descriptions complete

**Integration**:
- [ ] GossipSub integrated into T042's P2pService
- [ ] Reputation oracle connects to chain RPC
- [ ] Metrics exposed on Prometheus endpoint
- [ ] Service commands (Subscribe/Publish) work end-to-end

**Validation**:
- [ ] `cargo build --release -p nsn-p2p` succeeds
- [ ] `cargo test -p nsn-p2p` all tests pass
- [ ] Prometheus metrics endpoint returns valid metrics
- [ ] Ready for T044 (legacy-nodes removal)

**Definition of Done**:
Task is complete when ALL acceptance criteria are met, ALL tests pass, GossipSub is fully integrated with P2pService, reputation oracle syncs correctly, Prometheus metrics are exposed, and code is production-ready for T044.
