---
id: T022
title: GossipSub Configuration with Reputation Integration
status: pending
priority: 1
agent: backend
dependencies: [T021, T003]
blocked_by: []
created: 2025-12-24T00:00:00Z
updated: 2025-12-24T00:00:00Z
tags: [p2p, gossipsub, reputation, off-chain, phase1, critical-path]

context_refs:
  - context/project.md
  - context/architecture.md
  - context/acceptance-templates.md

docs_refs:
  - ../docs/prd.md (§17.3, §3.2)
  - ../docs/architecture.md (§4.4.2, §17.3)

est_tokens: 12000
actual_tokens: null
---

## Description

Implement GossipSub pub/sub messaging protocol with ICN-specific topic configuration and on-chain reputation-integrated peer scoring (v8.0.1 enhancement). This enables efficient broadcast of recipes, video chunks, BFT signals, attestations, and challenges across the hierarchical P2P swarm.

**Technical Approach:**
- Configure libp2p GossipSub with ICN topic definitions
- Set mesh parameters for optimal message propagation (n=6, n_low=4, n_high=12)
- Implement peer scoring with topic weights and invalid message penalties
- Create ReputationOracle to integrate on-chain reputation scores into GossipSub peer scoring
- Use strict validation mode with Ed25519 message signing
- Enable flood publishing for low-latency BFT signals
- Implement background sync loop to fetch on-chain reputation scores

**Integration Points:**
- Builds on T021 (libp2p core setup)
- Integrates with T003 (pallet-icn-reputation) via subxt RPC client
- Used by all off-chain nodes for message propagation

## Business Context

**User Story:** As an ICN Director, I want to broadcast BFT signals and video chunks efficiently to the network, so that consensus can be reached quickly and viewers receive content with low latency.

**Why This Matters:**
- Core messaging layer for off-chain coordination (BFT, video distribution, challenges)
- Reputation integration prevents Sybil attacks and prioritizes high-quality peers
- Low latency (flood publishing) critical for 45-second slot deadline

**What It Unblocks:**
- Director BFT coordination (embedding exchange)
- Video chunk distribution to Super-Nodes
- Recipe propagation to Directors
- Challenge broadcasts to Validators

**Priority Justification:** Critical path for Phase 1. BFT consensus and video distribution impossible without GossipSub.

## Acceptance Criteria

- [ ] **Topic Definitions**: 5 topics defined with correct names and versions:
  - `/icn/recipes/1.0.0`
  - `/icn/video/1.0.0`
  - `/icn/bft/1.0.0`
  - `/icn/attestations/1.0.0`
  - `/icn/challenges/1.0.0`
- [ ] **Topic Subscription**: Node can subscribe to topics and receive messages
- [ ] **Message Publishing**: Node can publish messages to topics with Ed25519 signature
- [ ] **Message Validation**: Strict validation mode rejects unsigned or invalid messages
- [ ] **Mesh Parameters**: Mesh size maintained between 4-12 peers per topic
- [ ] **Peer Scoring**: Peer scores computed based on topic delivery, invalid messages, and on-chain reputation
- [ ] **Reputation Oracle**: Background sync fetches on-chain reputation scores every 60 seconds
- [ ] **Reputation Integration**: GossipSub peer score boosted by on-chain reputation (0-50 bonus)
- [ ] **Flood Publishing**: BFT signals published to all mesh peers (flood_publish=true)
- [ ] **Max Transmit Size**: Video chunks up to 16MB accepted
- [ ] **Invalid Message Penalty**: Peers publishing invalid messages receive score penalty (-10 to -20)
- [ ] **Graylist Enforcement**: Peers with score < -100 are ignored entirely
- [ ] **Metrics Exposed**: Topics subscribed, messages sent/received, peer scores on Prometheus endpoint

## Test Scenarios

**Test Case 1: Topic Subscription and Message Propagation**
- Given: Three nodes (A, B, C) connected in a mesh
- When: Node A subscribes to `/icn/recipes/1.0.0` and Node B publishes a recipe message
- Then:
  - Node A receives the recipe message
  - Node C forwards the message (gossip)
  - Message delivery latency < 500ms

**Test Case 2: Message Signing and Validation**
- Given: Node A publishes a message to `/icn/bft/1.0.0`
- When: Node B receives the message
- Then:
  - Message signature is verified using Node A's PeerId public key
  - Message is accepted and forwarded
  - Metrics show valid_messages_received +1

**Test Case 3: Invalid Message Rejection**
- Given: Node A receives a message with invalid signature on `/icn/bft/1.0.0`
- When: Validation is performed
- Then:
  - Message is rejected
  - Sender peer score decreases by -20
  - Metrics show invalid_messages_rejected +1
  - Warning logged: "Invalid message from <peer_id> on topic /icn/bft/1.0.0"

**Test Case 4: Mesh Size Maintenance**
- Given: Node A subscribed to `/icn/video/1.0.0` with 15 potential peers
- When: GossipSub runs heartbeat (every 1 second)
- Then:
  - Mesh size stabilizes between 6-12 peers
  - Lowest-scoring peers are pruned when mesh > 12
  - Highest-scoring peers are grafted when mesh < 4

**Test Case 5: On-Chain Reputation Integration**
- Given: ReputationOracle has cached scores:
  - Peer A: on-chain reputation = 1000 (high)
  - Peer B: on-chain reputation = 100 (low)
- When: GossipSub computes peer scores
- Then:
  - Peer A receives +50 reputation boost (1000/1000 * 50)
  - Peer B receives +5 reputation boost (100/1000 * 50)
  - Peer A is more likely to be in mesh than Peer B

**Test Case 6: Reputation Oracle Sync**
- Given: ReputationOracle with empty cache
- When: Background sync loop runs
- Then:
  - Oracle queries pallet-icn-reputation via subxt
  - All account reputation scores fetched
  - PeerIds mapped to reputation scores
  - Cache populated within 5 seconds

**Test Case 7: Flood Publishing for BFT Signals**
- Given: Node A in mesh with 8 peers on `/icn/bft/1.0.0`
- When: Node A publishes BFT signal (CLIP embedding)
- Then:
  - Message sent immediately to all 8 peers (no gossip delay)
  - Delivery within 100ms
  - All peers receive message

**Test Case 8: Large Video Chunk Transmission**
- Given: Node A publishes 15MB video chunk on `/icn/video/1.0.0`
- When: Node B receives the message
- Then:
  - Message accepted (within 16MB max_transmit_size)
  - Message delivered completely
  - No fragmentation errors

**Test Case 9: Graylist Enforcement**
- Given: Peer C with score = -120 (below graylist_threshold of -100)
- When: Peer C attempts to publish message
- Then:
  - Node A ignores the message
  - Node A does not forward gossip from Peer C
  - Metrics show messages_ignored_from_graylisted_peers +1

## Technical Implementation

**Required Components:**

```
off-chain/src/p2p/
├── gossipsub.rs            # GossipSub configuration and behavior
├── topics.rs               # Topic definitions and constants
├── reputation_oracle.rs    # On-chain reputation sync
└── scoring.rs              # Custom peer scoring logic

off-chain/tests/
└── integration_gossipsub.rs  # GossipSub integration tests
```

**Key Rust Modules:**

```rust
// src/p2p/topics.rs
pub const RECIPES_TOPIC: &str = "/icn/recipes/1.0.0";
pub const VIDEO_CHUNKS_TOPIC: &str = "/icn/video/1.0.0";
pub const BFT_SIGNALS_TOPIC: &str = "/icn/bft/1.0.0";
pub const ATTESTATIONS_TOPIC: &str = "/icn/attestations/1.0.0";
pub const CHALLENGES_TOPIC: &str = "/icn/challenges/1.0.0";

pub fn all_topics() -> Vec<IdentTopic> {
    vec![
        IdentTopic::new(RECIPES_TOPIC),
        IdentTopic::new(VIDEO_CHUNKS_TOPIC),
        IdentTopic::new(BFT_SIGNALS_TOPIC),
        IdentTopic::new(ATTESTATIONS_TOPIC),
        IdentTopic::new(CHALLENGES_TOPIC),
    ]
}

// src/p2p/reputation_oracle.rs
use subxt::{OnlineClient, PolkadotConfig};
use sp_core::crypto::AccountId32;
use std::collections::HashMap;
use tokio::sync::RwLock;

pub struct ReputationOracle {
    cache: Arc<RwLock<HashMap<PeerId, u64>>>,
    chain_client: OnlineClient<PolkadotConfig>,
    account_to_peer_map: Arc<RwLock<HashMap<AccountId32, PeerId>>>,
}

impl ReputationOracle {
    pub async fn new(rpc_url: &str) -> Result<Self, Error> {
        let chain_client = OnlineClient::<PolkadotConfig>::from_url(rpc_url).await?;

        Ok(Self {
            cache: Arc::new(RwLock::new(HashMap::new())),
            chain_client,
            account_to_peer_map: Arc::new(RwLock::new(HashMap::new())),
        })
    }

    pub fn get_reputation(&self, peer_id: &PeerId) -> u64 {
        self.cache.blocking_read().get(peer_id).copied().unwrap_or(100)
    }

    pub async fn sync_loop(&self) {
        loop {
            if let Err(e) = self.fetch_all_reputations().await {
                tracing::error!("Reputation sync failed: {}", e);
            }
            tokio::time::sleep(Duration::from_secs(60)).await;
        }
    }

    async fn fetch_all_reputations(&self) -> Result<(), Error> {
        // Query pallet-icn-reputation storage
        let storage_query = icn_reputation::storage().reputation_scores_root();

        let mut iter = self.chain_client
            .storage()
            .at_latest()
            .await?
            .iter(storage_query)
            .await?;

        let mut new_cache = HashMap::new();

        while let Some((key, value)) = iter.next().await? {
            let account = key.0;
            let score = value;

            // Convert AccountId32 to PeerId (using account_to_peer_map)
            if let Some(peer_id) = self.account_to_peer(&account).await {
                new_cache.insert(peer_id, score.total());
            }
        }

        *self.cache.write().await = new_cache;
        tracing::info!("Synced {} reputation scores from chain", new_cache.len());

        Ok(())
    }

    async fn account_to_peer(&self, account: &AccountId32) -> Option<PeerId> {
        self.account_to_peer_map.read().await.get(account).copied()
    }

    pub async fn register_peer(&self, account: AccountId32, peer_id: PeerId) {
        self.account_to_peer_map.write().await.insert(account, peer_id);
    }
}

// src/p2p/gossipsub.rs
use libp2p::gossipsub::{
    Gossipsub, GossipsubConfig, MessageAuthenticity,
    ValidationMode, PeerScoreParams, TopicScoreParams,
    PeerScoreThresholds, IdentTopic,
};

pub fn build_gossipsub(
    keypair: &Keypair,
    reputation_oracle: Arc<ReputationOracle>,
) -> Gossipsub {
    let config = GossipsubConfig::builder()
        .heartbeat_interval(Duration::from_secs(1))
        .validation_mode(ValidationMode::Strict)
        .mesh_n(6)
        .mesh_n_low(4)
        .mesh_n_high(12)
        .gossip_lazy(6)
        .gossip_factor(0.25)
        .max_transmit_size(16 * 1024 * 1024)  // 16MB
        .flood_publish(true)  // Low-latency for BFT
        .history_length(12)
        .history_gossip(3)
        .build()
        .expect("Valid gossipsub config");

    let peer_score_params = build_peer_score_params();

    let mut gossipsub = Gossipsub::new(
        MessageAuthenticity::Signed(keypair.clone()),
        config,
    )
    .expect("Gossipsub creation failed")
    .with_peer_score(peer_score_params, Default::default())
    .expect("Peer scoring setup failed");

    // Integrate on-chain reputation into scoring
    let oracle_clone = reputation_oracle.clone();
    gossipsub.with_custom_score(move |peer_id: &PeerId| -> f64 {
        let on_chain_rep = oracle_clone.get_reputation(peer_id);
        (on_chain_rep as f64 / 1000.0) * 50.0
    });

    gossipsub
}

fn build_peer_score_params() -> PeerScoreParams {
    PeerScoreParams::builder()
        .topics(vec![
            (IdentTopic::new(RECIPES_TOPIC), TopicScoreParams {
                topic_weight: 1.0,
                first_message_deliveries_weight: 0.5,
                first_message_deliveries_decay: 0.9,
                first_message_deliveries_cap: 100.0,
                invalid_message_deliveries_weight: -10.0,
                invalid_message_deliveries_decay: 0.5,
                ..Default::default()
            }),
            (IdentTopic::new(VIDEO_CHUNKS_TOPIC), TopicScoreParams {
                topic_weight: 2.0,
                first_message_deliveries_weight: 1.0,
                mesh_message_deliveries_weight: 0.5,
                invalid_message_deliveries_weight: -10.0,
                ..Default::default()
            }),
            (IdentTopic::new(BFT_SIGNALS_TOPIC), TopicScoreParams {
                topic_weight: 3.0,
                invalid_message_deliveries_weight: -20.0,  // Harsh penalty
                ..Default::default()
            }),
            (IdentTopic::new(ATTESTATIONS_TOPIC), TopicScoreParams {
                topic_weight: 2.0,
                invalid_message_deliveries_weight: -10.0,
                ..Default::default()
            }),
            (IdentTopic::new(CHALLENGES_TOPIC), TopicScoreParams {
                topic_weight: 2.5,
                invalid_message_deliveries_weight: -15.0,
                ..Default::default()
            }),
        ])
        .thresholds(PeerScoreThresholds {
            gossip_threshold: -10.0,
            publish_threshold: -50.0,
            graylist_threshold: -100.0,
            accept_px_threshold: 0.0,
            opportunistic_graft_threshold: 5.0,
        })
        .build()
}

// Example usage in P2pService
impl P2pService {
    pub async fn new(config: P2pConfig, rpc_url: &str) -> Result<Self, Error> {
        let keypair = load_or_generate_keypair(&config.keypair_path)?;
        let transport = build_transport(&keypair)?;

        let reputation_oracle = Arc::new(ReputationOracle::new(rpc_url).await?);

        // Spawn background sync task
        let oracle_clone = reputation_oracle.clone();
        tokio::spawn(async move {
            oracle_clone.sync_loop().await;
        });

        let gossipsub = build_gossipsub(&keypair, reputation_oracle.clone());

        let behaviour = Behaviour {
            gossipsub,
            // ... other behaviors
        };

        let swarm = libp2p::SwarmBuilder::with_existing_identity(keypair)
            .with_tokio()
            .with_quic()
            .with_behaviour(|_| behaviour)?
            .with_swarm_config(|cfg| cfg.with_idle_connection_timeout(config.connection_timeout))
            .build();

        Ok(Self { swarm, config, reputation_oracle })
    }

    pub async fn subscribe(&mut self, topic: &str) -> Result<(), Error> {
        self.swarm.behaviour_mut().gossipsub.subscribe(&IdentTopic::new(topic))?;
        tracing::info!("Subscribed to topic: {}", topic);
        Ok(())
    }

    pub async fn publish(&mut self, topic: &str, data: Vec<u8>) -> Result<MessageId, Error> {
        let topic = IdentTopic::new(topic);
        let msg_id = self.swarm.behaviour_mut().gossipsub.publish(topic, data)?;
        tracing::debug!("Published message {} to topic: {}", msg_id, topic);
        Ok(msg_id)
    }
}
```

**Validation Commands:**

```bash
# Build with GossipSub
cargo build --release -p icn-off-chain --features gossipsub

# Run unit tests
cargo test -p icn-off-chain gossipsub::

# Run integration tests (requires chain and 3 nodes)
cargo test --test integration_gossipsub -- --nocapture

# Start node with GossipSub
RUST_LOG=debug cargo run --release -- \
  --port 9000 \
  --rpc-url ws://localhost:9944 \
  --subscribe recipes,video,bft

# Check GossipSub metrics
curl http://localhost:9100/metrics | grep gossipsub
```

**Code Patterns:**
- Use `IdentTopic` for topic definitions (string-based)
- Message authentication via `MessageAuthenticity::Signed`
- Custom scoring function for on-chain reputation integration
- Background tokio task for reputation sync
- Structured logging with peer_id and topic context

## Dependencies

**Hard Dependencies** (must be complete first):
- [T021] libp2p Core Setup - provides transport, encryption, PeerId
- [T003] pallet-icn-reputation - on-chain reputation scores

**Soft Dependencies:**
- None

**External Dependencies:**
- libp2p-gossipsub 0.53.0
- subxt 0.34+ (chain client)
- tokio (async runtime)

## Design Decisions

**Decision 1: Flood Publishing for BFT Signals**
- **Rationale:** BFT coordination requires lowest possible latency (<100ms). Flood publishing sends to all mesh peers immediately instead of waiting for gossip propagation.
- **Alternatives:**
  - Standard gossip: Slower (300-500ms), more bandwidth-efficient
  - Dedicated topic with separate mesh: More complex configuration
- **Trade-offs:** Higher bandwidth usage (acceptable for BFT signals which are small, ~512 bytes)

**Decision 2: On-Chain Reputation Integration**
- **Rationale:** Prevents Sybil attacks by prioritizing peers with proven on-chain track record. Aligns off-chain and on-chain incentives.
- **Alternatives:**
  - Pure GossipSub scoring: Vulnerable to new identity attacks
  - Manual peer whitelist: Not scalable, centralized
- **Trade-offs:** Requires chain sync (60s delay), additional complexity

**Decision 3: Strict Validation Mode**
- **Rationale:** Ensures all messages are signed by publisher, preventing impersonation and spam
- **Alternatives:**
  - Permissive mode: Faster but insecure
  - Anonymous mode: No sender attribution
- **Trade-offs:** Slightly higher CPU usage for signature verification (acceptable)

**Decision 4: 16MB Max Transmit Size**
- **Rationale:** Video chunks can be up to 15MB (45-second slot at high quality). 16MB provides headroom.
- **Alternatives:**
  - Default 4MB: Too small for video chunks
  - Unlimited: DoS risk
- **Trade-offs:** Larger messages increase memory usage and network bandwidth

## Risks & Mitigations

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| GossipSub performance degradation at scale | High | Medium | Monitor mesh size and message latency metrics, tune heartbeat interval and mesh parameters, hierarchical topic structure (T023) |
| Reputation oracle sync lag | Medium | Low | Cache on-chain reputation locally, 60s refresh is acceptable, fallback to default score (100) for unknown peers |
| Invalid message flood attack | High | Medium | Graylist peers with score < -100, rate limit messages per peer, slash stake on-chain for repeated violations |
| Topic subscription abuse | Medium | Low | Limit max subscribed topics per peer (e.g., 10), monitor subscription patterns, disconnect spammy peers |
| Message signature verification CPU bottleneck | Low | Low | Use Ed25519 (fast verification ~70k/sec), batch verification if needed, monitor CPU metrics |
| PeerId to AccountId mapping errors | Medium | Low | Implement robust conversion logic, validate mappings, log mismatches for debugging |

## Progress Log

### [2025-12-24T00:00:00Z] - Task Created

**Created By:** task-creator agent
**Reason:** User request for P2P networking layer tasks (Phase 1)
**Dependencies:** T021 (libp2p core), T003 (pallet-icn-reputation)
**Estimated Complexity:** Standard (well-defined GossipSub + custom reputation integration)

## Completion Checklist

**Code Complete:**
- [ ] All acceptance criteria met
- [ ] All test scenarios pass
- [ ] Code reviewed
- [ ] Documentation updated
- [ ] Clippy/linting passes
- [ ] Formatting applied
- [ ] No regression in existing tests

**Deployment Ready:**
- [ ] Integration tests pass on testnet
- [ ] Metrics verified in Grafana
- [ ] Logs structured and parseable
- [ ] Error paths tested
- [ ] Resource usage within limits
- [ ] Monitoring alerts configured

**Definition of Done:**
Task is complete when ALL acceptance criteria met, GossipSub configured with 5 topics, on-chain reputation integrated into peer scoring, integration tests demonstrate message propagation with reputation-based prioritization, and production-ready with metrics and graceful shutdown.
