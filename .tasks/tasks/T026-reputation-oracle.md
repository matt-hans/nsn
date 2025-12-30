---
id: T026
title: Reputation Oracle (On-Chain Sync for P2P Scoring)
status: pending
priority: 2
agent: backend
dependencies: [T003, T021]
blocked_by: []
created: 2025-12-24T00:00:00Z
updated: 2025-12-24T00:00:00Z
tags: [p2p, reputation, on-chain-sync, off-chain, phase1]

context_refs:
  - context/project.md
  - context/architecture.md
  - context/acceptance-templates.md

docs_refs:
  - ../docs/prd.md (§3.2, §17.3)
  - ../docs/architecture.md (§17.3)

est_tokens: 9000
actual_tokens: null
---

## Description

Implement ReputationOracle service that synchronizes on-chain reputation scores from pallet-nsn-reputation to local cache for integration with GossipSub peer scoring. Runs background sync every 60 seconds to keep scores current.

**Technical Approach:**
- Use subxt client to query pallet-nsn-reputation storage (ReputationScores)
- Cache reputation scores locally (HashMap<PeerId, u64>)
- Background tokio task for periodic sync (every 60 seconds)
- Map AccountId32 to PeerId (same Ed25519 keypair)
- Expose get_reputation(PeerId) -> u64 for GossipSub scoring
- Handle chain disconnections gracefully (use stale cache)

**Integration Points:**
- Builds on T003 (pallet-nsn-reputation)
- Used by T022 (GossipSub reputation-integrated scoring)
- Requires chain client (subxt)

## Business Context

**User Story:** As a GossipSub peer scoring algorithm, I want to factor in on-chain reputation, so that I prioritize high-quality peers and deprioritize malicious actors.

**Why This Matters:**
- Aligns off-chain and on-chain incentives (good reputation = better P2P connectivity)
- Prevents Sybil attacks (new identities start with low reputation)
- Reinforces economic security (slashed stake = damaged reputation = network isolation)

**What It Unblocks:**
- T022 (GossipSub with reputation scoring)
- BFT consensus (directors with high reputation preferred)
- Content distribution (super-nodes with high reputation preferred)

**Priority Justification:** Priority 2 (Important). Not critical for Phase 1 (can use default GossipSub scoring), but essential for mainnet security.

## Acceptance Criteria

- [ ] **Chain Client**: subxt OnlineClient connected to NSN Chain RPC
- [ ] **Storage Query**: Queries pallet-nsn-reputation::ReputationScores storage
- [ ] **Cache Update**: Updates local cache with fetched scores
- [ ] **PeerId Mapping**: Maps AccountId32 to PeerId correctly
- [ ] **Background Sync**: Runs sync loop every 60 seconds in background task
- [ ] **Get Reputation**: `get_reputation(PeerId)` returns cached score or default (100)
- [ ] **Register Peer**: `register_peer(AccountId32, PeerId)` updates mapping
- [ ] **Graceful Degradation**: Returns default score (100) if sync fails or peer unknown
- [ ] **Connection Recovery**: Reconnects to chain RPC if disconnected
- [ ] **Metrics Exposed**: Sync success/failures, cache size, query latency

## Test Scenarios

**Test Case 1: Initial Sync**
- Given: Reputation oracle started with chain RPC URL
- When: First sync runs
- Then:
  - Queries ReputationScores storage
  - Fetches all account scores (e.g., 50 accounts)
  - Updates cache with 50 entries
  - Metrics show cache_size=50

**Test Case 2: Reputation Query (Known Peer)**
- Given: Cache has entry for PeerId A with reputation 850
- When: `get_reputation(PeerId A)` called
- Then:
  - Returns 850
  - No RPC query (cached)
  - Query latency < 1ms

**Test Case 3: Reputation Query (Unknown Peer)**
- Given: Cache has no entry for PeerId B
- When: `get_reputation(PeerId B)` called
- Then:
  - Returns default 100
  - Warning logged: "Reputation unknown for peer B, using default"
  - Metrics show unknown_peer_queries +1

**Test Case 4: Periodic Sync**
- Given: Oracle running for 2 minutes
- When: Sync loops execute
- Then:
  - Sync runs at T=0, T=60s, T=120s
  - Cache updated each time
  - Metrics show sync_runs=3

**Test Case 5: Reputation Score Update**
- Given: Account A has reputation 500 at T=0
- When: Account A's reputation increases to 600 (on-chain event at T=30s)
  And sync runs at T=60s
- Then:
  - Cache updated to 600
  - Subsequent queries return 600

**Test Case 6: PeerId Registration**
- Given: New peer joins with AccountId A and PeerId P
- When: `register_peer(AccountId A, PeerId P)` called
- Then:
  - Mapping stored in account_to_peer_map
  - Next sync includes PeerId P in cache
  - `get_reputation(PeerId P)` returns AccountId A's score

**Test Case 7: Chain Disconnection**
- Given: Oracle syncing normally
- When: Chain RPC becomes unreachable
- Then:
  - Sync fails with error
  - Cache retains last known values
  - Queries still return cached scores
  - Metrics show sync_failures +1
  - Oracle retries next interval

**Test Case 8: Chain Reconnection**
- Given: Chain RPC was unreachable for 5 minutes
- When: Chain RPC becomes reachable again
- Then:
  - Next sync succeeds
  - Cache updated with latest scores
  - Metrics show sync_success +1

## Reference Documentation
- [Subxt Documentation](https://docs.rs/subxt/latest/subxt/)
- [Substrate Storage Queries](https://docs.substrate.io/reference/command-line-tools/subkey/#inspect-storage)

## Technical Implementation

**Required Components:**

```
off-chain/src/reputation/
├── mod.rs                  # ReputationOracle struct and API
├── sync.rs                 # Background sync task
├── cache.rs                # Reputation cache (HashMap)
└── mapping.rs              # AccountId ↔ PeerId mapping

off-chain/tests/
└── integration_reputation_oracle.rs # Integration tests
```

**Key Rust Modules:**

```rust
// src/reputation/mod.rs
use subxt::{OnlineClient, PolkadotConfig};
use sp_core::crypto::AccountId32;
use libp2p::PeerId;
use std::collections::HashMap;
use tokio::sync::RwLock;

pub struct ReputationOracle {
    cache: Arc<RwLock<HashMap<PeerId, u64>>>,
    chain_client: OnlineClient<PolkadotConfig>,
    account_to_peer_map: Arc<RwLock<HashMap<AccountId32, PeerId>>>,
    metrics: Arc<ReputationMetrics>,
}

impl ReputationOracle {
    pub async fn new(rpc_url: &str) -> Result<Self, Error> {
        let chain_client = OnlineClient::<PolkadotConfig>::from_url(rpc_url)
            .await
            .map_err(|e| Error::ChainConnectionFailed(e.to_string()))?;

        tracing::info!("Connected to chain RPC: {}", rpc_url);

        Ok(Self {
            cache: Arc::new(RwLock::new(HashMap::new())),
            chain_client,
            account_to_peer_map: Arc::new(RwLock::new(HashMap::new())),
            metrics: Arc::new(ReputationMetrics::new()),
        })
    }

    pub fn get_reputation(&self, peer_id: &PeerId) -> u64 {
        match self.cache.blocking_read().get(peer_id) {
            Some(&score) => score,
            None => {
                tracing::debug!("Reputation unknown for peer {}, using default", peer_id);
                self.metrics.unknown_peer_queries.inc();
                100  // Default score for unknown peers
            }
        }
    }

    pub async fn register_peer(&self, account: AccountId32, peer_id: PeerId) {
        self.account_to_peer_map.write().await.insert(account, peer_id);
        tracing::debug!("Registered peer mapping: {} -> {}", hex::encode(account), peer_id);
    }

    pub async fn sync_loop(&self) {
        let mut interval = tokio::time::interval(Duration::from_secs(60));

        loop {
            interval.tick().await;

            match self.fetch_all_reputations().await {
                Ok(count) => {
                    tracing::debug!("Synced {} reputation scores from chain", count);
                    self.metrics.sync_success.inc();
                }
                Err(e) => {
                    tracing::error!("Reputation sync failed: {}", e);
                    self.metrics.sync_failures.inc();
                }
            }
        }
    }

    async fn fetch_all_reputations(&self) -> Result<usize, Error> {
        use nsn_reputation_pallet as pallet;

        // Query storage iterator for ReputationScores
        let storage_query = pallet::storage().reputation_scores_root();

        let mut iter = self.chain_client
            .storage()
            .at_latest()
            .await
            .map_err(|e| Error::StorageQueryFailed(e.to_string()))?
            .iter(storage_query)
            .await
            .map_err(|e| Error::StorageIterFailed(e.to_string()))?;

        let mut new_cache = HashMap::new();
        let mut count = 0;

        while let Some(result) = iter.next().await {
            let (key, value) = result
                .map_err(|e| Error::StorageIterFailed(e.to_string()))?;

            let account = key.0;  // AccountId32
            let score = value;     // ReputationScore

            // Map AccountId to PeerId
            if let Some(peer_id) = self.account_to_peer(&account).await {
                new_cache.insert(peer_id, score.total());
                count += 1;
            } else {
                tracing::trace!("No PeerId mapping for account {}", hex::encode(&account));
            }
        }

        // Update cache atomically
        *self.cache.write().await = new_cache;
        self.metrics.cache_size.set(count as f64);

        Ok(count)
    }

    async fn account_to_peer(&self, account: &AccountId32) -> Option<PeerId> {
        self.account_to_peer_map.read().await.get(account).copied()
    }
}

// src/reputation/metrics.rs
use prometheus::{IntCounter, Gauge};

pub struct ReputationMetrics {
    pub sync_success: IntCounter,
    pub sync_failures: IntCounter,
    pub cache_size: Gauge,
    pub unknown_peer_queries: IntCounter,
}

impl ReputationMetrics {
    pub fn new() -> Self {
        Self {
            sync_success: IntCounter::new(
                "nsn_reputation_sync_success_total",
                "Total successful reputation syncs"
            ).unwrap(),
            sync_failures: IntCounter::new(
                "nsn_reputation_sync_failures_total",
                "Total failed reputation syncs"
            ).unwrap(),
            cache_size: Gauge::new(
                "nsn_reputation_cache_size",
                "Number of cached reputation scores"
            ).unwrap(),
            unknown_peer_queries: IntCounter::new(
                "nsn_reputation_unknown_peer_queries_total",
                "Queries for peers with unknown reputation"
            ).unwrap(),
        }
    }
}

// Example integration with P2pService
impl P2pService {
    pub async fn new_with_reputation_oracle(
        config: P2pConfig,
        rpc_url: &str,
    ) -> Result<Self, Error> {
        let keypair = load_or_generate_keypair(&config.keypair_path)?;
        let local_peer_id = PeerId::from(keypair.public());

        // Create reputation oracle
        let reputation_oracle = Arc::new(ReputationOracle::new(rpc_url).await?);

        // Register local peer mapping
        let account_id = peer_id_to_account_id(&local_peer_id);
        reputation_oracle.register_peer(account_id, local_peer_id.clone()).await;

        // Spawn background sync task
        let oracle_clone = reputation_oracle.clone();
        tokio::spawn(async move {
            oracle_clone.sync_loop().await;
        });

        // Build GossipSub with reputation oracle
        let gossipsub = build_gossipsub(&keypair, reputation_oracle.clone());

        // ... rest of P2pService initialization

        Ok(Self {
            // ...
            reputation_oracle: Some(reputation_oracle),
        })
    }
}
```

**Validation Commands:**

```bash
# Build reputation oracle module
cargo build --release -p nsn-off-chain --features reputation-oracle

# Run unit tests
cargo test -p nsn-off-chain reputation::

# Run integration tests (requires running chain)
cargo test --test integration_reputation_oracle -- --nocapture

# Start node with reputation oracle
RUST_LOG=debug cargo run --release -- \
  --port 9000 \
  --rpc-url ws://localhost:9944

# Check reputation metrics
curl http://localhost:9100/metrics | grep reputation

# Query specific peer reputation
cargo run --example query_reputation -- --peer-id 12D3KooW...
```

**Code Patterns:**
- Use subxt for type-safe chain queries
- RwLock for concurrent cache access
- Background tokio task for sync loop
- Prometheus metrics for observability
- Graceful degradation (default score) on errors

## Dependencies

**Hard Dependencies** (must be complete first):
- [T003] pallet-nsn-reputation - on-chain reputation storage
- [T021] libp2p Core Setup - PeerId and AccountId mapping

**Soft Dependencies:**
- [T022] GossipSub Configuration - uses reputation oracle for scoring (can be integrated later)

**External Dependencies:**
- subxt 0.34+ (chain client)
- tokio (async runtime)
- prometheus (metrics)

## Design Decisions

**Decision 1: 60-Second Sync Interval**
- **Rationale:** Balance between freshness and RPC load. Reputation changes are not time-critical (minutes-scale is acceptable).
- **Alternatives:**
  - 10s: More current but 6× RPC load
  - 5 min: Lower load but staler data
- **Trade-offs:** 60s is reasonable for reputation (not real-time data)

**Decision 2: Default Score 100 for Unknown Peers**
- **Rationale:** Neutral starting point for new peers. Not punished, but not rewarded.
- **Alternatives:**
  - 0 score: Too harsh, prevents new nodes from joining
  - 1000 score: Too generous, Sybil attack risk
- **Trade-offs:** 100 is standard "unknown" score in many reputation systems

**Decision 3: Local Cache (HashMap)**
- **Rationale:** Fast lookups (O(1)), no RPC queries during GossipSub scoring (latency-sensitive)
- **Alternatives:**
  - Query chain on every get_reputation: Too slow (10-100ms per query)
  - Persistent cache (RocksDB): Overkill for ephemeral data
- **Trade-offs:** Cache uses memory (~50KB for 1000 peers), but this is negligible

**Decision 4: Background Sync Task**
- **Rationale:** Decouples sync from main event loop, prevents blocking P2P operations
- **Alternatives:**
  - Sync on-demand: Slower, unpredictable latency
  - Subscription (chain events): More complex, requires event parsing
- **Trade-offs:** 60s sync lag, but acceptable for reputation use case

## Risks & Mitigations

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| Chain RPC unavailable | Medium | Medium | Use cached scores, retry with exponential backoff, fallback to default scores, monitor metrics |
| Cache inconsistency (reputation changes missed) | Low | Low | 60s sync interval catches most changes, accept eventual consistency model |
| PeerId mapping incomplete | Medium | Medium | Require nodes to register mapping on startup, log unmapped accounts for debugging |
| Memory usage (large cache) | Low | Low | Monitor cache_size metric, consider LRU eviction for >10k entries |
| Sync latency spike (slow RPC) | Low | Medium | Timeout queries after 30s, log slow queries, consider faster RPC provider |

## Progress Log

### [2025-12-24T00:00:00Z] - Task Created

**Created By:** task-creator agent
**Reason:** User request for P2P networking layer tasks (Phase 1)
**Dependencies:** T003 (pallet-nsn-reputation), T021 (libp2p core)
**Estimated Complexity:** Standard (chain sync, caching, background task)

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
Task is complete when ALL acceptance criteria met, reputation oracle syncs from chain every 60 seconds, cache updated with latest scores, integration tests demonstrate chain sync and cache updates, and production-ready with metrics and graceful degradation on chain failures.
