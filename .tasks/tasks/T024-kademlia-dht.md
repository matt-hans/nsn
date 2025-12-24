---
id: T024
title: Kademlia DHT for Peer Discovery and Content Addressing
status: pending
priority: 2
agent: backend
dependencies: [T021]
blocked_by: []
created: 2025-12-24T00:00:00Z
updated: 2025-12-24T00:00:00Z
tags: [p2p, dht, kademlia, off-chain, phase1]

context_refs:
  - context/project.md
  - context/architecture.md
  - context/acceptance-templates.md

docs_refs:
  - ../docs/prd.md (§17.2, §18.1)
  - ../docs/architecture.md (§4.4.2)

est_tokens: 9500
actual_tokens: null
---

## Description

Implement Kademlia Distributed Hash Table (DHT) for decentralized peer discovery, content addressing, and provider records for erasure-coded video shards. Enables nodes to find each other without centralized bootstrap servers.

**Technical Approach:**
- Configure libp2p Kademlia DHT with ICN-specific protocol ID
- Use DHT for peer discovery (find other Directors, Super-Nodes, Relays)
- Store and retrieve provider records for video shard assignments
- Implement content addressing for shard hashes
- Bootstrap from known peers and DNS seeds
- Periodic routing table refresh and republish

**Integration Points:**
- Builds on T021 (libp2p core)
- Used by T023 (NAT traversal for relay discovery)
- Used by T025 (bootstrap protocol)
- Erasure shard discovery for Super-Nodes

## Business Context

**User Story:** As an ICN Super-Node, I want to discover which peers have specific video shards, so that I can retrieve content for distribution and handle shard redundancy.

**Why This Matters:**
- Decentralized peer discovery (no central directory)
- Content addressing enables efficient shard retrieval
- Routing table self-healing ensures network resilience

**What It Unblocks:**
- Peer discovery for new nodes joining network
- Shard assignment and retrieval
- Relay node discovery for NAT traversal

**Priority Justification:** Priority 2 (Important). Not on critical path for Phase 1 (can use bootstrap peers), but essential for decentralization and mainnet scale.

## Acceptance Criteria

- [ ] **DHT Initialization**: Kademlia DHT initialized with ICN protocol ID `/icn/kad/1.0.0`
- [ ] **Peer Discovery**: Node can discover peers via DHT `get_closest_peers` query
- [ ] **Provider Records**: Node can publish provider record for shard hash
- [ ] **Provider Lookup**: Node can query providers for specific shard hash
- [ ] **Content Addressing**: Shard hashes stored as CID (Content Identifier)
- [ ] **Bootstrap**: DHT bootstraps from known peers (bootstrap list)
- [ ] **Routing Table**: k-bucket routing table maintained (k=20)
- [ ] **Periodic Refresh**: Routing table refreshed every 5 minutes
- [ ] **Republish**: Provider records republished every 12 hours
- [ ] **Query Timeout**: DHT queries timeout after 10 seconds
- [ ] **Metrics Exposed**: Routing table size, query latency, provider record count

## Test Scenarios

**Test Case 1: Peer Discovery**
- Given: Three nodes (A, B, C) bootstrapped to DHT
- When: Node A queries `get_closest_peers(random_peer_id)`
- Then:
  - Nodes B and C are returned in results
  - Results sorted by XOR distance to target
  - Query completes within 5 seconds

**Test Case 2: Provider Record Publication**
- Given: Super-Node A has shard with hash `0xABCD`
- When: Node A publishes provider record `put_provider(0xABCD)`
- Then:
  - Provider record stored in DHT
  - Record includes Node A's PeerId and multiaddrs
  - Record TTL = 12 hours
  - Metrics show provider_records_published +1

**Test Case 3: Provider Record Lookup**
- Given: Node A published provider record for shard `0xABCD`
- When: Node B queries `get_providers(0xABCD)`
- Then:
  - Node A returned as provider
  - Node B can connect to Node A via returned multiaddr
  - Query latency < 2 seconds

**Test Case 4: DHT Bootstrap**
- Given: New node with bootstrap list of 3 peers
- When: Node starts and bootstraps to DHT
- Then:
  - Connects to bootstrap peers
  - Populates routing table via FIND_NODE queries
  - Routing table size > 10 within 30 seconds

**Test Case 5: Routing Table Refresh**
- Given: Node with routing table of 50 peers
- When: 5 minutes elapse (refresh interval)
- Then:
  - FIND_NODE queries sent to random targets
  - Stale peers removed (unreachable)
  - New peers discovered
  - Routing table size maintained

**Test Case 6: Provider Record Expiry**
- Given: Node A published provider record at T=0
- When: 13 hours elapse (> 12h TTL)
- Then:
  - Provider record expires
  - Queries for shard return empty or stale results
  - Node A republishes record (if still providing)

**Test Case 7: Query Timeout**
- Given: DHT query to unreachable target
- When: Query initiated
- Then:
  - Query times out after 10 seconds
  - Error returned: "DHT query timeout"
  - Metrics show dht_query_timeouts +1

**Test Case 8: k-Bucket Replacement**
- Given: k-bucket full (20 peers) with one stale peer
- When: New responsive peer discovered
- Then:
  - Stale peer pinged
  - If stale peer unresponsive, replaced by new peer
  - k-bucket size remains 20

## Technical Implementation

**Required Components:**

```
off-chain/src/p2p/
├── kademlia.rs             # Kademlia DHT configuration
├── provider_records.rs     # Provider record publishing and lookup
├── routing_table.rs        # Routing table maintenance (optional wrapper)
└── content_id.rs           # CID generation for shard hashes

off-chain/tests/
└── integration_kademlia.rs # Kademlia integration tests
```

**Key Rust Modules:**

```rust
// src/p2p/kademlia.rs
use libp2p::kad::{
    Kademlia, KademliaConfig, KademliaEvent,
    store::MemoryStore, QueryResult, GetProvidersOk,
};
use libp2p::kad::record::Key;
use std::time::Duration;

pub fn build_kademlia(local_peer_id: PeerId) -> Kademlia<MemoryStore> {
    let mut config = KademliaConfig::default();
    config.set_protocol_names(vec![b"/icn/kad/1.0.0".to_vec()]);
    config.set_query_timeout(Duration::from_secs(10));
    config.set_replication_factor(20);  // k-bucket size
    config.set_publication_interval(Some(Duration::from_secs(12 * 3600)));  // 12h
    config.set_record_ttl(Some(Duration::from_secs(12 * 3600)));

    let store = MemoryStore::new(local_peer_id);
    Kademlia::with_config(local_peer_id, store, config)
}

pub struct KademliaService {
    kademlia: Kademlia<MemoryStore>,
    bootstrap_peers: Vec<(PeerId, Multiaddr)>,
    metrics: Arc<KademliaMetrics>,
}

impl KademliaService {
    pub fn new(local_peer_id: PeerId, bootstrap_peers: Vec<(PeerId, Multiaddr)>) -> Self {
        let kademlia = build_kademlia(local_peer_id);

        Self {
            kademlia,
            bootstrap_peers,
            metrics: Arc::new(KademliaMetrics::new()),
        }
    }

    pub fn bootstrap(&mut self) -> Result<(), Error> {
        // Add bootstrap peers to routing table
        for (peer_id, addr) in &self.bootstrap_peers {
            self.kademlia.add_address(peer_id, addr.clone());
        }

        // Bootstrap DHT
        self.kademlia.bootstrap().map_err(|e| Error::BootstrapFailed(e))?;
        tracing::info!("DHT bootstrap initiated with {} peers", self.bootstrap_peers.len());

        Ok(())
    }

    pub async fn start_periodic_refresh(&mut self) {
        let mut interval = tokio::time::interval(Duration::from_secs(300)); // 5 minutes

        loop {
            interval.tick().await;
            self.refresh_routing_table();
        }
    }

    fn refresh_routing_table(&mut self) {
        // Generate random PeerId and query for closest peers
        let random_peer = PeerId::random();
        let query_id = self.kademlia.get_closest_peers(random_peer);
        tracing::debug!("Routing table refresh: query_id={:?}", query_id);
        self.metrics.routing_table_refreshes.inc();
    }

    pub fn handle_event(&mut self, event: KademliaEvent) {
        match event {
            KademliaEvent::OutboundQueryProgressed { result, .. } => {
                self.handle_query_result(result);
            }
            KademliaEvent::RoutingUpdated { peer, .. } => {
                tracing::debug!("Routing table updated: added {}", peer);
                self.metrics.routing_table_size.set(self.kademlia.iter_peers().count() as f64);
            }
            _ => {}
        }
    }

    fn handle_query_result(&mut self, result: QueryResult) {
        match result {
            QueryResult::GetClosestPeers(Ok(peers)) => {
                tracing::info!("Found {} closest peers", peers.peers.len());
                self.metrics.peer_discovery_success.inc();
            }
            QueryResult::GetProviders(Ok(GetProvidersOk { key, providers, .. })) => {
                tracing::info!("Found {} providers for key {:?}", providers.len(), key);
                self.metrics.provider_lookups_success.inc();
            }
            QueryResult::GetClosestPeers(Err(e)) => {
                tracing::warn!("Get closest peers failed: {:?}", e);
                self.metrics.peer_discovery_failures.inc();
            }
            QueryResult::GetProviders(Err(e)) => {
                tracing::warn!("Get providers failed: {:?}", e);
                self.metrics.provider_lookups_failures.inc();
            }
            _ => {}
        }
    }
}

// src/p2p/provider_records.rs
use libp2p::kad::record::Key;
use sp_core::H256;

pub struct ProviderRecordService {
    kademlia: Arc<Mutex<Kademlia<MemoryStore>>>,
}

impl ProviderRecordService {
    pub async fn publish_shard(&self, shard_hash: H256) -> Result<(), Error> {
        let key = Key::new(&shard_hash.as_bytes());

        let mut kad = self.kademlia.lock().await;
        kad.start_providing(key.clone())
            .map_err(|e| Error::ProviderPublishFailed(e))?;

        tracing::info!("Published provider record for shard {}", hex::encode(shard_hash));
        Ok(())
    }

    pub async fn lookup_shard_providers(&self, shard_hash: H256) -> Result<Vec<PeerId>, Error> {
        let key = Key::new(&shard_hash.as_bytes());

        let mut kad = self.kademlia.lock().await;
        let query_id = kad.get_providers(key.clone());

        tracing::debug!("Querying providers for shard {}: query_id={:?}", hex::encode(shard_hash), query_id);

        // Wait for query result (in real impl, use futures channel)
        // For now, return placeholder
        Ok(vec![])
    }

    pub async fn republish_all_shards(&self) -> Result<(), Error> {
        // Re-publish all provider records (called every 12 hours)
        let mut kad = self.kademlia.lock().await;

        for record in kad.store_mut().records() {
            kad.start_providing(record.key.clone())
                .map_err(|e| Error::ProviderPublishFailed(e))?;
        }

        tracing::info!("Republished all provider records");
        Ok(())
    }
}

// Example integration with P2pService
impl P2pService {
    pub async fn new_with_dht(config: P2pConfig, bootstrap_peers: Vec<(PeerId, Multiaddr)>) -> Result<Self, Error> {
        let keypair = load_or_generate_keypair(&config.keypair_path)?;
        let local_peer_id = PeerId::from(keypair.public());

        let transport = build_transport(&keypair)?;
        let gossipsub = build_gossipsub(&keypair, reputation_oracle.clone());
        let mut kademlia = KademliaService::new(local_peer_id, bootstrap_peers.clone());

        // Bootstrap DHT
        kademlia.bootstrap()?;

        let behaviour = Behaviour {
            gossipsub,
            kademlia: kademlia.kademlia,
            // ... other behaviors
        };

        let swarm = SwarmBuilder::with_tokio_executor(
            transport,
            behaviour,
            local_peer_id,
        ).build();

        // Spawn periodic refresh task
        let mut kad_clone = kademlia.clone();
        tokio::spawn(async move {
            kad_clone.start_periodic_refresh().await;
        });

        Ok(Self {
            swarm,
            config,
            kademlia: Some(kademlia),
        })
    }

    pub async fn publish_shard_provider(&mut self, shard_hash: H256) -> Result<(), Error> {
        if let Some(kad) = &mut self.kademlia {
            kad.publish_shard(shard_hash).await?;
        }
        Ok(())
    }

    pub async fn find_shard_providers(&mut self, shard_hash: H256) -> Result<Vec<PeerId>, Error> {
        if let Some(kad) = &mut self.kademlia {
            kad.lookup_shard_providers(shard_hash).await
        } else {
            Ok(vec![])
        }
    }
}
```

**Validation Commands:**

```bash
# Build with Kademlia DHT
cargo build --release -p icn-off-chain --features kademlia

# Run unit tests
cargo test -p icn-off-chain kademlia::

# Run integration tests (requires 3+ nodes)
cargo test --test integration_kademlia -- --nocapture

# Start node with DHT enabled
RUST_LOG=debug cargo run --release -- \
  --port 9000 \
  --bootstrap-peers /ip4/1.2.3.4/tcp/9000/p2p/12D3KooW...

# Check DHT metrics
curl http://localhost:9100/metrics | grep kademlia

# Query DHT for providers
cargo run --example dht_query -- --shard-hash 0xabcd...
```

**Code Patterns:**
- Use `KademliaEvent` for async query results
- Periodic refresh via tokio interval
- Metrics for routing table size and query success rates
- Structured logging with query IDs

## Dependencies

**Hard Dependencies** (must be complete first):
- [T021] libp2p Core Setup - provides transport and PeerId

**Soft Dependencies:**
- [T025] Bootstrap Protocol - provides bootstrap peer list

**External Dependencies:**
- libp2p-kad (Kademlia DHT)
- CID library for content addressing

## Design Decisions

**Decision 1: MemoryStore vs PersistentStore**
- **Rationale:** MemoryStore is simpler and sufficient for peer discovery. Provider records are ephemeral (12h TTL) and republished.
- **Alternatives:**
  - RocksDB-backed store: More durable but adds complexity
  - Custom store: Not needed
- **Trade-offs:** MemoryStore loses data on restart, but this is acceptable (bootstrap repopulates routing table)

**Decision 2: 12-Hour Provider Record TTL**
- **Rationale:** Balance between republish overhead and availability. Shards are long-lived (7 days default retention).
- **Alternatives:**
  - 24h TTL: Fewer republishes but higher risk of stale data
  - 6h TTL: More current but higher network overhead
- **Trade-offs:** 12h is standard for libp2p/IPFS

**Decision 3: k=20 Replication Factor**
- **Rationale:** Standard Kademlia configuration, provides good redundancy and query performance
- **Alternatives:**
  - k=16: Less redundant but acceptable
  - k=32: More redundant but larger routing tables
- **Trade-offs:** k=20 is battle-tested in libp2p ecosystem

**Decision 4: 5-Minute Routing Table Refresh**
- **Rationale:** Balances between up-to-date routing table and query overhead
- **Alternatives:**
  - 10 min: Less overhead but staler data
  - 1 min: Very current but high query volume
- **Trade-offs:** 5 min is reasonable for dynamic networks

## Risks & Mitigations

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| Eclipse attack (malicious peers flood routing table) | High | Medium | Require minimum reputation for routing table entry, limit inbound connections, diversity checks |
| DHT routing table partitions | High | Low | Multiple bootstrap peers, periodic full refresh, cross-region connectivity checks |
| Provider record spam | Medium | Medium | Rate limit provider publications per peer, require stake for provider status, monitor metrics |
| Query amplification DDoS | Medium | Low | Rate limit queries per peer, query timeout enforcement, graylist abusive peers |
| Stale provider records | Low | Medium | 12h TTL with automatic republish, background validation of providers |

## Progress Log

### [2025-12-24T00:00:00Z] - Task Created

**Created By:** task-creator agent
**Reason:** User request for P2P networking layer tasks (Phase 1)
**Dependencies:** T021 (libp2p core)
**Estimated Complexity:** Standard (well-defined Kademlia DHT, libp2p implementation available)

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
Task is complete when ALL acceptance criteria met, Kademlia DHT configured with ICN protocol ID, peer discovery and provider records functional, integration tests demonstrate multi-node DHT queries, and production-ready with metrics and periodic refresh.
