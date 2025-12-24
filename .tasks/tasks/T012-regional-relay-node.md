---
id: T012
title: Regional Relay Node Implementation (Tier 2 Distribution)
status: pending
priority: 3
agent: backend
dependencies: [T001, T002, T011]
blocked_by: []
created: 2025-12-24T00:00:00Z
updated: 2025-12-24T00:00:00Z
tags: [off-chain, relay, rust, distribution, phase1]

context_refs:
  - context/project.md
  - context/architecture.md
  - context/acceptance-templates.md

docs_refs:
  - docs/prd.md#section-4.1
  - docs/architecture.md#section-4.3

est_tokens: 8000
actual_tokens: null
---

## Description

Implement the Regional Relay Node, ICN's Tier 2 city-level distribution layer. Regional Relays sit between Tier 1 Super-Nodes and Tier 3 Viewers, providing low-latency content delivery through geographic proximity and efficient caching.

Regional Relays perform four core functions:
1. **Content Caching**: Cache video chunks from Super-Nodes (1TB capacity) with LRU eviction
2. **Viewer Distribution**: Serve video streams to viewers via QUIC transport
3. **Auto-Assignment**: Self-assign to regions based on latency clustering (ping-based)
4. **Failover**: Redirect viewers to Super-Nodes if local cache miss or relay failure

Relays operate with minimum 10 ICN stake, are distributed at city-level granularity (target: 100+ relays across 50+ cities), and earn modest rewards for bandwidth contribution. Unlike Super-Nodes (regional anchors), relays are lightweight and horizontally scalable.

**Technical Approach:**
- Rust 1.75+ with Tokio async runtime
- LRU cache via `lru` crate (1TB max, TTL-based eviction)
- rust-libp2p for P2P (Kademlia DHT for shard discovery, QUIC for content transfer)
- QUIC server for viewer connections (WebTransport-compatible for browser clients)
- subxt for minimal chain interaction (stake verification only)

**Integration Points:**
- Queries Kademlia DHT for shard manifests (published by Super-Nodes in T011)
- Fetches shards from Super-Nodes via QUIC
- Serves viewers via QUIC (WebTransport for Tauri clients in T013)
- Publishes relay availability to DHT (viewers discover nearby relays)

## Business Context

**User Story:** As a Regional Relay operator, I want a lightweight caching service that automatically discovers content from Super-Nodes and serves it to local viewers, so that I can earn rewards with minimal hardware investment (no 10TB storage required).

**Why This Matters:** Regional Relays reduce latency for viewers by providing city-level caching. Without Tier 2, viewers would need to fetch content from geographically distant Super-Nodes (100-300ms latency). Relays bring this down to 10-30ms (intra-city).

**What It Unblocks:**
- T013 (Viewer Clients) can connect to low-latency local relays
- Horizontal scaling of distribution capacity (add more relays per city)
- Geographic load balancing (relays auto-discover viewers in their region)
- Graceful degradation (viewers fall back to Super-Nodes if relay unavailable)

**Priority Justification:** Priority 3 (Enhancement) - Nice to have for mainnet but not critical for Moonriver MVP. Early testnet can rely on Super-Node direct connections. Relays optimize for <45s glass-to-glass latency SLA.

## Acceptance Criteria

- [ ] Binary compiles with `cargo build --release -p icn-relay`
- [ ] Latency-based region detection: relay pings Super-Nodes, selects closest region
- [ ] Kademlia DHT query retrieves shard manifest for video CID
- [ ] QUIC client fetches shards from Super-Node (upstream connection)
- [ ] LRU cache stores shards with 1TB capacity limit
- [ ] Cache hit serves shard from local disk (<50ms latency)
- [ ] Cache miss fetches from upstream Super-Node and caches result
- [ ] QUIC server accepts viewer connections on port 9003
- [ ] Viewer shard requests served within 100ms (cache hit) or 500ms (cache miss + upstream fetch)
- [ ] Relay publishes availability to DHT (key: region, value: relay multiaddrs)
- [ ] Graceful shutdown with cache flush (persists cached shards to disk)
- [ ] Configuration loaded from TOML (cache size, upstream super-nodes, region override)
- [ ] Prometheus metrics exposed on port 9103 (cache_hits, cache_misses, bytes_served, viewer_count)
- [ ] Unit tests for cache eviction, latency measurement, shard fetching
- [ ] Integration test: fetches shard from mock Super-Node, serves to mock viewer

## Test Scenarios

**Test Case 1: Latency-Based Region Detection**
- Given: Relay starts without explicit region configuration
  And: Super-Nodes available in regions: NA-WEST (50ms), EU-WEST (150ms), APAC (200ms)
- When: Relay pings all Super-Nodes
- Then: NA-WEST selected as assigned region (lowest latency)
  And: Relay publishes availability to DHT: `dht.put(key: "relay:NA-WEST", value: [relay_multiaddr])`

**Test Case 2: Cache Hit (Shard Already Cached)**
- Given: Shard for video CID 0xABCD is cached locally
  And: Viewer requests shard: `GET /shards/0xABCD/shard_5.bin`
- When: Relay checks cache
- Then: Shard served from disk cache within 50ms
  And: Metrics: `cache_hits_total +1`
  And: No upstream Super-Node request

**Test Case 3: Cache Miss (Fetch from Super-Node)**
- Given: Shard for video CID 0xDEF not in cache
  And: Viewer requests shard: `GET /shards/0xDEF/shard_7.bin`
- When: Relay checks cache (miss)
  And: Queries DHT for shard manifest
  And: Fetches shard from Super-Node via QUIC
- Then: Shard downloaded in 200ms (from Super-Node)
  And: Shard added to cache
  And: Shard served to viewer
  And: Metrics: `cache_misses_total +1`, `upstream_fetches_total +1`

**Test Case 4: LRU Cache Eviction**
- Given: Cache is at 1TB capacity (full)
  And: New shard (8MB) needs to be cached
- When: Cache eviction triggered
- Then: Least recently used shard(s) evicted (total ≥8MB freed)
  And: New shard added to cache
  And: Evicted shards deleted from disk
  And: Metrics: `cache_evictions_total +1`

**Test Case 5: Upstream Super-Node Failure (Failover)**
- Given: Relay's primary Super-Node unreachable (offline/network partition)
  And: DHT has 2 backup Super-Nodes in same region
- When: Relay attempts to fetch shard and primary fails
- Then: Relay retries with backup Super-Node from DHT
  And: Shard fetched successfully from backup
  And: Error logged for primary Super-Node
  And: Health check scheduled for primary (exponential backoff)

**Test Case 6: Viewer Connection Surge (Bandwidth Limit)**
- Given: Relay serving 50 concurrent viewers
  And: Bandwidth limit is 100 Mbps
- When: 10 more viewers attempt to connect (total 60)
- Then: Relay accepts connections but rate limits per-viewer bandwidth
  And: Each viewer gets ~1.66 Mbps (100 / 60)
  And: Playback remains smooth (1080p@24fps needs ~5 Mbps, 480p@24fps needs ~1 Mbps)
  And: Relay suggests viewers downgrade to 480p via adaptive bitrate signaling

**Test Case 7: Cache Persistence Across Restarts**
- Given: Relay has cached 50GB of shards
  And: Relay receives SIGTERM (graceful shutdown)
- When: Cache flush triggered
- Then: Cache metadata written to `cache_manifest.json` (CID, shard IDs, LRU timestamps)
  And: Cached shards remain on disk
  And: After restart, cache reloaded from manifest
  And: Cache state preserved (no re-fetch from Super-Nodes)

## Technical Implementation

**Required Components:**
- `icn-relay/src/main.rs` - Binary entrypoint with CLI args (--config, --cache-path, --region)
- `icn-relay/src/config.rs` - Configuration struct with TOML deserialization
- `icn-relay/src/latency_detector.rs` - Ping-based region selection (ICMP or TCP handshake timing)
- `icn-relay/src/cache.rs` - LRU cache implementation with disk persistence
- `icn-relay/src/upstream_client.rs` - QUIC client for fetching shards from Super-Nodes
- `icn-relay/src/quic_server.rs` - QUIC server for serving viewers (WebTransport-compatible)
- `icn-relay/src/p2p_service.rs` - libp2p Kademlia DHT for shard discovery and relay advertising
- `icn-relay/src/metrics.rs` - Prometheus metrics (cache stats, bandwidth, viewer count)
- `icn-relay/src/health_check.rs` - Periodic upstream Super-Node health checks

**Validation Commands:**
```bash
# Build
cargo build --release -p icn-relay

# Run unit tests
cargo test -p icn-relay --lib

# Run integration tests
cargo test -p icn-relay --features integration-tests

# Clippy
cargo clippy -p icn-relay -- -D warnings

# Format check
cargo fmt -p icn-relay -- --check

# Run relay
./target/release/icn-relay \
  --config config/relay.toml \
  --cache-path /mnt/icn-cache \
  --region NA-WEST  # Or auto-detect

# Check metrics
curl http://localhost:9103/metrics | grep icn_relay_

# Test cache hit/miss
curl http://localhost:9003/shards/<CID>/shard_0.bin
```

**Code Patterns:**
```rust
// Latency-based region detection
use std::time::{Duration, Instant};

pub async fn detect_region(super_nodes: &[Multiaddr]) -> Result<Region> {
    let mut latencies = Vec::new();

    for addr in super_nodes {
        let start = Instant::now();
        let result = tokio::time::timeout(
            Duration::from_secs(2),
            TcpStream::connect(addr.to_socket_addr())
        ).await;

        if let Ok(Ok(_)) = result {
            let latency = start.elapsed();
            latencies.push((addr.region(), latency));
        }
    }

    latencies.sort_by_key(|(_, lat)| *lat);
    Ok(latencies.first().map(|(r, _)| r.clone())
        .unwrap_or(Region::default()))
}

// LRU cache with disk persistence
use lru::LruCache;
use std::num::NonZeroUsize;

pub struct ShardCache {
    cache: LruCache<ShardKey, PathBuf>,
    cache_dir: PathBuf,
    max_size_bytes: u64,
    current_size: u64,
}

impl ShardCache {
    pub fn new(cache_dir: PathBuf, max_size_bytes: u64) -> Self {
        Self {
            cache: LruCache::new(NonZeroUsize::new(10000).unwrap()),
            cache_dir,
            max_size_bytes,
            current_size: 0,
        }
    }

    pub async fn get(&mut self, key: &ShardKey) -> Option<Vec<u8>> {
        if let Some(path) = self.cache.get(key) {
            tokio::fs::read(path).await.ok()
        } else {
            None
        }
    }

    pub async fn put(&mut self, key: ShardKey, data: Vec<u8>) -> Result<()> {
        let shard_size = data.len() as u64;

        // Evict if needed
        while self.current_size + shard_size > self.max_size_bytes {
            if let Some((old_key, old_path)) = self.cache.pop_lru() {
                let old_size = tokio::fs::metadata(&old_path).await?.len();
                tokio::fs::remove_file(old_path).await?;
                self.current_size -= old_size;
            } else {
                break; // Cache empty
            }
        }

        // Write shard to disk
        let path = self.cache_dir.join(format!("{}.bin", key.hash()));
        tokio::fs::write(&path, &data).await?;

        self.cache.put(key, path);
        self.current_size += shard_size;

        Ok(())
    }
}

// QUIC shard fetching from Super-Node
use quinn::{Endpoint, Connection};

pub async fn fetch_shard_from_upstream(
    endpoint: &Endpoint,
    super_node_addr: &Multiaddr,
    cid: &str,
    shard_id: u8,
) -> Result<Vec<u8>> {
    let conn = endpoint.connect(super_node_addr.to_socket_addr(), "icn")?.await?;

    let mut send_stream = conn.open_uni().await?;
    send_stream.write_all(format!("GET /shards/{}/shard_{}.bin\n", cid, shard_id).as_bytes()).await?;
    send_stream.finish().await?;

    let mut recv_stream = conn.accept_uni().await?;
    let mut data = Vec::new();
    recv_stream.read_to_end(10 * 1024 * 1024, &mut data).await?; // Max 10MB

    Ok(data)
}
```

## Dependencies

**Hard Dependencies** (must be complete first):
- [T001] Moonbeam Repository Fork - For chain types (minimal usage)
- [T002] pallet-icn-stake - Relays must stake minimum 10 ICN
- [T011] Super-Node - Relays fetch shards from Super-Nodes

**Soft Dependencies** (nice to have):
- [T013] Viewer Client - Relays serve viewers, but can test with curl/browser
- Adaptive bitrate encoding - For bandwidth-constrained scenarios (future enhancement)

**External Dependencies:**
- 1TB storage (SSD preferred for cache performance)
- 100 Mbps symmetric bandwidth
- Low-latency network (<50ms to nearest Super-Node)

## Design Decisions

**Decision 1: LRU eviction instead of TTL-based expiration**
- **Rationale:** Popular content stays cached longer, unpopular content evicted quickly. Better cache hit rate than fixed TTL (e.g., 1 hour).
- **Alternatives:**
  - TTL-based: Simpler but wastes cache on unpopular content
  - ARC (Adaptive Replacement Cache): Better hit rate but more complex
- **Trade-offs:** (+) Simple, good hit rate. (-) Doesn't account for recency vs frequency (ARC does).

**Decision 2: Auto-region detection via latency instead of manual config**
- **Rationale:** Reduces operator burden, adapts to network conditions. Relay can re-detect if Super-Node latencies change.
- **Alternatives:**
  - Manual region config: Error-prone, doesn't adapt
  - GeoIP-based: Inaccurate for latency (network routing matters)
- **Trade-offs:** (+) Automatic, accurate. (-) Requires periodic re-detection (every 1 hour).

**Decision 3: QUIC for upstream and downstream instead of HTTP/2**
- **Rationale:** Lower latency (1-RTT handshake), multiplexed streams without head-of-line blocking. WebTransport support for browser clients.
- **Alternatives:**
  - HTTP/2 over TLS/TCP: Higher latency, head-of-line blocking
  - Raw TCP: No multiplexing, no encryption
- **Trade-offs:** (+) Best performance for video streaming. (-) UDP-based, some NAT issues (mitigated by ICE).

**Decision 4: Cache persistence across restarts instead of ephemeral**
- **Rationale:** Avoid re-fetching 1TB of content after restart. Faster recovery, reduced upstream load.
- **Alternatives:**
  - Ephemeral cache: Simpler, but wastes bandwidth on restarts
  - Database-backed cache: Overkill, adds latency
- **Trade-offs:** (+) Faster restarts, lower bandwidth. (-) Requires manifest file management.

## Risks & Mitigations

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| Cache thrashing (high eviction rate) | Medium (cache ineffective) | Medium | Monitor cache hit rate. If <70%, increase cache size or add more relays to distribute load. Implement frequency-based eviction (LFU hybrid). |
| Upstream Super-Node single point of failure | High (relay can't serve) | Low | DHT provides multiple Super-Nodes per region. Relay maintains connection pool to 3+ Super-Nodes. Failover within 1 second. |
| Bandwidth exhaustion (too many viewers) | Medium (playback buffering) | High | Rate limiting per viewer. Adaptive bitrate signaling (suggest lower quality). Redirect overflow viewers to other relays or Super-Nodes. |
| Latency detection inaccuracy | Low (suboptimal region) | Medium | Ping multiple times (3 samples), use median. Re-detect every 1 hour. Allow manual region override in config. |
| Disk I/O bottleneck (cache reads) | Medium (high latency) | Low | Use SSD for cache storage. Pre-allocate cache file space (avoid fragmentation). Monitor disk IOPS with `iostat`. |

## Progress Log

### [2025-12-24] - Task Created

**Created By:** task-creator agent
**Reason:** User request to create comprehensive off-chain node tasks for ICN project
**Dependencies:** T001 (Moonbeam fork), T002 (stake pallet), T011 (Super-Node)
**Estimated Complexity:** Standard (8,000 tokens) - Caching and distribution layer, relatively straightforward

## Completion Checklist

**Code Complete:**
- [ ] All acceptance criteria met and verified
- [ ] Unit tests pass with >80% coverage
- [ ] Integration tests pass (mock Super-Node, mock viewer)
- [ ] Clippy warnings resolved
- [ ] Code formatted with rustfmt
- [ ] Documentation comments complete

**Integration Ready:**
- [ ] Latency detection verified (selects lowest-latency Super-Node)
- [ ] Cache hit/miss tested (50ms vs 500ms latency)
- [ ] LRU eviction tested (cache stays under 1TB)
- [ ] Upstream failover tested (switches to backup Super-Node)
- [ ] Metrics verified in Prometheus

**Production Ready:**
- [ ] Cache persistence tested (survives restart)
- [ ] Resource limits tested (max 1TB disk, 100 Mbps bandwidth)
- [ ] Logs structured and parseable
- [ ] Error paths tested (upstream timeout, disk full, DHT failure)
- [ ] Monitoring alerts configured (low cache hit rate, upstream failures)
- [ ] Deployment guide written (systemd service, Docker volume)

**Definition of Done:**
Task is complete when relay runs for 24 hours on testnet serving 100+ viewers, achieves >80% cache hit rate, maintains <100ms average shard delivery latency, and successfully fails over to backup Super-Node within 1 second during simulated upstream outage.
