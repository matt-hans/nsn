---
id: T011
title: Super-Node Implementation (Tier 1 Storage and Relay)
status: pending
priority: 2
agent: backend
dependencies: [T001, T002, T003, T006, T009]
blocked_by: []
created: 2025-12-24T00:00:00Z
updated: 2025-12-24T00:00:00Z
tags: [off-chain, super-node, rust, storage, erasure-coding, phase1]

context_refs:
  - context/project.md
  - context/architecture.md
  - context/acceptance-templates.md

docs_refs:
  - docs/prd.md#section-3.4
  - docs/architecture.md#section-4.3

est_tokens: 12000
actual_tokens: null
---

## Description

Implement the Super-Node, ICN's Tier 1 regional anchor providing high-availability storage and relay services. Super-Nodes form the backbone of the hierarchical swarm architecture, storing erasure-coded video shards with Reed-Solomon (10+4) encoding and relaying content to Tier 2 Regional Relays and Tier 3 Viewers.

Super-Nodes perform five core functions:
1. **Erasure Coding**: Receive canonical video from directors, encode to 14 shards (10 data + 4 parity) using Reed-Solomon
2. **Shard Storage**: Persist shards to local storage (10TB+ capacity) with IPFS CID addressing
3. **Regional Relay**: Distribute video chunks to Tier 2 relays via QUIC transport
4. **Audit Response**: Respond to pinning audits from pallet-icn-pinning (prove shard possession)
5. **CLIP Validation**: Optional validator capability (dual role to maximize stake utility)

Super-Nodes operate with minimum 50 ICN stake, are geographically distributed across 7 regions (NA-WEST, NA-EAST, EU-WEST, EU-EAST, APAC, LATAM, MENA), and earn rewards via pallet-icn-pinning for shard storage.

**Technical Approach:**
- Rust 1.75+ with Tokio async runtime
- Reed-Solomon erasure coding via `reed-solomon-erasure` crate
- Storage backend: filesystem with IPFS CID-based addressing (future: IPFS integration)
- rust-libp2p for P2P (GossipSub for chunk reception, Kademlia DHT for shard discovery)
- QUIC transport for efficient video streaming to relays
- subxt for pinning deal monitoring and audit submission

**Integration Points:**
- Subscribes to GossipSub `/icn/video/1.0.0` for canonical video chunks from directors
- Publishes shard manifests to Kademlia DHT (key: video CID, value: shard locations)
- Monitors `pallet-icn-pinning::PendingAudits` for audit challenges
- Submits audit proofs via `pallet-icn-pinning::submit_audit_proof` extrinsic

## Business Context

**User Story:** As a Super-Node operator, I want an automated storage service that receives video chunks, encodes them with erasure coding, stores shards durably, and responds to audits, so that I can earn pinning rewards while ensuring content availability.

**Why This Matters:** Super-Nodes are ICN's durability layer. Without redundant storage across geographic regions, video content would be lost if directors go offline. Erasure coding provides 1.4× overhead (vs 3× for simple replication) with same durability guarantees.

**What It Unblocks:**
- T012 (Regional Relays) can fetch video from Super-Nodes
- T013 (Viewer Clients) can fall back to Super-Nodes if relays unavailable
- Long-term content archival (7-day retention default, governance-adjustable)
- Geographic redundancy (withstand entire region outages)

**Priority Justification:** Priority 2 (Important) - Required for mainnet availability SLA (99.5%) but not critical for Moonriver MVP. Initial testnet can rely on directors keeping content temporarily. Super-Nodes add production-grade durability.

## Acceptance Criteria

- [ ] Binary compiles with `cargo build --release -p icn-super-node`
- [ ] GossipSub subscription receives video chunks from directors (topic: `/icn/video/1.0.0`)
- [ ] Reed-Solomon encoding creates 14 shards (10 data + 4 parity) from video chunk
- [ ] Any 10 of 14 shards can reconstruct original video (verified with test data)
- [ ] Shards persisted to filesystem with CID-based paths (`storage/<CID>/shard_<N>.bin`)
- [ ] Shard manifests published to Kademlia DHT (key: video CID, value: JSON with shard locations)
- [ ] QUIC server accepts connections from Regional Relays on port 9002
- [ ] Shard retrieval API responds to relay requests within 100ms (local disk read)
- [ ] Audit monitor detects pending audits from on-chain `PendingAudits` storage
- [ ] Audit proof generation: read challenged bytes from shard, hash with nonce, submit extrinsic
- [ ] Audit proof submission completes within 10-minute deadline (100 blocks)
- [ ] Storage usage metrics tracked (total shards, bytes stored, disk utilization)
- [ ] Prometheus metrics exposed on port 9102 (shard_count, bytes_stored, audit_success_rate)
- [ ] Graceful shutdown with flush of pending writes
- [ ] Configuration loaded from TOML (storage path, regions, chain endpoint, pinning deal monitoring)
- [ ] Unit tests for Reed-Solomon encoding/decoding, shard persistence, audit proof generation
- [ ] Integration test: receives mock video chunk, stores shards, retrieves via DHT

## Test Scenarios

**Test Case 1: Video Chunk Reception and Erasure Encoding**
- Given: Super-Node subscribed to `/icn/video/1.0.0` GossipSub topic
  And: Director publishes 50MB video chunk for slot 100
- When: Super-Node receives chunk
  And: Encodes with Reed-Solomon (10+4)
- Then: 14 shards created, each ~7MB (50MB / 10 data shards × 1.4 overhead)
  And: Shards stored at `storage/<CID>/shard_{00..13}.bin`
  And: Shard manifest created: `{ cid: <CID>, shards: 14, locations: [<multiaddr1>, ...] }`
  And: Manifest published to DHT

**Test Case 2: Shard Reconstruction (Data Loss Simulation)**
- Given: 14 shards stored for video chunk with CID 0xABCD
  And: 4 shards deleted (simulating disk failure)
- When: Reconstruction requested with remaining 10 shards
- Then: Original 50MB video chunk reconstructed bit-for-bit
  And: Checksum matches original: `sha256(reconstructed) == sha256(original)`

**Test Case 3: Audit Challenge Response (Success)**
- Given: Super-Node storing shard 5 for video CID 0xABCD
  And: On-chain audit initiated: `PendingAudits[audit_id] = { pinner: super_node_id, shard_hash: hash(shard_5), challenge: { offset: 1000, length: 64, nonce: <random> } }`
- When: Audit monitor detects pending audit
  And: Super-Node reads bytes 1000-1063 from shard 5
  And: Computes proof: `hash(bytes || nonce)`
- Then: Audit proof submitted: `submit_audit_proof(audit_id, proof)`
  And: Proof accepted on-chain (verified against challenge)
  And: Reputation +10 for `PinningAuditPassed`

**Test Case 4: Audit Challenge Failure (Missing Shard)**
- Given: Audit challenge for shard that Super-Node does not have (deleted or never stored)
- When: Audit monitor detects pending audit
  And: Super-Node attempts to read shard file
- Then: File not found error
  And: Audit deadline (100 blocks) expires
  And: Stake slashed 10 ICN for `AuditTimeout`
  And: Reputation -50

**Test Case 5: DHT Shard Discovery by Relay**
- Given: Super-Node published shard manifest to DHT for video CID 0xABCD
- When: Regional Relay queries DHT: `dht.get(key: 0xABCD)`
- Then: DHT returns manifest: `{ cid: 0xABCD, shards: 14, locations: [/ip4/1.2.3.4/tcp/9002/p2p/<peer_id>] }`
  And: Relay establishes QUIC connection to Super-Node
  And: Requests shards 0-9 (data shards only, not parity)

**Test Case 6: QUIC Shard Transfer to Relay**
- Given: QUIC server listening on port 9002
  And: Regional Relay connected via QUIC
- When: Relay sends request: `GET /shards/0xABCD/shard_5.bin`
- Then: Super-Node reads shard from disk
  And: Streams shard bytes over QUIC (7MB in ~140ms at 500 Mbps)
  And: Relay acknowledges receipt

**Test Case 7: Storage Cleanup (Expired Pinning Deals)**
- Given: Pinning deal for video CID 0xABCD expires at block 10000
  And: Current block is 10001
- When: Storage cleanup task runs (every 1000 blocks)
- Then: Shards for 0xABCD deleted from disk
  And: DHT manifest removed
  And: Storage metrics updated (bytes_stored reduced by 98MB)

## Technical Implementation

**Required Components:**
- `icn-super-node/src/main.rs` - Binary entrypoint with CLI args (--config, --storage-path, --region)
- `icn-super-node/src/config.rs` - Configuration struct with TOML deserialization
- `icn-super-node/src/erasure.rs` - Reed-Solomon encoding/decoding using `reed-solomon-erasure` crate
- `icn-super-node/src/storage.rs` - Shard persistence layer (filesystem with CID-based paths)
- `icn-super-node/src/p2p_service.rs` - libp2p GossipSub for chunk reception, Kademlia DHT for manifest publishing
- `icn-super-node/src/quic_server.rs` - QUIC transport server for shard transfers to relays
- `icn-super-node/src/audit_monitor.rs` - Polling `PendingAudits` storage, proof generation
- `icn-super-node/src/chain_client.rs` - subxt integration for audit submission, pinning deal monitoring
- `icn-super-node/src/metrics.rs` - Prometheus metrics (shard_count, bytes_stored, audit_responses)
- `icn-super-node/src/storage_cleanup.rs` - Background task to remove expired pinning deals

**Validation Commands:**
```bash
# Build
cargo build --release -p icn-super-node

# Run unit tests
cargo test -p icn-super-node --lib

# Run integration tests
cargo test -p icn-super-node --features integration-tests

# Clippy
cargo clippy -p icn-super-node -- -D warnings

# Format check
cargo fmt -p icn-super-node -- --check

# Run super-node
./target/release/icn-super-node \
  --config config/super-node.toml \
  --storage-path /mnt/icn-storage \
  --region NA-WEST \
  --chain-endpoint wss://moonriver.api.onfinality.io

# Check metrics
curl http://localhost:9102/metrics | grep icn_super_node_

# Test Reed-Solomon encoding/decoding
cargo test -p icn-super-node test_erasure_coding -- --nocapture
```

**Code Patterns:**
```rust
// Reed-Solomon erasure coding
use reed_solomon_erasure::galois_8::ReedSolomon;

pub struct ErasureCoder {
    encoder: ReedSolomon,
    data_shards: usize,
    parity_shards: usize,
}

impl ErasureCoder {
    pub fn new() -> Self {
        Self {
            encoder: ReedSolomon::new(10, 4).unwrap(),
            data_shards: 10,
            parity_shards: 4,
        }
    }

    pub fn encode(&self, data: &[u8]) -> Result<Vec<Vec<u8>>> {
        // Split data into 10 equal-sized chunks
        let shard_size = (data.len() + self.data_shards - 1) / self.data_shards;
        let mut shards: Vec<Vec<u8>> = data.chunks(shard_size)
            .map(|chunk| {
                let mut shard = chunk.to_vec();
                shard.resize(shard_size, 0); // Pad if needed
                shard
            })
            .collect();

        // Add 4 empty parity shards
        for _ in 0..self.parity_shards {
            shards.push(vec![0u8; shard_size]);
        }

        // Compute parity
        self.encoder.encode(&mut shards)?;

        Ok(shards)
    }

    pub fn decode(&self, mut shards: Vec<Option<Vec<u8>>>) -> Result<Vec<u8>> {
        // Reconstruct missing shards
        self.encoder.reconstruct(&mut shards)?;

        // Concatenate data shards (skip parity)
        let data: Vec<u8> = shards.into_iter()
            .take(self.data_shards)
            .filter_map(|s| s)
            .flatten()
            .collect();

        Ok(data)
    }
}

// Audit proof generation
use sha2::{Sha256, Digest};

pub fn generate_audit_proof(
    shard_path: &Path,
    challenge: &AuditChallenge,
) -> Result<Vec<u8>> {
    // Read challenged bytes from shard
    let mut file = File::open(shard_path)?;
    file.seek(SeekFrom::Start(challenge.byte_offset as u64))?;

    let mut buffer = vec![0u8; challenge.byte_length as usize];
    file.read_exact(&mut buffer)?;

    // Hash with nonce
    let mut hasher = Sha256::new();
    hasher.update(&buffer);
    hasher.update(&challenge.nonce);

    Ok(hasher.finalize().to_vec())
}

// DHT shard manifest publishing
use libp2p::kad::{Kademlia, Record, RecordKey};

pub async fn publish_shard_manifest(
    kademlia: &mut Kademlia,
    cid: &str,
    locations: Vec<Multiaddr>,
) -> Result<()> {
    let manifest = ShardManifest {
        cid: cid.to_string(),
        shards: 14,
        locations,
        created_at: SystemTime::now(),
    };

    let key = RecordKey::new(&cid.as_bytes());
    let value = serde_json::to_vec(&manifest)?;
    let record = Record::new(key, value);

    kademlia.put_record(record, Quorum::One)?;
    Ok(())
}
```

## Dependencies

**Hard Dependencies** (must be complete first):
- [T001] Moonbeam Repository Fork - Need chain metadata
- [T002] pallet-icn-stake - Super-Nodes must stake minimum 50 ICN
- [T003] pallet-icn-reputation - Earn reputation for audit responses
- [T006] pallet-icn-pinning - Pinning deals, audit challenges, rewards
- [T009] Director Node - Super-Nodes receive video chunks from directors

**Soft Dependencies** (nice to have):
- [T010] Validator Node - Super-Nodes could optionally run validator logic (dual role)
- IPFS integration - For now use filesystem, future migrate to IPFS

**External Dependencies:**
- 10TB+ storage (SSD for latency, HDD for cost)
- 500 Mbps symmetric bandwidth (for shard distribution)
- `reed-solomon-erasure` Rust crate

## Design Decisions

**Decision 1: Reed-Solomon (10+4) instead of simple replication (3×)**
- **Rationale:** 1.4× overhead vs 3× overhead for same durability (withstand 4 failures). Significant cost savings at scale (10TB × 1.4 = 14TB vs 10TB × 3 = 30TB).
- **Alternatives:**
  - Simple replication: Easier implementation, higher cost
  - Fountain codes: Better for streaming, but overkill for static shards
- **Trade-offs:** (+) 53% storage cost reduction. (-) Higher CPU cost for encoding (acceptable, one-time cost).

**Decision 2: Filesystem storage instead of IPFS for MVP**
- **Rationale:** Simpler implementation, lower latency (direct disk reads), avoid IPFS daemon dependency. CID-based path structure (`storage/<CID>/shard_<N>.bin`) maintains content-addressability.
- **Alternatives:**
  - IPFS: Better for decentralized discovery, but adds complexity
  - S3-compatible object storage: Centralization risk
- **Trade-offs:** (+) Faster MVP, predictable latency. (-) Manual shard replication across Super-Nodes (vs IPFS automatic).

**Decision 3: Kademlia DHT for shard discovery instead of centralized registry**
- **Rationale:** Decentralized, no single point of failure. DHT is libp2p-native, proven at scale (IPFS, Ethereum).
- **Alternatives:**
  - On-chain registry: Would exceed TPS limits, expensive gas
  - Centralized API: Single point of failure
- **Trade-offs:** (+) Scalable, censorship-resistant. (-) Eventual consistency (DHT propagation delay ~5s).

**Decision 4: QUIC transport for shard transfers instead of TCP**
- **Rationale:** QUIC has lower latency (1-RTT handshake vs 3-RTT for TLS/TCP), better congestion control, multiplexed streams. Native to libp2p.
- **Alternatives:**
  - TCP + TLS: Higher latency, head-of-line blocking
  - HTTP/3 (QUIC): Similar performance, but we need raw streams not HTTP
- **Trade-offs:** (+) 30-50% latency reduction. (-) Requires UDP, some NAT/firewall issues (mitigated by NAT traversal stack).

**Decision 5: Stake-weighted audit probability (v8.0.1 enhancement)**
- **Rationale:** Higher-stake Super-Nodes are more trusted, audited less frequently. Reduces on-chain audit load while maintaining security.
- **Alternatives:**
  - Uniform audit probability: Wastes resources on trusted nodes
  - Reputation-weighted: Circular dependency (reputation depends on audits)
- **Trade-offs:** (+) Incentivizes higher stake, reduces audit overhead. (-) Complexity in probability calculation.

## Risks & Mitigations

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| Disk failure (data loss) | High (shards unrecoverable) | Medium | RAID-1 or RAID-5 for storage. Monitor disk health with SMART metrics. Alert on high error rate. Require 5× geographic replication (4 shard failures tolerable). |
| Reed-Solomon encoding too slow | Medium (bottleneck for high-throughput) | Low | Profile on target hardware. Pre-allocate shard buffers. Use SIMD-optimized RS library. If still slow, reduce to (8+4) shards (1.5× overhead). |
| DHT manifest not found (discovery failure) | Medium (relays can't find shards) | Medium | Publish to multiple DHT nodes (Quorum::Majority). Re-publish every 1 hour (DHT refresh). Fallback to GossipSub query if DHT miss. |
| Audit proof generation timeout | High (10 ICN slash) | Low | Pre-index shard byte offsets for fast seeks. Use O_DIRECT I/O to bypass page cache. Set internal deadline (50 of 100 blocks) to allow retries. |
| Storage cleanup deletes active content | Critical (availability loss) | Very Low | Double-check on-chain `PinningDeals` before deletion. Only delete if `expires_at < current_block`. Dry-run mode for testing. Manual recovery from backups. |
| QUIC bandwidth exhaustion | Medium (relays starved) | Medium | Rate limiting per relay (max 100 Mbps per peer). Prioritize based on relay reputation. Horizontal scaling (add more Super-Nodes per region). |

## Progress Log

### [2025-12-24] - Task Created

**Created By:** task-creator agent
**Reason:** User request to create comprehensive off-chain node tasks for ICN project
**Dependencies:** T001 (Moonbeam fork), T002 (stake pallet), T003 (reputation pallet), T006 (pinning pallet), T009 (director node)
**Estimated Complexity:** Complex (12,000 tokens) - Storage layer with erasure coding, DHT integration, audit response

## Completion Checklist

**Code Complete:**
- [ ] All acceptance criteria met and verified
- [ ] Unit tests pass with >85% coverage
- [ ] Integration tests pass (mock video chunks, erasure encode/decode)
- [ ] Clippy warnings resolved
- [ ] Code formatted with rustfmt
- [ ] Documentation comments complete

**Integration Ready:**
- [ ] Reed-Solomon encoding verified (10+4 shards, any 10 reconstruct original)
- [ ] Shards successfully stored to disk and retrieved
- [ ] DHT manifest publishing tested (retrieval by mock relay)
- [ ] QUIC shard transfers tested (100 Mbps sustained throughput)
- [ ] Audit proof generation completes within deadline
- [ ] Metrics verified in Prometheus

**Production Ready:**
- [ ] Storage path configurable, supports external mounts
- [ ] Disk usage monitoring alerts (>80% full)
- [ ] Resource limits tested (max 10TB storage, 500 Mbps bandwidth)
- [ ] Logs structured and parseable
- [ ] Error paths tested (disk full, audit timeout, DHT failure)
- [ ] Monitoring alerts configured (audit failures, storage cleanup errors)
- [ ] Deployment guide written (systemd service, Docker volume mounts)
- [ ] Disaster recovery procedure (restore from erasure shards)

**Definition of Done:**
Task is complete when Super-Node runs for 7 days on Moonriver testnet storing 1000+ video chunks (total 50GB), successfully responds to 100+ audits with 100% success rate, serves 10,000+ shard transfers to relays, and maintains <1s average shard retrieval latency.
