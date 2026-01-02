# NSN Complete Remediation Design
## Resolving All Audit Findings

**Date:** 2026-01-02
**Status:** Approved for Implementation
**Authors:** Claude Code + Human Operator

---

## Executive Summary

This design document addresses all findings from the security audit plus gaps from the existing remediation plan. It covers Lane 0/1 orchestration implementation, DHT decentralization, validator integration, sandbox hardening, and minimal agent support.

### Key Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Rust↔Python | gRPC | Matches sidecar pattern, clean separation |
| Sandbox | Docker + Hardening | Practical for testnet, preserves GPU access |
| Agent Pallet | Minimal | Derived accounts + pause/revoke only (YAGNI) |
| DHT Walk | Random walk only | MVP decentralization, GossipSub for roles |
| CID Verification | Optional flag | Trust boundary: internal=skip, external=verify |

---

## 1. Architecture Overview

### System Flow After Remediation

```
Chain Events (subxt) → Scheduler → Lane0/Lane1 (Rust)
                                        │
                                        │ gRPC
                                        ▼
                           Container (Docker+Hardened)
                                        │
                                        │ Internal
                                        ▼
                              Vortex Pipeline (Python)
                           Flux → LivePortrait → Kokoro
                                        │
                                        │ CLIP Verification
                                        ▼
                           Storage (IPFS) → P2P Distribution
```

### Implementation Status

| Component | Before | After |
|-----------|--------|-------|
| Lane 0 Orchestration | Placeholder | Full gRPC integration |
| Lane 1 Orchestration | Placeholder | Full gRPC integration |
| Cross-Lane Pipeline | N/A | LLM → Script → Video |
| DHT Walk | Placeholder | Random walk functional |
| Validator | Placeholder | CLIP verification via gRPC |
| Docker Sandbox | Basic | Hardened (seccomp, AppArmor) |
| CID Verification | Missing | Trust-boundary aware |
| Agent Pallet | Missing | Minimal (derived accounts) |

---

## 2. Vortex gRPC Service Interface

**New Component:** `vortex/src/vortex/grpc/service.py`

### Proto Definition

```protobuf
syntax = "proto3";
package vortex;

service VortexPipeline {
  // Lane 0: Full video generation pipeline
  rpc GenerateVideo(VideoRequest) returns (VideoResponse);

  // Lane 1: General model inference
  rpc RunInference(InferenceRequest) returns (InferenceResponse);

  // CLIP verification (used by validators)
  rpc VerifyClip(ClipRequest) returns (ClipResponse);

  // Health & model status
  rpc GetStatus(StatusRequest) returns (StatusResponse);
}

message VideoRequest {
  string request_id = 1;
  string prompt = 2;
  string voice_id = 3;
  string script = 4;
  int32 duration_sec = 5;
  int32 seed = 6;
}

message VideoResponse {
  string request_id = 1;
  bytes video_data = 2;
  string output_cid = 3;
  bytes clip_embedding = 4;
  float clip_score_b = 5;
  float clip_score_l = 6;
  bool self_check_passed = 7;
}

message InferenceRequest {
  string request_id = 1;
  string model_id = 2;
  bytes input_data = 3;
  map<string, string> parameters = 4;
}

message InferenceResponse {
  string request_id = 1;
  bytes output_data = 2;
  string output_cid = 3;
}

message ClipRequest {
  bytes video_data = 1;
  string prompt = 2;
}

message ClipResponse {
  float score_b = 1;
  float score_l = 2;
  float ensemble_score = 3;
  bytes embedding = 4;
  bool self_check_passed = 5;
  bool outlier_detected = 6;
}

message StatusRequest {}

message StatusResponse {
  bool healthy = 1;
  repeated string loaded_models = 2;
  float vram_used_gb = 3;
  float vram_total_gb = 4;
}
```

### Design Points

1. **Stateless requests** - Each call is self-contained
2. **Embedding included** - Video response includes CLIP embedding for BFT
3. **Self-check in response** - Director knows immediately if output passed

---

## 3. Lane 0/Lane 1 Rust Orchestration

### New Crates

- `node-core/crates/lane-common/` - Shared gRPC client
- `node-core/crates/lane0/src/lib.rs` - Video generation
- `node-core/crates/lane1/src/lib.rs` - General inference

### Shared Client (lane-common)

```rust
pub struct VortexClient {
    inner: VortexPipelineClient<Channel>,
    endpoint: String,
}

impl VortexClient {
    pub async fn connect(endpoint: &str) -> Result<Self, LaneError>;
    pub async fn health_check(&self) -> Result<VortexStatus, LaneError>;
    pub async fn generate_video(&self, req: VideoRequest) -> Result<VideoResponse, LaneError>;
    pub async fn run_inference(&self, req: InferenceRequest) -> Result<InferenceResponse, LaneError>;
    pub async fn verify_clip(&self, req: ClipRequest) -> Result<ClipResponse, LaneError>;
}
```

### Lane 0 Executor

```rust
pub struct Lane0Executor {
    client: VortexClient,
    storage: StorageManager,
    metrics: Lane0Metrics,
}

impl Lane0Executor {
    pub async fn execute(&self, recipe: Recipe) -> Result<Lane0Output, Lane0Error> {
        // 1. Call Vortex gRPC
        let response = self.client.generate_video(recipe.into()).await?;

        // 2. Store output (verify=false, we generated it)
        self.storage.put_trusted(&response.output_cid, &response.video_data).await?;

        // 3. Return embedding for BFT
        Ok(Lane0Output {
            cid: response.output_cid,
            embedding: response.clip_embedding,
            self_check_passed: response.self_check_passed,
        })
    }
}
```

### Lane 1 Executor

```rust
pub struct Lane1Executor {
    client: VortexClient,
    storage: StorageManager,
}

impl Lane1Executor {
    pub async fn execute(&self, task: TaskSpec) -> Result<Lane1Output, Lane1Error> {
        // 1. Fetch input from storage if needed
        let input_data = self.storage.get(&task.input_cid).await?;

        // 2. Call Vortex gRPC
        let response = self.client.run_inference(task.into()).await?;

        // 3. Store output
        self.storage.put_trusted(&response.output_cid, &response.data).await?;

        Ok(Lane1Output { cid: response.output_cid })
    }
}
```

### Cross-Lane Communication

```rust
/// Cross-lane output that can trigger downstream lanes
pub enum LaneOutput {
    Terminal { cid: Cid },
    Chainable { cid: Cid, next_lane: LaneTarget },
}

pub enum LaneTarget {
    Lane0 { recipe: Recipe },
    Lane1 { task: TaskSpec },
}

/// Pipeline coordinator for cross-lane workflows
pub struct LanePipeline {
    lane0: Lane0Executor,
    lane1: Lane1Executor,
}

impl LanePipeline {
    pub async fn execute_chain(&self, task: ChainableTask) -> Result<PipelineOutput, LaneError> {
        let mut current = task;
        let mut outputs = Vec::new();

        loop {
            let result = match current.lane {
                LaneTarget::Lane0 { recipe } => self.lane0.execute(recipe).await?.into(),
                LaneTarget::Lane1 { task } => self.lane1.execute(task).await?.into(),
            };

            outputs.push(result.cid.clone());

            match result {
                LaneOutput::Terminal { .. } => break,
                LaneOutput::Chainable { next_lane, .. } => {
                    current = ChainableTask { lane: next_lane };
                }
            }
        }

        Ok(PipelineOutput { outputs })
    }
}
```

**Example Flow - AI-Generated Video:**

```
1. Lane 1: LLM generates script from prompt
   └─► Output: script.txt (CID: Qm123...)
   └─► next_lane: Lane0 { recipe with script_cid: Qm123 }

2. Lane 0: Vortex generates video using script
   └─► Output: video.mp4 (CID: Qm456...)
   └─► Terminal (broadcast to network)
```

---

## 4. DHT Random Walk Implementation

**File:** `node-core/crates/p2p/src/bootstrap/dht_walk.rs`

```rust
use libp2p::kad::{Kademlia, QueryId};
use libp2p::PeerId;
use rand::Rng;

pub struct DhtWalker {
    min_peers: usize,
    target_peers: usize,
    concurrency: usize,
}

impl DhtWalker {
    pub async fn walk(
        &self,
        kademlia: &mut Kademlia<MemoryStore>,
        connected_peers: usize,
    ) -> Result<Vec<PeerInfo>, BootstrapError> {
        if connected_peers < self.min_peers {
            return Err(BootstrapError::InsufficientPeers {
                have: connected_peers,
                need: self.min_peers,
            });
        }

        let mut pending_queries = Vec::new();

        // Generate random target PeerIds across keyspace
        for _ in 0..self.concurrency {
            let random_peer_id = PeerId::random();
            let query_id = kademlia.get_closest_peers(random_peer_id);
            pending_queries.push(query_id);
        }

        // Responses collected via Swarm event loop
        Ok(vec![])
    }
}
```

**Integration:**

```rust
pub async fn bootstrap_with_dht_fallback(
    config: &BootstrapConfig,
    kademlia: &mut Kademlia<MemoryStore>,
) -> Result<Vec<PeerInfo>, BootstrapError> {
    // 1. Try configured sources first
    let mut peers = Vec::new();
    for source in &config.sources {
        if let Ok(source_peers) = source.fetch().await {
            peers.extend(source_peers);
        }
    }

    // 2. If insufficient, use DHT walk
    if peers.len() < config.min_peers {
        let walker = DhtWalker::default();
        let dht_peers = walker.walk(kademlia, peers.len()).await?;
        peers.extend(dht_peers);
    }

    Ok(peers)
}
```

---

## 5. Validator CLIP Integration

**File:** `node-core/crates/validator/src/lib.rs`

```rust
pub struct ClipValidator {
    client: VortexClient,
    threshold_b: f32,  // 0.70
    threshold_l: f32,  // 0.72
}

pub struct ValidationResult {
    pub is_valid: bool,
    pub score_b: f32,
    pub score_l: f32,
    pub ensemble_score: f32,
    pub outlier_detected: bool,
    pub embedding: ClipEmbedding,
}

impl ClipValidator {
    pub async fn verify(
        &self,
        video_cid: &Cid,
        prompt: &str,
        storage: &StorageManager,
    ) -> Result<ValidationResult, ValidatorError> {
        // 1. Fetch video from storage (verify=true, external data)
        let video_data = storage.get(video_cid).await?;

        // 2. Call Vortex CLIP verification
        let response = self.client.verify_clip(ClipRequest {
            video_data,
            prompt: prompt.to_string(),
        }).await?;

        // 3. Check thresholds
        let is_valid = response.score_b >= self.threshold_b
                    && response.score_l >= self.threshold_l
                    && !response.outlier_detected;

        Ok(ValidationResult {
            is_valid,
            score_b: response.score_b,
            score_l: response.score_l,
            ensemble_score: response.ensemble_score,
            outlier_detected: response.outlier_detected,
            embedding: response.embedding.into(),
        })
    }

    pub fn verify_embedding(
        &self,
        claimed: &ClipEmbedding,
        computed: &ClipEmbedding,
        tolerance: f32,
    ) -> bool {
        claimed.l2_distance(computed) <= tolerance
    }
}
```

---

## 6. Docker Sandbox Hardening

**Files:**
- `node-core/sidecar/src/container/seccomp.rs`
- `node-core/sidecar/seccomp/ai-workload.json`

### Seccomp Profile

```json
{
  "defaultAction": "SCMP_ACT_ERRNO",
  "architectures": ["SCMP_ARCH_X86_64"],
  "syscalls": [
    {
      "names": [
        "read", "write", "open", "close", "stat", "fstat",
        "mmap", "mprotect", "munmap", "brk",
        "ioctl", "clone", "futex", "set_robust_list",
        "epoll_create1", "epoll_ctl", "epoll_wait",
        "socket", "bind", "listen", "accept4", "connect",
        "sendto", "recvfrom", "getrandom"
      ],
      "action": "SCMP_ACT_ALLOW"
    },
    {
      "names": ["ioctl"],
      "action": "SCMP_ACT_ALLOW",
      "args": [{ "index": 1, "value": 3222817548, "op": "SCMP_CMP_EQ" }],
      "comment": "Allow NVIDIA ioctl for GPU access"
    }
  ]
}
```

### Enhanced Configuration

```rust
impl ContainerManagerConfig {
    pub fn hardened_defaults() -> Self {
        Self {
            network_mode: "none".to_string(),
            seccomp_profile: Some("ai-workload.json".to_string()),
            drop_all_caps: true,
            read_only_rootfs: true,
            no_new_privileges: true,
            apparmor_profile: Some("nsn-container".to_string()),
            ulimits: UlimitConfig {
                nofile: 1024,
                nproc: 512,
                memlock: 16 * 1024 * 1024 * 1024,
            },
            tmpfs_mounts: vec![
                TmpfsMount { path: "/tmp", size_mb: 512, noexec: true },
            ],
            audit_logging: true,
            ..Default::default()
        }
    }
}
```

---

## 7. Minimal Agent Pallet

**File:** `nsn-chain/pallets/nsn-agent/src/lib.rs`

Minimal implementation: derived accounts + pause/revoke only.

```rust
#[derive(Clone, Encode, Decode, TypeInfo, MaxEncodedLen, PartialEq)]
pub enum AgentStatus {
    Active,
    Paused,
    Revoked,
}

#[derive(Clone, Encode, Decode, TypeInfo, MaxEncodedLen)]
pub struct AgentInfo<AccountId, BlockNumber> {
    pub operator: AccountId,
    pub index: u32,
    pub status: AgentStatus,
    pub created_at: BlockNumber,
}
```

### Extrinsics

| Call | Purpose |
|------|---------|
| `register_agent()` | Create agent under operator |
| `pause_agent(agent)` | Immediately stop agent |
| `resume_agent(agent)` | Restart paused agent |
| `revoke_agent(agent)` | Permanently disable |

### Derived Account

```rust
pub fn derive_agent_account(operator: &AccountId, index: u32) -> AccountId {
    let mut data = operator.encode();
    data.extend_from_slice(&index.to_le_bytes());
    T::Hashing::hash(&data).into()
}
```

### Integration

Other pallets call `is_agent_active()` to gate operations:

```rust
fn ensure_can_operate(account: &T::AccountId) -> DispatchResult {
    if pallet_nsn_agent::Agents::<T>::contains_key(account) {
        ensure!(
            pallet_nsn_agent::Pallet::<T>::is_agent_active(account),
            Error::<T>::AgentNotActive
        );
    }
    Ok(())
}
```

---

## 8. CID Verification with Trust Boundaries

**File:** `node-core/crates/storage/src/lib.rs`

### Trait Update

```rust
#[async_trait]
pub trait StorageBackend: Send + Sync {
    async fn put(&self, cid: &Cid, data: &[u8], verify: bool) -> Result<(), StorageError>;
    async fn get(&self, cid: &Cid) -> Result<Vec<u8>, StorageError>;
    // ... unchanged methods
}
```

### Verification Function

```rust
fn verify_cid(cid: &Cid, data: &[u8]) -> Result<(), StorageError> {
    let expected_hash = cid::Cid::try_from(cid.as_str())?;

    let computed = match expected_hash.hash().code() {
        0x12 => Code::Sha2_256.digest(data),
        0x1b => Code::Keccak256.digest(data),
        0xb220 => Code::Blake2b256.digest(data),
        code => return Err(StorageError::UnsupportedHashAlgorithm(code)),
    };

    if computed.digest() != expected_hash.hash().digest() {
        return Err(StorageError::CidMismatch {
            expected: cid.clone(),
            actual: format_cid(&computed),
        });
    }

    Ok(())
}
```

### Convenience Methods

```rust
impl StorageManager {
    /// Store from trusted internal source (skip verification)
    pub async fn put_trusted(&self, cid: &Cid, data: &[u8]) -> Result<(), StorageError> {
        self.backend.put(cid, data, false).await
    }

    /// Store from untrusted external source (verify)
    pub async fn put_untrusted(&self, cid: &Cid, data: &[u8]) -> Result<(), StorageError> {
        self.backend.put(cid, data, true).await
    }
}
```

### Usage

| Call Site | Method | Rationale |
|-----------|--------|-----------|
| Lane0/Lane1 output | `put_trusted()` | We generated it |
| GossipSub handler | `put_untrusted()` | Received from network |
| Model download | `put_untrusted()` | Initial fetch |
| Model cache load | `put_trusted()` | Already verified |

---

## 9. Implementation Plan

### Task Breakdown

| Priority | Task | Estimate | Dependencies |
|----------|------|----------|--------------|
| P0 | Vortex gRPC Service | 3 days | None |
| P0 | lane-common crate | 2 days | Vortex gRPC |
| P0 | Lane 0 Orchestration | 3 days | lane-common |
| P0 | Lane 1 Orchestration | 2 days | lane-common |
| P0 | Cross-Lane Pipeline | 2 days | Lane 0, Lane 1 |
| P1 | DHT Random Walk | 3 days | None |
| P1 | Validator CLIP Integration | 2 days | Vortex gRPC |
| P1 | CID Verification | 1 day | None |
| P1 | Docker Hardening | 2 days | None |
| P2 | pallet-nsn-agent | 3 days | None |
| P2 | Agent Integration | 1 day | pallet-nsn-agent |

### Critical Path

```
Week 1:
  Vortex gRPC ──► lane-common ──► Lane 0 ──► Lane 1
       │
       └──► Validator Integration

Week 2:
  Cross-Lane Pipeline
  DHT Random Walk (parallel)
  CID Verification (parallel)
  Docker Hardening (parallel)

Week 3:
  pallet-nsn-agent
  Agent Integration
  Integration Testing
```

### Testing Strategy

**Unit Tests:**
- Lane 0/1: Mock VortexClient, verify orchestration logic
- DHT Walk: Mock Kademlia, verify random walk behavior
- Validator: Mock storage + Vortex, verify threshold logic
- Storage: Test verify=true/false paths

**Integration Tests:**

| Test | Description |
|------|-------------|
| `test_lane0_end_to_end` | Recipe → Vortex → Storage → CID |
| `test_lane1_to_lane0_chain` | LLM → Script → Video generation |
| `test_dht_bootstrap_fallback` | All sources fail → DHT discovery |
| `test_validator_clip_verify` | Fetch → CLIP → Attestation |
| `test_agent_pause_blocks_operations` | Paused agent can't submit |
| `test_container_seccomp_enforcement` | Blocked syscalls fail |

---

## 10. Success Criteria

- [ ] Lane 0 generates video end-to-end with CLIP self-check
- [ ] Lane 1 executes inference task and stores output
- [ ] Cross-lane pipeline: LLM script → video generation works
- [ ] DHT walk discovers peers when bootstrap sources unavailable
- [ ] Validator verifies content and produces attestation
- [ ] CID mismatch rejected on untrusted put
- [ ] Hardened container blocks disallowed syscalls
- [ ] Agent pause immediately stops agent operations
- [ ] All existing tests continue to pass

---

## Appendix: SOLID Principles Applied

| Principle | Application |
|-----------|-------------|
| **Single Responsibility** | Lane0 handles video, Lane1 handles inference, lane-common handles connection |
| **Open/Closed** | New lane types extend without modifying existing code |
| **Liskov Substitution** | All StorageBackend implementations interchangeable |
| **Interface Segregation** | VortexClient exposes focused methods per use case |
| **Dependency Inversion** | Executors depend on traits (VortexClient, StorageManager), not implementations |

---

**Document Status:** Approved for Implementation
**Next Steps:** Create implementation tasks, begin with Vortex gRPC service

*End of Design Document*
