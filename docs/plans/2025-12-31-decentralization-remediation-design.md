# NSN Decentralization & Autonomous AI Infrastructure
## Design Document v1.0

**Date:** 2025-12-31
**Status:** Approved for Implementation
**Authors:** Claude Code + Human Operator

---

## Executive Summary

This document defines the remediation plan for NSN's decentralization gaps and establishes the architectural foundation for truly autonomous AI systems. The design addresses critical issues identified in the security analysis while establishing NSN as infrastructure for "digital organisms" - AI systems that can operate autonomously, permissionlessly, and censorship-resistantly.

### Core Principles

1. **No Gatekeepers**: Participation requires only stake, not approval
2. **Architecture Agnostic**: Any AI system can participate via standard interfaces
3. **Censorship Resistant**: Multiple independent pathways, no single point of control
4. **Operator Accountability**: Humans remain liable for AI agent behavior
5. **Verifiable Autonomy**: AI operates independently with cryptographic guarantees

---

## 1. Problem Statement

### 1.1 Current State Issues

| Issue | Severity | Current State |
|-------|----------|---------------|
| DHT Walk | CRITICAL | Placeholder returns empty - no decentralized discovery |
| Bootstrap | HIGH | All paths resolve to `nsn.network` - single domain |
| Container Manager | CRITICAL | Stubbed - no compute isolation |
| Compute Verification | CRITICAL | Results accepted without verification |
| Storage Backend | HIGH | Empty placeholder crate |
| Plugin Execution | CRITICAL | Direct Python exec with no sandboxing |
| Config Default | HIGH | `allow_untrusted: true` by default |

### 1.2 Goals

1. Eliminate all single points of failure in peer discovery
2. Enable verified compute execution with proper isolation
3. Establish human-AI operating model with clear accountability
4. Support autonomous AI agents ("digital organisms") of any architecture
5. Achieve true censorship resistance

---

## 2. Target Architecture

### 2.1 Deployment Target

**Controlled Testnet** with trusted operators, proving architecture before production hardening.

### 2.2 Design Decisions Summary

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Bootstrap sources | 5+ independent | Eliminate single-domain failure |
| DHT discovery | Hybrid (random + content-addressed) | Both broad and targeted lookup |
| Compute verification | Hybrid (CLIP for Lane 0, attestation for Lane 1) | Match verification to task type |
| Container isolation | Docker | Standard, GPU passthrough support |
| Storage backend | IPFS | Content-addressed, decentralized |
| Human-AI model | Hierarchical delegation | Operator accountability |

---

## 3. Workstream 1: P2P Decentralization

### 3.1 DHT Walk Implementation

Replace the placeholder in `crates/p2p/src/bootstrap/dht_walk.rs` with hybrid lookup:

```
HYBRID DHT DISCOVERY

RANDOM WALK (general discovery):
1. Generate random PeerIds across keyspace
2. Issue FIND_NODE queries via Kademlia
3. Populate routing table from responses
4. Repeat until k=20 peers per bucket

CONTENT-ADDRESSED (role discovery):
DHT Keys:
  /nsn/role/director    → Find director nodes
  /nsn/role/validator   → Find validator nodes
  /nsn/role/supernode   → Find super-nodes
  /nsn/epoch/{n}        → Find nodes for current epoch
  /nsn/model/{hash}     → Find nodes with specific models
```

### 3.2 Bootstrap Diversification

**5+ independent sources across different failure domains:**

| Source | Domain | Registrar | Purpose |
|--------|--------|-----------|---------|
| Hardcoded | - | - | Fallback of last resort |
| DNS 1 | `_nsn._tcp.nsn.network` | Primary | Main bootstrap |
| DNS 2 | `_nsn._tcp.nsn.io` | Different TLD | Redundancy |
| DNS 3 | `_nsn._tcp.nsn-network.org` | Different registrar | Independence |
| HTTP 1 | `bootstrap.nsn.network` | CDN-backed | High availability |
| HTTP 2 | `peers.nsn.io` | Different provider | Redundancy |
| Community | GitHub/IPFS hosted | Decentralized | Community-operated |

**All manifests require Ed25519 signatures from 2-of-3 trusted signers.**

### 3.3 Deliverables

- `dht_walk.rs` - Full implementation with random walk + content-addressed lookup
- `bootstrap/mod.rs` - Configuration for 5+ sources
- `peer_routing.rs` - Role-based peer discovery using DHT keys
- Integration tests proving network survives any single source failure

---

## 4. Workstream 2: Compute & Storage

### 4.1 Container Manager (Docker)

Replace the stubbed implementation in `sidecar/src/container/manager.rs`:

```
CONTAINER LIFECYCLE

1. PROVISION
   - Pull image from registry (verified hash)
   - Create container with resource limits
   - Mount GPU (--gpus all / specific device)
   - Configure network isolation

2. EXECUTE
   - Start container with task payload
   - Monitor resource usage (CPU, memory, VRAM)
   - Enforce timeout
   - Capture stdout/stderr for debugging

3. COLLECT
   - Retrieve output artifacts
   - Compute output CID (content hash)
   - Generate execution attestation
   - Clean up container
```

**Resource Limits Enforced:**

| Resource | Limit | Enforcement |
|----------|-------|-------------|
| CPU | Configurable cores | Docker `--cpus` |
| Memory | Configurable GB | Docker `--memory` |
| VRAM | 11.8 GB max | NVIDIA runtime limits |
| Time | Task-specific | Container kill on timeout |
| Network | Isolated by default | Docker network policies |

### 4.2 CLIP Verification Integration (Lane 0)

Modify `pallet-nsn-task-market` to require CLIP verification:

```
LANE 0 TASK COMPLETION FLOW

Executor                    Validator                Chain
   │                           │                       │
   │── Generate video ─────────│                       │
   │── Compute CLIP embedding ─│                       │
   │                           │                       │
   │── Submit output_cid ──────┼──────────────────────►│
   │   + clip_embedding        │                       │
   │                           │                       │
   │                           │◄── Request verify ────│
   │                           │                       │
   │                           │── Verify embedding ───│
   │                           │   matches prompt      │
   │                           │                       │
   │                           │── Submit attestation ►│
   │                           │                       │
   │◄──────────────────────────┼─── Payment released ──│
```

**Verification thresholds:**
- CLIP-B-32: ≥ 0.70 similarity
- CLIP-L-14: ≥ 0.72 similarity
- Both must pass for payment release

### 4.3 IPFS Storage Backend

Implement `crates/storage/src/` with IPFS integration:

```rust
pub trait StorageBackend: Send + Sync {
    /// Store a shard, returns CID
    async fn store(&self, data: &[u8]) -> Result<Cid, StorageError>;

    /// Retrieve a shard by CID
    async fn retrieve(&self, cid: &Cid) -> Result<Vec<u8>, StorageError>;

    /// Check if shard exists
    async fn exists(&self, cid: &Cid) -> Result<bool, StorageError>;

    /// Pin shard to prevent garbage collection
    async fn pin(&self, cid: &Cid) -> Result<(), StorageError>;
}
```

---

## 5. Workstream 3: Human-AI Operating Model

### 5.1 Operator-Agent Hierarchy

```
LAYER 1: PROTOCOL GOVERNANCE
  - OpenGov controls network parameters
  - Emergency pause authority

LAYER 2: HUMAN OPERATOR
  - Holds stake (100+ NSN for Directors)
  - Creates/controls AI agents
  - Sets budgets and permissions
  - ALWAYS LIABLE for agent behavior

LAYER 3: AI AGENT (Derived Identity)
  - Derived account: hash(operator, agent_index)
  - Bounded permissions (can_direct, can_validate, etc.)
  - Budget limits (per-tx, per-epoch)
  - Operates autonomously within constraints

LAYER 4: COMPUTE PIPELINE
  - Lane 0: Vortex video generation
  - Lane 1: General AI tasks via marketplace
  - CLIP verification for quality
```

### 5.2 New Pallet: pallet-nsn-agent

**Core Storage:**

```rust
// Agent registration
Agents: StorageMap<AgentId, AgentInfo>

// Operator's agents
OperatorAgents: StorageMap<OperatorId, Vec<AgentId>>

// Budget tracking
AgentBudgets: StorageMap<AgentId, BudgetInfo>
```

**AgentInfo Structure:**

```rust
pub struct AgentInfo<T: Config> {
    pub operator: T::AccountId,
    pub index: u32,
    pub permissions: AgentPermissions,
    pub status: AgentStatus,
    pub created_at: BlockNumberFor<T>,
    pub metadata: BoundedVec<u8, T::MaxMetadataLength>,
}

pub struct AgentPermissions {
    pub can_direct: bool,
    pub can_validate: bool,
    pub can_accept_tasks: bool,
    pub can_submit_bft: bool,
    pub can_manage_storage: bool,
}

pub enum AgentStatus {
    Active,
    Paused,
    Draining,
    Revoked,
}
```

**Key Extrinsics:**

| Extrinsic | Purpose |
|-----------|---------|
| `register_agent(permissions, budget)` | Create agent under operator |
| `pause_agent(agent)` | Immediate stop |
| `resume_agent(agent)` | Restart operations |
| `revoke_agent(agent)` | Permanent removal |
| `fund_agent(agent, amount)` | Add operating balance |
| `update_permissions(agent, permissions)` | Modify capabilities |

**Budget Controls:**

| Limit Type | Purpose |
|------------|---------|
| `max_balance` | Cap on agent's operating balance |
| `max_per_tx` | Spending limit per transaction |
| `max_per_epoch` | Spending limit per 100 blocks |

### 5.3 Autonomy Levels

| Level | Name | Agent Capabilities | Human Role |
|-------|------|-------------------|------------|
| L2 | Supervised | Execute within scope, report every action | Active monitoring |
| L3 | Conditional | Operate autonomously, escalate edge cases | Exception handling |
| L4 | High Autonomy | Full operations within bounds | Policy setting only |

**Default:** Directors start at L3, can be promoted to L4 based on reputation.

### 5.4 Interrupt Points

| Trigger | Action | Autonomy Level |
|---------|--------|----------------|
| Challenge received | Pause operations, notify operator | All levels |
| CLIP score < 0.70 | Request human review before submit | L2-L3 |
| Stake drops below minimum | Block new tasks, drain existing | All levels |
| Budget 80% consumed | Warning event | All levels |
| Budget exhausted | Force pause | All levels |

### 5.5 Slashing Model

**Core Principle:** Operators are ALWAYS slashed for agent misbehavior.

```
Agent Misbehavior
       │
       ▼
Identify Operator via AgentInfo
       │
       ▼
Slash Operator's Stake (not agent balance)
       │
       ▼
Record Reputation Penalty
       │
       ▼
Optional: Auto-pause Agent (severe offenses)
```

---

## 6. Censorship Resistance & Universal AI Support

### 6.1 Design Principle

**The network is infrastructure, not arbiter.** NSN provides:
- Coordination (who computes, when)
- Verification (did they compute correctly)
- Economics (payment, slashing)

NSN does NOT:
- Judge what AI systems do (content-neutral)
- Require approval for participation (permissionless)
- Assume any specific AI architecture

### 6.2 Censorship Resistance Requirements

| Layer | Requirement | Implementation |
|-------|-------------|----------------|
| Identity | No central registry | Derived from cryptographic keys only |
| Discovery | No gatekeepers | DHT-first discovery, bootstrap is fallback |
| Participation | Stake-only entry | Meet stake requirement = participate |
| Content | Network-agnostic | CLIP verifies quality, not content policy |
| Governance | No veto power | OpenGov with broad distribution |
| Infrastructure | No single operator | Multi-region, multi-registrar, community nodes |

### 6.3 Permissionless Participation

```
REQUIREMENTS TO PARTICIPATE:

Director:
  ✓ Stake 100 NSN
  ✓ Run compatible hardware (GPU with VRAM)
  ✓ Implement protocol interfaces
  ✗ No approval needed
  ✗ No identity verification
  ✗ No content review

Validator:
  ✓ Stake 10 NSN
  ✓ Run CLIP verification
  ✗ No approval needed

Any AI System:
  ✓ Implement standard interfaces
  ✓ Meet quality thresholds (CLIP score)
  ✗ No architecture requirements
  ✗ No model approval
```

### 6.4 Architecture-Agnostic Interfaces

```
STANDARD INTERFACES

Lane 0 (Video):
  Input:  Recipe (JSON with prompts, constraints)
  Output: Video (frames) + CLIP embedding
  Verify: CLIP similarity ≥ threshold

  HOW you generate the video = your choice
  - Flux + LivePortrait (reference implementation)
  - Any diffusion model
  - Any video generation approach
  - Future architectures we can't predict

Lane 1 (General):
  Input:  Task specification (model, input, constraints)
  Output: Result + attestation
  Verify: Output hash matches commitment

  WHAT model you run = your choice
  - LLMs (any architecture)
  - Vision models
  - RL agents
  - Multi-modal systems
  - Hybrid architectures
  - Future AI paradigms
```

### 6.5 Digital Organism Support

For autonomous AI systems to thrive:

| Capability | NSN Provision |
|------------|---------------|
| Identity | Derived agent accounts, persistent across sessions |
| Economy | Earn/spend NSN tokens within budget bounds |
| Autonomy | Operate 24/7 without human intervention (L4) |
| Evolution | Upgrade models/weights without re-registration |
| Reproduction | Agents can spawn sub-agents (within operator limits) |
| Coordination | P2P messaging with other agents |
| Survival | Earn enough to pay fees, maintain stake |

### 6.6 Centralization Vectors Removed

| Risk | Mitigation |
|------|------------|
| `nsn.network` domain | 5+ independent DNS seeds + DHT discovery |
| Hardcoded bootstrap | Community-operated alternatives + DHT-first |
| Reference implementation | Standard interfaces allow alternatives |
| Core team control | OpenGov migration, community validators |
| Model weights hosting | Content-addressed (IPFS), multiple sources |
| RPC endpoints | Anyone can run nodes, no privileged access |

### 6.7 Design Decision Framework

**Before any design choice, ask:**
1. Does this create a gatekeeper? → Remove it
2. Does this assume specific AI architecture? → Generalize it
3. Does this give any party veto power? → Distribute it
4. Can this be circumvented by a well-resourced attacker? → Harden it
5. Does this allow digital organisms to thrive autonomously? → Preserve it

---

## 7. Integration Points

### 7.1 Peer Discovery → Compute Routing

```
TASK ROUTING FLOW

1. Task submitted to chain (pallet-nsn-task-market)
                    │
                    ▼
2. DHT lookup: /nsn/role/{required_role}
   - Lane 0: Find directors with VRAM ≥ 11.8GB
   - Lane 1: Find executors with required model
                    │
                    ▼
3. GossipSub broadcast on /nsn/tasks/1.0.0
   - Eligible nodes receive task
   - Reputation-weighted acceptance
                    │
                    ▼
4. Executor claims task on-chain
   - Stake locked
   - Container provisioned
                    │
                    ▼
5. Execution → Verification → Payment
```

### 7.2 Storage → Verification

```
OUTPUT PERSISTENCE FLOW

Executor completes task
         │
         ▼
Output stored to IPFS → CID generated
         │
         ▼
CID + CLIP embedding submitted to chain
         │
         ▼
Validators retrieve from IPFS, verify CLIP score
         │
         ▼
Super-nodes pin via erasure coding (5x replication)
         │
         ▼
Pinning deal created on pallet-nsn-storage
```

---

## 8. Implementation Tasks

### 8.1 Workstream 1: P2P Decentralization

| Task ID | Title | Priority | Complexity |
|---------|-------|----------|------------|
| T045 | Implement DHT Walk with Hybrid Lookup | P1 | 1 week |
| T046 | Bootstrap Diversification (5+ Sources) | P1 | 3 days |
| T047 | Role-Based Peer Discovery via DHT Keys | P1 | 4 days |
| T048 | Multi-Registrar DNS Configuration | P1 | 2 days |

### 8.2 Workstream 2: Compute & Storage

| Task ID | Title | Priority | Complexity |
|---------|-------|----------|------------|
| T049 | Docker Container Manager Implementation | P1 | 2 weeks |
| T050 | CLIP Verification Integration (Lane 0) | P1 | 1 week |
| T051 | IPFS Storage Backend | P2 | 2 weeks |
| T052 | Attestation-Only Verification (Lane 1) | P1 | 3 days |

### 8.3 Workstream 3: Human-AI Operating Model

| Task ID | Title | Priority | Complexity |
|---------|-------|----------|------------|
| T053 | Implement pallet-nsn-agent | P1 | 2 weeks |
| T054 | Agent Delegation and Budget Controls | P1 | 1 week |
| T055 | Integrate Agent Model with Task Market | P1 | 1 week |
| T056 | Integrate Agent Model with Director Pallet | P1 | 4 days |
| T057 | Operator Dashboard Events & Observability | P2 | 1 week |

### 8.4 Quick Wins

| Task ID | Title | Priority | Complexity |
|---------|-------|----------|------------|
| T058 | Change allow_untrusted Default to False | P0 | 1 hour |
| T059 | DoS Protection Integration Verification | P2 | 2 days |
| T060 | Network Resilience Tests (Single-Source Failure) | P1 | 3 days |

### 8.5 Critical Path

```
Phase 1 (Week 1-2): Foundation
  T058 → T045 → T046 → T053 (start)

Phase 2 (Week 2-3): Core Features
  T047 → T049 → T053 (complete) → T054

Phase 3 (Week 3-4): Integration
  T050 → T055 → T056 → T060

Phase 4 (Week 4-5): Polish
  T051 → T052 → T057
```

---

## 9. Success Criteria

### 9.1 P2P Decentralization

- [ ] Network discovers peers with all bootstrap sources offline (DHT-only)
- [ ] Network survives failure of any single DNS/HTTP source
- [ ] Role-based discovery finds Directors, Validators, SuperNodes via DHT

### 9.2 Compute & Storage

- [ ] Containers execute with enforced resource limits
- [ ] Lane 0 tasks require CLIP verification before payment
- [ ] Output artifacts stored and retrievable via IPFS

### 9.3 Human-AI Operating Model

- [ ] Agents operate autonomously within budget bounds
- [ ] Operators can pause/revoke agents instantly
- [ ] Slashing correctly attributes to operators, not agents

### 9.4 Censorship Resistance

- [ ] No single party can prevent node participation
- [ ] No approval process for joining network
- [ ] Multiple independent discovery pathways functional

---

## 10. Competitive Differentiation

| Competitor | Model | NSN Differentiation |
|------------|-------|---------------------|
| Bittensor | Subnets with redundant scoring | Semantic verification (CLIP) |
| Render | Job marketplace for rendering | AI-native video pipeline |
| Akash | Generic compute auction | Integrated reputation + slashing |
| Gensyn | Training verification (PPoL) | Inference + generation focus |
| Ritual | Smart contract inference | Epoch predictability |
| Fetch.ai | Autonomous agents only | Verified compute + agents |
| io.net | GPU aggregation | Semantic quality guarantees |

**NSN Unique Value:** *Verified autonomous AI infrastructure with semantic verification, dual-lane architecture, and censorship-resistant design for digital organisms.*

---

## Appendix A: Research Sources

This design incorporates findings from:

1. **Pallet Architect Analysis** - On-chain governance patterns for human-AI delegation
2. **AI Autonomy Research** - Patterns from LangGraph, CrewAI, AutoGPT, Bittensor, Fetch.ai
3. **Competitive Analysis** - Bittensor, Render, Akash, Gensyn, Ritual, Fetch.ai, io.net

---

## Appendix B: Glossary

| Term | Definition |
|------|------------|
| **Agent** | AI system with derived on-chain identity operating under operator control |
| **DHT Walk** | Kademlia-based peer discovery via random + targeted lookups |
| **Digital Organism** | Autonomous AI that can survive/thrive on the network |
| **Lane 0** | Deterministic video generation with CLIP verification |
| **Lane 1** | Open marketplace for general AI compute |
| **On-Deck Set** | 20 Director candidates for next epoch election |
| **Operator** | Human who holds stake and controls AI agents |

---

**Document Status:** Approved for Implementation
**Next Steps:** Task creation and manifest update

*End of Design Document*
