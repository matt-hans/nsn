# Architecture Context: Neuro-Symbolic Exocortex v3.0

## Tech Stack

### Language Strategy

| Component | Language | Rationale |
|-----------|----------|-----------|
| **Daemon (`exod`)** | Python 3.11+ | Native ecosystem for local AI (PyTorch, LanceDB, KùzuDB, ExLlamaV2) |
| **CLI (`cortex`)** | Rust 1.75+ | Instant startup, memory safety, fast file operations |
| **Shared Libraries** | Rust + PyO3 | Performance-critical paths (hashing, compression) |

### Core Dependencies

| Category | Technology | Version | Purpose |
|----------|------------|---------|---------|
| **Vector Store** | LanceDB | Latest | Zero-copy mmap, columnar storage |
| **Graph Store** | KùzuDB | 0.4.0+ | Embedded graph DB, Cypher support |
| **Embeddings** | bge-small-en-v1.5 | 1.5 | 384-dim, <100MB, high quality |
| **Tensor Format** | SafeTensors | Latest | Safe model/vector serialization |
| **Serialization** | Protocol Buffers | 3.x | gRPC message format |
| **Compression** | Zstandard | 1.5+ | Archive compression (level 19) |
| **Hashing** | BLAKE3 | Latest | Fast integrity verification |
| **Encryption** | age / ring | Latest | AES-256-GCM implementation |

## System Architecture

### Container Overview

The system consists of three major subsystems:

1. **Core Runtime:** Mount Manager, Adaptive Router, Union Retriever, Dual-Stream Injector
2. **Lifecycle Management:** Consolidation Engine, Capacity Manager
3. **Storage Engines:** LanceDB (vectors), KùzuDB (graph)

### Key Components

#### Mount Manager

- **Purpose:** Manages cortex lifecycle (mount/unmount/hot-swap)
- **Design:** Layer stack with RCU semantics for lock-free reads
- **Limits:** Soft cap 5 layers, hard cap 16 layers
- **Session:** Persists mount state to `~/.exocortex/session_state.json`

#### Adaptive Router

- **Purpose:** Decides retrieval strategy (FAST/DEEP/PARALLEL)
- **Algorithm:** Thompson Sampling with Beta distributions
- **Learning:** Per-bucket priors aggregated from all mounted layers
- **Latency:** <5ms routing decision

#### Union Retriever

- **Purpose:** Scatter-gather queries across layers, fuse results
- **Strategies:**
  - FAST: Vector search only (20-50ms)
  - DEEP: Graph traversal (100-300ms)
  - PARALLEL: Both paths with timeout (50-150ms)
- **Fusion:** Reciprocal Rank Fusion (RRF) with layer boosting

#### Dual-Stream Injector

- **Stream A:** Text context (XML-formatted facts and relationships)
- **Stream B:** Steering vectors (tensor injection into LLM activations)
- **Modes:** Native (ExLlamaV2), Shared Memory (llama.cpp), Text Fallback (Ollama)

#### Consolidation Engine

- **Trigger:** Idle >15min AND (vectors >500 OR last_sleep >24hr)
- **Process:** Rotate diff → HDBSCAN clustering → Ghost node generation → Merge to base
- **Concurrency:** LSM-tree style with atomic buffer rotation

#### Capacity Manager

- **Algorithm:** Stability-Modulated Ebbinghaus decay
- **Tiers:** Active (hot, mounted) and Archive (cold, compressed)
- **Protection:** Anchored entities, pinned memories, recent items (<7 days)

## Data Architecture

### The .cortex Format

**Directory Structure (.cortex.d):**
```
MyMemory.cortex.d/
├── manifest.toml           # Configuration & metadata
├── memory.lance/           # Vector store (LanceDB)
├── topology.kuzu/          # Knowledge graph (KùzuDB)
├── steering/               # Behavioral vectors (SafeTensors)
├── intuition/              # Router priors (JSON)
├── assets/                 # Source documents
└── checksums.blake3        # Integrity verification
```

**Archive Format:** `.cortex.tar.zst` (tar + Zstandard level 19)

### Vector Store Schema

- **Engine:** LanceDB (Lance columnar format)
- **Schema:** Arrow schema with 384-dim vectors, metadata, retention fields
- **Index:** IVF-PQ or HNSW for ANN search
- **Embedding Model:** bge-small-en-v1.5

### Graph Store Schema

- **Engine:** KùzuDB
- **Nodes:** Entity, Document, Tombstone
- **Edges:** RELATES_TO, MENTIONS, HAS_CHUNK, ABSTRACTS
- **Query Language:** Cypher

## Design Decisions (ADRs)

### ADR-001: Split-State Container Format
- **Decision:** Runtime as directory (.cortex.d), transport as archive (.cortex.tar.zst)
- **Rationale:** Zero-copy mmap reads, easy composition, standard tools

### ADR-002: KùzuDB for Graph Storage
- **Decision:** Use KùzuDB as embedded graph engine
- **Rationale:** Embedded, Cypher support, columnar storage, 10-100x faster than SQLite for multi-hop

### ADR-003: LanceDB for Vector Storage
- **Decision:** Use LanceDB as embedded vector engine
- **Rationale:** Zero-copy mmap, Lance format, native versioning, handles 10M+ vectors

### ADR-004: Hybrid Inference Hook Strategy
- **Decision:** Graceful degradation (Native → Shared Memory → Text Fallback)
- **Rationale:** Works on 100% of local setups, optimal where hooks available

### ADR-005: Rejection of Docker Distribution
- **Decision:** Reject Docker for runtime
- **Rationale:** RAM tax, GPU passthrough fragility, UX barrier, latency overhead

### ADR-006: Thompson Sampling for Routing
- **Decision:** Use Thompson Sampling with per-bucket Beta distributions
- **Rationale:** O(1) computation, online learning, no training data required

### ADR-007: Async Graph Extraction
- **Decision:** Vector write synchronous (<50ms), graph extraction background (2-5s)
- **Rationale:** Responsive UI, eventual consistency acceptable for graph

## Critical Workflows

### Mount Workflow
1. Parse manifest.toml (<5ms)
2. Validate schema version (<1ms)
3. mmap vector store (<10ms)
4. Open graph store (<20ms)
5. Add to stack (<5ms)
6. **Total:** <50ms (background checksum verification)

### Query Workflow
1. Route query (<5ms)
2. Embed query (<30ms, cached)
3. Scatter to layers (parallel)
4. Gather and fuse (<10ms)
5. Pack context (<5ms)
6. **Total:** <100ms p95

### Consolidation Workflow
1. **PRE_FLIGHT:** Check disk space (<1s)
2. **ROTATING:** Atomic diff swap (<100ms)
3. **CLUSTERING:** HDBSCAN + ghost nodes (10s-5min)
4. **MERGING:** Insert to base (30s-10min)
5. **SWAPPING:** Atomic mount swap (<100ms)
6. **CLEANUP:** Delete frozen layer (1-30s)

## Performance Targets

| Operation | Target (p95) | Max (p99) |
|-----------|--------------|-----------|
| Mount (warm) | <50ms | 200ms |
| Mount (cold) | <500ms | 5s |
| Vector search | <30ms | 100ms |
| Graph traversal | <50ms | 200ms |
| Full query | <100ms | 250ms |
| Write (vector) | <40ms | 100ms |
| Consolidation | <5min | 30min |

## Validation Commands

### Testing Strategy

**Phase 1 (Foundation):**
```bash
cortex init test.cortex.d
python -c "from storage import vectors; vectors.insert(...)"
python -c "from storage import graph; graph.create_node(...)"
cortex pack test.cortex.d test.cortex.tar.zst
cortex unpack test.cortex.tar.zst test2.cortex.d
cortex verify test2.cortex.d
```

**Phase 2 (Runtime Core):**
```bash
cortex mount base.cortex.d --mode ro
cortex mount skills.cortex.d --mode ro
cortex mount user.cortex.d --mode rw
cortex list
cortex daemon stop && cortex daemon start
cortex list  # Verify session restore
```

**Phase 3 (Retrieval):**
```bash
cortex query "What is Project Orion?" --debug
cortex benchmark --queries 100  # p50: 45ms, p95: 89ms
```

**Phase 7 (Polish):**
```bash
cortex benchmark --profile full
cortex doctor  # Health check
./scripts/security_audit.sh
```
