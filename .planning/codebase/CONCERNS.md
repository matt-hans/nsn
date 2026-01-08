# Codebase Concerns

**Analysis Date:** 2026-01-08

## Tech Debt

**Task Management System Complexity:**
- Issue: 64 tasks in `.tasks/manifest.json` with complex dependency graph
- Files: `.tasks/manifest.json`, `.tasks/tasks/*.md`
- Why: Evolved organically during development phases
- Impact: Hard to determine which tasks are still relevant vs outdated
- Fix approach: Review and archive completed/obsolete tasks, simplify manifest

**Multiple Python Virtual Environments:**
- Issue: Both `.venv/` and `.venv311/` directories exist
- Why: Version testing or developer preference inconsistency
- Impact: Unclear which environment is canonical
- Fix approach: Standardize on single venv, document in CLAUDE.md

**Scattered Configuration:**
- Issue: Configuration spread across multiple locations
- Files: `.env.example`, `docker-compose.yml`, individual Cargo.toml files
- Why: Multi-language project with different conventions
- Impact: Onboarding complexity, easy to miss required config
- Fix approach: Create unified configuration documentation

## Known Bugs

**No Critical Bugs Documented:**
- Task system tracks issues but no blocking bugs identified in current analysis
- Monitor: `.tasks/manifest.json` for bug-related tasks

## Security Considerations

**Key Management:**
- Risk: Substrate keys and P2P identity keys need secure storage
- Files: Key generation in `nsn-chain/node/`, `node-core/crates/p2p/`
- Current mitigation: Substrate keystore, file-based PeerId
- Recommendations: Document key backup procedures, consider HSM for production

**RPC Endpoint Exposure:**
- Risk: Substrate RPC exposes chain state and transaction submission
- Files: `nsn-chain/node/src/rpc.rs` (if exists)
- Current mitigation: Default to localhost binding
- Recommendations: Document safe RPC configuration for production

**GPU Model Loading:**
- Risk: Model weights from external sources (Hugging Face)
- Files: `vortex/src/vortex/models/`
- Current mitigation: safetensors format, known model IDs
- Recommendations: Pin model versions, verify checksums

## Performance Bottlenecks

**Lane 0 Glass-to-Glass Latency:**
- Problem: 45 second target from prompt to video output
- Files: `vortex/src/vortex/pipeline/`
- Measurement: Not yet benchmarked in production
- Cause: Multi-model pipeline (Flux + LivePortrait + Kokoro + CLIP)
- Improvement path: Model quantization (NF4), batch optimization, pipeline parallelization

**VRAM Constraints:**
- Problem: 11.8GB VRAM requirement limits hardware compatibility
- Files: `vortex/src/vortex/utils/vram.py`
- Measurement: Minimum RTX 3060 12GB
- Cause: Multiple models loaded simultaneously
- Improvement path: Model offloading, sequential loading, lower precision

## Fragile Areas

**Pallet Dependency Chain:**
- Files: `nsn-chain/pallets/*/`
- Why fragile: Pallets depend on each other (stake → reputation → epochs → bft)
- Common failures: Breaking changes in base pallets cascade
- Safe modification: Test full pallet stack, review dependency graph
- Test coverage: Individual pallet tests exist, limited cross-pallet integration tests

**P2P Topic Coordination:**
- Files: `node-core/crates/p2p/src/`
- Why fragile: GossipSub topics must be synchronized across nodes
- Common failures: Topic name changes break message routing
- Safe modification: Version topics, backwards compatibility period
- Test coverage: Unit tests for individual components, limited E2E

**Epoch Election Logic:**
- Files: `nsn-chain/pallets/nsn-director/`, `node-core/crates/scheduler/`
- Why fragile: On-chain and off-chain must agree on election results
- Common failures: Clock skew, network partitions during transitions
- Safe modification: Extensive testing around epoch boundaries
- Test coverage: Needs more integration tests

## Scaling Limits

**Single Director Set:**
- Current capacity: 5 elected directors per epoch
- Limit: Video throughput limited by director count
- Symptoms at limit: Queue buildup, increased latency
- Scaling path: Multiple director sets, sharding by content type

**P2P Mesh Size:**
- Current capacity: Designed for hundreds of nodes
- Limit: GossipSub message amplification at large scale
- Symptoms at limit: Network congestion, message delays
- Scaling path: Topic sharding, hierarchical gossip

**Storage Provider Capacity:**
- Current capacity: TBD (depends on deployment)
- Limit: Erasure coding overhead, storage deal matching
- Scaling path: Storage marketplace economics, replication tuning

## Dependencies at Risk

**polkadot-sdk Version:**
- Risk: Rapid Polkadot SDK evolution, breaking changes between versions
- Files: `nsn-chain/Cargo.toml` (version 2512.0.0)
- Impact: Potential incompatibility with ecosystem updates
- Migration plan: Track stable releases, plan upgrade windows

**libp2p Evolution:**
- Risk: libp2p 0.53 API changes, deprecations
- Files: `node-core/Cargo.toml`
- Impact: P2P networking disruption
- Migration plan: Monitor rust-libp2p releases, test upgrades

**PyTorch/Diffusers Updates:**
- Risk: Model compatibility with library updates
- Files: `vortex/pyproject.toml`
- Impact: Inference failures, changed outputs
- Migration plan: Pin versions, test model outputs after upgrades

## Missing Critical Features

**Task System Production Mode:**
- Problem: Task management designed for development, not production operation
- Files: `.tasks/`
- Current workaround: Manual task tracking
- Blocks: Clean production deployment
- Implementation complexity: Medium (audit and simplify task system)

**Viewer P2P Integration:**
- Problem: Viewer client P2P stream reception incomplete
- Files: `viewer/src-tauri/`, `viewer/src/services/`
- Current workaround: Direct URL playback
- Blocks: Decentralized video streaming to end users
- Implementation complexity: High (full P2P client in Tauri)

**Production Monitoring Dashboard:**
- Problem: No centralized monitoring for node operators
- Current workaround: Direct Prometheus queries
- Blocks: Operational visibility
- Implementation complexity: Medium (Grafana dashboards)

## Test Coverage Gaps

**Cross-Pallet Integration Tests:**
- What's not tested: Full pallet interaction chains
- Files: `nsn-chain/pallets/*/`
- Risk: Hidden interaction bugs between pallets
- Priority: High
- Difficulty: Need comprehensive mock runtime with all pallets

**P2P Network Simulation:**
- What's not tested: Multi-node network behavior
- Files: `node-core/crates/p2p/`
- Risk: Network partition handling, message ordering
- Priority: High
- Difficulty: Requires network simulation framework

**Vortex Pipeline End-to-End:**
- What's not tested: Full prompt-to-video pipeline
- Files: `vortex/src/vortex/pipeline/`
- Risk: Integration issues between models
- Priority: Medium
- Difficulty: Requires GPU resources for testing

**Viewer E2E Coverage:**
- What's not tested: Full user flows with P2P
- Files: `viewer/e2e/`
- Risk: UI/UX issues in production scenarios
- Priority: Medium
- Difficulty: Need mock P2P backend

---

*Concerns audit: 2026-01-08*
*Update as issues are fixed or new ones discovered*
