# Acceptance Criteria & Testing Templates

## Standard Acceptance Criteria Patterns

### For Substrate Pallets
1. Storage items defined with correct types and hashers
2. Events emitted for all state transitions
3. Errors defined for all failure modes
4. Extrinsics check origin and validate inputs
5. Unit tests cover success and failure paths
6. Integration tests verify inter-pallet communication
7. Benchmarks defined for weight calculation
8. Documentation comments complete and accurate

### For Off-Chain Components
1. Chain client successfully subscribes to events
2. P2P service connects and maintains peer count >10
3. Error handling with structured logging
4. Graceful shutdown on termination signals
5. Configuration loaded from environment/files
6. Metrics exposed on Prometheus endpoint
7. Integration tests with local dev chain
8. Resource cleanup on error paths

### For AI/ML Components
1. Models load successfully at startup
2. VRAM usage stays within 11.8GB budget
3. Generation completes within time budget (<15s)
4. Output quality meets CLIP threshold (>0.75)
5. Memory is freed after generation
6. GPU errors are caught and logged
7. Benchmark tests on target hardware
8. Graceful degradation on model load failure

## Validation Commands

### Substrate Pallets
```bash
# Build all pallets
cargo build --release --all-features

# Run all tests
cargo test --all-features -- --nocapture

# Clippy linting
cargo clippy --all-features -- -D warnings

# Check formatting
cargo fmt -- --check

# Build runtime WASM
cargo build --release --target wasm32-unknown-unknown -p nsn-runtime

# Verify runtime weights
cargo run --release -p nsn-weights-check
```

### Off-Chain Rust Components
```bash
# Build director node
cargo build --release -p nsn-director

# Build node-core (Lane 1)
cargo build --release -p scheduler -p sidecar

# Run integration tests
cargo test --features integration-tests

# Check dependencies
cargo audit
cargo deny check

# Benchmark
cargo bench --bench bft_coordination
```

### Vortex Engine (Python)
```bash
# Install dependencies
pip install -r vortex/requirements.txt

# Run unit tests
pytest vortex/tests/unit --cov=vortex

# Model loading test
python -c "from vortex.pipeline import VortexPipeline; p = VortexPipeline(); print(f'VRAM: {torch.cuda.memory_allocated() / 1e9:.2f} GB')"

# Generation benchmark
python vortex/benchmarks/slot_generation.py --slots 5 --max-time 15
```

### Viewer Client (Tauri)
```bash
# Install dependencies
npm install

# Run dev server
npm run tauri dev

# Build production
npm run tauri build

# Run tests
npm test

# Type checking
npx tsc --noEmit
```

## Test Scenario Format (Given/When/Then)

### Example: Director Election (Lane 0 - Epoch-Based)
```gherkin
GIVEN a network with 20 Directors in On-Deck set across 5 regions
  AND current epoch is 10 (block 1000)
  AND all Directors have reputation scores >500
WHEN the new epoch begins at block 1001
THEN exactly 5 Directors are elected from On-Deck set
  AND no more than 2 Directors are from the same region
  AND all elected Directors have stake ≥100 NSN
  AND DirectorsElected event is emitted with epoch number
  AND new On-Deck set is prepared for epoch 12
```

### Example: BFT Consensus with Challenge (Lane 0)
```gherkin
GIVEN 5 elected Directors for epoch 10
  AND all Directors have generated video with CLIP score >0.75
  AND 3 Directors agree on canonical embedding hash 0xABCD
WHEN submitter calls submit_bft_result(epoch=10, hash=0xABCD, agreeing=[D1,D2,D3])
THEN BFT result is stored as PENDING
  AND challenge period starts (50 blocks)
  AND FinalizedEpochs[10] = false

GIVEN the above pending BFT result
  AND 5 minutes (50 blocks) pass with no challenge
WHEN on_finalize runs after challenge period
THEN epoch 10 is auto-finalized
  AND FinalizedEpochs[10] = true
  AND reputation events recorded for D1, D2, D3 (+100 each)
```

### Example: Task Marketplace (Lane 1)
```gherkin
GIVEN a task submitter with 50 NSN balance
  AND model "flux-schnell" registered in model-registry
WHEN submitter calls create_task(model="flux-schnell", input_cid="Qm...", max_price=10)
THEN task is created with status PENDING
  AND TaskCreated event is emitted
  AND 10 NSN is reserved from submitter balance

GIVEN a pending task with matching capabilities
  AND a node operator with model "flux-schnell" loaded
WHEN operator calls accept_task(task_id)
THEN task status changes to ASSIGNED
  AND TaskAssigned event is emitted with operator account
```

## Definition of Done

### Code Complete Checklist
- [ ] All acceptance criteria met
- [ ] All test scenarios pass
- [ ] Code reviewed (if multi-person)
- [ ] Documentation updated
- [ ] Clippy/linting passes
- [ ] Formatting applied
- [ ] No regression in existing tests
- [ ] Performance impact assessed

### Deployment Ready Checklist
- [ ] Integration tests pass on testnet
- [ ] Metrics verified in Grafana
- [ ] Logs structured and parseable
- [ ] Error paths tested
- [ ] Resource usage within limits
- [ ] Security review completed (for critical pallets)
- [ ] Rollback plan documented
- [ ] Monitoring alerts configured
