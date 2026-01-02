# T063: Redundant Execution + Result Consensus Service

## Priority: P0 (Blocker)
## Complexity: 3-4 weeks
## Status: Pending
## Depends On: T062 (Task-Market Verification Gate), T049 (Container Manager), T018 (Dual CLIP Verification)

---

## Objective

Run tasks on multiple executors and compute consensus before submitting verification attestations to chain. Prevent single executor fraud.

## Background

Current executor workflow allows a single node to submit output. To harden, Lane 1 should use redundant execution with hash/semantic comparison and emit attestations for on-chain verification.

## Implementation

### Step 1: Scheduler Multi-Assignment

- Extend scheduler to assign each task to N executors (configurable, default 3)
- Track executor assignments and completion timing

### Step 2: Result Collection + Hashing

- Collect `output_cid` and compute content hash
- For AI outputs, compute CLIP semantic score vs recipe

### Step 3: Consensus Rule

- Majority hash match for deterministic outputs
- For AI outputs: require CLIP score above threshold and consensus within epsilon

### Step 4: Attestation Aggregation

- Produce attestation bundle signed by executors or validators
- Submit attestation bundle to `pallet-nsn-task-market::submit_attestation`

### Step 5: Failure + Timeout Handling

- If one executor fails, continue if quorum possible
- If quorum fails, mark task as rejected and report to chain

## Acceptance Criteria

- [ ] Tasks assigned to multiple executors by default
- [ ] Hash consensus or semantic consensus enforced
- [ ] Attestations submitted automatically upon consensus
- [ ] Failure handling leaves no orphaned tasks
- [ ] Metrics expose redundancy success rate and latency
- [ ] End-to-end integration with task-market verification gate

## Testing

- Unit test: 3 executors, 2 match -> consensus succeeds
- Unit test: all outputs differ -> consensus fails, task rejected
- Integration test: scheduler -> execution -> attestation -> chain verification
- Load test: 50 tasks with 3x redundancy, measure latency impact

## Deliverables

1. `node-core/crates/scheduler/` updates (multi-assignment + consensus)
2. `node-core/crates/validator/` or new attestation aggregator module
3. Metrics + logs for consensus pipeline
4. Documentation for redundancy policy

---

**This task is a production blocker until complete.**
