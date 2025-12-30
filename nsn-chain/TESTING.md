# NSN Chain Testing Documentation

This document describes the test coverage for the NSN Chain project.

## Test Organization

### Pallet Unit Tests

Each pallet contains comprehensive unit tests in `src/tests.rs`:

#### pallet-nsn-task-market

**Location:** `nsn-chain/pallets/nsn-task-market/src/tests.rs`

**Coverage:**
- Task creation with escrow reservation
- Task assignment (executor acceptance)
- Task completion with payment settlement
- Task failure with escrow refund
- Full lifecycle tests (success and failure paths)
- Multiple concurrent tasks
- Self-assignment and self-completion
- Boundary conditions (minimum escrow, queue limits)
- Error cases (insufficient balance, invalid states)

**Run tests:**
```bash
cd nsn-chain
cargo test -p pallet-nsn-task-market
```

**Key Integration Tests:**
- `full_task_lifecycle_success` - Complete flow from creation to completion
- `full_task_lifecycle_failure` - Complete flow from creation to failure
- `multiple_concurrent_tasks_lifecycle` - Concurrent task execution

### Off-Chain Scheduler Tests

**Location:** `node-core/crates/scheduler/src/tests.rs`

**Coverage:**
- State machine transitions (Starting → LoadingModels → Idle → Generating → etc.)
- Dual-lane task queuing (Lane 0 and Lane 1)
- Lane 0 priority over Lane 1
- Epoch transitions (Idle → On-Deck → EpochStart → EpochEnd)
- Lane 1 draining on On-Deck notification
- Task preemption logic
- Priority ordering within lanes
- Error recovery

**Run tests:**
```bash
cd node-core
cargo test -p nsn-scheduler
```

**Key Integration Tests:**
- `test_on_deck_starts_drain` - Lane 1 draining behavior
- `test_epoch_transition` - Full epoch lifecycle
- `test_next_task_priority` - Lane 0 > Lane 1 priority
- `test_should_preempt` - Preemption logic

## Test Requirements Met

### Task 4.2 Requirements

#### 1. Task Lifecycle Tests ✓

**Requirement:** Test complete Lane 1 task flow from creation to completion/failure

**Coverage:**
- `create_task_intent_succeeds` - Task creation with escrow
- `accept_assignment_succeeds` - Executor assignment
- `complete_task_succeeds` - Task completion with payment
- `fail_task_by_executor_succeeds` - Task failure with refund
- `full_task_lifecycle_success` - **Integration test: end-to-end success**
- `full_task_lifecycle_failure` - **Integration test: end-to-end failure**
- `test_task_escrow_insufficient` - Escrow validation

#### 2. Epoch Transition Tests ✓

**Requirement:** Test scheduler's dual-lane behavior and epoch transitions

**Coverage:**
- `test_on_deck_starts_drain` - **Lane 1 draining on On-Deck**
- `test_epoch_transition` - **Full epoch cycle (Idle → On-Deck → Start → End)**
- `test_next_task_priority` - **Lane 0 priority over Lane 1**
- `test_should_preempt` - **Preemption when Lane 0 task waiting**
- `test_lane0_priority_over_lane1` - Lane prioritization
- `test_epoch_tracker_isolation` - Epoch state isolation

## Running All Tests

### NSN Chain Tests

```bash
cd nsn-chain
cargo test --workspace
```

### Off-Chain Tests

```bash
cd node-core
cargo test --workspace
```

### Full Project Tests

```bash
# From project root
cd nsn-chain && cargo test --workspace && cd ../node-core && cargo test --workspace
```

## Test Coverage Summary

| Component | Unit Tests | Integration Tests | Coverage |
|-----------|------------|-------------------|----------|
| pallet-nsn-task-market | 36 tests | 3 full lifecycle | ✓ High |
| nsn-scheduler | 43 tests | 4 epoch/lane tests | ✓ High |
| pallet-nsn-stake | ✓ | - | ✓ Basic |
| pallet-nsn-reputation | ✓ | - | ✓ Basic |
| pallet-nsn-director | ✓ | - | ✓ Basic |
| pallet-nsn-bft | ✓ | - | ✓ Basic |
| pallet-nsn-storage | ✓ | - | ✓ Basic |
| pallet-nsn-treasury | ✓ | - | ✓ Basic |
| pallet-nsn-model-registry | ✓ | - | ✓ Basic |

## Test Patterns

### Pallet Test Pattern

```rust
#[test]
fn test_name() {
    ExtBuilder::default().build().execute_with(|| {
        // GIVEN: Setup initial state

        // WHEN: Execute action

        // THEN: Verify expected state
    });
}
```

### Scheduler Test Pattern

```rust
#[test]
fn test_name() {
    let mut scheduler = SchedulerState::new();

    // Phase 1: Setup
    // Phase 2: Action
    // Phase 3: Verification
}
```

## Future Testing Considerations

### Integration Testing

For true integration testing across on-chain and off-chain components:
- Consider end-to-end tests using `subxt` to interact with running chain
- Test director node interactions with on-chain state
- Verify epoch coordination between chain and off-chain nodes

### Performance Testing

- Benchmark pallet extrinsics
- Test scheduler under load
- Verify queue performance with large task counts

### Security Testing

- Fuzz testing for pallet inputs
- Adversarial testing for consensus logic
- Edge case testing for economic security

## Continuous Integration

Tests are run automatically on:
- Every commit (pre-commit hook)
- Pull requests (GitHub Actions)
- Before releases

**CI Command:**
```bash
./verify-build.sh
```

---

**Last Updated:** 2025-12-29
**Task:** 4.2 Integration Tests
