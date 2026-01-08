# Phase 1, Plan 1: Pallet Integration Validation

## Objective

Create a comprehensive integration test suite that validates cross-pallet interactions in the NSN chain runtime. The tests will verify that the stake→reputation→director→bft chain works correctly when pallets call each other through loosely-coupled traits and runtime wrappers.

## Execution Context

**Files to read:**
- `nsn-chain/pallets/nsn-stake/src/lib.rs` (foundation pallet)
- `nsn-chain/pallets/nsn-reputation/src/lib.rs` (reputation scoring)
- `nsn-chain/pallets/nsn-director/src/lib.rs` (epoch elections)
- `nsn-chain/pallets/nsn-bft/src/lib.rs` (consensus storage)
- `nsn-chain/pallets/nsn-treasury/src/lib.rs` (reward distribution)
- `nsn-chain/pallets/nsn-task-market/src/lib.rs` (Lane 1 marketplace)
- `nsn-chain/runtime/src/configs/mod.rs` (runtime configuration)

**Test commands:**
```bash
cd nsn-chain && cargo test -p pallet-nsn-stake
cd nsn-chain && cargo test -p pallet-nsn-director
cd nsn-chain && cargo test -p pallet-nsn-reputation
cd nsn-chain && cargo test -p pallet-nsn-bft
cd nsn-chain && cargo test -p pallet-nsn-treasury
cd nsn-chain && cargo test -p pallet-nsn-task-market
cd nsn-chain && cargo test --workspace
```

## Context

**Current State:**
- 9 pallets implemented with individual unit tests (8 have mock.rs + tests.rs)
- No integration tests that span multiple pallets
- Runtime configuration exists with wrapper structs connecting pallets
- Key integration points implemented via traits: NodeModeUpdater, NodeRoleUpdater, LaneNodeProvider, ReputationUpdater, TaskSlashHandler

**Critical Integration Chains:**

1. **Stake → Director Election Chain:**
   - Stakes::stakes() queried by Director::elect_directors()
   - NodeMode and NodeRole updated via trait impls
   - Director role eligibility depends on stake amount ≥ MinStakeDirector

2. **Director → Reputation Chain:**
   - Director::finalize_slot() calls Reputation::record_event()
   - Director::resolve_challenge() slashes via Stake::slash()
   - Positive/negative reputation affects future elections

3. **Director → BFT Chain:**
   - BFT stores consensus results from finalized director slots
   - Director pallet should call BFT::store_embeddings_hash() on finalization

4. **Task Market → Stake/Reputation Chain:**
   - LaneNodeProvider queries eligible nodes from Stakes
   - ReputationUpdater records task outcomes
   - TaskSlashHandler slashes for abandonment

5. **Treasury → Work Recording Chain:**
   - Treasury::record_director_work() tracks slots for rewards
   - Treasury::record_validator_work() tracks votes for rewards
   - Reward distribution based on accumulated contributions

## Tasks

### Task 1: Create Integration Test Infrastructure

Create a shared mock runtime that includes all 6 core pallets (stake, reputation, director, bft, treasury, task-market) configured together, similar to individual pallet mocks but with full inter-pallet coupling.

**Files to create:**
- `nsn-chain/test/integration/mod.rs` - Integration test module
- `nsn-chain/test/integration/mock.rs` - Shared multi-pallet mock runtime

**Acceptance criteria:**
- [ ] Mock runtime compiles with all 6 pallets configured
- [ ] Runtime uses real trait implementations (not mocks) for inter-pallet calls
- [ ] Helper functions for common test setup (fund accounts, stake, etc.)

**Checkpoint:** Run `cargo test -p nsn-integration` (assuming we add a test crate) or add tests to runtime crate

### Task 2: Test Stake → Director Election Integration

Test that director elections correctly query stake eligibility and update node modes/roles.

**Test cases:**
1. `test_director_election_respects_stake_eligibility` - Only accounts with stake ≥ MinStakeDirector become eligible
2. `test_director_election_updates_node_mode` - Elected directors get NodeMode::Lane0Active
3. `test_director_election_updates_node_role` - Elected directors get NodeRole::ActiveDirector
4. `test_director_epoch_transition_restores_lane1` - Previous directors return to Lane1Active after epoch

**Files to modify/create:**
- `nsn-chain/test/integration/stake_director_tests.rs`

**Acceptance criteria:**
- [ ] All test cases pass
- [ ] Tests verify storage state changes in both pallets
- [ ] Tests verify events emitted correctly

**Checkpoint:** `cargo test stake_director` shows all tests passing

### Task 3: Test Director → Reputation Integration

Test that director slot finalization and challenge resolution correctly update reputation scores.

**Test cases:**
1. `test_finalize_slot_records_positive_reputation` - Successful slot → +100 director reputation
2. `test_challenge_upheld_records_negative_reputation` - Slashed → -200 director reputation
3. `test_reputation_affects_election_weight` - Higher reputation → higher election weight
4. `test_reputation_decay_affects_future_elections` - Inactive accounts decay in weight

**Files to modify/create:**
- `nsn-chain/test/integration/director_reputation_tests.rs`

**Acceptance criteria:**
- [ ] All test cases pass
- [ ] Reputation score storage reflects correct deltas
- [ ] Election algorithm correctly incorporates reputation

**Checkpoint:** `cargo test director_reputation` shows all tests passing

### Task 4: Test Director → BFT Storage Integration

Test that director BFT consensus results are correctly stored in the BFT pallet.

**Test cases:**
1. `test_director_finalization_stores_bft_result` - Finalized slot creates BFT record
2. `test_bft_stats_updated_on_success` - ConsensusStats increments correctly
3. `test_bft_stats_updated_on_failure` - Failed slots increment failed_rounds
4. `test_bft_pruning_removes_old_consensus` - Retention period respected

**Files to modify/create:**
- `nsn-chain/test/integration/director_bft_tests.rs`

**Acceptance criteria:**
- [ ] BFT storage contains expected data after director operations
- [ ] Statistics accurately reflect consensus outcomes
- [ ] No orphaned or missing consensus records

**Checkpoint:** `cargo test director_bft` shows all tests passing

### Task 5: Test Task Market → Stake Integration

Test that the task market correctly queries node eligibility and handles slashing.

**Test cases:**
1. `test_lane0_nodes_filtered_correctly` - Only Lane0Active/ActiveDirector nodes eligible for Lane 0
2. `test_lane1_nodes_filtered_correctly` - Only Lane1Active/Reserve nodes eligible for Lane 1
3. `test_task_abandonment_slashes_stake` - TaskSlashHandler reduces stake
4. `test_renderer_registration_respects_stake` - Registration requires valid stake

**Files to modify/create:**
- `nsn-chain/test/integration/task_market_stake_tests.rs`

**Acceptance criteria:**
- [ ] LaneNodeProvider returns correct nodes for each lane
- [ ] Slashing reduces stake amount and emits events
- [ ] Eligibility updates when node mode changes

**Checkpoint:** `cargo test task_market_stake` shows all tests passing

### Task 6: Test Treasury → Work Recording Integration

Test that treasury reward distribution correctly accumulates work from director and validator operations.

**Test cases:**
1. `test_director_slots_accumulate_for_rewards` - record_director_work increments slot count
2. `test_validator_votes_accumulate_for_rewards` - record_validator_work increments vote count
3. `test_reward_distribution_proportional` - Rewards split by work contribution
4. `test_distribution_resets_accumulations` - Contributions reset after distribution

**Files to modify/create:**
- `nsn-chain/test/integration/treasury_tests.rs`

**Acceptance criteria:**
- [ ] Accumulated contributions storage correctly updated
- [ ] Reward distribution mints correct token amounts
- [ ] Treasury balance grows from allocations

**Checkpoint:** `cargo test treasury` shows all tests passing

### Task 7: End-to-End Epoch Lifecycle Test

Create a comprehensive test that exercises the full epoch lifecycle across all pallets.

**Test scenario:**
1. Setup: Fund accounts, stake as directors
2. Epoch 0: Bootstrap first epoch, elect directors
3. During epoch: Submit BFT results, record reputation
4. Epoch transition: Elect next directors, transition modes
5. Verify: Treasury accumulations, BFT records, reputation scores

**Files to modify/create:**
- `nsn-chain/test/integration/epoch_lifecycle_test.rs`

**Acceptance criteria:**
- [ ] Full lifecycle completes without errors
- [ ] All intermediate states verified
- [ ] Events emitted in correct order
- [ ] Final state consistent across all pallets

**Checkpoint:** `cargo test epoch_lifecycle` shows all tests passing

## Verification

**Run all integration tests:**
```bash
cd nsn-chain && cargo test --workspace 2>&1 | tee test_results.txt
```

**Expected output:**
- All unit tests continue to pass (regression check)
- New integration tests pass
- No clippy warnings: `cargo clippy --workspace -- -D warnings`

## Success Criteria

- [x] Integration test infrastructure created and compiles
- [x] All 7 task groups have passing tests
- [x] stake→reputation→director→bft chain validated end-to-end
- [x] task-market↔stake integration validated
- [x] treasury work recording validated
- [x] Full epoch lifecycle test passes
- [x] No regressions in existing unit tests
- [x] Code compiles with `cargo build --release`

## Execution Summary

**Completed:** 2026-01-08

**Tests created:** 55 integration tests across 6 test files
- `stake_director_tests.rs` (6 tests)
- `director_reputation_tests.rs` (9 tests)
- `director_bft_tests.rs` (10 tests)
- `task_market_stake_tests.rs` (11 tests)
- `treasury_tests.rs` (14 tests)
- `epoch_lifecycle_test.rs` (5 tests)

**Commits:**
- `9c2baea` - test(1-1): create integration test infrastructure
- `6c81906` - test(1-2..7): add integration tests for Phase 1 pallet validation

**Findings:**
- All cross-pallet trait implementations work correctly
- NodeModeUpdater/NodeRoleUpdater traits properly update stake storage
- LaneNodeProvider correctly filters nodes by mode
- ReputationUpdater callbacks update scores
- BFT storage tracks consensus statistics accurately
- Treasury work accumulation persists across blocks

## Output

**Artifacts:**
- Integration test crate or module in `nsn-chain/test/integration/`
- Test coverage for 5 critical integration chains
- Documentation comments in test files explaining integration points

**Next phase:** Phase 2 (Lane 0 Pipeline Stitching) depends on these integration tests passing to ensure on-chain logic is solid before wiring off-chain components.
