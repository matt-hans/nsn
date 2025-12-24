---
id: T035
title: Integration Tests (Staking → Election → Reputation → BFT Flow)
status: pending
priority: 1
agent: backend
dependencies: [T002, T003, T004, T005, T028, T034]
blocked_by: []
created: 2025-12-24T00:00:00Z
updated: 2025-12-24T00:00:00Z
tags: [testing, integration, e2e, quality, phase1]

context_refs:
  - context/acceptance-templates.md

docs_refs:
  - PRD Section 29 (Success Criteria - Moonriver)

est_tokens: 8000
actual_tokens: null
---

## Description

Implement end-to-end integration tests that verify the complete ICN protocol flow: accounts stake tokens → directors elected via VRF → BFT consensus coordinated → reputation updated → challenges resolved. Tests run against local Substrate node with all pallets deployed, simulating 10+ participant nodes.

**Technical Approach:**
- Local Substrate node from T028 docker-compose setup
- Polkadot.js API for extrinsic submission
- Rust integration tests with `subxt` client
- Simulate multi-node scenarios
- Test inter-pallet communication

**Key Flows to Test:**
1. Stake → Role assignment → Regional distribution
2. Director election → Cooldown enforcement → Multi-region requirement
3. BFT submission → Challenge period → Finalization
4. BFT challenge → Validator attestation → Slashing
5. Reputation events → Merkle tree → Decay
6. Pinning deal → Shard assignment → Audit → Rewards

## Acceptance Criteria

- [ ] Integration test suite in `tests/integration/` directory
- [ ] Local Substrate node starts automatically via test harness
- [ ] 10+ test accounts pre-funded and staked
- [ ] Full staking flow tested (deposit → delegate → slash → withdraw)
- [ ] Director election flow tested (elect → cooldown → re-elect)
- [ ] BFT consensus flow tested (submit → challenge → finalize)
- [ ] Reputation flow tested (record → batch → prune)
- [ ] Pinning flow tested (create_deal → assign → audit → distribute_rewards)
- [ ] Tests pass on CI/CD (GitHub Actions)
- [ ] Test duration <5 minutes total

## Test Scenarios

**Test Case 1: Full Staking Flow**
```rust
#[tokio::test]
async fn test_full_staking_flow() {
    let node = start_local_node().await;
    let alice = Account::new("//Alice");
    let bob = Account::new("//Bob");

    // Alice stakes
    alice.deposit_stake(100 * UNIT, Region::NaWest).await?;
    assert_eq!(alice.query_stake().await?.role, NodeRole::Director);

    // Bob delegates to Alice
    bob.delegate(alice.id(), 50 * UNIT).await?;
    assert_eq!(alice.query_stake().await?.delegated_to_me, 50 * UNIT);

    // Alice slashed for violation
    node.sudo_slash(alice.id(), 10 * UNIT, SlashReason::BftFailure).await?;
    assert_eq!(alice.query_stake().await?.amount, 90 * UNIT);

    // Alice withdraws after lock period
    advance_blocks(&node, 1000).await;
    alice.withdraw_stake(90 * UNIT).await?;
    assert_eq!(alice.query_balance().await?, initial_balance - 10 * UNIT);
}
```

**Test Case 2: Director Election Cycle**
```rust
#[tokio::test]
async fn test_director_election_cycle() {
    let node = start_local_node().await;

    // Stake 10 directors across 3 regions
    let directors = setup_multi_region_directors(10).await;

    // Trigger election for slot 100
    advance_to_slot(&node, 100).await;

    // Query elected directors
    let elected = node.query_elected_directors(100).await?;
    assert_eq!(elected.len(), 5);

    // Verify multi-region constraint
    let regions: HashSet<_> = elected.iter().map(|d| d.region).collect();
    assert!(regions.len() >= 3, "At least 3 different regions");

    // Verify no more than 2 from same region
    for region in regions {
        let count = elected.iter().filter(|d| d.region == region).count();
        assert!(count <= 2);
    }

    // Verify cooldown enforcement
    advance_to_slot(&node, 101).await;
    let re_elected = node.query_elected_directors(101).await?;
    for director in &elected {
        assert!(!re_elected.contains(director), "Cooldown violated");
    }
}
```

**Test Case 3: BFT Challenge Resolution**
```rust
#[tokio::test]
async fn test_bft_challenge_resolution() {
    let node = start_local_node().await;

    // Setup 5 directors
    let directors = setup_directors(5).await;

    // Elect for slot 50
    advance_to_slot(&node, 50).await;

    // Directors submit BFT result
    let canonical_hash = H256::random();
    directors[0].submit_bft_result(50, directors[0..3].to_vec(), canonical_hash).await?;

    // Verify result is PENDING
    let result = node.query_bft_result(50).await?;
    assert!(!result.finalized);

    // Challenger submits challenge
    let challenger = Account::new("//Charlie");
    challenger.challenge_bft_result(50, evidence_hash).await?;

    // Validator provides attestations
    let validator_attestations = vec![
        (validator1.id(), false, clip_embedding_1),  // Disagrees with result
        (validator2.id(), false, clip_embedding_2),  // Disagrees
        (validator3.id(), false, clip_embedding_3),  // Disagrees
    ];

    // Resolve challenge (upheld - majority disagree)
    node.sudo_resolve_challenge(50, validator_attestations).await?;

    // Verify directors slashed
    for director in &directors[0..3] {
        let stake = director.query_stake().await?;
        assert_eq!(stake.amount, initial_stake - 100 * UNIT);
    }

    // Verify challenger rewarded
    assert_eq!(challenger.query_balance().await?, initial_balance + 10 * UNIT);
}
```

**Test Case 4: Reputation Decay and Pruning**
```rust
#[tokio::test]
async fn test_reputation_decay_and_pruning() {
    let node = start_local_node().await;
    let alice = Account::new("//Alice");

    // Record reputation event
    node.sudo_record_reputation(alice.id(), DirectorSlotAccepted, 1).await?;
    assert_eq!(alice.query_reputation().await?.director_score, 100);

    // Advance 4 weeks (no activity)
    advance_blocks(&node, 4 * WEEKS_IN_BLOCKS).await;

    // Query reputation (should decay 10% per week)
    let rep = alice.query_reputation().await?;
    // 100 * 0.9^4 ≈ 65
    assert!(rep.director_score >= 64 && rep.director_score <= 66);

    // Advance past retention period (6 months)
    advance_blocks(&node, 6 * MONTHS_IN_BLOCKS).await;

    // Trigger pruning
    node.trigger_pruning().await?;

    // Verify old events removed
    let merkle_roots = node.query_merkle_roots().await?;
    assert_eq!(merkle_roots.len(), 0);  // Pruned
}
```

**Test Case 5: Pinning Audit Flow**
```rust
#[tokio::test]
async fn test_pinning_audit_flow() {
    let node = start_local_node().await;
    let pinner = Account::new("//Alice");

    // Pinner stakes
    pinner.deposit_stake(50 * UNIT, Region::NaWest).await?;

    // Create pinning deal
    let shards = vec![shard_hash_1, shard_hash_2];
    node.create_pinning_deal(shards.clone(), 1000, 100 * UNIT).await?;

    // Verify pinner assigned to shards
    let assignments = node.query_shard_assignments(shard_hash_1).await?;
    assert!(assignments.contains(&pinner.id()));

    // Trigger audit
    node.sudo_initiate_audit(pinner.id(), shard_hash_1).await?;

    // Pinner submits proof
    let proof = generate_merkle_proof(shard_hash_1, challenge);
    pinner.submit_audit_proof(audit_id, proof).await?;

    // Verify audit passed
    let audit = node.query_audit(audit_id).await?;
    assert_eq!(audit.status, AuditStatus::Passed);

    // Verify reputation updated
    let rep = pinner.query_reputation().await?;
    assert_eq!(rep.seeder_score, 10);  // +10 for passing audit
}
```

### Validation Commands

```bash
# Run all integration tests
cargo test --features integration-tests --test '*' -- --nocapture

# Run specific test
cargo test --features integration-tests test_bft_challenge_resolution

# Run with logging
RUST_LOG=debug cargo test --features integration-tests

# Run in CI
./.github/workflows/integration-tests.yml
```

## Dependencies

**Hard Dependencies:**
- [T002-T005] Pallet implementations
- [T028] Local dev environment
- [T034] Unit tests (validate pallets work independently first)

## Design Decisions

**Decision 1: Local Node vs. Moonriver Testnet**
- **Rationale:** Local node is faster (no network latency), deterministic, free
- **Trade-offs:** (+) Fast, isolated. (-) Doesn't test real network conditions

**Decision 2: Rust Tests vs. JavaScript (Polkadot.js)**
- **Rationale:** Rust `subxt` is type-safe, faster, better CI integration
- **Trade-offs:** (+) Type safety. (-) More verbose than JS

## Progress Log

### [2025-12-24] - Task Created
**Dependencies:** T002-T005, T028, T034

## Completion Checklist

- [ ] Integration test suite created
- [ ] All 5 key flows tested
- [ ] Tests pass on CI/CD
- [ ] Test duration <5 minutes

**Definition of Done:**
Integration tests verify full protocol flows (staking → election → BFT → reputation) on local Substrate node with 10+ simulated participants, all tests passing in <5 minutes.
