# T062: Task-Market Verification Gate and Escrow Release

## Priority: P0 (Blocker)
## Complexity: 2-3 weeks
## Status: Pending
## Depends On: T010 (Validator Node), T018 (Dual CLIP Verification), T050 (Lane 0 Verification)

---

## Objective

Prevent escrow payout for tasks until verification quorum is met. Executors must not receive payment for unverifiable or incorrect results.

## Background

`pallets/nsn-task-market` currently marks tasks complete and releases escrow immediately upon executor submission, with no verification gate. This enables fraudulent completion.

## Implementation

### Step 1: Add Verification States

Introduce explicit task status transitions:
- `Submitted` (executor submitted output)
- `PendingVerification`
- `Verified`
- `Rejected`

### Step 2: Add Attestation Storage

Store attestation quorum data:
- `Attestations[task_id] -> Vec<Attestation>`
- `RequiredQuorum[task_id] -> u8`
- `VerificationDeadline[task_id] -> BlockNumber`

### Step 3: New Extrinsics

- `submit_result(task_id, output_cid, attestation_cid)` by executor
- `submit_attestation(task_id, validator_id, score, signature)` by validators
- `finalize_task(task_id)` by system once quorum reached or deadline exceeded

### Step 4: Payment Gate

Release escrow only if:
- quorum reached AND verification passes threshold
- otherwise refund requester and slash executor bond if applicable

### Step 5: Events and Errors

Add events for:
- `TaskSubmitted`, `AttestationSubmitted`, `TaskVerified`, `TaskRejected`
- Errors for invalid attestations, duplicate submissions, deadline exceeded

## Acceptance Criteria

- [ ] Escrow never released on `submit_result` alone
- [ ] Task only marked `Verified` after quorum reached
- [ ] Executor payment only after `Verified`
- [ ] `Rejected` tasks refund requester and apply penalties
- [ ] Attestations validated (signature + role + uniqueness)
- [ ] Deadline path works (no infinite pending tasks)
- [ ] All new logic covered by pallet unit tests
- [ ] Runtime builds + tests pass

## Testing

- Unit test: executor submits result, escrow remains reserved
- Unit test: quorum attestations mark task verified + payout
- Unit test: invalid attestation rejected
- Unit test: deadline expires -> reject + refund
- Integration test: validator node submits attestations end-to-end

## Deliverables

1. `pallets/nsn-task-market/src/lib.rs` updates
2. New storage items + events + errors
3. Unit tests for task verification flow
4. Documentation updates for task lifecycle

---

**This task is a production blocker until complete.**
