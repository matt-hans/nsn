# BFT Consensus Constraints

Plugin interface for bft-prover validation.

---

## ICN Consensus Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Director count | 5 per slot | Balance between decentralization and latency |
| Attestation threshold | 3/5 (60%) | Byzantine tolerance with 2 faulty nodes |
| Challenge window | 50 blocks (~5 min) | Time for dispute submission |
| Challenger bond | 25 ICN | Prevent frivolous challenges |
| Director slashing | 100 ICN | Penalty for consensus failures |

---

## L0 Blocking Constraints

- Consensus threshold MUST be > 50% (Byzantine requirement)
- VRF election mechanism MUST be specified for director selection
- No deterministic election that could be gamed

## L1 Critical Constraints

- Attestation aggregation logic MUST be defined
- Slashing conditions MUST be enumerated exhaustively
- Challenge submission and resolution flow required

## L2 Mandatory Checks

- Adversarial scenario tests MUST exist
- Challenge window timing validated in tests
- Director rotation handles edge cases (insufficient stake, offline nodes)

## L3 Standard Guidance

- Edge cases documented (network partitions, simultaneous challenges)
- Formal verification hints where applicable

---

## Validation Commands

| Phase | Trigger | Check |
|-------|---------|-------|
| Planning | Before L1 plan creation | Query invariants for consensus design |
| Implementation | Write to `**/consensus/**/*.rs` | Threshold/invariant validation |
| Final | Before L4 completion | Adversarial tests, challenge flow verification |

## Adversarial Scenarios to Test

- 2/5 directors collude to produce invalid video
- Network partition during BFT round
- Director goes offline after election
- Challenge submitted at block 49 (edge of window)
- Multiple simultaneous challenges for same slot

