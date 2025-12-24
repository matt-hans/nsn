# Substrate Pallet Constraints

Plugin interface for substrate-architect validation.

---

## L0 Blocking Constraints

- All storage items MUST use Bounded variants (BoundedVec, BoundedBTreeMap)
- Storage migrations MUST have reversibility or explicit justification
- No unbounded iteration in extrinsics

## L1 Critical Constraints

- Each extrinsic MUST have weight annotation plan before implementation
- Storage types MUST derive MaxEncodedLen
- Cross-pallet coupling MUST use traits, not direct references

## L2 Mandatory Checks

- Benchmark functions MUST exist for all extrinsics
- Runtime integration MUST compile in icn-chain-runtime context
- Saturating arithmetic MUST be used (no overflow panics)

## L3 Standard Guidance

- Migration guide required if storage schema changes
- Pallet coupling diagram recommended for cross-pallet dependencies

---

## Validation Commands

| Phase | Trigger | Check |
|-------|---------|-------|
| Planning | Before L1 plan creation | Query constraints for storage/extrinsic design |
| Implementation | Write to `pallets/**/*.rs` | Lightweight macro/storage pattern check |
| Final | Before L4 completion | Full certification with runtime compilation |

## ICN-Specific Patterns

### Stake Pallet
- StakeInfo storage with role enum (Director, SuperNode, Validator, Relay)
- Deposit/withdraw with lock periods
- Delegation with saturation math

### Reputation Pallet
- Merkle-batched events for efficiency
- Decay calculation with configurable retention
- Score thresholds per role

### Director Pallet
- VRF election via ICN Chain randomness source
- BFT result storage with attestation mapping
- Challenge mechanism with 50-block window

