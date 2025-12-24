# Task T039: Cumulus Integration for Parachain Readiness

## Metadata
```yaml
id: T039
title: Cumulus Integration for Parachain Readiness
status: pending
priority: P3
tags: [parachain, cumulus, polkadot, on-chain, phase-c]
estimated_tokens: 12000
actual_tokens: 0
dependencies: [T001, T002, T003, T004, T005, T006, T007, T037]
created_at: 2025-12-24
updated_at: 2025-12-24
```

## Description

**PHASE C TASK** - Add Cumulus integration to ICN Chain for optional migration to Polkadot parachain status. This enables ICN to inherit Polkadot relay chain's ~$20B economic security when adoption justifies the operational complexity and cost.

This task is **NOT required for MVP** and should only be executed when:
- ICN Testnet/Mainnet has proven adoption
- Economic value justifies shared security cost
- Team has capacity for parachain operations

## Business Context

**Why this matters (when needed)**:
- **Shared Security**: Inherit Polkadot's relay chain security (~$20B economic security)
- **Interoperability**: XCM messaging with other parachains
- **Coretime Model**: Access to Polkadot's coretime allocation system
- **Ecosystem Recognition**: Legitimacy as Polkadot ecosystem member

**When to do this**:
- After successful ICN Mainnet launch
- When TVL or adoption justifies parachain costs
- When team has operational capacity

## Acceptance Criteria

1. Cumulus dependencies added to ICN runtime
2. Collator node implementation functional
3. Runtime compatible with relay chain validation
4. XCM configuration for cross-chain messaging (basic)
5. Parachain ID registered (on Rococo testnet initially)
6. ICN Chain syncs with relay chain
7. Blocks validated by relay chain validators
8. Migration path from solochain documented
9. Tested on Rococo/Westend before production

## Technical Implementation

### Step 1: Add Cumulus Dependencies

```toml
# runtime/Cargo.toml
[dependencies]
cumulus-pallet-aura-ext = { workspace = true }
cumulus-pallet-parachain-system = { workspace = true }
cumulus-pallet-xcm = { workspace = true }
cumulus-primitives-core = { workspace = true }
cumulus-primitives-utility = { workspace = true }
parachain-info = { workspace = true }
```

### Step 2: Configure Runtime for Parachain

```rust
// runtime/src/lib.rs
impl cumulus_pallet_parachain_system::Config for Runtime {
    type RuntimeEvent = RuntimeEvent;
    type OnSystemEvent = ();
    type SelfParaId = parachain_info::Pallet<Runtime>;
    type OutboundXcmpMessageSource = XcmpQueue;
    type DmpMessageHandler = DmpQueue;
    type ReservedDmpWeight = ReservedDmpWeight;
    type XcmpMessageHandler = XcmpQueue;
    type ReservedXcmpWeight = ReservedXcmpWeight;
    type CheckAssociatedRelayNumber = RelayNumberStrictlyIncreases;
}

impl parachain_info::Config for Runtime {}
```

### Step 3: Collator Node

```rust
// node/src/service.rs
// Convert from standalone Aura to parachain collator

pub fn start_parachain_node(
    parachain_config: Configuration,
    polkadot_config: Configuration,
    collator_options: CollatorOptions,
    para_id: ParaId,
) -> sc_service::error::Result<TaskManager> {
    // Cumulus collator implementation
    // ...
}
```

### Step 4: XCM Configuration (Basic)

```rust
// runtime/src/xcm_config.rs
parameter_types! {
    pub const RelayNetwork: NetworkId = NetworkId::Polkadot;
    pub RelayChainOrigin: RuntimeOrigin = cumulus_pallet_xcm::Origin::Relay.into();
    pub UniversalLocation: InteriorLocation = [GlobalConsensus(RelayNetwork::get()), Parachain(ParaId::get())].into();
}

pub struct XcmConfig;
impl xcm_executor::Config for XcmConfig {
    type RuntimeCall = RuntimeCall;
    type XcmSender = XcmRouter;
    type AssetTransactor = LocalAssetTransactor;
    // ... additional config
}
```

## Migration Path (Solochain → Parachain)

1. **Preparation**: Ensure all pallets compatible with Cumulus
2. **State Export**: Export solochain state at cutover block
3. **Genesis Update**: Create parachain genesis with exported state
4. **Relay Registration**: Register parachain on relay chain
5. **Collator Launch**: Start collator nodes
6. **Verification**: Verify blocks produced and validated
7. **Cutover**: Deprecate solochain validators

## Dependencies

- **T001-T007**: All pallets must be Cumulus-compatible
- **T037**: Successful testnet operation demonstrates readiness
- **Polkadot SDK**: Cumulus framework

## Risks & Mitigations

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Migration data loss | Critical | Low | Extensive testing on Rococo first |
| Incompatible pallet | High | Medium | Design pallets with Cumulus in mind from T001 |
| Relay chain changes | Medium | Medium | Pin to stable relay version |
| Operational complexity | Medium | High | Document runbooks, gradual rollout |

## Completion Checklist

- [ ] Cumulus dependencies integrated
- [ ] Collator node functional
- [ ] Tested on Rococo testnet
- [ ] XCM configuration complete
- [ ] Migration documentation written
- [ ] Parachain slot acquisition planned

