# T053: Implement pallet-nsn-agent (Human-AI Operating Model)

## Priority: P1 (Critical Path)
## Complexity: 2 weeks
## Status: Pending
## Depends On: T002 (pallet-nsn-stake)

---

## Objective

Implement a new Substrate pallet that enables humans to operate AI agents on the NSN network with clear delegation, budget controls, and accountability. This is the foundational infrastructure for autonomous AI systems ("digital organisms") to operate on the network.

## Background

The NSN network needs to support:
1. Human operators who fund and configure AI agents
2. AI agents that act semi-autonomously on behalf of operators
3. Clear accountability - operators are ALWAYS liable for agent behavior
4. Bounded autonomy - agents operate within permission and budget constraints
5. Instant control - operators can pause/revoke agents at any time

## Design Reference

See: `docs/plans/2025-12-31-decentralization-remediation-design.md` Section 5

## Core Components

### Storage

```rust
/// Agent registration
#[pallet::storage]
pub type Agents<T: Config> = StorageMap<
    _,
    Blake2_128Concat,
    T::AccountId,  // Agent account (derived)
    AgentInfo<T>,
    OptionQuery,
>;

/// Operator's agents
#[pallet::storage]
pub type OperatorAgents<T: Config> = StorageMap<
    _,
    Blake2_128Concat,
    T::AccountId,  // Operator account
    BoundedVec<T::AccountId, T::MaxAgentsPerOperator>,
    ValueQuery,
>;

/// Budget tracking
#[pallet::storage]
pub type AgentBudgets<T: Config> = StorageMap<
    _,
    Blake2_128Concat,
    T::AccountId,  // Agent account
    BudgetInfo<T>,
    OptionQuery,
>;

/// Global pause flag
#[pallet::storage]
pub type GlobalPaused<T: Config> = StorageValue<_, bool, ValueQuery>;
```

### Types

```rust
#[derive(Encode, Decode, Clone, PartialEq, Eq, RuntimeDebug, TypeInfo, MaxEncodedLen)]
pub struct AgentInfo<T: Config> {
    pub operator: T::AccountId,
    pub index: u32,
    pub permissions: AgentPermissions,
    pub status: AgentStatus,
    pub created_at: BlockNumberFor<T>,
    pub metadata: BoundedVec<u8, T::MaxMetadataLength>,
}

#[derive(Encode, Decode, Clone, PartialEq, Eq, RuntimeDebug, TypeInfo, MaxEncodedLen)]
pub struct AgentPermissions {
    pub can_direct: bool,
    pub can_validate: bool,
    pub can_accept_tasks: bool,
    pub can_submit_bft: bool,
    pub can_manage_storage: bool,
}

#[derive(Encode, Decode, Clone, PartialEq, Eq, RuntimeDebug, TypeInfo, MaxEncodedLen)]
pub enum AgentStatus {
    Active,
    Paused,
    Draining,
    Revoked,
}

#[derive(Encode, Decode, Clone, PartialEq, Eq, RuntimeDebug, TypeInfo, MaxEncodedLen)]
pub struct BudgetInfo<T: Config> {
    pub max_balance: BalanceOf<T>,
    pub max_per_tx: BalanceOf<T>,
    pub max_per_epoch: BalanceOf<T>,
    pub spent_this_epoch: BalanceOf<T>,
    pub last_epoch: u32,
}
```

### Extrinsics

| Extrinsic | Description | Origin |
|-----------|-------------|--------|
| `register_agent` | Create new agent under operator | Operator |
| `pause_agent` | Immediately pause agent | Operator |
| `resume_agent` | Resume paused agent | Operator |
| `revoke_agent` | Permanently revoke agent | Operator |
| `fund_agent` | Add operating balance | Operator |
| `update_permissions` | Modify agent capabilities | Operator |
| `emergency_pause_all` | Pause all operator's agents | Operator or Root |
| `global_agent_pause` | Network-wide pause | Root only |

### Agent Account Derivation

```rust
pub fn derive_agent_account<T: Config>(
    operator: &T::AccountId,
    agent_index: u32,
) -> T::AccountId {
    let entropy = (b"nsn/agent", operator, agent_index).using_encoded(blake2_256);
    T::AccountId::decode(&mut &entropy[..]).expect("32 bytes; qed")
}
```

### Budget Enforcement

```rust
pub fn check_and_spend(
    agent: &T::AccountId,
    amount: BalanceOf<T>,
) -> DispatchResult {
    AgentBudgets::<T>::try_mutate(agent, |maybe_budget| {
        let budget = maybe_budget.as_mut().ok_or(Error::<T>::NoBudgetSet)?;

        // Check per-transaction limit
        ensure!(amount <= budget.max_per_tx, Error::<T>::ExceedsPerTxLimit);

        // Reset epoch counter if needed
        let current_epoch = T::EpochProvider::current_epoch();
        if current_epoch > budget.last_epoch {
            budget.spent_this_epoch = Zero::zero();
            budget.last_epoch = current_epoch;
        }

        // Check per-epoch limit
        ensure!(
            budget.spent_this_epoch.saturating_add(amount) <= budget.max_per_epoch,
            Error::<T>::ExceedsPerEpochLimit
        );

        budget.spent_this_epoch = budget.spent_this_epoch.saturating_add(amount);
        Ok(())
    })
}
```

## Implementation Steps

### Week 1

1. [ ] Create pallet structure with Cargo.toml
2. [ ] Define storage items
3. [ ] Define types (AgentInfo, AgentPermissions, AgentStatus, BudgetInfo)
4. [ ] Implement agent account derivation
5. [ ] Implement register_agent extrinsic
6. [ ] Implement pause/resume/revoke extrinsics
7. [ ] Write unit tests for basic operations

### Week 2

1. [ ] Implement fund_agent extrinsic
2. [ ] Implement budget enforcement
3. [ ] Implement update_permissions extrinsic
4. [ ] Implement emergency controls
5. [ ] Implement helper function: ensure_agent()
6. [ ] Add to runtime configuration
7. [ ] Integration tests
8. [ ] Documentation

## Config Trait

```rust
#[pallet::config]
pub trait Config: frame_system::Config {
    type RuntimeEvent: From<Event<Self>> + IsType<<Self as frame_system::Config>::RuntimeEvent>;
    type Currency: ReservableCurrency<Self::AccountId>;
    type Staking: StakingInterface<Self::AccountId, Balance = BalanceOf<Self>>;
    type EpochProvider: EpochInfo;

    #[pallet::constant]
    type MaxAgentsPerOperator: Get<u32>;

    #[pallet::constant]
    type MinStakePerAgent: Get<BalanceOf<Self>>;

    #[pallet::constant]
    type MaxMetadataLength: Get<u32>;

    type WeightInfo: WeightInfo;
}
```

## Acceptance Criteria

- [ ] Operators can register agents with derived accounts
- [ ] Agents have bounded permissions
- [ ] Budget limits enforced (per-tx, per-epoch)
- [ ] Operators can pause/resume/revoke agents instantly
- [ ] Slashing correctly attributes to operators
- [ ] Emergency controls functional
- [ ] 85%+ test coverage
- [ ] Benchmarks for all extrinsics
- [ ] Integration with pallet-nsn-stake verified

## Security Considerations

1. **Operator Accountability**: Operators MUST be slashed for agent misbehavior
2. **Derived Accounts**: Agent accounts cannot be used without operator control
3. **Budget Enforcement**: Cannot be bypassed by agents
4. **Emergency Controls**: Root can pause all agents network-wide

## Events

```rust
#[pallet::event]
pub enum Event<T: Config> {
    AgentRegistered { operator: T::AccountId, agent: T::AccountId, index: u32 },
    AgentPaused { agent: T::AccountId },
    AgentResumed { agent: T::AccountId },
    AgentRevoked { agent: T::AccountId },
    AgentFunded { agent: T::AccountId, amount: BalanceOf<T> },
    PermissionsUpdated { agent: T::AccountId },
    BudgetWarning { agent: T::AccountId, spent: BalanceOf<T>, limit: BalanceOf<T> },
    AllAgentsPaused { operator: T::AccountId },
    GlobalAgentPause,
}
```

## Deliverables

1. `nsn-chain/pallets/nsn-agent/Cargo.toml`
2. `nsn-chain/pallets/nsn-agent/src/lib.rs`
3. `nsn-chain/pallets/nsn-agent/src/tests.rs`
4. `nsn-chain/pallets/nsn-agent/src/benchmarking.rs`
5. `nsn-chain/pallets/nsn-agent/src/weights.rs`
6. Runtime integration in `nsn-chain/runtime/src/lib.rs`
7. Documentation updates

---

**This pallet is foundational for autonomous AI on NSN.**
