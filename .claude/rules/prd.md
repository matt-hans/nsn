# Interdimensional Cable Network (ICN)
# Product Requirements Document v8.0

**Version:** 8.0.1-FINAL  
**Date:** 2025-12-24  
**Status:** Strategic Pivot - Moonbeam Pallet Architecture (Enhanced)  
**Classification:** Approved for Development  

---

## Document Control

### Strategic Pivot (v7.0.1 → v8.0)

| Aspect | v7.0.1 Approach | v8.0 Strategic Shift | Benefit |
|--------|-----------------|---------------------|---------|
| **On-Chain** | Custom L2 chain + Ethereum anchor | **Moonbeam custom pallets** | No coretime cost, shared security |
| **Timeline** | 9-18 months | **3-6 months to MVP** | 2-3× faster |
| **Cost** | $500k-$1M+ | **$80k-$200k** | 5-10× cheaper |
| **Security** | Self-managed | **Polkadot relay chain** | Battle-tested infrastructure |
| **Governance** | Custom DAO | **Moonbeam OpenGov** | Proven governance model |
| **Token** | Custom chain native | **ERC-20 + Substrate asset** | EVM compatibility |

### v8.0.1 Enhancements

| Enhancement | Section | Description |
|-------------|---------|-------------|
| **BFT Challenge Period** | §3.3 | On-chain dispute mechanism with 50-block window, stake slashing for fraud |
| **VRF Election Randomness** | §3.3 | Cryptographically secure director selection using Moonbeam VRF |
| **Governance-Adjustable Retention** | §3.2 | Reputation pruning period now a governance parameter |
| **Off-Chain Reputation Batching** | §3.2, §8.1 | TPS optimization via aggregated events |
| **Stake-Weighted Audit Probability** | §3.4 | Higher stake = lower audit frequency |
| **Reputation-Integrated GossipSub** | §17.3 | On-chain reputation influences P2P peer scoring |
| **Dual CLIP Self-Verification** | §12.2 | Directors use CLIP-B + CLIP-L ensemble before BFT |
| **Enhanced Risk Mitigations** | §8 | Detailed strategies for governance, TPS, regulatory risks |

### Revision History

| Version | Date | Changes |
|---------|------|---------|
| 7.0.1 | 2025-12-23 | True decentralization: Multi-Director BFT, NAT traversal, Global reputation |
| 8.0 | 2025-12-23 | **Moonbeam pivot**: Custom Rust pallets, Substrate integration, cost optimization |
| 8.0.1 | 2025-12-24 | **Hardening**: BFT challenges, VRF elections, reputation batching, dual CLIP |

---

## 1. Executive Summary

### 1.1 The Cost & Speed Problem

**v7.0.1 Reality Check:**
- Custom L2/parachain requires 9-18 months development
- Estimated cost: $500k-$1M+ (infrastructure, security, collators)
- High operational overhead (governance, validator management)
- Delayed time-to-market risks project viability

**v8.0 Solution:** Deploy ICN core logic as **custom Rust pallets** on **Moonbeam network**.

### 1.2 System Architecture (Revised)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         ICN v8.0 HYBRID ARCHITECTURE                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    POLKADOT RELAY CHAIN                              │    │
│  │                    (Shared Security Layer)                           │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                      MOONBEAM PARACHAIN                              │    │
│  │  ┌─────────────────────────────────────────────────────────────┐    │    │
│  │  │                 ICN CUSTOM PALLETS (Rust/FRAME)              │    │    │
│  │  │                                                              │    │    │
│  │  │  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐         │    │    │
│  │  │  │pallet-icn-   │ │pallet-icn-   │ │pallet-icn-   │         │    │    │
│  │  │  │   stake      │ │  reputation  │ │   pinning    │         │    │    │
│  │  │  └──────────────┘ └──────────────┘ └──────────────┘         │    │    │
│  │  │                                                              │    │    │
│  │  │  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐         │    │    │
│  │  │  │pallet-icn-   │ │pallet-icn-   │ │pallet-icn-   │         │    │    │
│  │  │  │  director    │ │     bft      │ │   treasury   │         │    │    │
│  │  │  └──────────────┘ └──────────────┘ └──────────────┘         │    │    │
│  │  └─────────────────────────────────────────────────────────────┘    │    │
│  │                                                                      │    │
│  │  ┌─────────────────────────────────────────────────────────────┐    │    │
│  │  │                 EVM / FRONTIER (Ethereum Layer)              │    │    │
│  │  │  • ICN Token (ERC-20)   • dApp Contracts   • Precompiles    │    │    │
│  │  └─────────────────────────────────────────────────────────────┘    │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                    │                                        │
│                         On-Chain Events                                     │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                      OFF-CHAIN LAYER (P2P)                           │    │
│  │                                                                      │    │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐    │    │
│  │  │  Director  │  │  Validator │  │  Super-    │  │   Viewer   │    │    │
│  │  │   Nodes    │  │   Nodes    │  │   Nodes    │  │   Nodes    │    │    │
│  │  │ (Vortex)   │  │  (CLIP)    │  │  (Relay)   │  │ (Consumer) │    │    │
│  │  └────────────┘  └────────────┘  └────────────┘  └────────────┘    │    │
│  │                                                                      │    │
│  │  libp2p + QUIC + GossipSub + Kademlia DHT                           │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.3 Key Technical Targets

| Metric | Target | Notes |
|--------|--------|-------|
| **Time to MVP** | 3-6 months | Moonriver → Moonbeam |
| **Development Cost** | $80k-$200k | 2-3 Rust/Substrate devs |
| **On-Chain TPS** | Moonbeam limits (~50 TPS) | Sufficient for ICN events |
| **Shared Security** | Polkadot Relay Chain | $20B+ economic security |
| **EVM Compatibility** | Full (Frontier) | Standard Web3 tooling |

### 1.4 What Stays the Same (from v7.0.1)

All core ICN protocol features are preserved:
- ✅ Multi-Director BFT (3-of-5 consensus)
- ✅ Semantic verification (CLIP ensemble)
- ✅ Static resident VRAM management
- ✅ Hierarchical swarm architecture
- ✅ NAT traversal stack
- ✅ Erasure coding + pinning incentives
- ✅ Reputation decay + anti-cartel mechanics

**Only the on-chain infrastructure changes** - from custom chain to Moonbeam pallets.

---

## 2. Moonbeam Integration Strategy

### 2.1 Why Moonbeam?

| Feature | Benefit for ICN |
|---------|-----------------|
| **Substrate-based** | Full FRAME pallet customization |
| **EVM + Frontier** | Standard dApp tooling (ethers.js, MetaMask) |
| **Polkadot Security** | Inherited from relay chain validators |
| **Active Governance** | OpenGov for runtime upgrades |
| **Moonriver Testnet** | Free testing environment |
| **XCM Support** | Future cross-chain interoperability |

### 2.2 Runtime Extension Architecture

```rust
// moonbeam-runtime/Cargo.toml additions
[dependencies]
pallet-icn-stake = { path = "../pallets/icn-stake" }
pallet-icn-reputation = { path = "../pallets/icn-reputation" }
pallet-icn-pinning = { path = "../pallets/icn-pinning" }
pallet-icn-director = { path = "../pallets/icn-director" }
pallet-icn-bft = { path = "../pallets/icn-bft" }
pallet-icn-treasury = { path = "../pallets/icn-treasury" }
```

```rust
// moonbeam-runtime/src/lib.rs
construct_runtime!(
    pub enum Runtime where
        Block = Block,
        NodeBlock = opaque::Block,
        UncheckedExtrinsic = UncheckedExtrinsic,
    {
        // ... existing Moonbeam pallets ...
        
        // ICN Custom Pallets
        IcnStake: pallet_icn_stake = 100,
        IcnReputation: pallet_icn_reputation = 101,
        IcnPinning: pallet_icn_pinning = 102,
        IcnDirector: pallet_icn_director = 103,
        IcnBft: pallet_icn_bft = 104,
        IcnTreasury: pallet_icn_treasury = 105,
    }
);
```

---

## 3. Custom Pallet Specifications

### 3.1 pallet-icn-stake

**Purpose:** Token staking, slashing, role eligibility, delegation

```rust
#![cfg_attr(not(feature = "std"), no_std)]

use frame_support::{
    decl_module, decl_storage, decl_event, decl_error,
    traits::{Currency, ReservableCurrency, LockableCurrency},
    dispatch::DispatchResult,
};
use frame_system::ensure_signed;
use sp_runtime::traits::Zero;

pub trait Config: frame_system::Config {
    type Event: From<Event<Self>> + Into<<Self as frame_system::Config>::Event>;
    type Currency: ReservableCurrency<Self::AccountId> + LockableCurrency<Self::AccountId>;
}

decl_storage! {
    trait Store for Module<T: Config> as IcnStake {
        /// Staked amounts per account
        pub Stakes get(fn stakes): map hasher(blake2_128_concat) T::AccountId => StakeInfo<T>;
        
        /// Total staked in network
        pub TotalStaked get(fn total_staked): BalanceOf<T>;
        
        /// Stake per region (anti-centralization)
        pub RegionStakes get(fn region_stakes): map hasher(blake2_128_concat) Region => BalanceOf<T>;
        
        /// Delegations: delegator -> (validator, amount)
        pub Delegations get(fn delegations): double_map
            hasher(blake2_128_concat) T::AccountId,
            hasher(blake2_128_concat) T::AccountId
            => BalanceOf<T>;
    }
}

#[derive(Encode, Decode, Clone, PartialEq, Eq, RuntimeDebug, TypeInfo)]
pub struct StakeInfo<T: Config> {
    pub amount: BalanceOf<T>,
    pub locked_until: T::BlockNumber,
    pub role: NodeRole,
    pub region: Region,
    pub delegated_to_me: BalanceOf<T>,
}

#[derive(Encode, Decode, Clone, PartialEq, Eq, RuntimeDebug, TypeInfo)]
pub enum NodeRole {
    Director,    // 100 ICN minimum
    SuperNode,   // 50 ICN minimum
    Validator,   // 10 ICN minimum
    Relay,       // 5 ICN minimum
    None,
}

decl_event! {
    pub enum Event<T> where
        AccountId = <T as frame_system::Config>::AccountId,
        Balance = BalanceOf<T>,
    {
        /// Stake deposited [account, amount, role]
        StakeDeposited(AccountId, Balance, NodeRole),
        /// Stake withdrawn [account, amount]
        StakeWithdrawn(AccountId, Balance),
        /// Stake slashed [account, amount, reason]
        StakeSlashed(AccountId, Balance, SlashReason),
        /// Delegation created [delegator, validator, amount]
        Delegated(AccountId, AccountId, Balance),
    }
}

decl_error! {
    pub enum Error for Module<T: Config> {
        InsufficientStake,
        StakeLocked,
        PerNodeCapExceeded,      // Max 1000 ICN per node
        RegionCapExceeded,       // Max 20% per region
        DelegationCapExceeded,   // Max 5x validator's own stake
        InvalidRegion,
    }
}

decl_module! {
    pub struct Module<T: Config> for enum Call where origin: T::Origin {
        type Error = Error<T>;
        fn deposit_event() = default;
        
        const MinStakeDirector: BalanceOf<T> = 100_000_000_000_000_000_000u128.saturated_into();
        const MinStakeSuperNode: BalanceOf<T> = 50_000_000_000_000_000_000u128.saturated_into();
        const MinStakeValidator: BalanceOf<T> = 10_000_000_000_000_000_000u128.saturated_into();
        const MaxStakePerNode: BalanceOf<T> = 1_000_000_000_000_000_000_000u128.saturated_into();
        const MaxRegionPercent: Percent = Percent::from_percent(20);
        
        /// Deposit stake to participate in network
        #[weight = 10_000]
        pub fn deposit_stake(
            origin,
            amount: BalanceOf<T>,
            lock_blocks: T::BlockNumber,
            region: Region,
        ) -> DispatchResult {
            let who = ensure_signed(origin)?;
            
            // Verify per-node cap
            let current = Self::stakes(&who).amount;
            ensure!(
                current.saturating_add(amount) <= Self::max_stake_per_node(),
                Error::<T>::PerNodeCapExceeded
            );
            
            // Verify per-region cap (20%)
            let region_total = Self::region_stakes(&region).saturating_add(amount);
            let network_total = Self::total_staked().saturating_add(amount);
            ensure!(
                region_total * 100 / network_total <= 20,
                Error::<T>::RegionCapExceeded
            );
            
            // Lock funds
            T::Currency::reserve(&who, amount)?;
            
            // Determine role
            let new_total = current.saturating_add(amount);
            let role = Self::determine_role(new_total);
            
            // Update storage
            let unlock_at = <frame_system::Pallet<T>>::block_number() + lock_blocks;
            Stakes::<T>::mutate(&who, |info| {
                info.amount = new_total;
                info.locked_until = unlock_at;
                info.role = role.clone();
                info.region = region.clone();
            });
            
            TotalStaked::<T>::mutate(|t| *t = t.saturating_add(amount));
            RegionStakes::<T>::mutate(&region, |r| *r = r.saturating_add(amount));
            
            Self::deposit_event(Event::StakeDeposited(who, amount, role));
            Ok(())
        }
        
        /// Delegate stake to a validator
        #[weight = 10_000]
        pub fn delegate(
            origin,
            validator: T::AccountId,
            amount: BalanceOf<T>,
        ) -> DispatchResult {
            let delegator = ensure_signed(origin)?;
            
            // Verify validator has sufficient own stake
            let validator_stake = Self::stakes(&validator);
            ensure!(
                validator_stake.role == NodeRole::Director || 
                validator_stake.role == NodeRole::SuperNode,
                Error::<T>::InsufficientStake
            );
            
            // Verify delegation cap (5x own stake)
            let current_delegated = validator_stake.delegated_to_me;
            ensure!(
                current_delegated.saturating_add(amount) <= validator_stake.amount.saturating_mul(5u32.into()),
                Error::<T>::DelegationCapExceeded
            );
            
            // Reserve delegator funds
            T::Currency::reserve(&delegator, amount)?;
            
            // Update storage
            Delegations::<T>::insert(&delegator, &validator, amount);
            Stakes::<T>::mutate(&validator, |info| {
                info.delegated_to_me = info.delegated_to_me.saturating_add(amount);
            });
            
            Self::deposit_event(Event::Delegated(delegator, validator, amount));
            Ok(())
        }
        
        /// Slash stake for protocol violations (called by other pallets)
        #[weight = 10_000]
        pub fn slash(
            origin,
            offender: T::AccountId,
            amount: BalanceOf<T>,
            reason: SlashReason,
        ) -> DispatchResult {
            // Only callable by ICN pallets (via ensure_root or pallet origin)
            ensure_root(origin)?;
            
            let stake = Self::stakes(&offender);
            let slash_amount = amount.min(stake.amount);
            
            // Slash reserved funds
            let (_, slashed) = T::Currency::slash_reserved(&offender, slash_amount);
            
            // Update storage
            Stakes::<T>::mutate(&offender, |info| {
                info.amount = info.amount.saturating_sub(slashed);
                info.role = Self::determine_role(info.amount);
            });
            TotalStaked::<T>::mutate(|t| *t = t.saturating_sub(slashed));
            
            Self::deposit_event(Event::StakeSlashed(offender, slashed, reason));
            Ok(())
        }
    }
}

impl<T: Config> Module<T> {
    fn determine_role(amount: BalanceOf<T>) -> NodeRole {
        if amount >= Self::min_stake_director() {
            NodeRole::Director
        } else if amount >= Self::min_stake_super_node() {
            NodeRole::SuperNode
        } else if amount >= Self::min_stake_validator() {
            NodeRole::Validator
        } else if amount >= 5_000_000_000_000_000_000u128.saturated_into() {
            NodeRole::Relay
        } else {
            NodeRole::None
        }
    }
}
```

### 3.2 pallet-icn-reputation

**Purpose:** Verifiable reputation events with Merkle proofs, pruning

```rust
#![cfg_attr(not(feature = "std"), no_std)]

use frame_support::{
    decl_module, decl_storage, decl_event, decl_error,
    dispatch::DispatchResult,
    traits::Get,
};
use sp_std::vec::Vec;
use sp_runtime::traits::Hash;

pub trait Config: frame_system::Config {
    type Event: From<Event<Self>> + Into<<Self as frame_system::Config>::Event>;
    type MaxEventsPerBlock: Get<u32>;
    type DefaultRetentionPeriod: Get<Self::BlockNumber>; // ~6 months default
}

decl_storage! {
    trait Store for Module<T: Config> as IcnReputation {
        /// Reputation scores per account
        pub ReputationScores get(fn reputation_scores): 
            map hasher(blake2_128_concat) T::AccountId => ReputationScore;
        
        /// Pending events (batched per block)
        pub PendingEvents get(fn pending_events): Vec<ReputationEvent<T>>;
        
        /// Merkle roots per block (for proof generation)
        pub MerkleRoots get(fn merkle_roots): 
            map hasher(twox_64_concat) T::BlockNumber => T::Hash;
        
        /// Checkpoints for pruning (every 1000 blocks)
        pub Checkpoints get(fn checkpoints):
            map hasher(twox_64_concat) T::BlockNumber => CheckpointData<T>;
        
        /// Governance-adjustable retention period (v8.0.1)
        /// Default: ~6 months (2,592,000 blocks at 6s/block)
        pub RetentionPeriod get(fn retention_period): T::BlockNumber = T::DefaultRetentionPeriod::get();
        
        /// Aggregated events from off-chain (TPS optimization, v8.0.1)
        pub AggregatedEvents get(fn aggregated_events):
            map hasher(blake2_128_concat) T::AccountId => AggregatedReputation;
    }
}

#[derive(Encode, Decode, Clone, PartialEq, Eq, RuntimeDebug, TypeInfo, Default)]
pub struct AggregatedReputation {
    pub net_director_delta: i64,
    pub net_validator_delta: i64,
    pub net_seeder_delta: i64,
    pub event_count: u32,
    pub last_aggregation_block: u64,
}

#[derive(Encode, Decode, Clone, PartialEq, Eq, RuntimeDebug, TypeInfo, Default)]
pub struct ReputationScore {
    pub director_score: u64,
    pub validator_score: u64,
    pub seeder_score: u64,
    pub last_activity: u64,  // Block number
}

impl ReputationScore {
    pub fn total(&self) -> u64 {
        // Weighted: 50% director, 30% validator, 20% seeder
        (self.director_score * 50 + self.validator_score * 30 + self.seeder_score * 20) / 100
    }
    
    pub fn apply_decay(&mut self, current_block: u64, decay_rate: u64) {
        let blocks_inactive = current_block.saturating_sub(self.last_activity);
        let weeks_inactive = blocks_inactive / (7 * 24 * 600); // ~1 block per 6s
        
        if weeks_inactive > 0 {
            let decay_factor = 100u64.saturating_sub(decay_rate * weeks_inactive);
            self.director_score = self.director_score * decay_factor / 100;
            self.validator_score = self.validator_score * decay_factor / 100;
            self.seeder_score = self.seeder_score * decay_factor / 100;
        }
    }
}

#[derive(Encode, Decode, Clone, PartialEq, Eq, RuntimeDebug, TypeInfo)]
pub enum ReputationEventType {
    DirectorSlotAccepted,      // +100
    DirectorSlotRejected,      // -200
    DirectorSlotMissed,        // -150
    ValidatorVoteCorrect,      // +5
    ValidatorVoteIncorrect,    // -10
    SeederChunkServed,         // +1
    PinningAuditPassed,        // +10
    PinningAuditFailed,        // -50
}

#[derive(Encode, Decode, Clone, PartialEq, Eq, RuntimeDebug, TypeInfo)]
pub struct ReputationEvent<T: Config> {
    pub account: T::AccountId,
    pub event_type: ReputationEventType,
    pub slot: u64,
    pub block: T::BlockNumber,
}

decl_event! {
    pub enum Event<T> where
        AccountId = <T as frame_system::Config>::AccountId,
        BlockNumber = <T as frame_system::Config>::BlockNumber,
        Hash = <T as frame_system::Config>::Hash,
    {
        /// Reputation event recorded [account, event_type, slot]
        ReputationRecorded(AccountId, ReputationEventType, u64),
        /// Merkle root published [block, root]
        MerkleRootPublished(BlockNumber, Hash),
        /// Checkpoint created [block, accounts_count]
        CheckpointCreated(BlockNumber, u32),
        /// Old events pruned [before_block, events_pruned]
        EventsPruned(BlockNumber, u32),
    }
}

decl_module! {
    pub struct Module<T: Config> for enum Call where origin: T::Origin {
        type Error = Error<T>;
        fn deposit_event() = default;
        
        /// Record a reputation event (called by other ICN pallets)
        #[weight = 5_000]
        pub fn record_event(
            origin,
            account: T::AccountId,
            event_type: ReputationEventType,
            slot: u64,
        ) -> DispatchResult {
            // Only ICN pallets can call this
            ensure_root(origin)?;
            
            let current_block = <frame_system::Pallet<T>>::block_number();
            
            // Apply score change
            ReputationScores::<T>::mutate(&account, |score| {
                let delta = Self::event_delta(&event_type);
                match event_type {
                    ReputationEventType::DirectorSlotAccepted |
                    ReputationEventType::DirectorSlotRejected |
                    ReputationEventType::DirectorSlotMissed => {
                        score.director_score = score.director_score.saturating_add_signed(delta);
                    },
                    ReputationEventType::ValidatorVoteCorrect |
                    ReputationEventType::ValidatorVoteIncorrect => {
                        score.validator_score = score.validator_score.saturating_add_signed(delta);
                    },
                    ReputationEventType::SeederChunkServed |
                    ReputationEventType::PinningAuditPassed |
                    ReputationEventType::PinningAuditFailed => {
                        score.seeder_score = score.seeder_score.saturating_add_signed(delta);
                    },
                }
                score.last_activity = current_block.saturated_into();
            });
            
            // Add to pending events for Merkle tree
            let event = ReputationEvent {
                account: account.clone(),
                event_type: event_type.clone(),
                slot,
                block: current_block,
            };
            PendingEvents::<T>::mutate(|events| {
                if events.len() < T::MaxEventsPerBlock::get() as usize {
                    events.push(event);
                }
            });
            
            Self::deposit_event(Event::ReputationRecorded(account, event_type, slot));
            Ok(())
        }
        
        /// Called at end of each block to finalize Merkle root
        fn on_finalize(block: T::BlockNumber) {
            let events = PendingEvents::<T>::take();
            if !events.is_empty() {
                // Compute Merkle root of events
                let root = Self::compute_merkle_root(&events);
                MerkleRoots::<T>::insert(block, root);
                Self::deposit_event(Event::MerkleRootPublished(block, root));
            }
            
            // Check if checkpoint needed (every 1000 blocks)
            if block % 1000u32.into() == Zero::zero() {
                Self::create_checkpoint(block);
            }
            
            // Prune old events beyond retention period
            let prune_before = block.saturating_sub(T::RetentionPeriod::get());
            Self::prune_old_events(prune_before);
        }
    }
}

impl<T: Config> Module<T> {
    fn event_delta(event_type: &ReputationEventType) -> i64 {
        match event_type {
            ReputationEventType::DirectorSlotAccepted => 100,
            ReputationEventType::DirectorSlotRejected => -200,
            ReputationEventType::DirectorSlotMissed => -150,
            ReputationEventType::ValidatorVoteCorrect => 5,
            ReputationEventType::ValidatorVoteIncorrect => -10,
            ReputationEventType::SeederChunkServed => 1,
            ReputationEventType::PinningAuditPassed => 10,
            ReputationEventType::PinningAuditFailed => -50,
        }
    }
    
    fn compute_merkle_root(events: &[ReputationEvent<T>]) -> T::Hash {
        let leaves: Vec<T::Hash> = events.iter()
            .map(|e| T::Hashing::hash_of(e))
            .collect();
        Self::build_merkle_tree(&leaves)
    }
    
    fn build_merkle_tree(leaves: &[T::Hash]) -> T::Hash {
        if leaves.is_empty() {
            return T::Hash::default();
        }
        if leaves.len() == 1 {
            return leaves[0];
        }
        
        let mut current = leaves.to_vec();
        while current.len() > 1 {
            let mut next = Vec::new();
            for chunk in current.chunks(2) {
                let combined = if chunk.len() == 2 {
                    T::Hashing::hash_of(&(chunk[0], chunk[1]))
                } else {
                    chunk[0]
                };
                next.push(combined);
            }
            current = next;
        }
        current[0]
    }
    
    fn create_checkpoint(block: T::BlockNumber) {
        let scores: Vec<_> = ReputationScores::<T>::iter().collect();
        let checkpoint = CheckpointData {
            block,
            score_count: scores.len() as u32,
            merkle_root: Self::compute_scores_merkle(&scores),
        };
        Checkpoints::<T>::insert(block, checkpoint);
        Self::deposit_event(Event::CheckpointCreated(block, scores.len() as u32));
    }
    
    fn prune_old_events(before_block: T::BlockNumber) {
        let mut pruned = 0u32;
        for (block, _) in MerkleRoots::<T>::iter() {
            if block < before_block {
                MerkleRoots::<T>::remove(block);
                pruned += 1;
            }
        }
        if pruned > 0 {
            Self::deposit_event(Event::EventsPruned(before_block, pruned));
        }
    }
}
```

### 3.3 pallet-icn-director

**Purpose:** Multi-director election, BFT coordination, cooldowns

```rust
#![cfg_attr(not(feature = "std"), no_std)]

use frame_support::{
    decl_module, decl_storage, decl_event, decl_error,
    dispatch::DispatchResult,
    traits::Randomness,
};

pub trait Config: frame_system::Config + pallet_icn_stake::Config + pallet_icn_reputation::Config {
    type Event: From<Event<Self>> + Into<<Self as frame_system::Config>::Event>;
    type Randomness: Randomness<Self::Hash, Self::BlockNumber>;
}

decl_storage! {
    trait Store for Module<T: Config> as IcnDirector {
        /// Current slot number
        pub CurrentSlot get(fn current_slot): u64;
        
        /// Elected directors for current slot
        pub ElectedDirectors get(fn elected_directors): Vec<T::AccountId>;
        
        /// Director cooldowns (last directed slot)
        pub Cooldowns get(fn cooldowns): 
            map hasher(blake2_128_concat) T::AccountId => u64;
        
        /// Slot results (for reputation updates)
        pub SlotResults get(fn slot_results):
            map hasher(twox_64_concat) u64 => SlotResult<T>;
        
        /// BFT consensus results per slot
        pub BftResults get(fn bft_results):
            map hasher(twox_64_concat) u64 => BftConsensusResult<T>;
        
        /// Challenge period for BFT results (v8.0.1)
        /// Maps slot -> (challenge_deadline, challenger, evidence_hash)
        pub PendingChallenges get(fn pending_challenges):
            map hasher(twox_64_concat) u64 => Option<BftChallenge<T>>;
        
        /// Finalized slots (past challenge period)
        pub FinalizedSlots get(fn finalized_slots):
            map hasher(twox_64_concat) u64 => bool;
    }
}

const DIRECTORS_PER_SLOT: usize = 5;
const BFT_THRESHOLD: usize = 3;
const COOLDOWN_SLOTS: u64 = 20;
const JITTER_FACTOR: u32 = 20; // ±20%
const CHALLENGE_PERIOD_BLOCKS: u32 = 50;  // ~5 minutes for disputes
const CHALLENGE_STAKE: u128 = 25_000_000_000_000_000_000; // 25 ICN bond

#[derive(Encode, Decode, Clone, PartialEq, Eq, RuntimeDebug, TypeInfo)]
pub struct BftChallenge<T: Config> {
    pub slot: u64,
    pub challenger: T::AccountId,
    pub challenge_block: T::BlockNumber,
    pub deadline: T::BlockNumber,
    pub evidence_hash: T::Hash,  // Hash of validator attestations proving fraud
    pub resolved: bool,
}

#[derive(Encode, Decode, Clone, PartialEq, Eq, RuntimeDebug, TypeInfo)]
pub struct SlotResult<T: Config> {
    pub slot: u64,
    pub canonical_director: Option<T::AccountId>,
    pub agreeing_directors: Vec<T::AccountId>,
    pub failed_directors: Vec<T::AccountId>,
    pub timestamp: u64,
}

#[derive(Encode, Decode, Clone, PartialEq, Eq, RuntimeDebug, TypeInfo)]
pub struct BftConsensusResult<T: Config> {
    pub slot: u64,
    pub success: bool,
    pub canonical_hash: T::Hash,  // Hash of CLIP embeddings
    pub attestations: Vec<(T::AccountId, bool)>,
}

decl_event! {
    pub enum Event<T> where
        AccountId = <T as frame_system::Config>::AccountId,
    {
        /// New slot started [slot_number]
        SlotStarted(u64),
        /// Directors elected [slot, directors]
        DirectorsElected(u64, Vec<AccountId>),
        /// BFT consensus reached [slot, canonical_director]
        BftConsensusReached(u64, AccountId),
        /// BFT consensus failed [slot]
        BftConsensusFailed(u64),
        /// Director slashed for failure [director, slot, reason]
        DirectorSlashed(AccountId, u64, SlashReason),
    }
}

decl_module! {
    pub struct Module<T: Config> for enum Call where origin: T::Origin {
        type Error = Error<T>;
        fn deposit_event() = default;
        
        /// Called each block to check for slot transitions
        fn on_initialize(block: T::BlockNumber) -> Weight {
            let block_num: u64 = block.saturated_into();
            
            // Slots are ~45 seconds = ~7.5 blocks at 6s/block
            let slot = block_num / 8; // Approximate
            
            if slot > Self::current_slot() {
                Self::start_new_slot(slot);
            }
            
            0
        }
        
        /// Submit BFT result from off-chain director coordination
        /// Result enters challenge period before finalization (v8.0.1)
        #[weight = 50_000]
        pub fn submit_bft_result(
            origin,
            slot: u64,
            canonical_director: T::AccountId,
            agreeing_directors: Vec<T::AccountId>,
            embeddings_hash: T::Hash,
        ) -> DispatchResult {
            let submitter = ensure_signed(origin)?;
            
            // Verify submitter is an elected director for this slot
            let elected = Self::elected_directors();
            ensure!(elected.contains(&submitter), Error::<T>::NotElectedDirector);
            
            // Verify minimum agreement (3-of-5)
            ensure!(agreeing_directors.len() >= BFT_THRESHOLD, Error::<T>::InsufficientAgreement);
            
            // Store BFT result (PENDING - not yet finalized)
            let result = BftConsensusResult {
                slot,
                success: true,
                canonical_hash: embeddings_hash,
                attestations: agreeing_directors.iter()
                    .map(|d| (d.clone(), true))
                    .collect(),
            };
            BftResults::<T>::insert(slot, result);
            
            // Start challenge period (50 blocks ≈ 5 minutes)
            let current_block = <frame_system::Pallet<T>>::block_number();
            let deadline = current_block + CHALLENGE_PERIOD_BLOCKS.into();
            
            // Mark slot as pending finalization
            FinalizedSlots::<T>::insert(slot, false);
            
            // Update cooldowns
            for director in &agreeing_directors {
                Cooldowns::<T>::insert(director, slot);
            }
            
            Self::deposit_event(Event::BftResultPending(slot, canonical_director, deadline));
            Ok(())
        }
        
        /// Challenge a BFT result during the challenge period (v8.0.1)
        /// Requires stake bond; successful challenges slash directors
        #[weight = 75_000]
        pub fn challenge_bft_result(
            origin,
            slot: u64,
            evidence_hash: T::Hash,  // Hash of validator attestations proving fraud
        ) -> DispatchResult {
            let challenger = ensure_signed(origin)?;
            
            // Verify result exists and is not finalized
            ensure!(BftResults::<T>::contains_key(slot), Error::<T>::ResultNotFound);
            ensure!(!Self::finalized_slots(slot), Error::<T>::AlreadyFinalized);
            ensure!(Self::pending_challenges(slot).is_none(), Error::<T>::ChallengeExists);
            
            // Verify challenger has sufficient stake (25 ICN bond)
            let challenger_stake = pallet_icn_stake::Module::<T>::stakes(&challenger);
            ensure!(
                challenger_stake.amount >= CHALLENGE_STAKE.saturated_into(),
                Error::<T>::InsufficientChallengeStake
            );
            
            // Lock challenge bond
            T::Currency::reserve(&challenger, CHALLENGE_STAKE.saturated_into())?;
            
            let current_block = <frame_system::Pallet<T>>::block_number();
            let challenge = BftChallenge {
                slot,
                challenger: challenger.clone(),
                challenge_block: current_block,
                deadline: current_block + CHALLENGE_PERIOD_BLOCKS.into(),
                evidence_hash,
                resolved: false,
            };
            
            PendingChallenges::<T>::insert(slot, Some(challenge));
            Self::deposit_event(Event::BftChallenged(slot, challenger));
            Ok(())
        }
        
        /// Resolve a challenge with validator attestations (v8.0.1)
        #[weight = 100_000]
        pub fn resolve_challenge(
            origin,
            slot: u64,
            validator_attestations: Vec<(T::AccountId, bool, T::Hash)>, // (validator, agrees_with_challenge, clip_embedding)
        ) -> DispatchResult {
            ensure_root(origin)?;
            
            let mut challenge = Self::pending_challenges(slot)
                .ok_or(Error::<T>::NoChallengeExists)?;
            ensure!(!challenge.resolved, Error::<T>::ChallengeAlreadyResolved);
            
            // Count validator votes (need >50% agreement with challenge)
            let agree_count = validator_attestations.iter()
                .filter(|(_, agrees, _)| *agrees)
                .count();
            let challenge_upheld = agree_count > validator_attestations.len() / 2;
            
            if challenge_upheld {
                // Challenge successful - slash directors, refund challenger
                let result = Self::bft_results(slot);
                for (director, _) in &result.attestations {
                    pallet_icn_stake::Module::<T>::slash(
                        frame_system::RawOrigin::Root.into(),
                        director.clone(),
                        100_000_000_000_000_000_000u128.saturated_into(), // 100 ICN (double penalty)
                        SlashReason::BftFraud,
                    )?;
                }
                
                // Refund challenger bond + reward
                T::Currency::unreserve(&challenge.challenger, CHALLENGE_STAKE.saturated_into());
                // Challenger gets 10% of slashed amount as reward
                let reward = 10_000_000_000_000_000_000u128.saturated_into(); // 10 ICN
                T::Currency::deposit_into_existing(&challenge.challenger, reward)?;
                
                Self::deposit_event(Event::ChallengeUpheld(slot, challenge.challenger.clone()));
            } else {
                // Challenge failed - slash challenger bond
                let (_, slashed) = T::Currency::slash_reserved(
                    &challenge.challenger,
                    CHALLENGE_STAKE.saturated_into()
                );
                
                // Finalize the original result
                Self::finalize_slot(slot)?;
                
                Self::deposit_event(Event::ChallengeRejected(slot, challenge.challenger.clone()));
            }
            
            challenge.resolved = true;
            PendingChallenges::<T>::insert(slot, Some(challenge));
            Ok(())
        }
        
        /// Called each block to finalize unchallenged results past deadline
        fn on_finalize(block: T::BlockNumber) {
            // Finalize slots past challenge period with no active challenges
            for (slot, result) in BftResults::<T>::iter() {
                if Self::finalized_slots(slot) {
                    continue;
                }
                
                if Self::pending_challenges(slot).is_some() {
                    continue; // Has active challenge
                }
                
                // Check if challenge period expired
                let submission_block = result.attestations.len(); // Approximate
                let deadline = block.saturating_sub(CHALLENGE_PERIOD_BLOCKS.into());
                
                // Auto-finalize after challenge period
                let _ = Self::finalize_slot(slot);
            }
        }
    }
}

impl<T: Config> Module<T> {
    fn finalize_slot(slot: u64) -> DispatchResult {
        let result = Self::bft_results(slot);
        
        // Record reputation events for agreeing directors
        for (director, agreed) in &result.attestations {
            if *agreed {
                pallet_icn_reputation::Module::<T>::record_event(
                    frame_system::RawOrigin::Root.into(),
                    director.clone(),
                    ReputationEventType::DirectorSlotAccepted,
                    slot,
                )?;
            }
        }
        
        FinalizedSlots::<T>::insert(slot, true);
        Self::deposit_event(Event::BftConsensusFinalized(slot));
        Ok(())
    }
}

impl<T: Config> Module<T> {
    fn start_new_slot(slot: u64) {
        CurrentSlot::put(slot);
        Self::deposit_event(Event::SlotStarted(slot));
        
        // Elect directors for slot + 2 (pipeline ahead)
        let election_slot = slot + 2;
        let directors = Self::elect_directors(election_slot);
        ElectedDirectors::put(directors.clone());
        
        Self::deposit_event(Event::DirectorsElected(election_slot, directors));
    }
    
    /// Elect directors using VRF-based randomness (v8.0.1)
    /// Moonbeam provides VRF via pallet-randomness (BABE-based)
    fn elect_directors(slot: u64) -> Vec<T::AccountId> {
        // Get all eligible candidates (Director role + not in cooldown)
        // Multi-region requirement: ensure geographic distribution
        let candidates: Vec<_> = pallet_icn_stake::Stakes::<T>::iter()
            .filter(|(account, stake)| {
                stake.role == NodeRole::Director &&
                Self::cooldowns(account) + COOLDOWN_SLOTS < slot
            })
            .collect();
        
        if candidates.len() < DIRECTORS_PER_SLOT {
            // Not enough candidates - use what we have
            return candidates.into_iter().map(|(a, _)| a).collect();
        }
        
        // Calculate weights with reputation, decay, and jitter
        let mut weighted: Vec<(T::AccountId, u64, Region)> = candidates.iter()
            .map(|(account, stake)| {
                let rep = pallet_icn_reputation::Module::<T>::reputation_scores(account);
                let base_weight = rep.total().saturating_add(1);
                
                // Sublinear scaling (sqrt-like via integer math)
                let scaled = Self::isqrt(base_weight);
                
                // Apply jitter (deterministic per slot+account)
                let jitter_seed = T::Hashing::hash_of(&(slot, account));
                let jitter_raw = u32::from_le_bytes(jitter_seed.as_ref()[0..4].try_into().unwrap());
                let jitter_pct = (jitter_raw % (JITTER_FACTOR * 2)) as i64 - JITTER_FACTOR as i64;
                let jittered = scaled as i64 * (100 + jitter_pct) / 100;
                
                (account.clone(), jittered.max(1) as u64, stake.region.clone())
            })
            .collect();
        
        // VRF-based randomness (Moonbeam's pallet-randomness uses BABE VRF)
        // This is cryptographically unpredictable and verifiable
        let mut selected = Vec::new();
        let mut selected_regions: Vec<Region> = Vec::new();
        
        // Use VRF output for secure randomness
        let (vrf_output, _) = T::Randomness::random(&slot.to_le_bytes());
        let mut rng_state: u64 = u64::from_le_bytes(vrf_output.as_ref()[0..8].try_into().unwrap());
        
        for selection_round in 0..DIRECTORS_PER_SLOT {
            if weighted.is_empty() {
                break;
            }
            
            // Boost weight for candidates from under-represented regions
            let total_weight: u64 = weighted.iter()
                .map(|(_, w, region)| {
                    if selected_regions.contains(region) {
                        *w / 2  // Reduce weight if region already represented
                    } else {
                        *w * 2  // Boost weight for new regions
                    }
                })
                .sum();
            
            // VRF-derived selection (each round uses different entropy)
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(selection_round as u64);
            let pick = rng_state % total_weight;
            
            let mut cumulative = 0u64;
            let mut chosen_idx = 0;
            for (i, (_, weight, region)) in weighted.iter().enumerate() {
                let adjusted_weight = if selected_regions.contains(region) {
                    *weight / 2
                } else {
                    *weight * 2
                };
                cumulative += adjusted_weight;
                if pick < cumulative {
                    chosen_idx = i;
                    break;
                }
            }
            
            let (account, _, region) = weighted.remove(chosen_idx);
            selected_regions.push(region);
            selected.push(account);
        }
        
        selected
    }
    
    fn isqrt(n: u64) -> u64 {
        if n == 0 { return 0; }
        let mut x = n;
        let mut y = (x + 1) / 2;
        while y < x {
            x = y;
            y = (x + n / x) / 2;
        }
        x
    }
}
```

### 3.4 pallet-icn-pinning

**Purpose:** Erasure shard pinning deals, rewards, and audits

```rust
#![cfg_attr(not(feature = "std"), no_std)]

use frame_support::{
    decl_module, decl_storage, decl_event, decl_error,
    dispatch::DispatchResult,
    traits::Currency,
};

pub trait Config: frame_system::Config + pallet_icn_stake::Config {
    type Event: From<Event<Self>> + Into<<Self as frame_system::Config>::Event>;
    type Currency: Currency<Self::AccountId>;
}

decl_storage! {
    trait Store for Module<T: Config> as IcnPinning {
        /// Active pinning deals
        pub PinningDeals get(fn pinning_deals):
            map hasher(blake2_128_concat) DealId => PinningDeal<T>;
        
        /// Shard assignments (shard_hash -> pinners)
        pub ShardAssignments get(fn shard_assignments):
            map hasher(blake2_128_concat) ShardHash => Vec<T::AccountId>;
        
        /// Pinner rewards accumulated
        pub PinnerRewards get(fn pinner_rewards):
            map hasher(blake2_128_concat) T::AccountId => BalanceOf<T>;
        
        /// Pending audits
        pub PendingAudits get(fn pending_audits):
            map hasher(blake2_128_concat) AuditId => PinningAudit<T>;
    }
}

type DealId = [u8; 32];
type ShardHash = [u8; 32];
type AuditId = [u8; 32];

const SHARD_REWARD_PER_BLOCK: u128 = 1_000_000_000_000_000; // 0.001 ICN
const AUDIT_SLASH_AMOUNT: u128 = 10_000_000_000_000_000_000; // 10 ICN
const REPLICATION_FACTOR: usize = 5;

// Stake-weighted audit probability (v8.0.1)
// Higher stake = lower audit frequency (trusted pinners)
// Base: 1% per hour, adjusted by stake tier
const BASE_AUDIT_PROBABILITY_PER_HOUR: u32 = 100;  // 1% = 100 basis points
const AUDIT_PROBABILITY_DIVISOR: u32 = 10000;      // Basis points
const MIN_AUDIT_PROBABILITY: u32 = 25;             // 0.25% floor for high-stake nodes
const MAX_AUDIT_PROBABILITY: u32 = 200;            // 2% ceiling for low-stake nodes

#[derive(Encode, Decode, Clone, PartialEq, Eq, RuntimeDebug, TypeInfo)]
pub struct PinningDeal<T: Config> {
    pub deal_id: DealId,
    pub creator: T::AccountId,
    pub shards: Vec<ShardHash>,
    pub created_at: T::BlockNumber,
    pub expires_at: T::BlockNumber,
    pub total_reward: BalanceOf<T>,
    pub status: DealStatus,
}

#[derive(Encode, Decode, Clone, PartialEq, Eq, RuntimeDebug, TypeInfo)]
pub enum DealStatus {
    Active,
    Expired,
    Cancelled,
}

#[derive(Encode, Decode, Clone, PartialEq, Eq, RuntimeDebug, TypeInfo)]
pub struct PinningAudit<T: Config> {
    pub audit_id: AuditId,
    pub pinner: T::AccountId,
    pub shard_hash: ShardHash,
    pub challenge: AuditChallenge,
    pub deadline: T::BlockNumber,
    pub status: AuditStatus,
}

#[derive(Encode, Decode, Clone, PartialEq, Eq, RuntimeDebug, TypeInfo)]
pub struct AuditChallenge {
    pub byte_offset: u32,
    pub byte_length: u32,
    pub nonce: [u8; 16],
}

#[derive(Encode, Decode, Clone, PartialEq, Eq, RuntimeDebug, TypeInfo)]
pub enum AuditStatus {
    Pending,
    Passed,
    Failed,
}

decl_event! {
    pub enum Event<T> where
        AccountId = <T as frame_system::Config>::AccountId,
        Balance = BalanceOf<T>,
    {
        /// Pinning deal created [deal_id, shards_count]
        DealCreated(DealId, u32),
        /// Shard assigned to pinner [shard, pinner]
        ShardAssigned(ShardHash, AccountId),
        /// Audit started [audit_id, pinner, shard]
        AuditStarted(AuditId, AccountId, ShardHash),
        /// Audit completed [audit_id, passed]
        AuditCompleted(AuditId, bool),
        /// Rewards distributed [pinner, amount]
        RewardsDistributed(AccountId, Balance),
    }
}

decl_module! {
    pub struct Module<T: Config> for enum Call where origin: T::Origin {
        type Error = Error<T>;
        fn deposit_event() = default;
        
        /// Create a pinning deal for shards
        #[weight = 50_000]
        pub fn create_deal(
            origin,
            shards: Vec<ShardHash>,
            duration_blocks: T::BlockNumber,
            payment: BalanceOf<T>,
        ) -> DispatchResult {
            let creator = ensure_signed(origin)?;
            
            // Reserve payment
            T::Currency::reserve(&creator, payment)?;
            
            let current_block = <frame_system::Pallet<T>>::block_number();
            let deal_id: DealId = T::Hashing::hash_of(&(&creator, current_block, &shards))
                .as_ref()[0..32].try_into().unwrap();
            
            let deal = PinningDeal {
                deal_id,
                creator: creator.clone(),
                shards: shards.clone(),
                created_at: current_block,
                expires_at: current_block + duration_blocks,
                total_reward: payment,
                status: DealStatus::Active,
            };
            
            PinningDeals::<T>::insert(deal_id, deal);
            
            // Assign shards to pinners (select from eligible super-nodes)
            for shard in &shards {
                let pinners = Self::select_pinners(shard, REPLICATION_FACTOR);
                ShardAssignments::<T>::insert(shard, pinners.clone());
                for pinner in pinners {
                    Self::deposit_event(Event::ShardAssigned(*shard, pinner));
                }
            }
            
            Self::deposit_event(Event::DealCreated(deal_id, shards.len() as u32));
            Ok(())
        }
        
        /// Initiate random audit of a pinner
        #[weight = 20_000]
        pub fn initiate_audit(
            origin,
            pinner: T::AccountId,
            shard_hash: ShardHash,
        ) -> DispatchResult {
            ensure_root(origin)?;
            
            let current_block = <frame_system::Pallet<T>>::block_number();
            let audit_id: AuditId = T::Hashing::hash_of(&(&pinner, &shard_hash, current_block))
                .as_ref()[0..32].try_into().unwrap();
            
            // Generate random challenge
            let (random, _) = T::Randomness::random(&audit_id);
            let challenge = AuditChallenge {
                byte_offset: u32::from_le_bytes(random.as_ref()[0..4].try_into().unwrap()) % 10000,
                byte_length: 64,
                nonce: random.as_ref()[4..20].try_into().unwrap(),
            };
            
            let audit = PinningAudit {
                audit_id,
                pinner: pinner.clone(),
                shard_hash,
                challenge,
                deadline: current_block + 100u32.into(), // ~10 minutes
                status: AuditStatus::Pending,
            };
            
            PendingAudits::<T>::insert(audit_id, audit);
            Self::deposit_event(Event::AuditStarted(audit_id, pinner, shard_hash));
            Ok(())
        }
        
        /// Submit audit proof (called by pinner)
        #[weight = 20_000]
        pub fn submit_audit_proof(
            origin,
            audit_id: AuditId,
            proof: Vec<u8>,
        ) -> DispatchResult {
            let pinner = ensure_signed(origin)?;
            
            let mut audit = Self::pending_audits(&audit_id)
                .ok_or(Error::<T>::AuditNotFound)?;
            ensure!(audit.pinner == pinner, Error::<T>::NotAuditTarget);
            ensure!(audit.status == AuditStatus::Pending, Error::<T>::AuditAlreadyCompleted);
            
            // Verify proof (simplified - real impl would verify merkle proof)
            let expected_hash = T::Hashing::hash_of(&(&audit.shard_hash, &audit.challenge, &proof));
            let valid = proof.len() >= audit.challenge.byte_length as usize;
            
            if valid {
                audit.status = AuditStatus::Passed;
                pallet_icn_reputation::Module::<T>::record_event(
                    frame_system::RawOrigin::Root.into(),
                    pinner.clone(),
                    ReputationEventType::PinningAuditPassed,
                    0,
                )?;
            } else {
                audit.status = AuditStatus::Failed;
                // Slash pinner
                pallet_icn_stake::Module::<T>::slash(
                    frame_system::RawOrigin::Root.into(),
                    pinner.clone(),
                    AUDIT_SLASH_AMOUNT.saturated_into(),
                    SlashReason::AuditFailure,
                )?;
            }
            
            PendingAudits::<T>::insert(audit_id, audit);
            Self::deposit_event(Event::AuditCompleted(audit_id, valid));
            Ok(())
        }
        
        /// Distribute rewards (called periodically)
        fn on_finalize(block: T::BlockNumber) {
            // Every 100 blocks, distribute rewards
            if block % 100u32.into() == Zero::zero() {
                Self::distribute_rewards();
            }
            
            // Check for expired audits
            Self::check_expired_audits(block);
        }
    }
}

impl<T: Config> Module<T> {
    fn select_pinners(shard: &ShardHash, count: usize) -> Vec<T::AccountId> {
        // Select super-nodes with best reputation in different regions
        let candidates: Vec<_> = pallet_icn_stake::Stakes::<T>::iter()
            .filter(|(_, stake)| stake.role == NodeRole::SuperNode)
            .collect();
        
        // Simple selection: take top N by reputation
        let mut sorted: Vec<_> = candidates.iter()
            .map(|(account, _)| {
                let rep = pallet_icn_reputation::Module::<T>::reputation_scores(account);
                (account.clone(), rep.total())
            })
            .collect();
        sorted.sort_by_key(|(_, rep)| core::cmp::Reverse(*rep));
        
        sorted.into_iter().take(count).map(|(a, _)| a).collect()
    }
    
    fn distribute_rewards() {
        for (deal_id, deal) in PinningDeals::<T>::iter() {
            if deal.status != DealStatus::Active {
                continue;
            }
            
            // Calculate per-shard-per-pinner reward
            let total_pinners = deal.shards.len() * REPLICATION_FACTOR;
            let reward_per_pinner = deal.total_reward / (total_pinners as u32).into();
            
            for shard in &deal.shards {
                let pinners = Self::shard_assignments(shard);
                for pinner in pinners {
                    PinnerRewards::<T>::mutate(&pinner, |r| {
                        *r = r.saturating_add(reward_per_pinner);
                    });
                    Self::deposit_event(Event::RewardsDistributed(pinner, reward_per_pinner));
                }
            }
        }
    }
    
    fn check_expired_audits(current_block: T::BlockNumber) {
        for (audit_id, audit) in PendingAudits::<T>::iter() {
            if audit.status == AuditStatus::Pending && current_block > audit.deadline {
                // Audit timed out - slash pinner
                let _ = pallet_icn_stake::Module::<T>::slash(
                    frame_system::RawOrigin::Root.into(),
                    audit.pinner.clone(),
                    AUDIT_SLASH_AMOUNT.saturated_into(),
                    SlashReason::AuditTimeout,
                );
                
                PendingAudits::<T>::mutate(&audit_id, |a| {
                    if let Some(audit) = a {
                        audit.status = AuditStatus::Failed;
                    }
                });
                
                Self::deposit_event(Event::AuditCompleted(audit_id, false));
            }
        }
    }
}
```

---

## 4. Off-Chain Components (Unchanged from v7.0.1)

The off-chain architecture remains identical to v7.0.1:

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Director Nodes** | Rust + Vortex Engine | Generate video using Flux + LivePortrait |
| **Validator Nodes** | Rust + CLIP | Semantic verification |
| **Super-Nodes** | Rust + libp2p | Tier 1 relay + erasure storage |
| **Regional Relays** | Rust + QUIC | Tier 2 distribution |
| **Viewer Clients** | Tauri + React | Consumption + optional seeding |

### 4.1 On-Chain Event Subscription

Off-chain nodes subscribe to Moonbeam events:

```rust
// Off-chain director node subscribing to election events
use subxt::{OnlineClient, PolkadotConfig};

async fn subscribe_to_elections(client: &OnlineClient<PolkadotConfig>) {
    let mut sub = client
        .blocks()
        .subscribe_finalized()
        .await?;
    
    while let Some(block) = sub.next().await {
        let events = block?.events().await?;
        
        for event in events.iter() {
            if let Ok(Some(election)) = event?.as_event::<icn_director::events::DirectorsElected>() {
                let (slot, directors) = election;
                
                if directors.contains(&my_account_id) {
                    log::info!("Elected as director for slot {}", slot);
                    start_generation_pipeline(slot).await;
                }
            }
        }
    }
}
```

### 4.2 BFT Result Submission

```rust
// Submit BFT consensus result to chain
async fn submit_bft_result(
    client: &OnlineClient<PolkadotConfig>,
    slot: u64,
    canonical_director: AccountId,
    agreeing_directors: Vec<AccountId>,
    embeddings_hash: H256,
) -> Result<(), Error> {
    let tx = icn_director::tx()
        .submit_bft_result(slot, canonical_director, agreeing_directors, embeddings_hash);
    
    let signed = client
        .tx()
        .sign_and_submit_then_watch_default(&tx, &my_keypair)
        .await?;
    
    signed.wait_for_finalized_success().await?;
    
    Ok(())
}
```

---

## 5. EVM Integration

### 5.1 ICN Token (ERC-20)

```solidity
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "@openzeppelin/contracts/token/ERC20/ERC20.sol";
import "@openzeppelin/contracts/access/Ownable.sol";

contract ICNToken is ERC20, Ownable {
    // Precompile addresses for Substrate interactions
    address constant STAKE_PRECOMPILE = 0x0000000000000000000000000000000000000900;
    
    constructor() ERC20("Interdimensional Cable Network", "ICN") {
        _mint(msg.sender, 1_000_000_000 * 10**18); // 1B tokens
    }
    
    /// @notice Stake tokens via Substrate pallet
    function stakeForRole(uint256 amount, uint8 role, bytes32 region) external {
        require(balanceOf(msg.sender) >= amount, "Insufficient balance");
        _burn(msg.sender, amount);
        
        // Call Substrate staking precompile
        (bool success,) = STAKE_PRECOMPILE.call(
            abi.encodeWithSignature(
                "deposit_stake(uint256,uint8,bytes32)",
                amount, role, region
            )
        );
        require(success, "Staking failed");
    }
    
    /// @notice Get reputation score for an account
    function getReputation(address account) external view returns (uint256) {
        // Call reputation precompile
        (bool success, bytes memory data) = REPUTATION_PRECOMPILE.staticcall(
            abi.encodeWithSignature("reputation_scores(address)", account)
        );
        require(success, "Query failed");
        return abi.decode(data, (uint256));
    }
}
```

### 5.2 Precompiles for Substrate Access

```rust
// Custom precompile for EVM access to ICN pallets
pub struct IcnStakePrecompile;

impl Precompile for IcnStakePrecompile {
    fn execute(handle: &mut impl PrecompileHandle) -> PrecompileResult {
        let input = handle.input();
        let selector = &input[0..4];
        
        match selector {
            // deposit_stake(uint256,uint8,bytes32)
            [0x12, 0x34, 0x56, 0x78] => {
                let amount = U256::from_big_endian(&input[4..36]);
                let role = input[36];
                let region = &input[37..69];
                
                // Convert to Substrate types and call pallet
                let account = handle.context().caller;
                let substrate_account = Self::evm_to_substrate(account);
                
                pallet_icn_stake::Pallet::<Runtime>::deposit_stake(
                    Origin::signed(substrate_account),
                    amount.as_u128().into(),
                    1000u32.into(), // lock blocks
                    region.try_into().unwrap(),
                ).map_err(|_| PrecompileFailure::Error { exit_status: ExitError::Other("Stake failed".into()) })?;
                
                Ok(PrecompileOutput {
                    exit_status: ExitSucceed::Returned,
                    output: vec![],
                })
            },
            _ => Err(PrecompileFailure::Error { exit_status: ExitError::Other("Unknown selector".into()) }),
        }
    }
}
```

---

## 6. Deployment Phases

### Phase 1: Moonriver Testnet (Weeks 1-8)

| Week | Milestone | Deliverable |
|------|-----------|-------------|
| 1-2 | Fork & Setup | Moonbeam repo forked, dev environment configured |
| 3-4 | Core Pallets | pallet-icn-stake, pallet-icn-reputation implemented |
| 5-6 | Director Logic | pallet-icn-director, pallet-icn-bft implemented |
| 7-8 | Integration | Full runtime deployed to Moonriver testnet |

**Exit Criteria:**
- [ ] All pallets compile and pass unit tests
- [ ] Runtime upgrade deployed to Moonriver
- [ ] Staking → Election → Reputation flow works end-to-end
- [ ] 10+ test nodes participating

### Phase 2: Moonbeam Mainnet (Weeks 9-16)

| Week | Milestone | Deliverable |
|------|-----------|-------------|
| 9-10 | Security Audit | Critical pallets audited |
| 11-12 | Governance Proposal | Runtime upgrade submitted to OpenGov |
| 13-14 | Token Launch | ICN ERC-20 deployed, liquidity bootstrapped |
| 15-16 | Mainnet Launch | Production deployment with 50+ nodes |

**Governance Process:**
1. Submit proposal with runtime Wasm + rationale
2. 7-day voting period (GLMR-weighted)
3. If approved: automatic enactment after 1-day delay
4. Rollback possible via emergency governance

### Phase 3: Post-Launch (Ongoing)

- Monitor performance and governance
- Add features incrementally (new pallets)
- Evaluate migration to dedicated parachain if TPS > 50

---

## 7. Cost & Resource Estimate

| Category | Details | Cost (USD) |
|----------|---------|------------|
| **Development** | 2-3 Rust/Substrate devs × 6 months | $60k-$120k |
| **Security Audit** | 1-2 audits (Oak Security, SRLabs) | $20k-$60k |
| **Infrastructure** | Testnet nodes, CI/CD, monitoring | $2k-$10k |
| **Legal & Token** | Tokenomics review, legal structure | $5k-$10k |
| **Contingency** | 15% buffer | $15k-$30k |
| **TOTAL** | | **$102k-$230k** |

### Comparison with v7.0.1 Custom Chain

| Aspect | v7.0.1 (Custom Chain) | v8.0 (Moonbeam Pallets) |
|--------|----------------------|-------------------------|
| Development | $200k-$400k | $60k-$120k |
| Security | $100k-$200k | $20k-$60k |
| Infrastructure | $100k-$300k/year | $2k-$10k/year |
| Time to MVP | 12-18 months | 3-6 months |
| **Total Year 1** | **$500k-$1M+** | **$100k-$230k** |

---

## 8. Risk Register (Enhanced)

| Risk | Severity | Likelihood | Mitigation | Status |
|------|----------|------------|------------|--------|
| **Governance rejection** | High | Medium | Start with non-critical pallets (treasury first), build GLMR holder support, engage Moonbeam community pre-proposal | Planned |
| **Runtime bugs** | High | Medium | Extensive Moonriver testing, formal verification for critical paths, staged rollout | Planned |
| **Moonbeam limitations** | Medium | Low | Astar as backup parachain; migration path documented | Contingency |
| **TPS bottleneck** | Medium | Medium | Off-chain reputation aggregation, batch events per block (max 50 events/block), monitor early | Planned |
| **Off-chain BFT trust** | High | Medium | On-chain challenge period (50 blocks), validator attestation disputes, stake slashing | **NEW** |
| **Precompile complexity** | Medium | Medium | Phase after core Substrate stability, use OpenZeppelin-audited patterns | Planned |
| **Token regulatory** | High | Low | Legal review, emphasize non-investment utility (network access), avoid yield/APY language | Planned |
| **Director collusion** | High | Low | Multi-region requirement, challenge mechanism, statistical anomaly detection | Planned |

### 8.1 Detailed Mitigation Strategies

#### Moonbeam Governance Risk
```
Strategy: Progressive Integration
├── Phase 1: Deploy treasury pallet only (low risk, high community value)
├── Phase 2: Add reputation pallet (read-only for validators)
├── Phase 3: Add stake + director pallets (full functionality)
└── Rollback: Runtime downgrade path documented, emergency governance available

Backup Plan: Astar Network
├── Astar supports WASM contracts + Substrate pallets
├── Similar EVM compatibility via ink! precompiles
└── Migration requires: re-deploy pallets, migrate state via XCM
```

#### TPS Bottleneck Mitigation
```rust
// Off-chain reputation aggregation
// Instead of 1 event per action, batch and submit once per block
pub struct ReputationAggregator {
    pending_events: HashMap<AccountId, Vec<ReputationDelta>>,
    
    // Batch and submit every N blocks
    const AGGREGATION_BLOCKS: u32 = 10;  // ~60 seconds
    const MAX_EVENTS_PER_BATCH: u32 = 50;
}

impl ReputationAggregator {
    pub fn aggregate(&mut self, account: AccountId, delta: ReputationDelta) {
        self.pending_events.entry(account).or_default().push(delta);
    }
    
    pub fn flush_to_chain(&mut self) -> Vec<AggregatedEvent> {
        // Combine multiple events per account into single on-chain event
        self.pending_events.drain()
            .map(|(account, deltas)| AggregatedEvent {
                account,
                net_director_delta: deltas.iter().map(|d| d.director).sum(),
                net_validator_delta: deltas.iter().map(|d| d.validator).sum(),
                net_seeder_delta: deltas.iter().map(|d| d.seeder).sum(),
                event_count: deltas.len() as u32,
            })
            .take(MAX_EVENTS_PER_BATCH as usize)
            .collect()
    }
}
```

#### Token Regulatory Compliance
```
Utility Token Framework:
├── Primary Use: Network access (staking for roles)
├── NOT Investment: No passive yield, no profit promises
├── Slashing: Active participation required (not "stake and forget")
├── Governance: Functional voting, not value extraction
└── Legal Review: Pre-launch consultation with securities counsel

Language Guidelines:
├── ✅ "Stake to participate as Director"
├── ✅ "Earn rewards for active validation work"
├── ❌ "Earn X% APY on your ICN"
├── ❌ "Investment opportunity"
```

---

## 9. Success Criteria

| Metric | Moonriver (Phase 1) | Moonbeam (Phase 2) |
|--------|---------------------|-------------------|
| Runtime deployed | ✓ | ✓ |
| Staking functional | 100% test coverage | Live with real tokens |
| Elections working | 10+ test nodes | 50+ mainnet nodes |
| BFT consensus | Simulated | Live 3-of-5 rounds |
| Reputation provable | Merkle proofs verified | On-chain + EVM query |
| Community | 50+ testnet users | 500+ mainnet users |

---

## 10. Architecture Decision Records

### ADR-001: Moonbeam over Custom Parachain

**Context:** ICN needs on-chain staking, reputation, and consensus signals.  
**Decision:** Deploy custom pallets on Moonbeam instead of building dedicated parachain.  
**Rationale:** 3-6× faster, 5-10× cheaper, shared security from day 1.  
**Consequences:** Limited to Moonbeam's TPS (~50); must migrate if growth exceeds.

### ADR-002: Hybrid On-Chain/Off-Chain Architecture

**Context:** Full BFT consensus on-chain is too expensive/slow.  
**Decision:** On-chain = signals, slashing, reputation. Off-chain = actual BFT, video generation.  
**Rationale:** Only critical state needs on-chain finality; computation stays off-chain.  
**Consequences:** Off-chain nodes must be trusted to submit accurate BFT results.

### ADR-003: EVM + Substrate Dual Interface

**Context:** Users expect EVM tooling (MetaMask, ethers.js).  
**Decision:** ICN token as ERC-20 with precompiles for Substrate pallet access.  
**Rationale:** Best of both worlds; EVM UX with Substrate power.  
**Consequences:** Precompile development adds complexity.

---

## 11. Development Roadmap (Updated)

### Sprint 1-2: Foundation (Weeks 1-4)
- [ ] Fork Moonbeam repo
- [ ] Implement pallet-icn-stake
- [ ] Implement pallet-icn-reputation
- [ ] Local dev node testing

### Sprint 3-4: Director Logic (Weeks 5-8)
- [ ] Implement pallet-icn-director
- [ ] Implement pallet-icn-bft
- [ ] Off-chain node integration
- [ ] Deploy to Moonriver testnet

### Sprint 5-6: Pinning & Treasury (Weeks 9-12)
- [ ] Implement pallet-icn-pinning
- [ ] Implement pallet-icn-treasury
- [ ] Security audit (critical pallets)
- [ ] EVM precompiles

### Sprint 7-8: Mainnet Launch (Weeks 13-16)
- [ ] Governance proposal submission
- [ ] ICN token launch
- [ ] Production deployment
- [ ] Community onboarding

---

## 12. Vortex Generation Engine (v8.0)

The Vortex engine remains the core off-chain AI generation component, unchanged from v7.0.1.

### 12.1 Static Resident VRAM Layout

**Critical Constraint:** All models remain loaded in VRAM at all times. No swapping.

| Component | Model | Precision | VRAM | Status |
|-----------|-------|-----------|------|--------|
| Actor Generation | Flux-Schnell | NF4 (4-bit) | ~6.0 GB | Resident |
| Video Warping | LivePortrait | FP16 (TensorRT) | ~3.5 GB | Resident |
| Text-to-Speech | Kokoro-82M | FP32 | ~0.4 GB | Resident |
| Semantic Verify (Primary) | CLIP-ViT-B-32 | INT8 | ~0.3 GB | Resident |
| Semantic Verify (Secondary) | CLIP-ViT-L-14 | INT8 | ~0.6 GB | Resident (v8.0.1) |
| System Overhead | PyTorch/CUDA | - | ~1.0 GB | Overhead |
| **TOTAL** | | | **~11.8 GB** | **RTX 3060 12GB** |

**Note (v8.0.1):** The second CLIP model (ViT-L-14) is added for director self-verification. 
This reduces off-chain disputes by ~40% by catching borderline content before BFT submission.
VRAM budget maintained via INT8 quantization of both CLIP models.

### 12.2 Generation Pipeline (Parallelized)

```python
# vortex/pipeline.py
import asyncio
from dataclasses import dataclass
from typing import Optional
import torch

@dataclass
class GenerationResult:
    video_frames: torch.Tensor  # [T, C, H, W]
    audio_waveform: torch.Tensor  # [samples]
    clip_embedding: torch.Tensor  # [512]
    generation_time_ms: int
    slot_id: int

class VortexPipeline:
    """Static-resident AI generation pipeline for Director nodes."""
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        
        # All models loaded once at startup - NEVER unloaded
        self.flux = self._load_flux()
        self.live_portrait = self._load_live_portrait()
        self.kokoro = self._load_kokoro()
        self.clip = self._load_clip()
        
        # Pre-allocate output buffers to avoid fragmentation
        self.actor_buffer = torch.zeros(1, 3, 512, 512, device=device)
        self.audio_buffer = torch.zeros(24000 * 45, device=device)  # 45s @ 24kHz
        self.video_buffer = torch.zeros(45 * 24, 3, 512, 512, device=device)
    
    async def generate_slot(self, recipe: "Recipe") -> GenerationResult:
        """Generate a complete slot (video + audio) from a Recipe."""
        start_time = time.monotonic()
        
        # PHASE 1: Parallel Audio + Actor Generation
        # Audio takes ~2s, Flux takes ~12s - run together
        audio_task = asyncio.create_task(
            self._generate_audio(recipe.audio)
        )
        actor_task = asyncio.create_task(
            self._generate_actor(recipe.visual)
        )
        
        # Wait for audio first (needed for lip-sync)
        audio_waveform = await audio_task
        
        # Wait for actor image
        actor_image = await actor_task
        
        # PHASE 2: Video Warping (sequential - needs both inputs)
        video_frames = await self._warp_video(
            actor_image=actor_image,
            driving_audio=audio_waveform,
            expression=recipe.visual.expression,
            duration_sec=recipe.slot_params.duration_sec
        )
        
        # PHASE 3: Semantic Verification (self-check before broadcast)
        clip_embedding = await self._compute_clip_embedding(
            video_frames=video_frames,
            prompt=recipe.visual.prompt
        )
        
        generation_time = int((time.monotonic() - start_time) * 1000)
        
        return GenerationResult(
            video_frames=video_frames,
            audio_waveform=audio_waveform,
            clip_embedding=clip_embedding,
            generation_time_ms=generation_time,
            slot_id=recipe.slot_id
        )
    
    async def _generate_audio(self, audio_spec: "AudioSpec") -> torch.Tensor:
        """TTS generation using Kokoro-82M."""
        with torch.no_grad():
            return self.kokoro.synthesize(
                text=audio_spec.script,
                voice_id=audio_spec.voice_id,
                speed=audio_spec.speed
            )
    
    async def _generate_actor(self, visual_spec: "VisualSpec") -> torch.Tensor:
        """Actor image generation using Flux-Schnell (NF4)."""
        with torch.no_grad():
            return self.flux.generate(
                prompt=visual_spec.prompt,
                num_inference_steps=4,  # Schnell = fast
                guidance_scale=0.0,
                output=self.actor_buffer
            )
    
    async def _warp_video(
        self,
        actor_image: torch.Tensor,
        driving_audio: torch.Tensor,
        expression: str,
        duration_sec: int
    ) -> torch.Tensor:
        """Video warping using LivePortrait."""
        with torch.no_grad():
            return self.live_portrait.animate(
                source_image=actor_image,
                driving_audio=driving_audio,
                expression_preset=expression,
                fps=24,
                duration=duration_sec,
                output=self.video_buffer[:duration_sec * 24]
            )
    
    async def _compute_clip_embedding(
        self,
        video_frames: torch.Tensor,
        prompt: str
    ) -> DualClipResult:
        """
        Compute CLIP embedding using dual models for self-verification (v8.0.1).
        
        Using both CLIP-B and CLIP-L reduces off-chain disputes by catching
        borderline content before it's submitted to BFT.
        """
        # Sample 5 keyframes for efficiency
        T = video_frames.shape[0]
        indices = torch.linspace(0, T-1, 5).long()
        keyframes = video_frames[indices]
        
        with torch.no_grad():
            # CLIP-ViT-B-32 (primary, fast)
            image_features_b = self.clip_b.encode_image(keyframes)
            text_features_b = self.clip_b.encode_text(prompt)
            score_b = F.cosine_similarity(
                image_features_b.mean(dim=0), 
                text_features_b
            ).item()
            
            # CLIP-ViT-L-14 (secondary, higher quality)
            image_features_l = self.clip_l.encode_image(keyframes)
            text_features_l = self.clip_l.encode_text(prompt)
            score_l = F.cosine_similarity(
                image_features_l.mean(dim=0), 
                text_features_l
            ).item()
            
            # Ensemble score (weighted average)
            ensemble_score = score_b * 0.4 + score_l * 0.6
            
            # Self-check: reject if either model scores below threshold
            self_check_passed = score_b >= 0.70 and score_l >= 0.72
            
            # Combined embedding for BFT exchange
            combined_b = (image_features_b.mean(dim=0) + text_features_b) / 2
            combined_l = (image_features_l.mean(dim=0) + text_features_l) / 2
            
            # Weighted combination for final embedding
            final_embedding = combined_b * 0.4 + combined_l * 0.6
            final_embedding = final_embedding / final_embedding.norm()
            
            return DualClipResult(
                embedding=final_embedding,
                score_clip_b=score_b,
                score_clip_l=score_l,
                ensemble_score=ensemble_score,
                self_check_passed=self_check_passed
            )

@dataclass
class DualClipResult:
    """Result from dual CLIP self-verification."""
    embedding: torch.Tensor      # Combined embedding for BFT
    score_clip_b: float          # CLIP-ViT-B-32 score
    score_clip_l: float          # CLIP-ViT-L-14 score
    ensemble_score: float        # Weighted average
    self_check_passed: bool      # Whether content passes self-check
```

### 12.3 Slot Timing Budget (v8.0)

```
┌──────────────────────────────────────────────────────────────────────────┐
│                    45-SECOND SLOT TIMELINE (OPTIMISTIC)                   │
├───────────────────┬──────────────────────────────────────────────────────┤
│  0-12s            │  GENERATION PHASE                                    │
│    ├─ 0-2s        │    Audio (Kokoro) - parallel                         │
│    └─ 0-12s       │    Actor (Flux) - parallel                           │
│                   │    Video Warp (LivePortrait) - after actor           │
├───────────────────┼──────────────────────────────────────────────────────┤
│  12-17s           │  BFT PHASE (Off-Chain)                               │
│    ├─ 12-14s      │    Directors exchange CLIP embeddings                │
│    ├─ 14-16s      │    Compute agreement matrix (3-of-5)                 │
│    └─ 16-17s      │    Submit result to Moonbeam (1 extrinsic)           │
├───────────────────┼──────────────────────────────────────────────────────┤
│  17-30s           │  PROPAGATION PHASE                                   │
│    ├─ 17-20s      │    Super-nodes download from canonical director      │
│    ├─ 20-25s      │    Regional relays download from super-nodes         │
│    └─ 25-30s      │    Edge viewers download + buffer                    │
├───────────────────┼──────────────────────────────────────────────────────┤
│  30-45s           │  PLAYBACK BUFFER                                     │
│                   │    Viewers have 15s buffer before deadline           │
└───────────────────┴──────────────────────────────────────────────────────┘
```

---

## 13. Security Model (v8.0 - Moonbeam Integrated)

### 13.1 Security Layers

| Layer | Component | Protection |
|-------|-----------|------------|
| 1 | **Polkadot Relay Chain** | $20B+ economic security, shared finality |
| 2 | **Moonbeam Collators** | Block production, censorship resistance |
| 3 | **ICN Staking (Pallet)** | Sybil resistance, role eligibility |
| 4 | **Reputation Slashing** | Malicious director/validator punishment |
| 5 | **CLIP Semantic Verify** | Content quality, policy compliance |
| 6 | **Ed25519 Signatures** | Message authenticity, non-repudiation |
| 7 | **Sandboxed Vortex** | RCE protection, resource isolation |
| 8 | **E2E Encryption** | Viewer privacy (AES-256-GCM) |

### 13.2 On-Chain Security Events

```rust
// Security events recorded on Moonbeam
#[derive(Encode, Decode, Clone, PartialEq, Eq, RuntimeDebug, TypeInfo)]
pub enum SecurityEvent {
    // Stake-related
    StakeSlashed { account: AccountId, amount: Balance, reason: SlashReason },
    StakeLocked { account: AccountId, until_block: BlockNumber },
    
    // BFT-related
    BftViolation { slot: u64, offender: AccountId, violation: BftViolationType },
    ConsensusFailure { slot: u64, directors: Vec<AccountId> },
    
    // Pinning-related
    AuditFailed { pinner: AccountId, shard: ShardHash },
    ShardLost { shard: ShardHash, last_known_pinners: Vec<AccountId> },
    
    // Reputation-related
    ReputationFrozen { account: AccountId, reason: FreezeReason },
    CartelDetected { accounts: Vec<AccountId>, pattern: CartelPattern },
}

#[derive(Encode, Decode, Clone, PartialEq, Eq, RuntimeDebug, TypeInfo)]
pub enum SlashReason {
    BftFailure,           // Director failed to reach consensus
    AuditTimeout,         // Pinner didn't respond to audit
    AuditInvalid,         // Pinner provided wrong data
    MissedSlot,           // Director didn't produce in time
    ContentViolation,     // CLIP score below threshold
}

#[derive(Encode, Decode, Clone, PartialEq, Eq, RuntimeDebug, TypeInfo)]
pub enum BftViolationType {
    DoubleSign,           // Signed two different results for same slot
    InvalidEmbedding,     // Submitted fake CLIP embedding
    CollusionDetected,    // Statistical anomaly in voting patterns
}
```

### 13.3 Off-Chain Security (P2P Layer)

```rust
// libp2p security configuration
use libp2p::{
    noise::NoiseAuthenticated,
    yamux::YamuxConfig,
    identity::Keypair,
};

pub struct SecureP2PConfig {
    // Ed25519 identity (same as on-chain account)
    pub keypair: Keypair,
    
    // Noise protocol for encryption
    pub noise: NoiseAuthenticated<XX, X25519Spec, ()>,
    
    // Connection limits (DoS protection)
    pub max_connections: usize,           // 256
    pub max_connections_per_peer: usize,  // 2
    pub connection_timeout: Duration,      // 30s
    
    // Rate limiting
    pub max_requests_per_minute: u32,     // 100
    pub max_bandwidth_mbps: u32,          // 100
}

impl SecureP2PConfig {
    pub fn new(keypair: Keypair) -> Self {
        Self {
            keypair: keypair.clone(),
            noise: NoiseAuthenticated::xx(&keypair)
                .expect("Noise XX auth failed"),
            max_connections: 256,
            max_connections_per_peer: 2,
            connection_timeout: Duration::from_secs(30),
            max_requests_per_minute: 100,
            max_bandwidth_mbps: 100,
        }
    }
}
```

### 13.4 CLIP Adversarial Hardening

```python
# Adversarial-hardened CLIP verification
class AdversarialHardenedCLIP:
    """Multi-model ensemble with outlier detection."""
    
    MODELS = [
        ("clip-vit-b-32", 0.3),   # Fast, baseline
        ("clip-vit-l-14", 0.5),   # High quality, main weight
        ("clip-rn50", 0.2),       # Different architecture, robustness
    ]
    
    OUTLIER_THRESHOLD = 0.15  # Max deviation from ensemble mean
    MIN_KEYFRAMES = 5
    
    async def verify(
        self,
        video_frames: torch.Tensor,
        prompt: str,
        threshold: float = 0.75
    ) -> VerificationResult:
        # Sample keyframes
        keyframes = self._sample_keyframes(video_frames, self.MIN_KEYFRAMES)
        
        # Ensemble verification
        scores = []
        embeddings = []
        for model_name, weight in self.MODELS:
            model = self.models[model_name]
            score = await self._compute_score(model, keyframes, prompt)
            scores.append((score, weight))
            embeddings.append(model.encode_image(keyframes))
        
        # Weighted average
        weighted_score = sum(s * w for s, w in scores) / sum(w for _, w in scores)
        
        # Outlier detection (check if any model disagrees significantly)
        mean_embedding = torch.stack(embeddings).mean(dim=0)
        max_deviation = max(
            (e - mean_embedding).norm().item()
            for e in embeddings
        )
        
        is_adversarial = max_deviation > self.OUTLIER_THRESHOLD
        
        return VerificationResult(
            score=weighted_score,
            passed=weighted_score >= threshold and not is_adversarial,
            is_adversarial=is_adversarial,
            embeddings=embeddings
        )
```

---

## 14. Tokenomics (ICN Token)

### 14.1 Token Distribution

| Allocation | Percentage | Amount | Vesting |
|------------|------------|--------|---------|
| **Community Rewards** | 40% | 400M ICN | 4-year linear release |
| **Development Fund** | 20% | 200M ICN | 2-year cliff, 2-year vest |
| **Ecosystem Growth** | 15% | 150M ICN | Grants, partnerships |
| **Team & Advisors** | 15% | 150M ICN | 1-year cliff, 3-year vest |
| **Initial Liquidity** | 10% | 100M ICN | Immediate |
| **TOTAL** | 100% | 1B ICN | |

### 14.2 Token Utility

| Use Case | Mechanism |
|----------|-----------|
| **Staking** | Lock ICN to become Director/SuperNode/Validator |
| **Slashing** | Forfeit staked ICN for protocol violations |
| **Delegation** | Delegate ICN to validators for shared rewards |
| **Pinning Rewards** | Earn ICN for storing erasure-coded shards |
| **Governance** | Vote on Moonbeam OpenGov proposals |
| **Gas Fees** | Pay transaction fees (or use GLMR) |

### 14.3 Reward Emission Schedule

```python
# Token emission curve (Python pseudocode)
import math

def annual_emission(year: int) -> int:
    """ICN tokens released per year."""
    base_emission = 100_000_000  # 100M Year 1
    decay_rate = 0.15  # 15% annual decay
    
    return int(base_emission * math.pow(1 - decay_rate, year - 1))

def reward_distribution(emission: int) -> dict:
    """How annual emission is distributed."""
    return {
        "director_rewards": emission * 0.40,   # 40% to directors
        "validator_rewards": emission * 0.25,  # 25% to validators
        "pinner_rewards": emission * 0.20,     # 20% to pinners
        "treasury": emission * 0.15,           # 15% to treasury
    }

# Year 1: 100M ICN emitted
# Year 2: 85M ICN emitted
# Year 3: 72.25M ICN emitted
# ...
# Year 10: 23.2M ICN emitted
```

### 14.4 Staking Requirements

| Role | Minimum Stake | Maximum Stake | Lock Period | Slashing Risk |
|------|---------------|---------------|-------------|---------------|
| **Director** | 100 ICN | 1,000 ICN | 30 days | High (50 ICN per violation) |
| **SuperNode** | 50 ICN | 500 ICN | 14 days | Medium (20 ICN per violation) |
| **Validator** | 10 ICN | 100 ICN | 7 days | Low (5 ICN per violation) |
| **Relay** | 5 ICN | 50 ICN | 3 days | Minimal (1 ICN per violation) |

---

## 15. Observability & Monitoring

### 15.1 Metrics Stack

```yaml
# docker-compose.observability.yml
version: '3.8'

services:
  prometheus:
    image: prom/prometheus:v2.47.0
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
    ports:
      - "9090:9090"
  
  grafana:
    image: grafana/grafana:10.0.0
    volumes:
      - ./grafana/dashboards:/var/lib/grafana/dashboards
    ports:
      - "3000:3000"
  
  jaeger:
    image: jaegertracing/all-in-one:1.50
    ports:
      - "16686:16686"  # UI
      - "4317:4317"    # OTLP gRPC
  
  alertmanager:
    image: prom/alertmanager:v0.26.0
    volumes:
      - ./alertmanager.yml:/etc/alertmanager/alertmanager.yml
```

### 15.2 Key Metrics

```yaml
# prometheus.yml (ICN-specific metrics)
scrape_configs:
  - job_name: 'icn-director'
    static_configs:
      - targets: ['director:9100']
    metrics:
      # Vortex engine
      - icn_vortex_generation_time_seconds
      - icn_vortex_vram_usage_bytes
      - icn_vortex_model_load_time_seconds
      
      # BFT consensus
      - icn_bft_round_duration_seconds
      - icn_bft_agreement_ratio
      - icn_bft_failures_total
      
      # P2P network
      - icn_p2p_connected_peers
      - icn_p2p_bandwidth_bytes_total
      - icn_p2p_message_latency_seconds

  - job_name: 'icn-pallet'
    static_configs:
      - targets: ['moonbeam-node:9615']  # Substrate metrics
    metrics:
      # On-chain state
      - icn_total_staked_tokens
      - icn_active_directors_count
      - icn_reputation_events_per_block
      - icn_slashing_events_total
```

### 15.3 Alerts

```yaml
# alertmanager.yml
groups:
  - name: icn-critical
    rules:
      - alert: DirectorSlotMissed
        expr: increase(icn_bft_failures_total[5m]) > 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Director slot missed"
          
      - alert: StakeConcentration
        expr: |
          (sum by (region) (icn_staked_amount)) / 
          sum(icn_staked_amount) > 0.25
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "Region has >25% of total stake"
          
      - alert: BftLatencyHigh
        expr: icn_bft_round_duration_seconds > 10
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "BFT round taking >10 seconds"
          
      - alert: VortexOOM
        expr: icn_vortex_vram_usage_bytes > 11.5e9
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Vortex VRAM near capacity"
```

---

## 16. CI/CD Pipeline

### 16.1 Pallet Testing

```yaml
# .github/workflows/pallets.yml
name: ICN Pallets CI

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  build-and-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Setup Rust
        uses: dtolnay/rust-action@stable
        with:
          toolchain: nightly-2024-01-01
          target: wasm32-unknown-unknown
      
      - name: Build Pallets
        run: |
          cd pallets
          cargo build --release --all-features
      
      - name: Unit Tests
        run: |
          cargo test --all-features -- --nocapture
      
      - name: Integration Tests
        run: |
          cargo test --features integration-tests
      
      - name: Clippy
        run: |
          cargo clippy --all-features -- -D warnings
      
      - name: Build Runtime Wasm
        run: |
          cargo build --release --target wasm32-unknown-unknown \
            -p moonbeam-runtime
      
      - name: Check Weights
        run: |
          cargo run --release -p icn-weights-check

  security-audit:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Cargo Audit
        run: cargo audit
      
      - name: Cargo Deny
        uses: EmbarkStudios/cargo-deny-action@v1

  deploy-moonriver:
    needs: [build-and-test, security-audit]
    if: github.ref == 'refs/heads/develop'
    runs-on: ubuntu-latest
    steps:
      - name: Build Runtime
        run: |
          cargo build --release --target wasm32-unknown-unknown \
            -p moonbeam-runtime
      
      - name: Submit Runtime Upgrade (Moonriver)
        run: |
          ./scripts/submit-runtime-upgrade.sh \
            --network moonriver \
            --wasm target/wasm32-unknown-unknown/release/moonbeam_runtime.wasm
```

### 16.2 Off-Chain Node Testing

```yaml
# .github/workflows/vortex.yml
name: Vortex Engine CI

on:
  push:
    paths:
      - 'vortex/**'
      - 'off-chain/**'

jobs:
  test-vortex:
    runs-on: [self-hosted, gpu]  # Need GPU runner
    steps:
      - uses: actions/checkout@v4
      
      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      
      - name: Install Dependencies
        run: |
          pip install -r vortex/requirements.txt
      
      - name: Unit Tests
        run: |
          pytest vortex/tests/unit --cov=vortex
      
      - name: Model Loading Test
        run: |
          python -c "
          from vortex.pipeline import VortexPipeline
          p = VortexPipeline()
          print(f'VRAM: {torch.cuda.memory_allocated() / 1e9:.2f} GB')
          assert torch.cuda.memory_allocated() < 12e9
          "
      
      - name: Generation Benchmark
        run: |
          python vortex/benchmarks/slot_generation.py \
            --slots 5 \
            --max-time 15
```

---

## 17. P2P Network Layer (Off-Chain)

### 17.1 NAT Traversal Stack

```rust
// off-chain/src/nat.rs
use libp2p::{
    autonat::Behaviour as AutoNat,
    dcutr::Behaviour as Dcutr,
    relay::client::Behaviour as RelayClient,
    identify::Behaviour as Identify,
};
use std::time::Duration;

pub struct NATTraversalStack {
    // Connection strategies (tried in order)
    strategies: Vec<ConnectionStrategy>,
    
    // TURN incentives (token rewards)
    turn_reward_per_hour: Decimal,
    
    // Circuit relay priority
    circuit_relay_priority: f64,
}

#[derive(Debug, Clone)]
pub enum ConnectionStrategy {
    Direct,           // No NAT / port forwarded
    STUN,             // UDP hole punching via ICE
    UPnP,             // Automatic port forwarding
    CircuitRelay,     // libp2p relay (incentivized)
    TURN,             // Fallback (expensive)
}

impl NATTraversalStack {
    pub fn new() -> Self {
        Self {
            strategies: vec![
                ConnectionStrategy::Direct,
                ConnectionStrategy::STUN,
                ConnectionStrategy::UPnP,
                ConnectionStrategy::CircuitRelay,  // Prioritized over TURN
                ConnectionStrategy::TURN,
            ],
            turn_reward_per_hour: Decimal::new(1, 2),  // 0.01 ICN
            circuit_relay_priority: 1.5,
        }
    }
    
    pub async fn establish_connection(
        &self,
        target: PeerId,
        target_addr: Multiaddr,
    ) -> Result<Connection, NATError> {
        for strategy in &self.strategies {
            match self.try_strategy(strategy, &target, &target_addr).await {
                Ok(conn) => {
                    log::info!("Connected via {:?}", strategy);
                    return Ok(conn);
                }
                Err(e) => {
                    log::debug!("Strategy {:?} failed: {}", strategy, e);
                    continue;
                }
            }
        }
        Err(NATError::AllStrategiesFailed)
    }
    
    async fn try_strategy(
        &self,
        strategy: &ConnectionStrategy,
        target: &PeerId,
        addr: &Multiaddr,
    ) -> Result<Connection, NATError> {
        match strategy {
            ConnectionStrategy::Direct => {
                // Direct TCP/QUIC connection
                self.dial_direct(target, addr).await
            }
            ConnectionStrategy::STUN => {
                // ICE/STUN hole punching
                let external = self.stun_client.discover_external().await?;
                self.dial_with_external(target, addr, external).await
            }
            ConnectionStrategy::UPnP => {
                // Request port mapping
                let port = self.upnp_client.request_mapping(9000).await?;
                self.dial_direct(target, addr).await
            }
            ConnectionStrategy::CircuitRelay => {
                // Find incentivized relay node
                let relay = self.find_best_relay(target).await?;
                self.dial_via_relay(target, relay).await
            }
            ConnectionStrategy::TURN => {
                // Fallback TURN server
                let turn_server = self.select_turn_server().await?;
                self.dial_via_turn(target, turn_server).await
            }
        }
    }
}
```

### 17.2 Hierarchical Swarm Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           TIER 0: DIRECTORS                             │
│   High-stake nodes with GPU, generate canonical video streams           │
│   Requirements: 100+ ICN staked, RTX 3060+, 100 Mbps symmetric          │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         TIER 1: SUPER-NODES                             │
│   Regional anchors, store erasure shards, relay to tier 2               │
│   Requirements: 50+ ICN staked, 10TB storage, 500 Mbps                  │
│   Regions: NA-WEST, NA-EAST, EU-WEST, EU-EAST, APAC, LATAM, MENA        │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        TIER 2: REGIONAL RELAYS                          │
│   City-level distribution, minimal storage, high bandwidth              │
│   Requirements: 10+ ICN staked, 100 Mbps, low latency                   │
│   Auto-assigned based on latency clustering                             │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          TIER 3: EDGE VIEWERS                           │
│   End users consuming streams, optional seeding                         │
│   Requirements: None (permissionless viewing)                           │
│   Incentive: Seed = earn small ICN rewards                              │
└─────────────────────────────────────────────────────────────────────────┘
```

### 17.3 GossipSub Configuration (v8.0.1 - Reputation Integrated)

```rust
// off-chain/src/gossipsub.rs
use libp2p::gossipsub::{
    Gossipsub, GossipsubConfig, MessageAuthenticity,
    ValidationMode, PeerScoreParams, TopicScoreParams,
    PeerScoreThresholds,
};

/// Build GossipSub with ICN reputation-integrated peer scoring (v8.0.1)
pub fn build_gossipsub(keypair: &Keypair, reputation_oracle: Arc<ReputationOracle>) -> Gossipsub {
    let config = GossipsubConfig::builder()
        // Performance tuning
        .heartbeat_interval(Duration::from_secs(1))
        .validation_mode(ValidationMode::Strict)
        .mesh_n(6)                    // Target peers in mesh
        .mesh_n_low(4)                // Minimum peers
        .mesh_n_high(12)              // Maximum peers
        .gossip_lazy(6)               // Peers for gossip
        .gossip_factor(0.25)          // Fraction to gossip
        .max_transmit_size(16 * 1024 * 1024)  // 16MB (video chunks)
        
        // Flood publishing for low-latency
        .flood_publish(true)
        
        // History settings
        .history_length(12)
        .history_gossip(3)
        
        .build()
        .expect("Valid gossipsub config");
    
    // Peer scoring with on-chain reputation integration (v8.0.1)
    let peer_score_params = PeerScoreParams::builder()
        .topics(vec![
            (RECIPES_TOPIC, TopicScoreParams {
                topic_weight: 1.0,
                first_message_deliveries_weight: 0.5,
                first_message_deliveries_decay: 0.9,
                first_message_deliveries_cap: 100.0,
                invalid_message_deliveries_weight: -10.0,
                invalid_message_deliveries_decay: 0.5,
                ..Default::default()
            }),
            (VIDEO_CHUNKS_TOPIC, TopicScoreParams {
                topic_weight: 2.0,  // More important
                first_message_deliveries_weight: 1.0,
                mesh_message_deliveries_weight: 0.5,
                ..Default::default()
            }),
            (BFT_SIGNALS_TOPIC, TopicScoreParams {
                topic_weight: 3.0,  // Critical for consensus
                invalid_message_deliveries_weight: -20.0,  // Harsh penalty
                ..Default::default()
            }),
        ])
        // Thresholds for peer behavior
        .thresholds(PeerScoreThresholds {
            gossip_threshold: -10.0,      // Below this, no gossip forwarding
            publish_threshold: -50.0,     // Below this, reject publishes
            graylist_threshold: -100.0,   // Below this, ignore entirely
            accept_px_threshold: 0.0,     // Only accept PX from positive peers
            opportunistic_graft_threshold: 5.0,
        })
        .build();
    
    // Custom scoring function that integrates on-chain reputation
    let scoring_fn = move |peer_id: &PeerId| -> f64 {
        // Query on-chain reputation (cached locally)
        let on_chain_rep = reputation_oracle.get_reputation(peer_id);
        
        // Convert to gossipsub score component
        // Scale: 0-1000 on-chain → 0-50 gossipsub boost
        let reputation_boost = (on_chain_rep as f64 / 1000.0) * 50.0;
        
        reputation_boost
    };
    
    Gossipsub::new(
        MessageAuthenticity::Signed(keypair.clone()),
        config,
    )
    .expect("Gossipsub creation failed")
    .with_peer_score(peer_score_params, Default::default())
    .expect("Peer scoring setup failed")
    .with_custom_score(scoring_fn)  // Add on-chain reputation
}

/// Oracle for querying on-chain reputation (cached locally)
pub struct ReputationOracle {
    cache: Arc<RwLock<HashMap<PeerId, u64>>>,
    chain_client: OnlineClient<PolkadotConfig>,
}

impl ReputationOracle {
    pub fn get_reputation(&self, peer_id: &PeerId) -> u64 {
        // Check cache first
        if let Some(score) = self.cache.read().get(peer_id) {
            return *score;
        }
        
        // Default for unknown peers
        100
    }
    
    /// Background task to sync reputation from chain
    pub async fn sync_loop(&self) {
        loop {
            // Fetch all reputation scores from chain every 60s
            if let Ok(scores) = self.fetch_all_reputations().await {
                let mut cache = self.cache.write();
                for (account, score) in scores {
                    if let Some(peer_id) = self.account_to_peer(&account) {
                        cache.insert(peer_id, score.total());
                    }
                }
            }
            tokio::time::sleep(Duration::from_secs(60)).await;
        }
    }
}

// Topic definitions
pub const RECIPES_TOPIC: &str = "/icn/recipes/1.0.0";
pub const VIDEO_CHUNKS_TOPIC: &str = "/icn/video/1.0.0";
pub const BFT_SIGNALS_TOPIC: &str = "/icn/bft/1.0.0";
pub const ATTESTATIONS_TOPIC: &str = "/icn/attestations/1.0.0";
pub const CHALLENGES_TOPIC: &str = "/icn/challenges/1.0.0";  // v8.0.1
```

---

## 18. Bootstrap Protocol

### 18.1 Multi-Layer Bootstrap

```rust
// off-chain/src/bootstrap.rs
use std::collections::HashSet;

pub struct BootstrapProtocol {
    // Trust levels (highest to lowest)
    hardcoded_peers: Vec<Multiaddr>,
    dns_seeds: Vec<String>,
    http_endpoints: Vec<String>,
    dht_topic: String,
    
    // Security
    require_signed_manifests: bool,
    manifest_signers: HashSet<PublicKey>,
}

impl BootstrapProtocol {
    pub fn mainnet() -> Self {
        Self {
            hardcoded_peers: vec![
                "/dns4/boot1.icn.network/tcp/9000/p2p/12D3KooW...".parse().unwrap(),
                "/dns4/boot2.icn.network/tcp/9000/p2p/12D3KooW...".parse().unwrap(),
                "/dns4/boot3.icn.network/tcp/9000/p2p/12D3KooW...".parse().unwrap(),
            ],
            dns_seeds: vec![
                "_icn-bootstrap._tcp.icn.network".to_string(),
            ],
            http_endpoints: vec![
                "https://bootstrap.icn.network/peers.json".to_string(),
                "https://backup.icn.network/peers.json".to_string(),
            ],
            dht_topic: "/icn/bootstrap/dht".to_string(),
            require_signed_manifests: true,
            manifest_signers: Self::trusted_signers(),
        }
    }
    
    pub async fn discover_peers(&self) -> Vec<PeerInfo> {
        let mut discovered = Vec::new();
        
        // Layer 1: Hardcoded peers (always trusted)
        for addr in &self.hardcoded_peers {
            discovered.push(PeerInfo::from_multiaddr(addr, TrustLevel::Hardcoded));
        }
        
        // Layer 2: DNS seeds (verify TXT record signatures)
        for dns_seed in &self.dns_seeds {
            match self.resolve_dns_seed(dns_seed).await {
                Ok(peers) => discovered.extend(peers),
                Err(e) => log::warn!("DNS seed {} failed: {}", dns_seed, e),
            }
        }
        
        // Layer 3: HTTP endpoints (verify JSON signature)
        for endpoint in &self.http_endpoints {
            match self.fetch_http_peers(endpoint).await {
                Ok(peers) => discovered.extend(peers),
                Err(e) => log::warn!("HTTP endpoint {} failed: {}", endpoint, e),
            }
        }
        
        // Layer 4: DHT walk (after connecting to some peers)
        if discovered.len() > 0 {
            let dht_peers = self.dht_discover().await;
            discovered.extend(dht_peers);
        }
        
        // Deduplicate and sort by trust level
        self.deduplicate_and_rank(discovered)
    }
    
    async fn resolve_dns_seed(&self, seed: &str) -> Result<Vec<PeerInfo>, BootstrapError> {
        let records = dns_lookup::lookup_txt(seed).await?;
        
        let mut peers = Vec::new();
        for record in records {
            // Parse: "icn:peer:<multiaddr>:sig:<hex_signature>"
            if let Some(peer_info) = self.parse_dns_record(&record)? {
                if self.verify_manifest_signature(&peer_info) {
                    peers.push(peer_info);
                }
            }
        }
        Ok(peers)
    }
    
    fn verify_manifest_signature(&self, peer: &PeerInfo) -> bool {
        if !self.require_signed_manifests {
            return true;
        }
        
        if let Some(sig) = &peer.signature {
            let message = peer.signing_message();
            self.manifest_signers.iter().any(|pk| pk.verify(&message, sig))
        } else {
            false
        }
    }
}
```

---

## 19. Recipe Schema (v8.0)

```json
{
  "$schema": "https://icn.network/schemas/recipe-v2.json",
  "recipe_id": "01942000-0000-7000-8000-000000000001",
  "version": "2.0.0",
  
  "slot_params": {
    "slot_number": 12345,
    "duration_sec": 45,
    "resolution": "512x512",
    "fps": 24,
    "deadline_unix": 1734984000
  },
  
  "audio_track": {
    "script": "Wubba lubba dub dub! In this dimension, we don't use cash!",
    "voice_id": "rick_c137",
    "speed": 1.1,
    "emotion": "manic"
  },
  
  "visual_track": {
    "prompt": "manic scientist, blue spiked hair, white lab coat, drool, green portal background",
    "negative_prompt": "blurry, low quality, watermark",
    "motion_preset": "excited_nodding",
    "expression_sequence": ["neutral", "excited", "manic", "calm"],
    "camera_motion": "slight_zoom_in"
  },
  
  "semantic_constraints": {
    "min_clip_score": 0.75,
    "banned_concepts": ["violence", "nsfw"],
    "required_concepts": ["scientist", "portal"]
  },
  
  "chain_reference": {
    "network": "moonbeam",
    "director_stake_tx": "0x1234...",
    "election_block": 5678901
  },
  
  "security": {
    "director_id": "12D3KooWAbCdEfGhIjKlMnOpQrStUvWxYz...",
    "ed25519_signature": "0xabcdef...",
    "timestamp": 1734983955
  }
}
```

---

## 20. Appendix A: Pallet Interface Summary

| Pallet | Key Extrinsics | Key Storage | Events |
|--------|----------------|-------------|--------|
| `icn-stake` | `deposit_stake`, `delegate`, `slash`, `withdraw` | `Stakes`, `Delegations`, `RegionStakes` | `StakeDeposited`, `Slashed` |
| `icn-reputation` | `record_event` | `ReputationScores`, `MerkleRoots`, `Checkpoints` | `ReputationRecorded`, `MerkleRootPublished` |
| `icn-director` | `submit_bft_result` | `ElectedDirectors`, `Cooldowns`, `BftResults` | `DirectorsElected`, `BftConsensusReached` |
| `icn-pinning` | `create_deal`, `initiate_audit`, `submit_audit_proof` | `PinningDeals`, `ShardAssignments`, `PendingAudits` | `DealCreated`, `AuditCompleted` |
| `icn-treasury` | `distribute_rewards`, `fund_treasury` | `TreasuryBalance`, `RewardSchedule` | `RewardsDistributed` |
| `icn-bft` | `submit_embeddings_hash` | `EmbeddingsHashes`, `ConsensusRounds` | `ConsensusFinalized` |

---

## 21. Appendix B: Glossary

| Term | Definition |
|------|------------|
| **BFT** | Byzantine Fault Tolerance - consensus requiring 3-of-5 directors to agree |
| **CLIP** | Contrastive Language-Image Pretraining - AI model for semantic verification |
| **Director** | High-stake node that generates canonical video streams |
| **Erasure Coding** | Reed-Solomon (10+4) redundancy for data availability |
| **FRAME** | Substrate's modular runtime framework for building pallets |
| **GLMR** | Moonbeam's native token (used for gas) |
| **Moonbeam** | Polkadot parachain with EVM compatibility |
| **Moonriver** | Moonbeam's canary network on Kusama (testnet) |
| **Pallet** | A Substrate runtime module (like a smart contract) |
| **Recipe** | JSON instruction set for AI generation (~100KB) |
| **Slot** | 45-90 second time window for content generation |
| **SuperNode** | Regional storage and relay infrastructure |
| **Vortex** | ICN's AI generation engine (Flux + LivePortrait + Kokoro) |
| **XCM** | Cross-Consensus Messaging for Polkadot interoperability |

---

## 22. Appendix C: External Dependencies

| Dependency | Version | License | Purpose |
|------------|---------|---------|---------|
| **Substrate** | polkadot-v1.0.0 | GPL-3.0 | Runtime framework |
| **Moonbeam** | v0.35.0 | GPL-3.0 | Base parachain |
| **libp2p** | 0.53.0 | MIT | P2P networking |
| **Flux-Schnell** | NF4 | Apache-2.0 | Image generation |
| **LivePortrait** | v1.0 | MIT | Video warping |
| **Kokoro-82M** | v1.0 | Apache-2.0 | Text-to-speech |
| **CLIP-ViT** | B-32, L-14, RN50 | MIT | Semantic verification |
| **PyTorch** | 2.1+ | BSD | ML runtime |
| **ONNX Runtime** | 1.16+ | MIT | Optimized inference |

---

**Document Status:** APPROVED FOR DEVELOPMENT  
**Architecture:** Moonbeam Custom Pallets (v8.0.1)  
**Strategic Pivot:** Custom Parachain → Moonbeam Runtime Extension  
**Key v8.0.1 Additions:** BFT Challenge Period, VRF Elections, Dual CLIP, Reputation Batching  
**Target MVP:** Q2 2026  
**Estimated Cost:** $100k-$230k  
**Timeline:** 3-6 months (vs. 12-18 months for custom chain)  
**Owner:** Core Engineering Team  
**Last Updated:** 2025-12-24

---

*End of Document*

