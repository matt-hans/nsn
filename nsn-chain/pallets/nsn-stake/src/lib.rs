// Copyright 2024 Neural Sovereign Network
// This file is part of NSN Chain.
//
// NSN Chain is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

//! # NSN Stake Pallet
//!
//! Token staking, slashing, role eligibility, and delegation for the Neural Sovereign Network.
//!
//! ## Overview
//!
//! This pallet implements:
//! - Token staking with role-based minimum/maximum stakes
//! - Slashing for protocol violations
//! - Regional anti-centralization (max 20% stake per region)
//! - Delegation with 5× validator stake cap
//!
//! ## Interface
//!
//! ### Dispatchable Functions
//!
//! - `deposit_stake`: Stake NSN tokens for a specific role and region
//! - `delegate`: Delegate stake to a validator
//! - `withdraw_stake`: Withdraw unstaked tokens after lock period
//! - `revoke_delegation`: Remove delegation from a validator
//! - `slash`: Slash tokens for violations (root/governance only)

#![cfg_attr(not(feature = "std"), no_std)]

pub use pallet::*;

mod types;
pub use types::{NodeMode, NodeRole, Region, SlashReason, StakeInfo};

#[cfg(test)]
mod mock;
#[cfg(test)]
mod tests;

#[cfg(feature = "runtime-benchmarks")]
mod benchmarking;

pub mod weights;
pub use weights::WeightInfo;

/// Trait for updating node operational mode from other pallets.
pub trait NodeModeUpdater<AccountId> {
    fn set_mode(account: &AccountId, mode: NodeMode) -> Weight;
}

/// Trait for updating node operational role from other pallets.
pub trait NodeRoleUpdater<AccountId> {
    fn set_role(account: &AccountId, role: NodeRole) -> Weight;
}

#[frame_support::pallet]
pub mod pallet {
    use super::*;
    use frame_support::{
        pallet_prelude::*,
        traits::{
            fungible::{Balanced, Inspect, InspectFreeze, Mutate, MutateFreeze},
            StorageVersion,
        },
    };
    use frame_system::pallet_prelude::*;
    use sp_runtime::traits::{CheckedAdd, Saturating, Zero};

    pub type BalanceOf<T> =
        <<T as Config>::Currency as Inspect<<T as frame_system::Config>::AccountId>>::Balance;

    /// The in-code storage version.
    const STORAGE_VERSION: StorageVersion = StorageVersion::new(0);

    #[pallet::pallet]
    #[pallet::storage_version(STORAGE_VERSION)]
    pub struct Pallet<T>(_);

    /// Configuration trait for the NSN Stake pallet
    #[pallet::config]
    pub trait Config: frame_system::Config<RuntimeEvent: From<Event<Self>>> {
        /// The currency type for staking
        type Currency: Inspect<Self::AccountId>
            + InspectFreeze<Self::AccountId>
            + Mutate<Self::AccountId>
            + MutateFreeze<Self::AccountId, Id = Self::RuntimeFreezeReason>
            + Balanced<Self::AccountId>;

        /// The overarching freeze reason
        type RuntimeFreezeReason: From<FreezeReason>;

        /// Minimum stake for Director role
        #[pallet::constant]
        type MinStakeDirector: Get<BalanceOf<Self>>;

        /// Minimum stake for SuperNode role
        #[pallet::constant]
        type MinStakeSuperNode: Get<BalanceOf<Self>>;

        /// Minimum stake for Validator role
        #[pallet::constant]
        type MinStakeValidator: Get<BalanceOf<Self>>;

        /// Minimum stake for Relay role
        #[pallet::constant]
        type MinStakeRelay: Get<BalanceOf<Self>>;

        /// Maximum stake per node (anti-centralization)
        #[pallet::constant]
        type MaxStakePerNode: Get<BalanceOf<Self>>;

        /// Maximum region percentage (20%)
        #[pallet::constant]
        type MaxRegionPercentage: Get<u32>;

        /// Stake total threshold before enforcing region caps (bootstrap phase)
        #[pallet::constant]
        type RegionCapBootstrapStake: Get<BalanceOf<Self>>;

        /// Delegation multiplier (5× validator stake)
        #[pallet::constant]
        type DelegationMultiplier: Get<u32>;

        /// Maximum delegations per delegator (L0 constraint: bounded)
        #[pallet::constant]
        type MaxDelegationsPerDelegator: Get<u32>;

        /// Maximum delegators per validator (L0 constraint: bounded)
        #[pallet::constant]
        type MaxDelegatorsPerValidator: Get<u32>;

        /// Weight information
        type WeightInfo: WeightInfo;
    }

    /// The reason for freezing funds
    #[pallet::composite_enum]
    pub enum FreezeReason {
        /// Funds frozen for staking
        Staking,
        /// Funds frozen for delegation
        Delegating,
    }

    /// Stakes for each account
    ///
    /// Maps an account to their stake information including amount, lock period,
    /// role, region, and total delegated to them. Uses `ValueQuery` so missing
    /// accounts return an empty `StakeInfo` (all zeros).
    ///
    /// # Storage Key
    /// Blake2_128Concat(AccountId) - safe for user-controlled keys
    #[pallet::storage]
    #[pallet::getter(fn stakes)]
    pub type Stakes<T: Config> = StorageMap<
        _,
        Blake2_128Concat,
        T::AccountId,
        StakeInfo<BalanceOf<T>, BlockNumberFor<T>>,
        ValueQuery,
    >;

    /// Total staked in the network
    ///
    /// Tracks the aggregate amount of NSN tokens staked across all accounts.
    /// Used for:
    /// - Calculating region percentage caps (20% per region)
    /// - Determining bootstrap phase (first MaxStakePerNode)
    /// - Network statistics and slashing adjustments
    ///
    /// # Value
    /// Sum of all `StakeInfo.amount` values
    #[pallet::storage]
    #[pallet::getter(fn total_staked)]
    pub type TotalStaked<T: Config> = StorageValue<_, BalanceOf<T>, ValueQuery>;

    /// Stakes per region (for anti-centralization)
    ///
    /// Maps each geographic region to the total stake amount in that region.
    /// Enforces the 20% regional cap to prevent geographic centralization.
    ///
    /// # Storage Key
    /// Blake2_128Concat(Region) - small enum key, fast hash
    ///
    /// # Regional Caps
    /// No single region may exceed 20% of total network stake
    #[pallet::storage]
    #[pallet::getter(fn region_stakes)]
    pub type RegionStakes<T: Config> =
        StorageMap<_, Blake2_128Concat, Region, BalanceOf<T>, ValueQuery>;

    /// Delegations: delegator → validator → amount
    ///
    /// Tracks stake delegations from delegators to validators. Each delegator
    /// can delegate to multiple validators, but total delegations per delegator
    /// are bounded by `MaxDelegationsPerDelegator`.
    ///
    /// # Storage Keys
    /// - Primary: Blake2_128Concat(DelegatorAccountId)
    /// - Secondary: Blake2_128Concat(ValidatorAccountId)
    ///
    /// # Delegation Caps
    /// Each validator can receive at most 5× their own stake in delegations
    /// (enforced in `delegate()` extrinsic).
    ///
    /// # L0 Compliance
    /// Iteration over delegations is bounded by `MaxDelegationsPerDelegator`.
    /// See `total_delegations_for()` for safe iteration pattern.
    #[pallet::storage]
    #[pallet::getter(fn delegations)]
    pub type Delegations<T: Config> = StorageDoubleMap<
        _,
        Blake2_128Concat,
        T::AccountId, // delegator
        Blake2_128Concat,
        T::AccountId, // validator
        BalanceOf<T>,
        ValueQuery,
    >;

    /// Node operational mode tracking
    ///
    /// Maps each account to its current operational mode in the dual-lane architecture.
    /// Modes control whether a node is ready for Lane 1 compute tasks, draining tasks
    /// before video generation, actively generating video in Lane 0, or offline.
    ///
    /// # Storage Key
    /// Blake2_128Concat(AccountId) - safe for user-controlled keys
    ///
    /// # Default Value
    /// `NodeMode::Lane1Active` for staked nodes
    #[pallet::storage]
    #[pallet::getter(fn node_modes)]
    pub type NodeModes<T: Config> =
        StorageMap<_, Blake2_128Concat, T::AccountId, NodeMode, ValueQuery>;

    /// Events emitted by the pallet
    #[pallet::event]
    #[pallet::generate_deposit(pub(super) fn deposit_event)]
    pub enum Event<T: Config> {
        /// Stake deposited
        StakeDeposited {
            who: T::AccountId,
            amount: BalanceOf<T>,
            role: NodeRole,
        },
        /// Stake withdrawn
        StakeWithdrawn {
            who: T::AccountId,
            amount: BalanceOf<T>,
        },
        /// Stake slashed
        StakeSlashed {
            offender: T::AccountId,
            amount: BalanceOf<T>,
            reason: SlashReason,
        },
        /// Delegation created
        Delegated {
            delegator: T::AccountId,
            validator: T::AccountId,
            amount: BalanceOf<T>,
        },
        /// Delegation revoked
        DelegationRevoked {
            delegator: T::AccountId,
            validator: T::AccountId,
            amount: BalanceOf<T>,
        },
        /// Node operational mode changed
        NodeModeChanged {
            account: T::AccountId,
            new_mode: NodeMode,
        },
    }

    /// Errors returned by the pallet
    #[pallet::error]
    pub enum Error<T> {
        /// Per-node stake cap exceeded (max 1000 NSN)
        PerNodeCapExceeded,
        /// Per-region stake cap exceeded (max 20%)
        RegionCapExceeded,
        /// Delegation cap exceeded (max 5× validator stake)
        DelegationCapExceeded,
        /// Stake is still locked
        StakeLocked,
        /// Insufficient stake to withdraw
        InsufficientStake,
        /// Validator not found (no stake)
        ValidatorNotFound,
        /// Delegation does not exist
        DelegationNotFound,
        /// Arithmetic overflow
        Overflow,
        /// Insufficient balance to stake
        InsufficientBalance,
        /// Account has no stake
        NotStaked,
    }

    #[pallet::hooks]
    impl<T: Config> Hooks<BlockNumberFor<T>> for Pallet<T> {
        /// Block initialization - no operations needed.
        fn on_initialize(_n: BlockNumberFor<T>) -> Weight {
            Weight::zero()
        }
    }

    /// Extrinsic calls
    #[pallet::call]
    impl<T: Config> Pallet<T> {
        /// Deposit stake for a specific role and region
        ///
        /// # Arguments
        /// * `amount` - Amount to stake
        /// * `lock_blocks` - Number of blocks to lock stake
        /// * `region` - Geographic region
        ///
        /// # Errors
        /// * `PerNodeCapExceeded` - Stake would exceed 1000 NSN per node
        /// * `RegionCapExceeded` - Stake would exceed 20% of total in region
        #[pallet::call_index(0)]
        #[pallet::weight(T::WeightInfo::deposit_stake())]
        pub fn deposit_stake(
            origin: OriginFor<T>,
            amount: BalanceOf<T>,
            lock_blocks: BlockNumberFor<T>,
            region: Region,
        ) -> DispatchResult {
            let who = ensure_signed(origin)?;

            // Verify per-node cap (L2: saturating arithmetic)
            let current_stake = Self::stakes(&who);
            let new_total = current_stake
                .amount
                .checked_add(&amount)
                .ok_or(Error::<T>::Overflow)?;
            ensure!(
                new_total <= T::MaxStakePerNode::get(),
                Error::<T>::PerNodeCapExceeded
            );

            // Verify per-region cap (20%)
            let current_region_stake = Self::region_stakes(region);
            let new_region_stake = current_region_stake
                .checked_add(&amount)
                .ok_or(Error::<T>::Overflow)?;
            let current_total = Self::total_staked();
            let new_network_total = current_total
                .checked_add(&amount)
                .ok_or(Error::<T>::Overflow)?;

            // Check: new_region_stake * 100 <= new_network_total * MaxRegionPercentage
            // Bootstrap: enforce region caps only after total stake exceeds threshold.
            let bootstrap_threshold = T::RegionCapBootstrapStake::get();
            if current_total >= bootstrap_threshold {
                let region_percent_scaled = new_region_stake.saturating_mul(100u32.into());
                let max_allowed_scaled =
                    new_network_total.saturating_mul(T::MaxRegionPercentage::get().into());
                ensure!(
                    region_percent_scaled <= max_allowed_scaled,
                    Error::<T>::RegionCapExceeded
                );
            }

            // Verify sufficient balance before freezing (explicit check for safety)
            // Use Expendable since freezing doesn't transfer - just locks existing balance
            let reducible = T::Currency::reducible_balance(
                &who,
                frame_support::traits::tokens::Preservation::Expendable,
                frame_support::traits::tokens::Fortitude::Polite,
            );
            ensure!(reducible >= amount, Error::<T>::InsufficientBalance);

            // Freeze tokens using fungible trait (Moonbeam pattern)
            T::Currency::set_freeze(
                &T::RuntimeFreezeReason::from(FreezeReason::Staking),
                &who,
                new_total,
            )?;

            // Determine role based on new stake amount while preserving director state
            let role = Self::adjust_role_for_amount(current_stake.role.clone(), new_total);

            // Calculate unlock block (never shorten an existing lock)
            let requested_unlock = <frame_system::Pallet<T>>::block_number()
                .checked_add(&lock_blocks)
                .ok_or(Error::<T>::Overflow)?;
            let unlock_at = sp_std::cmp::max(current_stake.locked_until, requested_unlock);

            // Update storage
            Stakes::<T>::insert(
                &who,
                StakeInfo {
                    amount: new_total,
                    locked_until: unlock_at,
                    role: role.clone(),
                    region,
                    delegated_to_me: current_stake.delegated_to_me, // Preserve existing delegations
                },
            );

            TotalStaked::<T>::mutate(|t| *t = t.saturating_add(amount));
            RegionStakes::<T>::mutate(region, |r| *r = r.saturating_add(amount));

            Self::deposit_event(Event::StakeDeposited { who, amount, role });
            Ok(())
        }

        /// Delegate stake to a validator
        ///
        /// # Arguments
        /// * `validator` - Account to delegate to
        /// * `amount` - Amount to delegate
        ///
        /// # Errors
        /// * `ValidatorNotFound` - Validator has no stake
        /// * `DelegationCapExceeded` - Would exceed 5× validator stake
        #[pallet::call_index(1)]
        #[pallet::weight(T::WeightInfo::delegate())]
        pub fn delegate(
            origin: OriginFor<T>,
            validator: T::AccountId,
            amount: BalanceOf<T>,
        ) -> DispatchResult {
            let delegator = ensure_signed(origin)?;

            // Verify validator exists
            let validator_stake = Self::stakes(&validator);
            ensure!(
                !validator_stake.amount.is_zero(),
                Error::<T>::ValidatorNotFound
            );

            // Verify delegation cap (5× validator stake)
            let current_delegated = validator_stake.delegated_to_me;
            let new_delegated = current_delegated
                .checked_add(&amount)
                .ok_or(Error::<T>::Overflow)?;
            let max_delegation = validator_stake
                .amount
                .saturating_mul(T::DelegationMultiplier::get().into());
            ensure!(
                new_delegated <= max_delegation,
                Error::<T>::DelegationCapExceeded
            );

            // Calculate new delegation to this validator
            let current_delegation = Self::delegations(&delegator, &validator);
            let new_delegation_to_validator = current_delegation
                .checked_add(&amount)
                .ok_or(Error::<T>::Overflow)?;

            // FIX VULN-001: Freeze total delegations across ALL validators, not just this one
            // Get current total across all validators, then add new amount
            let current_total_delegations = Self::total_delegations_for(&delegator);
            let new_total_freeze = current_total_delegations
                .checked_add(&amount)
                .ok_or(Error::<T>::Overflow)?;
            T::Currency::set_freeze(
                &T::RuntimeFreezeReason::from(FreezeReason::Delegating),
                &delegator,
                new_total_freeze,
            )?;

            // Update storage
            Delegations::<T>::insert(&delegator, &validator, new_delegation_to_validator);
            Stakes::<T>::mutate(&validator, |info| {
                info.delegated_to_me = info.delegated_to_me.saturating_add(amount);
            });

            Self::deposit_event(Event::Delegated {
                delegator,
                validator,
                amount,
            });
            Ok(())
        }

        /// Withdraw stake after lock period
        ///
        /// # Arguments
        /// * `amount` - Amount to withdraw
        ///
        /// # Errors
        /// * `StakeLocked` - Lock period not expired
        /// * `InsufficientStake` - Not enough stake to withdraw
        #[pallet::call_index(2)]
        #[pallet::weight(T::WeightInfo::withdraw_stake())]
        pub fn withdraw_stake(origin: OriginFor<T>, amount: BalanceOf<T>) -> DispatchResult {
            let who = ensure_signed(origin)?;

            let mut stake_info = Self::stakes(&who);

            // Verify lock period expired
            let current_block = <frame_system::Pallet<T>>::block_number();
            ensure!(
                current_block > stake_info.locked_until,
                Error::<T>::StakeLocked
            );

            // Verify sufficient stake
            ensure!(stake_info.amount >= amount, Error::<T>::InsufficientStake);

            // Calculate new stake
            let new_amount = stake_info.amount.saturating_sub(amount);

            // Enforce delegation cap after reducing validator stake
            let max_delegation = new_amount.saturating_mul(T::DelegationMultiplier::get().into());
            ensure!(
                stake_info.delegated_to_me <= max_delegation,
                Error::<T>::DelegationCapExceeded
            );

            // Update freeze (reduce or clear)
            if new_amount.is_zero() {
                T::Currency::thaw(&T::RuntimeFreezeReason::from(FreezeReason::Staking), &who)?;
            } else {
                T::Currency::set_freeze(
                    &T::RuntimeFreezeReason::from(FreezeReason::Staking),
                    &who,
                    new_amount,
                )?;
            }

            // Update role based on new amount while preserving director state
            let new_role = Self::adjust_role_for_amount(stake_info.role.clone(), new_amount);

            // Update storage
            stake_info.amount = new_amount;
            stake_info.role = new_role;
            Stakes::<T>::insert(&who, stake_info.clone());

            TotalStaked::<T>::mutate(|t| *t = t.saturating_sub(amount));
            RegionStakes::<T>::mutate(stake_info.region, |r| *r = r.saturating_sub(amount));

            Self::deposit_event(Event::StakeWithdrawn { who, amount });
            Ok(())
        }

        /// Revoke delegation from a validator
        ///
        /// # Arguments
        /// * `validator` - Validator to revoke from
        ///
        /// # Errors
        /// * `DelegationNotFound` - No delegation exists
        #[pallet::call_index(3)]
        #[pallet::weight(T::WeightInfo::revoke_delegation())]
        pub fn revoke_delegation(origin: OriginFor<T>, validator: T::AccountId) -> DispatchResult {
            let delegator = ensure_signed(origin)?;

            let amount = Self::delegations(&delegator, &validator);
            ensure!(!amount.is_zero(), Error::<T>::DelegationNotFound);

            // FIX VULN-002: Calculate remaining total delegations AFTER removing current one
            // Remove current delegation first
            Delegations::<T>::remove(&delegator, &validator);

            // Calculate new total freeze (remaining delegations across all other validators)
            let remaining_total = Self::total_delegations_for(&delegator);

            // Update freeze: thaw all if no remaining, otherwise set to remaining amount
            if remaining_total.is_zero() {
                T::Currency::thaw(
                    &T::RuntimeFreezeReason::from(FreezeReason::Delegating),
                    &delegator,
                )?;
            } else {
                T::Currency::set_freeze(
                    &T::RuntimeFreezeReason::from(FreezeReason::Delegating),
                    &delegator,
                    remaining_total,
                )?;
            }

            // Update validator stake
            Stakes::<T>::mutate(&validator, |info| {
                info.delegated_to_me = info.delegated_to_me.saturating_sub(amount);
            });

            Self::deposit_event(Event::DelegationRevoked {
                delegator,
                validator,
                amount,
            });
            Ok(())
        }

        /// Slash stake for protocol violations (root only)
        ///
        /// # Arguments
        /// * `offender` - Account to slash
        /// * `amount` - Amount to slash
        /// * `reason` - Reason for slashing
        ///
        /// # Note
        /// Only callable by root (governance or other pallets)
        #[pallet::call_index(4)]
        #[pallet::weight(T::WeightInfo::slash())]
        pub fn slash(
            origin: OriginFor<T>,
            offender: T::AccountId,
            amount: BalanceOf<T>,
            reason: SlashReason,
        ) -> DispatchResult {
            ensure_root(origin)?;
            Self::slash_internal(&offender, amount, reason)
        }

        /// Set node operational mode (root only)
        ///
        /// Updates the operational mode for a staked node in the dual-lane architecture.
        /// Called by the director pallet when electing directors for an epoch.
        ///
        /// # Arguments
        /// * `account` - Account whose mode to change
        /// * `mode` - New operational mode
        ///
        /// # Errors
        /// * `NotStaked` - Account has no stake
        ///
        /// # Note
        /// Only callable by root (director pallet or governance)
        #[pallet::call_index(5)]
        #[pallet::weight(T::WeightInfo::set_node_mode())]
        pub fn set_node_mode(
            origin: OriginFor<T>,
            account: T::AccountId,
            mode: NodeMode,
        ) -> DispatchResult {
            ensure_root(origin)?;

            // Verify account has stake
            ensure!(Stakes::<T>::contains_key(&account), Error::<T>::NotStaked);

            // Update mode
            NodeModes::<T>::insert(&account, mode.clone());

            Self::deposit_event(Event::NodeModeChanged {
                account,
                new_mode: mode,
            });

            Ok(())
        }
    }

    // Helper functions
    impl<T: Config> Pallet<T> {
        fn set_node_mode_internal(account: &T::AccountId, mode: NodeMode) -> Weight {
            if !Stakes::<T>::contains_key(account) {
                return T::DbWeight::get().reads(1);
            }
            NodeModes::<T>::insert(account, mode.clone());
            Self::deposit_event(Event::NodeModeChanged {
                account: account.clone(),
                new_mode: mode,
            });
            T::DbWeight::get().reads_writes(1, 1)
        }

        fn set_node_role_internal(account: &T::AccountId, role: NodeRole) -> Weight {
            if !Stakes::<T>::contains_key(account) {
                return T::DbWeight::get().reads(1);
            }
            Stakes::<T>::mutate(account, |stake| {
                stake.role = role;
            });
            T::DbWeight::get().reads_writes(1, 1)
        }

        /// Determine node role based on stake amount.
        ///
        /// Uses threshold comparison against configured minimum stakes for each role:
        /// - Director: ≥ MinStakeDirector (default 100 NSN)
        /// - SuperNode: ≥ MinStakeSuperNode (default 50 NSN)
        /// - Validator: ≥ MinStakeValidator (default 10 NSN)
        /// - Relay: ≥ MinStakeRelay (default 5 NSN)
        /// - None: < MinStakeRelay
        ///
        /// # Arguments
        /// * `amount` - The staked amount to classify
        ///
        /// # Returns
        /// The corresponding `NodeRole` for the stake amount
        fn determine_role(amount: BalanceOf<T>) -> NodeRole {
            if amount >= T::MinStakeDirector::get() {
                NodeRole::Director
            } else if amount >= T::MinStakeSuperNode::get() {
                NodeRole::SuperNode
            } else if amount >= T::MinStakeValidator::get() {
                NodeRole::Validator
            } else if amount >= T::MinStakeRelay::get() {
                NodeRole::Relay
            } else {
                NodeRole::None
            }
        }

        /// Determine role based on stake while preserving director state (Active/Reserve).
        fn adjust_role_for_amount(current_role: NodeRole, new_amount: BalanceOf<T>) -> NodeRole {
            let base_role = Self::determine_role(new_amount);
            match current_role {
                NodeRole::ActiveDirector | NodeRole::Reserve if base_role == NodeRole::Director => {
                    current_role
                }
                _ => base_role,
            }
        }

        /// Slash stake internally without requiring root origin.
        fn slash_internal(
            offender: &T::AccountId,
            amount: BalanceOf<T>,
            reason: SlashReason,
        ) -> DispatchResult {
            let mut stake_info = Self::stakes(offender);
            let slash_amount = amount.min(stake_info.amount);

            if slash_amount.is_zero() {
                return Ok(());
            }

            // Calculate new amount after slashing
            let new_amount = stake_info.amount.saturating_sub(slash_amount);

            // Enforce delegation cap after slashing
            let max_delegation = new_amount.saturating_mul(T::DelegationMultiplier::get().into());
            ensure!(
                stake_info.delegated_to_me <= max_delegation,
                Error::<T>::DelegationCapExceeded
            );

            // Burn slashed tokens by reducing freeze and burning balance
            T::Currency::set_freeze(
                &T::RuntimeFreezeReason::from(FreezeReason::Staking),
                offender,
                new_amount,
            )?;
            T::Currency::burn_from(
                offender,
                slash_amount,
                frame_support::traits::tokens::Preservation::Expendable,
                frame_support::traits::tokens::Precision::Exact,
                frame_support::traits::tokens::Fortitude::Force,
            )?;

            // Update stake info
            stake_info.amount = new_amount;
            stake_info.role = Self::adjust_role_for_amount(stake_info.role.clone(), new_amount);
            Stakes::<T>::insert(offender, stake_info.clone());

            // Update totals
            TotalStaked::<T>::mutate(|t| *t = t.saturating_sub(slash_amount));
            RegionStakes::<T>::mutate(stake_info.region, |r| *r = r.saturating_sub(slash_amount));

            Self::deposit_event(Event::StakeSlashed {
                offender: offender.clone(),
                amount: slash_amount,
                reason,
            });

            Ok(())
        }

        /// Slash for task abandonment without requiring root.
        pub fn slash_for_abandonment(
            offender: &T::AccountId,
            amount: BalanceOf<T>,
        ) -> DispatchResult {
            Self::slash_internal(offender, amount, SlashReason::TaskAbandonment)
        }

        /// Calculate total delegations across all validators for a given delegator.
        ///
        /// This iterates over all delegations from the delegator and sums them.
        /// Used to correctly calculate freeze amounts for multi-validator delegation.
        ///
        /// # Safety
        /// Iteration is bounded by `MaxDelegationsPerDelegator` to prevent
        /// unbounded loops and ensure L0 compliance.
        ///
        /// # Arguments
        /// * `delegator` - The account whose delegations to sum
        ///
        /// # Returns
        /// The total amount delegated across all validators
        pub fn total_delegations_for(delegator: &T::AccountId) -> BalanceOf<T> {
            let max_delegations = T::MaxDelegationsPerDelegator::get() as usize;
            Delegations::<T>::iter_prefix(delegator)
                .take(max_delegations) // L0: Bounded iteration
                .fold(Zero::zero(), |acc, (_, amount)| acc.saturating_add(amount))
        }
    }
}

impl<T: pallet::Config> NodeModeUpdater<T::AccountId> for pallet::Pallet<T> {
    fn set_mode(account: &T::AccountId, mode: NodeMode) -> Weight {
        pallet::Pallet::<T>::set_node_mode_internal(account, mode)
    }
}

impl<T: pallet::Config> NodeRoleUpdater<T::AccountId> for pallet::Pallet<T> {
    fn set_role(account: &T::AccountId, role: NodeRole) -> Weight {
        pallet::Pallet::<T>::set_node_role_internal(account, role)
    }
}
