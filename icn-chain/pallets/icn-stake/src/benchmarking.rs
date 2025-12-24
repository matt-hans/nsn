// Copyright 2024 Interdimensional Cable Network
// This file is part of ICN Moonbeam.
//
// ICN Moonbeam is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

//! Benchmarking for pallet-icn-stake
//!
//! # Running Benchmarks
//!
//! ```bash
//! # Build the node with benchmarking feature
//! cargo build --release --features runtime-benchmarks
//!
//! # Run benchmarks for this pallet
//! ./target/release/icn-node benchmark pallet \
//!   --chain dev \
//!   --pallet pallet_icn_stake \
//!   --extrinsics '*' \
//!   --steps 50 \
//!   --repeat 20 \
//!   --output ./pallets/icn-stake/src/weights.rs
//! ```
//!
//! # Benchmark Components
//!
//! Each benchmark measures:
//! - Database reads/writes
//! - Computation complexity
//! - Event emissions
//! - Balance operations (freeze/thaw/burn)

#![cfg(feature = "runtime-benchmarks")]

use super::*;
use crate::Pallet as IcnStake;
use frame_benchmarking::v2::*;
use frame_system::{Pallet as System, RawOrigin};
use sp_std::prelude::*;

/// Helper function to fund an account with tokens
fn fund_account<T: Config>(account: &T::AccountId, amount: BalanceOf<T>) {
    T::Currency::mint_into(account, amount).unwrap();
}

#[benchmarks]
mod benchmarks {
    use super::*;

    /// Benchmark `deposit_stake` extrinsic
    ///
    /// # Weight Components
    /// - Balance read: 1 (reducible_balance)
    /// - Storage reads: 3 (stakes, total_staked, region_stakes)
    /// - Storage writes: 3 (stakes, total_staked, region_stakes)
    /// - Currency operations: 1 (set_freeze)
    /// - Events: 1
    ///
    /// # Worst Case
    /// - Account already has stake (update, not insert)
    /// - Region cap check enabled (past bootstrap phase)
    #[benchmark]
    fn deposit_stake() {
        let caller: T::AccountId = whitelisted_caller();
        let amount = T::MinStakeDirector::get(); // 100 ICN
        let lock_blocks = 1000u32.into();
        let region = Region::NaWest;

        // Fund the caller
        fund_account::<T>(&caller, amount);

        #[extrinsic_call]
        deposit_stake(RawOrigin::Signed(caller.clone()), amount, lock_blocks, region);

        // Verify state changed
        assert_eq!(IcnStake::stakes(&caller).amount, amount);
    }

    /// Benchmark `delegate` extrinsic
    ///
    /// # Weight Components
    /// - Storage reads: 2 (validator stake, delegations)
    /// - Iteration: up to MaxDelegationsPerDelegator (bounded)
    /// - Storage writes: 2 (delegations, validator stake)
    /// - Currency operations: 1 (set_freeze)
    /// - Events: 1
    ///
    /// # Worst Case
    /// - Delegator has existing delegations (iteration cost)
    /// - Freeze amount recalculation required
    #[benchmark]
    fn delegate() {
        let delegator: T::AccountId = whitelisted_caller();
        let validator: T::AccountId = account("validator", 0, 0);
        let stake_amount = T::MinStakeDirector::get(); // 100 ICN
        let delegate_amount = 50u128.into();

        // Setup: validator has stake
        fund_account::<T>(&validator, stake_amount);
        IcnStake::deposit_stake(
            RawOrigin::Signed(validator.clone()).into(),
            stake_amount,
            1000u32.into(),
            Region::EuWest,
        )
        .unwrap();

        // Fund delegator
        fund_account::<T>(&delegator, delegate_amount);

        #[extrinsic_call]
        delegate(
            RawOrigin::Signed(delegator.clone()),
            validator.clone(),
            delegate_amount,
        );

        // Verify delegation created
        assert_eq!(IcnStake::delegations(&delegator, &validator), delegate_amount);
    }

    /// Benchmark `delegate` with maximum existing delegations (worst case)
    ///
    /// # Weight Components
    /// - Same as `delegate`, but with maximum iteration cost
    ///
    /// # Worst Case
    /// - Delegator already has (MaxDelegationsPerDelegator - 1) delegations
    /// - Full iteration over all existing delegations
    #[benchmark]
    fn delegate_with_max_existing() {
        let delegator: T::AccountId = whitelisted_caller();
        let delegate_amount = 10u128.into();
        let max_delegations = T::MaxDelegationsPerDelegator::get();

        // Setup: delegator already has (max - 1) delegations
        let total_needed = delegate_amount * ((max_delegations - 1) as u128);
        fund_account::<T>(&delegator, total_needed + delegate_amount);

        for i in 0..(max_delegations - 1) {
            let validator: T::AccountId = account("validator", i, 0);
            let stake_amount = 100u128.into();

            fund_account::<T>(&validator, stake_amount);
            IcnStake::deposit_stake(
                RawOrigin::Signed(validator.clone()).into(),
                stake_amount,
                1000u32.into(),
                Region::EuWest,
            )
            .unwrap();

            IcnStake::delegate(
                RawOrigin::Signed(delegator.clone()).into(),
                validator.clone(),
                delegate_amount,
            )
            .unwrap();
        }

        // Now benchmark one more delegation (worst case)
        let new_validator: T::AccountId = account("validator", max_delegations, 0);
        fund_account::<T>(&new_validator, 100u128.into());
        IcnStake::deposit_stake(
            RawOrigin::Signed(new_validator.clone()).into(),
            100u128.into(),
            1000u32.into(),
            Region::EuWest,
        )
        .unwrap();

        #[extrinsic_call]
        delegate(
            RawOrigin::Signed(delegator.clone()),
            new_validator.clone(),
            delegate_amount,
        );
    }

    /// Benchmark `withdraw_stake` extrinsic
    ///
    /// # Weight Components
    /// - Storage reads: 2 (stakes, system block_number)
    /// - Storage writes: 3 (stakes, total_staked, region_stakes)
    /// - Currency operations: 1 (thaw or set_freeze)
    /// - Events: 1
    ///
    /// # Worst Case
    /// - Partial withdrawal (set_freeze, not thaw)
    /// - Role downgrade occurs
    #[benchmark]
    fn withdraw_stake() {
        let caller: T::AccountId = whitelisted_caller();
        let stake_amount = 150u128.into();
        let withdraw_amount = 50u128.into();
        let lock_blocks = 100u32.into();

        // Setup: caller has stake that's unlocked
        fund_account::<T>(&caller, stake_amount);
        IcnStake::deposit_stake(
            RawOrigin::Signed(caller.clone()).into(),
            stake_amount,
            lock_blocks,
            Region::NaWest,
        )
        .unwrap();

        // Advance past lock period
        let current_block = System::block_number();
        System::set_block_number(current_block + lock_blocks + 1u32.into());

        #[extrinsic_call]
        withdraw_stake(RawOrigin::Signed(caller.clone()), withdraw_amount);

        // Verify withdrawal
        assert_eq!(
            IcnStake::stakes(&caller).amount,
            stake_amount - withdraw_amount
        );
    }

    /// Benchmark `revoke_delegation` extrinsic
    ///
    /// # Weight Components
    /// - Storage reads: 2 (delegations, then iteration)
    /// - Iteration: up to MaxDelegationsPerDelegator (bounded)
    /// - Storage writes: 2 (delegations remove, validator stake)
    /// - Currency operations: 1 (thaw or set_freeze)
    /// - Events: 1
    ///
    /// # Worst Case
    /// - Delegator has remaining delegations (iteration + set_freeze)
    #[benchmark]
    fn revoke_delegation() {
        let delegator: T::AccountId = whitelisted_caller();
        let validator: T::AccountId = account("validator", 0, 0);
        let stake_amount = 100u128.into();
        let delegate_amount = 50u128.into();

        // Setup: validator has stake, delegation exists
        fund_account::<T>(&validator, stake_amount);
        fund_account::<T>(&delegator, delegate_amount);

        IcnStake::deposit_stake(
            RawOrigin::Signed(validator.clone()).into(),
            stake_amount,
            1000u32.into(),
            Region::EuWest,
        )
        .unwrap();

        IcnStake::delegate(
            RawOrigin::Signed(delegator.clone()).into(),
            validator.clone(),
            delegate_amount,
        )
        .unwrap();

        #[extrinsic_call]
        revoke_delegation(RawOrigin::Signed(delegator.clone()), validator.clone());

        // Verify revocation
        assert_eq!(IcnStake::delegations(&delegator, &validator), 0);
    }

    /// Benchmark `slash` extrinsic (root only)
    ///
    /// # Weight Components
    /// - Storage reads: 1 (stakes)
    /// - Storage writes: 3 (stakes, total_staked, region_stakes)
    /// - Currency operations: 2 (set_freeze, burn_from)
    /// - Events: 1
    ///
    /// # Worst Case
    /// - Partial slash (not complete stake wipeout)
    /// - Role downgrade occurs
    #[benchmark]
    fn slash() {
        let offender: T::AccountId = account("offender", 0, 0);
        let stake_amount = 100u128.into();
        let slash_amount = 20u128.into();

        // Setup: offender has stake
        fund_account::<T>(&offender, stake_amount);
        IcnStake::deposit_stake(
            RawOrigin::Signed(offender.clone()).into(),
            stake_amount,
            1000u32.into(),
            Region::NaWest,
        )
        .unwrap();

        #[extrinsic_call]
        slash(
            RawOrigin::Root,
            offender.clone(),
            slash_amount,
            SlashReason::BftFailure,
        );

        // Verify slash
        assert_eq!(
            IcnStake::stakes(&offender).amount,
            stake_amount - slash_amount
        );
    }

    impl_benchmark_test_suite! {}
}
