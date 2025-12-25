// Copyright 2024 Interdimensional Cable Network
// This file is part of ICN Chain.
//
// ICN Chain is free software: you can redistribute it and/or modify
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

#![cfg(feature = "runtime-benchmarks")]

use super::*;
use frame_benchmarking::v2::*;
use frame_support::traits::fungible::Mutate;
use frame_support::traits::Get;
use frame_system::RawOrigin;

#[benchmarks]
mod benchmarks {
	use super::*;

	#[benchmark]
	fn deposit_stake() {
		let caller: T::AccountId = whitelisted_caller();
		let amount = T::MinStakeDirector::get();
		let lock_blocks = 1000u32.into();
		let region = Region::NaWest;

		#[extrinsic_call]
		deposit_stake(RawOrigin::Signed(caller.clone()), amount, lock_blocks, region);

		assert_eq!(Stakes::<T>::get(&caller).amount, amount);
	}

	#[benchmark]
	fn delegate() {
		let delegator: T::AccountId = whitelisted_caller();
		let validator: T::AccountId = account("validator", 0, 0);
		let stake_amount = T::MinStakeDirector::get();

		// Mint to validator using seeded amount
		let validator_stake = T::MinStakeDirector::get();
		T::Currency::mint_into(&validator, validator_stake).unwrap();
		Pallet::<T>::deposit_stake(
			RawOrigin::Signed(validator.clone()).into(),
			stake_amount,
			1000u32.into(),
			Region::EuWest,
		).unwrap();

		// Mint to delegator - use MinStakeDirector value as delegate amount
		let delegate_amount = T::MinStakeValidator::get();
		T::Currency::mint_into(&delegator, delegate_amount).unwrap();

		#[extrinsic_call]
		delegate(
			RawOrigin::Signed(delegator.clone()),
			validator.clone(),
			delegate_amount,
		);

		assert!(Delegations::<T>::get(&delegator, &validator) > 0u32.into());
	}

	#[benchmark]
	fn withdraw_stake() {
		let caller: T::AccountId = whitelisted_caller();
		let stake_amount = 100u32.into();

		T::Currency::mint_into(&caller, stake_amount).unwrap();
		Pallet::<T>::deposit_stake(
			RawOrigin::Signed(caller.clone()).into(),
			stake_amount,
			100u32.into(),
			Region::NaWest,
		).unwrap();

		let current_block = frame_system::Pallet::<T>::block_number();
		frame_system::Pallet::<T>::set_block_number(current_block + 200u32.into());

		let withdraw_amount = 50u32.into();

		#[extrinsic_call]
		withdraw_stake(RawOrigin::Signed(caller.clone()), withdraw_amount);

		assert!(Stakes::<T>::get(&caller).amount < stake_amount);
	}

	#[benchmark]
	fn revoke_delegation() {
		let delegator: T::AccountId = whitelisted_caller();
		let validator: T::AccountId = account("validator", 0, 0);

		let stake_amount = 100u32.into();
		let delegate_amount = 50u32.into();

		T::Currency::mint_into(&validator, stake_amount).unwrap();
		T::Currency::mint_into(&delegator, delegate_amount).unwrap();

		Pallet::<T>::deposit_stake(
			RawOrigin::Signed(validator.clone()).into(),
			stake_amount,
			1000u32.into(),
			Region::EuWest,
		).unwrap();

		Pallet::<T>::delegate(
			RawOrigin::Signed(delegator.clone()).into(),
			validator.clone(),
			delegate_amount,
		).unwrap();

		#[extrinsic_call]
		revoke_delegation(RawOrigin::Signed(delegator.clone()), validator.clone());

		assert_eq!(Delegations::<T>::get(&delegator, &validator), 0u32.into());
	}

	#[benchmark]
	fn slash() {
		let offender: T::AccountId = account("offender", 0, 0);
		let stake_amount = 100u32.into();
		let slash_amount = 20u32.into();

		T::Currency::mint_into(&offender, stake_amount).unwrap();
		Pallet::<T>::deposit_stake(
			RawOrigin::Signed(offender.clone()).into(),
			stake_amount,
			1000u32.into(),
			Region::NaWest,
		).unwrap();

		#[extrinsic_call]
		slash(
			RawOrigin::Root,
			offender.clone(),
			slash_amount,
			SlashReason::BftFailure,
		);

		assert!(Stakes::<T>::get(&offender).amount < stake_amount);
	}
}
