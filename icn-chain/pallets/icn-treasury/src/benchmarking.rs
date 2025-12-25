// Copyright 2024 Interdimensional Cable Network
// This file is part of ICN Chain.

//! Benchmarking setup for pallet-icn-treasury

#![cfg(feature = "runtime-benchmarks")]

use super::*;
use frame_benchmarking::v2::*;
use frame_system::RawOrigin;

#[benchmarks]
mod benchmarks {
	use super::*;

	#[benchmark]
	fn fund_treasury() {
		let caller: T::AccountId = whitelisted_caller();
		let amount = 1000u32.into();

		#[extrinsic_call]
		_(RawOrigin::Signed(caller.clone()), amount);

		assert!(TreasuryBalance::<T>::get() >= amount);
	}

	#[benchmark]
	fn approve_proposal() {
		let beneficiary: T::AccountId = whitelisted_caller();
		let amount = 1000u32.into();
		TreasuryBalance::<T>::put(amount * 2u32.into());

		#[extrinsic_call]
		_(RawOrigin::Root, beneficiary.clone(), amount, 1u32);

		assert!(TreasuryBalance::<T>::get() < amount * 2u32.into());
	}

	#[benchmark]
	fn record_director_work() {
		let account: T::AccountId = whitelisted_caller();

		#[extrinsic_call]
		_(RawOrigin::Root, account.clone(), 10u64);

		assert_eq!(AccumulatedContributionsMap::<T>::get(&account).director_slots, 10);
	}

	#[benchmark]
	fn record_validator_work() {
		let account: T::AccountId = whitelisted_caller();

		#[extrinsic_call]
		_(RawOrigin::Root, account.clone(), 20u64);

		assert_eq!(AccumulatedContributionsMap::<T>::get(&account).validator_votes, 20);
	}

	impl_benchmark_test_suite!(Pallet, crate::mock::new_test_ext(), crate::mock::Test);
}
