// Copyright 2024 Interdimensional Cable Network
// This file is part of ICN Chain.

//! Benchmarking for ICN Pinning pallet.

#![cfg(feature = "runtime-benchmarks")]

use super::*;
use crate::Pallet as Pinning;
use frame_benchmarking::v2::*;
use frame_support::{traits::fungible::Mutate, BoundedVec};
use frame_system::RawOrigin;
use pallet_icn_stake::{NodeRole, Region};
use sp_runtime::traits::StaticLookup;

fn create_super_nodes<T: Config>(count: u32) -> Vec<T::AccountId>
where
	T::AccountId: From<u64>,
{
	let mut accounts = Vec::new();
	let regions = vec![
		Region::NaWest,
		Region::NaEast,
		Region::EuWest,
		Region::Apac,
		Region::Latam,
		Region::Mena,
		Region::Mena, // Repeat last region if needed
	];

	for i in 0..count {
		let account: T::AccountId = (i as u64 + 1000).into();
		let region = regions[(i as usize) % regions.len()];

		// Fund account
		let balance = 1000u32.into();
		T::Currency::set_balance(&account, balance);

		// Stake as super-node
		let _ = pallet_icn_stake::Pallet::<T>::deposit_stake(
			RawOrigin::Signed(account.clone()).into(),
			50u32.into(),
			100u32.into(),
			region,
		);

		accounts.push(account);
	}

	accounts
}

#[benchmarks]
mod benchmarks {
	use super::*;

	#[benchmark]
	fn create_deal(s: Linear<10, 20>) {
		// Setup
		let caller: T::AccountId = whitelisted_caller();
		T::Currency::set_balance(&caller, 10000u32.into());
		create_super_nodes::<T>(10); // Create enough super-nodes

		let mut shard_vec = Vec::new();
		for i in 0..s {
			let mut shard = [0u8; 32];
			shard[0] = i as u8;
			shard_vec.push(shard);
		}
		let shards: BoundedVec<ShardHash, T::MaxShardsPerDeal> =
			BoundedVec::try_from(shard_vec).unwrap();

		#[extrinsic_call]
		create_deal(RawOrigin::Signed(caller), shards, 1000u32.into(), 100u32.into());
	}

	#[benchmark]
	fn initiate_audit() {
		let pinner: T::AccountId = whitelisted_caller();
		let shard_hash = [1u8; 32];

		#[extrinsic_call]
		initiate_audit(RawOrigin::Root, pinner, shard_hash);
	}

	#[benchmark]
	fn submit_audit_proof() {
		// Setup audit
		let pinner: T::AccountId = whitelisted_caller();
		T::Currency::set_balance(&pinner, 1000u32.into());

		let _ = pallet_icn_stake::Pallet::<T>::deposit_stake(
			RawOrigin::Signed(pinner.clone()).into(),
			50u32.into(),
			100u32.into(),
			Region::NaWest,
		);

		let shard_hash = [1u8; 32];
		Pinning::<T>::initiate_audit(RawOrigin::Root.into(), pinner.clone(), shard_hash)
			.unwrap();

		let audits: Vec<_> = crate::PendingAudits::<T>::iter().collect();
		let (audit_id, _) = audits[0];

		let proof: BoundedVec<u8, frame_support::traits::ConstU32<1024>> =
			BoundedVec::try_from(vec![0u8; 64]).unwrap();

		#[extrinsic_call]
		submit_audit_proof(RawOrigin::Signed(pinner), audit_id, proof);
	}

	impl_benchmark_test_suite!(Pinning, crate::mock::new_test_ext(), crate::mock::Test);
}
