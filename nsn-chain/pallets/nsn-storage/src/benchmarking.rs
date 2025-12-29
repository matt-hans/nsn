// Copyright 2024 Neural Sovereign Network
// This file is part of NSN Chain.

//! Benchmarking for NSN Storage pallet.

#![cfg(feature = "runtime-benchmarks")]

use super::*;
use crate::Pallet as Pinning;
use crate::types::MerkleProof;
use frame_benchmarking::v2::*;
use frame_support::traits::fungible::Mutate;
use frame_support::BoundedVec;
use frame_system::RawOrigin;
use sp_std::vec::Vec;

#[benchmarks]
mod benchmarks {
	use super::*;

	#[benchmark]
	fn create_deal(s: Linear<10, 20>) {
		// Setup
		let caller: T::AccountId = whitelisted_caller();
		<T as pallet::Config>::Currency::mint_into(&caller, 10000u32.into()).unwrap();

		let mut shard_vec = Vec::new();
		let mut merkle_vec = Vec::new();
		for i in 0..s {
			let mut shard = [0u8; 32];
			shard[0] = i as u8;
			shard_vec.push(shard);
			merkle_vec.push([2u8; 32]); // Dummy merkle root
		}
		let shards: BoundedVec<ShardHash, T::MaxShardsPerDeal> =
			BoundedVec::try_from(shard_vec).unwrap();
		let merkle_roots: BoundedVec<MerkleRoot, T::MaxShardsPerDeal> =
			BoundedVec::try_from(merkle_vec).unwrap();

		let duration_blocks = 1000u32.into();
		let payment = 100u32.into();

		#[extrinsic_call]
		create_deal(RawOrigin::Signed(caller), shards, merkle_roots, duration_blocks, payment);

		assert_eq!(crate::PinningDeals::<T>::iter().count(), 1);
	}

	#[benchmark]
	fn initiate_audit() {
		let pinner: T::AccountId = whitelisted_caller();
		let shard_hash = [1u8; 32];

		#[extrinsic_call]
		initiate_audit(RawOrigin::Root, pinner, shard_hash);

		assert_eq!(crate::PendingAudits::<T>::iter().count(), 1);
	}

	#[benchmark]
	fn submit_audit_proof() {
		// Setup audit
		let pinner: T::AccountId = whitelisted_caller();
		<T as pallet::Config>::Currency::mint_into(&pinner, 1000u32.into()).unwrap();

		let shard_hash = [1u8; 32];
		Pinning::<T>::initiate_audit(RawOrigin::Root.into(), pinner.clone(), shard_hash)
			.unwrap();

		let audits: Vec<_> = crate::PendingAudits::<T>::iter().collect();
		let (audit_id, _) = audits[0];

		let proof = MerkleProof {
			leaf_data: [0u8; 64],
			siblings: BoundedVec::default(),
			leaf_index: 0,
		};

		#[extrinsic_call]
		submit_audit_proof(RawOrigin::Signed(pinner), audit_id, proof);

		// Verify audit was processed
		let audit = crate::PendingAudits::<T>::get(audit_id);
		assert!(audit.is_some());
		assert!(audit.unwrap().status != crate::AuditStatus::Pending);
	}
}
