// Copyright 2024 Neural Sovereign Network
// This file is part of NSN Chain.
//
// NSN Chain is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

//! Test utilities for pallet-nsn-reputation

use crate as pallet_nsn_reputation;
use frame_support::{construct_runtime, parameter_types, traits::ConstU32};
use sp_core::H256;
use sp_runtime::{traits::IdentityLookup, BuildStorage};

type Block = frame_system::mocking::MockBlockU32<Test>;

// Configure mock runtime
construct_runtime!(
	pub enum Test
	{
		System: frame_system,
		IcnReputation: pallet_nsn_reputation,
	}
);

parameter_types! {
	pub const BlockHashCount: u32 = 250;
}

impl frame_system::Config for Test {
	type BaseCallFilter = frame_support::traits::Everything;
	type BlockWeights = ();
	type BlockLength = ();
	type DbWeight = ();
	type RuntimeOrigin = RuntimeOrigin;
	type RuntimeCall = RuntimeCall;
	type RuntimeTask = RuntimeTask;
	type Nonce = u64;
	type Hash = H256;
	type Hashing = sp_runtime::traits::BlakeTwo256;
	type AccountId = u64;
	type Lookup = IdentityLookup<Self::AccountId>;
	type Block = Block;
	type RuntimeEvent = RuntimeEvent;
	type BlockHashCount = BlockHashCount;
	type Version = ();
	type PalletInfo = PalletInfo;
	type AccountData = ();
	type OnNewAccount = ();
	type OnKilledAccount = ();
	type SystemWeightInfo = ();
	type SS58Prefix = ();
	type OnSetCode = ();
	type MaxConsumers = ConstU32<16>;
	type SingleBlockMigrations = ();
	type MultiBlockMigrator = ();
	type PreInherents = ();
	type PostInherents = ();
	type PostTransactions = ();
	type ExtensionsWeightInfo = ();
}

parameter_types! {
	// L0: Bounded storage limits
	pub const MaxEventsPerBlock: u32 = 50;

	// Retention and checkpointing
	pub const DefaultRetentionPeriod: u32 = 2592000; // ~6 months at 6s blocks
	pub const CheckpointInterval: u32 = 1000;
	pub const DecayRatePerWeek: u64 = 5; // 5% per week
	pub const MaxCheckpointAccounts: u32 = 10_000; // L0: bounded checkpoint iteration
	pub const MaxPrunePerBlock: u32 = 10_000; // L0: bounded pruning per block
}

impl pallet_nsn_reputation::Config for Test {
	type RuntimeEvent = RuntimeEvent;
	type MaxEventsPerBlock = MaxEventsPerBlock;
	type DefaultRetentionPeriod = DefaultRetentionPeriod;
	type CheckpointInterval = CheckpointInterval;
	type DecayRatePerWeek = DecayRatePerWeek;
	type MaxCheckpointAccounts = MaxCheckpointAccounts;
	type MaxPrunePerBlock = MaxPrunePerBlock;
	type WeightInfo = ();
}

// Test accounts
pub const ALICE: u64 = 1;
pub const BOB: u64 = 2;
pub const CHARLIE: u64 = 3;
pub const DAVE: u64 = 4;
pub const EVE: u64 = 5;

// Build test externalities
pub struct ExtBuilder;

impl Default for ExtBuilder {
	fn default() -> Self {
		Self
	}
}

impl ExtBuilder {
	pub fn build(self) -> sp_io::TestExternalities {
		let storage = frame_system::GenesisConfig::<Test>::default()
			.build_storage()
			.unwrap();

		let mut ext = sp_io::TestExternalities::new(storage);
		ext.execute_with(|| {
			System::set_block_number(1);
		});
		ext
	}
}

/// Convenience function to create test externalities
pub fn new_test_ext() -> sp_io::TestExternalities {
	ExtBuilder::default().build()
}

// Helper to get all events
pub fn events() -> Vec<RuntimeEvent> {
	System::events()
		.into_iter()
		.map(|e| e.event)
		.collect()
}
