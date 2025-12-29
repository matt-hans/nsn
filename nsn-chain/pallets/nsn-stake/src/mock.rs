// Copyright 2024 Neural Sovereign Network
// This file is part of NSN Chain.
//
// NSN Chain is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

//! Test utilities for pallet-nsn-stake

use crate as pallet_nsn_stake;
use frame_support::{
	construct_runtime, parameter_types,
	traits::{ConstU128, ConstU32, Everything, VariantCountOf},
};
use frame_system::pallet_prelude::BlockNumberFor;
use sp_core::H256;
use sp_runtime::{
	traits::{BlakeTwo256, IdentityLookup},
	BuildStorage,
};

pub type AccountId = u64;
pub type Balance = u128;
pub type BlockNumber = BlockNumberFor<Test>;

type Block = frame_system::mocking::MockBlockU32<Test>;

// Configure mock runtime
construct_runtime!(
	pub enum Test
	{
		System: frame_system,
		Balances: pallet_balances,
		NsnStake: pallet_nsn_stake::{Pallet, Call, Storage, Event<T>, FreezeReason},
	}
);

parameter_types! {
	pub const BlockHashCount: u32 = 250;
}

impl frame_system::Config for Test {
	type BaseCallFilter = Everything;
	type BlockWeights = ();
	type BlockLength = ();
	type DbWeight = ();
	type RuntimeOrigin = RuntimeOrigin;
	type RuntimeCall = RuntimeCall;
	type RuntimeTask = RuntimeTask;
	type Nonce = u64;
	type Hash = H256;
	type Hashing = BlakeTwo256;
	type AccountId = AccountId;
	type Lookup = IdentityLookup<Self::AccountId>;
	type Block = Block;
	type RuntimeEvent = RuntimeEvent;
	type BlockHashCount = BlockHashCount;
	type Version = ();
	type PalletInfo = PalletInfo;
	type AccountData = pallet_balances::AccountData<Balance>;
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

impl pallet_balances::Config for Test {
	type MaxLocks = ();
	type MaxReserves = ();
	type ReserveIdentifier = [u8; 8];
	type Balance = Balance;
	type RuntimeEvent = RuntimeEvent;
	type DustRemoval = ();
	type ExistentialDeposit = ConstU128<1>;
	type AccountStore = System;
	type WeightInfo = ();
	type RuntimeHoldReason = ();
	type FreezeIdentifier = RuntimeFreezeReason;
	type MaxFreezes = VariantCountOf<RuntimeFreezeReason>;
	type RuntimeFreezeReason = RuntimeFreezeReason;
	type DoneSlashHandler = ();
}

parameter_types! {
	// Role thresholds (in smallest unit)
	pub const MinStakeDirector: Balance = 100;
	pub const MinStakeSuperNode: Balance = 50;
	pub const MinStakeValidator: Balance = 10;
	pub const MinStakeRelay: Balance = 5;

	// Caps
	pub const MaxStakePerNode: Balance = 1000;
	pub const MaxRegionPercentage: u32 = 20; // 20%
	pub const DelegationMultiplier: u32 = 5; // 5× validator stake
	pub const RegionCapBootstrapStake: Balance = 1000; // Enforce caps after 1000 NSN total

	// Bounded limits (L0 constraint compliance)
	pub const MaxDelegationsPerDelegator: u32 = 100;
	pub const MaxDelegatorsPerValidator: u32 = 1000;
}

impl pallet_nsn_stake::Config for Test {
	type Currency = Balances;
	type RuntimeFreezeReason = RuntimeFreezeReason;
	type MinStakeDirector = MinStakeDirector;
	type MinStakeSuperNode = MinStakeSuperNode;
	type MinStakeValidator = MinStakeValidator;
	type MinStakeRelay = MinStakeRelay;
	type MaxStakePerNode = MaxStakePerNode;
	type MaxRegionPercentage = MaxRegionPercentage;
	type RegionCapBootstrapStake = RegionCapBootstrapStake;
	type DelegationMultiplier = DelegationMultiplier;
	type MaxDelegationsPerDelegator = MaxDelegationsPerDelegator;
	type MaxDelegatorsPerValidator = MaxDelegatorsPerValidator;
	type WeightInfo = ();
}

// Test accounts
pub const ALICE: AccountId = 1;
pub const BOB: AccountId = 2;
pub const CHARLIE: AccountId = 3;
pub const DAVE: AccountId = 4;
pub const EVE: AccountId = 5;
pub const FRANK: AccountId = 6;
pub const GEORGE: AccountId = 7;
pub const HELEN: AccountId = 8;

// Build test externalities
pub struct ExtBuilder {
	balances: Vec<(AccountId, Balance)>,
}

impl Default for ExtBuilder {
	fn default() -> Self {
		Self {
			balances: vec![
				(ALICE, 1000),
				(BOB, 1000),
				(CHARLIE, 1000),
				(DAVE, 1000),
				(EVE, 1000),
				(FRANK, 1000),
				(GEORGE, 1000),
				(HELEN, 1000),
			],
		}
	}
}

impl ExtBuilder {
	pub fn with_balances(mut self, balances: Vec<(AccountId, Balance)>) -> Self {
		self.balances = balances;
		self
	}

	pub fn build(self) -> sp_io::TestExternalities {
		let mut storage = frame_system::GenesisConfig::<Test>::default()
			.build_storage()
			.unwrap();

		pallet_balances::GenesisConfig::<Test> {
			balances: self.balances,
			dev_accounts: None,
		}
		.assimilate_storage(&mut storage)
		.unwrap();

		let mut ext = sp_io::TestExternalities::new(storage);
		ext.execute_with(|| {
			System::set_block_number(1);
		});
		ext
	}
}

// Helper function to advance blocks
pub fn roll_to(n: BlockNumber) {
	while System::block_number() < n {
		System::set_block_number(System::block_number() + 1);
	}
}

// Helper to get last event
pub fn last_event() -> RuntimeEvent {
	System::events().pop().expect("Event expected").event
}
