// Copyright 2024 Neural Sovereign Network
// This file is part of NSN Chain.

//! Mock runtime for pallet-nsn-treasury testing

use crate as pallet_nsn_treasury;
use frame_support::{
	derive_impl, parameter_types,
	traits::ConstU32,
};
use sp_runtime::BuildStorage;

type Block = frame_system::mocking::MockBlock<Test>;

// Configure a mock runtime to test the pallet
frame_support::construct_runtime!(
	pub enum Test {
		System: frame_system,
		Balances: pallet_balances,
		Treasury: pallet_nsn_treasury,
	}
);

#[derive_impl(frame_system::config_preludes::TestDefaultConfig)]
impl frame_system::Config for Test {
	type Block = Block;
	type AccountData = pallet_balances::AccountData<u128>;
}

parameter_types! {
	pub const ExistentialDeposit: u128 = 1;
}

impl pallet_balances::Config for Test {
	type MaxLocks = ConstU32<50>;
	type MaxReserves = ();
	type ReserveIdentifier = [u8; 8];
	type Balance = u128;
	type RuntimeEvent = RuntimeEvent;
	type DustRemoval = ();
	type ExistentialDeposit = ExistentialDeposit;
	type AccountStore = System;
	type WeightInfo = ();
	type FreezeIdentifier = ();
	type MaxFreezes = ();
	type RuntimeHoldReason = ();
	type RuntimeFreezeReason = ();
	type DoneSlashHandler = ();
}

parameter_types! {
	pub const TreasuryPalletId: frame_support::PalletId = frame_support::PalletId(*b"nsn/trea");
	pub const DistributionFrequency: u64 = 14400; // ~1 day at 6s blocks
}

impl pallet_nsn_treasury::Config for Test {
	type Currency = Balances;
	type PalletId = TreasuryPalletId;
	type DistributionFrequency = DistributionFrequency;
	type WeightInfo = ();
}

// Build genesis storage according to the mock runtime
pub fn new_test_ext() -> sp_io::TestExternalities {
	let mut t = frame_system::GenesisConfig::<Test>::default().build_storage().unwrap();

	pallet_balances::GenesisConfig::<Test> {
		balances: vec![
			(1, 1_000_000_000_000_000_000_000_000_000u128), // 1B NSN for account 1
			(2, 1_000_000_000_000_000_000_000_000_000u128), // 1B NSN for account 2
			(3, 500_000_000_000_000_000_000_000_000u128),   // 500M NSN for account 3
		],
		dev_accounts: Default::default(),
	}
	.assimilate_storage(&mut t)
	.unwrap();

	let mut ext = sp_io::TestExternalities::new(t);
	ext.execute_with(|| System::set_block_number(1));
	ext
}
