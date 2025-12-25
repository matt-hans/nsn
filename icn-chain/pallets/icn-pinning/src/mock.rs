// Copyright 2024 Interdimensional Cable Network
// This file is part of ICN Chain.

//! Mock runtime for ICN Pinning pallet tests.

use crate as pallet_icn_pinning;
use frame_support::{construct_runtime, parameter_types, traits::ConstU32, traits::Everything, PalletId};
use sp_core::H256;
use sp_runtime::{traits::IdentityLookup, BuildStorage, traits::BlakeTwo256};

type Block = frame_system::mocking::MockBlock<Test>;

// Configure a mock runtime to test the pallet.
construct_runtime!(
	pub enum Test
	{
		System: frame_system,
		Balances: pallet_balances,
		Stake: pallet_icn_stake::{Pallet, Call, Storage, Event<T>, FreezeReason},
		Reputation: pallet_icn_reputation::{Pallet, Call, Storage, Event<T>},
		Pinning: pallet_icn_pinning::{Pallet, Call, Storage, Event<T>, HoldReason},
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
	type AccountId = u64;
	type Lookup = IdentityLookup<Self::AccountId>;
	type Block = Block;
	type RuntimeEvent = RuntimeEvent;
	type BlockHashCount = BlockHashCount;
	type Version = ();
	type PalletInfo = PalletInfo;
	type AccountData = pallet_balances::AccountData<u128>;
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
	pub const ExistentialDeposit: u128 = 1;
}

impl pallet_balances::Config for Test {
	type MaxLocks = ();
	type MaxReserves = ();
	type ReserveIdentifier = [u8; 8];
	type Balance = u128;
	type RuntimeEvent = RuntimeEvent;
	type DustRemoval = ();
	type ExistentialDeposit = ExistentialDeposit;
	type AccountStore = System;
	type WeightInfo = ();
	type RuntimeHoldReason = RuntimeHoldReason;
	type FreezeIdentifier = RuntimeFreezeReason;
	type MaxFreezes = ConstU32<10>;
	type RuntimeFreezeReason = RuntimeFreezeReason;
	type DoneSlashHandler = ();
}

parameter_types! {
	pub const MinStakeDirector: u128 = 100_000_000_000_000_000_000; // 100 ICN
	pub const MinStakeSuperNode: u128 = 50_000_000_000_000_000_000; // 50 ICN
	pub const MinStakeValidator: u128 = 10_000_000_000_000_000_000; // 10 ICN
	pub const MinStakeRelay: u128 = 5_000_000_000_000_000_000; // 5 ICN
	pub const MaxStakePerNode: u128 = 1_000_000_000_000_000_000_000; // 1000 ICN
	pub const MaxRegionPercentage: u32 = 20;
	pub const RegionCapBootstrapStake: u128 = 1_000_000_000_000_000_000_000; // 1000 ICN
	pub const DelegationMultiplier: u32 = 5;
}

impl pallet_icn_stake::Config for Test {
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
	type MaxDelegationsPerDelegator = ConstU32<100>;
	type MaxDelegatorsPerValidator = ConstU32<1000>;
	type WeightInfo = ();
}

parameter_types! {
	pub const DefaultRetentionPeriod: u64 = 2_592_000; // ~6 months
	pub const CheckpointInterval: u64 = 1000;
	pub const DecayRatePerWeek: u64 = 5; // 5% per week
}

impl pallet_icn_reputation::Config for Test {
	type RuntimeEvent = RuntimeEvent;
	type MaxEventsPerBlock = ConstU32<1000>;
	type DefaultRetentionPeriod = DefaultRetentionPeriod;
	type CheckpointInterval = CheckpointInterval;
	type DecayRatePerWeek = DecayRatePerWeek;
	type MaxCheckpointAccounts = ConstU32<10000>;
	type MaxPrunePerBlock = ConstU32<100>;
	type WeightInfo = ();
}

parameter_types! {
	pub const AuditSlashAmount: u128 = 10_000_000_000_000_000_000; // 10 ICN
	pub const MaxSelectableCandidates: u32 = 1000; // Max candidates to consider
	pub const PinningPalletId: PalletId = PalletId(*b"icn/pinn");
}

impl pallet_icn_pinning::Config for Test {
	type RuntimeEvent = RuntimeEvent;
	type Currency = Balances;
	type RuntimeHoldReason = RuntimeHoldReason;
	type Randomness = TestRandomness;
	type AuditSlashAmount = AuditSlashAmount;
	type MaxShardsPerDeal = ConstU32<20>; // Support up to 20 shards
	type MaxPinnersPerShard = ConstU32<10>; // Support up to 10 pinners
	type MaxActiveDeals = ConstU32<100>;
	type MaxPendingAudits = ConstU32<100>;
	type MaxSelectableCandidates = MaxSelectableCandidates;
	type PalletId = PinningPalletId;
	type WeightInfo = ();
}

// Simplified randomness for testing
pub struct TestRandomness;
impl frame_support::traits::Randomness<H256, u64> for TestRandomness {
	fn random(subject: &[u8]) -> (H256, u64) {
		let hash = sp_io::hashing::blake2_256(subject);
		(H256::from(hash), 0)
	}
}

// Build genesis storage according to the mock runtime.
pub fn new_test_ext() -> sp_io::TestExternalities {
	use sp_runtime::traits::AccountIdConversion;

	let mut t = frame_system::GenesisConfig::<Test>::default()
		.build_storage()
		.unwrap();

	// Get the pallet account ID from the PalletId
	let pallet_account: u64 = PinningPalletId::get().into_account_truncating();

	pallet_balances::GenesisConfig::<Test> {
		balances: vec![
			(1, 1_000_000_000_000_000_000_000), // Alice: 1000 ICN
			(2, 1_000_000_000_000_000_000_000), // Bob: 1000 ICN
			(3, 1_000_000_000_000_000_000_000), // Charlie: 1000 ICN
			(4, 1_000_000_000_000_000_000_000), // Dave: 1000 ICN
			(5, 1_000_000_000_000_000_000_000), // Eve: 1000 ICN
			(6, 1_000_000_000_000_000_000_000), // Account 6: 1000 ICN
			(pallet_account, 10_000_000_000_000_000_000_000), // Pallet account: 10000 ICN initial balance
		],
		dev_accounts: None,
	}
	.assimilate_storage(&mut t)
	.unwrap();

	let mut ext = sp_io::TestExternalities::new(t);
	ext.execute_with(|| System::set_block_number(1));
	ext
}
