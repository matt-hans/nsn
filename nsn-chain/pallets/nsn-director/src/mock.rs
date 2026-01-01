// Copyright 2024 Neural Sovereign Network
// This file is part of NSN Chain.
//
// NSN Chain is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

//! Test utilities for pallet-nsn-director

use crate as pallet_nsn_director;
use frame_support::{
    construct_runtime, derive_impl, parameter_types,
    traits::{ConstU32, Hooks},
};
use pallet_nsn_stake::Region;
use sp_core::H256;
use sp_runtime::{
    traits::{BlakeTwo256, IdentityLookup},
    BuildStorage,
};

type Block = frame_system::mocking::MockBlockU32<Test>;

// Configure mock runtime
construct_runtime!(
    pub enum Test
    {
        System: frame_system,
        Balances: pallet_balances,
        NsnStake: pallet_nsn_stake,
        NsnReputation: pallet_nsn_reputation,
        NsnDirector: pallet_nsn_director,
    }
);

parameter_types! {
    pub const BlockHashCount: u32 = 250;
}

#[derive_impl(frame_system::config_preludes::TestDefaultConfig)]
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
}

parameter_types! {
    pub const ExistentialDeposit: u128 = 1;
}

impl pallet_balances::Config for Test {
    type MaxLocks = ConstU32<50>;
    type MaxReserves = ConstU32<50>;
    type ReserveIdentifier = [u8; 8];
    type Balance = u128;
    type RuntimeEvent = RuntimeEvent;
    type DustRemoval = ();
    type ExistentialDeposit = ExistentialDeposit;
    type AccountStore = System;
    type WeightInfo = ();
    type FreezeIdentifier = RuntimeFreezeReason;
    type MaxFreezes = ConstU32<50>;
    type RuntimeHoldReason = RuntimeHoldReason;
    type RuntimeFreezeReason = RuntimeFreezeReason;
    type DoneSlashHandler = ();
}

parameter_types! {
    // Stake pallet constants
    pub const MinStakeDirector: u128 = 100_000_000_000_000_000_000; // 100 NSN
    pub const MinStakeSuperNode: u128 = 50_000_000_000_000_000_000; // 50 NSN
    pub const MinStakeValidator: u128 = 10_000_000_000_000_000_000; // 10 NSN
    pub const MinStakeRelay: u128 = 5_000_000_000_000_000_000; // 5 NSN
    pub const MaxStakePerNode: u128 = 1_000_000_000_000_000_000_000; // 1000 NSN
    pub const MaxRegionPercentage: u32 = 20;
    pub const DelegationMultiplier: u32 = 5;
    pub const MaxDelegationsPerDelegator: u32 = 10;
    pub const MaxDelegatorsPerValidator: u32 = 100;
    pub const RegionCapBootstrapStake: u128 = 1_000_000_000_000_000_000_000; // 1000 NSN
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
    type DelegationMultiplier = DelegationMultiplier;
    type MaxDelegationsPerDelegator = MaxDelegationsPerDelegator;
    type MaxDelegatorsPerValidator = MaxDelegatorsPerValidator;
    type RegionCapBootstrapStake = RegionCapBootstrapStake;
    type WeightInfo = ();
}

parameter_types! {
    // Reputation pallet constants
    pub const MaxEventsPerBlock: u32 = 50;
    pub const DefaultRetentionPeriod: u32 = 2592000; // ~6 months
    pub const CheckpointInterval: u32 = 1000;
    pub const DecayRatePerWeek: u64 = 5;
    pub const MaxCheckpointAccounts: u32 = 10_000;
    pub const MaxPrunePerBlock: u32 = 10_000;
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

parameter_types! {
    // Director pallet constants
    pub const ChallengeBond: u128 = 25_000_000_000_000_000_000; // 25 NSN
    pub const DirectorSlashAmount: u128 = 100_000_000_000_000_000_000; // 100 NSN
    pub const ChallengerReward: u128 = 10_000_000_000_000_000_000; // 10 NSN
    pub const MaxDirectorsPerSlot: u32 = 5;
    pub const MaxPendingSlots: u32 = 100;
    // Epoch constants (for testing, use shorter durations)
    pub const EpochDuration: u32 = 600; // 1 hour at 6s/block
    pub const EpochLookahead: u32 = 20; // 2 minutes (20 blocks)
    pub const MaxDirectorsPerEpoch: u32 = 5;
}

/// Simple randomness implementation for testing
pub struct TestRandomness;

impl frame_support::traits::Randomness<H256, u32> for TestRandomness {
    fn random(subject: &[u8]) -> (H256, u32) {
        // Use a simple hash for deterministic testing
        let hash = BlakeTwo256::hash(subject);
        (hash, 0)
    }
}

impl pallet_nsn_director::Config for Test {
    type Currency = Balances;
    type RuntimeHoldReason = RuntimeHoldReason;
    type Randomness = TestRandomness;
    type ChallengeBond = ChallengeBond;
    type DirectorSlashAmount = DirectorSlashAmount;
    type ChallengerReward = ChallengerReward;
    type MaxDirectorsPerSlot = MaxDirectorsPerSlot;
    type MaxPendingSlots = MaxPendingSlots;
    type EpochDuration = EpochDuration;
    type EpochLookahead = EpochLookahead;
    type MaxDirectorsPerEpoch = MaxDirectorsPerEpoch;
    type WeightInfo = ();
}

// Test accounts
pub const ALICE: u64 = 1;
pub const BOB: u64 = 2;
pub const CHARLIE: u64 = 3;
pub const DAVE: u64 = 4;
pub const EVE: u64 = 5;
pub const FRANK: u64 = 6;
pub const GRACE: u64 = 7;
pub const HENRY: u64 = 8;
pub const IVAN: u64 = 9;
pub const JULIA: u64 = 10;

// NSN token amounts (18 decimals)
pub const NSN: u128 = 1_000_000_000_000_000_000;

// Build test externalities
pub struct ExtBuilder {
    balances: Vec<(u64, u128)>,
}

impl Default for ExtBuilder {
    fn default() -> Self {
        Self {
            balances: vec![
                (ALICE, 1000 * NSN),
                (BOB, 1000 * NSN),
                (CHARLIE, 1000 * NSN),
                (DAVE, 1000 * NSN),
                (EVE, 1000 * NSN),
                (FRANK, 1000 * NSN),
                (GRACE, 1000 * NSN),
                (HENRY, 1000 * NSN),
                (IVAN, 1000 * NSN),
                (JULIA, 1000 * NSN),
            ],
        }
    }
}

impl ExtBuilder {
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

/// Convenience function to create test externalities
pub fn new_test_ext() -> sp_io::TestExternalities {
    ExtBuilder::default().build()
}

// Helper function to advance blocks
pub fn roll_to(n: u32) {
    while System::block_number() < n {
        let current = System::block_number();
        <NsnDirector as Hooks<u32>>::on_finalize(current);
        System::set_block_number(current + 1);
        <NsnDirector as Hooks<u32>>::on_initialize(current + 1);
    }
}

// Helper to stake as Director
pub fn stake_as_director(who: u64, amount: u128, region: Region) {
    assert!(Balances::free_balance(who) >= amount);
    pallet_nsn_stake::Pallet::<Test>::deposit_stake(
        RuntimeOrigin::signed(who),
        amount,
        100, // lock blocks
        region,
    )
    .expect("Staking should succeed");
}

// Helper to record reputation event
pub fn record_reputation(
    who: u64,
    event_type: pallet_nsn_reputation::ReputationEventType,
    slot: u64,
) {
    pallet_nsn_reputation::Pallet::<Test>::record_event(
        RuntimeOrigin::root(),
        who,
        event_type,
        slot,
    )
    .expect("Recording reputation should succeed");
}

use sp_runtime::traits::Hash;

/// Helper to create a test hash
pub fn test_hash(data: &[u8]) -> H256 {
    BlakeTwo256::hash(data)
}
