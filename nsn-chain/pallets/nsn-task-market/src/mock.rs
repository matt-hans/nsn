// Copyright 2024 Neural Sovereign Network
// This file is part of NSN Chain.
//
// NSN Chain is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

//! Test utilities for pallet-nsn-task-market

use crate as pallet_nsn_task_market;
use frame_support::{
    construct_runtime, parameter_types, traits::ConstU128, traits::ConstU32, traits::Everything,
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
        NsnTaskMarket: pallet_nsn_task_market::{Pallet, Call, Storage, Event<T>},
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
    type MaxReserves = ConstU32<50>;
    type ReserveIdentifier = [u8; 8];
    type Balance = Balance;
    type RuntimeEvent = RuntimeEvent;
    type DustRemoval = ();
    type ExistentialDeposit = ConstU128<1>;
    type AccountStore = System;
    type WeightInfo = ();
    type RuntimeHoldReason = ();
    type FreezeIdentifier = ();
    type MaxFreezes = ConstU32<0>;
    type RuntimeFreezeReason = ();
    type DoneSlashHandler = ();
}

parameter_types! {
    /// Maximum number of pending (open) tasks
    pub const MaxPendingTasks: u32 = 100;
    /// Maximum number of assigned Lane 1 tasks
    pub const MaxAssignedLane1Tasks: u32 = 100;
    /// Maximum number of assignment candidates to consider
    pub const MaxAssignmentCandidates: u32 = 10;
    /// Maximum expired tasks to process per block
    pub const MaxExpiredPerBlock: u32 = 25;
    /// Maximum Lane 1 preemptions per block
    pub const MaxPreemptionsPerBlock: u32 = 10;
    /// Maximum length of model identifier
    pub const MaxModelIdLen: u32 = 64;
    /// Maximum length of content identifier (CID)
    pub const MaxCidLen: u32 = 64;
    /// Minimum escrow amount required for task creation
    pub const MinEscrow: Balance = 10;
    /// Slash amount for task abandonment
    pub const TaskAbandonmentSlash: Balance = 5;
}

pub struct MockLaneNodeProvider;
impl pallet_nsn_task_market::LaneNodeProvider<AccountId, Balance> for MockLaneNodeProvider {
    fn eligible_nodes(lane: pallet_nsn_task_market::TaskLane, max: u32) -> Vec<(AccountId, Balance)> {
        let mut candidates = match lane {
            pallet_nsn_task_market::TaskLane::Lane0 => vec![(ALICE, 100)],
            pallet_nsn_task_market::TaskLane::Lane1 => vec![(ALICE, 100), (BOB, 90), (CHARLIE, 80)],
        };
        let limit = max as usize;
        if candidates.len() > limit {
            candidates.truncate(limit);
        }
        candidates
    }

    fn is_eligible(account: &AccountId, lane: pallet_nsn_task_market::TaskLane) -> bool {
        match lane {
            pallet_nsn_task_market::TaskLane::Lane0 => *account == ALICE,
            pallet_nsn_task_market::TaskLane::Lane1 => matches!(account, &ALICE | &BOB | &CHARLIE),
        }
    }
}

pub struct MockReputationUpdater;
impl pallet_nsn_task_market::ReputationUpdater<AccountId> for MockReputationUpdater {
    fn record_task_result(_account: &AccountId, _success: bool) {}
}

pub struct MockTaskSlashHandler;
impl pallet_nsn_task_market::TaskSlashHandler<AccountId, Balance> for MockTaskSlashHandler {
    fn slash_for_abandonment(_account: &AccountId, _amount: Balance) -> frame_support::dispatch::DispatchResult {
        Ok(())
    }
}

impl pallet_nsn_task_market::Config for Test {
    type Currency = Balances;
    type MaxPendingTasks = MaxPendingTasks;
    type MaxAssignedLane1Tasks = MaxAssignedLane1Tasks;
    type MaxAssignmentCandidates = MaxAssignmentCandidates;
    type MaxExpiredPerBlock = MaxExpiredPerBlock;
    type MaxPreemptionsPerBlock = MaxPreemptionsPerBlock;
    type MaxModelIdLen = MaxModelIdLen;
    type MaxCidLen = MaxCidLen;
    type MinEscrow = MinEscrow;
    type LaneNodeProvider = MockLaneNodeProvider;
    type ReputationUpdater = MockReputationUpdater;
    type TaskSlashHandler = MockTaskSlashHandler;
    type TaskAbandonmentSlash = TaskAbandonmentSlash;
    type TreasuryAccount = TreasuryAccount;
    type WeightInfo = ();
}

// Test accounts
pub const ALICE: AccountId = 1;
pub const BOB: AccountId = 2;
pub const CHARLIE: AccountId = 3;
pub const DAVE: AccountId = 4;
pub const EVE: AccountId = 5;
pub const TREASURY_ACCOUNT: AccountId = EVE;

pub struct TreasuryAccount;
impl frame_support::traits::Get<AccountId> for TreasuryAccount {
    fn get() -> AccountId {
        TREASURY_ACCOUNT
    }
}

// Build test externalities
pub struct ExtBuilder {
    balances: Vec<(AccountId, Balance)>,
}

impl Default for ExtBuilder {
    fn default() -> Self {
        Self {
            balances: vec![
                (ALICE, 10_000),
                (BOB, 10_000),
                (CHARLIE, 10_000),
                (DAVE, 10_000),
                (EVE, 10_000),
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

// Helper to get all events
#[allow(dead_code)]
pub fn events() -> Vec<RuntimeEvent> {
    System::events().into_iter().map(|r| r.event).collect()
}
