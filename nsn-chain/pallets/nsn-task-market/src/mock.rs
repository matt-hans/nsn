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
    /// Maximum length of model identifier
    pub const MaxModelIdLen: u32 = 64;
    /// Maximum length of content identifier (CID)
    pub const MaxCidLen: u32 = 64;
    /// Minimum escrow amount required for task creation
    pub const MinEscrow: Balance = 10;
}

impl pallet_nsn_task_market::Config for Test {
    type Currency = Balances;
    type MaxPendingTasks = MaxPendingTasks;
    type MaxModelIdLen = MaxModelIdLen;
    type MaxCidLen = MaxCidLen;
    type MinEscrow = MinEscrow;
    type WeightInfo = ();
}

// Test accounts
pub const ALICE: AccountId = 1;
pub const BOB: AccountId = 2;
pub const CHARLIE: AccountId = 3;
pub const DAVE: AccountId = 4;
pub const EVE: AccountId = 5;

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
