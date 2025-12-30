// Copyright 2024 Neural Sovereign Network
// This file is part of NSN Chain.
//
// NSN Chain is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

//! Test utilities for pallet-nsn-model-registry

use crate as pallet_nsn_model_registry;
use frame_support::{construct_runtime, parameter_types, traits::ConstU32, traits::Everything};
use frame_system::pallet_prelude::BlockNumberFor;
use sp_core::H256;
use sp_runtime::{
    traits::{BlakeTwo256, IdentityLookup},
    BuildStorage,
};

pub type AccountId = u64;
pub type BlockNumber = BlockNumberFor<Test>;

type Block = frame_system::mocking::MockBlockU32<Test>;

// Configure mock runtime
construct_runtime!(
    pub enum Test
    {
        System: frame_system,
        NsnModelRegistry: pallet_nsn_model_registry::{Pallet, Call, Storage, Event<T>},
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
    /// Maximum length of model identifier
    pub const MaxModelIdLen: u32 = 64;
    /// Maximum length of container CID
    pub const MaxCidLen: u32 = 128;
    /// Maximum number of hot models a node can advertise
    pub const MaxHotModels: u32 = 10;
    /// Maximum number of warm models a node can advertise
    pub const MaxWarmModels: u32 = 20;
}

impl pallet_nsn_model_registry::Config for Test {
    type MaxModelIdLen = MaxModelIdLen;
    type MaxCidLen = MaxCidLen;
    type MaxHotModels = MaxHotModels;
    type MaxWarmModels = MaxWarmModels;
    type WeightInfo = ();
}

// Test accounts
pub const ALICE: AccountId = 1;
pub const BOB: AccountId = 2;
pub const CHARLIE: AccountId = 3;
#[allow(dead_code)]
pub const DAVE: AccountId = 4;
#[allow(dead_code)]
pub const EVE: AccountId = 5;

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

// Helper function to advance blocks
#[allow(dead_code)]
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
