// Copyright 2024 Interdimensional Cable Network
// This file is part of ICN Chain.

//! Mock runtime for pallet-nsn-bft tests

use crate as pallet_nsn_bft;
use frame_support::{derive_impl, parameter_types, traits::Hooks};
use sp_runtime::BuildStorage;

type Block = frame_system::mocking::MockBlock<Test>;

// Configure a mock runtime to test the pallet.
frame_support::construct_runtime!(
    pub enum Test
    {
        System: frame_system,
        IcnBft: pallet_nsn_bft,
    }
);

#[derive_impl(frame_system::config_preludes::TestDefaultConfig)]
impl frame_system::Config for Test {
    type Block = Block;
}

parameter_types! {
    /// Default retention period for tests: 2,592,000 blocks (~6 months)
    pub const DefaultRetentionPeriod: u64 = 2_592_000;
}

impl pallet_nsn_bft::Config for Test {
    type DefaultRetentionPeriod = DefaultRetentionPeriod;
    type WeightInfo = ();
}

/// Build test externalities
pub fn new_test_ext() -> sp_io::TestExternalities {
    let t = frame_system::GenesisConfig::<Test>::default()
        .build_storage()
        .unwrap();
    let mut ext = sp_io::TestExternalities::new(t);
    ext.execute_with(|| System::set_block_number(1));
    ext
}

/// Run test to block N
pub fn run_to_block(n: u64) {
    while System::block_number() < n {
        if System::block_number() > 1 {
            <frame_system::Pallet<Test> as Hooks<u64>>::on_finalize(System::block_number());
        }
        System::set_block_number(System::block_number() + 1);
        <frame_system::Pallet<Test> as Hooks<u64>>::on_initialize(System::block_number());
        <IcnBft as Hooks<u64>>::on_finalize(System::block_number());
    }
}
