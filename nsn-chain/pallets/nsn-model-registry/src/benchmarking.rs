// Copyright 2024 Neural Sovereign Network
// This file is part of NSN Chain.

//! Benchmarking for pallet-nsn-model-registry

#![cfg(feature = "runtime-benchmarks")]

use super::*;

#[allow(unused)]
use crate::Pallet as ModelRegistry;
use frame_benchmarking::v2::*;
use frame_system::RawOrigin;

#[benchmarks]
mod benchmarks {
    use super::*;

    #[benchmark]
    fn register_model() {
        let caller: T::AccountId = whitelisted_caller();
        let model_id = BoundedVec::try_from(b"test-model".to_vec()).unwrap();
        let container_cid = BoundedVec::try_from(b"Qm123456789".to_vec()).unwrap();

        #[extrinsic_call]
        register_model(
            RawOrigin::Signed(caller),
            model_id,
            container_cid,
            1000,
            crate::ModelCapabilities::Lane1,
        );
    }

    #[benchmark]
    fn update_capabilities() {
        let caller: T::AccountId = whitelisted_caller();
        let model_id = BoundedVec::try_from(b"test-model".to_vec()).unwrap();

        #[extrinsic_call]
        update_capabilities(
            RawOrigin::Signed(caller),
            model_id,
            crate::ModelState::Hot,
        );
    }

    impl_benchmark_test_suite!(ModelRegistry, crate::mock::new_test_ext(), crate::mock::Test);
}
