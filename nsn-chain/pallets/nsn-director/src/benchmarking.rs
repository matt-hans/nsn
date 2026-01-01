// Copyright 2024 Neural Sovereign Network
// This file is part of NSN Chain.
//
// NSN Chain is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

//! Benchmarking setup for pallet-nsn-director

#![cfg(feature = "runtime-benchmarks")]

use super::*;
use frame_benchmarking::v2::*;
use frame_support::BoundedVec;
use frame_system::RawOrigin;
use sp_std::vec;

#[benchmarks]
mod benchmarks {
    use super::*;

    #[benchmark]
    fn submit_bft_result() {
        // Setup: Create caller and stake
        let caller: T::AccountId = whitelisted_caller();

        // TODO: Set up elected directors for benchmark slot

        #[extrinsic_call]
        submit_bft_result(
            RawOrigin::Signed(caller),
            100u64,
            BoundedVec::try_from(vec![]).unwrap(),
            T::Hash::default(),
        );
    }

    #[benchmark]
    fn challenge_bft_result() {
        let caller: T::AccountId = whitelisted_caller();

        // TODO: Set up BFT result to challenge

        #[extrinsic_call]
        challenge_bft_result(RawOrigin::Signed(caller), 100u64, T::Hash::default());
    }

    #[benchmark]
    fn resolve_challenge() {
        // TODO: Set up pending challenge

        #[extrinsic_call]
        resolve_challenge(
            RawOrigin::Root,
            100u64,
            BoundedVec::try_from(vec![]).unwrap(),
        );
    }
}
