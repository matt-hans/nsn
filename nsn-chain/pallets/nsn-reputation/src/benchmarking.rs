// Copyright 2024 Neural Sovereign Network
// This file is part of NSN Chain.
//
// NSN Chain is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

//! Benchmarking for pallet-nsn-reputation
//!
//! # Running Benchmarks
//!
//! ```bash
//! # Build the node with benchmarking feature
//! cargo build --release --features runtime-benchmarks
//!
//! # Run benchmarks for this pallet
//! ./target/release/icn-node benchmark pallet \
//!   --chain dev \
//!   --pallet pallet_nsn_reputation \
//!   --extrinsics '*' \
//!   --steps 50 \
//!   --repeat 20 \
//!   --output ./pallets/nsn-reputation/src/weights.rs
//! ```

#![cfg(feature = "runtime-benchmarks")]

use super::*;
use frame_benchmarking::v2::*;
use frame_support::pallet_prelude::Hooks;
use frame_support::traits::Get;
use frame_system::RawOrigin;

#[benchmarks]
mod benchmarks {
    use super::*;

    #[benchmark]
    fn record_event() {
        let caller: T::AccountId = whitelisted_caller();
        let event_type = ReputationEventType::DirectorSlotAccepted;
        let slot = 100u64;

        #[extrinsic_call]
        record_event(RawOrigin::Root, caller.clone(), event_type, slot);

        assert_eq!(ReputationScores::<T>::get(caller).director_score, 100);
    }

    #[benchmark]
    fn on_finalize_with_events() {
        let block = 1000u32.into();

        let max_events = T::MaxEventsPerBlock::get();
        for i in 0..max_events {
            let account = account::<T::AccountId>("account", i as u32, 0);
            let event = ReputationEvent {
                account: account.clone(),
                event_type: ReputationEventType::SeederChunkServed,
                slot: 0u64,
                block,
            };
            PendingEvents::<T>::mutate(|events| {
                let _ = events.try_push(event);
            });
        }

        #[block]
        {
            <Pallet<T> as Hooks<_>>::on_finalize(block);
        }

        assert!(MerkleRoots::<T>::get(block).is_some());
    }

    #[benchmark]
    fn on_finalize_with_checkpoint() {
        let block = 1000u32.into();

        for i in 0..1000u32 {
            let account = account::<T::AccountId>("account", i, 0);
            let score = ReputationScore {
                director_score: (i * 10) as u64,
                validator_score: (i * 5) as u64,
                seeder_score: (i * 2) as u64,
                last_activity: 100,
            };
            ReputationScores::<T>::insert(account, score);
        }

        #[block]
        {
            <Pallet<T> as Hooks<_>>::on_finalize(block);
        }

        assert!(Checkpoints::<T>::get(block).is_some());
    }

    #[benchmark]
    fn update_retention() {
        let new_period = 1_000_000u32.into();

        #[extrinsic_call]
        update_retention(RawOrigin::Root, new_period);

        assert_eq!(RetentionPeriod::<T>::get(), new_period);
    }
}
