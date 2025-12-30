// Copyright 2024 Neural Sovereign Network
// This file is part of NSN Chain.

//! Weights for pallet-nsn-task-market
//!
//! THIS FILE WAS AUTO-GENERATED USING THE SUBSTRATE BENCHMARK CLI VERSION 4.0.0-dev
//! DATE: Placeholder - benchmarks to be run
//! HOSTNAME: Placeholder
//! CPU: Placeholder
//!
//! NOTE: Runtime benchmarking not yet performed. These are placeholder weights
//! with estimated PoV (Proof of Validity) sizes for Cumulus compatibility.
//!
//! PoV Size Estimation:
//! - Storage item size is estimated from MaxEncodedLen
//! - PoV includes: storage key prefix (32 bytes) + key (32 bytes) + value

#![cfg_attr(rustfmt, rustfmt_skip)]
#![allow(unused_parens)]
#![allow(unused_imports)]

use frame_support::{traits::Get, weights::Weight};
use sp_std::marker::PhantomData;

/// Weight functions needed for pallet_nsn_task_market.
pub trait WeightInfo {
    fn create_task_intent() -> Weight;
    fn accept_assignment() -> Weight;
    fn complete_task() -> Weight;
    fn fail_task() -> Weight;
}

/// Weights for pallet_nsn_task_market using the Substrate node and recommended hardware.
pub struct SubstrateWeight<T>(PhantomData<T>);
impl<T: frame_system::Config> WeightInfo for SubstrateWeight<T> {
    /// Storage: NsnTaskMarket NextTaskId (r:1 w:1)
    /// Proof: NsnTaskMarket NextTaskId (max_values: Some(1), max_size: Some(8), added: 503, mode: MaxEncodedLen)
    /// Storage: NsnTaskMarket Tasks (r:0 w:1)
    /// Proof: NsnTaskMarket Tasks (max_values: None, max_size: Some(200), added: 2675, mode: MaxEncodedLen)
    /// Storage: NsnTaskMarket OpenTasks (r:1 w:1)
    /// Proof: NsnTaskMarket OpenTasks (max_values: Some(1), max_size: Some(800), added: 1295, mode: MaxEncodedLen)
    fn create_task_intent() -> Weight {
        // PoV size: NextTaskId(8) + Tasks(200) + OpenTasks(800) + overhead(192) = 1200 bytes
        Weight::from_parts(45_000_000, 4473)
            .saturating_add(T::DbWeight::get().reads(2))
            .saturating_add(T::DbWeight::get().writes(3))
    }

    /// Storage: NsnTaskMarket Tasks (r:1 w:1)
    /// Proof: NsnTaskMarket Tasks (max_values: None, max_size: Some(200), added: 2675, mode: MaxEncodedLen)
    /// Storage: NsnTaskMarket OpenTasks (r:1 w:1)
    /// Proof: NsnTaskMarket OpenTasks (max_values: Some(1), max_size: Some(800), added: 1295, mode: MaxEncodedLen)
    fn accept_assignment() -> Weight {
        // PoV size: Tasks(200) + OpenTasks(800) + overhead(128) = 1128 bytes
        Weight::from_parts(40_000_000, 3970)
            .saturating_add(T::DbWeight::get().reads(2))
            .saturating_add(T::DbWeight::get().writes(2))
    }

    /// Storage: NsnTaskMarket Tasks (r:1 w:1)
    /// Proof: NsnTaskMarket Tasks (max_values: None, max_size: Some(200), added: 2675, mode: MaxEncodedLen)
    fn complete_task() -> Weight {
        // PoV size: Tasks(200) + overhead(64) = 264 bytes
        Weight::from_parts(50_000_000, 2739)
            .saturating_add(T::DbWeight::get().reads(1))
            .saturating_add(T::DbWeight::get().writes(1))
    }

    /// Storage: NsnTaskMarket Tasks (r:1 w:1)
    /// Proof: NsnTaskMarket Tasks (max_values: None, max_size: Some(200), added: 2675, mode: MaxEncodedLen)
    /// Storage: NsnTaskMarket OpenTasks (r:1 w:1)
    /// Proof: NsnTaskMarket OpenTasks (max_values: Some(1), max_size: Some(800), added: 1295, mode: MaxEncodedLen)
    fn fail_task() -> Weight {
        // PoV size: Tasks(200) + OpenTasks(800) + overhead(128) = 1128 bytes
        Weight::from_parts(45_000_000, 3970)
            .saturating_add(T::DbWeight::get().reads(2))
            .saturating_add(T::DbWeight::get().writes(2))
    }
}

// For backwards compatibility and tests
impl WeightInfo for () {
    fn create_task_intent() -> Weight {
        Weight::from_parts(45_000_000, 4473)
    }
    fn accept_assignment() -> Weight {
        Weight::from_parts(40_000_000, 3970)
    }
    fn complete_task() -> Weight {
        Weight::from_parts(50_000_000, 2739)
    }
    fn fail_task() -> Weight {
        Weight::from_parts(45_000_000, 3970)
    }
}
