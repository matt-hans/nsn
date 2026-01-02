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
    fn start_task() -> Weight;
    fn submit_result() -> Weight;
    fn submit_attestation() -> Weight;
    fn finalize_task() -> Weight;
    fn fail_task() -> Weight;
    fn register_renderer() -> Weight;
    fn deregister_renderer() -> Weight;
}

/// Weights for pallet_nsn_task_market using the Substrate node and recommended hardware.
pub struct SubstrateWeight<T>(PhantomData<T>);
impl<T: frame_system::Config> WeightInfo for SubstrateWeight<T> {
    /// Storage: NsnTaskMarket NextTaskId (r:1 w:1)
    /// Proof: NsnTaskMarket NextTaskId (max_values: Some(1), max_size: Some(8), added: 503, mode: MaxEncodedLen)
    /// Storage: NsnTaskMarket Tasks (r:0 w:1)
    /// Proof: NsnTaskMarket Tasks (max_values: None, max_size: Some(200), added: 2675, mode: MaxEncodedLen)
    /// Storage: NsnTaskMarket OpenLane0Tasks (r:1 w:1)
    /// Storage: NsnTaskMarket OpenLane1Tasks (r:1 w:1)
    fn create_task_intent() -> Weight {
        // PoV size: NextTaskId + Tasks + OpenLane* queue + overhead
        Weight::from_parts(50_000_000, 5000)
            .saturating_add(T::DbWeight::get().reads(2))
            .saturating_add(T::DbWeight::get().writes(3))
    }

    /// Storage: NsnTaskMarket Tasks (r:1 w:1)
    /// Proof: NsnTaskMarket Tasks (max_values: None, max_size: Some(200), added: 2675, mode: MaxEncodedLen)
    /// Storage: NsnTaskMarket OpenLane0Tasks (r:1 w:1)
    /// Storage: NsnTaskMarket OpenLane1Tasks (r:1 w:1)
    /// Storage: NsnTaskMarket AssignedLane1Tasks (r:1 w:1)
    fn accept_assignment() -> Weight {
        Weight::from_parts(55_000_000, 5500)
            .saturating_add(T::DbWeight::get().reads(3))
            .saturating_add(T::DbWeight::get().writes(3))
    }

    /// Storage: NsnTaskMarket Tasks (r:1 w:1)
    fn start_task() -> Weight {
        Weight::from_parts(35_000_000, 2500)
            .saturating_add(T::DbWeight::get().reads(1))
            .saturating_add(T::DbWeight::get().writes(1))
    }

    /// Storage: NsnTaskMarket Tasks (r:1 w:1)
    /// Proof: NsnTaskMarket Tasks (max_values: None, max_size: Some(200), added: 2675, mode: MaxEncodedLen)
    fn submit_result() -> Weight {
        Weight::from_parts(65_000_000, 4000)
            .saturating_add(T::DbWeight::get().reads(3))
            .saturating_add(T::DbWeight::get().writes(5))
    }

    /// Storage: NsnTaskMarket Tasks (r:1 w:1)
    /// Storage: NsnTaskMarket Attestations (r:1 w:1)
    fn submit_attestation() -> Weight {
        Weight::from_parts(55_000_000, 3000)
            .saturating_add(T::DbWeight::get().reads(3))
            .saturating_add(T::DbWeight::get().writes(3))
    }

    /// Storage: NsnTaskMarket Tasks (r:1 w:1)
    /// Storage: NsnTaskMarket Attestations (r:1 w:1)
    fn finalize_task() -> Weight {
        Weight::from_parts(70_000_000, 4500)
            .saturating_add(T::DbWeight::get().reads(3))
            .saturating_add(T::DbWeight::get().writes(4))
    }

    /// Storage: NsnTaskMarket Tasks (r:1 w:1)
    /// Proof: NsnTaskMarket Tasks (max_values: None, max_size: Some(200), added: 2675, mode: MaxEncodedLen)
    /// Storage: NsnTaskMarket OpenLane0Tasks (r:1 w:1)
    /// Storage: NsnTaskMarket OpenLane1Tasks (r:1 w:1)
    /// Storage: NsnTaskMarket AssignedLane1Tasks (r:1 w:1)
    fn fail_task() -> Weight {
        Weight::from_parts(50_000_000, 4000)
            .saturating_add(T::DbWeight::get().reads(3))
            .saturating_add(T::DbWeight::get().writes(3))
    }

    /// Storage: NsnTaskMarket RendererRegistry (r:1 w:1)
    fn register_renderer() -> Weight {
        Weight::from_parts(20_000_000, 2000)
            .saturating_add(T::DbWeight::get().reads(1))
            .saturating_add(T::DbWeight::get().writes(1))
    }

    /// Storage: NsnTaskMarket RendererRegistry (r:1 w:1)
    fn deregister_renderer() -> Weight {
        Weight::from_parts(15_000_000, 1500)
            .saturating_add(T::DbWeight::get().reads(1))
            .saturating_add(T::DbWeight::get().writes(1))
    }
}

// For backwards compatibility and tests
impl WeightInfo for () {
    fn create_task_intent() -> Weight {
        Weight::from_parts(50_000_000, 5000)
    }
    fn accept_assignment() -> Weight {
        Weight::from_parts(55_000_000, 5500)
    }
    fn start_task() -> Weight {
        Weight::from_parts(35_000_000, 2500)
    }
    fn submit_result() -> Weight {
        Weight::from_parts(65_000_000, 4000)
    }
    fn submit_attestation() -> Weight {
        Weight::from_parts(55_000_000, 3000)
    }
    fn finalize_task() -> Weight {
        Weight::from_parts(70_000_000, 4500)
    }
    fn fail_task() -> Weight {
        Weight::from_parts(50_000_000, 4000)
    }
    fn register_renderer() -> Weight {
        Weight::from_parts(20_000_000, 2000)
    }
    fn deregister_renderer() -> Weight {
        Weight::from_parts(15_000_000, 1500)
    }
}
