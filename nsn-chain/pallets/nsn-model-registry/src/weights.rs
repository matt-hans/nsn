// Copyright 2024 Neural Sovereign Network
// This file is part of NSN Chain.

//! Weights for pallet-nsn-model-registry
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

/// Weight functions needed for pallet_nsn_model_registry.
pub trait WeightInfo {
    fn register_model() -> Weight;
    fn update_capabilities() -> Weight;
}

/// Weights for pallet_nsn_model_registry using the Substrate node and recommended hardware.
pub struct SubstrateWeight<T>(PhantomData<T>);
impl<T: frame_system::Config> WeightInfo for SubstrateWeight<T> {
    /// Storage: NsnModelRegistry ModelCatalog (r:1 w:1)
    /// Proof: NsnModelRegistry ModelCatalog (max_values: None, max_size: Some(200), added: 2675, mode: MaxEncodedLen)
    fn register_model() -> Weight {
        // PoV size: ModelCatalog(~200) + overhead(64) = 264 bytes
        Weight::from_parts(35_000_000, 2739)
            .saturating_add(T::DbWeight::get().reads(1))
            .saturating_add(T::DbWeight::get().writes(1))
    }

    /// Storage: NsnModelRegistry NodeCapabilities (r:0 w:1)
    /// Proof: NsnModelRegistry NodeCapabilities (max_values: None, max_size: Some(800), added: 3275, mode: MaxEncodedLen)
    fn update_capabilities() -> Weight {
        // PoV size: NodeCapabilities(~800) + overhead(64) = 864 bytes
        Weight::from_parts(30_000_000, 3339)
            .saturating_add(T::DbWeight::get().writes(1))
    }
}

// For backwards compatibility and tests
impl WeightInfo for () {
    fn register_model() -> Weight {
        Weight::from_parts(35_000_000, 2739)
    }
    fn update_capabilities() -> Weight {
        Weight::from_parts(30_000_000, 3339)
    }
}
