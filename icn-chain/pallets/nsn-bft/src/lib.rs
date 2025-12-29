// Copyright 2024 Interdimensional Cable Network
// This file is part of ICN Chain.
//
// ICN Chain is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

//! # ICN BFT Pallet
//!
//! BFT consensus result storage and finalization for the Interdimensional Cable Network.
//!
//! ## Overview
//!
//! This pallet implements:
//! - CLIP embeddings hash storage for finalized director consensus rounds
//! - Consensus round metadata tracking (slot, directors, timestamp, success)
//! - Historical slot result queries for auditing and analytics
//! - Aggregate consensus statistics (success rate, average agreement)
//! - Automatic pruning of old consensus data (default: 6 months retention)
//!
//! ## Interface
//!
//! ### Dispatchable Functions (Root-Only)
//!
//! - `store_embeddings_hash`: Record finalized BFT result (called by pallet-nsn-director)
//! - `prune_old_consensus`: Remove consensus data older than retention period
//!
//! ### Query Helpers
//!
//! - `get_slot_result(slot)`: Retrieve consensus round metadata for a slot
//! - `get_embeddings_hash(slot)`: Get canonical CLIP embeddings hash
//! - `get_stats()`: Get aggregate consensus statistics
//! - `get_slot_range(start, end)`: Batch query for slot range
//!
//! ## Hooks
//!
//! - `on_finalize`: Auto-prune every 10000 blocks (~16.7 hours)
//!
//! ## Integration
//!
//! This pallet is designed to be called by `pallet-nsn-director` after BFT finalization:
//!
//! ```ignore
//! // In pallet-nsn-director::finalize_slot()
//! pallet_nsn_bft::Pallet::<T>::store_embeddings_hash(
//!     origin,
//!     slot,
//!     canonical_hash,
//!     directors,
//!     success,
//! )?;
//! ```

#![cfg_attr(not(feature = "std"), no_std)]

pub use pallet::*;

mod types;
pub use types::*;

#[cfg(test)]
mod mock;
#[cfg(test)]
mod tests;

#[cfg(feature = "runtime-benchmarks")]
mod benchmarking;

pub mod weights;
pub use weights::WeightInfo;

extern crate alloc;
use alloc::vec::Vec;

#[frame_support::pallet]
pub mod pallet {
    use super::*;
    use frame_support::{pallet_prelude::*, traits::StorageVersion, BoundedVec};
    use frame_system::pallet_prelude::*;
    use sp_runtime::traits::{Saturating, Zero};

    /// The in-code storage version.
    const STORAGE_VERSION: StorageVersion = StorageVersion::new(0);

    #[pallet::pallet]
    #[pallet::storage_version(STORAGE_VERSION)]
    pub struct Pallet<T>(_);

    /// Configuration trait for the ICN BFT pallet
    #[pallet::config]
    pub trait Config: frame_system::Config<RuntimeEvent: From<Event<Self>>> {
        /// Default retention period in blocks (6 months = 2,592,000 blocks)
        #[pallet::constant]
        type DefaultRetentionPeriod: Get<BlockNumberFor<Self>>;

        /// Weight information for extrinsics
        type WeightInfo: WeightInfo;
    }

    // =========================================================================
    // Storage Items
    // =========================================================================

    /// Canonical CLIP embeddings hash for each slot.
    ///
    /// Maps `slot → Hash` for quick lookup of the agreed embeddings.
    /// ZERO_HASH (all zeros) indicates failed consensus.
    ///
    /// # Storage Key
    ///
    /// Uses `Twox64Concat` hasher for performance (non-cryptographic is acceptable
    /// since slot numbers are not attacker-controlled).
    #[pallet::storage]
    #[pallet::getter(fn embeddings_hashes)]
    pub type EmbeddingsHashes<T: Config> = StorageMap<_, Twox64Concat, u64, T::Hash, OptionQuery>;

    /// Full consensus round metadata for each slot.
    ///
    /// Stores `ConsensusRound` struct with directors, timestamp, success flag.
    /// Provides historical record for auditing and analytics.
    #[pallet::storage]
    #[pallet::getter(fn consensus_rounds)]
    pub type ConsensusRounds<T: Config> =
        StorageMap<_, Twox64Concat, u64, ConsensusRound<T>, OptionQuery>;

    /// Aggregate consensus statistics across all rounds.
    ///
    /// Tracks total rounds, successful rounds, failed rounds, and average directors agreeing.
    /// Updated atomically with each `store_embeddings_hash()` call.
    #[pallet::storage]
    #[pallet::getter(fn consensus_stats)]
    pub type ConsensusRoundStats<T: Config> = StorageValue<_, ConsensusStats, ValueQuery>;

    /// Consensus data retention period in blocks.
    ///
    /// Governance-adjustable parameter controlling how long historical consensus
    /// data is kept before auto-pruning.
    ///
    /// Default: 2,592,000 blocks (~6 months at 6s/block)
    #[pallet::storage]
    #[pallet::getter(fn retention_period)]
    pub type RetentionPeriod<T: Config> =
        StorageValue<_, BlockNumberFor<T>, ValueQuery, T::DefaultRetentionPeriod>;

    // =========================================================================
    // Events
    // =========================================================================

    #[pallet::event]
    #[pallet::generate_deposit(pub(super) fn deposit_event)]
    pub enum Event<T: Config> {
        /// Consensus result stored for a slot
        ConsensusStored {
            /// Slot number
            slot: u64,
            /// Canonical embeddings hash
            embeddings_hash: T::Hash,
            /// Whether consensus was successful
            success: bool,
        },
        /// Old consensus data pruned
        ConsensusPruned {
            /// Slots before this were pruned
            before_slot: u64,
            /// Number of slots pruned
            count: u32,
        },
    }

    // =========================================================================
    // Errors
    // =========================================================================

    #[pallet::error]
    pub enum Error<T> {
        /// Too many directors provided (max 5)
        TooManyDirectors,
        /// Slot already has stored consensus
        SlotAlreadyStored,
        /// Arithmetic overflow in statistics calculation
        ArithmeticOverflow,
    }

    // =========================================================================
    // Extrinsics
    // =========================================================================

    #[pallet::call]
    impl<T: Config> Pallet<T> {
        /// Store finalized BFT consensus result.
        ///
        /// **Origin**: Root only (called by other pallets)
        ///
        /// This is the primary write operation for the BFT pallet. It records:
        /// - Canonical CLIP embeddings hash
        /// - Directors who participated
        /// - Timestamp (current block)
        /// - Success/failure flag
        ///
        /// Also updates aggregate statistics atomically.
        ///
        /// # Arguments
        ///
        /// * `slot` - Slot number this result applies to
        /// * `embeddings_hash` - Canonical CLIP embeddings hash (or ZERO_HASH if failed)
        /// * `directors` - Directors who participated (max 5)
        /// * `success` - Whether BFT consensus was reached
        ///
        /// # Errors
        ///
        /// * `TooManyDirectors` - More than 5 directors provided
        /// * `SlotAlreadyStored` - Slot already has consensus stored
        ///
        /// # Example
        ///
        /// ```ignore
        /// // Called from pallet-nsn-director after finalization
        /// pallet_nsn_bft::Pallet::<T>::store_embeddings_hash(
        ///     frame_system::RawOrigin::Root.into(),
        ///     slot,
        ///     canonical_hash,
        ///     directors,
        ///     true, // success
        /// )?;
        /// ```
        #[pallet::call_index(0)]
        #[pallet::weight(T::WeightInfo::store_embeddings_hash())]
        pub fn store_embeddings_hash(
            origin: OriginFor<T>,
            slot: u64,
            embeddings_hash: T::Hash,
            directors: Vec<T::AccountId>,
            success: bool,
        ) -> DispatchResult {
            // Only callable by root (other pallets)
            ensure_root(origin)?;

            // Validate directors count
            ensure!(
                directors.len() <= MAX_DIRECTORS_PER_ROUND as usize,
                Error::<T>::TooManyDirectors
            );

            // Ensure slot not already stored (prevent double-storage)
            ensure!(
                !EmbeddingsHashes::<T>::contains_key(slot),
                Error::<T>::SlotAlreadyStored
            );

            let current_block = <frame_system::Pallet<T>>::block_number();

            // Store embeddings hash
            EmbeddingsHashes::<T>::insert(slot, embeddings_hash);

            // Store full round metadata
            let bounded_directors =
                BoundedVec::<T::AccountId, ConstU32<MAX_DIRECTORS_PER_ROUND>>::try_from(
                    directors.clone(),
                )
                .expect("Directors length already checked");

            let round = ConsensusRound {
                slot,
                embeddings_hash,
                directors: bounded_directors,
                timestamp: current_block,
                success,
            };
            ConsensusRounds::<T>::insert(slot, round);

            // Update aggregate statistics
            ConsensusRoundStats::<T>::mutate(|stats| {
                stats.total_rounds = stats.total_rounds.saturating_add(1);

                if success {
                    stats.successful_rounds = stats.successful_rounds.saturating_add(1);

                    // Update moving average of directors agreeing (fixed-point × 100)
                    let director_count = directors.len() as u64;
                    let prev_total = stats.total_rounds.saturating_sub(1);

                    if prev_total == 0 {
                        // First successful round
                        stats.average_directors_agreeing = (director_count * 100) as u32;
                    } else {
                        // Moving average: ((prev_avg × prev_count) + (new_count × 100)) / total_count
                        let prev_sum =
                            (stats.average_directors_agreeing as u64).saturating_mul(prev_total);
                        let new_contribution = director_count.saturating_mul(100);
                        let new_avg = prev_sum
                            .saturating_add(new_contribution)
                            .checked_div(stats.total_rounds)
                            .unwrap_or(stats.average_directors_agreeing as u64);

                        stats.average_directors_agreeing = new_avg as u32;
                    }
                } else {
                    stats.failed_rounds = stats.failed_rounds.saturating_add(1);
                }
            });

            Self::deposit_event(Event::ConsensusStored {
                slot,
                embeddings_hash,
                success,
            });

            Ok(())
        }

        /// Prune old consensus data beyond retention period.
        ///
        /// **Origin**: Root only
        ///
        /// Removes consensus rounds older than `before_slot` to manage storage costs.
        /// Typically called by governance or automated via `on_finalize` hook.
        ///
        /// # Arguments
        ///
        /// * `before_slot` - Remove all consensus data for slots < this value
        ///
        /// # Weight
        ///
        /// O(N) where N is number of slots pruned. Bounded by max storage iterations.
        ///
        /// # Example
        ///
        /// ```ignore
        /// // Prune slots before slot 1000
        /// pallet_nsn_bft::Pallet::<T>::prune_old_consensus(
        ///     frame_system::RawOrigin::Root.into(),
        ///     1000,
        /// )?;
        /// ```
        #[pallet::call_index(1)]
        #[pallet::weight(T::WeightInfo::prune_old_consensus())]
        pub fn prune_old_consensus(origin: OriginFor<T>, before_slot: u64) -> DispatchResult {
            ensure_root(origin)?;

            let mut pruned = 0u32;

            // Iterate over all stored embeddings hashes
            // Note: In production, this should be paginated to avoid excessive weight
            let keys_to_remove: Vec<u64> = EmbeddingsHashes::<T>::iter_keys()
                .filter(|&slot| slot < before_slot)
                .collect();

            for slot in keys_to_remove {
                EmbeddingsHashes::<T>::remove(slot);
                ConsensusRounds::<T>::remove(slot);
                pruned = pruned.saturating_add(1);
            }

            if pruned > 0 {
                Self::deposit_event(Event::ConsensusPruned {
                    before_slot,
                    count: pruned,
                });
            }

            Ok(())
        }
    }

    // =========================================================================
    // Hooks
    // =========================================================================

    #[pallet::hooks]
    impl<T: Config> Hooks<BlockNumberFor<T>> for Pallet<T> {
        /// Reserve weight for on_finalize operations.
        fn on_initialize(_n: BlockNumberFor<T>) -> Weight {
            // Reserve weight for potential pruning in on_finalize
            T::DbWeight::get().reads_writes(10, 10)
        }

        /// Auto-prune old consensus data every 10000 blocks.
        ///
        /// Calculates cutoff based on retention period and current block,
        /// then removes consensus data older than that threshold.
        fn on_finalize(block: BlockNumberFor<T>) {
            // Auto-prune every AUTO_PRUNE_FREQUENCY blocks (~16.7 hours)
            let frequency: BlockNumberFor<T> = AUTO_PRUNE_FREQUENCY.into();

            if block % frequency == Zero::zero() {
                let retention = RetentionPeriod::<T>::get();
                let cutoff_block = block.saturating_sub(retention);

                // Convert block number to approximate slot
                // Assuming BLOCKS_PER_SLOT = 8 (from pallet-nsn-director)
                let cutoff_slot = TryInto::<u64>::try_into(cutoff_block)
                    .unwrap_or(0)
                    .saturating_div(8);

                // Attempt to prune (ignore errors in hook)
                let _ =
                    Self::prune_old_consensus(frame_system::RawOrigin::Root.into(), cutoff_slot);
            }
        }
    }

    // =========================================================================
    // Query Helpers (Public API)
    // =========================================================================

    impl<T: Config> Pallet<T> {
        /// Get consensus result for a specific slot.
        ///
        /// Returns full `ConsensusRound` metadata if available.
        ///
        /// # Weight
        ///
        /// Single storage read: O(1)
        ///
        /// # Example
        ///
        /// ```ignore
        /// if let Some(round) = pallet_nsn_bft::Pallet::<T>::get_slot_result(slot) {
        ///     println!("Slot {} consensus: {:?}", slot, round.success);
        /// }
        /// ```
        pub fn get_slot_result(slot: u64) -> Option<ConsensusRound<T>> {
            ConsensusRounds::<T>::get(slot)
        }

        /// Get embeddings hash for a specific slot.
        ///
        /// Returns canonical CLIP embeddings hash, or None if slot not found.
        ///
        /// # Weight
        ///
        /// Single storage read: O(1)
        pub fn get_embeddings_hash(slot: u64) -> Option<T::Hash> {
            EmbeddingsHashes::<T>::get(slot)
        }

        /// Get aggregate consensus statistics.
        ///
        /// Returns current network health metrics.
        ///
        /// # Weight
        ///
        /// Single storage read: O(1)
        pub fn get_stats() -> ConsensusStats {
            ConsensusRoundStats::<T>::get()
        }

        /// Get consensus results for a range of slots.
        ///
        /// Returns Vec of `ConsensusRound` for slots in [start, end] (inclusive).
        /// Only returns slots that have stored consensus.
        ///
        /// # Weight
        ///
        /// O(N) where N = end - start + 1
        ///
        /// # Arguments
        ///
        /// * `start` - First slot in range (inclusive)
        /// * `end` - Last slot in range (inclusive)
        ///
        /// # Example
        ///
        /// ```ignore
        /// let rounds = pallet_nsn_bft::Pallet::<T>::get_slot_range(100, 110);
        /// // Returns up to 11 ConsensusRound structs
        /// ```
        pub fn get_slot_range(start: u64, end: u64) -> Vec<ConsensusRound<T>> {
            (start..=end)
                .filter_map(|slot| Self::get_slot_result(slot))
                .collect()
        }
    }
}
