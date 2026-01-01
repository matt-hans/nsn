// Copyright 2024 Neural Sovereign Network
// This file is part of NSN Chain.
//
// NSN Chain is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

//! # NSN Director Pallet
//!
//! Multi-director election, BFT coordination, and challenge mechanism for the
//! Neural Sovereign Network.
//!
//! ## Overview
//!
//! This pallet implements:
//! - VRF-based election of 5 directors per slot
//! - Multi-region distribution (max 2 directors per region)
//! - 3-of-5 BFT consensus tracking
//! - 50-block challenge period with stake slashing
//! - Reputation-weighted selection with sqrt scaling and ±20% jitter
//!
//! ## Interface
//!
//! ### Dispatchable Functions
//!
//! - `submit_bft_result`: Submit BFT consensus result for a slot
//! - `challenge_bft_result`: Challenge a submitted result (25 NSN bond)
//! - `resolve_challenge`: Resolve challenge with validator attestations
//!
//! ## Hooks
//!
//! - `on_initialize`: Triggers slot transitions and director elections
//! - `on_finalize`: Auto-finalizes unchallenged results after 50 blocks
//!
//! ## Election Algorithm
//!
//! 1. Get eligible candidates (Director role + past cooldown)
//! 2. Calculate weights: sqrt(reputation + 1) × (100 ± jitter%)
//! 3. Apply region boosting: under-represented regions get 2× weight
//! 4. Use VRF randomness for cryptographically secure selection
//! 5. Select 5 directors respecting max 2 per region constraint

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
    use frame_support::{
        pallet_prelude::*,
        traits::{
            fungible::{Balanced, Inspect, MutateHold},
            tokens::{Fortitude, Precision, Preservation},
            Randomness, StorageVersion,
        },
    };
    use frame_system::pallet_prelude::*;
    use pallet_nsn_reputation::ReputationEventType;
    use pallet_nsn_stake::{NodeMode, NodeModeUpdater, NodeRole, NodeRoleUpdater, SlashReason};
    use sp_runtime::traits::{Hash, SaturatedConversion, Saturating, Zero};

    /// Balance type from the currency trait (our pallet's Currency)
    pub type BalanceOf<T> =
        <<T as Config>::Currency as Inspect<<T as frame_system::Config>::AccountId>>::Balance;

    /// Balance type from the stake pallet's currency trait
    pub type StakeBalanceOf<T> = <<T as pallet_nsn_stake::Config>::Currency as Inspect<
        <T as frame_system::Config>::AccountId,
    >>::Balance;

    /// The in-code storage version.
    const STORAGE_VERSION: StorageVersion = StorageVersion::new(0);

    #[pallet::pallet]
    #[pallet::storage_version(STORAGE_VERSION)]
    pub struct Pallet<T>(_);

    /// Configuration trait for the NSN Director pallet
    #[pallet::config]
    pub trait Config:
        frame_system::Config + pallet_nsn_stake::Config + pallet_nsn_reputation::Config
    {
        /// The currency type for bonds and slashing
        type Currency: Inspect<Self::AccountId>
            + MutateHold<Self::AccountId, Reason = Self::RuntimeHoldReason>
            + Balanced<Self::AccountId>;

        /// The overarching hold reason
        type RuntimeHoldReason: From<HoldReason>;

        /// Randomness source for VRF-based election
        type Randomness: Randomness<Self::Hash, BlockNumberFor<Self>>;

        /// Challenge bond amount (25 NSN)
        #[pallet::constant]
        type ChallengeBond: Get<BalanceOf<Self>>;

        /// Director slash amount for fraud (100 NSN) - must be convertible to stake pallet's balance
        #[pallet::constant]
        type DirectorSlashAmount: Get<StakeBalanceOf<Self>>;

        /// Challenger reward for upheld challenge (10 NSN)
        #[pallet::constant]
        type ChallengerReward: Get<BalanceOf<Self>>;

        /// Maximum elected directors per slot (L0 constraint)
        #[pallet::constant]
        type MaxDirectorsPerSlot: Get<u32>;

        /// Maximum pending slots to track (L0 constraint)
        #[pallet::constant]
        type MaxPendingSlots: Get<u32>;

        /// Epoch duration in blocks (1 hour at 6s/block = 600 blocks)
        #[pallet::constant]
        type EpochDuration: Get<BlockNumberFor<Self>>;

        /// Lookahead blocks for On-Deck notification (2 min = 20 blocks)
        #[pallet::constant]
        type EpochLookahead: Get<BlockNumberFor<Self>>;

        /// Maximum directors per epoch
        #[pallet::constant]
        type MaxDirectorsPerEpoch: Get<u32>;

        /// Node mode updater (loose coupling to stake pallet)
        type NodeModeUpdater: NodeModeUpdater<Self::AccountId>;

        /// Node role updater (loose coupling to stake pallet)
        type NodeRoleUpdater: NodeRoleUpdater<Self::AccountId>;

        /// Weight information for extrinsics
        type WeightInfo: WeightInfo;
    }

    /// The reason for holding funds
    #[pallet::composite_enum]
    pub enum HoldReason {
        /// Funds held for challenge bond
        ChallengeBond,
    }

    // =========================================================================
    // Storage Items
    // =========================================================================

    /// Current slot number
    ///
    /// Calculated from block number: slot = block / BLOCKS_PER_SLOT
    #[pallet::storage]
    #[pallet::getter(fn current_slot)]
    pub type CurrentSlot<T: Config> = StorageValue<_, u64, ValueQuery>;

    /// Elected directors for a given slot
    ///
    /// Populated 2 slots ahead (lookahead) to give directors time
    /// for off-chain BFT coordination before their slot starts.
    #[pallet::storage]
    #[pallet::getter(fn elected_directors)]
    pub type ElectedDirectors<T: Config> = StorageMap<
        _,
        Twox64Concat,
        u64, // slot
        BoundedVec<T::AccountId, T::MaxDirectorsPerSlot>,
        ValueQuery,
    >;

    /// Cooldown tracker for each director
    ///
    /// Maps account to the last slot they directed.
    /// Directors cannot be re-elected until last_slot + COOLDOWN_SLOTS.
    #[pallet::storage]
    #[pallet::getter(fn cooldowns)]
    pub type Cooldowns<T: Config> = StorageMap<_, Blake2_128Concat, T::AccountId, u64, ValueQuery>;

    /// BFT results for each slot
    ///
    /// Stored when `submit_bft_result()` is called.
    /// Contains the consensus hash and attestations.
    #[pallet::storage]
    #[pallet::getter(fn bft_results)]
    pub type BftResults<T: Config> = StorageMap<
        _,
        Twox64Concat,
        u64, // slot
        BftConsensusResult<T::AccountId, T::Hash>,
        OptionQuery,
    >;

    /// Pending challenges for each slot
    ///
    /// Created when `challenge_bft_result()` is called.
    /// Resolved via `resolve_challenge()` or auto-expired.
    #[pallet::storage]
    #[pallet::getter(fn pending_challenges)]
    pub type PendingChallenges<T: Config> = StorageMap<
        _,
        Twox64Concat,
        u64, // slot
        BftChallenge<T::AccountId, T::Hash>,
        OptionQuery,
    >;

    /// Finalized slots tracker
    ///
    /// True if slot has been finalized (either after challenge period
    /// or after challenge resolution).
    #[pallet::storage]
    #[pallet::getter(fn finalized_slots)]
    pub type FinalizedSlots<T: Config> = StorageMap<_, Twox64Concat, u64, bool, ValueQuery>;

    /// Slot status for tracking lifecycle
    #[pallet::storage]
    #[pallet::getter(fn slot_status)]
    pub type SlotStatuses<T: Config> = StorageMap<_, Twox64Concat, u64, SlotStatus, ValueQuery>;

    // =========================================================================
    // Epoch Management Storage
    // =========================================================================

    /// Current epoch information
    ///
    /// Tracks the active epoch with its directors and lifecycle status.
    #[pallet::storage]
    #[pallet::getter(fn current_epoch)]
    pub type CurrentEpoch<T: Config> = StorageValue<
        _,
        Epoch<BlockNumberFor<T>, T::AccountId, T::MaxDirectorsPerEpoch>,
        OptionQuery,
    >;

    /// Directors elected for next epoch (lookahead)
    ///
    /// Populated 20 blocks before epoch transition to enable On-Deck protocol.
    /// Directors in this list are notified to prepare for their upcoming shift.
    #[pallet::storage]
    #[pallet::getter(fn next_epoch_directors)]
    pub type NextEpochDirectors<T: Config> =
        StorageValue<_, BoundedVec<T::AccountId, T::MaxDirectorsPerEpoch>, ValueQuery>;

    // =========================================================================
    // Events
    // =========================================================================

    #[pallet::event]
    #[pallet::generate_deposit(pub(super) fn deposit_event)]
    pub enum Event<T: Config> {
        /// New slot started
        SlotStarted { slot: u64 },
        /// Directors elected for a future slot
        DirectorsElected {
            slot: u64,
            directors: Vec<T::AccountId>,
        },
        /// BFT result submitted, entering challenge period
        BftResultPending {
            slot: u64,
            canonical_director: T::AccountId,
            deadline: u64,
        },
        /// BFT result challenged
        BftChallenged { slot: u64, challenger: T::AccountId },
        /// Challenge upheld - directors slashed
        ChallengeUpheld { slot: u64, challenger: T::AccountId },
        /// Challenge rejected - challenger slashed
        ChallengeRejected { slot: u64, challenger: T::AccountId },
        /// BFT consensus finalized for slot
        BftConsensusFinalized { slot: u64 },
        /// Slot failed - insufficient directors or consensus
        SlotFailed { slot: u64, reason: Vec<u8> },
        /// Directors elected for upcoming epoch (On-Deck notification)
        OnDeckElection {
            epoch_id: EpochId,
            directors: Vec<T::AccountId>,
            start_block: BlockNumberFor<T>,
        },
        /// New epoch started
        EpochStarted {
            epoch_id: EpochId,
            directors: Vec<T::AccountId>,
            end_block: BlockNumberFor<T>,
        },
        /// Epoch completed
        EpochCompleted { epoch_id: EpochId },
    }

    // =========================================================================
    // Errors
    // =========================================================================

    #[pallet::error]
    pub enum Error<T> {
        /// Submitter is not an elected director for this slot
        NotElectedDirector,
        /// Insufficient agreement (need 3-of-5)
        InsufficientAgreement,
        /// BFT result not found for slot
        ResultNotFound,
        /// Slot already finalized
        AlreadyFinalized,
        /// Challenge already exists for slot
        ChallengeExists,
        /// Insufficient stake for challenge bond
        InsufficientChallengeStake,
        /// No challenge exists for slot
        NoChallengeExists,
        /// Challenge already resolved
        ChallengeAlreadyResolved,
        /// BFT result already submitted for slot
        ResultAlreadySubmitted,
        /// Not enough eligible directors for election
        InsufficientDirectors,
        /// Challenge deadline has passed
        ChallengeDeadlinePassed,
        /// Director list exceeds maximum
        TooManyDirectors,
        /// Arithmetic overflow
        Overflow,
    }

    // =========================================================================
    // Hooks
    // =========================================================================

    #[pallet::hooks]
    impl<T: Config> Hooks<BlockNumberFor<T>> for Pallet<T> {
        /// Block initialization hook
        ///
        /// Handles epoch-based elections and transitions:
        /// 1. Checks if it's time to elect next epoch's directors (lookahead)
        /// 2. Checks if current epoch is ending and needs transition
        /// 3. Falls back to slot-based election if no epoch is active
        fn on_initialize(block: BlockNumberFor<T>) -> Weight {
            let block_num: u64 = block.saturated_into();
            let mut total_weight = T::DbWeight::get().reads(1);

            // Handle epoch-based elections
            if let Some(current_epoch) = Self::current_epoch() {
                let lookahead = T::EpochLookahead::get();
                let election_trigger = current_epoch.end_block.saturating_sub(lookahead);

                // Check 1: Is it time to elect next epoch's directors?
                if block == election_trigger && NextEpochDirectors::<T>::get().is_empty() {
                    total_weight = total_weight.saturating_add(Self::run_epoch_election(block));
                }

                // Check 2: Is current epoch ending?
                if block >= current_epoch.end_block {
                    total_weight = total_weight.saturating_add(Self::transition_epoch(block));
                }
            } else {
                // No active epoch - bootstrap first epoch
                total_weight = total_weight.saturating_add(Self::bootstrap_first_epoch(block));
            }

            // Also handle slot-based logic for backwards compatibility
            let slot = block_num / BLOCKS_PER_SLOT;
            if slot > Self::current_slot() {
                Self::start_new_slot(slot);
                total_weight = total_weight.saturating_add(T::DbWeight::get().reads_writes(2, 3));
            }

            total_weight
        }

        /// Block finalization hook
        ///
        /// Auto-finalizes unchallenged BFT results after challenge period expires.
        fn on_finalize(block: BlockNumberFor<T>) {
            let block_num: u64 = block.saturated_into();

            // Iterate pending slots that might need finalization
            // L0: Bounded by MaxPendingSlots
            for (slot, result) in BftResults::<T>::iter().take(T::MaxPendingSlots::get() as usize) {
                // Skip already finalized slots
                if Self::finalized_slots(slot) {
                    continue;
                }

                // Skip slots with pending challenges
                if Self::pending_challenges(slot).is_some() {
                    // Check if challenge deadline expired
                    if let Some(challenge) = Self::pending_challenges(slot) {
                        if block_num > challenge.deadline && !challenge.resolved {
                            // Challenge expired without resolution - forfeit bond, finalize original
                            let _ = Self::handle_expired_challenge(slot, &challenge);
                        }
                    }
                    continue;
                }

                // Check if challenge period expired
                let deadline = result
                    .submitted_at_block
                    .saturating_add(CHALLENGE_PERIOD_BLOCKS as u64);
                if block_num >= deadline {
                    let _ = Self::finalize_slot(slot);
                }
            }
        }
    }

    // =========================================================================
    // Extrinsics
    // =========================================================================

    #[pallet::call]
    impl<T: Config> Pallet<T> {
        /// Submit BFT consensus result for a slot.
        ///
        /// Called by an elected director after off-chain BFT coordination.
        /// Requires at least 3 agreeing directors (BFT_THRESHOLD).
        ///
        /// # Arguments
        /// * `slot` - Slot number for this result
        /// * `agreeing_directors` - Directors who agreed on the hash (at least 3)
        /// * `embeddings_hash` - Hash of the agreed CLIP embeddings
        ///
        /// # Errors
        /// * `NotElectedDirector` - Submitter not elected for this slot
        /// * `InsufficientAgreement` - Less than 3 directors agreed
        /// * `ResultAlreadySubmitted` - Result already exists for slot
        ///
        /// # Events
        /// * `BftResultPending` - Result submitted, challenge period started
        #[pallet::call_index(0)]
        #[pallet::weight(<T as pallet::Config>::WeightInfo::submit_bft_result())]
        pub fn submit_bft_result(
            origin: OriginFor<T>,
            slot: u64,
            agreeing_directors: BoundedVec<T::AccountId, T::MaxDirectorsPerSlot>,
            embeddings_hash: T::Hash,
        ) -> DispatchResult {
            let submitter = ensure_signed(origin)?;

            // Verify submitter is elected director
            let elected = Self::elected_directors(slot);
            ensure!(elected.contains(&submitter), Error::<T>::NotElectedDirector);

            // Verify sufficient agreement (3-of-5)
            ensure!(
                agreeing_directors.len() >= BFT_THRESHOLD as usize,
                Error::<T>::InsufficientAgreement
            );

            // Verify no existing result
            ensure!(
                !BftResults::<T>::contains_key(slot),
                Error::<T>::ResultAlreadySubmitted
            );

            let current_block: u64 = <frame_system::Pallet<T>>::block_number().saturated_into();
            let deadline = current_block.saturating_add(CHALLENGE_PERIOD_BLOCKS as u64);

            // Store BFT result
            let result = BftConsensusResult {
                slot,
                success: true,
                canonical_hash: embeddings_hash,
                submitter: submitter.clone(),
                submitted_at_block: current_block,
            };
            BftResults::<T>::insert(slot, result);

            // Update slot status
            SlotStatuses::<T>::insert(slot, SlotStatus::Submitted);

            // Update cooldowns for agreeing directors
            for director in agreeing_directors.iter() {
                Cooldowns::<T>::insert(director, slot);
            }

            Self::deposit_event(Event::BftResultPending {
                slot,
                canonical_director: submitter,
                deadline,
            });
            Ok(())
        }

        /// Challenge a BFT result.
        ///
        /// Any staker with at least 25 NSN can challenge a submitted result
        /// before the challenge period expires.
        ///
        /// # Arguments
        /// * `slot` - Slot number to challenge
        /// * `evidence_hash` - Hash of off-chain evidence
        ///
        /// # Errors
        /// * `ResultNotFound` - No BFT result for this slot
        /// * `AlreadyFinalized` - Slot already finalized
        /// * `ChallengeExists` - Challenge already pending
        /// * `InsufficientChallengeStake` - Not enough stake for bond
        ///
        /// # Events
        /// * `BftChallenged` - Challenge submitted
        #[pallet::call_index(1)]
        #[pallet::weight(<T as pallet::Config>::WeightInfo::challenge_bft_result())]
        pub fn challenge_bft_result(
            origin: OriginFor<T>,
            slot: u64,
            evidence_hash: T::Hash,
        ) -> DispatchResult {
            let challenger = ensure_signed(origin)?;

            // Verify result exists and not finalized
            ensure!(
                BftResults::<T>::contains_key(slot),
                Error::<T>::ResultNotFound
            );
            ensure!(!Self::finalized_slots(slot), Error::<T>::AlreadyFinalized);
            ensure!(
                Self::pending_challenges(slot).is_none(),
                Error::<T>::ChallengeExists
            );

            // Verify challenger has sufficient stake
            let challenger_stake = pallet_nsn_stake::Pallet::<T>::stakes(&challenger);
            let challenge_bond = T::ChallengeBond::get();
            // Check that the challenger has sufficient balance for the bond
            let challenger_balance = <T as pallet::Config>::Currency::reducible_balance(
                &challenger,
                Preservation::Preserve,
                Fortitude::Polite,
            );
            ensure!(
                challenger_balance >= challenge_bond,
                Error::<T>::InsufficientChallengeStake
            );
            // Also verify they have stake in the system
            ensure!(
                !challenger_stake.amount.is_zero(),
                Error::<T>::InsufficientChallengeStake
            );

            // Hold the challenge bond
            <T as pallet::Config>::Currency::hold(
                &HoldReason::ChallengeBond.into(),
                &challenger,
                challenge_bond,
            )?;

            let current_block: u64 = <frame_system::Pallet<T>>::block_number().saturated_into();
            let deadline = current_block.saturating_add(CHALLENGE_PERIOD_BLOCKS as u64);

            // Store challenge
            let challenge = BftChallenge {
                slot,
                challenger: challenger.clone(),
                challenge_block: current_block,
                deadline,
                evidence_hash,
                resolved: false,
            };
            PendingChallenges::<T>::insert(slot, challenge);

            // Update slot status
            SlotStatuses::<T>::insert(slot, SlotStatus::Challenged);

            Self::deposit_event(Event::BftChallenged { slot, challenger });
            Ok(())
        }

        /// Resolve a challenge with validator attestations.
        ///
        /// Called by root (governance or automated oracle) after validators
        /// have submitted their attestations off-chain.
        ///
        /// # Arguments
        /// * `slot` - Slot number with pending challenge
        /// * `validator_attestations` - List of (validator, agrees, hash) tuples
        ///
        /// # Resolution Logic
        /// * If majority of validators agree with challenge: Slash directors
        /// * If majority reject challenge: Slash challenger
        ///
        /// # Errors
        /// * `NoChallengeExists` - No pending challenge for slot
        /// * `ChallengeAlreadyResolved` - Challenge already processed
        ///
        /// # Events
        /// * `ChallengeUpheld` - Challenge valid, directors slashed
        /// * `ChallengeRejected` - Challenge invalid, challenger slashed
        #[pallet::call_index(2)]
        #[pallet::weight(<T as pallet::Config>::WeightInfo::resolve_challenge())]
        pub fn resolve_challenge(
            origin: OriginFor<T>,
            slot: u64,
            validator_attestations: BoundedVec<
                ValidatorAttestation<T::AccountId, T::Hash>,
                ConstU32<MAX_VALIDATOR_ATTESTATIONS>,
            >,
        ) -> DispatchResult {
            ensure_root(origin)?;

            let mut challenge =
                Self::pending_challenges(slot).ok_or(Error::<T>::NoChallengeExists)?;
            ensure!(!challenge.resolved, Error::<T>::ChallengeAlreadyResolved);

            // Tally attestations
            let agree_count = validator_attestations
                .iter()
                .filter(|a| a.agrees_with_challenge)
                .count();
            let total = validator_attestations.len();
            let challenge_upheld = agree_count > total / 2;

            if challenge_upheld {
                // Slash directors
                let _result = Self::bft_results(slot).ok_or(Error::<T>::ResultNotFound)?;
                let elected = Self::elected_directors(slot);

                for director in elected.iter() {
                    let _ = pallet_nsn_stake::Pallet::<T>::slash(
                        frame_system::RawOrigin::Root.into(),
                        director.clone(),
                        T::DirectorSlashAmount::get(),
                        SlashReason::BftFailure,
                    );

                    // Record negative reputation
                    let _ = pallet_nsn_reputation::Pallet::<T>::record_event(
                        frame_system::RawOrigin::Root.into(),
                        director.clone(),
                        ReputationEventType::DirectorSlotRejected,
                        slot,
                    );
                }

                // Refund challenger bond + reward
                <T as pallet::Config>::Currency::release(
                    &HoldReason::ChallengeBond.into(),
                    &challenge.challenger,
                    T::ChallengeBond::get(),
                    Precision::Exact,
                )?;

                // Mint reward to challenger
                let _ = <T as pallet::Config>::Currency::deposit(
                    &challenge.challenger,
                    T::ChallengerReward::get(),
                    Precision::Exact,
                );

                // Mark slot as failed
                SlotStatuses::<T>::insert(slot, SlotStatus::Failed);
                FinalizedSlots::<T>::insert(slot, true);

                Self::deposit_event(Event::ChallengeUpheld {
                    slot,
                    challenger: challenge.challenger.clone(),
                });
            } else {
                // Slash challenger bond
                <T as pallet::Config>::Currency::burn_held(
                    &HoldReason::ChallengeBond.into(),
                    &challenge.challenger,
                    T::ChallengeBond::get(),
                    Precision::Exact,
                    Fortitude::Force,
                )?;

                // Finalize original result
                let _ = Self::finalize_slot(slot);

                Self::deposit_event(Event::ChallengeRejected {
                    slot,
                    challenger: challenge.challenger.clone(),
                });
            }

            // Mark challenge as resolved
            challenge.resolved = true;
            PendingChallenges::<T>::insert(slot, challenge);

            Ok(())
        }
    }

    // =========================================================================
    // Helper Functions
    // =========================================================================

    impl<T: Config> Pallet<T> {
        /// Start a new slot and trigger director election.
        ///
        /// Called from `on_initialize` when block crosses slot boundary.
        fn start_new_slot(slot: u64) {
            CurrentSlot::<T>::put(slot);
            Self::deposit_event(Event::SlotStarted { slot });

            // Elect directors for future slot (2-slot lookahead)
            let election_slot = slot.saturating_add(ELECTION_LOOKAHEAD);
            let directors = Self::elect_directors(election_slot);

            // Store elected directors
            if let Ok(bounded_directors) = BoundedVec::try_from(directors.clone()) {
                ElectedDirectors::<T>::insert(election_slot, bounded_directors);
                SlotStatuses::<T>::insert(election_slot, SlotStatus::Elected);

                Self::deposit_event(Event::DirectorsElected {
                    slot: election_slot,
                    directors,
                });
            }
        }

        /// Elect directors for a slot using VRF-weighted selection.
        ///
        /// # Algorithm
        /// 1. Get all eligible candidates (Director role + past cooldown)
        /// 2. Calculate weight = sqrt(reputation + 1) with ±20% jitter
        /// 3. Apply region boost (2× for under-represented regions)
        /// 4. Use VRF randomness for weighted selection
        /// 5. Enforce max 2 directors per region
        pub fn elect_directors(slot: u64) -> Vec<T::AccountId> {
            // Get eligible candidates
            let candidates: Vec<_> = pallet_nsn_stake::Stakes::<T>::iter()
                .filter(|(account, stake)| {
                    // Must be Director role
                    if !stake.role.is_director_like() {
                        return false;
                    }
                    // Check cooldown: either never directed (0) or past cooldown period
                    let last_directed = Self::cooldowns(account);
                    last_directed == 0 || last_directed.saturating_add(COOLDOWN_SLOTS) < slot
                })
                .collect();

            if candidates.is_empty() {
                return Vec::new();
            }

            // Build weighted candidates
            let mut weighted: Vec<(T::AccountId, u64, pallet_nsn_stake::Region)> = candidates
                .iter()
                .map(|(account, stake)| {
                    // Get reputation (apply decay first)
                    let current_block: u64 =
                        <frame_system::Pallet<T>>::block_number().saturated_into();
                    pallet_nsn_reputation::Pallet::<T>::apply_decay(account, current_block);
                    let rep = pallet_nsn_reputation::Pallet::<T>::get_reputation_total(account);

                    // Calculate weight with sqrt scaling
                    let base_weight = rep.saturating_add(1);
                    let scaled = Self::isqrt(base_weight);

                    // Apply deterministic jitter based on slot + account
                    let jitter_seed = T::Hashing::hash_of(&(slot, account));
                    let jitter_bytes: [u8; 4] =
                        jitter_seed.as_ref()[0..4].try_into().unwrap_or([0u8; 4]);
                    let jitter_raw = u32::from_le_bytes(jitter_bytes);
                    let jitter_pct =
                        (jitter_raw % (JITTER_PERCENT * 2)) as i64 - JITTER_PERCENT as i64;
                    let jittered = (scaled as i64)
                        .saturating_mul(100i64.saturating_add(jitter_pct))
                        .saturating_div(100);

                    (account.clone(), jittered.max(1) as u64, stake.region)
                })
                .collect();

            // VRF-based selection
            let mut selected: Vec<T::AccountId> = Vec::new();
            let mut selected_regions: Vec<pallet_nsn_stake::Region> = Vec::new();

            // Get VRF output
            let (vrf_output, _) = T::Randomness::random(&slot.to_le_bytes());
            let vrf_bytes: [u8; 8] = vrf_output.as_ref()[0..8].try_into().unwrap_or([0u8; 8]);
            let mut rng_state: u64 = u64::from_le_bytes(vrf_bytes);

            for selection_round in 0..DIRECTORS_PER_SLOT {
                if weighted.is_empty() {
                    break;
                }

                // Calculate adjusted weights with region boost
                let total_weight: u64 = weighted
                    .iter()
                    .map(|(_, w, region)| {
                        let region_count = selected_regions.iter().filter(|r| *r == region).count();
                        if region_count >= 2 {
                            0 // Exclude if region already has 2
                        } else if region_count == 1 {
                            w.saturating_div(2) // Reduce weight
                        } else {
                            w.saturating_mul(2) // Boost under-represented
                        }
                    })
                    .sum();

                if total_weight == 0 {
                    break;
                }

                // LCG random number generator
                rng_state = rng_state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(selection_round as u64);
                let pick = rng_state % total_weight;

                // Select candidate
                let mut cumulative = 0u64;
                let mut chosen_idx = 0;
                for (i, (_, weight, region)) in weighted.iter().enumerate() {
                    let region_count = selected_regions.iter().filter(|r| *r == region).count();
                    let adjusted_weight = if region_count >= 2 {
                        0
                    } else if region_count == 1 {
                        weight.saturating_div(2)
                    } else {
                        weight.saturating_mul(2)
                    };
                    cumulative = cumulative.saturating_add(adjusted_weight);
                    if pick < cumulative {
                        chosen_idx = i;
                        break;
                    }
                }

                let (account, _, region) = weighted.remove(chosen_idx);
                selected_regions.push(region);
                selected.push(account);
            }

            selected
        }

        /// Finalize a slot after challenge period expires.
        fn finalize_slot(slot: u64) -> DispatchResult {
            let _result = Self::bft_results(slot).ok_or(Error::<T>::ResultNotFound)?;
            let elected = Self::elected_directors(slot);

            // Record positive reputation for agreeing directors
            for director in elected.iter() {
                let _ = pallet_nsn_reputation::Pallet::<T>::record_event(
                    frame_system::RawOrigin::Root.into(),
                    director.clone(),
                    ReputationEventType::DirectorSlotAccepted,
                    slot,
                );
            }

            // Mark as finalized
            FinalizedSlots::<T>::insert(slot, true);
            SlotStatuses::<T>::insert(slot, SlotStatus::Finalized);

            Self::deposit_event(Event::BftConsensusFinalized { slot });
            Ok(())
        }

        /// Handle expired challenge (challenger failed to get resolution in time).
        fn handle_expired_challenge(
            slot: u64,
            challenge: &BftChallenge<T::AccountId, T::Hash>,
        ) -> DispatchResult {
            // Forfeit challenger's bond (griefing penalty)
            <T as pallet::Config>::Currency::burn_held(
                &HoldReason::ChallengeBond.into(),
                &challenge.challenger,
                T::ChallengeBond::get(),
                Precision::Exact,
                Fortitude::Force,
            )?;

            // Mark challenge as resolved
            let mut updated_challenge = challenge.clone();
            updated_challenge.resolved = true;
            PendingChallenges::<T>::insert(slot, updated_challenge);

            // Finalize original result
            Self::finalize_slot(slot)?;

            Ok(())
        }

        /// Integer square root using Newton's method.
        pub fn isqrt(n: u64) -> u64 {
            if n == 0 {
                return 0;
            }
            let mut x = n;
            let mut y = (x + 1) / 2;
            while y < x {
                x = y;
                y = (x + n / x) / 2;
            }
            x
        }

        /// Get elected directors for a specific slot.
        pub fn get_elected_directors(slot: u64) -> Vec<T::AccountId> {
            Self::elected_directors(slot).into_inner()
        }

        /// Check if an account is an elected director for a slot.
        pub fn is_elected_director(slot: u64, account: &T::AccountId) -> bool {
            Self::elected_directors(slot).contains(account)
        }

        /// Get the current slot number.
        pub fn get_current_slot() -> u64 {
            Self::current_slot()
        }

        // =====================================================================
        // Epoch Management Functions
        // =====================================================================

        /// Bootstrap the first epoch when pallet initializes.
        ///
        /// Creates an initial epoch starting at the current block.
        fn bootstrap_first_epoch(now: BlockNumberFor<T>) -> Weight {
            let epoch_duration = T::EpochDuration::get();
            let end_block = now.saturating_add(epoch_duration);

            // Elect initial directors
            let directors_vec = Self::elect_directors_for_epoch(0);
            let directors = match BoundedVec::try_from(directors_vec.clone()) {
                Ok(bounded) => bounded,
                Err(_) => {
                    // Failed to create bounded vec - return minimal weight
                    return T::DbWeight::get().reads(1);
                }
            };

            let epoch = Epoch {
                id: 0,
                start_block: now,
                end_block,
                directors: directors.clone(),
                status: EpochStatus::Active,
            };

            CurrentEpoch::<T>::put(epoch);

            // Mark elected directors as active for the epoch
            let mut mode_weight = Weight::zero();
            for director in directors.iter() {
                mode_weight = mode_weight.saturating_add(T::NodeModeUpdater::set_mode(
                    director,
                    NodeMode::Lane0Active {
                        epoch_end: end_block.saturated_into(),
                    },
                ));
                mode_weight = mode_weight.saturating_add(T::NodeRoleUpdater::set_role(
                    director,
                    NodeRole::ActiveDirector,
                ));
            }

            Self::deposit_event(Event::EpochStarted {
                epoch_id: 0,
                directors: directors_vec,
                end_block,
            });

            // Weight: 1 read (Stakes iteration) + 1 write (CurrentEpoch) + mode updates
            T::DbWeight::get()
                .reads_writes(1, 1)
                .saturating_add(mode_weight)
        }

        /// Run election for next epoch (On-Deck notification).
        ///
        /// Called 20 blocks before epoch transition to give directors time to prepare.
        fn run_epoch_election(_now: BlockNumberFor<T>) -> Weight {
            let current_epoch = match Self::current_epoch() {
                Some(epoch) => epoch,
                None => return T::DbWeight::get().reads(1),
            };

            let next_epoch_id = current_epoch.id.saturating_add(1);

            // Elect directors for next epoch
            let directors_vec = Self::elect_directors_for_epoch(next_epoch_id);
            let directors = match BoundedVec::try_from(directors_vec.clone()) {
                Ok(bounded) => bounded,
                Err(_) => {
                    // Failed to create bounded vec - return
                    return T::DbWeight::get().reads(1);
                }
            };

            NextEpochDirectors::<T>::put(directors);

            // Notify On-Deck directors to drain Lane 1 tasks
            let mut mode_weight = Weight::zero();
            for director in NextEpochDirectors::<T>::get().iter() {
                mode_weight = mode_weight.saturating_add(T::NodeModeUpdater::set_mode(
                    director,
                    NodeMode::Draining {
                        epoch_start: current_epoch.end_block.saturated_into(),
                    },
                ));
                mode_weight = mode_weight
                    .saturating_add(T::NodeRoleUpdater::set_role(director, NodeRole::Reserve));
            }

            Self::deposit_event(Event::OnDeckElection {
                epoch_id: next_epoch_id,
                directors: directors_vec,
                start_block: current_epoch.end_block,
            });

            // Weight: 1 read (Stakes iteration) + 1 write (NextEpochDirectors) + mode updates
            T::DbWeight::get()
                .reads_writes(1, 1)
                .saturating_add(mode_weight)
        }

        /// Transition to next epoch.
        ///
        /// Called when current epoch ends. Moves next epoch directors to active,
        /// updates epoch status, and clears lookahead storage.
        fn transition_epoch(now: BlockNumberFor<T>) -> Weight {
            let mut current_epoch = match Self::current_epoch() {
                Some(epoch) => epoch,
                None => return T::DbWeight::get().reads(1),
            };

            // Mark current epoch as completed
            current_epoch.status = EpochStatus::Completed;
            Self::deposit_event(Event::EpochCompleted {
                epoch_id: current_epoch.id,
            });

            // Get next epoch directors
            let next_directors = NextEpochDirectors::<T>::get();
            if next_directors.is_empty() {
                // No directors elected for next epoch - emergency fallback
                // Re-elect immediately
                let directors_vec =
                    Self::elect_directors_for_epoch(current_epoch.id.saturating_add(1));
                let directors = match BoundedVec::try_from(directors_vec) {
                    Ok(bounded) => bounded,
                    Err(_) => return T::DbWeight::get().reads_writes(2, 1),
                };
                NextEpochDirectors::<T>::put(directors.clone());
            }

            let next_directors = NextEpochDirectors::<T>::get();
            let next_epoch_id = current_epoch.id.saturating_add(1);
            let epoch_duration = T::EpochDuration::get();
            let end_block = now.saturating_add(epoch_duration);

            // Move previous epoch directors back to reserve (Lane 1)
            let mut mode_weight = Weight::zero();
            for director in current_epoch.directors.iter() {
                mode_weight = mode_weight.saturating_add(T::NodeModeUpdater::set_mode(
                    director,
                    NodeMode::Lane1Active,
                ));
                mode_weight = mode_weight
                    .saturating_add(T::NodeRoleUpdater::set_role(director, NodeRole::Reserve));
            }

            // Activate next epoch directors for Lane 0
            for director in next_directors.iter() {
                mode_weight = mode_weight.saturating_add(T::NodeModeUpdater::set_mode(
                    director,
                    NodeMode::Lane0Active {
                        epoch_end: end_block.saturated_into(),
                    },
                ));
                mode_weight = mode_weight.saturating_add(T::NodeRoleUpdater::set_role(
                    director,
                    NodeRole::ActiveDirector,
                ));
            }

            // Create new epoch
            let new_epoch = Epoch {
                id: next_epoch_id,
                start_block: now,
                end_block,
                directors: next_directors.clone(),
                status: EpochStatus::Active,
            };

            CurrentEpoch::<T>::put(new_epoch);
            NextEpochDirectors::<T>::kill(); // Clear lookahead

            Self::deposit_event(Event::EpochStarted {
                epoch_id: next_epoch_id,
                directors: next_directors.into_inner(),
                end_block,
            });

            // Weight: 2 reads (CurrentEpoch, NextEpochDirectors) + 2 writes (CurrentEpoch, NextEpochDirectors) + mode updates
            T::DbWeight::get()
                .reads_writes(2, 2)
                .saturating_add(mode_weight)
        }

        /// Elect directors for a specific epoch.
        ///
        /// Reuses the existing `elect_directors` logic but takes epoch ID instead of slot.
        /// For now, we use epoch_id as a pseudo-slot for the VRF seed.
        fn elect_directors_for_epoch(epoch_id: EpochId) -> Vec<T::AccountId> {
            // Reuse slot-based election logic with epoch_id as seed
            Self::elect_directors(epoch_id)
        }
    }
}
