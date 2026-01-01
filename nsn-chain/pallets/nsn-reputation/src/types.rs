// Copyright 2024 Neural Sovereign Network
// This file is part of NSN Chain.
//
// NSN Chain is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

//! Types for the NSN Reputation pallet.

use parity_scale_codec::{Decode, DecodeWithMemTracking, Encode, MaxEncodedLen};
use scale_info::TypeInfo;
use sp_runtime::RuntimeDebug;

/// Reputation event types with associated score deltas.
///
/// Each event type affects one of the three score components:
/// - Director score: Slot acceptance/rejection/missed
/// - Validator score: Vote correctness
/// - Seeder score: Chunk serving, audit results
///
/// # Deltas
/// - DirectorSlotAccepted: +100 director
/// - DirectorSlotRejected: -200 director
/// - DirectorSlotMissed: -150 director
/// - ValidatorVoteCorrect: +5 validator
/// - ValidatorVoteIncorrect: -10 validator
/// - SeederChunkServed: +1 seeder
/// - PinningAuditPassed: +10 seeder
/// - PinningAuditFailed: -50 seeder
/// - TaskCompleted: +5 seeder
/// - TaskFailed: -10 seeder
#[derive(
    Encode,
    Decode,
    DecodeWithMemTracking,
    Clone,
    PartialEq,
    Eq,
    RuntimeDebug,
    TypeInfo,
    MaxEncodedLen,
)]
pub enum ReputationEventType {
    /// Director slot successfully completed
    DirectorSlotAccepted,
    /// Director slot rejected (content quality failure)
    DirectorSlotRejected,
    /// Director missed slot (timeout/failure)
    DirectorSlotMissed,
    /// Validator voted correctly (matched BFT consensus)
    ValidatorVoteCorrect,
    /// Validator voted incorrectly (disagreed with consensus)
    ValidatorVoteIncorrect,
    /// Seeder served a chunk to viewer
    SeederChunkServed,
    /// Pinning audit passed (shard available)
    PinningAuditPassed,
    /// Pinning audit failed (shard missing/corrupt)
    PinningAuditFailed,
    /// Task completed successfully (Lane 1 compute)
    TaskCompleted,
    /// Task failed or abandoned (Lane 1 compute)
    TaskFailed,
}

impl ReputationEventType {
    /// Get the score delta for this event type.
    ///
    /// # Returns
    /// Positive deltas for positive events, negative for penalties.
    ///
    /// # Values
    /// Matches PRD specification exactly.
    pub fn delta(&self) -> i64 {
        match self {
            ReputationEventType::DirectorSlotAccepted => 100,
            ReputationEventType::DirectorSlotRejected => -200,
            ReputationEventType::DirectorSlotMissed => -150,
            ReputationEventType::ValidatorVoteCorrect => 5,
            ReputationEventType::ValidatorVoteIncorrect => -10,
            ReputationEventType::SeederChunkServed => 1,
            ReputationEventType::PinningAuditPassed => 10,
            ReputationEventType::PinningAuditFailed => -50,
            ReputationEventType::TaskCompleted => 5,
            ReputationEventType::TaskFailed => -10,
        }
    }

    /// Check if this is a director-related event.
    pub fn is_director_event(&self) -> bool {
        matches!(
            self,
            Self::DirectorSlotAccepted | Self::DirectorSlotRejected | Self::DirectorSlotMissed
        )
    }

    /// Check if this is a validator-related event.
    pub fn is_validator_event(&self) -> bool {
        matches!(
            self,
            Self::ValidatorVoteCorrect | Self::ValidatorVoteIncorrect
        )
    }

    /// Check if this is a seeder-related event.
    pub fn is_seeder_event(&self) -> bool {
        matches!(
            self,
            Self::SeederChunkServed
                | Self::PinningAuditPassed
                | Self::PinningAuditFailed
                | Self::TaskCompleted
                | Self::TaskFailed
        )
    }
}

/// Reputation score for an account.
///
/// Tracks three independent score components with a weighted total:
/// - 50% director score (content generation quality)
/// - 30% validator score (verification accuracy)
/// - 20% seeder score (infrastructure reliability)
///
/// Scores never go below zero (saturating arithmetic).
/// Decay applied weekly based on inactivity.
///
/// # Example
/// ```text
/// let score = ReputationScore {
///     director_score: 200,
///     validator_score: 5,
///     seeder_score: 1,
///     last_activity: 1000,
/// };
/// // total() = (200*50 + 5*30 + 1*20) / 100 = 101
/// ```
#[derive(
    Encode,
    Decode,
    DecodeWithMemTracking,
    Clone,
    PartialEq,
    Eq,
    RuntimeDebug,
    TypeInfo,
    Default,
    MaxEncodedLen,
)]
pub struct ReputationScore {
    /// Director-specific score (slot acceptance/rejection/missed)
    pub director_score: u64,
    /// Validator-specific score (vote correctness)
    pub validator_score: u64,
    /// Seeder-specific score (chunk serving, audits)
    pub seeder_score: u64,
    /// Block number of last activity (for decay calculation)
    pub last_activity: u64,
}

impl ReputationScore {
    /// Calculate weighted total reputation score.
    ///
    /// # Formula
    /// (director_score * 50 + validator_score * 30 + seeder_score * 20) / 100
    ///
    /// # Returns
    /// Weighted total in range [0, ∞). Used for director election probability.
    ///
    /// # Example
    /// If director=200, validator=5, seeder=1:
    /// total = (200*50 + 5*30 + 1*20) / 100 = 10170 / 100 = 101
    pub fn total(&self) -> u64 {
        // L2: Saturating arithmetic to prevent overflow
        let director_weighted = self.director_score.saturating_mul(50);
        let validator_weighted = self.validator_score.saturating_mul(30);
        let seeder_weighted = self.seeder_score.saturating_mul(20);

        director_weighted
            .saturating_add(validator_weighted)
            .saturating_add(seeder_weighted)
            .saturating_div(100)
    }

    /// Apply decay to inactive accounts.
    ///
    /// # Decay Formula
    /// weeks_inactive = (current_block - last_activity) / (7 * 24 * 600)
    /// decay_factor = max(0, 100 - decay_rate * weeks_inactive)
    /// new_score = old_score * decay_factor / 100
    ///
    /// # Arguments
    /// * `current_block` - Current block number
    /// * `decay_rate` - Decay rate per week in percent (default: 5)
    ///
    /// # Example
    /// If last_activity was 12 weeks ago with 5% decay:
    /// decay_factor = 100 - (5 * 12) = 40%
    /// All scores multiplied by 0.40
    pub fn apply_decay(&mut self, current_block: u64, decay_rate: u64) {
        // Assume ~600 blocks/hour, 24 hours/day, 7 days/week = 100,800 blocks/week
        const BLOCKS_PER_WEEK: u64 = 7 * 24 * 600;

        let blocks_inactive = current_block.saturating_sub(self.last_activity);
        let weeks_inactive = blocks_inactive / BLOCKS_PER_WEEK;

        if weeks_inactive > 0 {
            // Calculate decay factor (e.g., 5% * 12 weeks = 60% decay → 40% remaining)
            let decay_total = decay_rate.saturating_mul(weeks_inactive);
            let decay_factor = 100u64.saturating_sub(decay_total);

            // Apply decay to all components (saturating at 0)
            self.director_score = self
                .director_score
                .saturating_mul(decay_factor)
                .saturating_div(100);
            self.validator_score = self
                .validator_score
                .saturating_mul(decay_factor)
                .saturating_div(100);
            self.seeder_score = self
                .seeder_score
                .saturating_mul(decay_factor)
                .saturating_div(100);
            self.last_activity = current_block;
        }
    }

    /// Apply a delta to a specific score component with floor at zero.
    ///
    /// # Arguments
    /// * `delta` - Signed delta to apply (positive or negative)
    /// * `component` - Which component to update (0=director, 1=validator, 2=seeder)
    ///
    /// # Behavior
    /// Uses saturating arithmetic to prevent underflow/overflow.
    /// Negative deltas floor at zero (no negative scores).
    pub fn apply_delta(&mut self, delta: i64, component: u8) {
        match component {
            0 => self.director_score = self.director_score.saturating_add_signed(delta),
            1 => self.validator_score = self.validator_score.saturating_add_signed(delta),
            2 => self.seeder_score = self.seeder_score.saturating_add_signed(delta),
            _ => (),
        }
    }

    /// Update last activity to current block.
    pub fn update_activity(&mut self, current_block: u64) {
        self.last_activity = current_block;
    }
}

/// A reputation event recorded in a block.
///
/// These events are batched into Merkle trees for efficient off-chain
/// verification. Each event represents a single reputation change.
///
/// # Merkle Leaf
/// The hash of this struct (using T::Hashing) becomes a leaf in the
/// Merkle tree for that block.
#[derive(
    Encode,
    Decode,
    DecodeWithMemTracking,
    Clone,
    PartialEq,
    Eq,
    RuntimeDebug,
    TypeInfo,
    MaxEncodedLen,
)]
#[scale_info(skip_type_params(AccountId, BlockNumber))]
pub struct ReputationEvent<AccountId, BlockNumber> {
    /// Account affected by this event
    pub account: AccountId,
    /// Type of event (determines delta)
    pub event_type: ReputationEventType,
    /// Slot number (for director events)
    pub slot: u64,
    /// Block number when event occurred
    pub block: BlockNumber,
}

/// Checkpoint data created every 1000 blocks.
///
/// Checkpoints provide a snapshot of all reputation scores at a specific
/// block. Used for efficient proof generation and recovery.
///
/// # Storage
/// Stored in `Checkpoints` storage map at block intervals.
///
/// # Merkle Root
/// Computed over all (account, score) pairs at this block.
#[derive(
    Encode,
    Decode,
    DecodeWithMemTracking,
    Clone,
    PartialEq,
    Eq,
    RuntimeDebug,
    TypeInfo,
    MaxEncodedLen,
)]
pub struct CheckpointData<Hash, BlockNumber> {
    /// Block number of this checkpoint
    pub block: BlockNumber,
    /// Number of accounts with reputation at this block
    pub score_count: u32,
    /// Merkle root of all reputation scores at this block
    pub merkle_root: Hash,
}

/// Aggregated event item for batched submissions.
///
/// Represents a single reputation event in a batch for one account.
#[derive(
    Encode,
    Decode,
    DecodeWithMemTracking,
    Clone,
    PartialEq,
    Eq,
    RuntimeDebug,
    TypeInfo,
    MaxEncodedLen,
)]
pub struct AggregatedEvent {
    /// Type of event (determines delta)
    pub event_type: ReputationEventType,
    /// Slot number (for director events)
    pub slot: u64,
}

/// Aggregated reputation events for TPS optimization.
///
/// Instead of submitting every individual event, off-chain aggregators
/// can batch multiple events for the same account into a single transaction.
///
/// # Example
/// If Alice has 4 events in a block:
/// - DirectorSlotAccepted (+100)
/// - DirectorSlotAccepted (+100)
/// - DirectorSlotRejected (-200)
/// - ValidatorVoteCorrect (+5)
///
/// The aggregated event would have:
/// - net_director_delta = 0
/// - net_validator_delta = 5
/// - net_seeder_delta = 0
/// - event_count = 4
#[derive(
    Encode,
    Decode,
    DecodeWithMemTracking,
    Clone,
    PartialEq,
    Eq,
    RuntimeDebug,
    TypeInfo,
    Default,
    MaxEncodedLen,
)]
pub struct AggregatedReputation {
    /// Net director score change (sum of all director event deltas)
    pub net_director_delta: i64,
    /// Net validator score change (sum of all validator event deltas)
    pub net_validator_delta: i64,
    /// Net seeder score change (sum of all seeder event deltas)
    pub net_seeder_delta: i64,
    /// Number of events aggregated (for tracking/rewards)
    pub event_count: u32,
    /// Block number of last aggregation
    pub last_aggregation_block: u64,
}

impl AggregatedReputation {
    /// Create a new empty aggregation.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add an event delta to this aggregation.
    ///
    /// # Arguments
    /// * `event_type` - Type of event to add
    pub fn add_event(&mut self, event_type: &ReputationEventType) {
        let delta = event_type.delta();

        if event_type.is_director_event() {
            self.net_director_delta = self.net_director_delta.saturating_add(delta);
        } else if event_type.is_validator_event() {
            self.net_validator_delta = self.net_validator_delta.saturating_add(delta);
        } else if event_type.is_seeder_event() {
            self.net_seeder_delta = self.net_seeder_delta.saturating_add(delta);
        }

        self.event_count = self.event_count.saturating_add(1);
    }

    /// Check if this aggregation is empty (no events).
    pub fn is_empty(&self) -> bool {
        self.event_count == 0
    }

    /// Reset aggregation to empty state.
    pub fn clear(&mut self) {
        *self = Self::default();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_event_deltas() {
        assert_eq!(ReputationEventType::DirectorSlotAccepted.delta(), 100);
        assert_eq!(ReputationEventType::DirectorSlotRejected.delta(), -200);
        assert_eq!(ReputationEventType::DirectorSlotMissed.delta(), -150);
        assert_eq!(ReputationEventType::ValidatorVoteCorrect.delta(), 5);
        assert_eq!(ReputationEventType::ValidatorVoteIncorrect.delta(), -10);
        assert_eq!(ReputationEventType::SeederChunkServed.delta(), 1);
        assert_eq!(ReputationEventType::PinningAuditPassed.delta(), 10);
        assert_eq!(ReputationEventType::PinningAuditFailed.delta(), -50);
        assert_eq!(ReputationEventType::TaskCompleted.delta(), 5);
        assert_eq!(ReputationEventType::TaskFailed.delta(), -10);
    }

    #[test]
    fn test_reputation_total() {
        let score = ReputationScore {
            director_score: 200,
            validator_score: 5,
            seeder_score: 1,
            last_activity: 1000,
        };
        // (200*50 + 5*30 + 1*20) / 100 = 10170 / 100 = 101
        assert_eq!(score.total(), 101);
    }

    #[test]
    fn test_apply_decay() {
        let mut score = ReputationScore {
            director_score: 1000,
            validator_score: 500,
            seeder_score: 100,
            last_activity: 10000,
        };

        // 12 weeks later, 5% decay per week
        // decay_factor = 100 - (5 * 12) = 40%
        let current_block = 10000 + (12 * 7 * 24 * 600);
        score.apply_decay(current_block, 5);

        assert_eq!(score.director_score, 400);
        assert_eq!(score.validator_score, 200);
        assert_eq!(score.seeder_score, 40);
    }

    #[test]
    fn test_apply_delta_floor() {
        let mut score = ReputationScore {
            director_score: 50,
            validator_score: 10,
            seeder_score: 5,
            last_activity: 1000,
        };

        // Apply -200 director delta (more than current score)
        score.apply_delta(-200, 0);

        // Should floor at 0, not underflow
        assert_eq!(score.director_score, 0);
        assert_eq!(score.validator_score, 10);
        assert_eq!(score.seeder_score, 5);
    }

    #[test]
    fn test_aggregated_reputation() {
        let mut agg = AggregatedReputation::new();

        agg.add_event(&ReputationEventType::DirectorSlotAccepted);
        agg.add_event(&ReputationEventType::DirectorSlotAccepted);
        agg.add_event(&ReputationEventType::DirectorSlotRejected);
        agg.add_event(&ReputationEventType::ValidatorVoteCorrect);

        assert_eq!(agg.net_director_delta, 0); // 100 + 100 - 200
        assert_eq!(agg.net_validator_delta, 5);
        assert_eq!(agg.net_seeder_delta, 0);
        assert_eq!(agg.event_count, 4);
        assert!(!agg.is_empty());
    }
}
