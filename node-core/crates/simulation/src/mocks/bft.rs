//! Mock BFT participant for testing consensus without real network.

use std::collections::{HashSet, VecDeque};
use std::time::Duration;

use nsn_lane0::{BftConsensusResult, BftError, BftResult};

/// Byzantine behavior modes for BFT testing.
#[derive(Debug, Clone, PartialEq)]
pub enum ByzantineMode {
    /// Honest node behavior
    Honest,
    /// Return divergent embeddings
    DivergentEmbedding(Vec<f32>),
    /// Delay all responses
    Delay(Duration),
    /// Drop all messages (crash fault)
    Crash,
    /// Return invalid signatures
    InvalidSignature,
}

impl Default for ByzantineMode {
    fn default() -> Self {
        Self::Honest
    }
}

/// Mock BFT participant for simulation testing.
///
/// Provides configurable consensus behavior including Byzantine fault injection.
///
/// # Example
///
/// ```rust,ignore
/// use nsn_simulation::mocks::MockBftParticipant;
///
/// let mut bft = MockBftParticipant::new()
///     .with_consensus(&[1, 2, 3]);
///
/// let result = bft.run_consensus(1, embedding, 5000).await?;
/// assert!(result.success);
/// ```
#[derive(Debug, Clone)]
pub struct MockBftParticipant {
    /// Slots that will reach consensus
    consensus_slots: HashSet<u64>,
    /// Consensus results (for verification)
    pub results: VecDeque<BftConsensusResult>,
    /// Byzantine mode
    byzantine_mode: ByzantineMode,
    /// Custom similarity threshold
    similarity_threshold: f32,
    /// Custom vote count for success
    vote_threshold: usize,
    /// Total directors expected
    total_directors: usize,
    /// Simulated latency
    latency: Option<Duration>,
}

impl Default for MockBftParticipant {
    fn default() -> Self {
        Self::new()
    }
}

impl MockBftParticipant {
    /// Create a new mock BFT participant.
    pub fn new() -> Self {
        Self {
            consensus_slots: HashSet::new(),
            results: VecDeque::new(),
            byzantine_mode: ByzantineMode::default(),
            similarity_threshold: 0.85,
            vote_threshold: 3,
            total_directors: 5,
            latency: None,
        }
    }

    /// Configure slots that will reach consensus.
    pub fn with_consensus(mut self, slots: &[u64]) -> Self {
        self.consensus_slots.extend(slots);
        self
    }

    /// Configure Byzantine behavior.
    pub fn with_byzantine_mode(mut self, mode: ByzantineMode) -> Self {
        self.byzantine_mode = mode;
        self
    }

    /// Configure similarity threshold.
    pub fn with_similarity_threshold(mut self, threshold: f32) -> Self {
        self.similarity_threshold = threshold;
        self
    }

    /// Configure vote threshold (e.g., 3 of 5).
    pub fn with_vote_threshold(mut self, votes: usize, total: usize) -> Self {
        self.vote_threshold = votes;
        self.total_directors = total;
        self
    }

    /// Configure simulated latency.
    pub fn with_latency(mut self, latency: Duration) -> Self {
        self.latency = Some(latency);
        self
    }

    /// Add a consensus slot dynamically.
    pub fn add_consensus_slot(&mut self, slot: u64) {
        self.consensus_slots.insert(slot);
    }

    /// Remove a consensus slot.
    pub fn remove_consensus_slot(&mut self, slot: u64) {
        self.consensus_slots.remove(&slot);
    }

    /// Get the number of consensus results.
    pub fn consensus_count(&self) -> usize {
        self.results.len()
    }

    /// Clear consensus results.
    pub fn clear_results(&mut self) {
        self.results.clear();
    }

    /// Run BFT consensus for a slot.
    pub async fn run_consensus(
        &mut self,
        slot: u64,
        embedding: Vec<f32>,
        timeout_ms: u64,
    ) -> BftResult<BftConsensusResult> {
        // Handle Byzantine modes
        match &self.byzantine_mode {
            ByzantineMode::Crash => {
                // Simulate crash - never respond
                tokio::time::sleep(Duration::from_millis(timeout_ms + 1000)).await;
                return Err(BftError::Timeout {
                    timeout_ms,
                    collected: 0,
                    expected: 5,
                });
            }
            ByzantineMode::Delay(d) => {
                tokio::time::sleep(*d).await;
            }
            _ => {
                if let Some(latency) = self.latency {
                    tokio::time::sleep(latency).await;
                }
            }
        }

        let success = self.consensus_slots.contains(&slot);
        let similarity: f32 = if success {
            0.95
        } else {
            // Below threshold - consensus fails
            self.similarity_threshold - 0.20
        };

        // Modify embedding for Byzantine divergent mode
        let _final_embedding = match &self.byzantine_mode {
            ByzantineMode::DivergentEmbedding(divergent) => divergent.clone(),
            _ => embedding,
        };

        let result = BftConsensusResult {
            slot,
            canonical_hash: [0u8; 32],
            signers: vec![],
            success,
            similarity,
        };

        if !success {
            return Err(BftError::ConsensusFailed {
                slot,
                similarity,
                threshold: self.similarity_threshold,
            });
        }

        self.results.push_back(result.clone());
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_consensus_success() {
        let mut bft = MockBftParticipant::new().with_consensus(&[1, 2, 3]);

        let embedding = vec![0.5f32; 512];
        let result = bft.run_consensus(1, embedding, 5000).await;

        assert!(result.is_ok());
        let consensus = result.unwrap();
        assert!(consensus.success);
        assert_eq!(consensus.slot, 1);
    }

    #[tokio::test]
    async fn test_consensus_failure() {
        let mut bft = MockBftParticipant::new().with_consensus(&[1]);

        let embedding = vec![0.5f32; 512];
        let result = bft.run_consensus(99, embedding, 5000).await;

        assert!(result.is_err());
        match result {
            Err(BftError::ConsensusFailed { slot, .. }) => assert_eq!(slot, 99),
            _ => panic!("Expected ConsensusFailed"),
        }
    }

    #[tokio::test]
    async fn test_consensus_tracking() {
        let mut bft = MockBftParticipant::new().with_consensus(&[1, 2, 3]);

        for slot in 1..=3 {
            let embedding = vec![0.5f32; 512];
            bft.run_consensus(slot, embedding, 5000).await.unwrap();
        }

        assert_eq!(bft.consensus_count(), 3);
    }

    #[tokio::test]
    async fn test_byzantine_crash() {
        // Use tokio::time::pause() for deterministic timing
        tokio::time::pause();

        let mut bft =
            MockBftParticipant::new().with_byzantine_mode(ByzantineMode::Crash);

        let embedding = vec![0.5f32; 512];

        // The crash mode should timeout
        let result = tokio::time::timeout(
            Duration::from_millis(100),
            bft.run_consensus(1, embedding, 50),
        )
        .await;

        assert!(result.is_err()); // Should timeout
    }
}
