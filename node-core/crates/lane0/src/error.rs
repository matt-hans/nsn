//! Error types for Lane 0 orchestration.
//!
//! Provides comprehensive error handling for director lifecycle, slot generation,
//! BFT consensus, and video publishing operations.

use thiserror::Error;

/// Top-level error type for Lane 0 operations.
#[derive(Debug, Error)]
pub enum Lane0Error {
    /// Director lifecycle errors.
    #[error("director error: {0}")]
    Director(#[from] DirectorError),

    /// Recipe processing errors.
    #[error("recipe error: {0}")]
    Recipe(#[from] RecipeError),

    /// Vortex client errors.
    #[error("vortex error: {0}")]
    Vortex(#[from] VortexError),

    /// BFT consensus errors.
    #[error("BFT error: {0}")]
    Bft(#[from] BftError),

    /// Video publishing errors.
    #[error("publish error: {0}")]
    Publish(#[from] PublishError),

    /// Slot generation pipeline errors.
    #[error("slot error: {0}")]
    Slot(#[from] SlotError),
}

/// Errors during director lifecycle management.
#[derive(Debug, Error)]
pub enum DirectorError {
    /// Invalid state transition attempted.
    #[error("invalid state transition from {from:?} to {to:?}")]
    InvalidTransition { from: String, to: String },

    /// Epoch notification handling failed.
    #[error("epoch notification failed: {0}")]
    EpochNotification(String),

    /// Director not elected for current epoch.
    #[error("not elected as director for epoch {epoch}")]
    NotElected { epoch: u64 },

    /// Channel communication failed.
    #[error("channel error: {0}")]
    Channel(String),

    /// Configuration error.
    #[error("configuration error: {0}")]
    Config(String),
}

/// Errors during recipe processing.
#[derive(Debug, Error)]
pub enum RecipeError {
    /// Recipe deserialization failed.
    #[error("recipe deserialization failed: {0}")]
    Deserialization(String),

    /// Recipe validation failed.
    #[error("recipe validation failed: {0}")]
    Validation(String),

    /// Missing required field.
    #[error("missing required field: {field}")]
    MissingField { field: String },

    /// Invalid slot parameters.
    #[error("invalid slot parameters: {0}")]
    InvalidSlotParams(String),

    /// Recipe queue full.
    #[error("recipe queue full, capacity: {capacity}")]
    QueueFull { capacity: usize },

    /// Duplicate recipe for same slot.
    #[error("duplicate recipe for slot {slot}")]
    Duplicate { slot: u64 },

    /// Recipe expired (slot already passed).
    #[error("recipe expired for slot {slot}")]
    Expired { slot: u64 },

    /// P2P subscription error.
    #[error("P2P subscription error: {0}")]
    Subscription(String),
}

/// Errors during Vortex pipeline execution.
#[derive(Debug, Error)]
pub enum VortexError {
    /// Sidecar gRPC connection failed.
    #[error("sidecar connection failed: {0}")]
    Connection(String),

    /// Task execution failed.
    #[error("task execution failed: {0}")]
    Execution(String),

    /// Task timed out.
    #[error("task timed out after {timeout_ms}ms")]
    Timeout { timeout_ms: u64 },

    /// Response parsing failed.
    #[error("response parsing failed: {0}")]
    ResponseParse(String),

    /// Model not loaded.
    #[error("model not loaded: {model}")]
    ModelNotLoaded { model: String },

    /// Generation produced invalid output.
    #[error("invalid generation output: {0}")]
    InvalidOutput(String),
}

/// Errors during BFT consensus.
#[derive(Debug, Error)]
pub enum BftError {
    /// Not enough embeddings collected for consensus.
    #[error("not enough embeddings: got {got}, need {need}")]
    InsufficientEmbeddings { got: usize, need: usize },

    /// Consensus failed (embeddings too dissimilar).
    #[error("consensus failed for slot {slot}: similarity {similarity:.4} below threshold {threshold:.4}")]
    ConsensusFailed {
        slot: u64,
        similarity: f32,
        threshold: f32,
    },

    /// Timeout waiting for other directors.
    #[error("BFT timeout after {timeout_ms}ms, collected {collected} of {expected} embeddings")]
    Timeout {
        timeout_ms: u64,
        collected: usize,
        expected: usize,
    },

    /// P2P publish failed.
    #[error("failed to publish embedding: {0}")]
    PublishFailed(String),

    /// Invalid embedding received.
    #[error("invalid embedding from {peer}: {reason}")]
    InvalidEmbedding { peer: String, reason: String },

    /// Signature verification failed.
    #[error("signature verification failed for {peer}")]
    InvalidSignature { peer: String },
}

/// Errors during video chunk publishing.
#[derive(Debug, Error)]
pub enum PublishError {
    /// Video chunking failed.
    #[error("chunking failed: {0}")]
    ChunkingFailed(String),

    /// Chunk signing failed.
    #[error("signing failed: {0}")]
    SigningFailed(String),

    /// P2P publish failed.
    #[error("P2P publish failed: {0}")]
    P2pFailed(String),

    /// Empty video data.
    #[error("empty video data")]
    EmptyVideo,

    /// Invalid content ID.
    #[error("invalid content ID: {0}")]
    InvalidContentId(String),
}

/// Errors during complete slot generation pipeline.
#[derive(Debug, Error)]
pub enum SlotError {
    /// Vortex generation failed.
    #[error("generation failed: {0}")]
    Generation(#[from] VortexError),

    /// BFT consensus failed.
    #[error("BFT consensus failed: {0}")]
    Consensus(#[from] BftError),

    /// Video publishing failed.
    #[error("publishing failed: {0}")]
    Publishing(#[from] PublishError),

    /// Chain submission failed.
    #[error("chain submission failed: {0}")]
    ChainSubmission(String),

    /// Slot already processed.
    #[error("slot {slot} already processed")]
    AlreadyProcessed { slot: u64 },

    /// Recipe not found for slot.
    #[error("no recipe for slot {slot}")]
    NoRecipe { slot: u64 },
}

/// Result type alias for Lane 0 operations.
pub type Lane0Result<T> = Result<T, Lane0Error>;

/// Result type alias for director operations.
pub type DirectorResult<T> = Result<T, DirectorError>;

/// Result type alias for recipe operations.
pub type RecipeResult<T> = Result<T, RecipeError>;

/// Result type alias for Vortex operations.
pub type VortexResult<T> = Result<T, VortexError>;

/// Result type alias for BFT operations.
pub type BftResult<T> = Result<T, BftError>;

/// Result type alias for publish operations.
pub type PublishResult<T> = Result<T, PublishError>;

/// Result type alias for slot operations.
pub type SlotResult<T> = Result<T, SlotError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = DirectorError::NotElected { epoch: 42 };
        assert!(err.to_string().contains("42"));

        let err = BftError::ConsensusFailed {
            slot: 100,
            similarity: 0.75,
            threshold: 0.85,
        };
        assert!(err.to_string().contains("0.75"));
        assert!(err.to_string().contains("0.85"));
    }

    #[test]
    fn test_error_conversion() {
        let vortex_err = VortexError::Timeout { timeout_ms: 5000 };
        let slot_err: SlotError = vortex_err.into();
        assert!(matches!(slot_err, SlotError::Generation(_)));

        let bft_err = BftError::Timeout {
            timeout_ms: 5000,
            collected: 2,
            expected: 3,
        };
        let slot_err: SlotError = bft_err.into();
        assert!(matches!(slot_err, SlotError::Consensus(_)));
    }

    #[test]
    fn test_lane0_error_from() {
        let dir_err = DirectorError::NotElected { epoch: 1 };
        let lane0_err: Lane0Error = dir_err.into();
        assert!(matches!(lane0_err, Lane0Error::Director(_)));
    }
}
