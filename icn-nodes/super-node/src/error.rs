//! Error types for Super-Node operations

use thiserror::Error;

/// Super-Node error types
#[derive(Error, Debug)]
pub enum SuperNodeError {
    /// Configuration errors
    #[error("Configuration error: {0}")]
    Config(String),

    /// Storage layer errors
    #[error("Storage error: {0}")]
    Storage(String),

    /// Erasure coding errors
    #[error("Erasure coding error: {0}")]
    ErasureCoding(String),

    /// P2P network errors
    #[error("P2P error: {0}")]
    P2P(String),

    /// QUIC transport errors
    #[error("QUIC transport error: {0}")]
    QuicTransport(String),

    /// Chain client errors
    #[error("Chain client error: {0}")]
    ChainClient(String),

    /// Audit errors
    #[error("Audit error: {0}")]
    Audit(String),

    /// I/O errors
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Serialization errors
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    /// Generic errors
    #[error("Internal error: {0}")]
    Internal(String),
}

/// Result type for Super-Node operations
pub type Result<T> = std::result::Result<T, SuperNodeError>;
