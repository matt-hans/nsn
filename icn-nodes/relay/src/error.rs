//! Error types for Regional Relay Node

use std::io;
use thiserror::Error;

/// Regional Relay error type
#[derive(Debug, Error)]
pub enum RelayError {
    /// Configuration error
    #[error("Configuration error: {0}")]
    Config(String),

    /// P2P networking error
    #[error("P2P error: {0}")]
    P2P(String),

    /// QUIC transport error
    #[error("QUIC transport error: {0}")]
    QuicTransport(String),

    /// Cache error
    #[error("Cache error: {0}")]
    Cache(String),

    /// Upstream Super-Node error
    #[error("Upstream error: {0}")]
    Upstream(String),

    /// DHT query error
    #[error("DHT query error: {0}")]
    DHTQuery(String),

    /// Latency detection error
    #[error("Latency detection error: {0}")]
    LatencyDetection(String),

    /// I/O error
    #[error("I/O error: {0}")]
    Io(#[from] io::Error),

    /// Serialization error
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    /// Invalid shard request
    #[error("Invalid shard request: {0}")]
    InvalidRequest(String),

    /// Shard not found
    #[error("Shard not found: CID={0}, index={1}")]
    ShardNotFound(String, usize),

    /// Region not detected
    #[error("Region not detected: no reachable Super-Nodes")]
    RegionNotDetected,

    /// Cache eviction failed
    #[error("Cache eviction failed: {0}")]
    CacheEvictionFailed(String),

    /// Metrics error
    #[error("Metrics error: {0}")]
    Metrics(String),

    /// Unauthorized viewer (invalid or missing auth token)
    #[error("Unauthorized: {0}")]
    Unauthorized(String),

    /// Shard hash verification failed
    #[error("Shard hash verification failed: expected {0}, got {1}")]
    ShardHashMismatch(String, String),

    /// Merkle proof verification failed
    #[error("Merkle proof verification failed for shard {0}")]
    MerkleProofVerificationFailed(String),

    /// Invalid Merkle proof format
    #[error("Invalid Merkle proof format: {0}")]
    InvalidMerkleProof(String),

    /// DHT signature verification failed
    #[error("DHT signature verification failed from peer {0}")]
    DhtSignatureVerificationFailed(String),

    /// Missing DHT signature
    #[error("Missing DHT signature in record from peer {0}")]
    MissingDhtSignature(String),

    /// Upstream fetch failed
    #[error("Upstream fetch failed: {0}")]
    UpstreamFetchFailed(String),

    /// Invalid shard data
    #[error("Invalid shard: {0}")]
    InvalidShard(String),
}

/// Result type alias for Regional Relay operations
pub type Result<T> = std::result::Result<T, RelayError>;
