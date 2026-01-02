//! Core types for NSN off-chain nodes
//!
//! This crate provides shared type definitions used across all NSN node components.

use codec::{Decode, Encode};
use serde::{Deserialize, Serialize};

pub use nsn_primitives::Lane;

/// Node capability mode (static role configuration)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Encode, Decode)]
pub enum NodeCapability {
    /// Super-Node: Full capabilities (Director + Validator + Storage)
    SuperNode,
    /// Director-only: Generation capabilities
    DirectorOnly,
    /// Validator-only: CLIP verification
    ValidatorOnly,
    /// Storage-only: Pinning and distribution
    StorageOnly,
}

/// Lane operational mode synced from chain
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Encode, Decode)]
pub enum LaneMode {
    /// Ready for Lane 1 tasks (default reserve mode)
    Lane1Active,
    /// Draining Lane 1 tasks ahead of Lane 0 epoch
    Draining { epoch_start: u32 },
    /// Active Lane 0 director for current epoch
    Lane0Active { epoch_end: u32 },
    /// Offline or maintenance
    Offline,
}

/// Runtime node state
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum NodeState {
    /// Node is starting up
    Starting,
    /// Initializing models (Director/Validator modes)
    LoadingModels,
    /// Ready and waiting for tasks
    Idle,
    /// Actively generating content (Lane 0)
    GeneratingLane0,
    /// Actively generating content (Lane 1)
    GeneratingLane1,
    /// Validating content
    Validating,
    /// Serving pinned content
    Serving,
    /// Shutting down gracefully
    Stopping,
    /// Error state
    Error(String),
}

/// Container execution priority
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ContainerPriority {
    /// Critical models (always resident)
    Critical = 0,
    /// High priority models (preloaded)
    High = 1,
    /// Normal priority (loaded on demand)
    Normal = 2,
    /// Low priority (opportunistic)
    Low = 3,
}

/// Model loading state
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ModelState {
    /// Model not loaded
    Unloaded,
    /// Model currently loading
    Loading,
    /// Model resident in VRAM/memory
    Resident { vram_gb: f32 },
    /// Model unloading
    Unloading,
    /// Model load failed
    Failed(String),
}

/// Task execution status
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Encode, Decode)]
pub enum TaskStatus {
    /// Task queued
    Pending,
    /// Task executing
    Running,
    /// Task completed successfully
    Completed,
    /// Task failed
    Failed(FailureReason),
    /// Task cancelled
    Cancelled,
}

/// Failure reason for tasks
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Encode, Decode)]
pub enum FailureReason {
    /// Model not available
    ModelUnavailable,
    /// Out of VRAM
    OutOfMemory,
    /// Timeout exceeded
    Timeout,
    /// Invalid input
    InvalidInput,
    /// Semantic verification failed
    SemanticCheckFailed,
    /// gRPC communication error
    GrpcError,
    /// Other error
    Other(String),
}

/// VRAM allocation tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VramAllocation {
    /// Model identifier
    pub model_id: String,
    /// Allocated VRAM in GB
    pub allocated_gb: f32,
    /// Priority level
    pub priority: ContainerPriority,
}

/// Epoch/slot information from chain
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Encode, Decode)]
pub struct EpochInfo {
    /// Current epoch number
    pub epoch: u64,
    /// Current slot number within epoch
    pub slot: u64,
    /// Block number
    pub block_number: u64,
    /// Active lane (0 or 1)
    pub active_lane: u8,
}

impl EpochInfo {
    /// Check if this node should be active for the current lane
    pub fn is_lane_active(&self, lane: u8) -> bool {
        self.active_lane == lane
    }
}

/// Recipe specification for content generation
#[derive(Debug, Clone, Serialize, Deserialize, Encode, Decode)]
pub struct Recipe {
    /// Unique recipe identifier
    pub recipe_id: String,
    /// Recipe schema version
    pub version: String,
    /// Slot parameters
    pub slot_params: SlotParams,
    /// Audio track specification
    pub audio_track: AudioTrack,
    /// Visual track specification
    pub visual_track: VisualTrack,
    /// Semantic constraints
    pub semantic_constraints: SemanticConstraints,
    /// Security metadata
    pub security: SecurityMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize, Encode, Decode)]
pub struct SlotParams {
    pub slot_number: u64,
    pub duration_sec: u32,
    pub resolution: String,
    pub fps: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, Encode, Decode)]
pub struct AudioTrack {
    pub script: String,
    pub voice_id: String,
    pub speed: f32,
    pub emotion: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Encode, Decode)]
pub struct VisualTrack {
    pub prompt: String,
    pub negative_prompt: String,
    pub motion_preset: String,
    pub expression_sequence: Vec<String>,
    pub camera_motion: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Encode, Decode)]
pub struct SemanticConstraints {
    pub min_clip_score: f32,
    pub banned_concepts: Vec<String>,
    pub required_concepts: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Encode, Decode)]
pub struct SecurityMetadata {
    pub director_id: String,
    pub ed25519_signature: Vec<u8>,
    pub timestamp: u64,
}

/// Video chunk header for Lane 0 distribution.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Encode, Decode)]
pub struct VideoChunkHeader {
    /// Schema version for the chunk format.
    pub version: u16,
    /// Slot number for the video.
    pub slot: u64,
    /// Content identifier for the full video (IPFS CID or other).
    pub content_id: String,
    /// Zero-based chunk index.
    pub chunk_index: u32,
    /// Total number of chunks in the stream.
    pub total_chunks: u32,
    /// Timestamp when this chunk was published (ms since Unix epoch).
    pub timestamp_ms: u64,
    /// Whether this chunk represents a keyframe boundary.
    pub is_keyframe: bool,
    /// Blake3 hash of the chunk payload.
    pub payload_hash: [u8; 32],
}

/// Signed video chunk message for GossipSub distribution.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Encode, Decode)]
pub struct VideoChunk {
    /// Header metadata.
    pub header: VideoChunkHeader,
    /// Raw chunk payload bytes.
    pub payload: Vec<u8>,
    /// Signer public key (libp2p protobuf-encoded bytes).
    pub signer: Vec<u8>,
    /// Signature over the header + payload hash.
    pub signature: Vec<u8>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_epoch_info_lane_active() {
        let epoch = EpochInfo {
            epoch: 1,
            slot: 5,
            block_number: 100,
            active_lane: 0,
        };

        assert!(epoch.is_lane_active(0));
        assert!(!epoch.is_lane_active(1));
    }

    #[test]
    fn test_container_priority_ordering() {
        assert!(ContainerPriority::Critical < ContainerPriority::High);
        assert!(ContainerPriority::High < ContainerPriority::Normal);
        assert!(ContainerPriority::Normal < ContainerPriority::Low);
    }
}
