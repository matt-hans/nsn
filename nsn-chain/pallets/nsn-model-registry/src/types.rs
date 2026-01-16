// Copyright 2024 Neural Sovereign Network
// This file is part of NSN Chain.

//! Types for pallet-nsn-model-registry
//!
//! This module defines the core types used by the model registry pallet:
//! - ModelCapabilities: Bitfield representing what a model can do
//! - ModelMetadata: Full metadata for a registered model
//! - ModelState: Whether a model is hot/warm/cold on a node
//! - NodeCapabilityAd: A node's advertisement of its capabilities

use frame_support::pallet_prelude::*;
use parity_scale_codec::{Decode, DecodeWithMemTracking, Encode};
use scale_info::TypeInfo;
use sp_runtime::RuntimeDebug;

/// Capabilities bitfield for AI models
///
/// Represents what a model can do in the NSN dual-lane architecture:
/// - Lane 0: Video generation (video_generation)
/// - Lane 1: General AI compute (text_generation, code_generation, etc.)
#[derive(
    Clone,
    Copy,
    Encode,
    Decode,
    DecodeWithMemTracking,
    Eq,
    PartialEq,
    RuntimeDebug,
    TypeInfo,
    MaxEncodedLen,
    Default,
)]
pub struct ModelCapabilities {
    /// Text generation capability (LLMs, chat models)
    pub text_generation: bool,
    /// Image generation capability (Flux, Stable Diffusion, etc.)
    pub image_generation: bool,
    /// Code generation capability (Codex, StarCoder, etc.)
    pub code_generation: bool,
    /// Embedding generation capability (CLIP, sentence transformers)
    pub embedding: bool,
    /// Speech synthesis capability (TTS models like Kokoro)
    pub speech_synthesis: bool,
    /// Video generation capability (Lane 0 - LivePortrait, video diffusion)
    pub video_generation: bool,
}

impl ModelCapabilities {
    /// Create a new ModelCapabilities with all flags set to false
    pub fn new() -> Self {
        Self::default()
    }

    /// Create capabilities for a Lane 0 video generation model
    pub fn video_model() -> Self {
        Self {
            video_generation: true,
            ..Default::default()
        }
    }

    /// Create capabilities for a text generation model
    pub fn text_model() -> Self {
        Self {
            text_generation: true,
            ..Default::default()
        }
    }

    /// Check if any capability is set
    pub fn has_any(&self) -> bool {
        self.text_generation
            || self.image_generation
            || self.code_generation
            || self.embedding
            || self.speech_synthesis
            || self.video_generation
    }

    /// Check if this is a Lane 0 model (video generation)
    pub fn is_lane0(&self) -> bool {
        self.video_generation
    }
}

/// Metadata for a registered AI model
///
/// Contains all information about a model in the catalog, including
/// its container CID, VRAM requirements, and capabilities.
#[derive(Clone, Encode, Decode, DecodeWithMemTracking, Eq, PartialEq, RuntimeDebug, TypeInfo)]
#[scale_info(skip_type_params(MaxCidLen))]
pub struct ModelMetadata<AccountId, BlockNumber, MaxCidLen>
where
    MaxCidLen: Get<u32>,
{
    /// Content identifier for the model container (Docker/OCI image)
    pub container_cid: BoundedVec<u8, MaxCidLen>,
    /// VRAM required to load this model (in MB)
    pub vram_required_mb: u32,
    /// Model capabilities bitfield
    pub capabilities: ModelCapabilities,
    /// Account that registered this model
    pub registered_by: AccountId,
    /// Block number when the model was registered
    pub registered_at: BlockNumber,
}

impl<AccountId: Default, BlockNumber: Default, MaxCidLen: Get<u32>> Default
    for ModelMetadata<AccountId, BlockNumber, MaxCidLen>
{
    fn default() -> Self {
        Self {
            container_cid: BoundedVec::default(),
            vram_required_mb: 0,
            capabilities: ModelCapabilities::default(),
            registered_by: AccountId::default(),
            registered_at: BlockNumber::default(),
        }
    }
}

// Manual MaxEncodedLen implementation for ModelMetadata
impl<AccountId: MaxEncodedLen, BlockNumber: MaxEncodedLen, MaxCidLen: Get<u32>> MaxEncodedLen
    for ModelMetadata<AccountId, BlockNumber, MaxCidLen>
{
    fn max_encoded_len() -> usize {
        BoundedVec::<u8, MaxCidLen>::max_encoded_len() // container_cid
            + u32::max_encoded_len() // vram_required_mb
            + ModelCapabilities::max_encoded_len() // capabilities
            + AccountId::max_encoded_len() // registered_by
            + BlockNumber::max_encoded_len() // registered_at
    }
}

/// State of a model on a node
///
/// Represents whether a model is loaded in VRAM, cached on disk, or not present.
#[derive(
    Clone,
    Copy,
    Encode,
    Decode,
    DecodeWithMemTracking,
    Eq,
    PartialEq,
    RuntimeDebug,
    TypeInfo,
    MaxEncodedLen,
    Default,
)]
pub enum ModelState {
    /// Model is loaded in VRAM, ready for immediate inference
    Hot,
    /// Model is cached on disk, can be loaded quickly
    Warm,
    /// Model is not present on the node (default)
    #[default]
    Cold,
}

/// Node capability advertisement
///
/// Nodes advertise their available VRAM and which models they have loaded.
/// This enables task routing to nodes that can execute specific models.
#[derive(Clone, Encode, Decode, DecodeWithMemTracking, Eq, PartialEq, RuntimeDebug, TypeInfo)]
#[scale_info(skip_type_params(MaxModelIdLen, MaxHotModels, MaxWarmModels))]
pub struct NodeCapabilityAd<BlockNumber, MaxModelIdLen, MaxHotModels, MaxWarmModels>
where
    MaxModelIdLen: Get<u32>,
    MaxHotModels: Get<u32>,
    MaxWarmModels: Get<u32>,
{
    /// Available VRAM in MB (after currently loaded models)
    pub available_vram_mb: u32,
    /// Models currently loaded in VRAM (hot)
    pub hot_models: BoundedVec<BoundedVec<u8, MaxModelIdLen>, MaxHotModels>,
    /// Models cached on disk (warm)
    pub warm_models: BoundedVec<BoundedVec<u8, MaxModelIdLen>, MaxWarmModels>,
    /// Block number when this advertisement was last updated
    pub last_updated: BlockNumber,
}

impl<BlockNumber: Default, MaxModelIdLen: Get<u32>, MaxHotModels: Get<u32>, MaxWarmModels: Get<u32>>
    Default for NodeCapabilityAd<BlockNumber, MaxModelIdLen, MaxHotModels, MaxWarmModels>
{
    fn default() -> Self {
        Self {
            available_vram_mb: 0,
            hot_models: BoundedVec::default(),
            warm_models: BoundedVec::default(),
            last_updated: BlockNumber::default(),
        }
    }
}

// Manual MaxEncodedLen implementation for NodeCapabilityAd
impl<
        BlockNumber: MaxEncodedLen,
        MaxModelIdLen: Get<u32>,
        MaxHotModels: Get<u32>,
        MaxWarmModels: Get<u32>,
    > MaxEncodedLen for NodeCapabilityAd<BlockNumber, MaxModelIdLen, MaxHotModels, MaxWarmModels>
{
    fn max_encoded_len() -> usize {
        u32::max_encoded_len() // available_vram_mb
            + BoundedVec::<BoundedVec<u8, MaxModelIdLen>, MaxHotModels>::max_encoded_len() // hot_models
            + BoundedVec::<BoundedVec<u8, MaxModelIdLen>, MaxWarmModels>::max_encoded_len() // warm_models
            + BlockNumber::max_encoded_len() // last_updated
    }
}

/// Metadata for a registered Lane 0 video renderer
///
/// Renderers are modular video generation backends that implement the
/// DeterministicVideoRenderer interface. They must guarantee deterministic
/// output for BFT consensus verification.
///
/// # Lane 0 Requirements
/// - `deterministic` must be `true`
/// - `max_latency_ms` must be <= 15,000 ms
/// - `vram_required_mb` must be <= 11,500 MB
#[derive(Clone, Encode, Decode, DecodeWithMemTracking, Eq, PartialEq, RuntimeDebug, TypeInfo)]
#[scale_info(skip_type_params(MaxCidLen, MaxModelIdLen, MaxRendererModels))]
pub struct RendererMetadata<AccountId, BlockNumber, MaxCidLen, MaxModelIdLen, MaxRendererModels>
where
    MaxCidLen: Get<u32>,
    MaxModelIdLen: Get<u32>,
    MaxRendererModels: Get<u32>,
{
    /// Content identifier for the renderer container (Docker/OCI image)
    pub container_cid: BoundedVec<u8, MaxCidLen>,
    /// Total VRAM required for all models (in MB)
    pub vram_required_mb: u32,
    /// Maximum latency guarantee (in milliseconds)
    pub max_latency_ms: u32,
    /// Model IDs this renderer depends on (must exist in ModelCatalog)
    pub model_dependencies: BoundedVec<BoundedVec<u8, MaxModelIdLen>, MaxRendererModels>,
    /// Whether the renderer guarantees deterministic output (must be true for Lane 0)
    pub deterministic: bool,
    /// Account that registered this renderer
    pub registered_by: AccountId,
    /// Block number when the renderer was registered
    pub registered_at: BlockNumber,
}

impl<
        AccountId: Default,
        BlockNumber: Default,
        MaxCidLen: Get<u32>,
        MaxModelIdLen: Get<u32>,
        MaxRendererModels: Get<u32>,
    > Default for RendererMetadata<AccountId, BlockNumber, MaxCidLen, MaxModelIdLen, MaxRendererModels>
{
    fn default() -> Self {
        Self {
            container_cid: BoundedVec::default(),
            vram_required_mb: 0,
            max_latency_ms: 0,
            model_dependencies: BoundedVec::default(),
            deterministic: false,
            registered_by: AccountId::default(),
            registered_at: BlockNumber::default(),
        }
    }
}

// Manual MaxEncodedLen implementation for RendererMetadata
impl<
        AccountId: MaxEncodedLen,
        BlockNumber: MaxEncodedLen,
        MaxCidLen: Get<u32>,
        MaxModelIdLen: Get<u32>,
        MaxRendererModels: Get<u32>,
    > MaxEncodedLen
    for RendererMetadata<AccountId, BlockNumber, MaxCidLen, MaxModelIdLen, MaxRendererModels>
{
    fn max_encoded_len() -> usize {
        BoundedVec::<u8, MaxCidLen>::max_encoded_len() // container_cid
            + u32::max_encoded_len() // vram_required_mb
            + u32::max_encoded_len() // max_latency_ms
            + BoundedVec::<BoundedVec<u8, MaxModelIdLen>, MaxRendererModels>::max_encoded_len() // model_dependencies
            + bool::max_encoded_len() // deterministic
            + AccountId::max_encoded_len() // registered_by
            + BlockNumber::max_encoded_len() // registered_at
    }
}
