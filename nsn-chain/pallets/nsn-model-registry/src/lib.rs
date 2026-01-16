// Copyright 2024 Neural Sovereign Network
// This file is part of NSN Chain.
//
// NSN Chain is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

//! # NSN Model Registry Pallet
//!
//! Model catalog and node capability registry for the Neural Sovereign Network.
//!
//! ## Overview
//!
//! This pallet implements:
//! - Model catalog: Track available AI models (container CID, VRAM requirements, capabilities)
//! - Node capabilities: Track what models each node has loaded (hot in VRAM, warm on disk)
//!
//! ## Dual-Lane Architecture
//!
//! The NSN uses a dual-lane architecture:
//! - **Lane 0**: Video generation (priority, latency-sensitive)
//! - **Lane 1**: General AI compute (text, code, embeddings, etc.)
//!
//! The model capabilities bitfield indicates which lane(s) a model serves.
//!
//! ## Interface
//!
//! ### Dispatchable Functions
//!
//! - `register_model`: Add a new AI model to the catalog
//! - `update_capabilities`: Node advertises what models it has loaded

#![cfg_attr(not(feature = "std"), no_std)]

pub use pallet::*;

mod types;
pub use types::{ModelCapabilities, ModelMetadata, ModelState, NodeCapabilityAd, RendererMetadata};

#[cfg(test)]
mod mock;
#[cfg(test)]
mod tests;

//#[cfg(feature = "runtime-benchmarks")]
//pub mod benchmarking;

pub mod weights;
pub use weights::WeightInfo;

#[frame_support::pallet]
pub mod pallet {
    use super::*;
    use frame_support::{pallet_prelude::*, traits::StorageVersion};
    use frame_system::pallet_prelude::*;
    use types::RendererMetadata;

    /// Model ID type alias
    pub type ModelIdOf<T> = BoundedVec<u8, <T as Config>::MaxModelIdLen>;

    /// Container CID type alias
    pub type ContainerCidOf<T> = BoundedVec<u8, <T as Config>::MaxCidLen>;

    /// Model metadata type alias
    pub type ModelMetadataOf<T> = ModelMetadata<
        <T as frame_system::Config>::AccountId,
        BlockNumberFor<T>,
        <T as Config>::MaxCidLen,
    >;

    /// Node capability advertisement type alias
    pub type NodeCapabilityAdOf<T> = NodeCapabilityAd<
        BlockNumberFor<T>,
        <T as Config>::MaxModelIdLen,
        <T as Config>::MaxHotModels,
        <T as Config>::MaxWarmModels,
    >;

    /// Renderer ID type alias
    pub type RendererIdOf<T> = BoundedVec<u8, <T as Config>::MaxRendererIdLen>;

    /// Renderer metadata type alias
    pub type RendererMetadataOf<T> = RendererMetadata<
        <T as frame_system::Config>::AccountId,
        BlockNumberFor<T>,
        <T as Config>::MaxCidLen,
        <T as Config>::MaxModelIdLen,
        <T as Config>::MaxRendererModels,
    >;

    /// The in-code storage version.
    const STORAGE_VERSION: StorageVersion = StorageVersion::new(0);

    #[pallet::pallet]
    #[pallet::storage_version(STORAGE_VERSION)]
    pub struct Pallet<T>(_);

    /// Configuration trait for the NSN Model Registry pallet
    #[pallet::config]
    pub trait Config: frame_system::Config<RuntimeEvent: From<Event<Self>>> {
        /// Maximum length of model identifier
        #[pallet::constant]
        type MaxModelIdLen: Get<u32>;

        /// Maximum length of container CID
        #[pallet::constant]
        type MaxCidLen: Get<u32>;

        /// Maximum number of hot models a node can advertise
        #[pallet::constant]
        type MaxHotModels: Get<u32>;

        /// Maximum number of warm models a node can advertise
        #[pallet::constant]
        type MaxWarmModels: Get<u32>;

        /// Maximum length of renderer identifier
        #[pallet::constant]
        type MaxRendererIdLen: Get<u32>;

        /// Maximum number of model dependencies a renderer can declare
        #[pallet::constant]
        type MaxRendererModels: Get<u32>;

        /// Maximum VRAM a renderer can require (in MB) - Lane 0 constraint
        #[pallet::constant]
        type MaxRendererVramMb: Get<u32>;

        /// Maximum latency a renderer can declare (in ms) - Lane 0 constraint
        #[pallet::constant]
        type MaxRendererLatencyMs: Get<u32>;

        /// Weight information
        type WeightInfo: WeightInfo;
    }

    /// Model catalog: Maps model ID to model metadata
    ///
    /// Contains all registered AI models with their container CID,
    /// VRAM requirements, and capability flags.
    ///
    /// # Storage Key
    /// Blake2_128Concat(ModelId) - model IDs may be user-controlled
    #[pallet::storage]
    #[pallet::getter(fn model_catalog)]
    pub type ModelCatalog<T: Config> =
        StorageMap<_, Blake2_128Concat, ModelIdOf<T>, ModelMetadataOf<T>, OptionQuery>;

    /// Node capabilities: Maps node account to capability advertisement
    ///
    /// Nodes advertise what models they have loaded (hot/warm) and
    /// their available VRAM. Used for task routing.
    ///
    /// # Storage Key
    /// Blake2_128Concat(AccountId) - account IDs may be user-controlled
    #[pallet::storage]
    #[pallet::getter(fn node_capabilities)]
    pub type NodeCapabilities<T: Config> =
        StorageMap<_, Blake2_128Concat, T::AccountId, NodeCapabilityAdOf<T>, OptionQuery>;

    /// Renderer catalog: Maps renderer ID to renderer metadata
    ///
    /// Contains all registered Lane 0 video renderers. Renderers are modular
    /// video generation backends that implement the DeterministicVideoRenderer
    /// interface. Network users can register alternative renderers that meet
    /// Lane 0 constraints (determinism, VRAM, latency).
    ///
    /// # Storage Key
    /// Blake2_128Concat(RendererId) - renderer IDs may be user-controlled
    #[pallet::storage]
    #[pallet::getter(fn renderer_catalog)]
    pub type RendererCatalog<T: Config> =
        StorageMap<_, Blake2_128Concat, RendererIdOf<T>, RendererMetadataOf<T>, OptionQuery>;

    /// Events emitted by the pallet
    #[pallet::event]
    #[pallet::generate_deposit(pub(super) fn deposit_event)]
    pub enum Event<T: Config> {
        /// A new model was registered in the catalog
        ModelRegistered {
            model_id: ModelIdOf<T>,
            container_cid: ContainerCidOf<T>,
            vram_required_mb: u32,
            registered_by: T::AccountId,
        },
        /// A node updated its capability advertisement
        NodeCapabilityUpdated {
            node: T::AccountId,
            available_vram_mb: u32,
            hot_model_count: u32,
            warm_model_count: u32,
        },
        /// A new Lane 0 renderer was registered in the catalog
        RendererRegistered {
            renderer_id: RendererIdOf<T>,
            container_cid: ContainerCidOf<T>,
            vram_required_mb: u32,
            max_latency_ms: u32,
            model_count: u32,
            registered_by: T::AccountId,
        },
    }

    /// Errors returned by the pallet
    #[pallet::error]
    pub enum Error<T> {
        /// Model with this ID is already registered
        ModelAlreadyRegistered,
        /// Model not found in the catalog
        ModelNotFound,
        /// Container CID is empty or invalid
        InvalidContainerCid,
        /// Model ID is empty
        InvalidModelId,
        /// No capabilities specified for model
        NoCapabilities,
        /// Too many hot models in advertisement
        TooManyHotModels,
        /// Too many warm models in advertisement
        TooManyWarmModels,
        /// Renderer with this ID is already registered
        RendererAlreadyRegistered,
        /// Renderer ID is empty
        InvalidRendererId,
        /// Lane 0 renderers must be deterministic
        RendererNotDeterministic,
        /// Renderer VRAM exceeds maximum allowed for Lane 0
        RendererVramExceeded,
        /// Renderer latency exceeds maximum allowed for Lane 0
        RendererLatencyExceeded,
        /// A model dependency does not exist in the catalog
        ModelDependencyNotFound,
    }

    #[pallet::hooks]
    impl<T: Config> Hooks<BlockNumberFor<T>> for Pallet<T> {}

    /// Extrinsic calls
    #[pallet::call]
    impl<T: Config> Pallet<T> {
        /// Register a new AI model in the catalog
        ///
        /// # Arguments
        /// * `model_id` - Unique identifier for the model
        /// * `container_cid` - Content identifier for the model container
        /// * `vram_required_mb` - VRAM required to load this model (in MB)
        /// * `capabilities` - Bitfield of model capabilities
        ///
        /// # Errors
        /// * `ModelAlreadyRegistered` - Model ID already exists
        /// * `InvalidContainerCid` - Container CID is empty
        /// * `InvalidModelId` - Model ID is empty
        #[pallet::call_index(0)]
        #[pallet::weight(T::WeightInfo::register_model())]
        pub fn register_model(
            origin: OriginFor<T>,
            model_id: ModelIdOf<T>,
            container_cid: ContainerCidOf<T>,
            vram_required_mb: u32,
            capabilities: ModelCapabilities,
        ) -> DispatchResult {
            let who = ensure_signed(origin)?;

            // Validate inputs
            ensure!(!model_id.is_empty(), Error::<T>::InvalidModelId);
            ensure!(!container_cid.is_empty(), Error::<T>::InvalidContainerCid);

            // Check model doesn't already exist
            ensure!(
                !ModelCatalog::<T>::contains_key(&model_id),
                Error::<T>::ModelAlreadyRegistered
            );

            // Create model metadata
            let current_block = <frame_system::Pallet<T>>::block_number();
            let metadata = ModelMetadata {
                container_cid: container_cid.clone(),
                vram_required_mb,
                capabilities,
                registered_by: who.clone(),
                registered_at: current_block,
            };

            // Store in catalog
            ModelCatalog::<T>::insert(&model_id, metadata);

            Self::deposit_event(Event::ModelRegistered {
                model_id,
                container_cid,
                vram_required_mb,
                registered_by: who,
            });

            Ok(())
        }

        /// Update a node's capability advertisement
        ///
        /// Nodes call this to advertise what models they have loaded
        /// and their available VRAM. This enables task routing to
        /// nodes that can execute specific models.
        ///
        /// # Arguments
        /// * `available_vram_mb` - Available VRAM in MB after loaded models
        /// * `hot_models` - Model IDs currently loaded in VRAM
        /// * `warm_models` - Model IDs cached on disk
        ///
        /// # Errors
        /// * `TooManyHotModels` - Exceeded MaxHotModels
        /// * `TooManyWarmModels` - Exceeded MaxWarmModels
        #[pallet::call_index(1)]
        #[pallet::weight(T::WeightInfo::update_capabilities())]
        pub fn update_capabilities(
            origin: OriginFor<T>,
            available_vram_mb: u32,
            hot_models: BoundedVec<ModelIdOf<T>, T::MaxHotModels>,
            warm_models: BoundedVec<ModelIdOf<T>, T::MaxWarmModels>,
        ) -> DispatchResult {
            let who = ensure_signed(origin)?;

            let current_block = <frame_system::Pallet<T>>::block_number();
            let hot_count = hot_models.len() as u32;
            let warm_count = warm_models.len() as u32;

            // Create capability advertisement
            let capability_ad = NodeCapabilityAd {
                available_vram_mb,
                hot_models,
                warm_models,
                last_updated: current_block,
            };

            // Store/update capability advertisement
            NodeCapabilities::<T>::insert(&who, capability_ad);

            Self::deposit_event(Event::NodeCapabilityUpdated {
                node: who,
                available_vram_mb,
                hot_model_count: hot_count,
                warm_model_count: warm_count,
            });

            Ok(())
        }

        /// Register a new Lane 0 video renderer
        ///
        /// Renderers are modular video generation backends that implement the
        /// DeterministicVideoRenderer interface. Network users can register
        /// alternative renderers (e.g., StableVideoDiffusion, CogVideoX) that
        /// meet Lane 0 constraints.
        ///
        /// # Lane 0 Constraints (enforced)
        /// - Must be deterministic (same input + seed = same output)
        /// - VRAM must be <= MaxRendererVramMb (11,500 MB)
        /// - Latency must be <= MaxRendererLatencyMs (15,000 ms)
        /// - All model dependencies must exist in ModelCatalog
        ///
        /// # Arguments
        /// * `renderer_id` - Unique identifier for the renderer
        /// * `container_cid` - Content identifier for the renderer container
        /// * `vram_required_mb` - Total VRAM required for all models (in MB)
        /// * `max_latency_ms` - Maximum latency guarantee (in ms)
        /// * `model_dependencies` - Model IDs this renderer depends on
        /// * `deterministic` - Whether output is deterministic (must be true)
        ///
        /// # Errors
        /// * `RendererAlreadyRegistered` - Renderer ID already exists
        /// * `InvalidRendererId` - Renderer ID is empty
        /// * `InvalidContainerCid` - Container CID is empty
        /// * `RendererNotDeterministic` - Renderer not deterministic
        /// * `RendererVramExceeded` - VRAM exceeds Lane 0 limit
        /// * `RendererLatencyExceeded` - Latency exceeds Lane 0 limit
        /// * `ModelDependencyNotFound` - A model dependency doesn't exist
        #[pallet::call_index(2)]
        #[pallet::weight(T::WeightInfo::register_model())] // TODO: Add register_renderer weight
        pub fn register_renderer(
            origin: OriginFor<T>,
            renderer_id: RendererIdOf<T>,
            container_cid: ContainerCidOf<T>,
            vram_required_mb: u32,
            max_latency_ms: u32,
            model_dependencies: BoundedVec<ModelIdOf<T>, T::MaxRendererModels>,
            deterministic: bool,
        ) -> DispatchResult {
            let who = ensure_signed(origin)?;

            // Validate inputs
            ensure!(!renderer_id.is_empty(), Error::<T>::InvalidRendererId);
            ensure!(!container_cid.is_empty(), Error::<T>::InvalidContainerCid);

            // Check renderer doesn't already exist
            ensure!(
                !RendererCatalog::<T>::contains_key(&renderer_id),
                Error::<T>::RendererAlreadyRegistered
            );

            // Lane 0 constraints: must be deterministic
            ensure!(deterministic, Error::<T>::RendererNotDeterministic);

            // Lane 0 constraints: VRAM budget
            ensure!(
                vram_required_mb <= T::MaxRendererVramMb::get(),
                Error::<T>::RendererVramExceeded
            );

            // Lane 0 constraints: latency budget
            ensure!(
                max_latency_ms <= T::MaxRendererLatencyMs::get(),
                Error::<T>::RendererLatencyExceeded
            );

            // Validate all model dependencies exist
            for model_id in &model_dependencies {
                ensure!(
                    ModelCatalog::<T>::contains_key(model_id),
                    Error::<T>::ModelDependencyNotFound
                );
            }

            // Create renderer metadata
            let current_block = <frame_system::Pallet<T>>::block_number();
            let model_count = model_dependencies.len() as u32;
            let metadata = RendererMetadata {
                container_cid: container_cid.clone(),
                vram_required_mb,
                max_latency_ms,
                model_dependencies,
                deterministic,
                registered_by: who.clone(),
                registered_at: current_block,
            };

            // Store in catalog
            RendererCatalog::<T>::insert(&renderer_id, metadata);

            Self::deposit_event(Event::RendererRegistered {
                renderer_id,
                container_cid,
                vram_required_mb,
                max_latency_ms,
                model_count,
                registered_by: who,
            });

            Ok(())
        }
    }

    // Helper functions
    impl<T: Config> Pallet<T> {
        /// Check if a model exists in the catalog
        pub fn model_exists(model_id: &ModelIdOf<T>) -> bool {
            ModelCatalog::<T>::contains_key(model_id)
        }

        /// Get model metadata by ID
        pub fn get_model(model_id: &ModelIdOf<T>) -> Option<ModelMetadataOf<T>> {
            ModelCatalog::<T>::get(model_id)
        }

        /// Get node capability advertisement
        pub fn get_node_capabilities(node: &T::AccountId) -> Option<NodeCapabilityAdOf<T>> {
            NodeCapabilities::<T>::get(node)
        }

        /// Check if a node has a specific model hot (in VRAM)
        pub fn node_has_model_hot(node: &T::AccountId, model_id: &ModelIdOf<T>) -> bool {
            if let Some(ad) = NodeCapabilities::<T>::get(node) {
                ad.hot_models.iter().any(|m| m == model_id)
            } else {
                false
            }
        }

        /// Check if a node has a specific model warm (on disk)
        pub fn node_has_model_warm(node: &T::AccountId, model_id: &ModelIdOf<T>) -> bool {
            if let Some(ad) = NodeCapabilities::<T>::get(node) {
                ad.warm_models.iter().any(|m| m == model_id)
            } else {
                false
            }
        }

        /// Get the state of a model on a node
        pub fn get_model_state(node: &T::AccountId, model_id: &ModelIdOf<T>) -> ModelState {
            if Self::node_has_model_hot(node, model_id) {
                ModelState::Hot
            } else if Self::node_has_model_warm(node, model_id) {
                ModelState::Warm
            } else {
                ModelState::Cold
            }
        }

        /// Check if a renderer exists in the catalog
        pub fn renderer_exists(renderer_id: &RendererIdOf<T>) -> bool {
            RendererCatalog::<T>::contains_key(renderer_id)
        }

        /// Get renderer metadata by ID
        pub fn get_renderer(renderer_id: &RendererIdOf<T>) -> Option<RendererMetadataOf<T>> {
            RendererCatalog::<T>::get(renderer_id)
        }

        /// Check if a renderer has a specific model dependency
        pub fn renderer_has_model(renderer_id: &RendererIdOf<T>, model_id: &ModelIdOf<T>) -> bool {
            if let Some(renderer) = RendererCatalog::<T>::get(renderer_id) {
                renderer.model_dependencies.iter().any(|m| m == model_id)
            } else {
                false
            }
        }
    }
}
