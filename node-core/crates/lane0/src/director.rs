//! Director service for Lane 0 video generation.
//!
//! Manages the director lifecycle: epoch notifications, mode transitions,
//! and slot generation coordination. The DirectorService orchestrates the
//! complete slot generation pipeline from recipe to published video.

use std::collections::HashMap;

use libp2p::identity::Keypair;
use tokio::sync::mpsc;
use tracing::{debug, error, info, warn};

use nsn_scheduler::EpochEvent;
use nsn_types::Recipe;

use crate::bft::BftParticipant;
use crate::error::{DirectorError, DirectorResult, SlotError, SlotResult};
use crate::publisher::ChunkPublisher;
use crate::recipe::RecipeProcessor;
use crate::vortex_client::VortexClient;

/// Configuration for the DirectorService.
#[derive(Debug, Clone)]
pub struct DirectorConfig {
    /// Timeout for BFT consensus in milliseconds.
    pub bft_timeout_ms: u64,
    /// Maximum number of pending recipes to queue.
    pub max_pending_recipes: usize,
    /// Chunk size for video publishing in bytes.
    pub chunk_size: usize,
    /// Sidecar gRPC endpoint.
    pub sidecar_endpoint: String,
}

impl Default for DirectorConfig {
    fn default() -> Self {
        Self {
            bft_timeout_ms: 5000,
            max_pending_recipes: 10,
            chunk_size: 1024 * 1024, // 1 MiB
            sidecar_endpoint: "http://127.0.0.1:50050".to_string(),
        }
    }
}

/// Director lifecycle state machine.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DirectorState {
    /// Not elected, waiting for On-Deck notification.
    Standby,
    /// 120s before epoch, pre-warming models.
    OnDeck {
        /// The upcoming epoch number.
        epoch: u64,
    },
    /// Currently generating slots for this epoch.
    Active {
        /// Current epoch number.
        epoch: u64,
        /// Current slot being processed.
        slot: u64,
    },
    /// Finishing last slot, transitioning back to Standby.
    Draining {
        /// The epoch that just ended.
        epoch: u64,
    },
}

impl DirectorState {
    /// Get the epoch number if in an epoch-related state.
    pub fn epoch(&self) -> Option<u64> {
        match self {
            DirectorState::Standby => None,
            DirectorState::OnDeck { epoch } => Some(*epoch),
            DirectorState::Active { epoch, .. } => Some(*epoch),
            DirectorState::Draining { epoch } => Some(*epoch),
        }
    }

    /// Check if the director is actively generating content.
    pub fn is_active(&self) -> bool {
        matches!(self, DirectorState::Active { .. })
    }

    /// Check if the director is in standby (not elected).
    pub fn is_standby(&self) -> bool {
        matches!(self, DirectorState::Standby)
    }
}

impl std::fmt::Display for DirectorState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DirectorState::Standby => write!(f, "Standby"),
            DirectorState::OnDeck { epoch } => write!(f, "OnDeck(epoch={})", epoch),
            DirectorState::Active { epoch, slot } => {
                write!(f, "Active(epoch={}, slot={})", epoch, slot)
            }
            DirectorState::Draining { epoch } => write!(f, "Draining(epoch={})", epoch),
        }
    }
}

/// Result of processing a single slot.
#[derive(Debug, Clone)]
pub struct SlotResult2 {
    /// Slot number that was processed.
    pub slot_id: u64,
    /// Number of video chunks published.
    pub chunk_count: usize,
    /// Blake3 hash of BFT-agreed embedding.
    pub bft_hash: [u8; 32],
    /// Time spent generating the slot in milliseconds.
    pub generation_ms: u64,
}

/// Commands that can be sent to the DirectorService.
#[derive(Debug)]
pub enum DirectorCommand {
    /// Epoch event from scheduler.
    EpochEvent(EpochEvent),
    /// Recipe received from P2P.
    Recipe(Recipe),
    /// Shutdown the director service.
    Shutdown,
}

/// DirectorService manages the Lane 0 director lifecycle.
///
/// Coordinates between:
/// - Scheduler for epoch notifications
/// - RecipeProcessor for incoming recipes
/// - VortexClient for video generation
/// - BftParticipant for consensus
/// - ChunkPublisher for P2P distribution
pub struct DirectorService {
    /// Current director state.
    state: DirectorState,
    /// Service configuration.
    config: DirectorConfig,
    /// Node keypair for signing.
    keypair: Keypair,
    /// Receiver for epoch events from scheduler.
    epoch_rx: mpsc::Receiver<EpochEvent>,
    /// Vortex pipeline client.
    vortex_client: VortexClient,
    /// BFT consensus participant.
    bft: BftParticipant,
    /// Video chunk publisher.
    publisher: ChunkPublisher,
    /// Recipe processor.
    recipe_processor: RecipeProcessor,
    /// Processed slots in current epoch.
    processed_slots: HashMap<u64, SlotResult2>,
}

impl DirectorService {
    /// Create a new DirectorService.
    ///
    /// # Arguments
    ///
    /// * `config` - Service configuration
    /// * `keypair` - Node identity keypair for signing
    /// * `epoch_rx` - Channel for receiving epoch events from scheduler
    /// * `vortex_client` - Client for calling Vortex via sidecar
    /// * `bft` - BFT consensus participant
    /// * `publisher` - Video chunk publisher
    /// * `recipe_processor` - Recipe processor
    pub fn new(
        config: DirectorConfig,
        keypair: Keypair,
        epoch_rx: mpsc::Receiver<EpochEvent>,
        vortex_client: VortexClient,
        bft: BftParticipant,
        publisher: ChunkPublisher,
        recipe_processor: RecipeProcessor,
    ) -> Self {
        Self {
            state: DirectorState::Standby,
            config,
            keypair,
            epoch_rx,
            vortex_client,
            bft,
            publisher,
            recipe_processor,
            processed_slots: HashMap::new(),
        }
    }

    /// Get the current director state.
    pub fn state(&self) -> &DirectorState {
        &self.state
    }

    /// Get the service configuration.
    pub fn config(&self) -> &DirectorConfig {
        &self.config
    }

    /// Get the node keypair.
    pub fn keypair(&self) -> &Keypair {
        &self.keypair
    }

    /// Run the director service event loop.
    ///
    /// Processes epoch events and recipes, coordinating slot generation.
    pub async fn run(&mut self) -> DirectorResult<()> {
        info!(state = %self.state, "Director service started");

        loop {
            tokio::select! {
                // Handle epoch events from scheduler
                Some(event) = self.epoch_rx.recv() => {
                    if let Err(e) = self.handle_epoch_event(event).await {
                        error!(error = %e, "Failed to handle epoch event");
                    }
                }

                // Handle recipes from P2P (via recipe processor)
                Some(recipe) = self.recipe_processor.next_recipe() => {
                    if let Err(e) = self.handle_recipe(recipe).await {
                        warn!(error = %e, "Failed to handle recipe");
                    }
                }

                else => {
                    debug!("Director service channels closed, shutting down");
                    break;
                }
            }
        }

        info!("Director service stopped");
        Ok(())
    }

    /// Handle an epoch event.
    async fn handle_epoch_event(&mut self, event: EpochEvent) -> DirectorResult<()> {
        match event {
            EpochEvent::OnDeck { epoch, am_director } => {
                self.on_deck(epoch.epoch, am_director).await?;
            }
            EpochEvent::EpochStarted { epoch } => {
                self.on_epoch_start(epoch.epoch).await?;
            }
            EpochEvent::EpochEnded { epoch } => {
                self.on_epoch_end(epoch).await?;
            }
            EpochEvent::DirectorElected {
                epoch,
                directors,
                is_self,
            } => {
                debug!(
                    epoch,
                    directors = ?directors,
                    is_self,
                    "Director election result"
                );
            }
        }
        Ok(())
    }

    /// Handle On-Deck notification (120s before epoch).
    async fn on_deck(&mut self, epoch: u64, am_director: bool) -> DirectorResult<()> {
        if !am_director {
            debug!(epoch, "Not elected as director, staying in Standby");
            return Ok(());
        }

        let old_state = self.state.clone();
        if !old_state.is_standby() {
            return Err(DirectorError::InvalidTransition {
                from: old_state.to_string(),
                to: "OnDeck".to_string(),
            });
        }

        self.state = DirectorState::OnDeck { epoch };
        info!(epoch, "Transitioning to OnDeck state, pre-warming models");

        // TODO: Pre-warm Vortex models via sidecar
        // This could include loading models into VRAM if not already loaded

        Ok(())
    }

    /// Handle epoch start - transition to Active.
    async fn on_epoch_start(&mut self, epoch: u64) -> DirectorResult<()> {
        let old_state = self.state.clone();

        match old_state {
            DirectorState::OnDeck { epoch: pending } if pending == epoch => {
                self.state = DirectorState::Active { epoch, slot: 0 };
                self.processed_slots.clear();
                info!(epoch, "Transitioning to Active state");
                Ok(())
            }
            DirectorState::Standby => {
                // Not elected for this epoch, stay in standby
                debug!(epoch, "Epoch started but not OnDeck, staying Standby");
                Ok(())
            }
            _ => Err(DirectorError::InvalidTransition {
                from: old_state.to_string(),
                to: format!("Active(epoch={})", epoch),
            }),
        }
    }

    /// Handle epoch end - transition to Draining then Standby.
    async fn on_epoch_end(&mut self, epoch: u64) -> DirectorResult<()> {
        let old_state = self.state.clone();

        match old_state {
            DirectorState::Active {
                epoch: current, ..
            } if current == epoch => {
                self.state = DirectorState::Draining { epoch };
                info!(epoch, "Transitioning to Draining state");

                // Complete any pending work
                self.drain_pending_work().await?;

                self.state = DirectorState::Standby;
                info!(epoch, "Epoch complete, transitioning to Standby");
                Ok(())
            }
            DirectorState::Standby | DirectorState::OnDeck { .. } => {
                // Not active for this epoch
                debug!(epoch, "Epoch ended but wasn't active");
                Ok(())
            }
            DirectorState::Draining { epoch: current } if current == epoch => {
                // Already draining
                self.state = DirectorState::Standby;
                Ok(())
            }
            _ => Err(DirectorError::InvalidTransition {
                from: old_state.to_string(),
                to: "Standby".to_string(),
            }),
        }
    }

    /// Handle an incoming recipe.
    async fn handle_recipe(&mut self, recipe: Recipe) -> SlotResult<()> {
        if !self.state.is_active() {
            debug!(
                state = %self.state,
                recipe_id = %recipe.recipe_id,
                "Received recipe but not active, ignoring"
            );
            return Ok(());
        }

        let slot_id = recipe.slot_params.slot_number;

        if self.processed_slots.contains_key(&slot_id) {
            return Err(SlotError::AlreadyProcessed { slot: slot_id });
        }

        info!(
            slot = slot_id,
            recipe_id = %recipe.recipe_id,
            "Processing slot"
        );

        let result = self.process_slot(recipe).await?;

        // Update current slot in state
        if let DirectorState::Active { epoch, .. } = self.state {
            self.state = DirectorState::Active {
                epoch,
                slot: slot_id,
            };
        }

        self.processed_slots.insert(slot_id, result);
        Ok(())
    }

    /// Process a single slot through the complete pipeline.
    ///
    /// 1. Call Vortex for video generation
    /// 2. Run BFT consensus with other directors
    /// 3. Publish video chunks to P2P
    /// 4. Submit BFT result to chain
    pub async fn process_slot(&mut self, recipe: Recipe) -> SlotResult<SlotResult2> {
        let slot_id = recipe.slot_params.slot_number;
        info!(slot = slot_id, "Starting slot generation");

        // 1. Call Vortex for generation
        let output = self.vortex_client.generate_slot(&recipe).await?;
        info!(
            slot = slot_id,
            generation_ms = output.generation_time_ms,
            "Generation complete"
        );

        // 2. Run BFT consensus with other directors
        let bft_result = self
            .bft
            .run_consensus(slot_id, output.clip_embedding.clone(), self.config.bft_timeout_ms)
            .await?;

        if !bft_result.success {
            return Err(SlotError::Consensus(crate::error::BftError::ConsensusFailed {
                slot: slot_id,
                similarity: bft_result.similarity,
                threshold: 0.85,
            }));
        }
        info!(slot = slot_id, signers = bft_result.signers.len(), "BFT consensus reached");

        // 3. Publish video chunks to P2P
        let headers = self
            .publisher
            .publish_video(slot_id, &output.content_id, &output.video_data)
            .await?;
        info!(
            slot = slot_id,
            chunks = headers.len(),
            "Video chunks published"
        );

        // 4. Submit BFT result to chain (placeholder)
        // TODO: Implement chain submission via chain-client
        debug!(slot = slot_id, hash = ?bft_result.canonical_hash, "Would submit BFT result to chain");

        Ok(SlotResult2 {
            slot_id,
            chunk_count: headers.len(),
            bft_hash: bft_result.canonical_hash,
            generation_ms: output.generation_time_ms,
        })
    }

    /// Drain any pending work before transitioning out of Active state.
    async fn drain_pending_work(&mut self) -> DirectorResult<()> {
        // Process any remaining queued recipes
        while let Some(recipe) = self.recipe_processor.try_next_recipe() {
            if let Err(e) = self.process_slot(recipe).await {
                warn!(error = %e, "Failed to process recipe during drain");
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_director_state_display() {
        assert_eq!(DirectorState::Standby.to_string(), "Standby");
        assert_eq!(
            DirectorState::OnDeck { epoch: 5 }.to_string(),
            "OnDeck(epoch=5)"
        );
        assert_eq!(
            DirectorState::Active { epoch: 5, slot: 10 }.to_string(),
            "Active(epoch=5, slot=10)"
        );
        assert_eq!(
            DirectorState::Draining { epoch: 5 }.to_string(),
            "Draining(epoch=5)"
        );
    }

    #[test]
    fn test_director_state_epoch() {
        assert_eq!(DirectorState::Standby.epoch(), None);
        assert_eq!(DirectorState::OnDeck { epoch: 5 }.epoch(), Some(5));
        assert_eq!(
            DirectorState::Active { epoch: 5, slot: 10 }.epoch(),
            Some(5)
        );
        assert_eq!(DirectorState::Draining { epoch: 5 }.epoch(), Some(5));
    }

    #[test]
    fn test_director_state_is_active() {
        assert!(!DirectorState::Standby.is_active());
        assert!(!DirectorState::OnDeck { epoch: 5 }.is_active());
        assert!(DirectorState::Active { epoch: 5, slot: 10 }.is_active());
        assert!(!DirectorState::Draining { epoch: 5 }.is_active());
    }

    #[test]
    fn test_director_state_is_standby() {
        assert!(DirectorState::Standby.is_standby());
        assert!(!DirectorState::OnDeck { epoch: 5 }.is_standby());
        assert!(!DirectorState::Active { epoch: 5, slot: 10 }.is_standby());
        assert!(!DirectorState::Draining { epoch: 5 }.is_standby());
    }

    #[test]
    fn test_config_default() {
        let config = DirectorConfig::default();
        assert_eq!(config.bft_timeout_ms, 5000);
        assert_eq!(config.max_pending_recipes, 10);
        assert_eq!(config.chunk_size, 1024 * 1024);
        assert_eq!(config.sidecar_endpoint, "http://127.0.0.1:50050");
    }
}
