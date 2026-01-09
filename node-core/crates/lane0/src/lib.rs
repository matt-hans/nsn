//! Lane 0 video generation orchestration for NSN nodes.
//!
//! This crate provides the orchestration layer for Lane 0 (video generation),
//! connecting the existing components into a complete slot generation pipeline:
//!
//! - **DirectorService**: Manages director lifecycle and epoch transitions
//! - **RecipeProcessor**: Handles incoming recipes from P2P
//! - **VortexClient**: Calls Vortex pipeline via sidecar gRPC
//! - **BftParticipant**: Runs BFT consensus with CLIP embeddings
//! - **ChunkPublisher**: Publishes video chunks to P2P network
//!
//! # Architecture
//!
//! ```text
//! Scheduler                    Lane 0 Crate
//! ┌─────────────────┐          ┌──────────────────────────────┐
//! │ EpochTracker    │──OnDeck──▶│ DirectorService              │
//! │ SchedulerState  │          │   ├── RecipeProcessor        │
//! └─────────────────┘          │   ├── VortexClient           │
//!                              │   ├── BftParticipant         │
//!                              │   └── ChunkPublisher         │
//!                              └──────────────────────────────┘
//!                                        │
//!               ┌────────────────────────┼────────────────────────┐
//!               ▼                        ▼                        ▼
//!      Sidecar (gRPC)              P2P Layer               Chain Client
//!      ┌─────────────┐          ┌─────────────┐          ┌─────────────┐
//!      │ Vortex      │          │ VideoChunks │          │ submit_bft  │
//!      │ Pipeline    │          │ BftSignals  │          │ attestation │
//!      └─────────────┘          └─────────────┘          └─────────────┘
//! ```
//!
//! # Example
//!
//! ```no_run
//! use nsn_lane0::{DirectorService, DirectorConfig};
//! use nsn_scheduler::EpochEvent;
//! use libp2p::identity::Keypair;
//! use tokio::sync::mpsc;
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // Create dependencies (shown simplified)
//! let keypair = Keypair::generate_ed25519();
//! let (_epoch_tx, epoch_rx) = mpsc::channel::<EpochEvent>(16);
//!
//! // DirectorService coordinates the full slot generation pipeline
//! // (requires VortexClient, BftParticipant, ChunkPublisher, RecipeProcessor)
//! # Ok(())
//! # }
//! ```
//!
//! # Slot Generation Pipeline
//!
//! When a recipe is received, the DirectorService executes:
//!
//! 1. **Generation**: Call Vortex via sidecar for video generation
//! 2. **Consensus**: Run BFT with other directors using CLIP embeddings
//! 3. **Publish**: Chunk and sign video, publish to P2P
//! 4. **Submit**: Submit BFT result to chain (canonical hash + signers)

#![warn(missing_docs)]
#![warn(clippy::all)]

pub mod bft;
pub mod director;
pub mod error;
pub mod publisher;
pub mod recipe;
pub mod vortex_client;

// Re-export main types for convenience
pub use bft::{BftConfig, BftConsensusResult, BftParticipant, BftSignal};
pub use director::{DirectorCommand, DirectorConfig, DirectorService, DirectorState, SlotResult2};
pub use error::{
    BftError, BftResult, DirectorError, DirectorResult, Lane0Error, Lane0Result, PublishError,
    PublishResult, RecipeError, RecipeResult, SlotError, SlotResult, VortexError, VortexResult,
};
pub use publisher::{ChunkPublisher, PublisherConfig};
pub use recipe::{RecipeConfig, RecipeProcessor};
pub use vortex_client::{GenerationOutput, VortexClient, VortexClientConfig};
