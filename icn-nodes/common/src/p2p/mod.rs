//! P2P networking module for ICN off-chain nodes
//!
//! Provides libp2p-based P2P networking with QUIC transport, Noise XX encryption,
//! Ed25519 identity, connection management, and Prometheus metrics.
//!
//! # Example
//!
//! ```no_run
//! use icn_common::p2p::{P2pConfig, P2pService};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let config = P2pConfig::default();
//!     let (mut service, cmd_tx) = P2pService::new(config).await?;
//!
//!     // Start the service
//!     service.start().await?;
//!
//!     Ok(())
//! }
//! ```

mod behaviour;
mod config;
mod connection_manager;
mod event_handler;
mod identity;
mod metrics;
mod service;

// Re-export public API
pub use behaviour::{ConnectionTracker, IcnBehaviour};
pub use config::P2pConfig;
pub use identity::{
    generate_keypair, load_keypair, peer_id_to_account_id, save_keypair, IdentityError,
};
pub use metrics::{MetricsError, P2pMetrics};
pub use service::{P2pService, ServiceCommand, ServiceError};
