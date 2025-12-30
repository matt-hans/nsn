//! P2P networking module for NSN off-chain nodes
//!
//! Provides libp2p-based P2P networking with QUIC transport, Noise XX encryption,
//! Ed25519 identity, GossipSub messaging, connection management, and Prometheus metrics.
//!
//! # Example
//!
//! ```no_run
//! use nsn_p2p::{P2pConfig, P2pService};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let config = P2pConfig::default();
//!     let rpc_url = "ws://localhost:9944".to_string();
//!     let (mut service, cmd_tx) = P2pService::new(config, rpc_url).await?;
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
mod gossipsub;
mod identity;
mod metrics;
mod reputation_oracle;
mod scoring;
mod service;
#[cfg(test)]
mod test_helpers;
mod topics;

// Re-export public API
pub use behaviour::{ConnectionTracker, NsnBehaviour};
pub use config::P2pConfig;
pub use gossipsub::{create_gossipsub_behaviour, subscribe_to_all_topics, GossipsubError};
pub use identity::{
    generate_keypair, load_keypair, peer_id_to_account_id, save_keypair, IdentityError,
};
pub use metrics::{MetricsError, P2pMetrics};
pub use reputation_oracle::{OracleError, ReputationOracle, DEFAULT_REPUTATION, SYNC_INTERVAL};
pub use scoring::{
    build_peer_score_params, compute_app_specific_score, GOSSIP_THRESHOLD, GRAYLIST_THRESHOLD,
    PUBLISH_THRESHOLD,
};
pub use service::{P2pService, ServiceCommand, ServiceError};
pub use topics::{all_topics, lane_0_topics, lane_1_topics, parse_topic, TopicCategory};
