//! GossipSub protocol configuration and management
//!
//! PLACEHOLDER: Full implementation deferred to T043
//! This stub provides minimal types to satisfy service.rs compilation

use super::reputation_oracle::ReputationOracle;
use libp2p::gossipsub;
use libp2p::identity::Keypair;
use std::sync::Arc;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum GossipsubError {
    #[error("Subscription failed: {0}")]
    SubscriptionFailed(String),

    #[error("Publish failed: {0}")]
    PublishFailed(String),

    #[error("Configuration error: {0}")]
    ConfigError(String),
}

/// Create GossipSub behaviour (STUB - T043)
pub fn create_gossipsub_behaviour(
    _keypair: &Keypair,
    _reputation_oracle: Arc<ReputationOracle>,
) -> Result<gossipsub::Behaviour, GossipsubError> {
    use libp2p::gossipsub::{ConfigBuilder, MessageAuthenticity};

    let config = ConfigBuilder::default()
        .build()
        .map_err(|e| GossipsubError::ConfigError(e.to_string()))?;

    gossipsub::Behaviour::new(MessageAuthenticity::Signed(_keypair.clone()), config)
        .map_err(|e| GossipsubError::ConfigError(e.to_string()))
}

/// Subscribe to all NSN topics (STUB - T043)
pub fn subscribe_to_all_topics(
    _gossipsub: &mut gossipsub::Behaviour,
) -> Result<usize, GossipsubError> {
    // Stub: return 0 topics subscribed
    Ok(0)
}

/// Publish message to topic (STUB - T043)
pub fn publish_message(
    _gossipsub: &mut gossipsub::Behaviour,
    _category: &super::topics::TopicCategory,
    _data: Vec<u8>,
) -> Result<libp2p::gossipsub::MessageId, GossipsubError> {
    Err(GossipsubError::PublishFailed(
        "Not implemented (T043)".to_string(),
    ))
}
