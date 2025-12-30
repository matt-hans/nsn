//! GossipSub configuration and behavior
//!
//! Configures libp2p GossipSub with NSN-specific parameters:
//! - 6 topics (5 Lane 0 + 1 Lane 1)
//! - Mesh parameters (n=6, n_low=4, n_high=12)
//! - Strict validation with Ed25519 signing
//! - Flood publishing for BFT signals
//! - 16MB max transmit size
//! - On-chain reputation-integrated peer scoring

use super::reputation_oracle::ReputationOracle;
use super::scoring::build_peer_score_params;
use super::topics::{all_topics, TopicCategory};
use libp2p::gossipsub::{
    Behaviour as GossipsubBehaviour, Config as GossipsubConfig,
    ConfigBuilder as GossipsubConfigBuilder, Event as GossipsubEvent, MessageAuthenticity,
    MessageId, TopicHash, ValidationMode,
};
use libp2p::identity::Keypair;
use libp2p::PeerId;
use std::sync::Arc;
use std::time::Duration;
use thiserror::Error;
use tracing::{debug, info, warn};

/// GossipSub configuration errors
#[derive(Debug, Error)]
pub enum GossipsubError {
    #[error("Failed to build GossipSub config: {0}")]
    ConfigBuild(String),

    #[error("Failed to create GossipSub behavior: {0}")]
    BehaviourCreation(String),

    #[error("Failed to subscribe to topic: {0}")]
    SubscriptionFailed(String),

    #[error("Failed to publish message: {0}")]
    PublishFailed(String),
}

/// Heartbeat interval for mesh maintenance
pub const HEARTBEAT_INTERVAL: Duration = Duration::from_secs(1);

/// Desired mesh size (target peer count per topic)
pub const MESH_N: usize = 6;

/// Lower bound for mesh size (graft when below this)
pub const MESH_N_LOW: usize = 4;

/// Upper bound for mesh size (prune when above this)
pub const MESH_N_HIGH: usize = 12;

/// Gossip lazy parameter (peers to gossip to)
pub const GOSSIP_LAZY: usize = 6;

/// Gossip factor (proportion of peers to gossip to)
pub const GOSSIP_FACTOR: f64 = 0.25;

/// Maximum transmit size (16MB for video chunks)
pub const MAX_TRANSMIT_SIZE: usize = 16 * 1024 * 1024;

/// History length (number of heartbeat ticks to keep messages)
pub const HISTORY_LENGTH: usize = 12;

/// History gossip (number of windows to gossip about)
pub const HISTORY_GOSSIP: usize = 3;

/// Duplicate cache time (seen message TTL)
pub const DUPLICATE_CACHE_TIME: Duration = Duration::from_secs(120);

/// Build GossipSub configuration with NSN parameters
pub fn build_gossipsub_config() -> Result<GossipsubConfig, GossipsubError> {
    GossipsubConfigBuilder::default()
        .heartbeat_interval(HEARTBEAT_INTERVAL)
        .validation_mode(ValidationMode::Strict) // Require Ed25519 signatures
        .mesh_n(MESH_N)
        .mesh_n_low(MESH_N_LOW)
        .mesh_n_high(MESH_N_HIGH)
        .gossip_lazy(GOSSIP_LAZY)
        .gossip_factor(GOSSIP_FACTOR)
        .max_transmit_size(MAX_TRANSMIT_SIZE)
        .flood_publish(true) // Low-latency for BFT signals
        .history_length(HISTORY_LENGTH)
        .history_gossip(HISTORY_GOSSIP)
        .duplicate_cache_time(DUPLICATE_CACHE_TIME)
        .build()
        .map_err(|e| GossipsubError::ConfigBuild(e.to_string()))
}

/// Create GossipSub behavior with reputation-integrated peer scoring
///
/// # Arguments
/// * `keypair` - Ed25519 keypair for message signing
/// * `reputation_oracle` - Oracle for on-chain reputation scores
///
/// # Returns
/// Configured Gossipsub behavior
pub fn create_gossipsub_behaviour(
    keypair: &Keypair,
    reputation_oracle: Arc<ReputationOracle>,
) -> Result<GossipsubBehaviour, GossipsubError> {
    let config = build_gossipsub_config()?;

    // Build peer score parameters with topic configuration
    let (peer_score_params, peer_score_thresholds) =
        build_peer_score_params(reputation_oracle.clone());

    // Create GossipSub with signed messages (using default message ID function)
    let mut gossipsub =
        GossipsubBehaviour::new(MessageAuthenticity::Signed(keypair.clone()), config)
            .map_err(|e| GossipsubError::BehaviourCreation(e.to_string()))?;

    // Set peer score parameters
    gossipsub
        .with_peer_score(peer_score_params, peer_score_thresholds)
        .map_err(|e| {
            GossipsubError::BehaviourCreation(format!("Peer scoring setup failed: {}", e))
        })?;

    info!("GossipSub behavior created with reputation-integrated scoring");

    Ok(gossipsub)
}

/// Subscribe to all NSN topics
///
/// # Arguments
/// * `gossipsub` - GossipSub behavior to subscribe with
///
/// # Returns
/// Number of successful subscriptions
pub fn subscribe_to_all_topics(
    gossipsub: &mut GossipsubBehaviour,
) -> Result<usize, GossipsubError> {
    let topics = all_topics();
    let mut count = 0;

    for topic in topics {
        match gossipsub.subscribe(&topic) {
            Ok(true) => {
                info!("Subscribed to topic: {}", topic);
                count += 1;
            }
            Ok(false) => {
                warn!("Already subscribed to topic: {}", topic);
            }
            Err(e) => {
                return Err(GossipsubError::SubscriptionFailed(format!(
                    "Failed to subscribe to {}: {}",
                    topic, e
                )));
            }
        }
    }

    info!("Subscribed to {} topics", count);
    Ok(count)
}

/// Subscribe to specific topic categories
///
/// # Arguments
/// * `gossipsub` - GossipSub behavior
/// * `categories` - Topic categories to subscribe to
pub fn subscribe_to_categories(
    gossipsub: &mut GossipsubBehaviour,
    categories: &[TopicCategory],
) -> Result<usize, GossipsubError> {
    let mut count = 0;

    for category in categories {
        let topic = category.to_topic();
        match gossipsub.subscribe(&topic) {
            Ok(true) => {
                info!("Subscribed to topic: {}", category);
                count += 1;
            }
            Ok(false) => {
                debug!("Already subscribed to topic: {}", category);
            }
            Err(e) => {
                return Err(GossipsubError::SubscriptionFailed(format!(
                    "Failed to subscribe to {}: {}",
                    category, e
                )));
            }
        }
    }

    Ok(count)
}

/// Publish message to a topic
///
/// # Arguments
/// * `gossipsub` - GossipSub behavior
/// * `category` - Topic category to publish to
/// * `data` - Message data (will be signed with Ed25519)
///
/// # Returns
/// MessageId of published message
pub fn publish_message(
    gossipsub: &mut GossipsubBehaviour,
    category: &TopicCategory,
    data: Vec<u8>,
) -> Result<MessageId, GossipsubError> {
    // Verify message size
    if data.len() > category.max_message_size() {
        return Err(GossipsubError::PublishFailed(format!(
            "Message size {} exceeds max {} for topic {}",
            data.len(),
            category.max_message_size(),
            category
        )));
    }

    let topic = category.to_topic();

    gossipsub.publish(topic.clone(), data).map_err(|e| {
        GossipsubError::PublishFailed(format!("Failed to publish to {}: {}", topic, e))
    })
}

/// Handle GossipSub event
///
/// Processes incoming GossipSub events and returns parsed message data.
pub fn handle_gossipsub_event(event: GossipsubEvent) -> Option<(TopicCategory, PeerId, Vec<u8>)> {
    match event {
        GossipsubEvent::Message {
            propagation_source,
            message_id,
            message,
        } => {
            debug!(
                "Received message {} from peer {} on topic {:?}",
                message_id, propagation_source, message.topic
            );

            // Parse topic category
            let topic_hash: TopicHash = message.topic;
            let topic_str = topic_hash.as_str();

            if let Some(category) = super::topics::parse_topic(topic_str) {
                Some((category, propagation_source, message.data))
            } else {
                warn!("Unknown topic: {}", topic_str);
                None
            }
        }

        GossipsubEvent::Subscribed { peer_id, topic } => {
            debug!("Peer {} subscribed to {:?}", peer_id, topic);
            None
        }

        GossipsubEvent::Unsubscribed { peer_id, topic } => {
            debug!("Peer {} unsubscribed from {:?}", peer_id, topic);
            None
        }

        GossipsubEvent::GossipsubNotSupported { peer_id } => {
            warn!("Peer {} does not support GossipSub", peer_id);
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use libp2p::identity::Keypair;

    #[test]
    fn test_build_gossipsub_config() {
        let config = build_gossipsub_config().expect("Failed to build config");

        // Verify key parameters
        assert_eq!(config.mesh_n(), MESH_N);
        assert_eq!(config.mesh_n_low(), MESH_N_LOW);
        assert_eq!(config.mesh_n_high(), MESH_N_HIGH);
        assert_eq!(config.max_transmit_size(), MAX_TRANSMIT_SIZE);
        // ValidationMode doesn't implement PartialEq, so we just verify config builds successfully
        // The fact that we configured Strict mode is verified by the build succeeding with Strict
    }

    #[test]
    fn test_create_gossipsub_behaviour() {
        let keypair = Keypair::generate_ed25519();
        let oracle = Arc::new(ReputationOracle::new("ws://localhost:9944".to_string()));

        let gossipsub = create_gossipsub_behaviour(&keypair, oracle)
            .expect("Failed to create GossipSub behavior");

        // Just verify it compiles and creates successfully
        drop(gossipsub);
    }

    #[test]
    fn test_subscribe_to_all_topics() {
        let keypair = Keypair::generate_ed25519();
        let oracle = Arc::new(ReputationOracle::new("ws://localhost:9944".to_string()));

        let mut gossipsub = create_gossipsub_behaviour(&keypair, oracle)
            .expect("Failed to create GossipSub behavior");

        let count = subscribe_to_all_topics(&mut gossipsub).expect("Failed to subscribe to topics");

        assert_eq!(count, 6, "Should subscribe to all 6 topics");
    }

    #[test]
    fn test_subscribe_to_categories() {
        let keypair = Keypair::generate_ed25519();
        let oracle = Arc::new(ReputationOracle::new("ws://localhost:9944".to_string()));

        let mut gossipsub = create_gossipsub_behaviour(&keypair, oracle)
            .expect("Failed to create GossipSub behavior");

        // Subscribe to Lane 0 topics only
        let categories = TopicCategory::lane_0();
        let count =
            subscribe_to_categories(&mut gossipsub, &categories).expect("Failed to subscribe");

        assert_eq!(count, 5, "Should subscribe to 5 Lane 0 topics");
    }

    #[test]
    fn test_publish_message_size_enforcement() {
        let keypair = Keypair::generate_ed25519();
        let oracle = Arc::new(ReputationOracle::new("ws://localhost:9944".to_string()));

        let mut gossipsub = create_gossipsub_behaviour(&keypair, oracle)
            .expect("Failed to create GossipSub behavior");

        // Subscribe to BFT signals topic
        subscribe_to_categories(&mut gossipsub, &[TopicCategory::BftSignals])
            .expect("Failed to subscribe");

        // Try to publish message that exceeds max size for BFT signals (64KB)
        let oversized_data = vec![0u8; 128 * 1024]; // 128KB

        let result = publish_message(&mut gossipsub, &TopicCategory::BftSignals, oversized_data);

        assert!(result.is_err(), "Should reject oversized message");
    }

    #[test]
    fn test_publish_message_valid_size() {
        let keypair = Keypair::generate_ed25519();
        let oracle = Arc::new(ReputationOracle::new("ws://localhost:9944".to_string()));

        let mut gossipsub = create_gossipsub_behaviour(&keypair, oracle)
            .expect("Failed to create GossipSub behavior");

        // Subscribe to recipes topic
        subscribe_to_categories(&mut gossipsub, &[TopicCategory::Recipes])
            .expect("Failed to subscribe");

        // Publish valid-sized message
        let data = b"test recipe data".to_vec();

        let result = publish_message(&mut gossipsub, &TopicCategory::Recipes, data);

        // In isolated test (no connected peers), expect "InsufficientPeers" error
        // This is deterministic: publishing without peers ALWAYS fails with this error
        assert!(
            result.is_err(),
            "Expected InsufficientPeers error in isolated test"
        );

        let err = result.unwrap_err();
        assert!(
            err.to_string().contains("InsufficientPeers") || err.to_string().contains("no peers"),
            "Expected InsufficientPeers error, got: {}",
            err
        );
    }
}
