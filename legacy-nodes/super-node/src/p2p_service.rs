//! P2P service with GossipSub and Kademlia DHT
//!
//! Responsibilities:
//! - Subscribe to /icn/video/1.0.0 for video chunk reception from directors
//! - Publish shard manifests to Kademlia DHT for discovery by relays
//! - Maintain peer connections with reputation-weighted scoring

use futures::StreamExt;
use libp2p::{
    gossipsub::{self, IdentTopic, MessageAuthenticity, ValidationMode},
    identify,
    identity::Keypair,
    kad::{self, store::MemoryStore, Quorum, Record, RecordKey},
    noise,
    swarm::SwarmEvent,
    tcp, yamux, Multiaddr, PeerId, Swarm, SwarmBuilder,
};
use serde::{Deserialize, Serialize};
use std::time::Duration;
use tokio::sync::mpsc;
use tracing::{debug, info, warn};

/// Shard manifest for DHT publishing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardManifest {
    pub cid: String,
    pub shards: usize,
    pub locations: Vec<String>, // Multiaddrs
    pub created_at: u64,
}

/// Events from P2P service
#[derive(Debug, Clone)]
pub enum P2PEvent {
    /// Video chunk received from director
    VideoChunkReceived { slot: u64, data: Vec<u8> },
    /// Peer connected
    PeerConnected(PeerId),
    /// Peer disconnected
    PeerDisconnected(PeerId),
}

/// P2P network behaviour
#[derive(libp2p::swarm::NetworkBehaviour)]
struct P2PBehaviour {
    gossipsub: gossipsub::Behaviour,
    kademlia: kad::Behaviour<MemoryStore>,
    identify: identify::Behaviour,
}

/// P2P service managing libp2p swarm
pub struct P2PService {
    swarm: Swarm<P2PBehaviour>,
    event_tx: mpsc::UnboundedSender<P2PEvent>,
    video_topic: IdentTopic,
}

impl P2PService {
    /// Create new P2P service
    ///
    /// # Arguments
    /// * `config` - Super-Node configuration with P2P settings
    ///
    /// # Returns
    /// P2P service instance and event receiver
    pub async fn new(
        config: &crate::config::Config,
    ) -> crate::error::Result<(Self, mpsc::UnboundedReceiver<P2PEvent>)> {
        // Generate or load keypair
        let local_key = Keypair::generate_ed25519();
        let local_peer_id = PeerId::from(local_key.public());

        info!("Local peer ID: {}", local_peer_id);

        // Create GossipSub configuration
        let gossipsub_config = gossipsub::ConfigBuilder::default()
            .heartbeat_interval(Duration::from_secs(1))
            .validation_mode(ValidationMode::Strict)
            .message_id_fn(|message| {
                // Use message content hash as ID
                let mut hasher = blake3::Hasher::new();
                hasher.update(&message.data);
                gossipsub::MessageId::from(hasher.finalize().to_hex().as_bytes().to_vec())
            })
            .build()
            .map_err(|e| {
                crate::error::SuperNodeError::P2P(format!("GossipSub config error: {}", e))
            })?;

        // Create GossipSub behaviour
        let gossipsub = gossipsub::Behaviour::new(
            MessageAuthenticity::Signed(local_key.clone()),
            gossipsub_config,
        )
        .map_err(|e| crate::error::SuperNodeError::P2P(format!("GossipSub init error: {}", e)))?;

        // Create Kademlia DHT
        let store = MemoryStore::new(local_peer_id);
        let mut kademlia = kad::Behaviour::new(local_peer_id, store);
        kademlia.set_mode(Some(kad::Mode::Server)); // Enable DHT serving

        // Create Identify behaviour
        let identify = identify::Behaviour::new(identify::Config::new(
            "/icn/super-node/1.0.0".to_string(),
            local_key.public(),
        ));

        // Build swarm
        let behaviour = P2PBehaviour {
            gossipsub,
            kademlia,
            identify,
        };

        let swarm = SwarmBuilder::with_existing_identity(local_key)
            .with_tokio()
            .with_tcp(
                tcp::Config::default(),
                noise::Config::new,
                yamux::Config::default,
            )
            .map_err(|e| crate::error::SuperNodeError::P2P(format!("TCP config error: {}", e)))?
            .with_behaviour(|_| behaviour)
            .map_err(|e| crate::error::SuperNodeError::P2P(format!("Behaviour error: {}", e)))?
            .with_swarm_config(|c| c.with_idle_connection_timeout(Duration::from_secs(60)))
            .build();

        // Parse listen address
        let listen_addr: Multiaddr = config.p2p_listen_addr.parse().map_err(|e| {
            crate::error::SuperNodeError::P2P(format!("Invalid listen address: {}", e))
        })?;

        let (event_tx, event_rx) = mpsc::unbounded_channel();

        let video_topic = IdentTopic::new("/icn/video/1.0.0");

        let mut service = Self {
            swarm,
            event_tx,
            video_topic,
        };

        // Start listening
        service
            .swarm
            .listen_on(listen_addr.clone())
            .map_err(|e| crate::error::SuperNodeError::P2P(format!("Listen error: {}", e)))?;

        info!("P2P service listening on {}", listen_addr);

        // Add bootstrap peers
        for peer_str in &config.bootstrap_peers {
            if let Ok(multiaddr) = peer_str.parse::<Multiaddr>() {
                if let Some(peer_id) = multiaddr.iter().find_map(|p| {
                    if let libp2p::multiaddr::Protocol::P2p(id) = p {
                        Some(id)
                    } else {
                        None
                    }
                }) {
                    service
                        .swarm
                        .behaviour_mut()
                        .kademlia
                        .add_address(&peer_id, multiaddr);
                    info!("Added bootstrap peer: {}", peer_id);
                }
            } else {
                warn!("Invalid bootstrap peer address: {}", peer_str);
            }
        }

        // Bootstrap Kademlia DHT (non-fatal if no known peers)
        match service.swarm.behaviour_mut().kademlia.bootstrap() {
            Ok(_) => {
                info!("Kademlia DHT bootstrap initiated");
            }
            Err(kad::NoKnownPeers()) => {
                warn!("DHT bootstrap skipped: no known peers (will bootstrap when peers connect)");
            }
        }

        Ok((service, event_rx))
    }

    /// Subscribe to video topic for chunk reception
    pub async fn subscribe_video_topic(&mut self) -> crate::error::Result<()> {
        self.swarm
            .behaviour_mut()
            .gossipsub
            .subscribe(&self.video_topic)
            .map_err(|e| {
                crate::error::SuperNodeError::P2P(format!("GossipSub subscribe error: {}", e))
            })?;

        info!("Subscribed to {}", self.video_topic);
        Ok(())
    }

    /// Publish shard manifest to Kademlia DHT
    ///
    /// # Arguments
    /// * `manifest` - Shard manifest with CID and locations
    ///
    /// # Returns
    /// Result indicating success/failure
    pub async fn publish_shard_manifest(
        &mut self,
        manifest: ShardManifest,
    ) -> crate::error::Result<()> {
        let key = RecordKey::new(&manifest.cid.as_bytes());
        let value = serde_json::to_vec(&manifest)?;

        let record = Record {
            key,
            value,
            publisher: None,
            expires: None,
        };

        self.swarm
            .behaviour_mut()
            .kademlia
            .put_record(record, Quorum::One)
            .map_err(|e| crate::error::SuperNodeError::P2P(format!("DHT put error: {}", e)))?;

        debug!("Published shard manifest for CID: {}", manifest.cid);
        Ok(())
    }

    /// Run P2P service event loop
    ///
    /// This should be spawned as a background task
    pub async fn run(&mut self) -> crate::error::Result<()> {
        loop {
            match self.swarm.next().await {
                Some(SwarmEvent::Behaviour(event)) => {
                    self.handle_behaviour_event(event).await;
                }
                Some(SwarmEvent::NewListenAddr { address, .. }) => {
                    info!("Listening on {}", address);
                }
                Some(SwarmEvent::ConnectionEstablished { peer_id, .. }) => {
                    info!("Connected to peer: {}", peer_id);
                    let _ = self.event_tx.send(P2PEvent::PeerConnected(peer_id));
                }
                Some(SwarmEvent::ConnectionClosed { peer_id, cause, .. }) => {
                    debug!("Disconnected from peer: {} (cause: {:?})", peer_id, cause);
                    let _ = self.event_tx.send(P2PEvent::PeerDisconnected(peer_id));
                }
                Some(SwarmEvent::IncomingConnection { .. }) => {
                    debug!("Incoming connection");
                }
                Some(SwarmEvent::OutgoingConnectionError { peer_id, error, .. }) => {
                    warn!("Outgoing connection error to {:?}: {}", peer_id, error);
                }
                Some(SwarmEvent::IncomingConnectionError { .. }) => {
                    warn!("Incoming connection error");
                }
                _ => {}
            }
        }
    }

    /// Handle behaviour-specific events
    async fn handle_behaviour_event(&mut self, event: P2PBehaviourEvent) {
        match event {
            P2PBehaviourEvent::Gossipsub(gossipsub::Event::Message {
                propagation_source,
                message_id: _,
                message,
            }) => {
                debug!(
                    "Received GossipSub message from {}: {} bytes (topic: {})",
                    propagation_source,
                    message.data.len(),
                    message.topic
                );

                // Parse video chunk if from video topic
                if message.topic == self.video_topic.hash() {
                    // TODO: Parse slot number from message metadata
                    let slot = 0; // Placeholder
                    let _ = self.event_tx.send(P2PEvent::VideoChunkReceived {
                        slot,
                        data: message.data,
                    });
                }
            }
            P2PBehaviourEvent::Gossipsub(gossipsub::Event::Subscribed { peer_id, topic }) => {
                info!("Peer {} subscribed to topic: {}", peer_id, topic);
            }
            P2PBehaviourEvent::Gossipsub(gossipsub::Event::Unsubscribed { peer_id, topic }) => {
                info!("Peer {} unsubscribed from topic: {}", peer_id, topic);
            }
            P2PBehaviourEvent::Kademlia(kad::Event::OutboundQueryProgressed { result, .. }) => {
                match result {
                    kad::QueryResult::PutRecord(Ok(put_record_ok)) => {
                        debug!("DHT record published: {:?}", put_record_ok);
                    }
                    kad::QueryResult::PutRecord(Err(e)) => {
                        warn!("DHT put record failed: {:?}", e);
                    }
                    kad::QueryResult::GetRecord(Ok(get_record_ok)) => {
                        debug!("DHT record retrieved: {:?}", get_record_ok);
                    }
                    kad::QueryResult::GetRecord(Err(e)) => {
                        warn!("DHT get record failed: {:?}", e);
                    }
                    kad::QueryResult::Bootstrap(Ok(_)) => {
                        info!("Kademlia DHT bootstrap complete");
                    }
                    kad::QueryResult::Bootstrap(Err(e)) => {
                        warn!("Kademlia DHT bootstrap failed: {:?}", e);
                    }
                    _ => {}
                }
            }
            P2PBehaviourEvent::Identify(identify::Event::Received { peer_id, info }) => {
                debug!(
                    "Identify received from {}: {:?}",
                    peer_id, info.protocol_version
                );

                // Add peer addresses to Kademlia
                for addr in info.listen_addrs {
                    self.swarm
                        .behaviour_mut()
                        .kademlia
                        .add_address(&peer_id, addr);
                }
            }
            _ => {}
        }
    }

    /// Get number of connected peers
    pub fn peer_count(&self) -> usize {
        self.swarm.connected_peers().count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    /// Helper to create test config
    fn test_config() -> crate::config::Config {
        let tmp_dir = tempdir().unwrap();
        let storage_path = tmp_dir.path().join("storage");
        std::fs::create_dir(&storage_path).unwrap();

        crate::config::Config {
            chain_endpoint: "ws://127.0.0.1:9944".to_string(),
            storage_path,
            quic_port: 9002,
            metrics_port: 9102,
            p2p_listen_addr: "/ip4/127.0.0.1/tcp/0".to_string(),
            bootstrap_peers: vec![],
            region: "NA-WEST".to_string(),
            max_storage_gb: 10_000,
            audit_poll_secs: 30,
            cleanup_interval_blocks: 1000,
        }
    }

    #[tokio::test]
    async fn test_shard_manifest_serialization() {
        let manifest = ShardManifest {
            cid: "bafytest123".to_string(),
            shards: 14,
            locations: vec!["/ip4/1.2.3.4/tcp/9002".to_string()],
            created_at: 1234567890,
        };

        let json = serde_json::to_string(&manifest).unwrap();
        let decoded: ShardManifest = serde_json::from_str(&json).unwrap();

        assert_eq!(decoded.cid, "bafytest123");
        assert_eq!(decoded.shards, 14);
    }

    #[tokio::test]
    async fn test_p2p_service_creation() {
        let config = test_config();
        let result = P2PService::new(&config).await;
        assert!(result.is_ok(), "P2P service creation should succeed");
    }

    #[tokio::test]
    async fn test_subscribe_video_topic() {
        let config = test_config();
        let (mut service, _rx) = P2PService::new(&config).await.unwrap();

        let result = service.subscribe_video_topic().await;
        assert!(result.is_ok(), "Video topic subscription should succeed");
    }

    #[tokio::test]
    async fn test_publish_shard_manifest() {
        let config = test_config();
        let (mut service, _rx) = P2PService::new(&config).await.unwrap();

        let manifest = ShardManifest {
            cid: "bafytest456".to_string(),
            shards: 14,
            locations: vec!["/ip4/10.0.0.1/tcp/9002".to_string()],
            created_at: 1234567890,
        };

        let result = service.publish_shard_manifest(manifest).await;
        assert!(result.is_ok(), "Manifest publishing should succeed");
    }

    #[tokio::test]
    async fn test_peer_count() {
        let config = test_config();
        let (service, _rx) = P2PService::new(&config).await.unwrap();

        // Initially no connected peers
        let count = service.peer_count();
        assert_eq!(count, 0);
    }
}
