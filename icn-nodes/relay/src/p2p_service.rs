//! P2P service with Kademlia DHT for shard discovery
//!
//! Responsibilities:
//! - Query Kademlia DHT for shard manifests published by Super-Nodes
//! - Publish relay availability to DHT for viewer discovery
//! - Maintain peer connections

use base64::Engine;
use ed25519_dalek::{Signature, Verifier, VerifyingKey};
use libp2p::{
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

/// Shard manifest retrieved from DHT
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardManifest {
    pub cid: String,
    pub shards: usize,
    pub locations: Vec<String>, // Super-Node multiaddrs
    pub created_at: u64,
    /// Publisher PeerId (for signature verification)
    pub publisher_peer_id: String,
    /// Ed25519 signature (base64 encoded)
    pub signature: String,
}

impl ShardManifest {
    /// Verify manifest signature
    pub fn verify(&self, verifying_key: &VerifyingKey) -> crate::error::Result<()> {
        let message = self.canonical_message();

        let sig_bytes = base64::engine::general_purpose::STANDARD
            .decode(&self.signature)
            .map_err(|e| {
                crate::error::RelayError::P2P(format!("Invalid signature encoding: {}", e))
            })?;

        let signature =
            Signature::from_bytes(&sig_bytes.try_into().map_err(|_| {
                crate::error::RelayError::P2P("Invalid signature length".to_string())
            })?);

        verifying_key
            .verify(message.as_bytes(), &signature)
            .map_err(|e| {
                crate::error::RelayError::P2P(format!("Signature verification failed: {}", e))
            })?;

        Ok(())
    }

    /// Get canonical message for signing/verification
    fn canonical_message(&self) -> String {
        format!("{}:{}:{}", self.cid, self.shards, self.locations.join(","))
    }
}

/// P2P events
#[derive(Debug, Clone)]
pub enum P2PEvent {
    PeerConnected(PeerId),
    PeerDisconnected(PeerId),
    ShardManifestFound(ShardManifest),
}

/// P2P network behaviour
#[derive(libp2p::swarm::NetworkBehaviour)]
struct P2PBehaviour {
    kademlia: kad::Behaviour<MemoryStore>,
    identify: identify::Behaviour,
}

/// P2P service for DHT operations
pub struct P2PService {
    swarm: Swarm<P2PBehaviour>,
    event_tx: mpsc::UnboundedSender<P2PEvent>,
}

impl P2PService {
    /// Create new P2P service
    pub async fn new(
        config: &crate::config::Config,
    ) -> crate::error::Result<(Self, mpsc::UnboundedReceiver<P2PEvent>)> {
        let local_key = Keypair::generate_ed25519();
        let local_peer_id = PeerId::from(local_key.public());

        info!("P2P Service: Local peer ID: {}", local_peer_id);

        // Create Kademlia DHT
        let store = MemoryStore::new(local_peer_id);
        let mut kademlia = kad::Behaviour::new(local_peer_id, store);
        kademlia.set_mode(Some(kad::Mode::Client)); // Relay is DHT client

        // Create Identify behaviour
        let identify = identify::Behaviour::new(identify::Config::new(
            "/icn/relay/1.0.0".to_string(),
            local_key.public(),
        ));

        let behaviour = P2PBehaviour { kademlia, identify };

        let swarm = SwarmBuilder::with_existing_identity(local_key)
            .with_tokio()
            .with_tcp(
                tcp::Config::default(),
                noise::Config::new,
                yamux::Config::default,
            )
            .map_err(|e| crate::error::RelayError::P2P(format!("TCP config error: {}", e)))?
            .with_behaviour(|_| behaviour)
            .map_err(|e| crate::error::RelayError::P2P(format!("Behaviour error: {}", e)))?
            .with_swarm_config(|c| c.with_idle_connection_timeout(Duration::from_secs(60)))
            .build();

        let (event_tx, event_rx) = mpsc::unbounded_channel();

        let mut service = Self { swarm, event_tx };

        // Add bootstrap peers (Super-Nodes)
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
                    info!("Added bootstrap peer (Super-Node): {}", peer_id);
                }
            }
        }

        // Bootstrap DHT
        match service.swarm.behaviour_mut().kademlia.bootstrap() {
            Ok(_) => info!("Kademlia DHT bootstrap initiated"),
            Err(kad::NoKnownPeers()) => {
                warn!("DHT bootstrap skipped: no known peers (will bootstrap when peers connect)")
            }
        }

        Ok((service, event_rx))
    }

    /// Query DHT for shard manifest
    pub fn query_shard_manifest(&mut self, cid: &str) {
        let key = RecordKey::new(&cid.as_bytes());
        self.swarm.behaviour_mut().kademlia.get_record(key);
        debug!("Querying DHT for shard manifest: {}", cid);
    }

    /// Publish relay availability to DHT
    pub fn publish_relay_availability(
        &mut self,
        region: &str,
        multiaddr: String,
    ) -> crate::error::Result<()> {
        let key = RecordKey::new(&format!("relay:{}", region).as_bytes());
        let value = serde_json::to_vec(&vec![multiaddr])?;

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
            .map_err(|e| crate::error::RelayError::P2P(format!("DHT put error: {}", e)))?;

        info!("Published relay availability for region: {}", region);
        Ok(())
    }

    /// Run P2P service event loop (stub)
    pub async fn run(&mut self) -> crate::error::Result<()> {
        use futures::StreamExt;

        loop {
            match self.swarm.next().await {
                Some(SwarmEvent::Behaviour(event)) => {
                    self.handle_behaviour_event(event).await;
                }
                Some(SwarmEvent::ConnectionEstablished { peer_id, .. }) => {
                    info!("Connected to peer: {}", peer_id);
                    let _ = self.event_tx.send(P2PEvent::PeerConnected(peer_id));
                }
                Some(SwarmEvent::ConnectionClosed { peer_id, .. }) => {
                    debug!("Disconnected from peer: {}", peer_id);
                    let _ = self.event_tx.send(P2PEvent::PeerDisconnected(peer_id));
                }
                _ => {}
            }
        }
    }

    async fn handle_behaviour_event(&mut self, event: P2PBehaviourEvent) {
        match event {
            P2PBehaviourEvent::Kademlia(kad::Event::OutboundQueryProgressed {
                result: kad::QueryResult::GetRecord(Ok(kad::GetRecordOk::FoundRecord(peer_record))),
                ..
            }) => {
                if let Ok(manifest) =
                    serde_json::from_slice::<ShardManifest>(&peer_record.record.value)
                {
                    debug!("DHT: Found shard manifest for CID: {}", manifest.cid);

                    // Signature verification - accept unsigned manifests for backward compatibility
                    // TODO: Implement key management and enforce signature verification (Phase 6)
                    if manifest.signature.is_empty() {
                        warn!(
                            "WARNING: Manifest for CID {} has no signature - accepting for testnet compatibility (INSECURE)",
                            manifest.cid
                        );
                        // In future: reject unsigned manifests if require_signed_manifests config is true
                    } else {
                        // Future: Verify signature once key management is in place
                        // if let Some(verifying_key) = get_super_node_public_key(&manifest.publisher_peer_id) {
                        //     if let Err(e) = manifest.verify(&verifying_key) {
                        //         warn!("Manifest signature verification failed for CID {}: {}", manifest.cid, e);
                        //         return; // Reject invalid manifest
                        //     }
                        //     debug!("Manifest signature verified for CID {}", manifest.cid);
                        // } else {
                        //     warn!("Unknown publisher {} for CID {}, rejecting manifest", manifest.publisher_peer_id, manifest.cid);
                        //     return;
                        // }
                        debug!("Manifest has signature but verification not yet implemented for CID {}", manifest.cid);
                    }

                    let _ = self.event_tx.send(P2PEvent::ShardManifestFound(manifest));
                }
            }
            P2PBehaviourEvent::Identify(identify::Event::Received { peer_id, info }) => {
                debug!(
                    "Identify received from {}: {:?}",
                    peer_id, info.protocol_version
                );
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
}
