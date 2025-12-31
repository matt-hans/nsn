//! Kademlia DHT for NSN
//!
//! Provides decentralized peer discovery, content addressing, and provider
//! records for erasure-coded video shards.
//!
//! # Key Features
//! - Protocol ID: `/nsn/kad/1.0.0`
//! - k-bucket size: k=20
//! - Query timeout: 10 seconds
//! - Routing table refresh: every 5 minutes
//! - Provider record TTL: 12 hours (with automatic republish)

use libp2p::kad::store::MemoryStore;
use libp2p::kad::{
    Behaviour as KademliaBehaviour, Config as KademliaConfig, Event as KademliaEvent,
    GetClosestPeersError, GetProvidersError, GetProvidersOk, QueryId, QueryResult, RecordKey,
};
use libp2p::StreamProtocol;
use libp2p::{Multiaddr, PeerId};
use std::collections::HashMap;
use std::time::Duration;
use thiserror::Error;
use tokio::sync::oneshot;
use tracing::{debug, info, warn};

/// NSN Kademlia protocol ID
pub const NSN_KAD_PROTOCOL_ID: &str = "/nsn/kad/1.0.0";

/// k-bucket size (number of peers per bucket)
pub const K_VALUE: usize = 20;

/// DHT query timeout
pub const QUERY_TIMEOUT: Duration = Duration::from_secs(10);

/// Routing table refresh interval
pub const ROUTING_TABLE_REFRESH_INTERVAL: Duration = Duration::from_secs(300); // 5 minutes

/// Provider record TTL (12 hours)
pub const PROVIDER_RECORD_TTL: Duration = Duration::from_secs(12 * 3600);

/// Provider record republish interval (12 hours)
pub const PROVIDER_REPUBLISH_INTERVAL: Duration = Duration::from_secs(12 * 3600);

#[derive(Debug, Error)]
pub enum KademliaError {
    #[error("No known peers")]
    NoKnownPeers,

    #[error("Query failed: {0}")]
    QueryFailed(String),

    #[error("Timeout")]
    Timeout,

    #[error("Bootstrap failed: {0}")]
    BootstrapFailed(String),

    #[error("Provider publish failed: {0}")]
    ProviderPublishFailed(String),
}

/// Kademlia service configuration
#[derive(Debug, Clone)]
pub struct KademliaServiceConfig {
    /// Bootstrap peers (PeerId, Multiaddr)
    pub bootstrap_peers: Vec<(PeerId, Multiaddr)>,

    /// Enable automatic routing table refresh
    pub auto_refresh: bool,

    /// Enable automatic provider record republish
    pub auto_republish: bool,
}

impl Default for KademliaServiceConfig {
    fn default() -> Self {
        Self {
            bootstrap_peers: Vec::new(),
            auto_refresh: true,
            auto_republish: true,
        }
    }
}

/// Kademlia service for DHT operations
///
/// Manages Kademlia DHT behavior, handles queries, provider records,
/// and routing table maintenance.
pub struct KademliaService {
    /// Kademlia behavior
    pub(crate) kademlia: KademliaBehaviour<MemoryStore>,

    /// Configuration
    config: KademliaServiceConfig,

    /// Pending queries for get_closest_peers
    pending_get_closest_peers:
        HashMap<QueryId, oneshot::Sender<Result<Vec<PeerId>, KademliaError>>>,

    /// Pending queries for get_providers
    pending_get_providers: HashMap<QueryId, oneshot::Sender<Result<Vec<PeerId>, KademliaError>>>,

    /// Pending queries for start_providing
    pending_start_providing: HashMap<QueryId, oneshot::Sender<Result<bool, KademliaError>>>,

    /// Local shards being provided (for republish)
    local_provided_shards: Vec<[u8; 32]>,
}

impl KademliaService {
    /// Create new Kademlia service
    ///
    /// # Arguments
    /// * `local_peer_id` - Local peer ID
    /// * `config` - Kademlia service configuration
    pub fn new(local_peer_id: PeerId, config: KademliaServiceConfig) -> Self {
        let mut kad_config = KademliaConfig::default();

        // Set NSN protocol ID
        let protocol = StreamProtocol::try_from_owned(NSN_KAD_PROTOCOL_ID.to_string())
            .expect("NSN_KAD_PROTOCOL_ID is a valid protocol string");
        kad_config.set_protocol_names(vec![protocol]);

        // Set query timeout
        kad_config.set_query_timeout(QUERY_TIMEOUT);

        // Set replication factor (k-bucket size)
        kad_config
            .set_replication_factor(K_VALUE.try_into().expect("K_VALUE fits in NonZeroUsize"));

        // Set provider record TTL and publication interval
        if config.auto_republish {
            kad_config.set_provider_publication_interval(Some(PROVIDER_REPUBLISH_INTERVAL));
        }
        kad_config.set_provider_record_ttl(Some(PROVIDER_RECORD_TTL));

        // Set record TTL (for future record storage)
        kad_config.set_record_ttl(Some(PROVIDER_RECORD_TTL));

        // Create memory store
        let store = MemoryStore::new(local_peer_id);

        // Create Kademlia behavior
        let mut kademlia = KademliaBehaviour::with_config(local_peer_id, store, kad_config);

        // Add bootstrap peers to routing table
        for (peer_id, addr) in &config.bootstrap_peers {
            kademlia.add_address(peer_id, addr.clone());
        }

        info!(
            "Kademlia DHT initialized with protocol ID: {}, k={}, timeout={:?}",
            NSN_KAD_PROTOCOL_ID, K_VALUE, QUERY_TIMEOUT
        );

        Self {
            kademlia,
            config,
            pending_get_closest_peers: HashMap::new(),
            pending_get_providers: HashMap::new(),
            pending_start_providing: HashMap::new(),
            local_provided_shards: Vec::new(),
        }
    }

    /// Bootstrap the DHT
    ///
    /// Initiates bootstrap queries to populate routing table.
    pub fn bootstrap(&mut self) -> Result<QueryId, KademliaError> {
        if self.config.bootstrap_peers.is_empty() {
            return Err(KademliaError::BootstrapFailed(
                "No bootstrap peers configured".to_string(),
            ));
        }

        self.kademlia
            .bootstrap()
            .map_err(|e| KademliaError::BootstrapFailed(format!("{:?}", e)))
    }

    /// Find closest peers to a target peer ID
    ///
    /// # Arguments
    /// * `target` - Target peer ID
    /// * `result_tx` - Channel to send query results
    pub fn get_closest_peers(
        &mut self,
        target: PeerId,
        result_tx: oneshot::Sender<Result<Vec<PeerId>, KademliaError>>,
    ) -> QueryId {
        let query_id = self.kademlia.get_closest_peers(target);
        self.pending_get_closest_peers.insert(query_id, result_tx);
        debug!("get_closest_peers query initiated: query_id={:?}", query_id);
        query_id
    }

    /// Publish provider record for shard hash
    ///
    /// # Arguments
    /// * `shard_hash` - 32-byte shard hash
    /// * `result_tx` - Channel to send publish result
    pub fn start_providing(
        &mut self,
        shard_hash: [u8; 32],
        result_tx: oneshot::Sender<Result<bool, KademliaError>>,
    ) -> QueryId {
        let key = RecordKey::new(&shard_hash);
        let query_id = self
            .kademlia
            .start_providing(key.clone())
            .expect("start_providing should not fail immediately");

        self.pending_start_providing.insert(query_id, result_tx);

        // Track shard for republishing
        if !self.local_provided_shards.contains(&shard_hash) {
            self.local_provided_shards.push(shard_hash);
        }

        info!(
            "start_providing query initiated: shard_hash={}, query_id={:?}",
            hex::encode(shard_hash),
            query_id
        );

        query_id
    }

    /// Query providers for shard hash
    ///
    /// # Arguments
    /// * `shard_hash` - 32-byte shard hash
    /// * `result_tx` - Channel to send query results (list of provider PeerIds)
    pub fn get_providers(
        &mut self,
        shard_hash: [u8; 32],
        result_tx: oneshot::Sender<Result<Vec<PeerId>, KademliaError>>,
    ) -> QueryId {
        let key = RecordKey::new(&shard_hash);
        let query_id = self.kademlia.get_providers(key.clone());
        self.pending_get_providers.insert(query_id, result_tx);
        debug!(
            "get_providers query initiated: shard_hash={}, query_id={:?}",
            hex::encode(shard_hash),
            query_id
        );
        query_id
    }

    /// Trigger routing table refresh
    ///
    /// Sends FIND_NODE queries to random targets to discover new peers
    /// and validate existing peers.
    pub fn refresh_routing_table(&mut self) {
        let random_peer = PeerId::random();
        let query_id = self.kademlia.get_closest_peers(random_peer);
        debug!("Routing table refresh initiated: query_id={:?}", query_id);
    }

    /// Republish all local provider records
    ///
    /// Called periodically (every 12 hours) to refresh provider records.
    pub fn republish_providers(&mut self) {
        for shard_hash in &self.local_provided_shards {
            let key = RecordKey::new(shard_hash);
            match self.kademlia.start_providing(key.clone()) {
                Ok(query_id) => {
                    debug!(
                        "Republishing provider record: shard_hash={}, query_id={:?}",
                        hex::encode(shard_hash),
                        query_id
                    );
                }
                Err(e) => {
                    warn!(
                        "Failed to republish provider record for shard {}: {:?}",
                        hex::encode(shard_hash),
                        e
                    );
                }
            }
        }

        info!(
            "Republished {} provider records",
            self.local_provided_shards.len()
        );
    }

    /// Get routing table size (number of peers)
    pub fn routing_table_size(&mut self) -> usize {
        self.kademlia
            .kbuckets()
            .map(|bucket| bucket.num_entries())
            .sum()
    }

    /// Handle Kademlia event
    ///
    /// Processes query results and updates pending query channels.
    pub fn handle_event(&mut self, event: KademliaEvent) {
        match event {
            KademliaEvent::OutboundQueryProgressed { id, result, .. } => {
                self.handle_query_result(id, result);
            }
            KademliaEvent::RoutingUpdated { peer, .. } => {
                debug!("Routing table updated: added peer {}", peer);
            }
            KademliaEvent::InboundRequest { request } => {
                debug!("Received inbound DHT request: {:?}", request);
            }
            KademliaEvent::ModeChanged { new_mode } => {
                info!("Kademlia mode changed: {:?}", new_mode);
            }
            _ => {}
        }
    }

    /// Handle query result
    fn handle_query_result(&mut self, query_id: QueryId, result: QueryResult) {
        match result {
            QueryResult::GetClosestPeers(Ok(ok)) => {
                debug!(
                    "get_closest_peers succeeded: query_id={:?}, peers={}",
                    query_id,
                    ok.peers.len()
                );

                if let Some(tx) = self.pending_get_closest_peers.remove(&query_id) {
                    let _ = tx.send(Ok(ok.peers));
                }
            }

            QueryResult::GetClosestPeers(Err(err)) => {
                warn!(
                    "get_closest_peers failed: query_id={:?}, err={:?}",
                    query_id, err
                );

                if let Some(tx) = self.pending_get_closest_peers.remove(&query_id) {
                    let error = match err {
                        GetClosestPeersError::Timeout { .. } => KademliaError::Timeout,
                    };
                    let _ = tx.send(Err(error));
                }
            }

            QueryResult::GetProviders(Ok(GetProvidersOk::FoundProviders { key: _, providers })) => {
                debug!(
                    "get_providers found providers: query_id={:?}, providers={}",
                    query_id,
                    providers.len()
                );

                if let Some(tx) = self.pending_get_providers.remove(&query_id) {
                    let _ = tx.send(Ok(providers.into_iter().collect()));
                }
            }

            QueryResult::GetProviders(Ok(GetProvidersOk::FinishedWithNoAdditionalRecord {
                closest_peers,
            })) => {
                debug!(
                    "get_providers finished: query_id={:?}, closest_peers={}",
                    query_id,
                    closest_peers.len()
                );

                // No additional providers found; return empty if query still pending
                if let Some(tx) = self.pending_get_providers.remove(&query_id) {
                    let _ = tx.send(Ok(Vec::new()));
                }
            }

            QueryResult::GetProviders(Err(err)) => {
                warn!(
                    "get_providers failed: query_id={:?}, err={:?}",
                    query_id, err
                );

                if let Some(tx) = self.pending_get_providers.remove(&query_id) {
                    let error = match err {
                        GetProvidersError::Timeout { .. } => KademliaError::Timeout,
                    };
                    let _ = tx.send(Err(error));
                }
            }

            QueryResult::StartProviding(Ok(_)) => {
                debug!("start_providing succeeded: query_id={:?}", query_id);

                if let Some(tx) = self.pending_start_providing.remove(&query_id) {
                    let _ = tx.send(Ok(true));
                }
            }

            QueryResult::StartProviding(Err(err)) => {
                warn!(
                    "start_providing failed: query_id={:?}, err={:?}",
                    query_id, err
                );

                if let Some(tx) = self.pending_start_providing.remove(&query_id) {
                    let _ = tx.send(Err(KademliaError::ProviderPublishFailed(format!(
                        "{:?}",
                        err
                    ))));
                }
            }

            QueryResult::Bootstrap(Ok(_)) => {
                info!("DHT bootstrap completed: query_id={:?}", query_id);
            }

            QueryResult::Bootstrap(Err(err)) => {
                warn!(
                    "DHT bootstrap failed: query_id={:?}, err={:?}",
                    query_id, err
                );
            }

            _ => {
                debug!("Unhandled query result: query_id={:?}", query_id);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use libp2p::identity::Keypair;

    #[test]
    fn test_kademlia_service_creation() {
        let keypair = Keypair::generate_ed25519();
        let peer_id = PeerId::from(keypair.public());

        let config = KademliaServiceConfig::default();
        let mut service = KademliaService::new(peer_id, config);

        assert_eq!(service.routing_table_size(), 0);
        assert!(service.local_provided_shards.is_empty());
    }

    #[test]
    fn test_kademlia_bootstrap_no_peers_fails() {
        let keypair = Keypair::generate_ed25519();
        let peer_id = PeerId::from(keypair.public());

        let config = KademliaServiceConfig::default();
        let mut service = KademliaService::new(peer_id, config);

        let result = service.bootstrap();
        assert!(result.is_err());
        assert!(matches!(result, Err(KademliaError::BootstrapFailed(_))));
    }

    #[test]
    fn test_provider_record_tracking() {
        let keypair = Keypair::generate_ed25519();
        let peer_id = PeerId::from(keypair.public());

        let config = KademliaServiceConfig::default();
        let mut service = KademliaService::new(peer_id, config);

        let shard_hash: [u8; 32] = [0xAB; 32];
        let (tx, _rx) = oneshot::channel();

        service.start_providing(shard_hash, tx);

        assert_eq!(service.local_provided_shards.len(), 1);
        assert_eq!(service.local_provided_shards[0], shard_hash);
    }

    #[test]
    fn test_routing_table_refresh() {
        let keypair = Keypair::generate_ed25519();
        let peer_id = PeerId::from(keypair.public());

        let config = KademliaServiceConfig::default();
        let mut service = KademliaService::new(peer_id, config);

        // Should not panic
        service.refresh_routing_table();
    }

    #[test]
    fn test_republish_providers() {
        let keypair = Keypair::generate_ed25519();
        let peer_id = PeerId::from(keypair.public());

        let config = KademliaServiceConfig::default();
        let mut service = KademliaService::new(peer_id, config);

        let shard_hash: [u8; 32] = [0xAB; 32];
        let (tx, _rx) = oneshot::channel();
        service.start_providing(shard_hash, tx);

        // Should not panic
        service.republish_providers();
    }
}
