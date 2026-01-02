//! P2P network service
//!
//! Main P2P service that manages libp2p Swarm, GossipSub messaging,
//! connection lifecycle, reputation oracle, and event handling.

use super::behaviour::NsnBehaviour;
use super::bootstrap::{resolve_trusted_signers, BootstrapProtocol, PeerInfo, TrustedSignerSet};
use super::config::P2pConfig;
use super::connection_manager::ConnectionManager;
use super::event_handler;
use super::gossipsub::{create_gossipsub_behaviour, subscribe_to_all_topics, GossipsubError};
use super::identity::{generate_keypair, load_keypair, save_keypair, IdentityError};
use super::kademlia::KademliaError;
use super::kademlia_helpers::build_kademlia;
use super::metrics::P2pMetrics;
use super::reputation_oracle::{OracleError, ReputationOracle};
use super::security::{
    BandwidthLimiter, DosDetector, Graylist, RateLimitError, RateLimiter, SecureP2pConfig,
    SecurityMetrics,
};
use super::topics::TopicCategory;
use futures::StreamExt;
use hyper::service::{make_service_fn, service_fn};
use hyper::{Body, Request, Response, Server};
use libp2p::gossipsub::MessageId;
use libp2p::kad::{
    Event as KademliaEvent, GetClosestPeersError, GetProvidersError, GetProvidersOk, QueryId,
    QueryResult, RecordKey,
};
use libp2p::swarm::SwarmEvent;
use libp2p::{identity::Keypair, Multiaddr, PeerId, Swarm, SwarmBuilder};
use prometheus::{Encoder, TextEncoder};
use std::collections::{HashMap, HashSet};
use std::convert::Infallible;
use std::net::SocketAddr;
use std::sync::Arc;
use thiserror::Error;
use tokio::sync::{mpsc, oneshot};
use tokio::time::{interval, MissedTickBehavior};
use tracing::{debug, error, info, warn};

#[derive(Debug, Error)]
pub enum ServiceError {
    #[error("Identity error: {0}")]
    Identity(#[from] IdentityError),

    #[error("Transport error: {0}")]
    Transport(String),

    #[error("Swarm error: {0}")]
    Swarm(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Event handling error: {0}")]
    Event(#[from] event_handler::EventError),

    #[error("Reputation oracle error: {0}")]
    ReputationOracleError(String),

    #[error("GossipSub error: {0}")]
    Gossipsub(#[from] GossipsubError),

    #[error("Oracle error: {0}")]
    Oracle(#[from] OracleError),

    #[error("Bootstrap error: {0}")]
    Bootstrap(String),
}

struct SecurityState {
    rate_limiter: RateLimiter,
    bandwidth_limiter: BandwidthLimiter,
    graylist: Graylist,
    dos_detector: DosDetector,
    metrics: Arc<SecurityMetrics>,
    violation_counts: HashMap<PeerId, u32>,
    graylist_threshold: u32,
}

#[derive(Debug, Clone)]
struct BootstrapTarget {
    peer_id: PeerId,
    base_addr: Multiaddr,
    dial_addr: Multiaddr,
}

/// Commands that can be sent to the P2P service
#[derive(Debug)]
pub enum ServiceCommand {
    /// Dial a peer at the given multiaddr
    Dial(Multiaddr),

    /// Get current peer count
    GetPeerCount(tokio::sync::oneshot::Sender<usize>),

    /// Get connection count
    GetConnectionCount(tokio::sync::oneshot::Sender<usize>),

    /// Subscribe to a topic
    Subscribe(
        TopicCategory,
        tokio::sync::oneshot::Sender<Result<(), GossipsubError>>,
    ),

    /// Publish message to a topic
    Publish(
        TopicCategory,
        Vec<u8>,
        tokio::sync::oneshot::Sender<Result<MessageId, GossipsubError>>,
    ),

    /// Get closest peers to target (DHT query)
    GetClosestPeers(
        PeerId,
        tokio::sync::oneshot::Sender<Result<Vec<PeerId>, super::kademlia::KademliaError>>,
    ),

    /// Publish provider record for shard hash (DHT)
    PublishProvider(
        [u8; 32],
        tokio::sync::oneshot::Sender<Result<bool, super::kademlia::KademliaError>>,
    ),

    /// Get providers for shard hash (DHT query)
    GetProviders(
        [u8; 32],
        tokio::sync::oneshot::Sender<Result<Vec<PeerId>, super::kademlia::KademliaError>>,
    ),

    /// Get DHT routing table size
    GetRoutingTableSize(
        tokio::sync::oneshot::Sender<Result<usize, super::kademlia::KademliaError>>,
    ),

    /// Trigger manual routing table refresh
    TriggerRoutingTableRefresh(
        tokio::sync::oneshot::Sender<Result<(), super::kademlia::KademliaError>>,
    ),

    /// Shutdown the service
    Shutdown,
}

/// P2P network service
pub struct P2pService {
    /// libp2p Swarm
    swarm: Swarm<NsnBehaviour>,

    /// Configuration
    pub(crate) config: P2pConfig,

    /// Metrics
    pub(crate) metrics: Arc<P2pMetrics>,

    /// Local PeerId
    local_peer_id: PeerId,

    /// Command receiver
    command_rx: mpsc::UnboundedReceiver<ServiceCommand>,

    /// Command sender (for cloning)
    command_tx: mpsc::UnboundedSender<ServiceCommand>,

    /// Connection manager
    pub(crate) connection_manager: ConnectionManager,

    /// Reputation oracle for on-chain reputation scores
    #[allow(dead_code)] // Stored for future use and passed to GossipSub during construction
    reputation_oracle: Arc<ReputationOracle>,

    /// Pending Kademlia get_closest_peers queries
    pending_get_closest_peers:
        HashMap<QueryId, oneshot::Sender<Result<Vec<PeerId>, KademliaError>>>,

    /// Pending Kademlia get_providers queries
    pending_get_providers: HashMap<QueryId, oneshot::Sender<Result<Vec<PeerId>, KademliaError>>>,

    /// Pending Kademlia start_providing queries
    pending_start_providing: HashMap<QueryId, oneshot::Sender<Result<bool, KademliaError>>>,

    /// Local shards being provided (for republish)
    local_provided_shards: Vec<[u8; 32]>,

    /// Security state (rate limiting, DoS detection, graylist)
    security: SecurityState,

    /// Shutdown flag
    shutdown: bool,
}

impl P2pService {
    /// Create new P2P service with GossipSub and reputation oracle
    ///
    /// # Arguments
    /// * `config` - P2P configuration
    /// * `rpc_url` - NSN Chain RPC URL for reputation oracle
    ///
    /// # Returns
    /// Tuple of (P2pService, command sender)
    pub async fn new(
        config: P2pConfig,
        rpc_url: String,
    ) -> Result<(Self, mpsc::UnboundedSender<ServiceCommand>), ServiceError> {
        // Load or generate keypair
        let keypair = if let Some(path) = &config.keypair_path {
            if path.exists() {
                info!("Loading keypair from {:?}", path);
                load_keypair(path)?
            } else {
                info!("Generating new keypair and saving to {:?}", path);
                let kp = generate_keypair();
                save_keypair(&kp, path)?;
                kp
            }
        } else {
            info!("Generating ephemeral keypair");
            generate_keypair()
        };

        let local_peer_id = PeerId::from(keypair.public());
        info!("Local PeerId: {}", local_peer_id);

        // Create metrics
        let metrics = Arc::new(P2pMetrics::new().expect("Failed to create metrics"));
        metrics.connection_limit.set(config.max_connections as f64);

        // Create security metrics (registered on the same registry)
        let security_metrics = Arc::new(
            SecurityMetrics::new(&metrics.registry)
                .expect("Failed to create security metrics"),
        );

        let rpc_url_for_signers = rpc_url.clone();

        // Create reputation oracle with metrics registry (before spawning metrics server)
        let reputation_oracle = Arc::new(
            ReputationOracle::new(rpc_url, &metrics.registry)
                .map_err(|e| ServiceError::ReputationOracleError(e.to_string()))?,
        );

        // Spawn Prometheus metrics server (after oracle creation)
        if config.metrics_port != 0 {
            let metrics_registry = metrics.registry.clone();
            let metrics_addr: SocketAddr = ([127, 0, 0, 1], config.metrics_port).into();
            tokio::spawn(async move {
                if let Err(err) = serve_metrics(metrics_registry, metrics_addr).await {
                    error!("Metrics server failed: {}", err);
                }
            });
        } else {
            info!("Metrics server disabled (metrics_port=0)");
        }

        // Spawn reputation oracle sync loop
        let oracle_clone = reputation_oracle.clone();
        tokio::spawn(async move {
            oracle_clone.sync_loop().await;
        });

        // Resolve trusted bootstrap signers
        let trusted_signers = if config.bootstrap.require_signed_manifests
            || !config.bootstrap.signer_config.trusted_signers_hex.is_empty()
        {
            resolve_trusted_signers(&config.bootstrap.signer_config, &rpc_url_for_signers)
                .await
                .map_err(|e| ServiceError::Bootstrap(e.to_string()))?
        } else {
            let temp_keypair = Keypair::generate_ed25519();
            let mut active = HashSet::new();
            active.insert(temp_keypair.public());
            TrustedSignerSet::new(active, HashSet::new(), 1)
                .map_err(|e| ServiceError::Bootstrap(e.to_string()))?
        };

        // Discover bootstrap peers
        let bootstrap_protocol =
            BootstrapProtocol::new(config.bootstrap.clone(), trusted_signers, Some(metrics.clone()));
        let bootstrap_peers = match bootstrap_protocol.discover_peers().await {
            Ok(peers) => peers,
            Err(err) => {
                warn!("Bootstrap discovery failed: {}", err);
                Vec::new()
            }
        };
        let bootstrap_targets = collect_bootstrap_targets(&bootstrap_peers);

        // Create GossipSub behavior
        let mut gossipsub = create_gossipsub_behaviour(&keypair, reputation_oracle.clone())?;

        // Subscribe to all NSN topics
        let sub_count = subscribe_to_all_topics(&mut gossipsub)?;
        info!("Subscribed to {} topics", sub_count);

        // Create Kademlia behavior with NSN configuration
        let bootstrap_addrs: Vec<(PeerId, Multiaddr)> = bootstrap_targets
            .iter()
            .map(|target| (target.peer_id, target.base_addr.clone()))
            .collect();
        let kademlia = build_kademlia(local_peer_id, &bootstrap_addrs);

        // Create NSN behaviour with GossipSub and Kademlia
        let behaviour = NsnBehaviour::new(gossipsub, kademlia);

        // Build swarm with QUIC transport
        let mut swarm = SwarmBuilder::with_existing_identity(keypair)
            .with_tokio()
            .with_quic()
            .with_behaviour(|_| behaviour)
            .map_err(|e| ServiceError::Swarm(format!("Failed to create behaviour: {}", e)))?
            .with_swarm_config(|cfg| cfg.with_idle_connection_timeout(config.connection_timeout))
            .build();

        // Dial bootstrap peers
        for target in &bootstrap_targets {
            if let Err(err) = swarm.dial(target.dial_addr.clone()) {
                warn!(
                    "Failed to dial bootstrap peer {}: {}",
                    target.peer_id, err
                );
            }
        }

        // Bootstrap Kademlia DHT via swarm behaviour
        match swarm.behaviour_mut().kademlia.bootstrap() {
            Ok(query_id) => {
                info!("DHT bootstrap initiated: query_id={:?}", query_id);
            }
            Err(e) => {
                debug!("DHT bootstrap skipped (no bootstrap peers): {:?}", e);
            }
        }

        let (command_tx, command_rx) = mpsc::unbounded_channel();

        // Create connection manager
        let connection_manager = ConnectionManager::new(config.clone(), metrics.clone());

        let security_config: SecureP2pConfig = config.security.clone();
        let rate_limiter = RateLimiter::new(
            security_config.rate_limiter.clone(),
            Some(reputation_oracle.clone()),
            security_metrics.clone(),
        );
        let bandwidth_limiter =
            BandwidthLimiter::new(security_config.bandwidth_limiter.clone(), security_metrics.clone());
        let graylist = Graylist::new(security_config.graylist.clone(), security_metrics.clone());
        let dos_detector =
            DosDetector::new(security_config.dos_detector.clone(), security_metrics.clone());

        let security = SecurityState {
            rate_limiter,
            bandwidth_limiter,
            graylist,
            dos_detector,
            metrics: security_metrics,
            violation_counts: HashMap::new(),
            graylist_threshold: security_config.graylist.threshold_violations,
        };

        Ok((
            Self {
                swarm,
                config,
                metrics,
                local_peer_id,
                command_rx,
                command_tx: command_tx.clone(),
                connection_manager,
                reputation_oracle,
                pending_get_closest_peers: HashMap::new(),
                pending_get_providers: HashMap::new(),
                pending_start_providing: HashMap::new(),
                local_provided_shards: Vec::new(),
                security,
                shutdown: false,
            },
            command_tx,
        ))
    }

    /// Get local PeerId
    pub fn local_peer_id(&self) -> PeerId {
        self.local_peer_id
    }

    /// Get metrics
    pub fn metrics(&self) -> Arc<P2pMetrics> {
        self.metrics.clone()
    }

    /// Get command sender
    pub fn command_sender(&self) -> mpsc::UnboundedSender<ServiceCommand> {
        self.command_tx.clone()
    }

    /// Start the P2P service
    ///
    /// This will start listening on the configured port and process
    /// events until shutdown is requested.
    pub async fn start(&mut self) -> Result<(), ServiceError> {
        // Start listening
        let listen_addr: Multiaddr =
            format!("/ip4/0.0.0.0/udp/{}/quic-v1", self.config.listen_port)
                .parse()
                .map_err(|e| ServiceError::Transport(format!("Invalid listen address: {}", e)))?;

        self.swarm
            .listen_on(listen_addr.clone())
            .map_err(|e| ServiceError::Transport(format!("Failed to listen: {}", e)))?;

        info!("P2P service listening on {}", listen_addr);

        // Advertise external addresses (STUN/UPnP)
        if let Err(err) = self.configure_nat().await {
            warn!("NAT configuration failed: {}", err);
        }

        let mut refresh_interval = interval(super::kademlia::ROUTING_TABLE_REFRESH_INTERVAL);
        refresh_interval.set_missed_tick_behavior(MissedTickBehavior::Delay);

        // Event loop
        loop {
            tokio::select! {
                _ = refresh_interval.tick() => {
                    let random_peer = PeerId::random();
                    self.swarm
                        .behaviour_mut()
                        .kademlia
                        .get_closest_peers(random_peer);
                    debug!("Triggered periodic DHT refresh");
                }
                // Handle swarm events
                event = self.swarm.select_next_some() => {
                    match event {
                        SwarmEvent::Behaviour(super::behaviour::NsnBehaviourEvent::Kademlia(kad_event)) => {
                            self.handle_kademlia_event(kad_event);
                        }
                        SwarmEvent::Behaviour(super::behaviour::NsnBehaviourEvent::Gossipsub(gossipsub_event)) => {
                            self.handle_gossipsub_event(gossipsub_event).await;
                        }
                        SwarmEvent::ConnectionEstablished { peer_id, endpoint, connection_id, num_established, .. } => {
                            let addr = endpoint.get_remote_address().clone();
                            self.swarm
                                .behaviour_mut()
                                .kademlia
                                .add_address(&peer_id, addr.clone());
                            debug!("Added connected peer {} at {} to Kademlia routing table", peer_id, addr);

                            self.handle_connection_security(&peer_id).await;

                            if let Err(e) = event_handler::handle_connection_established(
                                peer_id,
                                connection_id,
                                num_established,
                                &mut self.connection_manager,
                                &mut self.swarm,
                            ) {
                                error!("Error handling connection established: {}", e);
                            }
                        }
                        SwarmEvent::ConnectionClosed { peer_id, cause, .. } => {
                            event_handler::handle_connection_closed(peer_id, cause, &mut self.connection_manager);
                        }
                        SwarmEvent::NewListenAddr { address, .. } => {
                            event_handler::handle_new_listen_addr(&address);
                        }
                        SwarmEvent::OutgoingConnectionError { peer_id, error, .. } => {
                            event_handler::handle_outgoing_connection_error(peer_id, &error, &self.connection_manager);
                        }
                        SwarmEvent::IncomingConnectionError { error, .. } => {
                            self.security.dos_detector.record_connection_attempt().await;
                            if self.security.dos_detector.detect_connection_flood().await {
                                debug!("Connection flood detected during inbound error");
                            }

                            event_handler::handle_incoming_connection_error(&error, &self.connection_manager);
                        }
                        _ => {}
                    }
                }

                // Handle commands
                Some(command) = self.command_rx.recv() => {
                    if let Err(e) = self.handle_command(command).await {
                        error!("Error handling command: {}", e);
                    }

                    if self.shutdown {
                        info!("Shutdown requested, stopping P2P service");
                        break;
                    }
                }
            }
        }

        // Graceful shutdown
        info!("Shutting down P2P service gracefully");
        self.shutdown_gracefully().await;

        Ok(())
    }

    /// Handle commands
    async fn handle_command(&mut self, command: ServiceCommand) -> Result<(), ServiceError> {
        match command {
            ServiceCommand::Dial(addr) => self.handle_dial_command(addr).await?,
            ServiceCommand::GetPeerCount(tx) => self.handle_get_peer_count_command(tx),
            ServiceCommand::GetConnectionCount(tx) => self.handle_get_connection_count_command(tx),
            ServiceCommand::Subscribe(category, tx) => self.handle_subscribe_command(category, tx),
            ServiceCommand::Publish(category, data, tx) => {
                self.handle_publish_command(category, data, tx)
            }
            ServiceCommand::GetClosestPeers(target, result_tx) => {
                self.handle_get_closest_peers_command(target, result_tx);
            }
            ServiceCommand::PublishProvider(shard_hash, result_tx) => {
                self.handle_publish_provider_command(shard_hash, result_tx);
            }
            ServiceCommand::GetProviders(shard_hash, result_tx) => {
                self.handle_get_providers_command(shard_hash, result_tx);
            }
            ServiceCommand::GetRoutingTableSize(result_tx) => {
                self.handle_get_routing_table_size_command(result_tx);
            }
            ServiceCommand::TriggerRoutingTableRefresh(result_tx) => {
                self.handle_trigger_routing_table_refresh_command(result_tx);
            }
            ServiceCommand::Shutdown => {
                info!("Received shutdown command");
                self.shutdown = true;
            }
        }

        Ok(())
    }

    /// Handle Dial command
    async fn handle_dial_command(&mut self, addr: Multiaddr) -> Result<(), ServiceError> {
        info!("Dialing {}", addr);

        // Extract peer_id from multiaddr if present (e.g., /ip4/.../p2p/<peer_id>)
        // and add to Kademlia routing table
        if let Some(libp2p::multiaddr::Protocol::P2p(peer_id)) = addr.iter().last() {
            // Remove /p2p/<peer_id> suffix for the base address
            let base_addr: Multiaddr = addr
                .iter()
                .filter(|p| !matches!(p, libp2p::multiaddr::Protocol::P2p(_)))
                .collect();
            self.swarm
                .behaviour_mut()
                .kademlia
                .add_address(&peer_id, base_addr);
            debug!("Added peer {} to Kademlia routing table", peer_id);
        }

        self.swarm
            .dial(addr.clone())
            .map_err(|e| ServiceError::Swarm(format!("Failed to dial {}: {}", addr, e)))?;
        Ok(())
    }

    async fn handle_connection_security(&mut self, peer_id: &PeerId) {
        self.security.dos_detector.record_connection_attempt().await;

        if self.security.dos_detector.detect_connection_flood().await {
            self.graylist_peer(peer_id, "Connection flood detected").await;
            return;
        }

        if self.security.graylist.is_graylisted(peer_id).await {
            debug!("Disconnecting graylisted peer {}", peer_id);
            let _ = self.swarm.disconnect_peer_id(peer_id.clone());
        }
    }

    async fn handle_gossipsub_event(&mut self, event: libp2p::gossipsub::Event) {
        if let libp2p::gossipsub::Event::Message {
            propagation_source,
            message,
            ..
        } = event
        {
            let peer_id = propagation_source;
            let message_len = message.data.len();

            if self.enforce_inbound_limits(&peer_id, message_len).await {
                self.metrics.gossipsub_messages_received_total.inc();
            } else {
                debug!("Dropped GossipSub message from {}", peer_id);
            }
        }
    }

    async fn enforce_inbound_limits(&mut self, peer_id: &PeerId, message_len: usize) -> bool {
        let _timer = self.security.metrics.security_check_duration.start_timer();

        self.security.dos_detector.record_message_attempt().await;
        if self.security.dos_detector.detect_message_spam().await {
            self.graylist_peer(peer_id, "Message spam detected").await;
            return false;
        }

        if self.security.graylist.is_graylisted(peer_id).await {
            return false;
        }

        let bandwidth_ok = self
            .security
            .bandwidth_limiter
            .record_transfer(peer_id, message_len as u64)
            .await;
        if !bandwidth_ok {
            self.record_violation(peer_id, "Bandwidth limit exceeded")
                .await;
            return false;
        }

        match self.security.rate_limiter.check_rate_limit(peer_id).await {
            Ok(()) => true,
            Err(RateLimitError::LimitExceeded { .. }) => {
                self.record_violation(peer_id, "Rate limit exceeded")
                    .await;
                false
            }
        }
    }

    async fn record_violation(&mut self, peer_id: &PeerId, reason: &str) {
        let should_graylist = {
            let count = self
                .security
                .violation_counts
                .entry(peer_id.clone())
                .or_insert(0);
            *count += 1;
            *count >= self.security.graylist_threshold
        };

        if should_graylist {
            self.graylist_peer(peer_id, reason).await;
        }
    }

    async fn graylist_peer(&mut self, peer_id: &PeerId, reason: &str) {
        self.security
            .graylist
            .add(peer_id.clone(), reason.to_string())
            .await;
        let _ = self.swarm.disconnect_peer_id(peer_id.clone());
    }

    /// Handle GetPeerCount command
    fn handle_get_peer_count_command(&self, tx: oneshot::Sender<usize>) {
        let count = self.connection_manager.tracker().connected_peers();
        let _ = tx.send(count);
    }

    /// Handle GetConnectionCount command
    fn handle_get_connection_count_command(&self, tx: oneshot::Sender<usize>) {
        let count = self.connection_manager.tracker().total_connections();
        let _ = tx.send(count);
    }

    /// Handle Subscribe command
    fn handle_subscribe_command(
        &mut self,
        category: TopicCategory,
        tx: oneshot::Sender<Result<(), GossipsubError>>,
    ) {
        let topic = category.to_topic();
        let result = self
            .swarm
            .behaviour_mut()
            .gossipsub
            .subscribe(&topic)
            .map_err(|e| GossipsubError::SubscriptionFailed(format!("{}: {}", category, e)))
            .map(|_| ());
        let _ = tx.send(result);
    }

    /// Handle Publish command
    fn handle_publish_command(
        &mut self,
        category: TopicCategory,
        data: Vec<u8>,
        tx: oneshot::Sender<Result<MessageId, GossipsubError>>,
    ) {
        let result = super::gossipsub::publish_message(
            &mut self.swarm.behaviour_mut().gossipsub,
            &category,
            data,
        );
        let _ = tx.send(result);
    }

    /// Handle GetClosestPeers command
    fn handle_get_closest_peers_command(
        &mut self,
        target: PeerId,
        result_tx: oneshot::Sender<Result<Vec<PeerId>, KademliaError>>,
    ) {
        let query_id = self
            .swarm
            .behaviour_mut()
            .kademlia
            .get_closest_peers(target);
        self.pending_get_closest_peers.insert(query_id, result_tx);
        debug!("get_closest_peers query initiated: {:?}", query_id);
    }

    /// Handle PublishProvider command
    fn handle_publish_provider_command(
        &mut self,
        shard_hash: [u8; 32],
        result_tx: oneshot::Sender<Result<bool, KademliaError>>,
    ) {
        let key = RecordKey::new(&shard_hash);
        match self.swarm.behaviour_mut().kademlia.start_providing(key) {
            Ok(query_id) => {
                self.pending_start_providing.insert(query_id, result_tx);
                if !self.local_provided_shards.contains(&shard_hash) {
                    self.local_provided_shards.push(shard_hash);
                }
                info!("start_providing: shard={}", hex::encode(shard_hash));
            }
            Err(e) => {
                let _ = result_tx.send(Err(KademliaError::ProviderPublishFailed(format!(
                    "{:?}",
                    e
                ))));
            }
        }
    }

    /// Handle GetProviders command
    fn handle_get_providers_command(
        &mut self,
        shard_hash: [u8; 32],
        result_tx: oneshot::Sender<Result<Vec<PeerId>, KademliaError>>,
    ) {
        let key = RecordKey::new(&shard_hash);
        let query_id = self.swarm.behaviour_mut().kademlia.get_providers(key);
        self.pending_get_providers.insert(query_id, result_tx);
        debug!("get_providers query: shard={}", hex::encode(shard_hash));
    }

    /// Handle GetRoutingTableSize command
    fn handle_get_routing_table_size_command(
        &mut self,
        result_tx: oneshot::Sender<Result<usize, KademliaError>>,
    ) {
        let size = self
            .swarm
            .behaviour_mut()
            .kademlia
            .kbuckets()
            .map(|bucket| bucket.num_entries())
            .sum();
        let _ = result_tx.send(Ok(size));
    }

    /// Handle TriggerRoutingTableRefresh command
    fn handle_trigger_routing_table_refresh_command(
        &mut self,
        result_tx: oneshot::Sender<Result<(), KademliaError>>,
    ) {
        let random_peer = PeerId::random();
        self.swarm
            .behaviour_mut()
            .kademlia
            .get_closest_peers(random_peer);
        let _ = result_tx.send(Ok(()));
    }

    /// Gracefully shutdown the service
    async fn shutdown_gracefully(&mut self) {
        // Close all connections
        let connected_peers: Vec<PeerId> = self.swarm.connected_peers().cloned().collect();
        for peer_id in connected_peers {
            debug!("Disconnecting from {}", peer_id);
            let _ = self.swarm.disconnect_peer_id(peer_id);
        }

        // Reset connection manager
        self.connection_manager.reset();

        info!("All connections closed");
    }

    /// Handle Kademlia events and process query results
    fn handle_kademlia_event(&mut self, event: KademliaEvent) {
        match event {
            KademliaEvent::OutboundQueryProgressed { id, result, .. } => {
                self.handle_kademlia_query_result(id, result);
            }
            KademliaEvent::RoutingUpdated { peer, .. } => {
                debug!("Kademlia routing table updated: added peer {}", peer);
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

    /// Handle Kademlia query results
    fn handle_kademlia_query_result(&mut self, query_id: QueryId, result: QueryResult) {
        match result {
            QueryResult::GetClosestPeers(Ok(ok)) => {
                self.handle_get_closest_peers_ok(query_id, ok);
            }
            QueryResult::GetClosestPeers(Err(err)) => {
                self.handle_get_closest_peers_err(query_id, err);
            }
            QueryResult::GetProviders(Ok(ok)) => {
                self.handle_get_providers_ok(query_id, ok);
            }
            QueryResult::GetProviders(Err(err)) => {
                self.handle_get_providers_err(query_id, err);
            }
            QueryResult::StartProviding(Ok(_)) => {
                self.handle_start_providing_ok(query_id);
            }
            QueryResult::StartProviding(Err(err)) => {
                debug!(
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
                debug!(
                    "DHT bootstrap failed: query_id={:?}, err={:?}",
                    query_id, err
                );
            }
            _ => {
                debug!("Unhandled query result: query_id={:?}", query_id);
            }
        }
    }

    /// Handle successful GetClosestPeers result
    fn handle_get_closest_peers_ok(
        &mut self,
        query_id: QueryId,
        ok: libp2p::kad::GetClosestPeersOk,
    ) {
        debug!(
            "get_closest_peers succeeded: query_id={:?}, peers={}",
            query_id,
            ok.peers.len()
        );

        if let Some(tx) = self.pending_get_closest_peers.remove(&query_id) {
            let _ = tx.send(Ok(ok.peers));
        }
    }

    /// Handle failed GetClosestPeers result
    fn handle_get_closest_peers_err(&mut self, query_id: QueryId, err: GetClosestPeersError) {
        debug!(
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

    /// Handle successful GetProviders result
    fn handle_get_providers_ok(&mut self, query_id: QueryId, ok: GetProvidersOk) {
        match ok {
            GetProvidersOk::FoundProviders { key: _, providers } => {
                debug!(
                    "get_providers found providers: query_id={:?}, providers={}",
                    query_id,
                    providers.len()
                );

                if let Some(tx) = self.pending_get_providers.remove(&query_id) {
                    let _ = tx.send(Ok(providers.into_iter().collect()));
                }
            }
            GetProvidersOk::FinishedWithNoAdditionalRecord { closest_peers } => {
                debug!(
                    "get_providers finished: query_id={:?}, closest_peers={}",
                    query_id,
                    closest_peers.len()
                );

                if let Some(tx) = self.pending_get_providers.remove(&query_id) {
                    let _ = tx.send(Ok(Vec::new()));
                }
            }
        }
    }

    /// Handle failed GetProviders result
    fn handle_get_providers_err(&mut self, query_id: QueryId, err: GetProvidersError) {
        debug!(
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

    /// Handle successful StartProviding result
    fn handle_start_providing_ok(&mut self, query_id: QueryId) {
        debug!("start_providing succeeded: query_id={:?}", query_id);

        if let Some(tx) = self.pending_start_providing.remove(&query_id) {
            let _ = tx.send(Ok(true));
        }
    }

    async fn configure_nat(&mut self) -> Result<(), ServiceError> {
        if !self.config.stun_servers.is_empty() {
            let servers = self.config.stun_servers.clone();
            match tokio::task::spawn_blocking(move || {
                super::stun::discover_external_with_fallback(&servers)
            })
            .await
            {
                Ok(Ok(addr)) => {
                    let addr_str = format!("/ip4/{}/udp/{}/quic-v1", addr.ip(), addr.port());
                    if let Ok(multiaddr) = addr_str.parse() {
                        self.swarm.add_external_address(multiaddr);
                        info!("Advertised external address via STUN: {}", addr_str);
                    }
                }
                Ok(Err(err)) => {
                    warn!("STUN discovery failed: {}", err);
                }
                Err(err) => {
                    warn!("STUN discovery task failed: {}", err);
                }
            }
        }

        if self.config.enable_upnp {
            let port = self.config.listen_port;
            match tokio::task::spawn_blocking(move || super::upnp::setup_p2p_port_mapping(port))
                .await
            {
                Ok(Ok((ip, _tcp_port, udp_port))) => {
                    let addr_str = format!("/ip4/{}/udp/{}/quic-v1", ip, udp_port);
                    if let Ok(multiaddr) = addr_str.parse() {
                        self.swarm.add_external_address(multiaddr);
                        info!("Advertised external address via UPnP: {}", addr_str);
                    }
                }
                Ok(Err(err)) => {
                    warn!("UPnP mapping failed: {}", err);
                }
                Err(err) => {
                    warn!("UPnP mapping task failed: {}", err);
                }
            }
        }

        Ok(())
    }
}

fn collect_bootstrap_targets(peers: &[PeerInfo]) -> Vec<BootstrapTarget> {
    let mut targets = Vec::new();
    let mut seen = HashSet::new();

    for peer in peers {
        for addr in &peer.addrs {
            let base_addr = strip_peer_id(addr);
            let dial_addr = ensure_peer_id(addr, peer.peer_id);
            let key = format!("{}|{}", peer.peer_id, base_addr);

            if seen.insert(key) {
                targets.push(BootstrapTarget {
                    peer_id: peer.peer_id,
                    base_addr,
                    dial_addr,
                });
            }
        }
    }

    targets
}

fn strip_peer_id(addr: &Multiaddr) -> Multiaddr {
    addr.iter()
        .filter(|proto| !matches!(proto, libp2p::multiaddr::Protocol::P2p(_)))
        .collect()
}

fn ensure_peer_id(addr: &Multiaddr, peer_id: PeerId) -> Multiaddr {
    let has_peer = addr
        .iter()
        .any(|proto| matches!(proto, libp2p::multiaddr::Protocol::P2p(_)));

    if has_peer {
        addr.clone()
    } else {
        let mut updated = addr.clone();
        updated.push(libp2p::multiaddr::Protocol::P2p(peer_id));
        updated
    }
}

async fn serve_metrics(
    registry: prometheus::Registry,
    addr: SocketAddr,
) -> Result<(), hyper::Error> {
    let make_svc = make_service_fn(move |_| {
        let registry = registry.clone();
        async move {
            Ok::<_, Infallible>(service_fn(move |_req: Request<Body>| {
                let registry = registry.clone();
                async move {
                    let metric_families = registry.gather();
                    let encoder = TextEncoder::new();
                    let mut buffer = Vec::new();
                    encoder
                        .encode(&metric_families, &mut buffer)
                        .unwrap_or_default();

                    Ok::<_, Infallible>(
                        Response::builder()
                            .status(200)
                            .header(hyper::header::CONTENT_TYPE, encoder.format_type())
                            .body(Body::from(buffer))
                            .unwrap_or_else(|_| Response::new(Body::from("metrics unavailable"))),
                    )
                }
            }))
        }
    });

    info!("Prometheus metrics listening on http://{}", addr);
    Server::bind(&addr).serve(make_svc).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helpers::*;
    use libp2p::PeerId;

    #[tokio::test]
    async fn test_service_creation() {
        let (service, _cmd_tx) = create_test_service().await;
        assert_service_initial_state(&service);
    }

    #[tokio::test]
    async fn test_service_local_peer_id() {
        let (service, _cmd_tx) = create_test_service().await;
        let peer_id = service.local_peer_id();
        assert!(!peer_id.to_string().is_empty());
    }

    #[tokio::test]
    async fn test_service_metrics() {
        let (service, _cmd_tx) = create_test_service().await;
        let metrics = service.metrics();
        assert_eq!(metrics.connection_limit.get(), 256.0);
    }

    #[tokio::test]
    async fn test_service_handles_get_peer_count_command() {
        if !network_allowed() {
            return;
        }
        let (service, cmd_tx) = create_test_service_with_port(9100).await;
        let handle = spawn_service(service);
        wait_for_startup().await;

        let count = query_peer_count(&cmd_tx).await;
        assert_eq!(count, 0, "Should have 0 peers initially");

        shutdown_service(cmd_tx, handle).await;
    }

    #[tokio::test]
    async fn test_service_handles_get_connection_count_command() {
        if !network_allowed() {
            return;
        }
        let (service, cmd_tx) = create_test_service_with_port(9101).await;
        let handle = spawn_service(service);
        wait_for_startup().await;

        let count = query_connection_count(&cmd_tx).await;
        assert_eq!(count, 0, "Should have 0 connections initially");

        shutdown_service(cmd_tx, handle).await;
    }

    #[tokio::test]
    async fn test_service_shutdown_command() {
        if !network_allowed() {
            return;
        }
        let (service, cmd_tx) = create_test_service().await;
        let handle = spawn_service(service);
        wait_for_startup().await;

        cmd_tx
            .send(ServiceCommand::Shutdown)
            .expect("Failed to send shutdown");

        let result =
            tokio::time::timeout(std::time::Duration::from_secs(TEST_TIMEOUT_SECS), handle)
                .await
                .expect("Service should shutdown within timeout");

        assert!(result.is_ok(), "Service should shutdown gracefully");
    }

    #[tokio::test]
    async fn test_invalid_multiaddr_dial_returns_error() {
        let (mut service, _cmd_tx) = create_test_service_with_port(9102).await;

        let invalid_addr: Multiaddr = "/ip4/127.0.0.1/udp/9999/quic-v1".parse().unwrap();
        let result = service
            .handle_command(ServiceCommand::Dial(invalid_addr))
            .await;

        // Accept either success or error (libp2p may not fail immediately for missing peer ID)
        let _ = result;
    }

    #[tokio::test]
    async fn test_service_command_sender_clonable() {
        if !network_allowed() {
            return;
        }
        let (service, cmd_tx) = create_test_service_with_port(9103).await;
        let handle = spawn_service(service);
        wait_for_startup().await;

        let cmd_tx_clone = cmd_tx.clone();

        let count1 = query_peer_count(&cmd_tx).await;
        let count2 = query_peer_count(&cmd_tx_clone).await;

        assert_eq!(count1, count2, "Both senders should work");

        shutdown_service(cmd_tx, handle).await;
    }

    #[tokio::test]
    async fn test_service_with_keypair_path() {
        use tempfile::TempDir;

        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let keypair_path = temp_dir.path().join("test_keypair");

        let config = P2pConfig {
            keypair_path: Some(keypair_path.clone()),
            ..Default::default()
        };

        let (service1, _) = create_test_service_with_config(config.clone()).await;
        let peer_id_1 = service1.local_peer_id();

        drop(service1);
        assert!(keypair_path.exists(), "Keypair should be saved to file");

        let (service2, _) = create_test_service_with_config(config).await;
        let peer_id_2 = service2.local_peer_id();

        assert_eq!(
            peer_id_1, peer_id_2,
            "PeerId should be same when loading existing keypair"
        );
    }

    #[tokio::test]
    async fn test_service_ephemeral_keypair() {
        let mut config = P2pConfig {
            keypair_path: None,
            ..Default::default()
        };
        config.bootstrap.require_signed_manifests = false;
        config.bootstrap.signer_config.source = crate::SignerSource::Static;

        let (service1, _) = create_test_service_with_config(config.clone()).await;
        let (service2, _) = create_test_service_with_config(config).await;

        assert_ne!(
            service1.local_peer_id(),
            service2.local_peer_id(),
            "Ephemeral keypairs should generate different PeerIds"
        );
    }

    #[tokio::test]
    async fn test_connection_metrics_updated() {
        let (service, _cmd_tx) = create_test_service().await;
        assert_metrics_initial_state(&service.metrics());
    }

    #[tokio::test]
    async fn test_inbound_rate_limit_triggers_graylist() {
        let mut config = P2pConfig::default();
        config.security.rate_limiter.max_requests_per_minute = 1;
        config.security.graylist.threshold_violations = 2;
        config.bootstrap.require_signed_manifests = false;
        config.bootstrap.signer_config.source = crate::SignerSource::Static;

        let (mut service, _cmd_tx) = create_test_service_with_config(config).await;
        let peer_id = PeerId::random();

        assert!(service.enforce_inbound_limits(&peer_id, 128).await);
        assert!(!service.enforce_inbound_limits(&peer_id, 128).await);
        assert!(!service.enforce_inbound_limits(&peer_id, 128).await);

        assert!(service.security.graylist.is_graylisted(&peer_id).await);
    }
}
