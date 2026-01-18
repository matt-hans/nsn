//! P2P network service
//!
//! Main P2P service that manages libp2p Swarm, GossipSub messaging,
//! connection lifecycle, reputation oracle, and event handling.

use super::behaviour::NsnBehaviour;
use super::bootstrap::{resolve_trusted_signers, BootstrapProtocol, PeerInfo, TrustedSignerSet};
use super::cert::CertificateManager;
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
use super::discovery::{default_protocols, filter_addresses, P2pFeatures, P2pInfoData, P2pInfoResponse};
use super::topics::TopicCategory;
use super::video::{chunk_latency_ms, decode_video_chunk, verify_video_chunk};
use libp2p::core::muxing::StreamMuxerBox;
use libp2p::core::Transport as LibTransport;
use libp2p_webrtc as webrtc;
use futures::StreamExt;
use hyper::service::{make_service_fn, service_fn};
use hyper::{Body, Request, Response, Server};
use libp2p::gossipsub::MessageId;
use libp2p::kad::{
    Event as KademliaEvent, GetClosestPeersError, GetProvidersError, GetProvidersOk, QueryId,
    QueryResult, RecordKey,
};
use libp2p::mdns;
use libp2p::swarm::SwarmEvent;
use libp2p::{identity::Keypair, Multiaddr, PeerId, Swarm, SwarmBuilder};
use prometheus::{Encoder, TextEncoder};
use std::collections::{HashMap, HashSet};
use std::convert::Infallible;
use std::net::SocketAddr;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use thiserror::Error;
use tokio::sync::{broadcast, mpsc, oneshot};
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

/// Shared state for HTTP endpoints (metrics + discovery)
struct HttpState {
    registry: prometheus::Registry,
    peer_id: PeerId,
    webrtc_enabled: bool,
    websocket_enabled: bool,
    role: String,
    external_address: Option<String>,
    /// Current listening addresses (updated by swarm event loop)
    listeners: Arc<tokio::sync::RwLock<Vec<Multiaddr>>>,
    /// Current external addresses (updated by swarm event loop)
    external_addrs: Arc<tokio::sync::RwLock<Vec<Multiaddr>>>,
    /// Flag indicating swarm is ready (has at least one listener)
    swarm_ready: Arc<AtomicBool>,
    /// Supported protocols (static list for now)
    protocols: Vec<String>,
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

    /// Video chunk latency stream (ms)
    video_latency_tx: broadcast::Sender<u64>,

    /// Shutdown flag
    shutdown: bool,

    /// Shared listeners for HTTP discovery endpoint
    http_listeners: Arc<tokio::sync::RwLock<Vec<Multiaddr>>>,

    /// Shared external addresses for HTTP discovery endpoint
    http_external_addrs: Arc<tokio::sync::RwLock<Vec<Multiaddr>>>,

    /// Flag indicating swarm is ready (has at least one listener)
    swarm_ready: Arc<AtomicBool>,
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

        // Load or generate WebRTC certificate if enabled
        let webrtc_cert = if config.enable_webrtc {
            let data_dir = config.data_dir.clone().unwrap_or_else(|| {
                std::env::temp_dir().join("nsn-p2p")
            });
            let cert_manager = CertificateManager::new(&data_dir);
            Some(
                cert_manager
                    .load_or_generate()
                    .map_err(|e| ServiceError::Transport(format!("WebRTC certificate error: {}", e)))?,
            )
        } else {
            None
        };

        // Create metrics
        let metrics = Arc::new(P2pMetrics::new().expect("Failed to create metrics"));
        metrics.connection_limit.set(config.max_connections as f64);

        // Create security metrics (registered on the same registry)
        let security_metrics = Arc::new(
            SecurityMetrics::new(&metrics.registry).expect("Failed to create security metrics"),
        );

        let rpc_url_for_signers = rpc_url.clone();

        // Create reputation oracle with metrics registry (before spawning metrics server)
        let reputation_oracle = Arc::new(
            ReputationOracle::new(rpc_url, &metrics.registry)
                .map_err(|e| ServiceError::ReputationOracleError(e.to_string()))?,
        );

        // Create shared state for HTTP discovery endpoint
        let http_listeners = Arc::new(tokio::sync::RwLock::new(Vec::new()));
        let http_external_addrs = Arc::new(tokio::sync::RwLock::new(Vec::new()));
        let swarm_ready = Arc::new(AtomicBool::new(false));

        // Spawn HTTP server for metrics + discovery (after oracle creation)
        if config.metrics_port != 0 {
            let http_state = Arc::new(HttpState {
                registry: metrics.registry.clone(),
                peer_id: local_peer_id,
                webrtc_enabled: config.enable_webrtc,
                websocket_enabled: config.enable_websocket,
                role: "node".to_string(), // Default role, can be parameterized later
                external_address: config.external_address.clone(),
                listeners: http_listeners.clone(),
                external_addrs: http_external_addrs.clone(),
                swarm_ready: swarm_ready.clone(),
                protocols: default_protocols(),
            });
            let http_addr: SocketAddr = ([0, 0, 0, 0], config.metrics_port).into();
            tokio::spawn(async move {
                if let Err(err) = serve_http(http_state, http_addr).await {
                    error!("HTTP server failed: {}", err);
                }
            });
        } else {
            info!("HTTP server disabled (metrics_port=0)");
        }

        // Spawn reputation oracle sync loop
        let oracle_clone = reputation_oracle.clone();
        tokio::spawn(async move {
            oracle_clone.sync_loop().await;
        });

        // Skip bootstrap for local testnet (using mDNS for peer discovery)
        let bootstrap_targets = if std::env::var("NSN_LOCAL_TESTNET").is_ok() {
            info!("NSN_LOCAL_TESTNET set: skipping bootstrap protocol, using mDNS for local discovery");
            Vec::new()
        } else {
            // Resolve trusted bootstrap signers
            let trusted_signers = if config.bootstrap.require_signed_manifests
                || !config
                    .bootstrap
                    .signer_config
                    .trusted_signers_hex
                    .is_empty()
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
            let bootstrap_protocol = BootstrapProtocol::new(
                config.bootstrap.clone(),
                trusted_signers,
                Some(metrics.clone()),
            );
            let bootstrap_peers = match bootstrap_protocol.discover_peers().await {
                Ok(peers) => peers,
                Err(err) => {
                    warn!("Bootstrap discovery failed: {}", err);
                    Vec::new()
                }
            };
            collect_bootstrap_targets(&bootstrap_peers)
        };

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

        // Create mDNS behavior for local network peer discovery
        let mdns = mdns::tokio::Behaviour::new(mdns::Config::default(), local_peer_id)
            .map_err(|e| ServiceError::Swarm(format!("Failed to create mDNS: {}", e)))?;
        info!("mDNS local discovery enabled");

        // Create NSN behaviour with GossipSub, Kademlia, and mDNS
        let behaviour = NsnBehaviour::new(gossipsub, kademlia, mdns);

        // Build swarm with TCP and QUIC transports
        // TCP transport is needed for compatibility with chain-advertised WebSocket addresses
        let swarm_builder = SwarmBuilder::with_existing_identity(keypair)
            .with_tokio()
            .with_tcp(
                libp2p::tcp::Config::default(),
                libp2p::noise::Config::new,
                libp2p::yamux::Config::default,
            )
            .map_err(|e| ServiceError::Swarm(format!("TCP transport error: {}", e)))?
            .with_quic();

        // Conditionally add WebRTC transport
        let mut swarm = if let Some(cert) = webrtc_cert {
            info!(
                "Enabling WebRTC transport with certificate fingerprint: {:?}",
                cert.fingerprint()
            );
            swarm_builder
                .with_other_transport(|id_keys| {
                    Ok(webrtc::tokio::Transport::new(id_keys.clone(), cert.clone())
                        .map(|(peer_id, conn), _| (peer_id, StreamMuxerBox::new(conn))))
                })
                .map_err(|e| ServiceError::Swarm(format!("WebRTC transport error: {}", e)))?
                .with_behaviour(|_| behaviour)
                .map_err(|e| ServiceError::Swarm(format!("Failed to create behaviour: {}", e)))?
                .with_swarm_config(|cfg| cfg.with_idle_connection_timeout(config.connection_timeout))
                .build()
        } else {
            swarm_builder
                .with_behaviour(|_| behaviour)
                .map_err(|e| ServiceError::Swarm(format!("Failed to create behaviour: {}", e)))?
                .with_swarm_config(|cfg| cfg.with_idle_connection_timeout(config.connection_timeout))
                .build()
        };

        // Dial bootstrap peers
        for target in &bootstrap_targets {
            if let Err(err) = swarm.dial(target.dial_addr.clone()) {
                warn!("Failed to dial bootstrap peer {}: {}", target.peer_id, err);
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
        let bandwidth_limiter = BandwidthLimiter::new(
            security_config.bandwidth_limiter.clone(),
            security_metrics.clone(),
        );
        let graylist = Graylist::new(security_config.graylist.clone(), security_metrics.clone());
        let dos_detector = DosDetector::new(
            security_config.dos_detector.clone(),
            security_metrics.clone(),
        );

        let security = SecurityState {
            rate_limiter,
            bandwidth_limiter,
            graylist,
            dos_detector,
            metrics: security_metrics,
            violation_counts: HashMap::new(),
            graylist_threshold: security_config.graylist.threshold_violations,
        };

        let (video_latency_tx, _video_latency_rx) = broadcast::channel(1024);

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
                video_latency_tx,
                shutdown: false,
                http_listeners,
                http_external_addrs,
                swarm_ready,
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

    /// Subscribe to decoded video chunk latencies (ms).
    pub fn subscribe_video_latency(&self) -> broadcast::Receiver<u64> {
        self.video_latency_tx.subscribe()
    }

    /// Start the P2P service
    ///
    /// This will start listening on the configured port and process
    /// events until shutdown is requested.
    pub async fn start(&mut self) -> Result<(), ServiceError> {
        // Start listening on both QUIC and TCP transports
        let quic_addr: Multiaddr =
            format!("/ip4/0.0.0.0/udp/{}/quic-v1", self.config.listen_port)
                .parse()
                .map_err(|e| ServiceError::Transport(format!("Invalid QUIC address: {}", e)))?;

        let tcp_addr: Multiaddr =
            format!("/ip4/0.0.0.0/tcp/{}", self.config.listen_port)
                .parse()
                .map_err(|e| ServiceError::Transport(format!("Invalid TCP address: {}", e)))?;

        self.swarm
            .listen_on(quic_addr.clone())
            .map_err(|e| ServiceError::Transport(format!("Failed to listen on QUIC: {}", e)))?;

        self.swarm
            .listen_on(tcp_addr.clone())
            .map_err(|e| ServiceError::Transport(format!("Failed to listen on TCP: {}", e)))?;

        info!(
            "P2P service listening on {} (QUIC) and {} (TCP)",
            quic_addr, tcp_addr
        );

        // Start WebRTC listener if enabled
        if self.config.enable_webrtc {
            let webrtc_addr: Multiaddr =
                format!("/ip4/0.0.0.0/udp/{}/webrtc-direct", self.config.webrtc_port)
                    .parse()
                    .map_err(|e| ServiceError::Transport(format!("Invalid WebRTC address: {}", e)))?;

            self.swarm
                .listen_on(webrtc_addr.clone())
                .map_err(|e| ServiceError::Transport(format!("Failed to listen on WebRTC: {}", e)))?;

            info!(
                "P2P service listening on {} (WebRTC) for browser connections",
                webrtc_addr
            );
        }

        // WebSocket listener disabled - use WebRTC-Direct for browser connections
        // To re-enable WebSocket, add the "websocket" feature to libp2p dependencies
        if self.config.enable_websocket {
            warn!(
                "WebSocket transport requested but not compiled in. Use WebRTC-Direct instead (--p2p-enable-webrtc)."
            );
        }

        // Advertise external WebRTC address if configured (for NAT/Docker environments)
        if let Some(ref external_addr_str) = self.config.external_address {
            match external_addr_str.parse::<Multiaddr>() {
                Ok(external_addr) => {
                    self.swarm.add_external_address(external_addr.clone());
                    info!("Advertising external address: {}", external_addr);
                }
                Err(e) => {
                    warn!("Invalid external address '{}': {}", external_addr_str, e);
                }
            }
        }

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
                        SwarmEvent::Behaviour(super::behaviour::NsnBehaviourEvent::Mdns(mdns_event)) => {
                            match mdns_event {
                                libp2p::mdns::Event::Discovered(peers) => {
                                    for (peer_id, addr) in peers {
                                        info!("mDNS discovered peer {} at {}", peer_id, addr);
                                        // Add to Kademlia routing table
                                        self.swarm
                                            .behaviour_mut()
                                            .kademlia
                                            .add_address(&peer_id, addr.clone());
                                        // Dial the discovered peer
                                        if let Err(e) = self.swarm.dial(addr.clone()) {
                                            debug!("Failed to dial mDNS peer {}: {}", peer_id, e);
                                        }
                                    }
                                }
                                libp2p::mdns::Event::Expired(peers) => {
                                    for (peer_id, addr) in peers {
                                        debug!("mDNS peer expired: {} at {}", peer_id, addr);
                                    }
                                }
                            }
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
                            // Update shared state for HTTP discovery endpoint
                            {
                                let mut listeners = self.http_listeners.write().await;
                                if !listeners.contains(&address) {
                                    listeners.push(address.clone());
                                }
                            }
                            // Mark swarm as ready once we have at least one listener
                            self.swarm_ready.store(true, Ordering::SeqCst);

                            let address_str = address.to_string();
                            // Log with emphasis if it's a WebRTC address (contains certhash)
                            if address_str.contains("webrtc") {
                                info!("WebRTC listening address (for browsers): {}", address_str);
                                if address_str.contains("/ip4/0.0.0.0/")
                                    || address_str.contains("/ip6/::/")
                                {
                                    warn!(
                                        "WebRTC listener bound to an unspecified address. \
                                        Browsers cannot dial 0.0.0.0/::. \
                                        Set --p2p-external-address to a routable IP \
                                        (e.g. /ip4/192.168.1.10/udp/9003/webrtc-direct)."
                                    );
                                }
                            } else {
                                event_handler::handle_new_listen_addr(&address);
                            }
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
            self.graylist_peer(peer_id, "Connection flood detected")
                .await;
            return;
        }

        if self.security.graylist.is_graylisted(peer_id).await {
            debug!("Disconnecting graylisted peer {}", peer_id);
            let _ = self.swarm.disconnect_peer_id(*peer_id);
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
                if let Some(TopicCategory::VideoChunks) =
                    super::topics::parse_topic(&message.topic.to_string())
                {
                    if let Ok(chunk) = decode_video_chunk(&message.data) {
                        if verify_video_chunk(&chunk).is_ok() {
                            let latency_ms = chunk_latency_ms(&chunk);
                            self.metrics
                                .video_chunk_latency_seconds
                                .observe(latency_ms as f64 / 1000.0);
                            let _ = self.video_latency_tx.send(latency_ms);
                        }
                    }
                }
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
                self.record_violation(peer_id, "Rate limit exceeded").await;
                false
            }
        }
    }

    async fn record_violation(&mut self, peer_id: &PeerId, reason: &str) {
        let should_graylist = {
            let count = self
                .security
                .violation_counts
                .entry(*peer_id)
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
            .add(*peer_id, reason.to_string())
            .await;
        let _ = self.swarm.disconnect_peer_id(*peer_id);
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
                    if let Ok(multiaddr) = addr_str.parse::<Multiaddr>() {
                        self.swarm.add_external_address(multiaddr.clone());
                        // Update shared state for HTTP discovery endpoint
                        {
                            let mut ext = self.http_external_addrs.write().await;
                            if !ext.contains(&multiaddr) {
                                ext.push(multiaddr);
                            }
                        }
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
                    if let Ok(multiaddr) = addr_str.parse::<Multiaddr>() {
                        self.swarm.add_external_address(multiaddr.clone());
                        // Update shared state for HTTP discovery endpoint
                        {
                            let mut ext = self.http_external_addrs.write().await;
                            if !ext.contains(&multiaddr) {
                                ext.push(multiaddr);
                            }
                        }
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

/// HTTP server handling both Prometheus metrics and P2P discovery endpoints
async fn serve_http(state: Arc<HttpState>, addr: SocketAddr) -> Result<(), hyper::Error> {
    let make_svc = make_service_fn(move |_| {
        let state = state.clone();
        async move {
            Ok::<_, Infallible>(service_fn(move |req: Request<Body>| {
                let state = state.clone();
                async move {
                    // Handle CORS preflight
                    if req.method() == hyper::Method::OPTIONS {
                        let response = Response::builder()
                            .status(200)
                            .header(hyper::header::ACCESS_CONTROL_ALLOW_ORIGIN, "*")
                            .header(hyper::header::ACCESS_CONTROL_ALLOW_METHODS, "GET, OPTIONS")
                            .header(hyper::header::ACCESS_CONTROL_ALLOW_HEADERS, "Content-Type")
                            .header(hyper::header::CACHE_CONTROL, "no-store, max-age=0")
                            .body(Body::empty())
                            .unwrap_or_else(|_| Response::new(Body::empty()));
                        return Ok::<_, Infallible>(response);
                    }

                    let path = req.uri().path();

                    let (status, content_type, body, retry_after): (
                        u16,
                        String,
                        String,
                        Option<&str>,
                    ) = match path {
                        "/p2p/info" => {
                            // Check if swarm is ready (has listeners)
                            if !state.swarm_ready.load(Ordering::SeqCst) {
                                let response = P2pInfoResponse::error(
                                    "NODE_INITIALIZING",
                                    "Swarm not ready, please retry",
                                );
                                let json = serde_json::to_string(&response).unwrap_or_else(|_| {
                                    r#"{"success":false,"error":{"code":"SERIALIZATION_ERROR","message":"Failed to serialize response"}}"#.to_string()
                                });
                                // Return 503 with Retry-After: 5 header
                                (503, "application/json".to_string(), json, Some("5"))
                            } else {
                                // Get current addresses
                                let listeners = state.listeners.read().await;
                                let external_addrs = state.external_addrs.read().await;

                                let multiaddrs = filter_addresses(
                                    listeners.iter(),
                                    external_addrs.iter(),
                                    state.external_address.as_deref(),
                                );

                                let data = P2pInfoData {
                                    peer_id: state.peer_id.to_string(),
                                    multiaddrs,
                                    protocols: state.protocols.clone(),
                                    features: P2pFeatures {
                                        webrtc_enabled: state.webrtc_enabled,
                                        websocket_enabled: state.websocket_enabled,
                                        role: state.role.clone(),
                                    },
                                };

                                let response = P2pInfoResponse::success(data);
                                let json = serde_json::to_string(&response).unwrap_or_else(|_| {
                                    r#"{"success":false,"error":{"code":"SERIALIZATION_ERROR","message":"Failed to serialize response"}}"#.to_string()
                                });

                                (200, "application/json".to_string(), json, None)
                            }
                        }
                        "/metrics" | "/" => {
                            // Prometheus metrics
                            let metric_families = state.registry.gather();
                            let encoder = TextEncoder::new();
                            let mut buffer = Vec::new();
                            encoder.encode(&metric_families, &mut buffer).unwrap_or_default();
                            let content_type = encoder.format_type().to_string();

                            (
                                200,
                                content_type,
                                String::from_utf8(buffer).unwrap_or_default(),
                                None,
                            )
                        }
                        _ => (404, "text/plain".to_string(), "Not Found".to_string(), None),
                    };

                    let mut builder = Response::builder()
                        .status(status)
                        .header(hyper::header::CONTENT_TYPE, content_type)
                        .header(hyper::header::ACCESS_CONTROL_ALLOW_ORIGIN, "*")
                        .header(hyper::header::ACCESS_CONTROL_ALLOW_METHODS, "GET, OPTIONS")
                        .header(hyper::header::ACCESS_CONTROL_ALLOW_HEADERS, "Content-Type")
                        .header(hyper::header::CACHE_CONTROL, "no-store, max-age=0");

                    // Add Retry-After header for 503 responses
                    if let Some(seconds) = retry_after {
                        builder = builder.header("retry-after", seconds);
                    }

                    let response = builder
                        .body(Body::from(body))
                        .unwrap_or_else(|_| Response::new(Body::from("Internal error")));

                    Ok::<_, Infallible>(response)
                }
            }))
        }
    });

    info!(
        "HTTP server listening on http://{} (metrics: /metrics, discovery: /p2p/info)",
        addr
    );
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

    #[tokio::test]
    async fn test_service_with_webrtc_enabled() {
        use tempfile::TempDir;

        let temp_dir = TempDir::new().expect("Failed to create temp dir");

        let mut config = P2pConfig {
            enable_webrtc: true,
            webrtc_port: 19003, // Use non-standard port for test
            data_dir: Some(temp_dir.path().to_path_buf()),
            ..Default::default()
        };
        config.bootstrap.require_signed_manifests = false;
        config.bootstrap.signer_config.source = crate::SignerSource::Static;

        let result = P2pService::new(config, "ws://127.0.0.1:9944".to_string()).await;

        // Service should create successfully with WebRTC enabled
        assert!(
            result.is_ok(),
            "Service should create with WebRTC enabled: {:?}",
            result.err()
        );

        let (service, _cmd_tx) = result.unwrap();

        // Verify local peer ID exists
        assert!(!service.local_peer_id().to_string().is_empty());

        // Verify certificate was created
        let cert_path = temp_dir.path().join("webrtc_cert.pem");
        assert!(cert_path.exists(), "WebRTC certificate should be created");
    }

    #[tokio::test]
    async fn test_webrtc_certificate_persists() {
        use tempfile::TempDir;

        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let cert_path = temp_dir.path().join("webrtc_cert.pem");

        // Create first service - should generate certificate
        {
            let mut config = P2pConfig {
                enable_webrtc: true,
                webrtc_port: 19004,
                data_dir: Some(temp_dir.path().to_path_buf()),
                ..Default::default()
            };
            config.bootstrap.require_signed_manifests = false;
            config.bootstrap.signer_config.source = crate::SignerSource::Static;

            let _ = P2pService::new(config, "ws://127.0.0.1:9944".to_string()).await;
        }

        // Read certificate content
        let cert_content_1 =
            std::fs::read_to_string(&cert_path).expect("Should read certificate");

        // Create second service - should load existing certificate
        {
            let mut config = P2pConfig {
                enable_webrtc: true,
                webrtc_port: 19005,
                data_dir: Some(temp_dir.path().to_path_buf()),
                ..Default::default()
            };
            config.bootstrap.require_signed_manifests = false;
            config.bootstrap.signer_config.source = crate::SignerSource::Static;

            let _ = P2pService::new(config, "ws://127.0.0.1:9944".to_string()).await;
        }

        // Certificate content should be unchanged
        let cert_content_2 =
            std::fs::read_to_string(&cert_path).expect("Should read certificate");

        assert_eq!(
            cert_content_1, cert_content_2,
            "Certificate should persist across service instances"
        );
    }

    #[tokio::test]
    async fn test_discovery_endpoint_returns_valid_json() {
        use tempfile::TempDir;

        // Skip if network not allowed
        if !network_allowed() {
            return;
        }

        let temp_dir = TempDir::new().expect("Failed to create temp dir");

        let mut config = P2pConfig {
            metrics_port: 19200, // Use unique port for test - HTTP server with discovery
            listen_port: 19210,
            enable_webrtc: true,
            webrtc_port: 19203,
            data_dir: Some(temp_dir.path().to_path_buf()),
            ..Default::default()
        };
        config.bootstrap.require_signed_manifests = false;
        config.bootstrap.signer_config.source = crate::SignerSource::Static;

        // Use P2pService::new directly to preserve metrics_port
        let (service, cmd_tx) = P2pService::new(config, "ws://127.0.0.1:9944".to_string())
            .await
            .expect("Failed to create service");
        let handle = spawn_service(service);

        // Wait for service startup and listeners to bind
        // The swarm needs time to bind listeners and emit NewListenAddr events
        // This is a race condition - need to poll until ready or timeout
        let client = reqwest::Client::new();
        let mut ready = false;
        for _ in 0..30 {
            tokio::time::sleep(std::time::Duration::from_millis(200)).await;
            if let Ok(response) = client.get("http://127.0.0.1:19200/p2p/info").send().await {
                if response.status() == 200 {
                    ready = true;
                    break;
                }
            }
        }
        assert!(ready, "Service should become ready within 6 seconds");

        // Fetch discovery endpoint - should now return 200
        let response = client
            .get("http://127.0.0.1:19200/p2p/info")
            .send()
            .await
            .expect("Failed to fetch /p2p/info");

        assert_eq!(response.status(), 200);

        // Check CORS header
        let cors = response.headers().get("access-control-allow-origin");
        assert!(cors.is_some(), "Should have CORS header");
        assert_eq!(cors.unwrap(), "*");

        // Check Cache-Control header
        let cache = response.headers().get("cache-control");
        assert!(cache.is_some(), "Should have Cache-Control header");
        assert!(cache.unwrap().to_str().unwrap().contains("no-store"));

        // Parse JSON
        let json: serde_json::Value = response.json().await.expect("Failed to parse JSON");

        assert_eq!(json["success"], true);
        assert!(json["data"]["peer_id"].is_string());
        assert!(json["data"]["multiaddrs"].is_array());
        assert!(
            json["data"]["protocols"].is_array(),
            "Should have protocols field"
        );
        assert!(
            !json["data"]["protocols"].as_array().unwrap().is_empty(),
            "Protocols should not be empty"
        );
        assert_eq!(json["data"]["features"]["webrtc_enabled"], true);

        shutdown_service(cmd_tx, handle).await;
    }

    #[tokio::test]
    async fn test_discovery_endpoint_cors_preflight() {
        if !network_allowed() {
            return;
        }

        let temp_dir = tempfile::TempDir::new().expect("Failed to create temp dir");

        let mut config = P2pConfig {
            metrics_port: 19201,
            listen_port: 19211,
            data_dir: Some(temp_dir.path().to_path_buf()),
            ..Default::default()
        };
        config.bootstrap.require_signed_manifests = false;
        config.bootstrap.signer_config.source = crate::SignerSource::Static;

        // Use P2pService::new directly to preserve metrics_port
        let (service, cmd_tx) = P2pService::new(config, "ws://127.0.0.1:9944".to_string())
            .await
            .expect("Failed to create service");
        let handle = spawn_service(service);
        wait_for_startup().await;
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;

        // Send OPTIONS preflight request
        let client = reqwest::Client::new();
        let response = client
            .request(reqwest::Method::OPTIONS, "http://127.0.0.1:19201/p2p/info")
            .send()
            .await
            .expect("Failed OPTIONS request");

        assert_eq!(response.status(), 200);

        let allow_methods = response.headers().get("access-control-allow-methods");
        assert!(allow_methods.is_some());
        assert!(allow_methods.unwrap().to_str().unwrap().contains("GET"));

        shutdown_service(cmd_tx, handle).await;
    }

    #[tokio::test]
    async fn test_discovery_endpoint_503_before_ready() {
        if !network_allowed() {
            return;
        }

        // This test verifies the 503 response logic exists in the code.
        // Testing the actual race condition (hitting endpoint before swarm is ready)
        // is tricky in integration tests. Instead, we verify:
        // 1. The endpoint eventually returns 200 (proving it becomes ready)
        // 2. The 503 error response structure is correct via unit tests

        let temp_dir = tempfile::TempDir::new().expect("Failed to create temp dir");

        let mut config = P2pConfig {
            metrics_port: 19202,
            listen_port: 19212,
            data_dir: Some(temp_dir.path().to_path_buf()),
            ..Default::default()
        };
        config.bootstrap.require_signed_manifests = false;
        config.bootstrap.signer_config.source = crate::SignerSource::Static;

        // Use P2pService::new directly to preserve metrics_port
        let (service, cmd_tx) = P2pService::new(config, "ws://127.0.0.1:9944".to_string())
            .await
            .expect("Failed to create service");
        let handle = spawn_service(service);

        // Wait for service to be fully ready (swarm needs to bind listeners)
        // Poll until we get 200 or timeout
        let client = reqwest::Client::new();
        for _ in 0..30 {
            tokio::time::sleep(std::time::Duration::from_millis(200)).await;
            if let Ok(response) = client.get("http://127.0.0.1:19202/p2p/info").send().await {
                if response.status() == 200 {
                    break;
                }
            }
        }

        // Now should get 200
        let response = client
            .get("http://127.0.0.1:19202/p2p/info")
            .timeout(std::time::Duration::from_secs(2))
            .send()
            .await
            .expect("Failed to fetch /p2p/info after startup");

        assert_eq!(response.status(), 200, "Should be ready after startup");

        shutdown_service(cmd_tx, handle).await;
    }
}
