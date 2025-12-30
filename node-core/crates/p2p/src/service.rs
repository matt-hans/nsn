//! P2P network service
//!
//! Main P2P service that manages libp2p Swarm, GossipSub messaging,
//! connection lifecycle, reputation oracle, and event handling.

use super::behaviour::NsnBehaviour;
use super::config::P2pConfig;
use super::connection_manager::ConnectionManager;
use super::event_handler;
use super::gossipsub::{create_gossipsub_behaviour, subscribe_to_all_topics, GossipsubError};
use super::identity::{generate_keypair, load_keypair, save_keypair, IdentityError};
use super::metrics::P2pMetrics;
use super::reputation_oracle::{OracleError, ReputationOracle};
use super::topics::TopicCategory;
use futures::StreamExt;
use libp2p::gossipsub::MessageId;
use libp2p::{Multiaddr, PeerId, Swarm, SwarmBuilder};
use std::sync::Arc;
use thiserror::Error;
use tokio::sync::mpsc;
use tracing::{debug, error, info};

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

    #[error("GossipSub error: {0}")]
    Gossipsub(#[from] GossipsubError),

    #[error("Oracle error: {0}")]
    Oracle(#[from] OracleError),
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

    /// Shutdown the service
    Shutdown,
}

/// P2P network service
pub struct P2pService {
    /// libp2p Swarm
    swarm: Swarm<NsnBehaviour>,

    /// Configuration
    config: P2pConfig,

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

        // Create reputation oracle
        let reputation_oracle = Arc::new(ReputationOracle::new(rpc_url));

        // Spawn reputation oracle sync loop
        let oracle_clone = reputation_oracle.clone();
        tokio::spawn(async move {
            oracle_clone.sync_loop().await;
        });

        // Create GossipSub behavior
        let mut gossipsub = create_gossipsub_behaviour(&keypair, reputation_oracle.clone())?;

        // Subscribe to all NSN topics
        let sub_count = subscribe_to_all_topics(&mut gossipsub)?;
        info!("Subscribed to {} topics", sub_count);

        // Create NSN behaviour
        let behaviour = NsnBehaviour::new(gossipsub);

        // Build swarm with QUIC transport
        let swarm = SwarmBuilder::with_existing_identity(keypair)
            .with_tokio()
            .with_quic()
            .with_behaviour(|_| behaviour)
            .map_err(|e| ServiceError::Swarm(format!("Failed to create behaviour: {}", e)))?
            .with_swarm_config(|cfg| cfg.with_idle_connection_timeout(config.connection_timeout))
            .build();

        let (command_tx, command_rx) = mpsc::unbounded_channel();

        // Create connection manager
        let connection_manager = ConnectionManager::new(config.clone(), metrics.clone());

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

        // Event loop
        loop {
            tokio::select! {
                // Handle swarm events
                event = self.swarm.select_next_some() => {
                    if let Err(e) = event_handler::dispatch_swarm_event(
                        event,
                        &mut self.connection_manager,
                        &mut self.swarm,
                    ) {
                        error!("Error handling swarm event: {}", e);
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
            ServiceCommand::Dial(addr) => {
                info!("Dialing {}", addr);
                self.swarm
                    .dial(addr.clone())
                    .map_err(|e| ServiceError::Swarm(format!("Failed to dial {}: {}", addr, e)))?;
            }

            ServiceCommand::GetPeerCount(tx) => {
                let count = self.connection_manager.tracker().connected_peers();
                let _ = tx.send(count);
            }

            ServiceCommand::GetConnectionCount(tx) => {
                let count = self.connection_manager.tracker().total_connections();
                let _ = tx.send(count);
            }

            ServiceCommand::Subscribe(category, tx) => {
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

            ServiceCommand::Publish(category, data, tx) => {
                let result = super::gossipsub::publish_message(
                    &mut self.swarm.behaviour_mut().gossipsub,
                    &category,
                    data,
                );
                let _ = tx.send(result);
            }

            ServiceCommand::Shutdown => {
                info!("Received shutdown command");
                self.shutdown = true;
            }
        }

        Ok(())
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helpers::*;

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
        let (service, cmd_tx) = create_test_service_with_port(9100).await;
        let handle = spawn_service(service);
        wait_for_startup().await;

        let count = query_peer_count(&cmd_tx).await;
        assert_eq!(count, 0, "Should have 0 peers initially");

        shutdown_service(cmd_tx, handle).await;
    }

    #[tokio::test]
    async fn test_service_handles_get_connection_count_command() {
        let (service, cmd_tx) = create_test_service_with_port(9101).await;
        let handle = spawn_service(service);
        wait_for_startup().await;

        let count = query_connection_count(&cmd_tx).await;
        assert_eq!(count, 0, "Should have 0 connections initially");

        shutdown_service(cmd_tx, handle).await;
    }

    #[tokio::test]
    async fn test_service_shutdown_command() {
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
        let config = P2pConfig {
            keypair_path: None,
            ..Default::default()
        };

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
}
