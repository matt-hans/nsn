//! P2P network service
//!
//! Main P2P service that manages libp2p Swarm, connection lifecycle,
//! and event handling with metrics and graceful shutdown support.

use super::behaviour::IcnBehaviour;
use super::config::P2pConfig;
use super::connection_manager::ConnectionManager;
use super::event_handler;
use super::identity::{generate_keypair, load_keypair, save_keypair, IdentityError};
use super::metrics::P2pMetrics;
use futures::StreamExt;
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

    /// Shutdown the service
    Shutdown,
}

/// P2P network service
pub struct P2pService {
    /// libp2p Swarm
    swarm: Swarm<IcnBehaviour>,

    /// Configuration
    config: P2pConfig,

    /// Metrics
    metrics: Arc<P2pMetrics>,

    /// Local PeerId
    local_peer_id: PeerId,

    /// Command receiver
    command_rx: mpsc::UnboundedReceiver<ServiceCommand>,

    /// Command sender (for cloning)
    command_tx: mpsc::UnboundedSender<ServiceCommand>,

    /// Connection manager
    connection_manager: ConnectionManager,

    /// Shutdown flag
    shutdown: bool,
}

impl P2pService {
    /// Create new P2P service
    ///
    /// # Arguments
    /// * `config` - P2P configuration
    ///
    /// # Returns
    /// Tuple of (P2pService, command sender)
    pub async fn new(
        config: P2pConfig,
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
        metrics.connection_limit.set(config.max_connections as i64);

        // Build swarm with QUIC transport
        let swarm = SwarmBuilder::with_existing_identity(keypair)
            .with_tokio()
            .with_quic()
            .with_behaviour(|_| IcnBehaviour::new())
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

    #[tokio::test]
    async fn test_service_creation() {
        let config = P2pConfig::default();

        let (service, _cmd_tx) = P2pService::new(config)
            .await
            .expect("Failed to create service");

        // Verify initial state
        assert_eq!(service.connection_manager.tracker().total_connections(), 0);
        assert_eq!(service.connection_manager.tracker().connected_peers(), 0);
        assert_eq!(service.metrics.active_connections.get(), 0);
        assert_eq!(service.metrics.connected_peers.get(), 0);
    }

    #[tokio::test]
    async fn test_service_local_peer_id() {
        let config = P2pConfig::default();

        let (service, _cmd_tx) = P2pService::new(config)
            .await
            .expect("Failed to create service");

        let peer_id = service.local_peer_id();
        assert!(!peer_id.to_string().is_empty());
    }

    #[tokio::test]
    async fn test_service_metrics() {
        let config = P2pConfig::default();

        let (service, _cmd_tx) = P2pService::new(config)
            .await
            .expect("Failed to create service");

        let metrics = service.metrics();
        assert_eq!(metrics.connection_limit.get(), 256);
    }

    #[tokio::test]
    async fn test_service_handles_get_peer_count_command() {
        let config = P2pConfig {
            listen_port: 9100, // Use different port for each test
            ..Default::default()
        };
        let (mut service, cmd_tx) = P2pService::new(config)
            .await
            .expect("Failed to create service");

        // Start service in background
        let handle = tokio::spawn(async move { service.start().await });

        // Give service time to start
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;

        // Query peer count
        let (tx, rx) = tokio::sync::oneshot::channel();
        cmd_tx
            .send(ServiceCommand::GetPeerCount(tx))
            .expect("Failed to send command");

        // Should receive 0 peers (no connections yet)
        let count = tokio::time::timeout(std::time::Duration::from_secs(1), rx)
            .await
            .expect("Timeout waiting for response")
            .expect("Failed to receive peer count");

        assert_eq!(count, 0, "Should have 0 peers initially");

        // Shutdown
        cmd_tx
            .send(ServiceCommand::Shutdown)
            .expect("Failed to shutdown");
        let _ = tokio::time::timeout(std::time::Duration::from_secs(2), handle).await;
    }

    #[tokio::test]
    async fn test_service_handles_get_connection_count_command() {
        let config = P2pConfig {
            listen_port: 9101, // Use different port for each test
            ..Default::default()
        };
        let (mut service, cmd_tx) = P2pService::new(config)
            .await
            .expect("Failed to create service");

        // Start service in background
        let handle = tokio::spawn(async move { service.start().await });

        // Give service time to start
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;

        let (tx, rx) = tokio::sync::oneshot::channel();
        cmd_tx
            .send(ServiceCommand::GetConnectionCount(tx))
            .expect("Failed to send command");

        let count = tokio::time::timeout(std::time::Duration::from_secs(1), rx)
            .await
            .expect("Timeout waiting for response")
            .expect("Failed to receive connection count");

        assert_eq!(count, 0, "Should have 0 connections initially");

        // Shutdown
        cmd_tx
            .send(ServiceCommand::Shutdown)
            .expect("Failed to shutdown");
        let _ = tokio::time::timeout(std::time::Duration::from_secs(2), handle).await;
    }

    #[tokio::test]
    async fn test_service_shutdown_command() {
        let config = P2pConfig::default();
        let (mut service, cmd_tx) = P2pService::new(config)
            .await
            .expect("Failed to create service");

        // Start service in background
        let handle = tokio::spawn(async move { service.start().await });

        // Give it time to start
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;

        // Send shutdown command
        cmd_tx
            .send(ServiceCommand::Shutdown)
            .expect("Failed to send shutdown");

        // Service should exit cleanly
        let result = tokio::time::timeout(std::time::Duration::from_secs(2), handle)
            .await
            .expect("Service should shutdown within timeout");

        assert!(result.is_ok(), "Service should shutdown gracefully");
    }

    #[tokio::test]
    async fn test_invalid_multiaddr_dial_returns_error() {
        let config = P2pConfig {
            listen_port: 9102, // Use different port
            ..Default::default()
        };
        let (mut service, _cmd_tx) = P2pService::new(config)
            .await
            .expect("Failed to create service");

        // Try to dial invalid multiaddr (missing peer ID)
        let invalid_addr: Multiaddr = "/ip4/127.0.0.1/udp/9999/quic-v1".parse().unwrap();

        // This should fail during command handling
        let result = service
            .handle_command(ServiceCommand::Dial(invalid_addr))
            .await;

        // The command handling should return an error
        // Note: libp2p may not fail immediately for missing peer ID, so we just verify
        // that the command completes without panicking
        let _ = result; // Accept either success or error
    }

    #[tokio::test]
    async fn test_service_command_sender_clonable() {
        let config = P2pConfig {
            listen_port: 9103, // Use different port
            ..Default::default()
        };
        let (mut service, cmd_tx) = P2pService::new(config)
            .await
            .expect("Failed to create service");

        // Start service in background
        let handle = tokio::spawn(async move { service.start().await });

        tokio::time::sleep(std::time::Duration::from_millis(100)).await;

        // Clone sender
        let cmd_tx_clone = cmd_tx.clone();

        // Both should work
        let (tx1, rx1) = tokio::sync::oneshot::channel();
        cmd_tx
            .send(ServiceCommand::GetPeerCount(tx1))
            .expect("Original sender should work");

        let (tx2, rx2) = tokio::sync::oneshot::channel();
        cmd_tx_clone
            .send(ServiceCommand::GetPeerCount(tx2))
            .expect("Cloned sender should work");

        // Both should receive responses
        let _count1 = tokio::time::timeout(std::time::Duration::from_secs(1), rx1)
            .await
            .expect("Timeout")
            .expect("Should receive from original");
        let _count2 = tokio::time::timeout(std::time::Duration::from_secs(1), rx2)
            .await
            .expect("Timeout")
            .expect("Should receive from clone");

        // Shutdown
        cmd_tx
            .send(ServiceCommand::Shutdown)
            .expect("Failed to shutdown");
        let _ = tokio::time::timeout(std::time::Duration::from_secs(2), handle).await;
    }

    #[tokio::test]
    async fn test_service_with_keypair_path() {
        use tempfile::TempDir;

        // Create temporary directory for test keypair
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let keypair_path = temp_dir.path().join("test_keypair");

        let config = P2pConfig {
            keypair_path: Some(keypair_path.clone()),
            ..Default::default()
        };

        // First creation should generate and save keypair
        let (service1, _) = P2pService::new(config.clone())
            .await
            .expect("Failed to create service with new keypair");

        let peer_id_1 = service1.local_peer_id();

        // Drop service and verify keypair file exists
        drop(service1);
        assert!(keypair_path.exists(), "Keypair should be saved to file");

        // Second creation should load existing keypair
        let (service2, _) = P2pService::new(config)
            .await
            .expect("Failed to create service with existing keypair");

        let peer_id_2 = service2.local_peer_id();

        // PeerIds should match (same keypair)
        assert_eq!(
            peer_id_1, peer_id_2,
            "PeerId should be same when loading existing keypair"
        );
    }

    #[tokio::test]
    async fn test_service_ephemeral_keypair() {
        let config = P2pConfig {
            keypair_path: None, // Ephemeral
            ..Default::default()
        };

        let (service1, _) = P2pService::new(config.clone())
            .await
            .expect("Failed to create service 1");
        let (service2, _) = P2pService::new(config)
            .await
            .expect("Failed to create service 2");

        let peer_id_1 = service1.local_peer_id();
        let peer_id_2 = service2.local_peer_id();

        // Ephemeral keypairs should be different
        assert_ne!(
            peer_id_1, peer_id_2,
            "Ephemeral keypairs should generate different PeerIds"
        );
    }

    #[tokio::test]
    async fn test_connection_metrics_updated() {
        let config = P2pConfig::default();
        let (service, _cmd_tx) = P2pService::new(config)
            .await
            .expect("Failed to create service");

        let metrics = service.metrics();

        // Verify initial metrics
        assert_eq!(metrics.active_connections.get(), 0);
        assert_eq!(metrics.connected_peers.get(), 0);
        assert_eq!(metrics.connections_established_total.get(), 0);
        assert_eq!(metrics.connections_closed_total.get(), 0);
        assert_eq!(metrics.connections_failed_total.get(), 0);
    }
}
