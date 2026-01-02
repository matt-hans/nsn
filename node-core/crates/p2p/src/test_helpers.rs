//! Test utilities and helpers
//!
//! Common test setup, teardown, and helper functions to reduce duplication
//! across test modules.

use super::behaviour::NsnBehaviour;
use super::config::P2pConfig;
use super::connection_manager::ConnectionManager;
use super::metrics::P2pMetrics;
use super::service::{P2pService, ServiceCommand};
use libp2p::{identity::Keypair, Swarm, SwarmBuilder};
use std::sync::Arc;
use tokio::sync::mpsc;

/// Default RPC URL for tests (will fail to connect but that's fine for P2P tests)
pub const TEST_RPC_URL: &str = "ws://localhost:9944";

/// Default timeout for async operations in tests
pub const TEST_TIMEOUT_SECS: u64 = 2;

/// Service startup delay (allows event loop to initialize)
pub const TEST_STARTUP_DELAY_MS: u64 = 100;

/// Returns true if network sockets can be opened in this environment.
pub fn network_allowed() -> bool {
    std::net::UdpSocket::bind("127.0.0.1:0").is_ok()
}

/// Create default test config
pub fn test_config() -> P2pConfig {
    let mut config = P2pConfig::default();
    config.bootstrap.require_signed_manifests = false;
    config.bootstrap.signer_config.source = super::bootstrap::SignerSource::Static;
    config.metrics_port = 0;
    config
}

/// Create test config with custom listen port
pub fn test_config_with_port(port: u16) -> P2pConfig {
    P2pConfig {
        listen_port: port,
        ..Default::default()
    }
}

/// Create test config with custom connection limits
pub fn test_config_with_limits(max_connections: usize, max_per_peer: usize) -> P2pConfig {
    P2pConfig {
        max_connections,
        max_connections_per_peer: max_per_peer,
        ..Default::default()
    }
}

/// Create test P2P service
pub async fn create_test_service() -> (P2pService, mpsc::UnboundedSender<ServiceCommand>) {
    create_test_service_with_config(test_config()).await
}

/// Create test P2P service with custom config
pub async fn create_test_service_with_config(
    config: P2pConfig,
) -> (P2pService, mpsc::UnboundedSender<ServiceCommand>) {
    let mut config = config;
    config.bootstrap.require_signed_manifests = false;
    config.bootstrap.signer_config.source = super::bootstrap::SignerSource::Static;
    config.metrics_port = 0;
    P2pService::new(config, TEST_RPC_URL.to_string())
        .await
        .expect("Failed to create test service")
}

/// Create test P2P service with custom port
pub async fn create_test_service_with_port(
    port: u16,
) -> (P2pService, mpsc::UnboundedSender<ServiceCommand>) {
    create_test_service_with_config(test_config_with_port(port)).await
}

/// Start service in background and return join handle
pub fn spawn_service(
    mut service: P2pService,
) -> tokio::task::JoinHandle<Result<(), super::service::ServiceError>> {
    tokio::spawn(async move { service.start().await })
}

/// Wait for service to start up
pub async fn wait_for_startup() {
    tokio::time::sleep(std::time::Duration::from_millis(TEST_STARTUP_DELAY_MS)).await;
}

/// Shutdown service gracefully with timeout
pub async fn shutdown_service(
    cmd_tx: mpsc::UnboundedSender<ServiceCommand>,
    handle: tokio::task::JoinHandle<Result<(), super::service::ServiceError>>,
) {
    cmd_tx
        .send(ServiceCommand::Shutdown)
        .expect("Failed to send shutdown command");

    let _ = tokio::time::timeout(std::time::Duration::from_secs(TEST_TIMEOUT_SECS), handle).await;
}

/// Query peer count from service
pub async fn query_peer_count(cmd_tx: &mpsc::UnboundedSender<ServiceCommand>) -> usize {
    let (tx, rx) = tokio::sync::oneshot::channel();
    cmd_tx
        .send(ServiceCommand::GetPeerCount(tx))
        .expect("Failed to send GetPeerCount command");

    tokio::time::timeout(std::time::Duration::from_secs(TEST_TIMEOUT_SECS), rx)
        .await
        .expect("Timeout waiting for peer count")
        .expect("Failed to receive peer count")
}

/// Query connection count from service
pub async fn query_connection_count(cmd_tx: &mpsc::UnboundedSender<ServiceCommand>) -> usize {
    let (tx, rx) = tokio::sync::oneshot::channel();
    cmd_tx
        .send(ServiceCommand::GetConnectionCount(tx))
        .expect("Failed to send GetConnectionCount command");

    tokio::time::timeout(std::time::Duration::from_secs(TEST_TIMEOUT_SECS), rx)
        .await
        .expect("Timeout waiting for connection count")
        .expect("Failed to receive connection count")
}

/// Create test connection manager with default config
pub fn create_test_connection_manager() -> (ConnectionManager, Arc<P2pMetrics>) {
    let config = test_config();
    let metrics = Arc::new(P2pMetrics::new().expect("Failed to create test metrics"));
    let manager = ConnectionManager::new(config, metrics.clone());
    (manager, metrics)
}

/// Create test connection manager with custom config
pub fn create_test_connection_manager_with_config(
    config: P2pConfig,
) -> (ConnectionManager, Arc<P2pMetrics>) {
    let metrics = Arc::new(P2pMetrics::new().expect("Failed to create test metrics"));
    let manager = ConnectionManager::new(config, metrics.clone());
    (manager, metrics)
}

/// Create test swarm with NsnBehaviour
pub fn create_test_swarm(keypair: &Keypair) -> Swarm<NsnBehaviour> {
    SwarmBuilder::with_existing_identity(keypair.clone())
        .with_tokio()
        .with_quic()
        .with_behaviour(|_| NsnBehaviour::new_for_testing(keypair))
        .expect("Failed to create test swarm")
        .build()
}

/// Generate test keypair
pub fn generate_test_keypair() -> Keypair {
    libp2p::identity::Keypair::generate_ed25519()
}

/// Assert service initial state is correct
pub fn assert_service_initial_state(service: &P2pService) {
    assert_eq!(
        service.connection_manager.tracker().total_connections(),
        0,
        "Initial total connections should be 0"
    );
    assert_eq!(
        service.connection_manager.tracker().connected_peers(),
        0,
        "Initial connected peers should be 0"
    );
    assert_eq!(
        service.metrics.active_connections.get(),
        0.0,
        "Initial active_connections metric should be 0"
    );
    assert_eq!(
        service.metrics.connected_peers.get(),
        0.0,
        "Initial connected_peers metric should be 0"
    );
}

/// Assert metrics are at initial state
pub fn assert_metrics_initial_state(metrics: &P2pMetrics) {
    assert_eq!(
        metrics.active_connections.get(),
        0.0,
        "active_connections should be 0"
    );
    assert_eq!(
        metrics.connected_peers.get(),
        0.0,
        "connected_peers should be 0"
    );
    assert_eq!(
        metrics.connections_established_total.get(),
        0.0,
        "connections_established_total should be 0"
    );
    assert_eq!(
        metrics.connections_closed_total.get(),
        0.0,
        "connections_closed_total should be 0"
    );
    assert_eq!(
        metrics.connections_failed_total.get(),
        0.0,
        "connections_failed_total should be 0"
    );
}
