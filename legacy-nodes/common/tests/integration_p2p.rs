//! Integration tests for P2P networking
//!
//! Tests actual multi-node communication, connection establishment,
//! and P2P protocol behavior.

use icn_common::p2p::{P2pConfig, P2pService, ServiceCommand};
use libp2p::{Multiaddr, PeerId};
use std::time::Duration;
use tokio::sync::oneshot;
use tokio::time::timeout;

/// Test that two nodes can connect via QUIC and maintain connection
#[tokio::test]
async fn test_two_nodes_connect_via_quic() {
    // Initialize tracing for debugging
    let _ = tracing_subscriber::fmt::try_init();

    // Create node A with explicit port
    let config_a = P2pConfig {
        listen_port: 9001,
        keypair_path: None, // Ephemeral for tests
        ..Default::default()
    };

    let (mut service_a, cmd_tx_a) = P2pService::new(config_a)
        .await
        .expect("Failed to create service A");

    let peer_id_a = service_a.local_peer_id();

    // Create node B with different port
    let config_b = P2pConfig {
        listen_port: 9002,
        keypair_path: None,
        ..Default::default()
    };

    let (mut service_b, cmd_tx_b) = P2pService::new(config_b)
        .await
        .expect("Failed to create service B");

    let peer_id_b = service_b.local_peer_id();

    // Start both services in background tasks
    let handle_a = tokio::spawn(async move { service_a.start().await });

    let handle_b = tokio::spawn(async move { service_b.start().await });

    // Give services time to start listening
    tokio::time::sleep(Duration::from_millis(100)).await;

    // Node A dials Node B
    let multiaddr_b: Multiaddr = format!("/ip4/127.0.0.1/udp/9002/quic-v1/p2p/{}", peer_id_b)
        .parse()
        .expect("Failed to parse multiaddr");

    cmd_tx_a
        .send(ServiceCommand::Dial(multiaddr_b))
        .expect("Failed to send dial command");

    // Wait for connection to establish
    tokio::time::sleep(Duration::from_millis(500)).await;

    // Verify connections on both sides
    let (tx_a, rx_a) = oneshot::channel();
    cmd_tx_a
        .send(ServiceCommand::GetPeerCount(tx_a))
        .expect("Failed to query peer count A");

    let peer_count_a = timeout(Duration::from_secs(1), rx_a)
        .await
        .expect("Timeout waiting for peer count A")
        .expect("Failed to get peer count A");

    assert_eq!(
        peer_count_a, 1,
        "Node A should be connected to 1 peer (Node B)"
    );

    let (tx_b, rx_b) = oneshot::channel();
    cmd_tx_b
        .send(ServiceCommand::GetPeerCount(tx_b))
        .expect("Failed to query peer count B");

    let peer_count_b = timeout(Duration::from_secs(1), rx_b)
        .await
        .expect("Timeout waiting for peer count B")
        .expect("Failed to get peer count B");

    assert_eq!(
        peer_count_b, 1,
        "Node B should be connected to 1 peer (Node A)"
    );

    // Verify connection count
    let (tx_conn_a, rx_conn_a) = oneshot::channel();
    cmd_tx_a
        .send(ServiceCommand::GetConnectionCount(tx_conn_a))
        .expect("Failed to query connection count A");

    let conn_count_a = timeout(Duration::from_secs(1), rx_conn_a)
        .await
        .expect("Timeout waiting for connection count A")
        .expect("Failed to get connection count A");

    assert_eq!(conn_count_a, 1, "Node A should have 1 active connection");

    // Shutdown both nodes
    cmd_tx_a
        .send(ServiceCommand::Shutdown)
        .expect("Failed to shutdown A");
    cmd_tx_b
        .send(ServiceCommand::Shutdown)
        .expect("Failed to shutdown B");

    // Wait for graceful shutdown
    let _ = timeout(Duration::from_secs(2), handle_a).await;
    let _ = timeout(Duration::from_secs(2), handle_b).await;
}

/// Test connection timeout behavior
#[tokio::test]
async fn test_connection_timeout_after_inactivity() {
    let _ = tracing_subscriber::fmt::try_init();

    // Create config with very short timeout for testing
    let config_a = P2pConfig {
        listen_port: 9003,
        keypair_path: None,
        connection_timeout: Duration::from_secs(2), // Short timeout for test
        ..Default::default()
    };

    let (mut service_a, cmd_tx_a) = P2pService::new(config_a)
        .await
        .expect("Failed to create service A");

    let peer_id_a = service_a.local_peer_id();

    let config_b = P2pConfig {
        listen_port: 9004,
        keypair_path: None,
        connection_timeout: Duration::from_secs(2),
        ..Default::default()
    };

    let (mut service_b, cmd_tx_b) = P2pService::new(config_b)
        .await
        .expect("Failed to create service B");

    let peer_id_b = service_b.local_peer_id();

    // Start both services
    let handle_a = tokio::spawn(async move { service_a.start().await });

    let handle_b = tokio::spawn(async move { service_b.start().await });

    tokio::time::sleep(Duration::from_millis(100)).await;

    // Node A dials Node B
    let multiaddr_b: Multiaddr = format!("/ip4/127.0.0.1/udp/9004/quic-v1/p2p/{}", peer_id_b)
        .parse()
        .expect("Failed to parse multiaddr");

    cmd_tx_a
        .send(ServiceCommand::Dial(multiaddr_b))
        .expect("Failed to send dial command");

    // Wait for connection
    tokio::time::sleep(Duration::from_millis(500)).await;

    // Verify connection exists
    let (tx, rx) = oneshot::channel();
    cmd_tx_a
        .send(ServiceCommand::GetPeerCount(tx))
        .expect("Failed to query peer count");

    let peer_count = timeout(Duration::from_secs(1), rx)
        .await
        .expect("Timeout")
        .expect("Failed to get peer count");

    assert_eq!(peer_count, 1, "Connection should be established");

    // Wait for timeout (2s timeout + 1s buffer)
    tokio::time::sleep(Duration::from_secs(3)).await;

    // Verify connection is closed due to timeout
    let (tx, rx) = oneshot::channel();
    cmd_tx_a
        .send(ServiceCommand::GetPeerCount(tx))
        .expect("Failed to query peer count");

    let peer_count_after = timeout(Duration::from_secs(1), rx)
        .await
        .expect("Timeout")
        .expect("Failed to get peer count");

    assert_eq!(
        peer_count_after, 0,
        "Connection should timeout after inactivity"
    );

    // Cleanup
    cmd_tx_a
        .send(ServiceCommand::Shutdown)
        .expect("Failed to shutdown A");
    cmd_tx_b
        .send(ServiceCommand::Shutdown)
        .expect("Failed to shutdown B");

    let _ = timeout(Duration::from_secs(2), handle_a).await;
    let _ = timeout(Duration::from_secs(2), handle_b).await;
}

/// Test that multiple nodes can form a mesh network
#[tokio::test]
async fn test_multiple_nodes_mesh() {
    let _ = tracing_subscriber::fmt::try_init();

    // Create 3 nodes
    let config_a = P2pConfig {
        listen_port: 9005,
        keypair_path: None,
        ..Default::default()
    };
    let (mut service_a, cmd_tx_a) = P2pService::new(config_a).await.unwrap();
    let peer_id_a = service_a.local_peer_id();

    let config_b = P2pConfig {
        listen_port: 9006,
        keypair_path: None,
        ..Default::default()
    };
    let (mut service_b, cmd_tx_b) = P2pService::new(config_b).await.unwrap();
    let peer_id_b = service_b.local_peer_id();

    let config_c = P2pConfig {
        listen_port: 9007,
        keypair_path: None,
        ..Default::default()
    };
    let (mut service_c, cmd_tx_c) = P2pService::new(config_c).await.unwrap();
    let peer_id_c = service_c.local_peer_id();

    // Start all services
    let handle_a = tokio::spawn(async move { service_a.start().await });
    let handle_b = tokio::spawn(async move { service_b.start().await });
    let handle_c = tokio::spawn(async move { service_c.start().await });

    tokio::time::sleep(Duration::from_millis(100)).await;

    // Connect A -> B
    let multiaddr_b: Multiaddr = format!("/ip4/127.0.0.1/udp/9006/quic-v1/p2p/{}", peer_id_b)
        .parse()
        .unwrap();
    cmd_tx_a.send(ServiceCommand::Dial(multiaddr_b)).unwrap();

    // Connect A -> C
    let multiaddr_c: Multiaddr = format!("/ip4/127.0.0.1/udp/9007/quic-v1/p2p/{}", peer_id_c)
        .parse()
        .unwrap();
    cmd_tx_a.send(ServiceCommand::Dial(multiaddr_c)).unwrap();

    // Wait for connections
    tokio::time::sleep(Duration::from_millis(500)).await;

    // Verify A has 2 peers
    let (tx, rx) = oneshot::channel();
    cmd_tx_a.send(ServiceCommand::GetPeerCount(tx)).unwrap();
    let count = timeout(Duration::from_secs(1), rx).await.unwrap().unwrap();
    assert_eq!(count, 2, "Node A should be connected to 2 peers");

    // Cleanup
    cmd_tx_a.send(ServiceCommand::Shutdown).unwrap();
    cmd_tx_b.send(ServiceCommand::Shutdown).unwrap();
    cmd_tx_c.send(ServiceCommand::Shutdown).unwrap();

    let _ = timeout(Duration::from_secs(2), handle_a).await;
    let _ = timeout(Duration::from_secs(2), handle_b).await;
    let _ = timeout(Duration::from_secs(2), handle_c).await;
}

/// Test graceful shutdown closes all connections
#[tokio::test]
async fn test_graceful_shutdown_closes_connections() {
    let _ = tracing_subscriber::fmt::try_init();

    let config_a = P2pConfig {
        listen_port: 9008,
        keypair_path: None,
        ..Default::default()
    };
    let (mut service_a, cmd_tx_a) = P2pService::new(config_a).await.unwrap();
    let peer_id_a = service_a.local_peer_id();

    let config_b = P2pConfig {
        listen_port: 9009,
        keypair_path: None,
        ..Default::default()
    };
    let (mut service_b, cmd_tx_b) = P2pService::new(config_b).await.unwrap();
    let peer_id_b = service_b.local_peer_id();

    let handle_a = tokio::spawn(async move { service_a.start().await });
    let handle_b = tokio::spawn(async move { service_b.start().await });

    tokio::time::sleep(Duration::from_millis(100)).await;

    // Connect A -> B
    let multiaddr_b: Multiaddr = format!("/ip4/127.0.0.1/udp/9009/quic-v1/p2p/{}", peer_id_b)
        .parse()
        .unwrap();
    cmd_tx_a.send(ServiceCommand::Dial(multiaddr_b)).unwrap();

    tokio::time::sleep(Duration::from_millis(500)).await;

    // Verify connection
    let (tx, rx) = oneshot::channel();
    cmd_tx_a
        .send(ServiceCommand::GetConnectionCount(tx))
        .unwrap();
    let count = timeout(Duration::from_secs(1), rx).await.unwrap().unwrap();
    assert_eq!(count, 1, "Should have 1 connection before shutdown");

    // Shutdown A
    cmd_tx_a.send(ServiceCommand::Shutdown).unwrap();

    // Wait for shutdown
    tokio::time::sleep(Duration::from_millis(500)).await;

    // Verify B sees disconnection
    let (tx, rx) = oneshot::channel();
    cmd_tx_b.send(ServiceCommand::GetPeerCount(tx)).unwrap();
    let count = timeout(Duration::from_secs(1), rx).await.unwrap().unwrap();
    assert_eq!(
        count, 0,
        "Node B should see peer disconnected after A shutdown"
    );

    // Cleanup
    cmd_tx_b.send(ServiceCommand::Shutdown).unwrap();

    let _ = timeout(Duration::from_secs(2), handle_a).await;
    let _ = timeout(Duration::from_secs(2), handle_b).await;
}
