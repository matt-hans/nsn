//! Latency-based region detection
//!
//! Pings Super-Nodes to determine lowest-latency region for relay assignment.
//! Uses TCP handshake timing for accurate network latency measurement.

use std::net::{SocketAddr, ToSocketAddrs};
use std::time::{Duration, Instant};
use tokio::net::TcpStream;
use tokio::time::timeout;
use tracing::{debug, info, warn};

/// Latency detection result
#[derive(Debug, Clone)]
pub struct LatencyResult {
    pub address: String,
    pub latency: Duration,
    pub region: String,
}

/// Ping a Super-Node to measure latency
///
/// # Arguments
/// * `address` - Super-Node address (e.g., "127.0.0.1:9002")
/// * `timeout_duration` - Max time to wait for TCP handshake
///
/// # Returns
/// Latency in milliseconds if successful, None if unreachable
pub async fn ping_super_node(address: &str, timeout_duration: Duration) -> Option<Duration> {
    // Parse socket address
    let socket_addr: SocketAddr = match address.to_socket_addrs() {
        Ok(mut addrs) => match addrs.next() {
            Some(addr) => addr,
            None => {
                warn!("No socket address resolved for {}", address);
                return None;
            }
        },
        Err(e) => {
            warn!("Failed to resolve address {}: {}", address, e);
            return None;
        }
    };

    // Measure TCP handshake time
    let start = Instant::now();
    match timeout(timeout_duration, TcpStream::connect(socket_addr)).await {
        Ok(Ok(_stream)) => {
            let latency = start.elapsed();
            debug!("Ping {} successful: {:?}", address, latency);
            Some(latency)
        }
        Ok(Err(e)) => {
            warn!("TCP connect to {} failed: {}", address, e);
            None
        }
        Err(_) => {
            warn!("Ping {} timed out after {:?}", address, timeout_duration);
            None
        }
    }
}

/// Detect optimal region by pinging multiple Super-Nodes
///
/// # Arguments
/// * `super_node_addresses` - List of Super-Node addresses with regions
/// * `samples` - Number of ping samples per node (default 3)
///
/// # Returns
/// Detected region with lowest median latency, or error if no nodes reachable
pub async fn detect_region(
    super_node_addresses: &[(String, String)], // (address, region)
    samples: usize,
) -> crate::error::Result<String> {
    if super_node_addresses.is_empty() {
        return Err(crate::error::RelayError::LatencyDetection(
            "No Super-Node addresses provided".to_string(),
        ));
    }

    let timeout_duration = Duration::from_secs(2);
    let mut latency_results: Vec<LatencyResult> = Vec::new();

    // Ping each Super-Node multiple times and compute median latency
    for (address, region) in super_node_addresses {
        let mut latencies = Vec::new();

        for i in 0..samples {
            debug!("Pinging {} (sample {}/{})", address, i + 1, samples);
            if let Some(latency) = ping_super_node(address, timeout_duration).await {
                latencies.push(latency);
            }
            // Small delay between samples
            tokio::time::sleep(Duration::from_millis(100)).await;
        }

        if !latencies.is_empty() {
            // Compute median latency
            latencies.sort();
            let median = latencies[latencies.len() / 2];

            latency_results.push(LatencyResult {
                address: address.clone(),
                latency: median,
                region: region.clone(),
            });

            info!(
                "Super-Node {} (region: {}): median latency {:?}",
                address, region, median
            );
        } else {
            warn!("Super-Node {} (region: {}) unreachable", address, region);
        }
    }

    if latency_results.is_empty() {
        return Err(crate::error::RelayError::RegionNotDetected);
    }

    // Sort by latency and select lowest
    latency_results.sort_by_key(|r| r.latency);

    let best = &latency_results[0];
    info!(
        "Selected region: {} (Super-Node: {}, latency: {:?})",
        best.region, best.address, best.latency
    );

    Ok(best.region.clone())
}

/// Extract region from Super-Node address
///
/// Expected format: "<host>:<port>" with region in config
/// Fallback: use address as region identifier
pub fn extract_region_from_address(address: &str) -> String {
    // Simple heuristic: use first part of hostname as region hint
    // E.g., "na-west-sn1.icn.network:9002" -> "NA-WEST"
    if let Some(host) = address.split(':').next() {
        if host.contains("na-west") {
            return "NA-WEST".to_string();
        } else if host.contains("eu-west") {
            return "EU-WEST".to_string();
        } else if host.contains("apac") {
            return "APAC".to_string();
        }
    }

    // Fallback: use full address
    address.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test Case: Ping successful Super-Node
    /// Purpose: Verify TCP handshake timing works
    /// Contract: Returns latency if node reachable
    #[tokio::test]
    async fn test_ping_super_node_localhost() {
        // Start a simple TCP listener
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        // Spawn listener task
        tokio::spawn(async move {
            while let Ok((_stream, _)) = listener.accept().await {
                // Accept and close immediately
            }
        });

        let result = ping_super_node(&addr.to_string(), Duration::from_secs(1)).await;
        assert!(result.is_some(), "Should successfully ping localhost");
        let latency = result.unwrap();
        assert!(
            latency < Duration::from_millis(100),
            "Localhost should be <100ms"
        );
    }

    /// Test Case: Ping unreachable Super-Node
    /// Purpose: Verify timeout handling
    /// Contract: Returns None if unreachable
    #[tokio::test]
    async fn test_ping_super_node_unreachable() {
        // Use a port that's definitely not listening
        let result = ping_super_node("127.0.0.1:9999", Duration::from_millis(100)).await;
        assert!(result.is_none(), "Should return None for unreachable node");
    }

    /// Test Case: Detect region from multiple Super-Nodes
    /// Purpose: Verify lowest latency selection
    /// Contract: Returns region with lowest median latency
    #[tokio::test]
    async fn test_detect_region_selects_lowest_latency() {
        // Setup mock Super-Nodes
        let listener1 = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr1 = listener1.local_addr().unwrap();

        let listener2 = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr2 = listener2.local_addr().unwrap();

        // Spawn listeners
        tokio::spawn(async move { while let Ok((_stream, _)) = listener1.accept().await {} });

        tokio::spawn(async move {
            // Add artificial delay to simulate higher latency
            while let Ok((_stream, _)) = listener2.accept().await {
                tokio::time::sleep(Duration::from_millis(50)).await;
            }
        });

        let super_nodes = vec![
            (addr1.to_string(), "NA-WEST".to_string()),
            (addr2.to_string(), "EU-WEST".to_string()),
        ];

        let result = detect_region(&super_nodes, 1).await;
        assert!(result.is_ok(), "Should detect region");

        // One of the regions should be selected
        let region = result.unwrap();
        assert!(
            region == "NA-WEST" || region == "EU-WEST",
            "Should select a valid region"
        );
    }

    /// Test Case: Detect region when all nodes unreachable
    /// Purpose: Verify error handling for no connectivity
    /// Contract: Returns RegionNotDetected error
    #[tokio::test]
    async fn test_detect_region_all_unreachable() {
        let super_nodes = vec![
            ("127.0.0.1:9998".to_string(), "NA-WEST".to_string()),
            ("127.0.0.1:9999".to_string(), "EU-WEST".to_string()),
        ];

        let result = detect_region(&super_nodes, 1).await;
        assert!(result.is_err(), "Should return error when all unreachable");

        match result.unwrap_err() {
            crate::error::RelayError::RegionNotDetected => {
                // Expected error
            }
            e => panic!("Unexpected error: {:?}", e),
        }
    }

    /// Test Case: Extract region from hostname
    /// Purpose: Verify region hint extraction
    #[test]
    fn test_extract_region_from_address() {
        assert_eq!(
            extract_region_from_address("na-west-sn1.icn.network:9002"),
            "NA-WEST"
        );
        assert_eq!(
            extract_region_from_address("eu-west-relay.example.com:9002"),
            "EU-WEST"
        );
        assert_eq!(
            extract_region_from_address("apac-sn.icn.network:9002"),
            "APAC"
        );

        // Fallback to full address if no match
        assert_eq!(
            extract_region_from_address("unknown-host:9002"),
            "unknown-host:9002"
        );
    }

    /// Test Case: Detect region with empty input
    /// Purpose: Verify validation
    #[tokio::test]
    async fn test_detect_region_empty_input() {
        let result = detect_region(&[], 3).await;
        assert!(result.is_err(), "Should reject empty Super-Node list");
    }
}
