//! RelayNode - Main orchestrator for ICN Regional Relay
//!
//! Coordinates all components: cache, P2P, QUIC server, health checker

use crate::{
    cache::ShardCache,
    config::Config,
    error::Result,
    health_check::HealthChecker,
    latency_detector, metrics,
    p2p_service::P2PService,
    quic_server::{QuicServer, QuicServerConfig},
    upstream_client::UpstreamClient,
};
use std::sync::Arc;
use tokio::sync::Mutex;
use tracing::{error, info};

/// Main relay node orchestrator
pub struct RelayNode {
    config: Config,
    region: String,
    cache: Arc<Mutex<ShardCache>>,
    quic_server: QuicServer,
}

impl RelayNode {
    /// Create new relay node with configuration
    pub async fn new(config: Config, region_override: Option<String>) -> Result<Self> {
        config.validate()?;

        // Detect or use override region
        let region = detect_region(&config, region_override).await;

        // Initialize cache
        info!("Initializing shard cache...");
        let cache = Arc::new(Mutex::new(
            ShardCache::new(config.cache_path.clone(), config.max_cache_gb).await?,
        ));

        // Initialize upstream QUIC client
        info!("Initializing upstream client...");
        let dev_mode = is_dev_mode();
        let upstream_client = Arc::new(UpstreamClient::new(dev_mode)?);

        // Initialize P2P service and publish availability
        info!("Initializing P2P service...");
        let (p2p_service, p2p_rx) = P2PService::new(&config).await?;
        spawn_p2p_service(p2p_service, p2p_rx, &region, config.quic_port);

        // Start health checker
        spawn_health_checker(&config);

        // Initialize QUIC server
        info!("Starting QUIC server on port {}...", config.quic_port);
        let quic_config = create_quic_config(dev_mode);
        let quic_server = QuicServer::new(
            config.quic_port,
            Arc::clone(&cache),
            Arc::clone(&upstream_client),
            config.super_node_addresses.clone(),
            100, // Global connection rate: 100/s
            10,  // Per-IP connection rate: 10/s
            quic_config,
        )
        .await?;

        Ok(Self {
            config,
            region,
            cache,
            quic_server,
        })
    }

    /// Run the relay node (blocks until shutdown)
    pub async fn run(self) -> Result<()> {
        // Start metrics server
        spawn_metrics_server(self.config.metrics_port);

        // Setup graceful shutdown handler
        let shutdown_handler = spawn_shutdown_handler(Arc::clone(&self.cache));

        // Run QUIC server (blocks)
        info!("Regional Relay Node running (region: {})", self.region);
        tokio::select! {
            result = self.quic_server.run() => {
                if let Err(e) = result {
                    error!("QUIC server error: {}", e);
                }
            }
            _ = shutdown_handler => {
                info!("Graceful shutdown complete");
            }
        }

        Ok(())
    }
}

/// Detect region from config or auto-detect
async fn detect_region(config: &Config, override_region: Option<String>) -> String {
    if let Some(region) = override_region {
        info!("Region override: {}", region);
        return region;
    }

    if !config.region.is_empty() {
        info!("Region from config: {}", config.region);
        return config.region.clone();
    }

    info!("Auto-detecting region...");
    let super_node_pairs: Vec<(String, String)> = config
        .super_node_addresses
        .iter()
        .map(|addr| {
            let region = latency_detector::extract_region_from_address(addr);
            (addr.clone(), region)
        })
        .collect();

    match latency_detector::detect_region(&super_node_pairs, 3).await {
        Ok(detected_region) => {
            info!("Detected region: {}", detected_region);
            detected_region
        }
        Err(e) => {
            error!("Region detection failed: {}", e);
            error!("Falling back to default region: UNKNOWN");
            "UNKNOWN".to_string()
        }
    }
}

/// Check if running in development mode
fn is_dev_mode() -> bool {
    std::env::var("DEV_MODE").unwrap_or_else(|_| "false".to_string()) == "true"
}

/// Create QUIC server configuration with auth
fn create_quic_config(dev_mode: bool) -> QuicServerConfig {
    if dev_mode {
        info!("DEV MODE: Viewer authentication DISABLED");
        return QuicServerConfig::no_auth();
    }

    let auth_tokens = load_auth_tokens();
    if auth_tokens.is_empty() {
        warn_no_auth_tokens();
        QuicServerConfig::no_auth()
    } else {
        info!(
            "Viewer authentication ENABLED ({} tokens)",
            auth_tokens.len()
        );
        QuicServerConfig::with_auth(auth_tokens)
    }
}

/// Load authentication tokens from environment
fn load_auth_tokens() -> Vec<String> {
    std::env::var("AUTH_TOKENS")
        .unwrap_or_default()
        .split(',')
        .map(|s| s.to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

/// Warn about disabled authentication
fn warn_no_auth_tokens() {
    error!(
        "WARNING: No AUTH_TOKENS provided - viewer authentication DISABLED. \
         This is INSECURE and acceptable only for testnet. \
         For production deployment, set AUTH_TOKENS environment variable."
    );
}

/// Spawn P2P service with event handling
fn spawn_p2p_service(
    mut p2p_service: P2PService,
    mut p2p_rx: tokio::sync::mpsc::UnboundedReceiver<crate::p2p_service::P2PEvent>,
    region: &str,
    quic_port: u16,
) {
    let relay_multiaddr = format!("/ip4/0.0.0.0/tcp/{}", quic_port);
    if let Err(e) = p2p_service.publish_relay_availability(region, relay_multiaddr) {
        error!("Failed to publish relay availability: {}", e);
    }

    tokio::spawn(async move {
        if let Err(e) = p2p_service.run().await {
            error!("P2P service error: {}", e);
        }
    });

    tokio::spawn(async move {
        while let Some(event) = p2p_rx.recv().await {
            info!("P2P Event: {:?}", event);
        }
    });
}

/// Spawn health checker for Super-Nodes
fn spawn_health_checker(config: &Config) {
    info!("Starting Super-Node health checker...");
    let mut health_checker = HealthChecker::new(
        config.super_node_addresses.clone(),
        config.health_check_secs,
    );
    tokio::spawn(async move {
        health_checker.run().await;
    });
}

/// Spawn metrics HTTP server
fn spawn_metrics_server(port: u16) {
    info!("Starting metrics server on port {}...", port);
    tokio::spawn(async move {
        if let Err(e) = metrics::start_metrics_server(port).await {
            error!("Metrics server error: {}", e);
        }
    });
}

/// Spawn graceful shutdown handler
fn spawn_shutdown_handler(cache: Arc<Mutex<ShardCache>>) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        tokio::signal::ctrl_c()
            .await
            .expect("Failed to listen for Ctrl+C");
        info!("Shutdown signal received, flushing cache...");

        if let Err(e) = cache.lock().await.save_manifest().await {
            error!("Failed to save cache manifest: {}", e);
        } else {
            info!("Cache manifest saved successfully");
        }
    })
}
