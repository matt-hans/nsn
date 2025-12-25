//! ICN Director Node
//!
//! GPU-powered video generation node with BFT coordination.
//!
//! Responsibilities:
//! - Monitor chain for election events via subxt
//! - Schedule video generation with lookahead (slot + 2)
//! - Exchange CLIP embeddings via gRPC
//! - Compute 3-of-5 BFT consensus
//! - Submit results to chain
//! - Maintain P2P connectivity
//! - Interface with Python Vortex engine via PyO3

mod bft_coordinator;
mod chain_client;
mod config;
mod election_monitor;
mod error;
mod keystore;
mod metrics;
mod p2p_service;
mod slot_scheduler;
mod types;
mod vortex_bridge;

use bft_coordinator::BftCoordinator;
use chain_client::ChainClient;
use config::Config;
use election_monitor::ElectionMonitor;
use error::Result;
use keystore::Keystore;
use metrics::Metrics;
use p2p_service::P2pService;
use slot_scheduler::SlotScheduler;
use vortex_bridge::VortexBridge;

use clap::Parser;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::signal;
use tokio::sync::RwLock;
use tracing::{error, info};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

/// Director node CLI arguments
#[derive(Parser, Debug)]
#[command(name = "icn-director")]
#[command(about = "ICN Director Node - GPU-powered video generation with BFT coordination")]
struct Cli {
    /// Path to configuration file
    #[arg(short, long, default_value = "config/director.toml")]
    config: PathBuf,

    /// Override chain endpoint
    #[arg(long)]
    chain_endpoint: Option<String>,

    /// Override keypair path
    #[arg(long)]
    keypair: Option<PathBuf>,
}

/// Director node application state
struct DirectorNode {
    config: Config,
    metrics: Metrics,
    chain_client: Arc<ChainClient>,
    _election_monitor: ElectionMonitor,
    _slot_scheduler: Arc<RwLock<SlotScheduler>>,
    _bft_coordinator: BftCoordinator,
    p2p_service: P2pService,
    _vortex_bridge: VortexBridge,
}

impl DirectorNode {
    async fn new(config: Config) -> Result<Self> {
        info!("Initializing Director Node");

        // Validate configuration
        config.validate()?;

        // Initialize metrics
        let metrics = Metrics::new()?;

        // Initialize chain client
        let chain_client = Arc::new(ChainClient::connect(config.chain_endpoint.clone()).await?);

        // Load keypair from file
        let keystore = Keystore::load(&config.keypair_path)?;
        let own_peer_id = keystore.peer_id().to_string();

        info!("Node PeerId: {}", own_peer_id);

        // Initialize election monitor
        let _election_monitor = ElectionMonitor::new(own_peer_id.clone());

        // Initialize slot scheduler
        let _slot_scheduler = Arc::new(RwLock::new(SlotScheduler::new(config.pipeline_lookahead)));

        // Initialize BFT coordinator
        let _bft_coordinator =
            BftCoordinator::new(own_peer_id.clone(), config.bft_consensus_threshold);

        // Initialize P2P service
        let mut p2p_service = P2pService::new(own_peer_id.clone()).await?;
        p2p_service.start().await?;

        // Initialize Vortex bridge
        let _vortex_bridge = VortexBridge::initialize()?;

        info!("Director Node initialized successfully");

        Ok(Self {
            config,
            metrics,
            chain_client,
            _election_monitor,
            _slot_scheduler,
            _bft_coordinator,
            p2p_service,
            _vortex_bridge,
        })
    }

    async fn run(&mut self) -> Result<()> {
        info!("Starting Director Node main loop");

        // Start metrics server
        let metrics_addr = format!("0.0.0.0:{}", self.config.metrics_port);
        let metrics_registry = Arc::new(self.metrics.registry().clone());
        tokio::spawn(async move {
            if let Err(e) = start_metrics_server(&metrics_addr, metrics_registry).await {
                error!("Metrics server error: {}", e);
            }
        });

        // Main event loop
        loop {
            tokio::select! {
                // Monitor chain for blocks and events
                _ = tokio::time::sleep(tokio::time::Duration::from_secs(6)) => {
                    if let Ok(block) = self.chain_client.get_latest_block().await {
                        self.metrics.chain_latest_block.set(block as f64);
                    }
                }

                // Update P2P peer count metric
                _ = tokio::time::sleep(tokio::time::Duration::from_secs(10)) => {
                    let peer_count = self.p2p_service.peer_count();
                    self.metrics.connected_peers.set(peer_count as f64);
                }

                // Handle shutdown signal
                _ = signal::ctrl_c() => {
                    info!("Received shutdown signal");
                    break;
                }
            }
        }

        info!("Director Node shutting down");
        Ok(())
    }
}

/// Start Prometheus metrics HTTP server (STUB - simplified for MVP)
async fn start_metrics_server(addr: &str, _registry: Arc<prometheus::Registry>) -> Result<()> {
    // TODO: Implement full Prometheus HTTP server with hyper 1.0 API
    // For now, just log that metrics would be served
    info!("Metrics server would listen on http://{} (STUB)", addr);

    // Keep task alive
    loop {
        tokio::time::sleep(tokio::time::Duration::from_secs(3600)).await;
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::registry()
        .with(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| "info,icn_director=debug".into()),
        )
        .with(tracing_subscriber::fmt::layer().with_target(true))
        .init();

    info!("ICN Director Node starting...");

    // Parse CLI arguments
    let cli = Cli::parse();

    // Load configuration
    let mut config = Config::load(&cli.config)?;

    // Override with CLI arguments
    if let Some(endpoint) = cli.chain_endpoint {
        config.chain_endpoint = endpoint;
    }
    if let Some(keypair) = cli.keypair {
        config.keypair_path = keypair;
    }

    info!("Configuration loaded from {:?}", cli.config);
    info!("Chain endpoint: {}", config.chain_endpoint);
    info!("gRPC port: {}", config.grpc_port);
    info!("Metrics port: {}", config.metrics_port);
    info!("Region: {}", config.region);

    // Create and run director node
    let mut node = DirectorNode::new(config).await?;
    node.run().await?;

    info!("ICN Director Node stopped");
    Ok(())
}
