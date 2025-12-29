//! ICN Super-Node Binary
//!
//! Responsibilities:
//! - Store erasure-coded video shards (Reed-Solomon 10+4)
//! - Respond to pinning audits from pallet-icn-pinning
//! - Relay content to Regional Relays via QUIC
//! - Publish shard manifests to Kademlia DHT

use clap::Parser;
use icn_super_node::{
    audit_monitor::AuditMonitor, chain_client::ChainClient, config::Config, erasure::ErasureCoder,
    metrics, p2p_service::P2PService, quic_server::QuicServer, storage::Storage,
    storage_cleanup::StorageCleanup,
};
use std::sync::Arc;
use tracing::{debug, error, info};

#[derive(Parser)]
#[command(name = "icn-super-node")]
#[command(about = "ICN Super-Node - Tier 1 storage and relay", long_about = None)]
struct Cli {
    /// Path to configuration file
    #[arg(short, long, default_value = "config/super-node.toml")]
    config: String,

    /// Storage root path (overrides config)
    #[arg(long)]
    storage_path: Option<String>,

    /// Geographic region (overrides config)
    #[arg(long)]
    region: Option<String>,

    /// Chain RPC endpoint (overrides config)
    #[arg(long)]
    chain_endpoint: Option<String>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_target(false)
        .with_level(true)
        .init();

    info!("ICN Super-Node starting...");

    // Parse CLI args
    let cli = Cli::parse();

    // Load configuration
    let mut config = Config::load(&cli.config)?;

    // Override with CLI args if provided
    if let Some(storage_path) = cli.storage_path {
        config.storage_path = storage_path.into();
    }
    if let Some(region) = cli.region {
        config.region = region;
    }
    if let Some(chain_endpoint) = cli.chain_endpoint {
        config.chain_endpoint = chain_endpoint;
    }

    // Validate configuration
    config.validate()?;

    info!(
        "Configuration loaded: region={}, storage_path={:?}, chain_endpoint={}",
        config.region, config.storage_path, config.chain_endpoint
    );

    // Initialize components
    let erasure_coder = Arc::new(ErasureCoder::new()?);
    let storage = Arc::new(Storage::new(config.storage_path.clone()));

    // Connect to chain
    let (chain_client, chain_rx) = ChainClient::connect(config.chain_endpoint.clone()).await?;
    let chain_client = Arc::new(chain_client);

    // Initialize P2P service
    let (mut p2p_service, mut p2p_rx) = P2PService::new(&config).await?;

    // Initialize QUIC server
    let quic_server = QuicServer::new(config.quic_port, config.storage_path.clone()).await?;

    // Initialize audit monitor
    let audit_monitor = AuditMonitor::new(
        config.audit_poll_secs,
        Arc::clone(&chain_client),
        Arc::clone(&storage),
        chain_rx,
    );

    // Initialize storage cleanup
    let storage_cleanup = StorageCleanup::new(
        config.cleanup_interval_blocks,
        Arc::clone(&chain_client),
        Arc::clone(&storage),
    );

    // Start metrics server
    metrics::start_metrics_server(config.metrics_port).await?;

    info!("All services initialized successfully");
    info!("QUIC server listening on port {}", config.quic_port);
    info!(
        "Metrics available at http://localhost:{}/metrics",
        config.metrics_port
    );

    // Subscribe to GossipSub topics
    p2p_service.subscribe_video_topic().await?;

    // Spawn P2P event loop
    tokio::spawn(async move {
        if let Err(e) = p2p_service.run().await {
            error!("P2P service failed: {}", e);
        }
    });

    // Spawn QUIC server
    tokio::spawn(async move {
        if let Err(e) = quic_server.run().await {
            error!("QUIC server failed: {}", e);
        }
    });

    // Spawn chain client
    let chain_client_handle = Arc::clone(&chain_client);
    tokio::spawn(async move {
        if let Err(e) = chain_client_handle.run().await {
            error!("Chain client failed: {}", e);
        }
    });

    // Handle P2P events (spawn before background tasks to capture variables)
    let event_erasure_coder = Arc::clone(&erasure_coder);
    let event_storage = Arc::clone(&storage);

    tokio::spawn(async move {
        while let Some(event) = p2p_rx.recv().await {
            match event {
                icn_super_node::p2p_service::P2PEvent::VideoChunkReceived { slot, data } => {
                    info!(
                        "Received video chunk for slot {}: {} bytes",
                        slot,
                        data.len()
                    );

                    // Process video chunk: encode, store, publish manifest
                    match process_video_chunk(&event_erasure_coder, &event_storage, slot, data)
                        .await
                    {
                        Ok(cid) => {
                            info!("Video chunk processed successfully: CID={}", cid);
                        }
                        Err(e) => {
                            error!("Failed to process video chunk: {}", e);
                        }
                    }
                }
                icn_super_node::p2p_service::P2PEvent::PeerConnected(peer_id) => {
                    info!("Peer connected: {}", peer_id);
                }
                icn_super_node::p2p_service::P2PEvent::PeerDisconnected(peer_id) => {
                    info!("Peer disconnected: {}", peer_id);
                }
            }
        }
    });

    // Start background tasks
    tokio::select! {
        result = audit_monitor.start() => {
            if let Err(e) = result {
                error!("Audit monitor failed: {}", e);
            }
        }
        result = storage_cleanup.start() => {
            if let Err(e) = result {
                error!("Storage cleanup failed: {}", e);
            }
        }
        _ = tokio::signal::ctrl_c() => {
            info!("Received shutdown signal");
        }
    }

    info!("Super-Node shutting down gracefully...");
    Ok(())
}

/// Process video chunk: encode with Reed-Solomon, store shards, update metrics
///
/// # Arguments
/// * `erasure_coder` - Reed-Solomon encoder
/// * `storage` - Shard storage manager
/// * `slot` - Slot number for video chunk
/// * `data` - Raw video chunk bytes
///
/// # Returns
/// CID of stored content
async fn process_video_chunk(
    erasure_coder: &ErasureCoder,
    storage: &Storage,
    slot: u64,
    data: Vec<u8>,
) -> anyhow::Result<String> {
    let data_len = data.len();

    info!(
        "Processing video chunk for slot {}: {} bytes",
        slot, data_len
    );

    // Step 1: Encode with Reed-Solomon (10+4)
    let shards = erasure_coder.encode(&data)?;
    info!(
        "Encoded video chunk into {} shards ({} bytes each)",
        shards.len(),
        shards.first().map(|s| s.len()).unwrap_or(0)
    );

    // Step 2: Store shards to disk
    let cid = storage.store_shards(&data, shards.clone()).await?;
    info!("Stored {} shards for CID: {}", shards.len(), cid);

    // Step 3: Update metrics
    let total_shard_bytes: usize = shards.iter().map(|s| s.len()).sum();
    metrics::SHARD_COUNT.add(shards.len() as i64);
    metrics::BYTES_STORED.add(total_shard_bytes as i64);

    info!(
        "Video chunk processed: slot={}, cid={}, shards={}, bytes_stored={}",
        slot,
        cid,
        shards.len(),
        total_shard_bytes
    );

    // Step 4: TODO - Publish shard manifest to DHT
    // This requires P2P service reference, which we don't have in this context
    // The P2P service will need to be refactored to accept manifest publishing requests
    // via a channel or shared state. For now, we log that this step is needed.
    debug!(
        "TODO: Publish shard manifest to DHT for CID {} (requires P2P service refactor)",
        cid
    );

    Ok(cid)
}
