//! NSN Unified Node Binary
//!
//! This binary runs in one of four modes:
//! - SuperNode: Full capabilities (Director + Validator + Storage)
//! - DirectorOnly: Generation capabilities only
//! - ValidatorOnly: CLIP verification only
//! - StorageOnly: Pinning and distribution only

use anyhow::{anyhow, Result};
use clap::{Parser, Subcommand, ValueEnum};
use nsn_chain_client::{ChainAttestationSubmitter, ChainExecutorRegistry};
use nsn_p2p::{
    build_video_chunks, build_video_chunks_from_ivf, publish_video_chunks, P2pConfig, P2pService,
    VideoChunkConfig, DEFAULT_CHUNK_SIZE_BYTES,
};
use nsn_scheduler::{
    AttestationSubmitter, DualAttestationSubmitter, NoopAttestationSubmitter,
    P2pAttestationSubmitter, RedundancyConfig, RedundantScheduler,
};
use nsn_sidecar::proto::sidecar_server::SidecarServer;
use nsn_sidecar::{PluginPolicy, SidecarService, SidecarServiceConfig, TaskCompletionEvent};
use nsn_storage::{StorageAuditReport, StorageBackendKind, StorageManager};
use nsn_types::NodeCapability;
use sp_core::{sr25519, Pair};
use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;
use tonic::transport::Server;
use tracing::{error, info, warn, Level};
use tracing_subscriber::FmtSubscriber;

#[derive(Parser)]
#[command(name = "nsn-node")]
#[command(about = "NSN Unified Off-Chain Node", long_about = None)]
struct Cli {
    /// Node mode
    #[command(subcommand)]
    mode: Mode,

    /// Configuration file path
    #[arg(short, long, default_value = "config.toml")]
    config: String,

    /// Log level (trace, debug, info, warn, error)
    #[arg(short, long, default_value = "info")]
    log_level: String,

    /// Enable JSON logging
    #[arg(long)]
    json_logs: bool,

    /// RPC URL for NSN chain (used by P2P reputation oracle)
    #[arg(long, default_value = "ws://127.0.0.1:9944")]
    rpc_url: String,

    /// P2P listen port
    #[arg(long, default_value = "9000")]
    p2p_listen_port: u16,

    /// P2P metrics port
    #[arg(long, default_value = "9100")]
    p2p_metrics_port: u16,

    /// Executor registry refresh interval (seconds)
    #[arg(long, default_value = "30")]
    executor_registry_refresh_secs: u64,

    /// Attestation submission mode (p2p, chain, dual, none)
    #[arg(long, value_enum, default_value = "p2p")]
    attestation_submit_mode: AttestationSubmitMode,

    /// Attestation signer SURI (required for chain/dual modes)
    #[arg(long)]
    attestation_suri: Option<String>,

    /// P2P identity keypair path (used for signing video chunks)
    #[arg(long)]
    p2p_keypair_path: Option<PathBuf>,

    /// Enable WebRTC transport for browser connections
    #[arg(long)]
    p2p_enable_webrtc: bool,

    /// UDP port for WebRTC connections (default: 9003)
    #[arg(long, default_value = "9003")]
    p2p_webrtc_port: u16,

    /// Enable WebSocket transport for browser connections (via Tailscale, etc.)
    #[arg(long)]
    p2p_enable_websocket: bool,

    /// TCP port for WebSocket connections (default: 9004)
    #[arg(long, default_value = "9004")]
    p2p_websocket_port: u16,

    /// External address to advertise for WebRTC (for NAT/Docker)
    /// Format: /ip4/1.2.3.4/udp/9003/webrtc-direct
    #[arg(long)]
    p2p_external_address: Option<String>,

    /// Data directory for persistent state (certificates, etc.)
    #[arg(long, default_value = "/var/lib/nsn")]
    data_dir: PathBuf,

    /// Storage backend (local or ipfs)
    #[arg(long, value_enum, default_value = "ipfs")]
    storage_backend: StorageBackendMode,

    /// IPFS API URL (for ipfs storage backend)
    #[arg(long, default_value = "http://127.0.0.1:5001")]
    ipfs_api_url: String,

    /// Storage path (for local storage backend)
    #[arg(long, default_value = "/var/lib/nsn/storage")]
    storage_path: String,

    /// Optional CID to publish as a video stream over P2P
    #[arg(long)]
    publish_video_cid: Option<String>,

    /// Slot number to associate with published video
    #[arg(long, default_value = "0")]
    publish_slot: u64,

    /// Chunk size in bytes for video distribution
    #[arg(long, default_value_t = DEFAULT_CHUNK_SIZE_BYTES)]
    publish_chunk_size: usize,

    /// Keyframe interval in chunks (0 = only first chunk is keyframe)
    #[arg(long, default_value = "0")]
    publish_keyframe_interval: u32,

    /// Timeout for publish ack (ms)
    #[arg(long, default_value = "5000")]
    publish_ack_timeout_ms: u64,
}

#[derive(Subcommand)]
enum Mode {
    /// Run as SuperNode (Director + Validator + Storage)
    SuperNode {
        /// GPU device index (if multiple GPUs)
        #[arg(long, default_value = "0")]
        gpu_device: u8,
    },
    /// Run as Director-only node
    DirectorOnly {
        /// GPU device index (if multiple GPUs)
        #[arg(long, default_value = "0")]
        gpu_device: u8,
    },
    /// Run as Validator-only node
    ValidatorOnly {
        /// GPU device index (if multiple GPUs)
        #[arg(long, default_value = "0")]
        gpu_device: u8,
    },
    /// Run as Storage-only node
    StorageOnly {
        /// Storage path
        #[arg(long, default_value = "/var/lib/nsn/storage")]
        storage_path: String,
    },
    /// Run as a latency receiver for video chunks
    LatencyReceiver {
        /// Rolling window size for latency percentiles
        #[arg(long, default_value = "512")]
        window_size: usize,
        /// Reporting interval (ms)
        #[arg(long, default_value = "2000")]
        report_every_ms: u64,
    },
}

#[derive(Clone, Debug, ValueEnum)]
enum AttestationSubmitMode {
    P2p,
    Chain,
    Dual,
    None,
}

#[derive(Clone, Debug, ValueEnum)]
enum StorageBackendMode {
    Local,
    Ipfs,
}

impl Mode {
    fn to_node_mode(&self) -> Option<NodeCapability> {
        match self {
            Mode::SuperNode { .. } => Some(NodeCapability::SuperNode),
            Mode::DirectorOnly { .. } => Some(NodeCapability::DirectorOnly),
            Mode::ValidatorOnly { .. } => Some(NodeCapability::ValidatorOnly),
            Mode::StorageOnly { .. } => Some(NodeCapability::StorageOnly),
            Mode::LatencyReceiver { .. } => None,
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    // Initialize logging
    let log_level = match cli.log_level.to_lowercase().as_str() {
        "trace" => Level::TRACE,
        "debug" => Level::DEBUG,
        "info" => Level::INFO,
        "warn" => Level::WARN,
        "error" => Level::ERROR,
        _ => Level::INFO,
    };

    let subscriber = FmtSubscriber::builder()
        .with_max_level(log_level)
        .with_target(true)
        .with_thread_ids(true)
        .with_line_number(true)
        .finish();

    tracing::subscriber::set_global_default(subscriber)?;

    let latency_config = match &cli.mode {
        Mode::LatencyReceiver {
            window_size,
            report_every_ms,
        } => Some((*window_size, *report_every_ms)),
        _ => None,
    };

    let node_mode = cli.mode.to_node_mode();

    info!("Starting NSN Node");
    if let Some(mode) = node_mode {
        info!("Mode: {:?}", mode);
    } else {
        info!("Mode: LatencyReceiver");
    }
    info!("Config: {}", cli.config);

    // TODO: Load configuration from file
    // TODO: Initialize components based on mode
    // TODO: Start main event loop

    // Initialize P2P networking
    let mut p2p_config = P2pConfig::default();
    p2p_config.listen_port = cli.p2p_listen_port;
    p2p_config.metrics_port = cli.p2p_metrics_port;
    p2p_config.keypair_path = cli.p2p_keypair_path.clone();
    p2p_config.enable_webrtc = cli.p2p_enable_webrtc;
    p2p_config.webrtc_port = cli.p2p_webrtc_port;
    p2p_config.enable_websocket = cli.p2p_enable_websocket;
    p2p_config.websocket_port = cli.p2p_websocket_port;
    p2p_config.data_dir = Some(cli.data_dir.clone());
    p2p_config.external_address = cli.p2p_external_address.clone();

    if cli.p2p_enable_webrtc {
        info!(
            "WebRTC transport enabled on UDP port {}, data dir: {:?}",
            cli.p2p_webrtc_port, cli.data_dir
        );
        if let Some(ref addr) = cli.p2p_external_address {
            info!("External address for WebRTC: {}", addr);
        }
    }

    if cli.p2p_enable_websocket {
        info!(
            "WebSocket transport enabled on TCP port {}",
            cli.p2p_websocket_port
        );
    }

    let (mut p2p_service, p2p_cmd_tx) = P2pService::new(p2p_config, cli.rpc_url.clone()).await?;
    let mut latency_rx = latency_config.map(|_| p2p_service.subscribe_video_latency());
    tokio::spawn(async move {
        if let Err(err) = p2p_service.start().await {
            tracing::error!("P2P service failed: {}", err);
        }
    });

    if let Some((window_size, report_every_ms)) = latency_config {
        let rx = latency_rx
            .take()
            .ok_or_else(|| anyhow!("latency receiver not initialized"))?;
        run_latency_receiver(rx, window_size, Duration::from_millis(report_every_ms)).await?;
        return Ok(());
    }

    let mut attestation_submitter =
        Some(build_attestation_submitter(&cli, p2p_cmd_tx.clone()).await?);
    info!("Attestation submit mode: {:?}", cli.attestation_submit_mode);

    let registry_refresh = Duration::from_secs(cli.executor_registry_refresh_secs);

    let storage_path = match &cli.mode {
        Mode::StorageOnly { storage_path } => storage_path.clone(),
        _ => cli.storage_path.clone(),
    };

    let storage = build_storage_manager(
        cli.storage_backend.clone(),
        storage_path.clone(),
        cli.ipfs_api_url.clone(),
    )?;

    if let Some(cid) = cli.publish_video_cid.as_ref() {
        let report = publish_video_from_storage(
            &storage,
            &p2p_cmd_tx,
            cid,
            cli.publish_slot,
            cli.p2p_keypair_path.as_ref(),
            cli.publish_chunk_size,
            cli.publish_keyframe_interval,
            cli.publish_ack_timeout_ms,
        )
        .await?;
        info!(
            total_chunks = report.total_chunks,
            published = report.published,
            failed = report.failed,
            max_ack_ms = report.max_ack_ms,
            avg_ack_ms = report.avg_ack_ms,
            "Video distribution completed"
        );
    }

    match cli.mode {
        Mode::SuperNode { gpu_device } => {
            info!("SuperNode mode: GPU device {}", gpu_device);
            // TODO: Initialize scheduler, sidecar, storage
            let executor_registry =
                Arc::new(ChainExecutorRegistry::new(cli.rpc_url.clone(), registry_refresh).await?);
            let _executor_registry_task = executor_registry.clone().start_refresh();
            let submitter = attestation_submitter
                .take()
                .ok_or_else(|| anyhow!("attestation submitter already consumed"))?;
            let _redundant_scheduler =
                RedundantScheduler::new(RedundancyConfig::default(), submitter);
            let storage_root = PathBuf::from(storage_path);
            if matches!(storage.backend_kind(), StorageBackendKind::Local) {
                let _storage = StorageManager::local(storage_root)?;
            }
            start_sidecar_with_lane0_publish(storage.clone(), p2p_cmd_tx.clone(), &cli).await?;
        }
        Mode::DirectorOnly { gpu_device } => {
            info!("DirectorOnly mode: GPU device {}", gpu_device);
            // TODO: Initialize scheduler, sidecar
            let executor_registry =
                Arc::new(ChainExecutorRegistry::new(cli.rpc_url.clone(), registry_refresh).await?);
            let _executor_registry_task = executor_registry.clone().start_refresh();
            let submitter = attestation_submitter
                .take()
                .ok_or_else(|| anyhow!("attestation submitter already consumed"))?;
            let _redundant_scheduler =
                RedundantScheduler::new(RedundancyConfig::default(), submitter);
            start_sidecar_with_lane0_publish(storage.clone(), p2p_cmd_tx.clone(), &cli).await?;
        }
        Mode::ValidatorOnly { gpu_device } => {
            info!("ValidatorOnly mode: GPU device {}", gpu_device);
            // TODO: Initialize validator, sidecar
        }
        Mode::StorageOnly { storage_path } => {
            info!("StorageOnly mode: storage path {}", storage_path);
            // TODO: Initialize storage
            let storage_root = PathBuf::from(storage_path);
            if matches!(storage.backend_kind(), StorageBackendKind::Local) {
                let _storage = StorageManager::local(storage_root)?;
            }
        }
        Mode::LatencyReceiver { .. } => {}
    }

    info!("NSN Node initialized successfully");
    info!("Press Ctrl+C to stop");

    // Keep running until Ctrl+C
    tokio::signal::ctrl_c().await?;

    info!("Shutting down NSN Node");

    Ok(())
}

fn build_storage_manager(
    backend: StorageBackendMode,
    storage_path: String,
    ipfs_api_url: String,
) -> Result<StorageManager> {
    match backend {
        StorageBackendMode::Local => Ok(StorageManager::local(PathBuf::from(storage_path))?),
        StorageBackendMode::Ipfs => Ok(StorageManager::ipfs(ipfs_api_url)?),
    }
}

async fn publish_video_from_storage(
    storage: &StorageManager,
    cmd_tx: &tokio::sync::mpsc::UnboundedSender<nsn_p2p::ServiceCommand>,
    cid: &str,
    slot: u64,
    keypair_path: Option<&PathBuf>,
    chunk_size: usize,
    keyframe_interval: u32,
    ack_timeout_ms: u64,
) -> Result<nsn_p2p::VideoPublishReport> {
    let normalized = normalize_cid(cid);
    let payload = storage.get(&normalized).await?;

    let keypair = load_or_create_keypair(keypair_path)?;

    storage.pin(&normalized).await?;
    let audit = storage.audit_pin_status(&normalized).await?;
    log_audit(&audit);

    // Detect IVF format by magic bytes "DKIF"
    let is_ivf = payload.len() >= 4 && &payload[0..4] == b"DKIF";

    let chunks = if is_ivf {
        info!(
            slot,
            cid = %normalized,
            payload_bytes = payload.len(),
            "Detected IVF format, using frame-aware chunking"
        );
        build_video_chunks_from_ivf(&normalized, slot, &payload, &keypair)?
    } else {
        let config = VideoChunkConfig {
            chunk_size,
            keyframe_interval,
            ..VideoChunkConfig::default()
        };
        build_video_chunks(&normalized, slot, &payload, &keypair, &config)?
    };

    publish_video_chunks(cmd_tx, chunks, Duration::from_millis(ack_timeout_ms))
        .await
        .map_err(|err| anyhow!(err.to_string()))
}

fn normalize_cid(cid: &str) -> String {
    cid.trim()
        .strip_prefix("ipfs://")
        .or_else(|| cid.trim().strip_prefix("cid://"))
        .unwrap_or(cid.trim())
        .to_string()
}

fn load_or_create_keypair(path: Option<&PathBuf>) -> Result<libp2p::identity::Keypair> {
    if let Some(path) = path {
        if path.exists() {
            Ok(nsn_p2p::load_keypair(path)?)
        } else {
            let keypair = nsn_p2p::generate_keypair();
            nsn_p2p::save_keypair(&keypair, path)?;
            Ok(keypair)
        }
    } else {
        Ok(nsn_p2p::generate_keypair())
    }
}

fn log_audit(report: &StorageAuditReport) {
    info!(
        cid = %report.cid,
        backend = ?report.backend,
        status = ?report.status,
        checked_at_ms = report.checked_at_ms,
        "Storage audit"
    );
}

struct LatencyPercentiles {
    p50: u64,
    p95: u64,
    p99: u64,
}

struct RollingLatencyWindow {
    window: VecDeque<u64>,
    max_samples: usize,
}

impl RollingLatencyWindow {
    fn new(max_samples: usize) -> Self {
        Self {
            window: VecDeque::with_capacity(max_samples),
            max_samples,
        }
    }

    fn push(&mut self, value: u64) {
        if self.max_samples == 0 {
            return;
        }

        if self.window.len() == self.max_samples {
            self.window.pop_front();
        }
        self.window.push_back(value);
    }

    fn percentiles(&self) -> Option<LatencyPercentiles> {
        if self.window.is_empty() {
            return None;
        }

        let mut values: Vec<u64> = self.window.iter().copied().collect();
        values.sort_unstable();

        Some(LatencyPercentiles {
            p50: percentile(&values, 50.0),
            p95: percentile(&values, 95.0),
            p99: percentile(&values, 99.0),
        })
    }
}

fn percentile(sorted: &[u64], pct: f64) -> u64 {
    if sorted.is_empty() {
        return 0;
    }
    let clamped = pct.clamp(0.0, 100.0);
    let idx = ((clamped / 100.0) * (sorted.len().saturating_sub(1)) as f64).round() as usize;
    let safe_idx = idx.min(sorted.len() - 1);
    sorted[safe_idx]
}

async fn run_latency_receiver(
    mut rx: tokio::sync::broadcast::Receiver<u64>,
    window_size: usize,
    report_every: Duration,
) -> Result<()> {
    let mut window = RollingLatencyWindow::new(window_size.max(1));
    let mut ticker = tokio::time::interval(report_every);

    info!(
        window_size = window_size,
        report_every_ms = report_every.as_millis(),
        "Latency receiver started"
    );

    loop {
        tokio::select! {
            _ = ticker.tick() => {
                if let Some(p) = window.percentiles() {
                    println!(
                        "latency_ms p50={} p95={} p99={} samples={}",
                        p.p50,
                        p.p95,
                        p.p99,
                        window.window.len()
                    );
                }
            }
            msg = rx.recv() => {
                match msg {
                    Ok(latency_ms) => window.push(latency_ms),
                    Err(tokio::sync::broadcast::error::RecvError::Lagged(skipped)) => {
                        warn!(skipped = skipped, "Latency receiver lagged");
                    }
                    Err(tokio::sync::broadcast::error::RecvError::Closed) => {
                        warn!("Latency channel closed");
                        return Ok(());
                    }
                }
            }
        }
    }
}

async fn start_sidecar_with_lane0_publish(
    storage: StorageManager,
    p2p_cmd_tx: tokio::sync::mpsc::UnboundedSender<nsn_p2p::ServiceCommand>,
    cli: &Cli,
) -> Result<()> {
    let (task_tx, task_rx) = tokio::sync::mpsc::unbounded_channel::<TaskCompletionEvent>();

    // Configure sidecar with vortex-lane0 plugin allowed and increased latency limit
    let mut plugin_policy = PluginPolicy::default();
    plugin_policy.allowlist.insert("vortex-lane0".to_string());
    plugin_policy.lane0_max_latency_ms = 60_000; // 60 seconds for video generation

    let config = SidecarServiceConfig {
        plugin_policy,
        // Use wrapper script that activates the Vortex venv
        plugin_exec_command: Some(vec![
            "/home/matt/nsn/node-core/run-plugin.sh".to_string(),
        ]),
        ..SidecarServiceConfig::default()
    };

    let mut sidecar_service = SidecarService::with_config(config);
    sidecar_service.set_task_completion_sender(task_tx);
    let bind_addr = sidecar_service.config().bind_addr;

    tokio::spawn(async move {
        if let Err(err) = Server::builder()
            .add_service(SidecarServer::new(sidecar_service))
            .serve(bind_addr)
            .await
        {
            error!("Sidecar server failed: {}", err);
        }
    });

    let publish_slot = cli.publish_slot;
    let keypair_path = cli.p2p_keypair_path.clone();
    let chunk_size = cli.publish_chunk_size;
    let keyframe_interval = cli.publish_keyframe_interval;
    let ack_timeout_ms = cli.publish_ack_timeout_ms;
    let storage_clone = storage.clone();
    let p2p_cmd_tx_clone = p2p_cmd_tx.clone();

    tokio::spawn(async move {
        run_lane0_auto_publish(
            task_rx,
            storage_clone,
            p2p_cmd_tx_clone,
            publish_slot,
            keypair_path,
            chunk_size,
            keyframe_interval,
            ack_timeout_ms,
        )
        .await;
    });

    info!("Sidecar server started");
    Ok(())
}

async fn run_lane0_auto_publish(
    mut task_rx: tokio::sync::mpsc::UnboundedReceiver<TaskCompletionEvent>,
    storage: StorageManager,
    p2p_cmd_tx: tokio::sync::mpsc::UnboundedSender<nsn_p2p::ServiceCommand>,
    publish_slot: u64,
    keypair_path: Option<PathBuf>,
    chunk_size: usize,
    keyframe_interval: u32,
    ack_timeout_ms: u64,
) {
    while let Some(event) = task_rx.recv().await {
        if event.lane != 0 {
            continue;
        }

        if event.output_cid.is_empty() {
            warn!(task_id = %event.task_id, "Lane 0 task completed without output CID");
            continue;
        }

        info!(
            task_id = %event.task_id,
            output_cid = %event.output_cid,
            "Publishing Lane 0 output"
        );

        match publish_video_from_storage(
            &storage,
            &p2p_cmd_tx,
            &event.output_cid,
            publish_slot,
            keypair_path.as_ref(),
            chunk_size,
            keyframe_interval,
            ack_timeout_ms,
        )
        .await
        {
            Ok(report) => {
                info!(
                    task_id = %event.task_id,
                    total_chunks = report.total_chunks,
                    published = report.published,
                    failed = report.failed,
                    max_ack_ms = report.max_ack_ms,
                    avg_ack_ms = report.avg_ack_ms,
                    "Lane 0 video distribution completed"
                );
            }
            Err(err) => {
                error!(
                    task_id = %event.task_id,
                    error = %err,
                    "Lane 0 video distribution failed"
                );
            }
        }
    }
}

async fn build_attestation_submitter(
    cli: &Cli,
    p2p_cmd_tx: tokio::sync::mpsc::UnboundedSender<nsn_p2p::ServiceCommand>,
) -> Result<Box<dyn AttestationSubmitter>> {
    match cli.attestation_submit_mode {
        AttestationSubmitMode::P2p => Ok(Box::new(P2pAttestationSubmitter::new(p2p_cmd_tx))),
        AttestationSubmitMode::Chain => {
            let signer = parse_attestation_signer(&cli.attestation_suri)?;
            let submitter = ChainAttestationSubmitter::new(cli.rpc_url.clone(), signer).await?;
            Ok(Box::new(submitter))
        }
        AttestationSubmitMode::Dual => {
            let signer = parse_attestation_signer(&cli.attestation_suri)?;
            let chain_submitter =
                ChainAttestationSubmitter::new(cli.rpc_url.clone(), signer).await?;
            let p2p_submitter = P2pAttestationSubmitter::new(p2p_cmd_tx);
            Ok(Box::new(DualAttestationSubmitter::new(
                p2p_submitter,
                chain_submitter,
            )))
        }
        AttestationSubmitMode::None => Ok(Box::new(NoopAttestationSubmitter)),
    }
}

fn parse_attestation_signer(suri: &Option<String>) -> Result<sr25519::Pair> {
    let suri = suri
        .as_deref()
        .ok_or_else(|| anyhow!("attestation_suri is required for chain submission"))?;
    sr25519::Pair::from_string(suri, None)
        .map_err(|err| anyhow!("invalid attestation signer SURI: {err}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mode_to_node_mode() {
        let mode = Mode::SuperNode { gpu_device: 0 };
        assert_eq!(mode.to_node_mode(), Some(NodeCapability::SuperNode));

        let mode = Mode::DirectorOnly { gpu_device: 0 };
        assert_eq!(mode.to_node_mode(), Some(NodeCapability::DirectorOnly));

        let mode = Mode::LatencyReceiver {
            window_size: 10,
            report_every_ms: 1000,
        };
        assert!(mode.to_node_mode().is_none());
    }

    #[test]
    fn test_latency_percentiles() {
        let mut window = RollingLatencyWindow::new(5);
        for value in [10_u64, 20, 30, 40, 50] {
            window.push(value);
        }
        let p = window.percentiles().expect("percentiles available");
        assert_eq!(p.p50, 30);
        assert_eq!(p.p95, 50);
        assert_eq!(p.p99, 50);
    }

    #[test]
    fn test_ivf_detection() {
        // IVF magic bytes
        let ivf_data = b"DKIF\x00\x00\x20\x00VP90extra_content_here";
        let is_ivf = ivf_data.len() >= 4 && &ivf_data[0..4] == b"DKIF";
        assert!(is_ivf, "Should detect IVF magic bytes");

        // Non-IVF data
        let raw_data = b"NOT_IVF_DATA_HERE";
        let is_ivf = raw_data.len() >= 4 && &raw_data[0..4] == b"DKIF";
        assert!(!is_ivf, "Should not detect non-IVF data as IVF");

        // Short data
        let short_data = b"DKI";
        let is_ivf = short_data.len() >= 4 && &short_data[0..4] == b"DKIF";
        assert!(!is_ivf, "Should not detect short data as IVF");
    }

    #[test]
    fn test_ivf_chunking_uses_frame_aware_chunker() {
        // Create synthetic IVF with known frame structure
        let mut ivf_data = Vec::new();

        // IVF file header (32 bytes)
        ivf_data.extend_from_slice(b"DKIF");
        ivf_data.extend_from_slice(&0u16.to_le_bytes()); // version
        ivf_data.extend_from_slice(&32u16.to_le_bytes()); // header length
        ivf_data.extend_from_slice(b"VP90"); // FourCC
        ivf_data.extend_from_slice(&640u16.to_le_bytes()); // width
        ivf_data.extend_from_slice(&480u16.to_le_bytes()); // height
        ivf_data.extend_from_slice(&24u32.to_le_bytes()); // fps
        ivf_data.extend_from_slice(&1u32.to_le_bytes()); // time base
        ivf_data.extend_from_slice(&2u32.to_le_bytes()); // frame count
        ivf_data.extend_from_slice(&0u32.to_le_bytes()); // unused

        // Frame 0: Keyframe (200 bytes)
        let frame0 = vec![0x00; 200]; // bit 2 = 0 -> keyframe
        ivf_data.extend_from_slice(&(frame0.len() as u32).to_le_bytes());
        ivf_data.extend_from_slice(&0u64.to_le_bytes());
        ivf_data.extend_from_slice(&frame0);

        // Frame 1: Inter-frame (150 bytes)
        let frame1 = vec![0x04; 150]; // bit 2 = 1 -> inter-frame
        ivf_data.extend_from_slice(&(frame1.len() as u32).to_le_bytes());
        ivf_data.extend_from_slice(&1u64.to_le_bytes());
        ivf_data.extend_from_slice(&frame1);

        // Detect IVF
        let is_ivf = ivf_data.len() >= 4 && &ivf_data[0..4] == b"DKIF";
        assert!(is_ivf);

        // Verify we'd use frame-aware chunking
        let keypair = nsn_p2p::generate_keypair();
        let chunks = build_video_chunks_from_ivf("test_cid", 1, &ivf_data, &keypair)
            .expect("should chunk IVF");

        // Should have 2 chunks (one per frame), not MB-based chunks
        assert_eq!(chunks.len(), 2);
        assert!(chunks[0].header.is_keyframe);
        assert!(!chunks[1].header.is_keyframe);
        assert_eq!(chunks[0].payload.len(), 200);
        assert_eq!(chunks[1].payload.len(), 150);
    }
}
