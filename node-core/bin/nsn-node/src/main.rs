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
    build_video_chunks, publish_video_chunks, P2pConfig, P2pService, VideoChunkConfig,
    DEFAULT_CHUNK_SIZE_BYTES,
};
use nsn_scheduler::{
    AttestationSubmitter, DualAttestationSubmitter, NoopAttestationSubmitter,
    P2pAttestationSubmitter, RedundancyConfig, RedundantScheduler,
};
use nsn_storage::{StorageAuditReport, StorageBackendKind, StorageManager};
use nsn_types::NodeCapability;
use sp_core::{sr25519, Pair};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;
use tracing::{info, Level};
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

    /// Storage backend (local or ipfs)
    #[arg(long, value_enum, default_value = "ipfs")]
    storage_backend: StorageBackendMode,

    /// IPFS API URL (for ipfs storage backend)
    #[arg(long, default_value = "http://127.0.0.1:5001")]
    ipfs_api_url: String,

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
    fn to_node_mode(&self) -> NodeCapability {
        match self {
            Mode::SuperNode { .. } => NodeCapability::SuperNode,
            Mode::DirectorOnly { .. } => NodeCapability::DirectorOnly,
            Mode::ValidatorOnly { .. } => NodeCapability::ValidatorOnly,
            Mode::StorageOnly { .. } => NodeCapability::StorageOnly,
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

    let node_mode = cli.mode.to_node_mode();

    info!("Starting NSN Node");
    info!("Mode: {:?}", node_mode);
    info!("Config: {}", cli.config);

    // TODO: Load configuration from file
    // TODO: Initialize components based on mode
    // TODO: Start main event loop

    // Initialize P2P networking
    let mut p2p_config = P2pConfig::default();
    p2p_config.listen_port = cli.p2p_listen_port;
    p2p_config.metrics_port = cli.p2p_metrics_port;
    p2p_config.keypair_path = cli.p2p_keypair_path.clone();

    let (mut p2p_service, p2p_cmd_tx) =
        P2pService::new(p2p_config, cli.rpc_url.clone()).await?;
    tokio::spawn(async move {
        if let Err(err) = p2p_service.start().await {
            tracing::error!("P2P service failed: {}", err);
        }
    });

    let mut attestation_submitter =
        Some(build_attestation_submitter(&cli, p2p_cmd_tx.clone()).await?);
    info!(
        "Attestation submit mode: {:?}",
        cli.attestation_submit_mode
    );

    let registry_refresh = Duration::from_secs(cli.executor_registry_refresh_secs);

    let storage_path = match &cli.mode {
        Mode::StorageOnly { storage_path } => storage_path.clone(),
        _ => "/var/lib/nsn/storage".to_string(),
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
            let executor_registry = Arc::new(
                ChainExecutorRegistry::new(cli.rpc_url.clone(), registry_refresh).await?,
            );
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
        }
        Mode::DirectorOnly { gpu_device } => {
            info!("DirectorOnly mode: GPU device {}", gpu_device);
            // TODO: Initialize scheduler, sidecar
            let executor_registry = Arc::new(
                ChainExecutorRegistry::new(cli.rpc_url.clone(), registry_refresh).await?,
            );
            let _executor_registry_task = executor_registry.clone().start_refresh();
            let submitter = attestation_submitter
                .take()
                .ok_or_else(|| anyhow!("attestation submitter already consumed"))?;
            let _redundant_scheduler =
                RedundantScheduler::new(RedundancyConfig::default(), submitter);
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
    let config = VideoChunkConfig {
        chunk_size,
        keyframe_interval,
        ..VideoChunkConfig::default()
    };

    storage.pin(&normalized).await?;
    let audit = storage.audit_pin_status(&normalized).await?;
    log_audit(&audit);

    let chunks = build_video_chunks(&normalized, slot, &payload, &keypair, &config)?;
    publish_video_chunks(cmd_tx, chunks, Duration::from_millis(ack_timeout_ms)).await
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

async fn build_attestation_submitter(
    cli: &Cli,
    p2p_cmd_tx: tokio::sync::mpsc::UnboundedSender<nsn_p2p::ServiceCommand>,
) -> Result<Box<dyn AttestationSubmitter>> {
    match cli.attestation_submit_mode {
        AttestationSubmitMode::P2p => {
            Ok(Box::new(P2pAttestationSubmitter::new(p2p_cmd_tx)))
        }
        AttestationSubmitMode::Chain => {
            let signer = parse_attestation_signer(&cli.attestation_suri)?;
            let submitter =
                ChainAttestationSubmitter::new(cli.rpc_url.clone(), signer).await?;
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
        assert_eq!(mode.to_node_mode(), NodeCapability::SuperNode);

        let mode = Mode::DirectorOnly { gpu_device: 0 };
        assert_eq!(mode.to_node_mode(), NodeCapability::DirectorOnly);
    }
}
