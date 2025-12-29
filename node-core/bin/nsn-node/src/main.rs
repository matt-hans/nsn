//! NSN Unified Node Binary
//!
//! This binary runs in one of four modes:
//! - SuperNode: Full capabilities (Director + Validator + Storage)
//! - DirectorOnly: Generation capabilities only
//! - ValidatorOnly: CLIP verification only
//! - StorageOnly: Pinning and distribution only

use anyhow::Result;
use clap::{Parser, Subcommand};
use nsn_types::NodeMode;
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

impl Mode {
    fn to_node_mode(&self) -> NodeMode {
        match self {
            Mode::SuperNode { .. } => NodeMode::SuperNode,
            Mode::DirectorOnly { .. } => NodeMode::DirectorOnly,
            Mode::ValidatorOnly { .. } => NodeMode::ValidatorOnly,
            Mode::StorageOnly { .. } => NodeMode::StorageOnly,
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

    match cli.mode {
        Mode::SuperNode { gpu_device } => {
            info!("SuperNode mode: GPU device {}", gpu_device);
            // TODO: Initialize scheduler, sidecar, P2P, storage
        }
        Mode::DirectorOnly { gpu_device } => {
            info!("DirectorOnly mode: GPU device {}", gpu_device);
            // TODO: Initialize scheduler, sidecar, P2P
        }
        Mode::ValidatorOnly { gpu_device } => {
            info!("ValidatorOnly mode: GPU device {}", gpu_device);
            // TODO: Initialize validator, sidecar, P2P
        }
        Mode::StorageOnly { storage_path } => {
            info!("StorageOnly mode: storage path {}", storage_path);
            // TODO: Initialize storage, P2P
        }
    }

    info!("NSN Node initialized successfully");
    info!("Press Ctrl+C to stop");

    // Keep running until Ctrl+C
    tokio::signal::ctrl_c().await?;

    info!("Shutting down NSN Node");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mode_to_node_mode() {
        let mode = Mode::SuperNode { gpu_device: 0 };
        assert_eq!(mode.to_node_mode(), NodeMode::SuperNode);

        let mode = Mode::DirectorOnly { gpu_device: 0 };
        assert_eq!(mode.to_node_mode(), NodeMode::DirectorOnly);
    }
}
