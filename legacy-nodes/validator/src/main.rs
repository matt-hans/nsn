use clap::Parser;
use icn_validator::{ValidatorConfig, ValidatorNode};
use std::path::PathBuf;
use tracing::{error, info};
use tracing_subscriber::EnvFilter;

/// ICN Validator Node - Semantic verification for director-generated content
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to configuration file
    #[arg(short, long, default_value = "config/validator.toml")]
    config: PathBuf,

    /// ICN Chain WebSocket endpoint (overrides config file)
    #[arg(long)]
    chain_endpoint: Option<String>,

    /// Path to Ed25519 keypair JSON (overrides config file)
    #[arg(long)]
    keypair: Option<PathBuf>,

    /// Directory containing CLIP ONNX models (overrides config file)
    #[arg(long)]
    models_dir: Option<PathBuf>,

    /// Metrics server port (overrides config file)
    #[arg(long)]
    metrics_port: Option<u16>,

    /// Enable verbose logging
    #[arg(short, long)]
    verbose: bool,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Parse command line arguments
    let args = Args::parse();

    // Initialize tracing subscriber
    let filter = if args.verbose {
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("debug"))
    } else {
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"))
    };

    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(true)
        .init();

    info!("🚀 Starting ICN Validator Node");
    info!("Version: {}", env!("CARGO_PKG_VERSION"));

    // Load configuration
    let mut config = if args.config.exists() {
        info!("Loading configuration from {:?}", args.config);
        ValidatorConfig::from_file(&args.config)?
    } else {
        error!("Configuration file not found: {:?}", args.config);
        std::process::exit(1);
    };

    // Apply CLI overrides
    if let Some(endpoint) = args.chain_endpoint {
        info!("Overriding chain endpoint: {}", endpoint);
        config.chain_endpoint = endpoint;
    }

    if let Some(keypair) = args.keypair {
        info!("Overriding keypair path: {:?}", keypair);
        config.keypair_path = keypair;
    }

    if let Some(models_dir) = args.models_dir {
        info!("Overriding models directory: {:?}", models_dir);
        config.models_dir = models_dir;
    }

    if let Some(port) = args.metrics_port {
        info!("Overriding metrics port: {}", port);
        config.metrics.port = port;
    }

    // Validate configuration
    config.validate()?;

    // Create and run validator node
    info!("Initializing validator node...");
    let validator = ValidatorNode::new(config).await?;

    info!("✅ Validator node initialized successfully");
    info!("🔄 Entering main event loop...");

    // Set up graceful shutdown handler
    let (shutdown_tx, mut shutdown_rx) = tokio::sync::mpsc::channel::<()>(1);

    ctrlc::set_handler(move || {
        info!("Received shutdown signal (Ctrl+C)");
        let _ = shutdown_tx.try_send(());
    })?;

    // Run validator with shutdown handling
    tokio::select! {
        result = validator.run() => {
            if let Err(e) = result {
                error!("Validator node error: {}", e);
                std::process::exit(1);
            }
        }
        _ = shutdown_rx.recv() => {
            info!("Shutting down validator node gracefully...");
        }
    }

    info!("👋 Validator node stopped");
    Ok(())
}
