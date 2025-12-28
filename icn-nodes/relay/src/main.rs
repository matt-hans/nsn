//! ICN Regional Relay Node
//!
//! Tier 2 content distribution with LRU caching, latency-based region detection,
//! and QUIC-based serving to viewers.

use clap::Parser;
use icn_relay::{Config, RelayNode};
use std::path::PathBuf;
use tracing::info;

/// ICN Regional Relay CLI arguments
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Configuration file path
    #[arg(short, long, default_value = "config/relay.toml")]
    config: PathBuf,

    /// Override cache path
    #[arg(long)]
    cache_path: Option<PathBuf>,

    /// Override region (skip auto-detection)
    #[arg(long)]
    region: Option<String>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    setup_logging();

    info!("ICN Regional Relay Node starting...");

    let args = Args::parse();
    let config = load_config(&args)?;

    let relay = RelayNode::new(config, args.region).await?;
    relay.run().await?;

    Ok(())
}

/// Initialize tracing subscriber for logging
fn setup_logging() {
    tracing_subscriber::fmt::init();
}

/// Load and validate configuration
fn load_config(args: &Args) -> anyhow::Result<Config> {
    let mut config = Config::load(&args.config)?;

    if let Some(cache_path) = &args.cache_path {
        config.cache_path = cache_path.clone();
    }

    info!("Configuration loaded: {:?}", config);
    Ok(config)
}
