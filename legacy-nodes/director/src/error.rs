use thiserror::Error;

#[cfg_attr(feature = "stub", allow(dead_code))]
#[derive(Error, Debug)]
pub enum DirectorError {
    #[error("Chain client error: {0}")]
    ChainClient(String),

    #[error("Election monitor error: {0}")]
    ElectionMonitor(String),

    #[error("Slot scheduler error: {0}")]
    SlotScheduler(String),

    #[error("BFT coordinator error: {0}")]
    BftCoordinator(String),

    #[error("P2P service error: {0}")]
    P2pService(String),

    #[error("Vortex bridge error: {0}")]
    VortexBridge(String),

    #[error("Configuration error: {0}")]
    Config(String),

    #[error("Metrics error: {0}")]
    Metrics(String),

    #[error("Io error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Subxt error: {0}")]
    Subxt(String),

    #[error("gRPC error: {0}")]
    Grpc(#[from] tonic::Status),

    #[error("Serialization error: {0}")]
    Serialization(String),

    #[error("Prometheus error: {0}")]
    Prometheus(#[from] prometheus::Error),
}

pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
