//! Reputation oracle for on-chain reputation scores
//!
//! PLACEHOLDER: Full implementation deferred to T043
//! This stub provides minimal types to satisfy service.rs compilation

use thiserror::Error;

#[derive(Debug, Error)]
pub enum OracleError {
    #[error("RPC error: {0}")]
    Rpc(String),

    #[error("Sync error: {0}")]
    Sync(String),
}

/// Reputation oracle (STUB - T043)
pub struct ReputationOracle {
    _rpc_url: String,
}

impl ReputationOracle {
    /// Create new reputation oracle
    pub fn new(rpc_url: String) -> Self {
        Self { _rpc_url: rpc_url }
    }

    /// Sync loop (STUB - T043)
    pub async fn sync_loop(&self) {
        // Stub: do nothing
        loop {
            tokio::time::sleep(tokio::time::Duration::from_secs(60)).await;
        }
    }
}

/// Default reputation score
#[allow(dead_code)] // Stub for T043
pub const DEFAULT_REPUTATION: f64 = 0.5;

/// Sync interval
#[allow(dead_code)] // Stub for T043
pub const SYNC_INTERVAL: std::time::Duration = std::time::Duration::from_secs(60);
