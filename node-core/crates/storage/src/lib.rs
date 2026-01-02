//! Storage management for NSN nodes
//!
//! Provides a pluggable storage backend interface with local filesystem
//! and IPFS implementations.

mod ipfs;
mod local;
mod metrics;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use thiserror::Error;

pub use ipfs::IpfsBackend;
pub use local::LocalBackend;
pub use metrics::StorageMetrics;

/// Content identifier type alias.
pub type Cid = String;

/// Storage backend type label.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StorageBackendKind {
    Local,
    Ipfs,
}

/// Pin status for a content ID.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PinStatus {
    Pinned,
    NotPinned,
}

/// Audit report for storage pinning.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageAuditReport {
    pub cid: Cid,
    pub backend: StorageBackendKind,
    pub status: PinStatus,
    pub checked_at_ms: u64,
}

#[derive(Debug, Error)]
pub enum StorageError {
    #[error("Invalid CID: {0}")]
    InvalidCid(String),

    #[error("CID mismatch: expected {expected}, got {actual}")]
    CidMismatch { expected: String, actual: String },

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),

    #[error("Backend error: {0}")]
    Backend(String),
}

#[async_trait]
pub trait StorageBackend: Send + Sync {
    async fn put(&self, cid: &Cid, data: &[u8]) -> Result<(), StorageError>;
    async fn get(&self, cid: &Cid) -> Result<Vec<u8>, StorageError>;
    async fn pin(&self, cid: &Cid) -> Result<(), StorageError>;
    async fn unpin(&self, cid: &Cid) -> Result<(), StorageError>;
    async fn pin_status(&self, cid: &Cid) -> Result<PinStatus, StorageError>;
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StorageBackendConfig {
    Local { root: PathBuf },
    Ipfs { api_url: String },
}

#[derive(Clone)]
pub struct StorageManager {
    backend: Arc<dyn StorageBackend>,
    metrics: Option<Arc<StorageMetrics>>,
    kind: StorageBackendKind,
}

impl StorageManager {
    pub fn new(config: StorageBackendConfig) -> Result<Self, StorageError> {
        let (backend, kind): (Arc<dyn StorageBackend>, StorageBackendKind) = match config {
            StorageBackendConfig::Local { root } => (
                Arc::new(LocalBackend::new(root)?),
                StorageBackendKind::Local,
            ),
            StorageBackendConfig::Ipfs { api_url } => (
                Arc::new(IpfsBackend::new(api_url)?),
                StorageBackendKind::Ipfs,
            ),
        };

        Ok(Self {
            backend,
            metrics: None,
            kind,
        })
    }

    pub fn new_with_metrics(
        config: StorageBackendConfig,
        metrics: Arc<StorageMetrics>,
    ) -> Result<Self, StorageError> {
        let mut manager = Self::new(config)?;
        manager.metrics = Some(metrics);
        Ok(manager)
    }

    pub fn local(root: PathBuf) -> Result<Self, StorageError> {
        Self::new(StorageBackendConfig::Local { root })
    }

    pub fn ipfs(api_url: String) -> Result<Self, StorageError> {
        Self::new(StorageBackendConfig::Ipfs { api_url })
    }

    pub fn backend(&self) -> Arc<dyn StorageBackend> {
        self.backend.clone()
    }

    pub fn backend_kind(&self) -> StorageBackendKind {
        self.kind
    }

    pub async fn put(&self, cid: &Cid, data: &[u8]) -> Result<(), StorageError> {
        let _timer = self
            .metrics
            .as_ref()
            .map(|metrics| metrics.operation_duration_seconds.start_timer());
        let result = self.backend.put(cid, data).await;
        self.record_metric("put", result.is_err());
        result
    }

    pub async fn get(&self, cid: &Cid) -> Result<Vec<u8>, StorageError> {
        let _timer = self
            .metrics
            .as_ref()
            .map(|metrics| metrics.operation_duration_seconds.start_timer());
        let result = self.backend.get(cid).await;
        self.record_metric("get", result.is_err());
        result
    }

    pub async fn pin(&self, cid: &Cid) -> Result<(), StorageError> {
        let _timer = self
            .metrics
            .as_ref()
            .map(|metrics| metrics.operation_duration_seconds.start_timer());
        let result = self.backend.pin(cid).await;
        self.record_metric("pin", result.is_err());
        result
    }

    pub async fn unpin(&self, cid: &Cid) -> Result<(), StorageError> {
        let _timer = self
            .metrics
            .as_ref()
            .map(|metrics| metrics.operation_duration_seconds.start_timer());
        let result = self.backend.unpin(cid).await;
        self.record_metric("unpin", result.is_err());
        result
    }

    pub async fn pin_status(&self, cid: &Cid) -> Result<PinStatus, StorageError> {
        let _timer = self
            .metrics
            .as_ref()
            .map(|metrics| metrics.operation_duration_seconds.start_timer());
        let result = self.backend.pin_status(cid).await;
        self.record_metric("pin_status", result.is_err());
        result
    }

    pub async fn audit_pin_status(&self, cid: &Cid) -> Result<StorageAuditReport, StorageError> {
        let status = self.pin_status(cid).await?;
        Ok(StorageAuditReport {
            cid: cid.clone(),
            backend: self.kind,
            status,
            checked_at_ms: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64,
        })
    }

    fn record_metric(&self, operation: &str, failed: bool) {
        if let Some(metrics) = &self.metrics {
            metrics
                .operations_total
                .with_label_values(&[operation])
                .inc();
            if failed {
                metrics
                    .operations_failed_total
                    .with_label_values(&[operation])
                    .inc();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use prometheus::Registry;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_local_backend_put_get_pin() {
        let temp_dir = TempDir::new().expect("temp dir");
        let storage = StorageManager::local(temp_dir.path().to_path_buf()).expect("storage");
        let cid = "QmTestCid".to_string();
        let payload = b"hello".to_vec();

        storage.put(&cid, &payload).await.expect("put");
        storage.pin(&cid).await.expect("pin");

        let fetched = storage.get(&cid).await.expect("get");
        assert_eq!(fetched, payload);

        storage.unpin(&cid).await.expect("unpin");
    }

    #[tokio::test]
    async fn test_metrics_recorded_on_put() {
        let temp_dir = TempDir::new().expect("temp dir");
        let registry = Registry::new();
        let metrics = Arc::new(StorageMetrics::new(&registry).expect("metrics"));
        let storage = StorageManager::new_with_metrics(
            StorageBackendConfig::Local {
                root: temp_dir.path().to_path_buf(),
            },
            metrics.clone(),
        )
        .expect("storage");

        let cid = "QmMetricsCid".to_string();
        storage.put(&cid, b"data").await.expect("put");

        let count = metrics.operations_total.with_label_values(&["put"]).get();
        assert_eq!(count, 1);
    }

    #[tokio::test]
    async fn test_local_pin_status() {
        let temp_dir = TempDir::new().expect("temp dir");
        let storage = StorageManager::local(temp_dir.path().to_path_buf()).expect("storage");
        let cid = "QmStatusCid".to_string();
        storage.put(&cid, b"data").await.expect("put");

        let status = storage.pin_status(&cid).await.expect("pin status");
        assert_eq!(status, PinStatus::NotPinned);

        storage.pin(&cid).await.expect("pin");
        let status = storage.pin_status(&cid).await.expect("pin status");
        assert_eq!(status, PinStatus::Pinned);
    }
}
