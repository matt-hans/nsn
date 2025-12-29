//! Audit monitor for pinning challenge response
//!
//! Polls on-chain PendingAudits and generates proofs

use sha2::{Digest, Sha256};
use std::path::Path;
use tokio::fs::File;
use tokio::io::{AsyncReadExt, AsyncSeekExt};

/// Audit challenge from on-chain
#[derive(Debug, Clone)]
pub struct AuditChallenge {
    pub audit_id: u64,
    pub cid: String,
    pub shard_index: usize,
    pub byte_offset: u64,
    pub byte_length: u64,
    pub nonce: Vec<u8>,
}

/// Generate audit proof for challenged bytes
pub async fn generate_audit_proof(
    shard_path: &Path,
    challenge: &AuditChallenge,
) -> crate::error::Result<Vec<u8>> {
    // Read challenged bytes from shard
    let mut file = File::open(shard_path).await?;
    file.seek(std::io::SeekFrom::Start(challenge.byte_offset))
        .await?;

    let mut buffer = vec![0u8; challenge.byte_length as usize];
    file.read_exact(&mut buffer).await?;

    // Hash with nonce
    let mut hasher = Sha256::new();
    hasher.update(&buffer);
    hasher.update(&challenge.nonce);

    Ok(hasher.finalize().to_vec())
}

use crate::chain_client::{ChainClient, ChainEvent};
use crate::metrics;
use crate::storage::Storage;
use std::sync::Arc;
use tokio::sync::mpsc;
use tokio::time::{interval, Duration};
use tracing::{error, info, warn};

/// Audit monitor service
pub struct AuditMonitor {
    poll_interval_secs: u64,
    chain_client: Arc<ChainClient>,
    storage: Arc<Storage>,
    chain_rx: mpsc::UnboundedReceiver<ChainEvent>,
}

impl AuditMonitor {
    pub fn new(
        poll_interval_secs: u64,
        chain_client: Arc<ChainClient>,
        storage: Arc<Storage>,
        chain_rx: mpsc::UnboundedReceiver<ChainEvent>,
    ) -> Self {
        Self {
            poll_interval_secs,
            chain_client,
            storage,
            chain_rx,
        }
    }

    /// Start audit monitoring loop
    ///
    /// Polls chain for pending audits and generates proofs
    pub async fn start(mut self) -> crate::error::Result<()> {
        info!(
            "Audit monitor started (poll interval: {}s)",
            self.poll_interval_secs
        );

        let mut poll_timer = interval(Duration::from_secs(self.poll_interval_secs));

        loop {
            tokio::select! {
                _ = poll_timer.tick() => {
                    // Poll chain for pending audits
                    if let Err(e) = self.poll_audits().await {
                        error!("Audit poll failed: {}", e);
                    }
                }
                Some(event) = self.chain_rx.recv() => {
                    match event {
                        ChainEvent::PendingAudit(audit) => {
                            info!("Pending audit detected: audit_id={}", audit.audit_id);
                            if let Err(e) = self.handle_audit(audit).await {
                                error!("Audit handling failed: {}", e);
                            }
                        }
                        ChainEvent::BlockFinalized { block_number } => {
                            tracing::debug!("Block finalized: {}", block_number);
                        }
                    }
                }
            }
        }
    }

    /// Poll chain for pending audits
    async fn poll_audits(&self) -> crate::error::Result<()> {
        // TODO: Query PendingAudits storage from chain
        // For now, this is a no-op as chain_rx will receive events
        Ok(())
    }

    /// Handle pending audit
    async fn handle_audit(
        &self,
        audit: crate::chain_client::PendingAudit,
    ) -> crate::error::Result<()> {
        // Construct shard path
        let shard_path = self.storage.get_shard_path(&audit.cid, audit.shard_index);

        // Generate audit proof
        let challenge = AuditChallenge {
            audit_id: audit.audit_id,
            cid: audit.cid.clone(),
            shard_index: audit.shard_index,
            byte_offset: audit.byte_offset,
            byte_length: audit.byte_length,
            nonce: audit.nonce.clone(),
        };

        match generate_audit_proof(&shard_path, &challenge).await {
            Ok(proof) => {
                info!("Generated audit proof for audit_id={}", audit.audit_id);

                // Submit proof to chain
                match self
                    .chain_client
                    .submit_audit_proof(audit.audit_id, proof)
                    .await
                {
                    Ok(tx_hash) => {
                        info!("Audit proof submitted: {}", tx_hash);
                        // Update metrics - successful audit
                        metrics::AUDIT_SUCCESS_TOTAL.inc();
                        Ok(())
                    }
                    Err(e) => {
                        error!("Audit proof submission failed: {}", e);
                        // Update metrics - failed audit
                        metrics::AUDIT_FAILURE_TOTAL.inc();
                        Err(e)
                    }
                }
            }
            Err(e) => {
                warn!("Audit proof generation failed: {}", e);
                // Update metrics - failed audit (couldn't generate proof)
                metrics::AUDIT_FAILURE_TOTAL.inc();
                Err(e)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use tempfile::NamedTempFile;

    #[tokio::test]
    async fn test_audit_proof_generation() {
        // Create test shard file
        let tmp_file = NamedTempFile::new().unwrap();
        let test_data = b"This is shard data for audit proof testing. More bytes here...";
        tokio::fs::write(tmp_file.path(), test_data).await.unwrap();

        let challenge = AuditChallenge {
            audit_id: 123,
            cid: "bafytest".to_string(),
            shard_index: 5,
            byte_offset: 8,
            byte_length: 10,
            nonce: vec![1, 2, 3, 4],
        };

        let proof = generate_audit_proof(tmp_file.path(), &challenge)
            .await
            .expect("Proof generation failed");

        // Verify proof is hash (32 bytes for SHA256)
        assert_eq!(proof.len(), 32);
    }

    /// Test Case: Audit proof generation with invalid offset
    /// Purpose: Verify error handling when byte range exceeds file size
    /// Contract: Returns error without panic
    #[tokio::test]
    async fn test_audit_proof_invalid_offset() {
        let tmp_file = NamedTempFile::new().unwrap();
        let test_data = b"Short data";
        tokio::fs::write(tmp_file.path(), test_data).await.unwrap();

        let challenge = AuditChallenge {
            audit_id: 456,
            cid: "bafytest".to_string(),
            shard_index: 0,
            byte_offset: 100, // Beyond file size
            byte_length: 10,
            nonce: vec![5, 6, 7, 8],
        };

        let result = generate_audit_proof(tmp_file.path(), &challenge).await;

        // Should fail with IO error (seek beyond EOF or read exact failure)
        assert!(result.is_err());
    }

    /// Test Case: Audit proof with exact file size match
    /// Purpose: Verify boundary condition when reading entire file
    /// Contract: Successfully generates proof for full file
    #[tokio::test]
    async fn test_audit_proof_full_file() {
        let tmp_file = NamedTempFile::new().unwrap();
        let test_data = b"Exact size match test data";
        tokio::fs::write(tmp_file.path(), test_data).await.unwrap();

        let challenge = AuditChallenge {
            audit_id: 789,
            cid: "bafytest".to_string(),
            shard_index: 0,
            byte_offset: 0,
            byte_length: test_data.len() as u64,
            nonce: vec![9, 10, 11, 12],
        };

        let proof = generate_audit_proof(tmp_file.path(), &challenge)
            .await
            .expect("Full file proof generation failed");

        // Verify proof generated
        assert_eq!(proof.len(), 32);
    }

    /// Test Case: Audit proof for missing shard file
    /// Purpose: Verify error handling when shard file doesn't exist
    /// Contract: Returns error indicating file not found
    #[tokio::test]
    async fn test_audit_proof_missing_file() {
        let nonexistent_path = PathBuf::from("/tmp/nonexistent_shard_12345.bin");

        let challenge = AuditChallenge {
            audit_id: 999,
            cid: "bafymissing".to_string(),
            shard_index: 3,
            byte_offset: 0,
            byte_length: 100,
            nonce: vec![13, 14, 15, 16],
        };

        let result = generate_audit_proof(&nonexistent_path, &challenge).await;

        // Should fail with file not found error
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            crate::error::SuperNodeError::Io(_)
        ));
    }
}
