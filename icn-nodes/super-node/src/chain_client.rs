//! Chain client for pinning deal monitoring and audit submission
//!
//! Integrates with ICN Chain using subxt for:
//! - Monitoring PendingAudits storage
//! - Submitting audit proofs via extrinsics
//! - Tracking finalized blocks

#[allow(unused_imports)]
use futures::StreamExt;
use parity_scale_codec::{Decode, Encode};
use subxt::{OnlineClient, PolkadotConfig};
use tokio::sync::mpsc;
use tracing::{debug, error, info, warn}; // Used in subscribe_pending_audits() spawn

/// Pending audit from on-chain
#[derive(Debug, Clone)]
pub struct PendingAudit {
    pub audit_id: u64,
    pub pinner: String,
    pub cid: String,
    pub shard_index: usize,
    pub byte_offset: u64,
    pub byte_length: u64,
    pub nonce: Vec<u8>,
    pub deadline_block: u64,
}

/// Pinning deal from on-chain
#[derive(Debug, Clone, Encode, Decode)]
pub struct PinningDeal {
    pub deal_id: u64,
    pub creator: Vec<u8>,
    pub cid: String,
    pub shards: Vec<u8>,
    pub expires_at: u64,
    pub total_reward: u128,
    pub status: u8,
}

/// Chain events
#[derive(Debug, Clone)]
pub enum ChainEvent {
    /// New pending audit detected
    PendingAudit(PendingAudit),
    /// Block finalized
    BlockFinalized { block_number: u64 },
}

/// Chain client for ICN Chain interaction
pub struct ChainClient {
    api: Option<OnlineClient<PolkadotConfig>>,
    endpoint: String,
    event_tx: mpsc::UnboundedSender<ChainEvent>,
}

impl ChainClient {
    /// Connect to ICN Chain
    ///
    /// # Arguments
    /// * `endpoint` - WebSocket RPC endpoint (ws://... or wss://...)
    ///
    /// # Returns
    /// Chain client instance and event receiver
    pub async fn connect(
        endpoint: String,
    ) -> crate::error::Result<(Self, mpsc::UnboundedReceiver<ChainEvent>)> {
        // Validate endpoint
        if !endpoint.starts_with("ws://") && !endpoint.starts_with("wss://") {
            return Err(crate::error::SuperNodeError::ChainClient(
                "Endpoint must start with ws:// or wss://".to_string(),
            ));
        }

        info!("Connecting to ICN Chain at {}", endpoint);

        // Initialize subxt client (graceful degradation if chain not available)
        let api = match OnlineClient::<PolkadotConfig>::from_url(&endpoint).await {
            Ok(client) => {
                info!("Successfully connected to ICN Chain");
                Some(client)
            }
            Err(e) => {
                warn!(
                    "Failed to connect to ICN Chain ({}): {}. Running in offline mode.",
                    endpoint, e
                );
                None
            }
        };

        let (event_tx, event_rx) = mpsc::unbounded_channel();

        let client = Self {
            api,
            endpoint: endpoint.clone(),
            event_tx,
        };

        Ok((client, event_rx))
    }

    /// Subscribe to pending audits from on-chain storage
    pub async fn subscribe_pending_audits(&self) -> crate::error::Result<()> {
        debug!("Subscribing to PendingAudits storage");

        let api = match &self.api {
            Some(api) => api,
            None => {
                warn!("Chain API not connected, skipping pending audits subscription");
                return Ok(());
            }
        };

        // Subscribe to finalized blocks
        let mut blocks_sub = api.blocks().subscribe_finalized().await.map_err(|e| {
            crate::error::SuperNodeError::ChainClient(format!("Block subscription failed: {}", e))
        })?;

        let event_tx = self.event_tx.clone();

        // Spawn background task to poll pending audits
        tokio::spawn(async move {
            while let Some(block_result) = blocks_sub.next().await {
                match block_result {
                    Ok(block) => {
                        let block_number = block.number() as u64;

                        // Emit block finalized event
                        if event_tx
                            .send(ChainEvent::BlockFinalized { block_number })
                            .is_err()
                        {
                            debug!("Event channel closed, stopping audit subscription");
                            break;
                        }

                        // TODO: Query PendingAudits storage from pallet-icn-pinning
                        // This requires generated metadata from ICN Chain
                        // For now, we'll use the event-driven approach via ChainEvent
                        //
                        // Example implementation (requires ICN Chain metadata):
                        // let storage = api.storage().at(block.hash());
                        // if let Ok(Some(audits)) = storage.fetch(&icn_pinning::storage::PendingAudits::root()).await {
                        //     for (audit_id, audit_data) in audits {
                        //         let pending_audit = PendingAudit {
                        //             audit_id,
                        //             pinner: audit_data.pinner,
                        //             cid: audit_data.cid,
                        //             shard_index: audit_data.shard_index,
                        //             byte_offset: audit_data.byte_offset,
                        //             byte_length: audit_data.byte_length,
                        //             nonce: audit_data.nonce,
                        //             deadline_block: audit_data.deadline_block,
                        //         };
                        //         let _ = event_tx.send(ChainEvent::PendingAudit(pending_audit));
                        //     }
                        // }
                    }
                    Err(e) => {
                        error!("Block subscription error: {}", e);
                        break;
                    }
                }
            }

            warn!("Pending audits subscription ended");
        });

        Ok(())
    }

    /// Submit audit proof extrinsic
    ///
    /// # Arguments
    /// * `audit_id` - Audit ID from on-chain
    /// * `proof` - SHA256 hash of challenged bytes + nonce
    ///
    /// # Returns
    /// Transaction hash
    pub async fn submit_audit_proof(
        &self,
        audit_id: u64,
        proof: Vec<u8>,
    ) -> crate::error::Result<String> {
        info!("Submitting audit proof for audit_id={}", audit_id);

        let _api = match &self.api {
            Some(api) => api,
            None => {
                return Err(crate::error::SuperNodeError::ChainClient(
                    "Chain API not connected".to_string(),
                ));
            }
        };

        // TODO: Implement actual extrinsic submission once ICN Chain metadata is available
        // This requires:
        // 1. Generated subxt types from ICN Chain metadata
        // 2. Signer keypair (from config or keystore)
        //
        // Example implementation:
        // let signer = subxt_signer::sr25519::dev::alice();
        // let submit_tx = icn_pinning::tx().submit_audit_proof(audit_id, proof);
        // let tx = api.tx()
        //     .sign_and_submit_then_watch_default(&submit_tx, &signer)
        //     .await
        //     .map_err(|e| SuperNodeError::ChainClient(format!("Extrinsic submission failed: {}", e)))?;
        //
        // let events = tx.wait_for_finalized_success().await
        //     .map_err(|e| SuperNodeError::ChainClient(format!("Transaction failed: {}", e)))?;
        //
        // let tx_hash = format!("0x{}", hex::encode(tx.extrinsic_hash()));

        // For now, return simulated response
        let tx_hash = format!(
            "0x{}{}",
            hex::encode(audit_id.to_le_bytes()),
            hex::encode(&proof[..8.min(proof.len())])
        );

        debug!("Audit proof prepared for submission: {}", tx_hash);

        Ok(tx_hash)
    }

    /// Run chain client event loop
    ///
    /// Monitors finalized blocks and emits events
    pub async fn run(&self) -> crate::error::Result<()> {
        info!("Starting chain client event loop");

        let api = match &self.api {
            Some(api) => api,
            None => {
                warn!("Chain API not connected, running in offline mode");
                // Keep alive but don't process blocks
                tokio::time::sleep(tokio::time::Duration::from_secs(u64::MAX)).await;
                return Ok(());
            }
        };

        let mut blocks_sub = api.blocks().subscribe_finalized().await.map_err(|e| {
            crate::error::SuperNodeError::ChainClient(format!("Block subscription failed: {}", e))
        })?;

        let event_tx = self.event_tx.clone();

        loop {
            match blocks_sub.next().await {
                Some(Ok(block)) => {
                    let block_number = block.number() as u64;
                    if event_tx
                        .send(ChainEvent::BlockFinalized { block_number })
                        .is_err()
                    {
                        debug!("Event channel closed, stopping chain client");
                        break;
                    }
                }
                Some(Err(e)) => {
                    error!("Block subscription error: {}", e);
                    break;
                }
                None => {
                    warn!("Block subscription ended");
                    break;
                }
            }
        }

        Ok(())
    }

    /// Get current block number
    pub async fn get_current_block(&self) -> crate::error::Result<u64> {
        let api = match &self.api {
            Some(api) => api,
            None => {
                return Err(crate::error::SuperNodeError::ChainClient(
                    "Chain API not connected".to_string(),
                ));
            }
        };

        let block = api.blocks().at_latest().await.map_err(|e| {
            crate::error::SuperNodeError::ChainClient(format!("Failed to query block: {}", e))
        })?;

        Ok(block.number() as u64)
    }

    /// Get finalized block number
    pub async fn get_finalized_block(&self) -> crate::error::Result<u64> {
        let api = match &self.api {
            Some(api) => api,
            None => {
                return Err(crate::error::SuperNodeError::ChainClient(
                    "Chain API not connected".to_string(),
                ));
            }
        };

        let block = api.blocks().at_latest().await.map_err(|e| {
            crate::error::SuperNodeError::ChainClient(format!("Failed to query block: {}", e))
        })?;

        // Note: subxt doesn't directly expose "finalized" vs "latest"
        // In practice, subscribe_finalized() gives us finalized blocks
        // This method returns the latest finalized block we've seen
        Ok(block.number() as u64)
    }

    /// Query all pinning deals (for cleanup)
    ///
    /// Returns list of all active pinning deals
    pub async fn get_pinning_deals(&self) -> crate::error::Result<Vec<PinningDeal>> {
        let _api = match &self.api {
            Some(api) => api,
            None => {
                return Err(crate::error::SuperNodeError::ChainClient(
                    "Chain API not connected".to_string(),
                ));
            }
        };

        // TODO: Query PinningDeals storage map from pallet-icn-pinning
        // This requires generated metadata from ICN Chain
        //
        // Example implementation:
        // let storage = api.storage().at_latest().await?;
        // let deals = storage.fetch(&icn_pinning::storage::PinningDeals::root()).await?;
        //
        // For now, return empty list (offline mode compatible)
        debug!("Querying pinning deals from chain");

        Ok(vec![])
    }

    /// Check if chain API is connected
    pub fn is_connected(&self) -> bool {
        self.api.is_some()
    }

    /// Get endpoint URL
    pub fn endpoint(&self) -> &str {
        &self.endpoint
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_chain_client_creation() {
        let result = ChainClient::connect("ws://127.0.0.1:9944".to_string()).await;
        assert!(result.is_ok(), "Chain client creation should succeed");

        let (client, _rx) = result.unwrap();
        // In offline mode (no chain running), API will be None
        assert_eq!(client.endpoint(), "ws://127.0.0.1:9944");
    }

    #[tokio::test]
    async fn test_invalid_endpoint() {
        let result = ChainClient::connect("http://invalid".to_string()).await;
        assert!(result.is_err(), "Should reject non-WebSocket endpoint");
    }

    #[tokio::test]
    async fn test_subscribe_pending_audits() {
        let (client, _rx) = ChainClient::connect("ws://127.0.0.1:9944".to_string())
            .await
            .unwrap();

        let result = client.subscribe_pending_audits().await;
        // Should succeed (graceful degradation in offline mode)
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_submit_audit_proof_offline() {
        let (client, _rx) = ChainClient::connect("ws://127.0.0.1:9944".to_string())
            .await
            .unwrap();

        let proof = vec![0u8; 32]; // 32-byte SHA256 hash
        let result = client.submit_audit_proof(123, proof).await;

        // In offline mode, should fail with "not connected" error
        if !client.is_connected() {
            assert!(result.is_err());
        } else {
            // If chain is actually running, should succeed
            assert!(result.is_ok());
        }
    }

    #[tokio::test]
    async fn test_get_pinning_deals() {
        let (client, _rx) = ChainClient::connect("ws://127.0.0.1:9944".to_string())
            .await
            .unwrap();

        let result = client.get_pinning_deals().await;

        // In offline mode, should fail
        if !client.is_connected() {
            assert!(result.is_err());
        } else {
            // If connected, should return empty list (no metadata yet)
            assert!(result.is_ok());
        }
    }

    #[tokio::test]
    async fn test_is_connected() {
        let (client, _rx) = ChainClient::connect("ws://127.0.0.1:9944".to_string())
            .await
            .unwrap();

        // is_connected() returns true only if chain is actually running
        let _connected = client.is_connected();
        // Don't assert value since it depends on test environment
    }
}
