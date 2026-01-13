//! Result submitter for Lane 1 tasks.
//!
//! Submits execution results and status updates to the NSN chain.

use crate::error::{SubmissionError, SubmissionResult};
use async_trait::async_trait;
use sp_core::sr25519;
use subxt::{dynamic::tx, dynamic::Value, OnlineClient, PolkadotConfig};
use tracing::{debug, info, warn};

/// Trait for submitting results to chain.
///
/// This trait enables mock implementations for testing without requiring
/// a real chain connection.
#[async_trait]
pub trait ResultSubmitterTrait: Send + Sync {
    /// Connect to the chain.
    async fn connect(&mut self) -> SubmissionResult<()>;

    /// Disconnect from the chain.
    fn disconnect(&mut self);

    /// Check if connected to chain.
    fn is_connected(&self) -> bool;

    /// Notify chain that we're starting task execution.
    ///
    /// Calls `NsnTaskMarket::start_task(task_id)` extrinsic.
    async fn start_task(&mut self, task_id: u64) -> SubmissionResult<()>;

    /// Submit task execution result to chain.
    ///
    /// Calls `NsnTaskMarket::submit_result(task_id, output_cid, attestation_cid)` extrinsic.
    async fn submit_result(
        &mut self,
        task_id: u64,
        output_cid: &str,
        attestation_cid: Option<&str>,
    ) -> SubmissionResult<()>;

    /// Report task execution failure to chain.
    ///
    /// Calls `NsnTaskMarket::fail_task(task_id, reason)` extrinsic.
    async fn fail_task(&mut self, task_id: u64, reason: &str) -> SubmissionResult<()>;

    /// Cancel a task on chain.
    ///
    /// Calls `NsnTaskMarket::cancel_task(task_id, reason)` extrinsic.
    async fn cancel_task(&mut self, task_id: u64, reason: &str) -> SubmissionResult<()>;
}

/// Configuration for the result submitter.
#[derive(Debug, Clone)]
pub struct SubmitterConfig {
    /// Chain RPC endpoint.
    pub chain_rpc_url: String,
    /// Transaction timeout in milliseconds.
    pub tx_timeout_ms: u64,
}

impl Default for SubmitterConfig {
    fn default() -> Self {
        Self {
            chain_rpc_url: "ws://127.0.0.1:9944".to_string(),
            tx_timeout_ms: 30_000,
        }
    }
}

/// Result submitter for task marketplace extrinsics.
///
/// Handles submission of task lifecycle events to the NSN chain:
/// - Starting task execution
/// - Submitting results
/// - Reporting failures
pub struct ResultSubmitter {
    config: SubmitterConfig,
    client: Option<OnlineClient<PolkadotConfig>>,
    signer: subxt::tx::PairSigner<PolkadotConfig, sr25519::Pair>,
}

impl ResultSubmitter {
    /// Create a new result submitter.
    ///
    /// # Arguments
    /// * `config` - Submitter configuration
    /// * `keypair` - Sr25519 keypair for signing transactions
    pub fn new(config: SubmitterConfig, keypair: sr25519::Pair) -> Self {
        Self {
            config,
            client: None,
            signer: subxt::tx::PairSigner::new(keypair),
        }
    }

    /// Get the submitter configuration.
    pub fn config(&self) -> &SubmitterConfig {
        &self.config
    }

    /// Connect to the chain.
    pub async fn connect(&mut self) -> SubmissionResult<()> {
        if self.client.is_some() {
            return Ok(());
        }

        info!(
            rpc_url = %self.config.chain_rpc_url,
            "Connecting to chain for submissions"
        );

        let client = OnlineClient::<PolkadotConfig>::from_url(&self.config.chain_rpc_url)
            .await
            .map_err(|e| SubmissionError::Connection(e.to_string()))?;

        self.client = Some(client);
        Ok(())
    }

    /// Disconnect from the chain.
    pub fn disconnect(&mut self) {
        self.client = None;
    }

    /// Check if connected to chain.
    pub fn is_connected(&self) -> bool {
        self.client.is_some()
    }

    /// Notify chain that we're starting task execution.
    ///
    /// Calls `NsnTaskMarket::start_task(task_id)` extrinsic.
    pub async fn start_task(&mut self, task_id: u64) -> SubmissionResult<()> {
        self.connect().await?;

        let client = self.client.as_ref().ok_or_else(|| {
            SubmissionError::Connection("client not connected".to_string())
        })?;

        debug!(task_id = task_id, "Submitting start_task extrinsic");

        let call = tx(
            "NsnTaskMarket",
            "start_task",
            vec![Value::u128(task_id as u128)],
        );

        client
            .tx()
            .sign_and_submit_default(&call, &self.signer)
            .await
            .map_err(|e| SubmissionError::ExtrinsicFailed(e.to_string()))?;

        info!(task_id = task_id, "start_task submitted successfully");
        Ok(())
    }

    /// Submit task execution result to chain.
    ///
    /// Calls `NsnTaskMarket::submit_result(task_id, output_cid, attestation_cid)` extrinsic.
    pub async fn submit_result(
        &mut self,
        task_id: u64,
        output_cid: &str,
        attestation_cid: Option<&str>,
    ) -> SubmissionResult<()> {
        self.connect().await?;

        let client = self.client.as_ref().ok_or_else(|| {
            SubmissionError::Connection("client not connected".to_string())
        })?;

        debug!(
            task_id = task_id,
            output_cid = %output_cid,
            has_attestation = attestation_cid.is_some(),
            "Submitting submit_result extrinsic"
        );

        let attestation_value = match attestation_cid {
            Some(cid) => Value::unnamed_variant("Some", vec![Value::from_bytes(cid.as_bytes())]),
            None => Value::unnamed_variant("None", Vec::new()),
        };

        let call = tx(
            "NsnTaskMarket",
            "submit_result",
            vec![
                Value::u128(task_id as u128),
                Value::from_bytes(output_cid.as_bytes()),
                attestation_value,
            ],
        );

        client
            .tx()
            .sign_and_submit_default(&call, &self.signer)
            .await
            .map_err(|e| SubmissionError::ExtrinsicFailed(e.to_string()))?;

        info!(
            task_id = task_id,
            output_cid = %output_cid,
            "submit_result submitted successfully"
        );
        Ok(())
    }

    /// Report task execution failure to chain.
    ///
    /// Calls `NsnTaskMarket::fail_task(task_id, reason)` extrinsic.
    pub async fn fail_task(&mut self, task_id: u64, reason: &str) -> SubmissionResult<()> {
        self.connect().await?;

        let client = self.client.as_ref().ok_or_else(|| {
            SubmissionError::Connection("client not connected".to_string())
        })?;

        warn!(
            task_id = task_id,
            reason = %reason,
            "Submitting fail_task extrinsic"
        );

        let call = tx(
            "NsnTaskMarket",
            "fail_task",
            vec![
                Value::u128(task_id as u128),
                Value::from_bytes(reason.as_bytes()),
            ],
        );

        client
            .tx()
            .sign_and_submit_default(&call, &self.signer)
            .await
            .map_err(|e| SubmissionError::ExtrinsicFailed(e.to_string()))?;

        info!(task_id = task_id, "fail_task submitted successfully");
        Ok(())
    }

    /// Cancel a task on chain (if supported).
    ///
    /// Calls `NsnTaskMarket::cancel_task(task_id, reason)` extrinsic.
    pub async fn cancel_task(&mut self, task_id: u64, reason: &str) -> SubmissionResult<()> {
        self.connect().await?;

        let client = self.client.as_ref().ok_or_else(|| {
            SubmissionError::Connection("client not connected".to_string())
        })?;

        debug!(
            task_id = task_id,
            reason = %reason,
            "Submitting cancel_task extrinsic"
        );

        let call = tx(
            "NsnTaskMarket",
            "cancel_task",
            vec![
                Value::u128(task_id as u128),
                Value::from_bytes(reason.as_bytes()),
            ],
        );

        client
            .tx()
            .sign_and_submit_default(&call, &self.signer)
            .await
            .map_err(|e| SubmissionError::ExtrinsicFailed(e.to_string()))?;

        info!(task_id = task_id, "cancel_task submitted successfully");
        Ok(())
    }
}

#[async_trait]
impl ResultSubmitterTrait for ResultSubmitter {
    async fn connect(&mut self) -> SubmissionResult<()> {
        self.connect().await
    }

    fn disconnect(&mut self) {
        self.disconnect()
    }

    fn is_connected(&self) -> bool {
        self.is_connected()
    }

    async fn start_task(&mut self, task_id: u64) -> SubmissionResult<()> {
        self.start_task(task_id).await
    }

    async fn submit_result(
        &mut self,
        task_id: u64,
        output_cid: &str,
        attestation_cid: Option<&str>,
    ) -> SubmissionResult<()> {
        self.submit_result(task_id, output_cid, attestation_cid).await
    }

    async fn fail_task(&mut self, task_id: u64, reason: &str) -> SubmissionResult<()> {
        self.fail_task(task_id, reason).await
    }

    async fn cancel_task(&mut self, task_id: u64, reason: &str) -> SubmissionResult<()> {
        self.cancel_task(task_id, reason).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sp_core::Pair;

    #[test]
    fn test_submitter_config_default() {
        let config = SubmitterConfig::default();
        assert_eq!(config.chain_rpc_url, "ws://127.0.0.1:9944");
        assert_eq!(config.tx_timeout_ms, 30_000);
    }

    #[test]
    fn test_submitter_creation() {
        let config = SubmitterConfig::default();
        // Generate a test keypair
        let keypair = sr25519::Pair::from_string("//Alice", None).unwrap();
        let submitter = ResultSubmitter::new(config.clone(), keypair);

        assert!(!submitter.is_connected());
        assert_eq!(submitter.config().chain_rpc_url, config.chain_rpc_url);
    }

    #[test]
    fn test_disconnect() {
        let config = SubmitterConfig::default();
        let keypair = sr25519::Pair::from_string("//Alice", None).unwrap();
        let mut submitter = ResultSubmitter::new(config, keypair);

        assert!(!submitter.is_connected());
        submitter.disconnect();
        assert!(!submitter.is_connected());
    }
}
