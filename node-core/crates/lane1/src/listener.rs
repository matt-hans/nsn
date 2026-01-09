//! Chain event listener for Lane 1 task marketplace.
//!
//! Subscribes to task-market pallet events and routes them to the scheduler.

use crate::error::{ListenerError, ListenerResult};
use nsn_scheduler::task_queue::Priority;
use tokio::sync::mpsc;
use tracing::{debug, info, warn};

/// Task event types from the chain.
#[derive(Debug, Clone)]
pub enum TaskEvent {
    /// New task created on chain.
    Created {
        /// Task ID.
        task_id: u64,
        /// Model to use for execution.
        model_id: String,
        /// Input data CID.
        input_cid: String,
        /// Task priority.
        priority: Priority,
        /// Reward amount.
        reward: u128,
    },
    /// Task assigned to our node.
    AssignedToMe {
        /// Task ID.
        task_id: u64,
    },
    /// Task assigned to another node.
    AssignedToOther {
        /// Task ID.
        task_id: u64,
        /// Assigned executor.
        executor: String,
    },
    /// Task verified by validators.
    Verified {
        /// Task ID.
        task_id: u64,
    },
    /// Task was rejected.
    Rejected {
        /// Task ID.
        task_id: u64,
        /// Rejection reason.
        reason: String,
    },
    /// Task failed.
    Failed {
        /// Task ID.
        task_id: u64,
        /// Failure reason.
        reason: String,
    },
}

/// Configuration for the chain listener.
#[derive(Debug, Clone)]
pub struct ListenerConfig {
    /// Chain RPC endpoint.
    pub chain_rpc_url: String,
    /// Event buffer size.
    pub event_buffer_size: usize,
    /// Reconnect interval on failure (ms).
    pub reconnect_interval_ms: u64,
}

impl Default for ListenerConfig {
    fn default() -> Self {
        Self {
            chain_rpc_url: "ws://127.0.0.1:9944".to_string(),
            event_buffer_size: 256,
            reconnect_interval_ms: 5000,
        }
    }
}

/// Chain event listener for task marketplace events.
///
/// Subscribes to the NSN chain and listens for task-related events from
/// the `pallet-nsn-task-market`. Events are routed to the task executor
/// service via an mpsc channel.
pub struct ChainListener {
    config: ListenerConfig,
    my_account: String,
    event_tx: mpsc::Sender<TaskEvent>,
    shutdown_rx: Option<mpsc::Receiver<()>>,
}

impl ChainListener {
    /// Create a new chain listener.
    ///
    /// # Arguments
    /// * `config` - Listener configuration
    /// * `my_account` - This node's account ID (to detect assignments)
    /// * `event_tx` - Channel to send events to
    pub fn new(
        config: ListenerConfig,
        my_account: String,
        event_tx: mpsc::Sender<TaskEvent>,
    ) -> Self {
        Self {
            config,
            my_account,
            event_tx,
            shutdown_rx: None,
        }
    }

    /// Set the shutdown receiver for graceful termination.
    pub fn with_shutdown(mut self, shutdown_rx: mpsc::Receiver<()>) -> Self {
        self.shutdown_rx = Some(shutdown_rx);
        self
    }

    /// Get the listener configuration.
    pub fn config(&self) -> &ListenerConfig {
        &self.config
    }

    /// Run the event listener loop.
    ///
    /// This method subscribes to chain events and forwards task-related
    /// events to the executor service. It handles reconnection on failures.
    pub async fn run(&mut self) -> ListenerResult<()> {
        info!(
            rpc_url = %self.config.chain_rpc_url,
            account = %self.my_account,
            "Starting chain event listener"
        );

        loop {
            match self.subscribe_and_listen().await {
                Ok(()) => {
                    info!("Chain listener stopped gracefully");
                    return Ok(());
                }
                Err(e) => {
                    warn!(error = %e, "Chain subscription failed, reconnecting...");
                    tokio::time::sleep(std::time::Duration::from_millis(
                        self.config.reconnect_interval_ms,
                    ))
                    .await;
                }
            }
        }
    }

    /// Subscribe to chain events and process them.
    async fn subscribe_and_listen(&mut self) -> ListenerResult<()> {
        // In a real implementation, this would use subxt to subscribe to
        // finalized blocks and extract task-market pallet events.
        //
        // For now, we implement the event processing logic that will be
        // connected to the actual chain subscription.

        debug!("Connecting to chain at {}", self.config.chain_rpc_url);

        // Placeholder: The actual implementation would look like:
        //
        // let client = OnlineClient::<PolkadotConfig>::from_url(&self.config.chain_rpc_url).await?;
        // let mut blocks = client.blocks().subscribe_finalized().await?;
        //
        // while let Some(block) = blocks.next().await {
        //     let block = block?;
        //     for event in block.events().await?.iter() {
        //         self.process_event(event).await?;
        //     }
        // }

        // For now, return Ok to allow compilation. The actual chain subscription
        // will be connected when integrating with the live chain.
        if let Some(ref mut shutdown_rx) = self.shutdown_rx {
            let _ = shutdown_rx.recv().await;
        }

        Ok(())
    }

    /// Process a TaskCreated event.
    pub async fn on_task_created(
        &self,
        task_id: u64,
        model_id: String,
        input_cid: String,
        priority: Priority,
        reward: u128,
    ) -> ListenerResult<()> {
        debug!(
            task_id = task_id,
            model_id = %model_id,
            "TaskCreated event received"
        );

        let event = TaskEvent::Created {
            task_id,
            model_id,
            input_cid,
            priority,
            reward,
        };

        self.event_tx
            .send(event)
            .await
            .map_err(|_| ListenerError::Subscription("event channel closed".to_string()))?;

        Ok(())
    }

    /// Process a TaskAssigned event.
    pub async fn on_task_assigned(
        &self,
        task_id: u64,
        executor: String,
    ) -> ListenerResult<()> {
        debug!(
            task_id = task_id,
            executor = %executor,
            my_account = %self.my_account,
            "TaskAssigned event received"
        );

        let event = if executor == self.my_account {
            TaskEvent::AssignedToMe { task_id }
        } else {
            TaskEvent::AssignedToOther { task_id, executor }
        };

        self.event_tx
            .send(event)
            .await
            .map_err(|_| ListenerError::Subscription("event channel closed".to_string()))?;

        Ok(())
    }

    /// Process a TaskVerified event.
    pub async fn on_task_verified(&self, task_id: u64) -> ListenerResult<()> {
        debug!(task_id = task_id, "TaskVerified event received");

        let event = TaskEvent::Verified { task_id };

        self.event_tx
            .send(event)
            .await
            .map_err(|_| ListenerError::Subscription("event channel closed".to_string()))?;

        Ok(())
    }

    /// Process a TaskRejected event.
    pub async fn on_task_rejected(
        &self,
        task_id: u64,
        reason: String,
    ) -> ListenerResult<()> {
        debug!(
            task_id = task_id,
            reason = %reason,
            "TaskRejected event received"
        );

        let event = TaskEvent::Rejected { task_id, reason };

        self.event_tx
            .send(event)
            .await
            .map_err(|_| ListenerError::Subscription("event channel closed".to_string()))?;

        Ok(())
    }

    /// Process a TaskFailed event.
    pub async fn on_task_failed(
        &self,
        task_id: u64,
        reason: String,
    ) -> ListenerResult<()> {
        debug!(
            task_id = task_id,
            reason = %reason,
            "TaskFailed event received"
        );

        let event = TaskEvent::Failed { task_id, reason };

        self.event_tx
            .send(event)
            .await
            .map_err(|_| ListenerError::Subscription("event channel closed".to_string()))?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_listener_creation() {
        let config = ListenerConfig::default();
        let (tx, _rx) = mpsc::channel(16);
        let listener = ChainListener::new(config.clone(), "5GrwvaEF...".to_string(), tx);

        assert_eq!(listener.config().chain_rpc_url, config.chain_rpc_url);
    }

    #[tokio::test]
    async fn test_on_task_created() {
        let config = ListenerConfig::default();
        let (tx, mut rx) = mpsc::channel(16);
        let listener = ChainListener::new(config, "5GrwvaEF...".to_string(), tx);

        listener
            .on_task_created(1, "flux-schnell".to_string(), "QmInput".to_string(), Priority::Normal, 1000)
            .await
            .unwrap();

        let event = rx.recv().await.unwrap();
        match event {
            TaskEvent::Created { task_id, model_id, .. } => {
                assert_eq!(task_id, 1);
                assert_eq!(model_id, "flux-schnell");
            }
            _ => panic!("Expected TaskEvent::Created"),
        }
    }

    #[tokio::test]
    async fn test_on_task_assigned_to_me() {
        let config = ListenerConfig::default();
        let (tx, mut rx) = mpsc::channel(16);
        let my_account = "5GrwvaEF...".to_string();
        let listener = ChainListener::new(config, my_account.clone(), tx);

        listener
            .on_task_assigned(1, my_account)
            .await
            .unwrap();

        let event = rx.recv().await.unwrap();
        assert!(matches!(event, TaskEvent::AssignedToMe { task_id: 1 }));
    }

    #[tokio::test]
    async fn test_on_task_assigned_to_other() {
        let config = ListenerConfig::default();
        let (tx, mut rx) = mpsc::channel(16);
        let listener = ChainListener::new(config, "5GrwvaEF...".to_string(), tx);

        listener
            .on_task_assigned(1, "5FHneW46...".to_string())
            .await
            .unwrap();

        let event = rx.recv().await.unwrap();
        match event {
            TaskEvent::AssignedToOther { task_id, executor } => {
                assert_eq!(task_id, 1);
                assert_eq!(executor, "5FHneW46...");
            }
            _ => panic!("Expected TaskEvent::AssignedToOther"),
        }
    }

    #[tokio::test]
    async fn test_on_task_verified() {
        let config = ListenerConfig::default();
        let (tx, mut rx) = mpsc::channel(16);
        let listener = ChainListener::new(config, "5GrwvaEF...".to_string(), tx);

        listener.on_task_verified(42).await.unwrap();

        let event = rx.recv().await.unwrap();
        assert!(matches!(event, TaskEvent::Verified { task_id: 42 }));
    }
}
