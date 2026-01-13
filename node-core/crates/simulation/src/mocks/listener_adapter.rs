//! Adapter to bridge MockChainClient to ChainListener channel interface.
//!
//! The TaskExecutorService expects events via an `mpsc::Receiver<TaskEvent>`,
//! but MockChainClient provides a poll-based interface. This adapter bridges
//! the gap by polling the MockChainClient and pushing events to a channel.

use std::sync::Arc;
use std::time::Duration;

use nsn_lane1::{ListenerResult, TaskEvent};
use tokio::sync::{mpsc, RwLock};

use super::chain::MockChainClient;

/// Adapter that polls MockChainClient and pushes events to channel.
///
/// Bridges the poll-based MockChainClient interface to the channel-based
/// interface expected by TaskExecutorService.
///
/// # Example
///
/// ```rust,ignore
/// use nsn_simulation::mocks::{MockChainClient, ChainListenerAdapter};
/// use std::sync::Arc;
/// use tokio::sync::RwLock;
///
/// let chain = Arc::new(RwLock::new(MockChainClient::new()));
/// let (tx, rx) = tokio::sync::mpsc::channel(32);
/// let mut adapter = ChainListenerAdapter::new(chain.clone());
///
/// // Run adapter in background
/// tokio::spawn(async move {
///     adapter.run(tx).await.unwrap();
/// });
///
/// // Inject events via chain mock
/// chain.write().await.create_task(1, "model", "input", 1000);
///
/// // Receive via channel
/// let event = rx.recv().await.unwrap();
/// ```
pub struct ChainListenerAdapter {
    /// Mock chain client to poll for events.
    chain: Arc<RwLock<MockChainClient>>,
    /// Poll interval between event checks.
    poll_interval: Duration,
    /// Shutdown signal receiver.
    shutdown_rx: Option<mpsc::Receiver<()>>,
}

impl ChainListenerAdapter {
    /// Create a new adapter with the given chain client.
    pub fn new(chain: Arc<RwLock<MockChainClient>>) -> Self {
        Self {
            chain,
            poll_interval: Duration::from_millis(10),
            shutdown_rx: None,
        }
    }

    /// Configure the poll interval.
    pub fn with_poll_interval(mut self, interval: Duration) -> Self {
        self.poll_interval = interval;
        self
    }

    /// Set the shutdown receiver for graceful termination.
    pub fn with_shutdown(mut self, shutdown_rx: mpsc::Receiver<()>) -> Self {
        self.shutdown_rx = Some(shutdown_rx);
        self
    }

    /// Run the adapter, polling chain and pushing events to the channel.
    ///
    /// This method runs until:
    /// - The event channel is closed (receiver dropped)
    /// - A shutdown signal is received
    /// - The shutdown receiver is disconnected
    pub async fn run(&mut self, event_tx: mpsc::Sender<TaskEvent>) -> ListenerResult<()> {
        loop {
            // Check for shutdown signal
            if let Some(ref mut shutdown_rx) = self.shutdown_rx {
                match shutdown_rx.try_recv() {
                    Ok(()) => {
                        // Shutdown requested
                        break;
                    }
                    Err(mpsc::error::TryRecvError::Empty) => {
                        // No shutdown signal, continue
                    }
                    Err(mpsc::error::TryRecvError::Disconnected) => {
                        // Shutdown channel closed, exit
                        break;
                    }
                }
            }

            // Poll for events
            let event = {
                let mut chain = self.chain.write().await;
                chain.next_task_event()
            };

            if let Some(event) = event {
                // Try to send event
                if event_tx.send(event).await.is_err() {
                    // Channel closed, exit
                    break;
                }
            } else {
                // No events, sleep before next poll
                tokio::time::sleep(self.poll_interval).await;
            }
        }

        Ok(())
    }

    /// Run the adapter for a single poll cycle (for testing).
    ///
    /// Returns the number of events sent.
    pub async fn poll_once(&mut self, event_tx: &mpsc::Sender<TaskEvent>) -> usize {
        let mut count = 0;

        loop {
            let event = {
                let mut chain = self.chain.write().await;
                chain.next_task_event()
            };

            match event {
                Some(event) => {
                    if event_tx.send(event).await.is_ok() {
                        count += 1;
                    } else {
                        break;
                    }
                }
                None => break,
            }
        }

        count
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_adapter_creation() {
        let chain = Arc::new(RwLock::new(MockChainClient::new()));
        let adapter = ChainListenerAdapter::new(chain);

        assert_eq!(adapter.poll_interval, Duration::from_millis(10));
    }

    #[tokio::test]
    async fn test_poll_once_empty() {
        let chain = Arc::new(RwLock::new(MockChainClient::new()));
        let mut adapter = ChainListenerAdapter::new(chain);

        let (tx, _rx) = mpsc::channel(32);
        let count = adapter.poll_once(&tx).await;

        assert_eq!(count, 0);
    }

    #[tokio::test]
    async fn test_poll_once_with_events() {
        let chain = Arc::new(RwLock::new(MockChainClient::new()));

        // Inject events
        {
            let mut chain = chain.write().await;
            chain.create_task(1, "model-1", "QmInput1", 1000);
            chain.create_task(2, "model-2", "QmInput2", 2000);
        }

        let mut adapter = ChainListenerAdapter::new(chain);
        let (tx, mut rx) = mpsc::channel(32);

        let count = adapter.poll_once(&tx).await;
        assert_eq!(count, 2);

        // Verify events received
        let event1 = rx.recv().await.unwrap();
        assert!(matches!(
            event1,
            TaskEvent::Created { task_id: 1, .. }
        ));

        let event2 = rx.recv().await.unwrap();
        assert!(matches!(
            event2,
            TaskEvent::Created { task_id: 2, .. }
        ));
    }

    #[tokio::test]
    async fn test_run_with_shutdown() {
        let chain = Arc::new(RwLock::new(MockChainClient::new()));

        // Inject one event
        {
            let mut chain = chain.write().await;
            chain.create_task(1, "model-1", "QmInput1", 1000);
        }

        let (shutdown_tx, shutdown_rx) = mpsc::channel(1);
        let (event_tx, mut event_rx) = mpsc::channel(32);

        let mut adapter = ChainListenerAdapter::new(chain)
            .with_poll_interval(Duration::from_millis(1))
            .with_shutdown(shutdown_rx);

        // Run adapter in background
        let handle = tokio::spawn(async move {
            adapter.run(event_tx).await
        });

        // Wait for event to be received
        let event = tokio::time::timeout(Duration::from_millis(100), event_rx.recv())
            .await
            .expect("timeout waiting for event")
            .expect("channel closed");

        assert!(matches!(event, TaskEvent::Created { task_id: 1, .. }));

        // Send shutdown signal
        shutdown_tx.send(()).await.unwrap();

        // Adapter should exit gracefully
        let result = tokio::time::timeout(Duration::from_millis(100), handle)
            .await
            .expect("timeout waiting for adapter to stop");

        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_assigned_event() {
        let chain = Arc::new(RwLock::new(MockChainClient::new()));

        // Inject assign event
        {
            let mut chain = chain.write().await;
            chain.assign_task_to_me(1);
        }

        let mut adapter = ChainListenerAdapter::new(chain);
        let (tx, mut rx) = mpsc::channel(32);

        adapter.poll_once(&tx).await;

        let event = rx.recv().await.unwrap();
        assert!(matches!(event, TaskEvent::AssignedToMe { task_id: 1 }));
    }
}
