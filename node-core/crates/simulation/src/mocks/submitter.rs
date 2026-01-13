//! Mock ResultSubmitter for testing Lane 1 result submission without actual chain.

use std::collections::VecDeque;

use async_trait::async_trait;
use nsn_lane1::{ResultSubmitterTrait, SubmissionError, SubmissionResult};

/// A recorded submission event for verification.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SubmissionEvent {
    /// Start task execution notification.
    StartTask {
        /// Task ID.
        task_id: u64,
    },
    /// Task result submission.
    SubmitResult {
        /// Task ID.
        task_id: u64,
        /// Output CID.
        output_cid: String,
        /// Attestation CID (optional).
        attestation_cid: Option<String>,
    },
    /// Task failure notification.
    FailTask {
        /// Task ID.
        task_id: u64,
        /// Failure reason.
        reason: String,
    },
    /// Task cancellation.
    CancelTask {
        /// Task ID.
        task_id: u64,
        /// Cancellation reason.
        reason: String,
    },
}

/// Mock ResultSubmitter for simulation testing.
///
/// Tracks all submitted events for verification without requiring a real chain.
///
/// # Example
///
/// ```rust,ignore
/// use nsn_simulation::mocks::MockResultSubmitter;
///
/// let mut submitter = MockResultSubmitter::new();
/// submitter.start_task(1).await?;
/// submitter.submit_result(1, "QmOutput", None).await?;
///
/// assert_eq!(submitter.submission_count(), 2);
/// ```
#[derive(Debug, Clone)]
pub struct MockResultSubmitter {
    /// Submitted events (for verification).
    pub submissions: VecDeque<SubmissionEvent>,
    /// Whether to fail submissions.
    fail_mode: bool,
    /// Whether connected to chain.
    connected: bool,
}

impl Default for MockResultSubmitter {
    fn default() -> Self {
        Self::new()
    }
}

impl MockResultSubmitter {
    /// Create a new mock submitter.
    pub fn new() -> Self {
        Self {
            submissions: VecDeque::new(),
            fail_mode: false,
            connected: false,
        }
    }

    /// Configure failure mode (all submissions will fail).
    pub fn with_fail_mode(mut self, fail: bool) -> Self {
        self.fail_mode = fail;
        self
    }

    /// Enable failure mode dynamically.
    pub fn set_fail_mode(&mut self, fail: bool) {
        self.fail_mode = fail;
    }

    /// Get the number of submission events.
    pub fn submission_count(&self) -> usize {
        self.submissions.len()
    }

    /// Clear submission events.
    pub fn clear_submissions(&mut self) {
        self.submissions.clear();
    }

    /// Get submissions for a specific task.
    pub fn submissions_for_task(&self, task_id: u64) -> Vec<&SubmissionEvent> {
        self.submissions
            .iter()
            .filter(|e| match e {
                SubmissionEvent::StartTask { task_id: id } => *id == task_id,
                SubmissionEvent::SubmitResult { task_id: id, .. } => *id == task_id,
                SubmissionEvent::FailTask { task_id: id, .. } => *id == task_id,
                SubmissionEvent::CancelTask { task_id: id, .. } => *id == task_id,
            })
            .collect()
    }

    /// Check if a task has a start event.
    pub fn has_start_task(&self, task_id: u64) -> bool {
        self.submissions
            .iter()
            .any(|e| matches!(e, SubmissionEvent::StartTask { task_id: id } if *id == task_id))
    }

    /// Check if a task has a result submission.
    pub fn has_submit_result(&self, task_id: u64) -> bool {
        self.submissions
            .iter()
            .any(|e| matches!(e, SubmissionEvent::SubmitResult { task_id: id, .. } if *id == task_id))
    }

    /// Check if a task has a failure submission.
    pub fn has_fail_task(&self, task_id: u64) -> bool {
        self.submissions
            .iter()
            .any(|e| matches!(e, SubmissionEvent::FailTask { task_id: id, .. } if *id == task_id))
    }
}

#[async_trait]
impl ResultSubmitterTrait for MockResultSubmitter {
    async fn connect(&mut self) -> SubmissionResult<()> {
        self.connected = true;
        Ok(())
    }

    fn disconnect(&mut self) {
        self.connected = false;
    }

    fn is_connected(&self) -> bool {
        self.connected
    }

    async fn start_task(&mut self, task_id: u64) -> SubmissionResult<()> {
        if self.fail_mode {
            return Err(SubmissionError::ExtrinsicFailed(
                "Mock failure mode enabled".to_string(),
            ));
        }

        self.submissions
            .push_back(SubmissionEvent::StartTask { task_id });
        Ok(())
    }

    async fn submit_result(
        &mut self,
        task_id: u64,
        output_cid: &str,
        attestation_cid: Option<&str>,
    ) -> SubmissionResult<()> {
        if self.fail_mode {
            return Err(SubmissionError::ExtrinsicFailed(
                "Mock failure mode enabled".to_string(),
            ));
        }

        self.submissions.push_back(SubmissionEvent::SubmitResult {
            task_id,
            output_cid: output_cid.to_string(),
            attestation_cid: attestation_cid.map(|s| s.to_string()),
        });
        Ok(())
    }

    async fn fail_task(&mut self, task_id: u64, reason: &str) -> SubmissionResult<()> {
        if self.fail_mode {
            return Err(SubmissionError::ExtrinsicFailed(
                "Mock failure mode enabled".to_string(),
            ));
        }

        self.submissions.push_back(SubmissionEvent::FailTask {
            task_id,
            reason: reason.to_string(),
        });
        Ok(())
    }

    async fn cancel_task(&mut self, task_id: u64, reason: &str) -> SubmissionResult<()> {
        if self.fail_mode {
            return Err(SubmissionError::ExtrinsicFailed(
                "Mock failure mode enabled".to_string(),
            ));
        }

        self.submissions.push_back(SubmissionEvent::CancelTask {
            task_id,
            reason: reason.to_string(),
        });
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_start_task() {
        let mut submitter = MockResultSubmitter::new();

        submitter.start_task(1).await.unwrap();

        assert_eq!(submitter.submission_count(), 1);
        assert!(submitter.has_start_task(1));
    }

    #[tokio::test]
    async fn test_submit_result() {
        let mut submitter = MockResultSubmitter::new();

        submitter
            .submit_result(1, "QmOutput1", None)
            .await
            .unwrap();

        assert_eq!(submitter.submission_count(), 1);
        assert!(submitter.has_submit_result(1));
    }

    #[tokio::test]
    async fn test_submit_result_with_attestation() {
        let mut submitter = MockResultSubmitter::new();

        submitter
            .submit_result(1, "QmOutput1", Some("QmAttestation1"))
            .await
            .unwrap();

        let submissions = submitter.submissions_for_task(1);
        assert_eq!(submissions.len(), 1);

        match submissions[0] {
            SubmissionEvent::SubmitResult {
                attestation_cid, ..
            } => {
                assert_eq!(attestation_cid.as_deref(), Some("QmAttestation1"));
            }
            _ => panic!("Expected SubmitResult event"),
        }
    }

    #[tokio::test]
    async fn test_fail_task() {
        let mut submitter = MockResultSubmitter::new();

        submitter.fail_task(1, "execution failed").await.unwrap();

        assert_eq!(submitter.submission_count(), 1);
        assert!(submitter.has_fail_task(1));
    }

    #[tokio::test]
    async fn test_cancel_task() {
        let mut submitter = MockResultSubmitter::new();

        submitter.cancel_task(1, "user cancelled").await.unwrap();

        assert_eq!(submitter.submission_count(), 1);
        let event = &submitter.submissions[0];
        assert!(matches!(event, SubmissionEvent::CancelTask { task_id: 1, .. }));
    }

    #[tokio::test]
    async fn test_fail_mode() {
        let mut submitter = MockResultSubmitter::new().with_fail_mode(true);

        let result = submitter.start_task(1).await;
        assert!(result.is_err());

        let result = submitter.submit_result(1, "QmOutput", None).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_full_lifecycle() {
        let mut submitter = MockResultSubmitter::new();

        // Full successful lifecycle
        submitter.start_task(1).await.unwrap();
        submitter
            .submit_result(1, "QmOutput1", None)
            .await
            .unwrap();

        assert_eq!(submitter.submission_count(), 2);
        assert!(submitter.has_start_task(1));
        assert!(submitter.has_submit_result(1));

        let task_submissions = submitter.submissions_for_task(1);
        assert_eq!(task_submissions.len(), 2);
    }

    #[tokio::test]
    async fn test_connect_disconnect() {
        let mut submitter = MockResultSubmitter::new();

        assert!(!submitter.is_connected());
        submitter.connect().await.unwrap();
        assert!(submitter.is_connected());
        submitter.disconnect();
        assert!(!submitter.is_connected());
    }

    #[tokio::test]
    async fn test_clear_submissions() {
        let mut submitter = MockResultSubmitter::new();

        submitter.start_task(1).await.unwrap();
        submitter.start_task(2).await.unwrap();
        assert_eq!(submitter.submission_count(), 2);

        submitter.clear_submissions();
        assert_eq!(submitter.submission_count(), 0);
    }
}
