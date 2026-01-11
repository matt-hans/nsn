//! Mock chain client for testing on-chain interactions.

use std::collections::{HashMap, VecDeque};

use nsn_lane1::TaskEvent;
use nsn_scheduler::EpochEvent;
use nsn_scheduler::task_queue::Priority;
use nsn_types::EpochInfo;

/// A submitted extrinsic for verification.
#[derive(Debug, Clone)]
pub enum SubmittedExtrinsic {
    /// Start task execution
    StartTask { task_id: u64 },
    /// Submit task result
    SubmitResult {
        task_id: u64,
        output_cid: String,
    },
    /// Fail task
    FailTask { task_id: u64, reason: String },
}

/// Mock chain state for tracking on-chain data.
#[derive(Debug, Clone, Default)]
pub struct MockChainState {
    /// Current epoch
    pub current_epoch: u64,
    /// Current slot
    pub current_slot: u64,
    /// Active lane (0 or 1)
    pub active_lane: u8,
    /// Directors for current epoch
    pub directors: Vec<String>,
    /// Task assignments (task_id -> executor account)
    pub task_assignments: HashMap<u64, String>,
    /// Completed tasks
    pub completed_tasks: Vec<u64>,
    /// Failed tasks
    pub failed_tasks: Vec<u64>,
}

impl MockChainState {
    /// Create a new mock chain state.
    pub fn new() -> Self {
        Self::default()
    }

    /// Advance to next epoch.
    pub fn advance_epoch(&mut self) {
        self.current_epoch += 1;
        self.current_slot = 0;
    }

    /// Advance to next slot.
    pub fn advance_slot(&mut self) {
        self.current_slot += 1;
    }

    /// Set active lane.
    pub fn set_active_lane(&mut self, lane: u8) {
        self.active_lane = lane;
    }

    /// Get current epoch info.
    pub fn epoch_info(&self) -> EpochInfo {
        EpochInfo {
            epoch: self.current_epoch,
            slot: self.current_slot,
            block_number: self.current_epoch * 100 + self.current_slot,
            active_lane: self.active_lane,
        }
    }
}

/// Mock chain client for simulating on-chain events.
///
/// Provides event injection and extrinsic tracking for testing.
///
/// # Example
///
/// ```rust,ignore
/// use nsn_simulation::mocks::MockChainClient;
///
/// let mut client = MockChainClient::new();
///
/// // Inject task created event
/// client.inject_task_event(TaskEvent::Created { ... });
///
/// // Verify submitted extrinsics
/// assert_eq!(client.submitted_extrinsics().len(), 1);
/// ```
#[derive(Debug, Clone, Default)]
pub struct MockChainClient {
    /// Chain state
    pub state: MockChainState,
    /// Pending task events to emit
    task_event_queue: VecDeque<TaskEvent>,
    /// Pending epoch events to emit
    epoch_event_queue: VecDeque<EpochEvent>,
    /// Submitted extrinsics (for verification)
    pub submitted_extrinsics: VecDeque<SubmittedExtrinsic>,
    /// Whether to fail extrinsic submissions
    fail_submissions: bool,
    /// Account ID of this node
    my_account: String,
}

impl MockChainClient {
    /// Create a new mock chain client.
    pub fn new() -> Self {
        Self {
            state: MockChainState::new(),
            task_event_queue: VecDeque::new(),
            epoch_event_queue: VecDeque::new(),
            submitted_extrinsics: VecDeque::new(),
            fail_submissions: false,
            my_account: "5GrwvaEF...".to_string(),
        }
    }

    /// Configure the account ID for this node.
    pub fn with_account(mut self, account: String) -> Self {
        self.my_account = account;
        self
    }

    /// Configure failure mode for submissions.
    pub fn with_fail_submissions(mut self, fail: bool) -> Self {
        self.fail_submissions = fail;
        self
    }

    /// Inject a task event to be emitted.
    pub fn inject_task_event(&mut self, event: TaskEvent) {
        self.task_event_queue.push_back(event);
    }

    /// Inject an epoch event to be emitted.
    pub fn inject_epoch_event(&mut self, event: EpochEvent) {
        self.epoch_event_queue.push_back(event);
    }

    /// Create a TaskCreated event.
    pub fn create_task(&mut self, task_id: u64, model_id: &str, input_cid: &str, reward: u128) {
        self.inject_task_event(TaskEvent::Created {
            task_id,
            model_id: model_id.to_string(),
            input_cid: input_cid.to_string(),
            priority: Priority::Normal,
            reward,
        });
    }

    /// Assign task to this node.
    pub fn assign_task_to_me(&mut self, task_id: u64) {
        self.state
            .task_assignments
            .insert(task_id, self.my_account.clone());
        self.inject_task_event(TaskEvent::AssignedToMe { task_id });
    }

    /// Assign task to another node.
    pub fn assign_task_to_other(&mut self, task_id: u64, executor: &str) {
        self.state
            .task_assignments
            .insert(task_id, executor.to_string());
        self.inject_task_event(TaskEvent::AssignedToOther {
            task_id,
            executor: executor.to_string(),
        });
    }

    /// Emit OnDeck epoch event.
    pub fn emit_on_deck(&mut self, am_director: bool) {
        self.inject_epoch_event(EpochEvent::OnDeck {
            epoch: self.state.epoch_info(),
            am_director,
        });
    }

    /// Emit EpochStarted event.
    pub fn emit_epoch_started(&mut self) {
        self.inject_epoch_event(EpochEvent::EpochStarted {
            epoch: self.state.epoch_info(),
        });
    }

    /// Emit EpochEnded event.
    pub fn emit_epoch_ended(&mut self) {
        let epoch = self.state.current_epoch;
        self.inject_epoch_event(EpochEvent::EpochEnded { epoch });
    }

    /// Get the next task event.
    pub fn next_task_event(&mut self) -> Option<TaskEvent> {
        self.task_event_queue.pop_front()
    }

    /// Get the next epoch event.
    pub fn next_epoch_event(&mut self) -> Option<EpochEvent> {
        self.epoch_event_queue.pop_front()
    }

    /// Check if there are pending task events.
    pub fn has_pending_task_events(&self) -> bool {
        !self.task_event_queue.is_empty()
    }

    /// Check if there are pending epoch events.
    pub fn has_pending_epoch_events(&self) -> bool {
        !self.epoch_event_queue.is_empty()
    }

    /// Submit start_task extrinsic.
    pub async fn start_task(&mut self, task_id: u64) -> Result<(), String> {
        if self.fail_submissions {
            return Err("Mock failure mode enabled".to_string());
        }
        self.submitted_extrinsics
            .push_back(SubmittedExtrinsic::StartTask { task_id });
        Ok(())
    }

    /// Submit task result extrinsic.
    pub async fn submit_result(&mut self, task_id: u64, output_cid: String) -> Result<(), String> {
        if self.fail_submissions {
            return Err("Mock failure mode enabled".to_string());
        }
        self.state.completed_tasks.push(task_id);
        self.submitted_extrinsics
            .push_back(SubmittedExtrinsic::SubmitResult { task_id, output_cid });
        Ok(())
    }

    /// Submit fail_task extrinsic.
    pub async fn fail_task(&mut self, task_id: u64, reason: String) -> Result<(), String> {
        if self.fail_submissions {
            return Err("Mock failure mode enabled".to_string());
        }
        self.state.failed_tasks.push(task_id);
        self.submitted_extrinsics
            .push_back(SubmittedExtrinsic::FailTask { task_id, reason });
        Ok(())
    }

    /// Get count of submitted extrinsics.
    pub fn submission_count(&self) -> usize {
        self.submitted_extrinsics.len()
    }

    /// Clear submitted extrinsics.
    pub fn clear_submissions(&mut self) {
        self.submitted_extrinsics.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chain_state_epoch_advance() {
        let mut state = MockChainState::new();
        assert_eq!(state.current_epoch, 0);

        state.advance_epoch();
        assert_eq!(state.current_epoch, 1);
        assert_eq!(state.current_slot, 0);

        state.advance_slot();
        assert_eq!(state.current_slot, 1);
    }

    #[test]
    fn test_chain_state_epoch_info() {
        let mut state = MockChainState::new();
        state.current_epoch = 5;
        state.current_slot = 10;
        state.active_lane = 0;

        let info = state.epoch_info();
        assert_eq!(info.epoch, 5);
        assert_eq!(info.slot, 10);
        assert_eq!(info.active_lane, 0);
    }

    #[tokio::test]
    async fn test_task_event_injection() {
        let mut client = MockChainClient::new();

        client.create_task(1, "model-1", "QmInput", 1000);

        let event = client.next_task_event().unwrap();
        match event {
            TaskEvent::Created { task_id, model_id, .. } => {
                assert_eq!(task_id, 1);
                assert_eq!(model_id, "model-1");
            }
            _ => panic!("Expected Created event"),
        }
    }

    #[tokio::test]
    async fn test_epoch_event_injection() {
        let mut client = MockChainClient::new();

        client.emit_on_deck(true);

        let event = client.next_epoch_event().unwrap();
        match event {
            EpochEvent::OnDeck { am_director, .. } => {
                assert!(am_director);
            }
            _ => panic!("Expected OnDeck event"),
        }
    }

    #[tokio::test]
    async fn test_extrinsic_submission() {
        let mut client = MockChainClient::new();

        client.start_task(1).await.unwrap();
        client
            .submit_result(1, "QmResult".to_string())
            .await
            .unwrap();

        assert_eq!(client.submission_count(), 2);
        assert_eq!(client.state.completed_tasks.len(), 1);
    }

    #[tokio::test]
    async fn test_fail_mode() {
        let mut client = MockChainClient::new().with_fail_submissions(true);

        let result = client.start_task(1).await;
        assert!(result.is_err());
    }
}
