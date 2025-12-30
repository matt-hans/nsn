//! Core scheduler state machine for NSN off-chain nodes
//!
//! Manages:
//! - Node state transitions (Idle <-> GeneratingLane0/Lane1 <-> Idle)
//! - Dual-lane task queuing (Lane 0 = video, Lane 1 = general AI)
//! - Epoch transitions and On-Deck notifications
//! - Task lifecycle management

use crate::epoch::EpochTracker;
use crate::task_queue::{Lane, Priority, Task, TaskId, TaskQueue, TaskResult};
use nsn_types::{EpochInfo, NodeState};
use std::collections::HashMap;

/// Handle to a running task
#[derive(Debug, Clone)]
pub struct TaskHandle {
    /// Task ID
    pub task_id: TaskId,
    /// Which lane the task is running on
    pub lane: Lane,
    /// When the task started
    pub started_at: std::time::Instant,
}

impl TaskHandle {
    /// Create a new task handle
    pub fn new(task_id: TaskId, lane: Lane) -> Self {
        Self {
            task_id,
            lane,
            started_at: std::time::Instant::now(),
        }
    }

    /// Get the elapsed time since task started
    pub fn elapsed(&self) -> std::time::Duration {
        self.started_at.elapsed()
    }
}

/// Errors that can occur during scheduler operations
#[derive(Debug, Clone, thiserror::Error)]
pub enum SchedulerError {
    /// Invalid state transition attempted
    #[error("Invalid state transition from {from:?} to {to:?}")]
    InvalidTransition { from: NodeState, to: NodeState },

    /// Queue is at maximum capacity
    #[error("Task queue is full")]
    QueueFull,

    /// Task not found in any queue
    #[error("Task not found: {0}")]
    TaskNotFound(TaskId),

    /// Already in draining state
    #[error("Already draining Lane 1 tasks")]
    AlreadyDraining,

    /// Cannot perform action in current state
    #[error("Invalid operation in state {0:?}")]
    InvalidStateForOperation(NodeState),

    /// Lane mismatch
    #[error("Task lane mismatch: expected {expected:?}, got {actual:?}")]
    LaneMismatch { expected: Lane, actual: Lane },
}

/// Core scheduler state machine
#[derive(Debug)]
pub struct SchedulerState {
    /// Current node state
    current_state: NodeState,
    /// Lane 0 queue (video generation - priority)
    lane0_queue: TaskQueue,
    /// Lane 1 queue (general AI compute)
    lane1_queue: TaskQueue,
    /// Current epoch information
    current_epoch: Option<EpochInfo>,
    /// Pending epoch (On-Deck)
    pending_epoch: Option<EpochInfo>,
    /// Currently running task
    active_task: Option<TaskHandle>,
    /// Whether we're draining Lane 1 for epoch transition
    is_draining: bool,
    /// Epoch tracker
    epoch_tracker: EpochTracker,
    /// Completed tasks (for result retrieval)
    completed_tasks: HashMap<TaskId, TaskResult>,
    /// Task ID counter (shared across both lanes)
    next_task_id: u64,
}

impl Default for SchedulerState {
    fn default() -> Self {
        Self::new()
    }
}

impl SchedulerState {
    /// Create a new scheduler state
    pub fn new() -> Self {
        Self {
            current_state: NodeState::Starting,
            lane0_queue: TaskQueue::new(Lane::Lane0),
            lane1_queue: TaskQueue::new(Lane::Lane1),
            current_epoch: None,
            pending_epoch: None,
            active_task: None,
            is_draining: false,
            epoch_tracker: EpochTracker::new(),
            completed_tasks: HashMap::new(),
            next_task_id: 0,
        }
    }

    /// Get the current node state
    pub fn current_state(&self) -> &NodeState {
        &self.current_state
    }

    /// Get the current epoch info
    pub fn current_epoch(&self) -> Option<&EpochInfo> {
        self.current_epoch.as_ref()
    }

    /// Get the pending (On-Deck) epoch info
    pub fn pending_epoch(&self) -> Option<&EpochInfo> {
        self.pending_epoch.as_ref()
    }

    /// Get the currently active task
    pub fn active_task(&self) -> Option<&TaskHandle> {
        self.active_task.as_ref()
    }

    /// Check if we're currently draining Lane 1
    pub fn is_draining(&self) -> bool {
        self.is_draining
    }

    /// Get the epoch tracker
    pub fn epoch_tracker(&self) -> &EpochTracker {
        &self.epoch_tracker
    }

    /// Get Lane 0 queue length
    pub fn lane0_queue_len(&self) -> usize {
        self.lane0_queue.len()
    }

    /// Get Lane 1 queue length
    pub fn lane1_queue_len(&self) -> usize {
        self.lane1_queue.len()
    }

    /// Generate a new task ID
    fn generate_task_id(&mut self) -> TaskId {
        let id = TaskId::new(self.next_task_id);
        self.next_task_id += 1;
        id
    }

    /// Attempt to transition to a new state
    pub fn transition(&mut self, to: NodeState) -> Result<(), SchedulerError> {
        let from = self.current_state.clone();

        // Validate the transition
        if !self.is_valid_transition(&from, &to) {
            return Err(SchedulerError::InvalidTransition { from, to });
        }

        tracing::info!(
            from = ?self.current_state,
            to = ?to,
            "State transition"
        );

        self.current_state = to;
        Ok(())
    }

    /// Check if a state transition is valid
    fn is_valid_transition(&self, from: &NodeState, to: &NodeState) -> bool {
        use NodeState::*;

        match (from, to) {
            // Starting can go to LoadingModels or Error
            (Starting, LoadingModels) => true,
            (Starting, Error(_)) => true,

            // LoadingModels can go to Idle or Error
            (LoadingModels, Idle) => true,
            (LoadingModels, Error(_)) => true,

            // Idle can go to any active state or Stopping
            (Idle, GeneratingLane0) => true,
            (Idle, GeneratingLane1) => true,
            (Idle, Validating) => true,
            (Idle, Serving) => true,
            (Idle, Stopping) => true,
            (Idle, Error(_)) => true,

            // Generating states can go back to Idle or Error
            (GeneratingLane0, Idle) => true,
            (GeneratingLane0, Error(_)) => true,
            (GeneratingLane1, Idle) => true,
            (GeneratingLane1, Error(_)) => true,

            // Validating can go back to Idle or Error
            (Validating, Idle) => true,
            (Validating, Error(_)) => true,

            // Serving can go back to Idle or Error
            (Serving, Idle) => true,
            (Serving, Error(_)) => true,

            // Stopping is terminal (except Error)
            (Stopping, Error(_)) => true,

            // Error can go to Stopping or back to Idle (recovery)
            (Error(_), Stopping) => true,
            (Error(_), Idle) => true,

            // Any other transition is invalid
            _ => false,
        }
    }

    /// Enqueue a task to Lane 0 (video generation)
    pub fn enqueue_lane0(
        &mut self,
        model_id: String,
        input_cid: String,
    ) -> Result<TaskId, SchedulerError> {
        self.enqueue_lane0_with_priority(model_id, input_cid, Priority::Normal)
    }

    /// Enqueue a task to Lane 0 with specific priority
    pub fn enqueue_lane0_with_priority(
        &mut self,
        model_id: String,
        input_cid: String,
        priority: Priority,
    ) -> Result<TaskId, SchedulerError> {
        if self.lane0_queue.is_full() {
            return Err(SchedulerError::QueueFull);
        }

        let task_id = self.generate_task_id();
        let task = Task::new(task_id, model_id, input_cid, Lane::Lane0).with_priority(priority);

        self.lane0_queue
            .enqueue(task)
            .map_err(|_| SchedulerError::QueueFull)?;

        tracing::debug!(task_id = %task_id, "Enqueued Lane 0 task");
        Ok(task_id)
    }

    /// Enqueue a task to Lane 1 (general AI compute)
    pub fn enqueue_lane1(
        &mut self,
        model_id: String,
        input_cid: String,
    ) -> Result<TaskId, SchedulerError> {
        self.enqueue_lane1_with_priority(model_id, input_cid, Priority::Normal)
    }

    /// Enqueue a task to Lane 1 with specific priority
    pub fn enqueue_lane1_with_priority(
        &mut self,
        model_id: String,
        input_cid: String,
        priority: Priority,
    ) -> Result<TaskId, SchedulerError> {
        // Don't accept new Lane 1 tasks if we're draining
        if self.is_draining {
            tracing::warn!("Rejecting Lane 1 task - currently draining for epoch transition");
            return Err(SchedulerError::AlreadyDraining);
        }

        if self.lane1_queue.is_full() {
            return Err(SchedulerError::QueueFull);
        }

        let task_id = self.generate_task_id();
        let task = Task::new(task_id, model_id, input_cid, Lane::Lane1).with_priority(priority);

        self.lane1_queue
            .enqueue(task)
            .map_err(|_| SchedulerError::QueueFull)?;

        tracing::debug!(task_id = %task_id, "Enqueued Lane 1 task");
        Ok(task_id)
    }

    /// Get the next task to execute
    ///
    /// Priority order:
    /// 1. Lane 0 tasks always have priority (video generation is latency-sensitive)
    /// 2. Lane 1 tasks when Lane 0 is empty
    /// 3. If draining, only process remaining Lane 1 tasks (no new ones accepted)
    pub fn next_task(&mut self) -> Option<Task> {
        // Check if we're already running a task
        if self.active_task.is_some() {
            return None;
        }

        // Lane 0 always has priority
        if let Some(task) = self.lane0_queue.dequeue() {
            return Some(task);
        }

        // If not draining, process Lane 1 tasks
        if !self.is_draining || !self.lane1_queue.is_empty() {
            if let Some(task) = self.lane1_queue.dequeue() {
                return Some(task);
            }
        }

        None
    }

    /// Start executing a task
    pub fn start_task(&mut self, task: &Task) -> Result<TaskHandle, SchedulerError> {
        if self.active_task.is_some() {
            return Err(SchedulerError::InvalidStateForOperation(
                self.current_state.clone(),
            ));
        }

        let handle = TaskHandle::new(task.id, task.lane);
        self.active_task = Some(handle.clone());

        // Transition to appropriate generating state
        let target_state = match task.lane {
            Lane::Lane0 => NodeState::GeneratingLane0,
            Lane::Lane1 => NodeState::GeneratingLane1,
        };

        if self.current_state == NodeState::Idle {
            self.transition(target_state)?;
        }

        Ok(handle)
    }

    /// Handle On-Deck notification (2 minutes before becoming director)
    pub fn on_deck_received(&mut self, epoch: EpochInfo) {
        tracing::info!(
            epoch = epoch.epoch,
            slot = epoch.slot,
            "On-Deck notification received - starting Lane 1 drain"
        );

        self.pending_epoch = Some(epoch.clone());
        self.is_draining = true;
        self.epoch_tracker.on_deck(epoch, true);
    }

    /// Handle epoch start
    pub fn epoch_started(&mut self, epoch: EpochInfo) {
        tracing::info!(
            epoch = epoch.epoch,
            slot = epoch.slot,
            active_lane = epoch.active_lane,
            "Epoch started"
        );

        self.current_epoch = Some(epoch.clone());
        self.pending_epoch = None;
        self.is_draining = false;
        self.epoch_tracker.epoch_started(epoch);
    }

    /// Handle epoch end
    pub fn epoch_ended(&mut self) {
        if let Some(ref epoch) = self.current_epoch {
            tracing::info!(epoch = epoch.epoch, "Epoch ended");
        }

        self.current_epoch = None;
        self.epoch_tracker.epoch_ended();
    }

    /// Complete a running task
    pub fn complete_task(
        &mut self,
        task_id: TaskId,
        result: TaskResult,
    ) -> Result<(), SchedulerError> {
        // Verify the task is actually running
        let handle = self
            .active_task
            .as_ref()
            .ok_or(SchedulerError::TaskNotFound(task_id))?;

        if handle.task_id != task_id {
            return Err(SchedulerError::TaskNotFound(task_id));
        }

        tracing::info!(
            task_id = %task_id,
            elapsed_ms = handle.elapsed().as_millis(),
            "Task completed"
        );

        // Store the result
        self.completed_tasks.insert(task_id, result);

        // Clear active task
        self.active_task = None;

        // Transition back to Idle
        if matches!(
            self.current_state,
            NodeState::GeneratingLane0 | NodeState::GeneratingLane1
        ) {
            self.transition(NodeState::Idle)?;
        }

        Ok(())
    }

    /// Cancel a task
    pub fn cancel_task(&mut self, task_id: TaskId) -> Result<(), SchedulerError> {
        // Check if it's the active task
        if let Some(ref handle) = self.active_task {
            if handle.task_id == task_id {
                tracing::info!(task_id = %task_id, "Cancelling active task");
                self.active_task = None;

                // Transition back to Idle
                if matches!(
                    self.current_state,
                    NodeState::GeneratingLane0 | NodeState::GeneratingLane1
                ) {
                    self.transition(NodeState::Idle)?;
                }
                return Ok(());
            }
        }

        // Try to remove from Lane 0 queue
        if let Some(mut task) = self.lane0_queue.remove(task_id) {
            task.mark_cancelled();
            tracing::info!(task_id = %task_id, "Cancelled Lane 0 task");
            return Ok(());
        }

        // Try to remove from Lane 1 queue
        if let Some(mut task) = self.lane1_queue.remove(task_id) {
            task.mark_cancelled();
            tracing::info!(task_id = %task_id, "Cancelled Lane 1 task");
            return Ok(());
        }

        Err(SchedulerError::TaskNotFound(task_id))
    }

    /// Check if a Lane 0 task is waiting (should preempt Lane 1)
    pub fn should_preempt(&self) -> bool {
        // Only preempt if we're currently doing Lane 1 work
        // and there's a Lane 0 task waiting
        if !self.lane0_queue.is_empty() {
            if let Some(ref handle) = self.active_task {
                return handle.lane == Lane::Lane1;
            }
        }
        false
    }

    /// Get a completed task result
    pub fn get_task_result(&self, task_id: TaskId) -> Option<&TaskResult> {
        self.completed_tasks.get(&task_id)
    }

    /// Clear old completed task results
    pub fn clear_completed_before(&mut self, task_id: TaskId) {
        self.completed_tasks.retain(|id, _| id.0 >= task_id.0);
    }

    /// Get statistics about the scheduler
    pub fn stats(&self) -> SchedulerStats {
        SchedulerStats {
            current_state: self.current_state.clone(),
            lane0_queue_len: self.lane0_queue.len(),
            lane1_queue_len: self.lane1_queue.len(),
            is_draining: self.is_draining,
            has_active_task: self.active_task.is_some(),
            completed_tasks_count: self.completed_tasks.len(),
            current_epoch: self.current_epoch.as_ref().map(|e| e.epoch),
        }
    }
}

/// Scheduler statistics for monitoring
#[derive(Debug, Clone)]
pub struct SchedulerStats {
    pub current_state: NodeState,
    pub lane0_queue_len: usize,
    pub lane1_queue_len: usize,
    pub is_draining: bool,
    pub has_active_task: bool,
    pub completed_tasks_count: usize,
    pub current_epoch: Option<u64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_epoch(epoch: u64, slot: u64, active_lane: u8) -> EpochInfo {
        EpochInfo {
            epoch,
            slot,
            block_number: epoch * 100 + slot,
            active_lane,
        }
    }

    #[test]
    fn test_scheduler_new() {
        let scheduler = SchedulerState::new();
        assert_eq!(*scheduler.current_state(), NodeState::Starting);
        assert!(scheduler.current_epoch().is_none());
        assert!(scheduler.active_task().is_none());
        assert!(!scheduler.is_draining());
    }

    #[test]
    fn test_state_transitions() {
        let mut scheduler = SchedulerState::new();

        // Starting -> LoadingModels
        scheduler.transition(NodeState::LoadingModels).unwrap();
        assert_eq!(*scheduler.current_state(), NodeState::LoadingModels);

        // LoadingModels -> Idle
        scheduler.transition(NodeState::Idle).unwrap();
        assert_eq!(*scheduler.current_state(), NodeState::Idle);

        // Idle -> GeneratingLane0
        scheduler.transition(NodeState::GeneratingLane0).unwrap();
        assert_eq!(*scheduler.current_state(), NodeState::GeneratingLane0);

        // GeneratingLane0 -> Idle
        scheduler.transition(NodeState::Idle).unwrap();
        assert_eq!(*scheduler.current_state(), NodeState::Idle);
    }

    #[test]
    fn test_invalid_state_transition() {
        let mut scheduler = SchedulerState::new();

        // Starting -> GeneratingLane0 is invalid
        let result = scheduler.transition(NodeState::GeneratingLane0);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            SchedulerError::InvalidTransition { .. }
        ));
    }

    #[test]
    fn test_enqueue_lane0() {
        let mut scheduler = SchedulerState::new();

        let task_id = scheduler
            .enqueue_lane0("flux-schnell".to_string(), "QmInput123".to_string())
            .unwrap();

        assert_eq!(scheduler.lane0_queue_len(), 1);
        assert_eq!(task_id, TaskId::new(0));
    }

    #[test]
    fn test_enqueue_lane1() {
        let mut scheduler = SchedulerState::new();

        let task_id = scheduler
            .enqueue_lane1("llama-70b".to_string(), "QmInput456".to_string())
            .unwrap();

        assert_eq!(scheduler.lane1_queue_len(), 1);
        assert_eq!(task_id, TaskId::new(0));
    }

    #[test]
    fn test_next_task_priority() {
        let mut scheduler = SchedulerState::new();

        // Add Lane 1 task first
        scheduler
            .enqueue_lane1("llama-70b".to_string(), "QmLane1".to_string())
            .unwrap();

        // Add Lane 0 task second
        scheduler
            .enqueue_lane0("flux-schnell".to_string(), "QmLane0".to_string())
            .unwrap();

        // Lane 0 should come out first (priority)
        let task = scheduler.next_task().unwrap();
        assert_eq!(task.lane, Lane::Lane0);

        // Then Lane 1
        let task = scheduler.next_task().unwrap();
        assert_eq!(task.lane, Lane::Lane1);
    }

    #[test]
    fn test_on_deck_starts_drain() {
        let mut scheduler = SchedulerState::new();
        let epoch = make_epoch(1, 0, 0);

        assert!(!scheduler.is_draining());

        scheduler.on_deck_received(epoch);

        assert!(scheduler.is_draining());
        assert!(scheduler.pending_epoch().is_some());
    }

    #[test]
    fn test_draining_rejects_lane1() {
        let mut scheduler = SchedulerState::new();
        let epoch = make_epoch(1, 0, 0);

        scheduler.on_deck_received(epoch);
        assert!(scheduler.is_draining());

        // Lane 1 tasks should be rejected
        let result = scheduler.enqueue_lane1("model".to_string(), "cid".to_string());
        assert!(matches!(result, Err(SchedulerError::AlreadyDraining)));

        // Lane 0 tasks should still work
        let result = scheduler.enqueue_lane0("model".to_string(), "cid".to_string());
        assert!(result.is_ok());
    }

    #[test]
    fn test_epoch_transition() {
        let mut scheduler = SchedulerState::new();

        // On-Deck notification
        let epoch1 = make_epoch(1, 0, 0);
        scheduler.on_deck_received(epoch1.clone());
        assert!(scheduler.is_draining());

        // Epoch starts
        scheduler.epoch_started(epoch1);
        assert!(!scheduler.is_draining());
        assert!(scheduler.current_epoch().is_some());
        assert_eq!(scheduler.current_epoch().unwrap().epoch, 1);

        // Epoch ends
        scheduler.epoch_ended();
        assert!(scheduler.current_epoch().is_none());
    }

    #[test]
    fn test_should_preempt() {
        let mut scheduler = SchedulerState::new();

        // Not preempting when queues are empty
        assert!(!scheduler.should_preempt());

        // Simulate running a Lane 1 task
        scheduler.transition(NodeState::LoadingModels).unwrap();
        scheduler.transition(NodeState::Idle).unwrap();

        scheduler
            .enqueue_lane1("model".to_string(), "cid".to_string())
            .unwrap();
        let task = scheduler.next_task().unwrap();
        scheduler.start_task(&task).unwrap();

        // Still not preempting (no Lane 0 task waiting)
        assert!(!scheduler.should_preempt());

        // Add a Lane 0 task
        scheduler
            .enqueue_lane0("flux".to_string(), "cid".to_string())
            .unwrap();

        // Now we should preempt
        assert!(scheduler.should_preempt());
    }

    #[test]
    fn test_complete_task() {
        let mut scheduler = SchedulerState::new();
        scheduler.transition(NodeState::LoadingModels).unwrap();
        scheduler.transition(NodeState::Idle).unwrap();

        // Enqueue and start a task
        let task_id = scheduler
            .enqueue_lane0("flux".to_string(), "cid".to_string())
            .unwrap();
        let task = scheduler.next_task().unwrap();
        scheduler.start_task(&task).unwrap();

        assert_eq!(*scheduler.current_state(), NodeState::GeneratingLane0);

        // Complete the task
        let result = TaskResult::success("QmOutput".to_string(), 1500);
        scheduler.complete_task(task_id, result).unwrap();

        assert_eq!(*scheduler.current_state(), NodeState::Idle);
        assert!(scheduler.active_task().is_none());
        assert!(scheduler.get_task_result(task_id).is_some());
    }

    #[test]
    fn test_cancel_queued_task() {
        let mut scheduler = SchedulerState::new();

        let task_id = scheduler
            .enqueue_lane0("flux".to_string(), "cid".to_string())
            .unwrap();

        assert_eq!(scheduler.lane0_queue_len(), 1);

        scheduler.cancel_task(task_id).unwrap();

        assert_eq!(scheduler.lane0_queue_len(), 0);
    }

    #[test]
    fn test_cancel_active_task() {
        let mut scheduler = SchedulerState::new();
        scheduler.transition(NodeState::LoadingModels).unwrap();
        scheduler.transition(NodeState::Idle).unwrap();

        let task_id = scheduler
            .enqueue_lane0("flux".to_string(), "cid".to_string())
            .unwrap();
        let task = scheduler.next_task().unwrap();
        scheduler.start_task(&task).unwrap();

        assert!(scheduler.active_task().is_some());

        scheduler.cancel_task(task_id).unwrap();

        assert!(scheduler.active_task().is_none());
        assert_eq!(*scheduler.current_state(), NodeState::Idle);
    }

    #[test]
    fn test_stats() {
        let mut scheduler = SchedulerState::new();
        scheduler.transition(NodeState::LoadingModels).unwrap();
        scheduler.transition(NodeState::Idle).unwrap();

        scheduler
            .enqueue_lane0("flux".to_string(), "cid1".to_string())
            .unwrap();
        scheduler
            .enqueue_lane1("llama".to_string(), "cid2".to_string())
            .unwrap();

        let stats = scheduler.stats();

        assert_eq!(stats.current_state, NodeState::Idle);
        assert_eq!(stats.lane0_queue_len, 1);
        assert_eq!(stats.lane1_queue_len, 1);
        assert!(!stats.is_draining);
        assert!(!stats.has_active_task);
    }
}
