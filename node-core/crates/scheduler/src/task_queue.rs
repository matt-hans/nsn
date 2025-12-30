//! Task queue implementation for NSN scheduler
//!
//! Provides lane-specific task queuing with priority support for dual-lane architecture:
//! - Lane 0: Video generation (priority, latency-sensitive)
//! - Lane 1: General AI compute (LLM inference, image gen, etc.)

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::time::Instant;

/// Unique identifier for a task
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TaskId(pub u64);

impl TaskId {
    /// Create a new TaskId
    pub fn new(id: u64) -> Self {
        Self(id)
    }
}

impl std::fmt::Display for TaskId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "TaskId({})", self.0)
    }
}

/// Lane designation for task queuing
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Lane {
    /// Lane 0: Video generation (priority, latency-sensitive)
    Lane0,
    /// Lane 1: General AI compute (LLM inference, image gen, etc.)
    Lane1,
}

impl Lane {
    /// Get the lane number
    pub fn as_u8(&self) -> u8 {
        match self {
            Lane::Lane0 => 0,
            Lane::Lane1 => 1,
        }
    }
}

/// Task priority levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize, Default)]
pub enum Priority {
    /// Critical priority - immediate execution
    Critical = 0,
    /// High priority
    High = 1,
    /// Normal priority
    #[default]
    Normal = 2,
    /// Low priority - opportunistic execution
    Low = 3,
}

/// Result of a completed task
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskResult {
    /// Output CID (content identifier)
    pub output_cid: Option<String>,
    /// Execution time in milliseconds
    pub execution_time_ms: u64,
    /// Any additional metadata
    pub metadata: Option<serde_json::Value>,
}

impl TaskResult {
    /// Create a successful task result
    pub fn success(output_cid: String, execution_time_ms: u64) -> Self {
        Self {
            output_cid: Some(output_cid),
            execution_time_ms,
            metadata: None,
        }
    }

    /// Create a failed task result
    pub fn failed(execution_time_ms: u64) -> Self {
        Self {
            output_cid: None,
            execution_time_ms,
            metadata: None,
        }
    }
}

/// A task to be executed by the scheduler
#[derive(Debug, Clone)]
pub struct Task {
    /// Unique task identifier
    pub id: TaskId,
    /// Model identifier (e.g., "flux-schnell", "llama-70b")
    pub model_id: String,
    /// Input content identifier (CID)
    pub input_cid: String,
    /// Task priority
    pub priority: Priority,
    /// Optional deadline for task completion
    pub deadline: Option<Instant>,
    /// Current task status
    pub status: nsn_types::TaskStatus,
    /// When the task was created
    pub created_at: Instant,
    /// Which lane this task belongs to
    pub lane: Lane,
}

impl Task {
    /// Create a new task
    pub fn new(id: TaskId, model_id: String, input_cid: String, lane: Lane) -> Self {
        Self {
            id,
            model_id,
            input_cid,
            priority: Priority::default(),
            deadline: None,
            status: nsn_types::TaskStatus::Pending,
            created_at: Instant::now(),
            lane,
        }
    }

    /// Set the task priority
    pub fn with_priority(mut self, priority: Priority) -> Self {
        self.priority = priority;
        self
    }

    /// Set a deadline for the task
    pub fn with_deadline(mut self, deadline: Instant) -> Self {
        self.deadline = Some(deadline);
        self
    }

    /// Check if the task has exceeded its deadline
    pub fn is_expired(&self) -> bool {
        if let Some(deadline) = self.deadline {
            Instant::now() > deadline
        } else {
            false
        }
    }

    /// Mark task as running
    pub fn mark_running(&mut self) {
        self.status = nsn_types::TaskStatus::Running;
    }

    /// Mark task as completed
    pub fn mark_completed(&mut self) {
        self.status = nsn_types::TaskStatus::Completed;
    }

    /// Mark task as cancelled
    pub fn mark_cancelled(&mut self) {
        self.status = nsn_types::TaskStatus::Cancelled;
    }
}

/// Task queue for a specific lane
#[derive(Debug)]
pub struct TaskQueue {
    /// Queue of pending tasks
    tasks: VecDeque<Task>,
    /// Lane this queue serves
    lane: Lane,
    /// Maximum queue size
    max_size: usize,
    /// Counter for task IDs
    next_id: u64,
}

impl TaskQueue {
    /// Create a new task queue for the specified lane
    pub fn new(lane: Lane) -> Self {
        Self {
            tasks: VecDeque::new(),
            lane,
            max_size: 1000,
            next_id: 0,
        }
    }

    /// Create a new task queue with a custom max size
    pub fn with_max_size(lane: Lane, max_size: usize) -> Self {
        Self {
            tasks: VecDeque::new(),
            lane,
            max_size,
            next_id: 0,
        }
    }

    /// Get the lane this queue serves
    pub fn lane(&self) -> Lane {
        self.lane
    }

    /// Get the number of pending tasks
    pub fn len(&self) -> usize {
        self.tasks.len()
    }

    /// Check if the queue is empty
    pub fn is_empty(&self) -> bool {
        self.tasks.is_empty()
    }

    /// Check if the queue is full
    pub fn is_full(&self) -> bool {
        self.tasks.len() >= self.max_size
    }

    /// Generate a new task ID
    pub fn generate_task_id(&mut self) -> TaskId {
        let id = TaskId::new(self.next_id);
        self.next_id += 1;
        id
    }

    /// Enqueue a task, maintaining priority order
    pub fn enqueue(&mut self, task: Task) -> Result<TaskId, QueueError> {
        if self.is_full() {
            return Err(QueueError::QueueFull);
        }

        let task_id = task.id;

        // Insert based on priority (lower priority value = higher priority)
        // Also consider deadline for same-priority tasks
        let insert_pos = self.tasks.iter().position(|t| {
            if t.priority > task.priority {
                true
            } else if t.priority == task.priority {
                // For same priority, earlier deadline comes first
                match (&t.deadline, &task.deadline) {
                    (Some(t_deadline), Some(task_deadline)) => t_deadline > task_deadline,
                    (None, Some(_)) => true,
                    _ => false,
                }
            } else {
                false
            }
        });

        match insert_pos {
            Some(pos) => self.tasks.insert(pos, task),
            None => self.tasks.push_back(task),
        }

        tracing::debug!(
            task_id = %task_id,
            lane = ?self.lane,
            queue_len = self.tasks.len(),
            "Task enqueued"
        );

        Ok(task_id)
    }

    /// Dequeue the next task
    pub fn dequeue(&mut self) -> Option<Task> {
        let task = self.tasks.pop_front();
        if let Some(ref t) = task {
            tracing::debug!(
                task_id = %t.id,
                lane = ?self.lane,
                queue_len = self.tasks.len(),
                "Task dequeued"
            );
        }
        task
    }

    /// Peek at the next task without removing it
    pub fn peek(&self) -> Option<&Task> {
        self.tasks.front()
    }

    /// Find a task by ID
    pub fn find(&self, task_id: TaskId) -> Option<&Task> {
        self.tasks.iter().find(|t| t.id == task_id)
    }

    /// Find a task by ID mutably
    pub fn find_mut(&mut self, task_id: TaskId) -> Option<&mut Task> {
        self.tasks.iter_mut().find(|t| t.id == task_id)
    }

    /// Remove a task by ID
    pub fn remove(&mut self, task_id: TaskId) -> Option<Task> {
        if let Some(pos) = self.tasks.iter().position(|t| t.id == task_id) {
            self.tasks.remove(pos)
        } else {
            None
        }
    }

    /// Remove all expired tasks
    pub fn remove_expired(&mut self) -> Vec<Task> {
        let mut expired = Vec::new();
        self.tasks.retain(|t| {
            if t.is_expired() {
                expired.push(t.clone());
                false
            } else {
                true
            }
        });
        expired
    }

    /// Clear all tasks from the queue
    pub fn clear(&mut self) -> Vec<Task> {
        self.tasks.drain(..).collect()
    }

    /// Get an iterator over pending tasks
    pub fn iter(&self) -> impl Iterator<Item = &Task> {
        self.tasks.iter()
    }
}

/// Errors that can occur during queue operations
#[derive(Debug, Clone, thiserror::Error)]
pub enum QueueError {
    /// Queue has reached maximum capacity
    #[error("Queue is full")]
    QueueFull,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_task_creation() {
        let task = Task::new(
            TaskId::new(1),
            "flux-schnell".to_string(),
            "QmTest123".to_string(),
            Lane::Lane0,
        );

        assert_eq!(task.id, TaskId::new(1));
        assert_eq!(task.model_id, "flux-schnell");
        assert_eq!(task.lane, Lane::Lane0);
        assert_eq!(task.priority, Priority::Normal);
        assert!(!task.is_expired());
    }

    #[test]
    fn test_task_priority_ordering() {
        assert!(Priority::Critical < Priority::High);
        assert!(Priority::High < Priority::Normal);
        assert!(Priority::Normal < Priority::Low);
    }

    #[test]
    fn test_queue_enqueue_dequeue() {
        let mut queue = TaskQueue::new(Lane::Lane0);

        let task1 = Task::new(
            TaskId::new(1),
            "model1".to_string(),
            "cid1".to_string(),
            Lane::Lane0,
        );
        let task2 = Task::new(
            TaskId::new(2),
            "model2".to_string(),
            "cid2".to_string(),
            Lane::Lane0,
        );

        queue.enqueue(task1).unwrap();
        queue.enqueue(task2).unwrap();

        assert_eq!(queue.len(), 2);

        let dequeued = queue.dequeue().unwrap();
        assert_eq!(dequeued.id, TaskId::new(1));

        assert_eq!(queue.len(), 1);
    }

    #[test]
    fn test_queue_priority_ordering() {
        let mut queue = TaskQueue::new(Lane::Lane1);

        // Add normal priority first
        let task_normal = Task::new(
            TaskId::new(1),
            "model".to_string(),
            "cid1".to_string(),
            Lane::Lane1,
        );

        // Add high priority second
        let task_high = Task::new(
            TaskId::new(2),
            "model".to_string(),
            "cid2".to_string(),
            Lane::Lane1,
        )
        .with_priority(Priority::High);

        // Add critical priority last
        let task_critical = Task::new(
            TaskId::new(3),
            "model".to_string(),
            "cid3".to_string(),
            Lane::Lane1,
        )
        .with_priority(Priority::Critical);

        queue.enqueue(task_normal).unwrap();
        queue.enqueue(task_high).unwrap();
        queue.enqueue(task_critical).unwrap();

        // Critical should come out first
        assert_eq!(queue.dequeue().unwrap().id, TaskId::new(3));
        // Then high
        assert_eq!(queue.dequeue().unwrap().id, TaskId::new(2));
        // Then normal
        assert_eq!(queue.dequeue().unwrap().id, TaskId::new(1));
    }

    #[test]
    fn test_queue_full() {
        let mut queue = TaskQueue::with_max_size(Lane::Lane0, 2);

        let task1 = Task::new(
            TaskId::new(1),
            "model".to_string(),
            "cid1".to_string(),
            Lane::Lane0,
        );
        let task2 = Task::new(
            TaskId::new(2),
            "model".to_string(),
            "cid2".to_string(),
            Lane::Lane0,
        );
        let task3 = Task::new(
            TaskId::new(3),
            "model".to_string(),
            "cid3".to_string(),
            Lane::Lane0,
        );

        queue.enqueue(task1).unwrap();
        queue.enqueue(task2).unwrap();

        let result = queue.enqueue(task3);
        assert!(matches!(result, Err(QueueError::QueueFull)));
    }

    #[test]
    fn test_queue_remove() {
        let mut queue = TaskQueue::new(Lane::Lane0);

        let task1 = Task::new(
            TaskId::new(1),
            "model".to_string(),
            "cid1".to_string(),
            Lane::Lane0,
        );
        let task2 = Task::new(
            TaskId::new(2),
            "model".to_string(),
            "cid2".to_string(),
            Lane::Lane0,
        );

        queue.enqueue(task1).unwrap();
        queue.enqueue(task2).unwrap();

        let removed = queue.remove(TaskId::new(1));
        assert!(removed.is_some());
        assert_eq!(removed.unwrap().id, TaskId::new(1));
        assert_eq!(queue.len(), 1);

        // Try to remove non-existent task
        let not_found = queue.remove(TaskId::new(99));
        assert!(not_found.is_none());
    }

    #[test]
    fn test_lane_as_u8() {
        assert_eq!(Lane::Lane0.as_u8(), 0);
        assert_eq!(Lane::Lane1.as_u8(), 1);
    }
}
