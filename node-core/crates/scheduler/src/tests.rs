//! Unit tests for NSN scheduler
//!
//! Contains the 8 required test cases plus additional edge case coverage.

use crate::epoch::EpochTracker;
use crate::state_machine::{SchedulerError, SchedulerState};
use crate::task_queue::{Lane, Priority, Task, TaskId, TaskQueue, TaskResult};
use nsn_types::{EpochInfo, NodeState};

fn make_epoch(epoch: u64, slot: u64, active_lane: u8) -> EpochInfo {
    EpochInfo {
        epoch,
        slot,
        block_number: epoch * 100 + slot,
        active_lane,
    }
}

// =============================================================================
// Required Test 1: test_state_transitions - Valid state transitions
// =============================================================================
#[test]
fn test_state_transitions() {
    let mut scheduler = SchedulerState::new();

    // Starting -> LoadingModels
    assert!(scheduler.transition(NodeState::LoadingModels).is_ok());
    assert_eq!(*scheduler.current_state(), NodeState::LoadingModels);

    // LoadingModels -> Idle
    assert!(scheduler.transition(NodeState::Idle).is_ok());
    assert_eq!(*scheduler.current_state(), NodeState::Idle);

    // Idle -> GeneratingLane0
    assert!(scheduler.transition(NodeState::GeneratingLane0).is_ok());
    assert_eq!(*scheduler.current_state(), NodeState::GeneratingLane0);

    // GeneratingLane0 -> Idle
    assert!(scheduler.transition(NodeState::Idle).is_ok());
    assert_eq!(*scheduler.current_state(), NodeState::Idle);

    // Idle -> GeneratingLane1
    assert!(scheduler.transition(NodeState::GeneratingLane1).is_ok());
    assert_eq!(*scheduler.current_state(), NodeState::GeneratingLane1);

    // GeneratingLane1 -> Idle
    assert!(scheduler.transition(NodeState::Idle).is_ok());
    assert_eq!(*scheduler.current_state(), NodeState::Idle);

    // Idle -> Validating
    assert!(scheduler.transition(NodeState::Validating).is_ok());
    assert_eq!(*scheduler.current_state(), NodeState::Validating);

    // Validating -> Idle
    assert!(scheduler.transition(NodeState::Idle).is_ok());
    assert_eq!(*scheduler.current_state(), NodeState::Idle);

    // Idle -> Serving
    assert!(scheduler.transition(NodeState::Serving).is_ok());
    assert_eq!(*scheduler.current_state(), NodeState::Serving);

    // Serving -> Idle
    assert!(scheduler.transition(NodeState::Idle).is_ok());
    assert_eq!(*scheduler.current_state(), NodeState::Idle);

    // Idle -> Stopping (terminal state)
    assert!(scheduler.transition(NodeState::Stopping).is_ok());
    assert_eq!(*scheduler.current_state(), NodeState::Stopping);
}

// =============================================================================
// Required Test 2: test_invalid_state_transition - Rejects invalid transitions
// =============================================================================
#[test]
fn test_invalid_state_transition() {
    let mut scheduler = SchedulerState::new();
    assert_eq!(*scheduler.current_state(), NodeState::Starting);

    // Starting -> GeneratingLane0 is INVALID (must go through LoadingModels -> Idle first)
    let result = scheduler.transition(NodeState::GeneratingLane0);
    assert!(result.is_err());
    match result.unwrap_err() {
        SchedulerError::InvalidTransition { from, to } => {
            assert_eq!(from, NodeState::Starting);
            assert_eq!(to, NodeState::GeneratingLane0);
        }
        _ => panic!("Expected InvalidTransition error"),
    }

    // State should remain unchanged
    assert_eq!(*scheduler.current_state(), NodeState::Starting);

    // Starting -> Idle is also INVALID
    let result = scheduler.transition(NodeState::Idle);
    assert!(result.is_err());

    // Starting -> Validating is INVALID
    let result = scheduler.transition(NodeState::Validating);
    assert!(result.is_err());

    // Now do valid transitions
    scheduler.transition(NodeState::LoadingModels).unwrap();
    scheduler.transition(NodeState::Idle).unwrap();

    // Idle -> LoadingModels is INVALID (can't go backwards)
    let result = scheduler.transition(NodeState::LoadingModels);
    assert!(result.is_err());

    // GeneratingLane0 -> GeneratingLane1 is INVALID (must go through Idle)
    scheduler.transition(NodeState::GeneratingLane0).unwrap();
    let result = scheduler.transition(NodeState::GeneratingLane1);
    assert!(result.is_err());
}

// =============================================================================
// Required Test 3: test_enqueue_lane0 - Adds task to Lane 0 queue
// =============================================================================
#[test]
fn test_enqueue_lane0() {
    let mut scheduler = SchedulerState::new();

    // Enqueue first task
    let task_id1 = scheduler
        .enqueue_lane0("flux-schnell".to_string(), "QmInput123".to_string())
        .unwrap();
    assert_eq!(task_id1, TaskId::new(0));
    assert_eq!(scheduler.lane0_queue_len(), 1);

    // Enqueue second task
    let task_id2 = scheduler
        .enqueue_lane0("liveportrait".to_string(), "QmInput456".to_string())
        .unwrap();
    assert_eq!(task_id2, TaskId::new(1));
    assert_eq!(scheduler.lane0_queue_len(), 2);

    // Lane 1 should still be empty
    assert_eq!(scheduler.lane1_queue_len(), 0);

    // Enqueue with high priority
    let task_id3 = scheduler
        .enqueue_lane0_with_priority(
            "kokoro-tts".to_string(),
            "QmInput789".to_string(),
            Priority::High,
        )
        .unwrap();
    assert_eq!(task_id3, TaskId::new(2));
    assert_eq!(scheduler.lane0_queue_len(), 3);
}

// =============================================================================
// Required Test 4: test_enqueue_lane1 - Adds task to Lane 1 queue
// =============================================================================
#[test]
fn test_enqueue_lane1() {
    let mut scheduler = SchedulerState::new();

    // Enqueue first task
    let task_id1 = scheduler
        .enqueue_lane1("llama-70b".to_string(), "QmPrompt123".to_string())
        .unwrap();
    assert_eq!(task_id1, TaskId::new(0));
    assert_eq!(scheduler.lane1_queue_len(), 1);

    // Enqueue second task
    let task_id2 = scheduler
        .enqueue_lane1("stable-diffusion".to_string(), "QmPrompt456".to_string())
        .unwrap();
    assert_eq!(task_id2, TaskId::new(1));
    assert_eq!(scheduler.lane1_queue_len(), 2);

    // Lane 0 should still be empty
    assert_eq!(scheduler.lane0_queue_len(), 0);

    // Enqueue with low priority
    let task_id3 = scheduler
        .enqueue_lane1_with_priority(
            "whisper".to_string(),
            "QmAudio789".to_string(),
            Priority::Low,
        )
        .unwrap();
    assert_eq!(task_id3, TaskId::new(2));
    assert_eq!(scheduler.lane1_queue_len(), 3);
}

// =============================================================================
// Required Test 5: test_next_task_priority - Lane 0 has priority over Lane 1
// =============================================================================
#[test]
fn test_next_task_priority() {
    let mut scheduler = SchedulerState::new();

    // Add Lane 1 task first (timestamp earlier)
    let lane1_id = scheduler
        .enqueue_lane1("llama-70b".to_string(), "QmLane1First".to_string())
        .unwrap();

    // Add Lane 0 task second (timestamp later)
    let lane0_id = scheduler
        .enqueue_lane0("flux-schnell".to_string(), "QmLane0Second".to_string())
        .unwrap();

    // Lane 0 should come out first despite being enqueued later
    let task1 = scheduler.next_task().unwrap();
    assert_eq!(task1.id, lane0_id);
    assert_eq!(task1.lane, Lane::Lane0);

    // Then Lane 1
    let task2 = scheduler.next_task().unwrap();
    assert_eq!(task2.id, lane1_id);
    assert_eq!(task2.lane, Lane::Lane1);

    // Queue should be empty
    assert!(scheduler.next_task().is_none());

    // Test with multiple tasks in each lane
    scheduler
        .enqueue_lane1("model1".to_string(), "cid1".to_string())
        .unwrap();
    scheduler
        .enqueue_lane1("model2".to_string(), "cid2".to_string())
        .unwrap();
    scheduler
        .enqueue_lane0("model3".to_string(), "cid3".to_string())
        .unwrap();
    scheduler
        .enqueue_lane1("model4".to_string(), "cid4".to_string())
        .unwrap();
    scheduler
        .enqueue_lane0("model5".to_string(), "cid5".to_string())
        .unwrap();

    // All Lane 0 tasks should come out first
    assert_eq!(scheduler.next_task().unwrap().lane, Lane::Lane0);
    assert_eq!(scheduler.next_task().unwrap().lane, Lane::Lane0);

    // Then all Lane 1 tasks
    assert_eq!(scheduler.next_task().unwrap().lane, Lane::Lane1);
    assert_eq!(scheduler.next_task().unwrap().lane, Lane::Lane1);
    assert_eq!(scheduler.next_task().unwrap().lane, Lane::Lane1);
}

// =============================================================================
// Required Test 6: test_on_deck_starts_drain - On-Deck sets draining flag
// =============================================================================
#[test]
fn test_on_deck_starts_drain() {
    let mut scheduler = SchedulerState::new();

    // Initially not draining
    assert!(!scheduler.is_draining());

    // Receive On-Deck notification
    let epoch = make_epoch(1, 0, 0);
    scheduler.on_deck_received(epoch.clone());

    // Now draining
    assert!(scheduler.is_draining());

    // Pending epoch should be set
    assert!(scheduler.pending_epoch().is_some());
    assert_eq!(scheduler.pending_epoch().unwrap().epoch, 1);

    // Epoch tracker should also reflect On-Deck state
    assert!(scheduler.epoch_tracker().is_on_deck());
    assert!(scheduler.epoch_tracker().should_drain_lane1());

    // New Lane 1 tasks should be rejected
    let result = scheduler.enqueue_lane1("model".to_string(), "cid".to_string());
    assert!(matches!(result, Err(SchedulerError::AlreadyDraining)));

    // But existing Lane 1 tasks can still be processed
    // (they should be drained, not dropped)
}

// =============================================================================
// Required Test 7: test_epoch_transition - Full epoch transition flow
// =============================================================================
#[test]
fn test_epoch_transition() {
    let mut scheduler = SchedulerState::new();

    // === Phase 1: Normal operation (no director role) ===
    assert!(scheduler.current_epoch().is_none());
    assert!(!scheduler.is_draining());

    // Add some Lane 1 tasks
    scheduler
        .enqueue_lane1("llama".to_string(), "cid1".to_string())
        .unwrap();
    scheduler
        .enqueue_lane1("whisper".to_string(), "cid2".to_string())
        .unwrap();

    // === Phase 2: On-Deck notification (2 min before epoch) ===
    let epoch1 = make_epoch(1, 0, 0);
    scheduler.on_deck_received(epoch1.clone());

    // Should be draining Lane 1
    assert!(scheduler.is_draining());
    assert!(scheduler.pending_epoch().is_some());

    // Existing Lane 1 tasks can be dequeued (draining them)
    let task = scheduler.next_task().unwrap();
    assert_eq!(task.lane, Lane::Lane1);

    // New Lane 1 tasks rejected
    assert!(scheduler
        .enqueue_lane1("new".to_string(), "cid".to_string())
        .is_err());

    // Lane 0 tasks still accepted
    assert!(scheduler
        .enqueue_lane0("flux".to_string(), "cid".to_string())
        .is_ok());

    // === Phase 3: Epoch starts ===
    scheduler.epoch_started(epoch1.clone());

    // Draining should stop
    assert!(!scheduler.is_draining());

    // Current epoch should be set
    assert!(scheduler.current_epoch().is_some());
    assert_eq!(scheduler.current_epoch().unwrap().epoch, 1);
    assert_eq!(scheduler.current_epoch().unwrap().active_lane, 0);

    // Pending epoch should be cleared
    assert!(scheduler.pending_epoch().is_none());

    // Lane 1 enqueuing should work again
    assert!(scheduler
        .enqueue_lane1("llama".to_string(), "cid".to_string())
        .is_ok());

    // === Phase 4: Epoch ends ===
    scheduler.epoch_ended();

    // Current epoch should be cleared
    assert!(scheduler.current_epoch().is_none());

    // Epoch tracker should reflect end
    assert!(!scheduler.epoch_tracker().is_director_current());
}

// =============================================================================
// Required Test 8: test_should_preempt - Returns true when Lane 0 task waiting
// =============================================================================
#[test]
fn test_should_preempt() {
    let mut scheduler = SchedulerState::new();
    scheduler.transition(NodeState::LoadingModels).unwrap();
    scheduler.transition(NodeState::Idle).unwrap();

    // No preemption when nothing is running
    assert!(!scheduler.should_preempt());

    // Start a Lane 1 task
    scheduler
        .enqueue_lane1("llama".to_string(), "cid1".to_string())
        .unwrap();
    let task = scheduler.next_task().unwrap();
    assert_eq!(task.lane, Lane::Lane1);
    scheduler.start_task(&task).unwrap();

    // Still no preemption (no Lane 0 task waiting)
    assert!(!scheduler.should_preempt());

    // Add a Lane 0 task
    scheduler
        .enqueue_lane0("flux".to_string(), "cid2".to_string())
        .unwrap();

    // NOW we should preempt
    assert!(scheduler.should_preempt());

    // Complete the Lane 1 task
    let result = TaskResult::success("output".to_string(), 1000);
    scheduler.complete_task(task.id, result).unwrap();

    // No longer need to preempt (no active Lane 1 task)
    assert!(!scheduler.should_preempt());

    // Start the Lane 0 task
    let task = scheduler.next_task().unwrap();
    assert_eq!(task.lane, Lane::Lane0);
    scheduler.start_task(&task).unwrap();

    // No preemption for Lane 0 tasks
    assert!(!scheduler.should_preempt());
}

// =============================================================================
// Additional Edge Case Tests
// =============================================================================

#[test]
fn test_task_queue_priority_ordering() {
    let mut queue = TaskQueue::new(Lane::Lane1);

    // Add tasks in reverse priority order
    let task_low = Task::new(
        TaskId::new(1),
        "m".to_string(),
        "c".to_string(),
        Lane::Lane1,
    )
    .with_priority(Priority::Low);
    let task_normal = Task::new(
        TaskId::new(2),
        "m".to_string(),
        "c".to_string(),
        Lane::Lane1,
    )
    .with_priority(Priority::Normal);
    let task_high = Task::new(
        TaskId::new(3),
        "m".to_string(),
        "c".to_string(),
        Lane::Lane1,
    )
    .with_priority(Priority::High);
    let task_critical = Task::new(
        TaskId::new(4),
        "m".to_string(),
        "c".to_string(),
        Lane::Lane1,
    )
    .with_priority(Priority::Critical);

    queue.enqueue(task_low).unwrap();
    queue.enqueue(task_normal).unwrap();
    queue.enqueue(task_high).unwrap();
    queue.enqueue(task_critical).unwrap();

    // Should come out in priority order
    assert_eq!(queue.dequeue().unwrap().id, TaskId::new(4)); // Critical
    assert_eq!(queue.dequeue().unwrap().id, TaskId::new(3)); // High
    assert_eq!(queue.dequeue().unwrap().id, TaskId::new(2)); // Normal
    assert_eq!(queue.dequeue().unwrap().id, TaskId::new(1)); // Low
}

#[test]
fn test_epoch_tracker_isolation() {
    let mut tracker = EpochTracker::new();

    // Test On-Deck without becoming director
    let epoch = make_epoch(1, 0, 0);
    tracker.on_deck(epoch.clone(), false);

    // Should NOT be On-Deck (not a director)
    assert!(!tracker.is_on_deck());
    assert!(!tracker.should_drain_lane1());

    // Now become director
    tracker.on_deck(epoch.clone(), true);
    assert!(tracker.is_on_deck());
    assert!(tracker.should_drain_lane1());
}

#[test]
fn test_cancel_nonexistent_task() {
    let mut scheduler = SchedulerState::new();

    let result = scheduler.cancel_task(TaskId::new(999));
    assert!(matches!(result, Err(SchedulerError::TaskNotFound(_))));
}

#[test]
fn test_complete_wrong_task() {
    let mut scheduler = SchedulerState::new();
    scheduler.transition(NodeState::LoadingModels).unwrap();
    scheduler.transition(NodeState::Idle).unwrap();

    // Start task 0
    scheduler
        .enqueue_lane0("flux".to_string(), "cid".to_string())
        .unwrap();
    let task = scheduler.next_task().unwrap();
    scheduler.start_task(&task).unwrap();

    // Try to complete task 999 (wrong ID)
    let result = TaskResult::success("output".to_string(), 1000);
    let complete_result = scheduler.complete_task(TaskId::new(999), result);
    assert!(matches!(
        complete_result,
        Err(SchedulerError::TaskNotFound(_))
    ));
}

#[test]
fn test_error_state_recovery() {
    let mut scheduler = SchedulerState::new();
    scheduler.transition(NodeState::LoadingModels).unwrap();
    scheduler.transition(NodeState::Idle).unwrap();

    // Transition to error
    scheduler
        .transition(NodeState::Error("Test error".to_string()))
        .unwrap();
    assert!(matches!(scheduler.current_state(), NodeState::Error(_)));

    // Recover back to Idle
    scheduler.transition(NodeState::Idle).unwrap();
    assert_eq!(*scheduler.current_state(), NodeState::Idle);

    // Or transition to Stopping
    scheduler
        .transition(NodeState::Error("Fatal error".to_string()))
        .unwrap();
    scheduler.transition(NodeState::Stopping).unwrap();
    assert_eq!(*scheduler.current_state(), NodeState::Stopping);
}

#[test]
fn test_scheduler_stats() {
    let mut scheduler = SchedulerState::new();
    scheduler.transition(NodeState::LoadingModels).unwrap();
    scheduler.transition(NodeState::Idle).unwrap();

    // Add tasks to both lanes
    scheduler
        .enqueue_lane0("flux".to_string(), "cid1".to_string())
        .unwrap();
    scheduler
        .enqueue_lane0("lp".to_string(), "cid2".to_string())
        .unwrap();
    scheduler
        .enqueue_lane1("llama".to_string(), "cid3".to_string())
        .unwrap();

    let stats = scheduler.stats();

    assert_eq!(stats.current_state, NodeState::Idle);
    assert_eq!(stats.lane0_queue_len, 2);
    assert_eq!(stats.lane1_queue_len, 1);
    assert!(!stats.is_draining);
    assert!(!stats.has_active_task);
    assert!(stats.current_epoch.is_none());

    // Start On-Deck
    let epoch = make_epoch(1, 0, 0);
    scheduler.on_deck_received(epoch);

    let stats = scheduler.stats();
    assert!(stats.is_draining);
}
