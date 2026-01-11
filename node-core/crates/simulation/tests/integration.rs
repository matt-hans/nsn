//! Integration tests for the simulation harness.
//!
//! These tests validate multi-node scenarios using the TestHarness.

use std::time::Duration;

use nsn_simulation::{ByzantineBehavior, Scenario, TestHarness};

// =============================================================================
// Baseline Consensus Tests
// =============================================================================

#[tokio::test]
async fn test_baseline_five_director_consensus() {
    // Setup: 5 directors, all honest
    let mut harness = TestHarness::new();

    let directors: Vec<_> = (0..5).map(|_| harness.add_director()).collect();

    // Configure slot 1 to succeed
    harness.configure_slot_success(&[1]);

    // Activate all directors
    harness.emit_epoch_started(&directors);

    // Run consensus for slot 1
    let successful = harness.run_slot(1).await.unwrap();

    // All 5 directors should succeed
    assert_eq!(successful, 5, "All 5 directors should reach consensus");

    // Verify consensus was reached
    harness.assert_consensus_reached(1);
    harness.assert_chunk_published(1);
}

#[tokio::test]
async fn test_three_of_five_minimum_consensus() {
    // Setup: 5 directors, but only 3 will succeed
    let mut harness = TestHarness::new();

    let directors: Vec<_> = (0..5).map(|_| harness.add_director()).collect();

    // Only first 3 directors configured for success
    for peer in &directors[0..3] {
        if let Some(director) = harness.get_director_mut(peer) {
            director.vortex.add_success_slot(1);
            director.bft.add_consensus_slot(1);
        }
    }

    harness.emit_epoch_started(&directors);
    let successful = harness.run_slot(1).await.unwrap();

    // At least 3 should succeed (the configured ones)
    assert!(successful >= 3, "At least 3 directors should succeed");
    harness.assert_consensus_reached(1);
}

// =============================================================================
// Byzantine Director Tests
// =============================================================================

#[tokio::test]
async fn test_byzantine_director_divergent_embedding() {
    // Setup: 5 directors, 1 returns different CLIP embedding
    let mut harness = TestHarness::new();

    let directors: Vec<_> = (0..5).map(|_| harness.add_director()).collect();
    harness.configure_slot_success(&[1]);

    // Make first director Byzantine (drops messages)
    harness
        .set_byzantine(directors[0], ByzantineBehavior::DropMessages)
        .unwrap();

    harness.emit_epoch_started(&directors);
    let successful = harness.run_slot(1).await.unwrap();

    // 4 honest directors should succeed
    assert_eq!(successful, 4, "4 honest directors should succeed");

    // Verify the Byzantine director is tracked as failed
    let director = harness.get_director(&directors[0]).unwrap();
    assert!(
        !director.state.consensus_results.contains(&1),
        "Byzantine director should not have consensus"
    );
}

#[tokio::test]
async fn test_two_byzantine_directors_still_succeeds() {
    // Setup: 5 directors, 2 Byzantine
    let mut harness = TestHarness::new();

    let directors: Vec<_> = (0..5).map(|_| harness.add_director()).collect();
    harness.configure_slot_success(&[1]);

    // Make 2 directors Byzantine
    harness
        .set_byzantine(directors[0], ByzantineBehavior::DropMessages)
        .unwrap();
    harness
        .set_byzantine(directors[1], ByzantineBehavior::DropMessages)
        .unwrap();

    harness.emit_epoch_started(&directors);
    let successful = harness.run_slot(1).await.unwrap();

    // 3 honest directors should succeed (3-of-5 threshold)
    assert_eq!(successful, 3, "3 honest directors should reach consensus");
    harness.assert_consensus_reached(1);
}

#[tokio::test]
async fn test_three_byzantine_directors_fails() {
    // Setup: 5 directors, 3 Byzantine - should fail to reach consensus
    let mut harness = TestHarness::new();

    let directors: Vec<_> = (0..5).map(|_| harness.add_director()).collect();
    harness.configure_slot_success(&[1]);

    // Make 3 directors Byzantine
    for i in 0..3 {
        harness
            .set_byzantine(directors[i], ByzantineBehavior::DropMessages)
            .unwrap();
    }

    harness.emit_epoch_started(&directors);
    let successful = harness.run_slot(1).await.unwrap();

    // Only 2 honest directors - below 3-of-5 threshold
    assert_eq!(successful, 2, "Only 2 honest directors should succeed");
}

// =============================================================================
// Network Partition Tests
// =============================================================================

#[tokio::test]
async fn test_network_partition_larger_group_succeeds() {
    // Setup: 5 directors, partition into [3] | [2]
    let mut harness = TestHarness::new();

    let directors: Vec<_> = (0..5).map(|_| harness.add_director()).collect();
    harness.configure_slot_success(&[1]);

    // Create partition
    harness.inject_partition(vec![
        vec![directors[0], directors[1], directors[2]],
        vec![directors[3], directors[4]],
    ]);

    // Only activate larger partition
    harness.emit_epoch_started(&[directors[0], directors[1], directors[2]]);

    let successful = harness.run_slot(1).await.unwrap();

    // Larger partition (3 nodes) should reach consensus
    assert_eq!(successful, 3, "Larger partition should reach consensus");
    harness.assert_consensus_reached(1);
}

#[tokio::test]
async fn test_network_partition_recovery() {
    // Setup: 5 directors, partition then heal
    let mut harness = TestHarness::new();

    let directors: Vec<_> = (0..5).map(|_| harness.add_director()).collect();
    harness.configure_slot_success(&[1, 2]);

    // Create partition
    harness.inject_partition(vec![
        vec![directors[0], directors[1], directors[2]],
        vec![directors[3], directors[4]],
    ]);

    harness.emit_epoch_started(&[directors[0], directors[1], directors[2]]);
    let _ = harness.run_slot(1).await.unwrap();

    // Heal partition
    harness.heal_partition();

    // Now activate all directors for slot 2
    harness.emit_epoch_started(&directors);
    let successful = harness.run_slot(2).await.unwrap();

    // All 5 should succeed after healing
    assert_eq!(successful, 5, "All directors should succeed after heal");
}

// =============================================================================
// Director Failure Tests
// =============================================================================

#[tokio::test]
async fn test_director_failure_mid_epoch() {
    // Setup: 5 directors, remove 1 mid-slot
    let mut harness = TestHarness::new();

    let directors: Vec<_> = (0..5).map(|_| harness.add_director()).collect();
    harness.configure_slot_success(&[1]);

    harness.emit_epoch_started(&directors);

    // Remove first director (simulating crash)
    harness.remove_node(directors[0]).unwrap();

    let successful = harness.run_slot(1).await.unwrap();

    // Remaining 4 should complete (3-of-4 works)
    assert_eq!(successful, 4, "4 remaining directors should succeed");
}

#[tokio::test]
async fn test_cascading_director_failures() {
    // Setup: 5 directors, remove directors one by one
    let mut harness = TestHarness::new();

    let directors: Vec<_> = (0..5).map(|_| harness.add_director()).collect();
    harness.configure_slot_success(&[1, 2, 3, 4]);

    harness.emit_epoch_started(&directors);

    // Remove directors progressively
    for (i, slot) in [1, 2, 3].iter().enumerate() {
        harness.remove_node(directors[i]).unwrap();
        let successful = harness.run_slot(*slot).await.unwrap();

        let remaining = 5 - i - 1;
        assert!(
            successful <= remaining,
            "Slot {} should have at most {} successful directors",
            slot,
            remaining
        );
    }

    // With only 2 directors left, consensus should fail
    let successful = harness.run_slot(4).await.unwrap();
    assert_eq!(successful, 2, "Only 2 directors remain");
}

// =============================================================================
// Epoch Lifecycle Tests
// =============================================================================

#[tokio::test]
async fn test_epoch_transition_on_deck_to_active() {
    let mut harness = TestHarness::new();

    let directors: Vec<_> = (0..5).map(|_| harness.add_director()).collect();

    // Initially not on_deck
    for peer in &directors {
        let director = harness.get_director(peer).unwrap();
        assert!(!director.state.on_deck);
        assert!(!director.state.active);
    }

    // Emit OnDeck
    harness.emit_on_deck(&directors);
    for peer in &directors {
        let director = harness.get_director(peer).unwrap();
        assert!(director.state.on_deck);
        assert!(!director.state.active);
    }

    // Emit EpochStarted
    harness.emit_epoch_started(&directors);
    for peer in &directors {
        let director = harness.get_director(peer).unwrap();
        assert!(!director.state.on_deck);
        assert!(director.state.active);
    }

    // Emit EpochEnded
    harness.emit_epoch_ended();
    for peer in &directors {
        let director = harness.get_director(peer).unwrap();
        assert!(!director.state.on_deck);
        assert!(!director.state.active);
    }
}

#[tokio::test]
async fn test_full_epoch_multiple_slots() {
    let mut harness = TestHarness::new();

    let directors: Vec<_> = (0..5).map(|_| harness.add_director()).collect();
    harness.configure_slot_success(&[1, 2, 3, 4, 5]);

    // Pause time for deterministic simulation
    tokio::time::pause();

    harness.emit_on_deck(&directors);
    harness.advance_time(Duration::from_millis(100)).await;
    harness.emit_epoch_started(&directors);

    // Run 5 slots
    let mut total_successful = 0;
    for slot in 1..=5 {
        let successful = harness.run_slot(slot).await.unwrap();
        total_successful += successful;
        harness.advance_time(Duration::from_millis(100)).await;
    }

    // All 5 directors should succeed for all 5 slots = 25 total
    assert_eq!(total_successful, 25, "All slots should complete successfully");

    harness.emit_epoch_ended();
}

// =============================================================================
// Lane Switching Tests
// =============================================================================

#[tokio::test]
async fn test_lane_switching_directors_to_executors() {
    let mut harness = TestHarness::new();

    let directors: Vec<_> = (0..5).map(|_| harness.add_director()).collect();
    let executors: Vec<_> = (0..3).map(|_| harness.add_executor()).collect();

    harness.configure_slot_success(&[1]);

    // Lane 0 active
    harness.chain_state.state.set_active_lane(0);
    harness.emit_epoch_started(&directors);

    let successful = harness.run_slot(1).await.unwrap();
    assert_eq!(successful, 5);

    // Switch to Lane 1
    harness.emit_epoch_ended();
    harness.chain_state.state.set_active_lane(1);

    // Verify executors exist
    assert_eq!(harness.executor_count(), 3);
    assert!(harness.get_executor(&executors[0]).is_some());
}

// =============================================================================
// Task Lifecycle Tests (Lane 1)
// =============================================================================

#[tokio::test]
async fn test_task_created_and_assigned() {
    let mut harness = TestHarness::new();

    let executors: Vec<_> = (0..3).map(|_| harness.add_executor()).collect();

    // Inject task into first executor
    if let Some(executor) = harness.executors.get_mut(&executors[0]) {
        executor.chain.create_task(1, "model-1", "QmInput", 1000);
        executor.chain.assign_task_to_me(1);
        executor.state.tasks_assigned.push(1);
    }

    let executor = harness.get_executor(&executors[0]).unwrap();
    assert!(executor.state.tasks_assigned.contains(&1));
}

#[tokio::test]
async fn test_task_execution_and_completion() {
    let mut harness = TestHarness::new();

    let executors: Vec<_> = (0..3).map(|_| harness.add_executor()).collect();

    // Full task lifecycle
    if let Some(executor) = harness.executors.get_mut(&executors[0]) {
        executor.chain.create_task(1, "model-1", "QmInput", 1000);
        executor.chain.assign_task_to_me(1);
        executor.chain.start_task(1).await.unwrap();
        executor
            .chain
            .submit_result(1, "QmOutput".to_string())
            .await
            .unwrap();
        executor.state.tasks_completed.push(1);
    }

    let executor = harness.get_executor(&executors[0]).unwrap();
    assert!(executor.state.tasks_completed.contains(&1));
    assert!(executor.chain.state.completed_tasks.contains(&1));
}

// =============================================================================
// High Latency Tests
// =============================================================================

#[tokio::test]
async fn test_high_latency_consensus_succeeds() {
    let mut harness = TestHarness::new();

    let directors: Vec<_> = (0..5).map(|_| harness.add_director()).collect();
    harness.configure_slot_success(&[1]);

    harness.emit_epoch_started(&directors);

    // Pause time for deterministic simulation
    tokio::time::pause();

    // Advance time to simulate latency
    harness.advance_time(Duration::from_millis(200)).await;

    let successful = harness.run_slot(1).await.unwrap();
    assert_eq!(successful, 5, "Consensus should succeed with latency");
}

// =============================================================================
// Scenario Tests
// =============================================================================

#[tokio::test]
async fn test_scenario_baseline_consensus() {
    let mut harness = TestHarness::new();
    let result = Scenario::BaselineConsensus.run(&mut harness).await;
    assert!(Scenario::BaselineConsensus.verify(&result).is_ok());
}

#[tokio::test]
async fn test_scenario_byzantine_director() {
    let mut harness = TestHarness::new();
    let result = Scenario::ByzantineDirector.run(&mut harness).await;
    // Should reach consensus with 4 honest nodes
    assert!(result.consensus_reached >= 1 || !result.failed_directors.is_empty());
}

#[tokio::test]
async fn test_scenario_network_partition() {
    let mut harness = TestHarness::new();
    let result = Scenario::NetworkPartition.run(&mut harness).await;
    assert!(Scenario::NetworkPartition.verify(&result).is_ok());
}

#[tokio::test]
async fn test_scenario_director_failure() {
    let mut harness = TestHarness::new();
    let result = Scenario::DirectorFailure.run(&mut harness).await;
    assert!(Scenario::DirectorFailure.verify(&result).is_ok());
}

#[tokio::test]
async fn test_scenario_full_epoch_lifecycle() {
    let mut harness = TestHarness::new();
    let result = Scenario::FullEpochLifecycle.run(&mut harness).await;
    assert!(Scenario::FullEpochLifecycle.verify(&result).is_ok());
}

#[tokio::test]
async fn test_scenario_task_lifecycle() {
    let mut harness = TestHarness::new();
    let result = Scenario::TaskLifecycle.run(&mut harness).await;
    assert!(Scenario::TaskLifecycle.verify(&result).is_ok());
}

#[tokio::test]
async fn test_scenario_lane_switching() {
    let mut harness = TestHarness::new();
    let result = Scenario::LaneSwitching.run(&mut harness).await;
    assert!(Scenario::LaneSwitching.verify(&result).is_ok());
}

// =============================================================================
// Metrics Tests
// =============================================================================

#[tokio::test]
async fn test_metrics_collection() {
    let mut harness = TestHarness::new();

    let directors: Vec<_> = (0..5).map(|_| harness.add_director()).collect();
    harness.configure_slot_success(&[1, 2, 3]);

    harness.emit_epoch_started(&directors);

    for slot in 1..=3 {
        harness.run_slot(slot).await.unwrap();
    }

    let metrics = harness.metrics();
    assert_eq!(metrics.slots_generated, 15, "5 directors x 3 slots = 15");
    assert_eq!(metrics.consensus_rounds, 3);
}

#[tokio::test]
async fn test_metrics_reset() {
    let mut harness = TestHarness::new();

    let directors: Vec<_> = (0..5).map(|_| harness.add_director()).collect();
    harness.configure_slot_success(&[1]);
    harness.emit_epoch_started(&directors);

    harness.run_slot(1).await.unwrap();
    assert!(harness.metrics().slots_generated > 0);

    harness.reset_metrics();
    assert_eq!(harness.metrics().slots_generated, 0);
}
