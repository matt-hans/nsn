// Copyright 2024 Neural Sovereign Network
// This file is part of NSN Chain.

//! Unit tests for pallet-nsn-task-market

use crate::{mock::*, Error, Event, FailReason, TaskStatus};
use frame_support::{assert_noop, assert_ok};

// ============================================================================
// Green Path Tests (Happy Flows)
// ============================================================================

#[test]
fn create_task_intent_succeeds() {
    ExtBuilder::default().build().execute_with(|| {
        // GIVEN: Alice has 10,000 balance
        assert_eq!(Balances::free_balance(ALICE), 10_000);
        assert_eq!(Balances::reserved_balance(ALICE), 0);

        // WHEN: Alice creates a task with 100 escrow and 100 block deadline
        assert_ok!(NsnTaskMarket::create_task_intent(
            RuntimeOrigin::signed(ALICE),
            100,
            100
        ));

        // THEN: Task is created, escrow reserved
        let task = NsnTaskMarket::tasks(0).expect("Task should exist");
        assert_eq!(task.id, 0);
        assert_eq!(task.requester, ALICE);
        assert_eq!(task.executor, None);
        assert_eq!(task.status, TaskStatus::Open);
        assert_eq!(task.escrow, 100);
        assert_eq!(task.created_at, 1);
        assert_eq!(task.deadline, 101); // current_block (1) + deadline_blocks (100)

        // Verify escrow reserved
        assert_eq!(Balances::free_balance(ALICE), 9_900);
        assert_eq!(Balances::reserved_balance(ALICE), 100);

        // Verify task is in open queue
        assert_eq!(NsnTaskMarket::open_tasks().len(), 1);
        assert!(NsnTaskMarket::open_tasks().contains(&0));

        // Verify next task ID incremented
        assert_eq!(NsnTaskMarket::next_task_id(), 1);

        // Verify event
        let event = last_event();
        assert!(matches!(
            event,
            RuntimeEvent::NsnTaskMarket(Event::TaskCreated {
                task_id: 0,
                requester: ALICE,
                escrow: 100,
                deadline: 101
            })
        ));
    });
}

#[test]
fn accept_assignment_succeeds() {
    ExtBuilder::default().build().execute_with(|| {
        // GIVEN: Alice has created a task
        assert_ok!(NsnTaskMarket::create_task_intent(
            RuntimeOrigin::signed(ALICE),
            100,
            100
        ));

        // WHEN: Bob accepts the assignment
        assert_ok!(NsnTaskMarket::accept_assignment(
            RuntimeOrigin::signed(BOB),
            0
        ));

        // THEN: Task is assigned to Bob
        let task = NsnTaskMarket::tasks(0).expect("Task should exist");
        assert_eq!(task.status, TaskStatus::Assigned);
        assert_eq!(task.executor, Some(BOB));

        // Verify task removed from open queue
        assert_eq!(NsnTaskMarket::open_tasks().len(), 0);

        // Verify event
        let event = last_event();
        assert!(matches!(
            event,
            RuntimeEvent::NsnTaskMarket(Event::TaskAssigned {
                task_id: 0,
                executor: BOB
            })
        ));
    });
}

#[test]
fn complete_task_succeeds() {
    ExtBuilder::default().build().execute_with(|| {
        // GIVEN: Alice created a task, Bob accepted it
        assert_ok!(NsnTaskMarket::create_task_intent(
            RuntimeOrigin::signed(ALICE),
            100,
            100
        ));
        assert_ok!(NsnTaskMarket::accept_assignment(
            RuntimeOrigin::signed(BOB),
            0
        ));

        let alice_balance_before = Balances::free_balance(ALICE);
        let bob_balance_before = Balances::free_balance(BOB);

        // WHEN: Bob completes the task
        assert_ok!(NsnTaskMarket::complete_task(RuntimeOrigin::signed(BOB), 0));

        // THEN: Task is completed, payment transferred
        let task = NsnTaskMarket::tasks(0).expect("Task should exist");
        assert_eq!(task.status, TaskStatus::Completed);

        // Verify payment: unreserved from Alice, transferred to Bob
        assert_eq!(Balances::reserved_balance(ALICE), 0);
        assert_eq!(
            Balances::free_balance(ALICE),
            alice_balance_before // Was already unreserved before transfer
        );
        assert_eq!(Balances::free_balance(BOB), bob_balance_before + 100);

        // Verify event
        let event = last_event();
        assert!(matches!(
            event,
            RuntimeEvent::NsnTaskMarket(Event::TaskCompleted {
                task_id: 0,
                executor: BOB,
                payment: 100
            })
        ));
    });
}

#[test]
fn fail_task_by_executor_succeeds() {
    ExtBuilder::default().build().execute_with(|| {
        // GIVEN: Alice created a task, Bob accepted it
        assert_ok!(NsnTaskMarket::create_task_intent(
            RuntimeOrigin::signed(ALICE),
            100,
            100
        ));
        assert_ok!(NsnTaskMarket::accept_assignment(
            RuntimeOrigin::signed(BOB),
            0
        ));

        let alice_balance_before = Balances::free_balance(ALICE);

        // WHEN: Bob fails the task
        assert_ok!(NsnTaskMarket::fail_task(
            RuntimeOrigin::signed(BOB),
            0,
            FailReason::ExecutorFailed
        ));

        // THEN: Task is failed, escrow returned to Alice
        let task = NsnTaskMarket::tasks(0).expect("Task should exist");
        assert_eq!(task.status, TaskStatus::Failed);

        // Verify escrow returned
        assert_eq!(Balances::reserved_balance(ALICE), 0);
        assert_eq!(Balances::free_balance(ALICE), alice_balance_before + 100);

        // Verify event
        let event = last_event();
        assert!(matches!(
            event,
            RuntimeEvent::NsnTaskMarket(Event::TaskFailed {
                task_id: 0,
                reason: FailReason::ExecutorFailed
            })
        ));
    });
}

#[test]
fn fail_task_by_requester_on_assigned_task_succeeds() {
    ExtBuilder::default().build().execute_with(|| {
        // GIVEN: Alice created a task, Bob accepted it
        assert_ok!(NsnTaskMarket::create_task_intent(
            RuntimeOrigin::signed(ALICE),
            100,
            100
        ));
        assert_ok!(NsnTaskMarket::accept_assignment(
            RuntimeOrigin::signed(BOB),
            0
        ));

        // WHEN: Alice (requester) fails the assigned task
        assert_ok!(NsnTaskMarket::fail_task(
            RuntimeOrigin::signed(ALICE),
            0,
            FailReason::Cancelled
        ));

        // THEN: Task is failed
        let task = NsnTaskMarket::tasks(0).expect("Task should exist");
        assert_eq!(task.status, TaskStatus::Failed);

        // Verify escrow returned
        assert_eq!(Balances::reserved_balance(ALICE), 0);
    });
}

#[test]
fn cancel_open_task_succeeds() {
    ExtBuilder::default().build().execute_with(|| {
        // GIVEN: Alice created a task (still open)
        assert_ok!(NsnTaskMarket::create_task_intent(
            RuntimeOrigin::signed(ALICE),
            100,
            100
        ));
        assert_eq!(NsnTaskMarket::open_tasks().len(), 1);

        let alice_balance_before = Balances::free_balance(ALICE);

        // WHEN: Alice cancels the task
        assert_ok!(NsnTaskMarket::fail_task(
            RuntimeOrigin::signed(ALICE),
            0,
            FailReason::Cancelled
        ));

        // THEN: Task is failed, removed from open queue, escrow returned
        let task = NsnTaskMarket::tasks(0).expect("Task should exist");
        assert_eq!(task.status, TaskStatus::Failed);
        assert_eq!(NsnTaskMarket::open_tasks().len(), 0);
        assert_eq!(Balances::free_balance(ALICE), alice_balance_before + 100);
    });
}

// ============================================================================
// Red Path Tests (Error Cases)
// ============================================================================

#[test]
fn create_task_insufficient_balance_fails() {
    ExtBuilder::default()
        .with_balances(vec![(ALICE, 50)])
        .build()
        .execute_with(|| {
            // WHEN: Alice tries to create a task with more escrow than balance
            assert_noop!(
                NsnTaskMarket::create_task_intent(RuntimeOrigin::signed(ALICE), 100, 100),
                Error::<Test>::InsufficientBalance
            );
        });
}

#[test]
fn create_task_invalid_deadline_fails() {
    ExtBuilder::default().build().execute_with(|| {
        // WHEN: Alice tries to create a task with zero deadline
        assert_noop!(
            NsnTaskMarket::create_task_intent(RuntimeOrigin::signed(ALICE), 100, 0),
            Error::<Test>::InvalidDeadline
        );
    });
}

#[test]
fn create_task_exceeds_max_pending_fails() {
    ExtBuilder::default()
        .with_balances(vec![(ALICE, 1_000_000)])
        .build()
        .execute_with(|| {
            // GIVEN: Create MaxPendingTasks tasks (100)
            for i in 0..100 {
                assert_ok!(NsnTaskMarket::create_task_intent(
                    RuntimeOrigin::signed(ALICE),
                    10,
                    100
                ));
                assert_eq!(NsnTaskMarket::next_task_id(), i + 1);
            }

            // WHEN: Try to create one more
            assert_noop!(
                NsnTaskMarket::create_task_intent(RuntimeOrigin::signed(ALICE), 10, 100),
                Error::<Test>::TooManyPendingTasks
            );
        });
}

#[test]
fn accept_assignment_task_not_found_fails() {
    ExtBuilder::default().build().execute_with(|| {
        // WHEN: Bob tries to accept a non-existent task
        assert_noop!(
            NsnTaskMarket::accept_assignment(RuntimeOrigin::signed(BOB), 999),
            Error::<Test>::TaskNotFound
        );
    });
}

#[test]
fn accept_assignment_task_not_open_fails() {
    ExtBuilder::default().build().execute_with(|| {
        // GIVEN: Task is already assigned
        assert_ok!(NsnTaskMarket::create_task_intent(
            RuntimeOrigin::signed(ALICE),
            100,
            100
        ));
        assert_ok!(NsnTaskMarket::accept_assignment(
            RuntimeOrigin::signed(BOB),
            0
        ));

        // WHEN: Charlie tries to accept the same task
        assert_noop!(
            NsnTaskMarket::accept_assignment(RuntimeOrigin::signed(CHARLIE), 0),
            Error::<Test>::TaskNotOpen
        );
    });
}

#[test]
fn complete_task_not_found_fails() {
    ExtBuilder::default().build().execute_with(|| {
        // WHEN: Bob tries to complete a non-existent task
        assert_noop!(
            NsnTaskMarket::complete_task(RuntimeOrigin::signed(BOB), 999),
            Error::<Test>::TaskNotFound
        );
    });
}

#[test]
fn complete_task_not_assigned_fails() {
    ExtBuilder::default().build().execute_with(|| {
        // GIVEN: Task is open (not assigned)
        assert_ok!(NsnTaskMarket::create_task_intent(
            RuntimeOrigin::signed(ALICE),
            100,
            100
        ));

        // WHEN: Bob tries to complete an open task
        assert_noop!(
            NsnTaskMarket::complete_task(RuntimeOrigin::signed(BOB), 0),
            Error::<Test>::TaskNotAssigned
        );
    });
}

#[test]
fn complete_task_not_executor_fails() {
    ExtBuilder::default().build().execute_with(|| {
        // GIVEN: Task is assigned to Bob
        assert_ok!(NsnTaskMarket::create_task_intent(
            RuntimeOrigin::signed(ALICE),
            100,
            100
        ));
        assert_ok!(NsnTaskMarket::accept_assignment(
            RuntimeOrigin::signed(BOB),
            0
        ));

        // WHEN: Charlie tries to complete Bob's task
        assert_noop!(
            NsnTaskMarket::complete_task(RuntimeOrigin::signed(CHARLIE), 0),
            Error::<Test>::NotExecutor
        );
    });
}

#[test]
fn fail_task_not_found_fails() {
    ExtBuilder::default().build().execute_with(|| {
        // WHEN: Bob tries to fail a non-existent task
        assert_noop!(
            NsnTaskMarket::fail_task(RuntimeOrigin::signed(BOB), 999, FailReason::ExecutorFailed),
            Error::<Test>::TaskNotFound
        );
    });
}

#[test]
fn fail_open_task_not_requester_fails() {
    ExtBuilder::default().build().execute_with(|| {
        // GIVEN: Alice created an open task
        assert_ok!(NsnTaskMarket::create_task_intent(
            RuntimeOrigin::signed(ALICE),
            100,
            100
        ));

        // WHEN: Bob (not requester) tries to cancel the open task
        assert_noop!(
            NsnTaskMarket::fail_task(RuntimeOrigin::signed(BOB), 0, FailReason::Cancelled),
            Error::<Test>::NotRequester
        );
    });
}

#[test]
fn fail_assigned_task_by_third_party_fails() {
    ExtBuilder::default().build().execute_with(|| {
        // GIVEN: Alice created a task, Bob accepted it
        assert_ok!(NsnTaskMarket::create_task_intent(
            RuntimeOrigin::signed(ALICE),
            100,
            100
        ));
        assert_ok!(NsnTaskMarket::accept_assignment(
            RuntimeOrigin::signed(BOB),
            0
        ));

        // WHEN: Charlie (neither executor nor requester) tries to fail
        assert_noop!(
            NsnTaskMarket::fail_task(
                RuntimeOrigin::signed(CHARLIE),
                0,
                FailReason::ExecutorFailed
            ),
            Error::<Test>::NotExecutor
        );
    });
}

#[test]
fn complete_already_completed_task_fails() {
    ExtBuilder::default().build().execute_with(|| {
        // GIVEN: Task is completed
        assert_ok!(NsnTaskMarket::create_task_intent(
            RuntimeOrigin::signed(ALICE),
            100,
            100
        ));
        assert_ok!(NsnTaskMarket::accept_assignment(
            RuntimeOrigin::signed(BOB),
            0
        ));
        assert_ok!(NsnTaskMarket::complete_task(RuntimeOrigin::signed(BOB), 0));

        // WHEN: Bob tries to complete again
        assert_noop!(
            NsnTaskMarket::complete_task(RuntimeOrigin::signed(BOB), 0),
            Error::<Test>::TaskNotAssigned
        );
    });
}

#[test]
fn fail_already_failed_task_fails() {
    ExtBuilder::default().build().execute_with(|| {
        // GIVEN: Task is failed
        assert_ok!(NsnTaskMarket::create_task_intent(
            RuntimeOrigin::signed(ALICE),
            100,
            100
        ));
        assert_ok!(NsnTaskMarket::accept_assignment(
            RuntimeOrigin::signed(BOB),
            0
        ));
        assert_ok!(NsnTaskMarket::fail_task(
            RuntimeOrigin::signed(BOB),
            0,
            FailReason::ExecutorFailed
        ));

        // WHEN: Bob tries to fail again
        assert_noop!(
            NsnTaskMarket::fail_task(RuntimeOrigin::signed(BOB), 0, FailReason::ExecutorFailed),
            Error::<Test>::TaskNotAssigned
        );
    });
}

// ============================================================================
// Boundary and Edge Case Tests
// ============================================================================

#[test]
fn multiple_tasks_created_correctly() {
    ExtBuilder::default().build().execute_with(|| {
        // WHEN: Multiple tasks are created
        for i in 0..5 {
            assert_ok!(NsnTaskMarket::create_task_intent(
                RuntimeOrigin::signed(ALICE),
                100 + i as u128,
                100
            ));
        }

        // THEN: All tasks exist with correct IDs and escrows
        for i in 0..5 {
            let task = NsnTaskMarket::tasks(i).expect("Task should exist");
            assert_eq!(task.id, i);
            assert_eq!(task.escrow, 100 + i as u128);
        }

        assert_eq!(NsnTaskMarket::next_task_id(), 5);
        assert_eq!(NsnTaskMarket::open_tasks().len(), 5);
        assert_eq!(
            Balances::reserved_balance(ALICE),
            100 + 101 + 102 + 103 + 104
        );
    });
}

#[test]
fn open_tasks_queue_maintained_correctly() {
    ExtBuilder::default().build().execute_with(|| {
        // Create 3 tasks
        for _ in 0..3 {
            assert_ok!(NsnTaskMarket::create_task_intent(
                RuntimeOrigin::signed(ALICE),
                100,
                100
            ));
        }
        assert_eq!(NsnTaskMarket::open_tasks().len(), 3);
        assert!(NsnTaskMarket::open_tasks().contains(&0));
        assert!(NsnTaskMarket::open_tasks().contains(&1));
        assert!(NsnTaskMarket::open_tasks().contains(&2));

        // Accept task 1 (middle one)
        assert_ok!(NsnTaskMarket::accept_assignment(
            RuntimeOrigin::signed(BOB),
            1
        ));
        assert_eq!(NsnTaskMarket::open_tasks().len(), 2);
        assert!(NsnTaskMarket::open_tasks().contains(&0));
        assert!(!NsnTaskMarket::open_tasks().contains(&1));
        assert!(NsnTaskMarket::open_tasks().contains(&2));

        // Cancel task 0
        assert_ok!(NsnTaskMarket::fail_task(
            RuntimeOrigin::signed(ALICE),
            0,
            FailReason::Cancelled
        ));
        assert_eq!(NsnTaskMarket::open_tasks().len(), 1);
        assert!(!NsnTaskMarket::open_tasks().contains(&0));
        assert!(NsnTaskMarket::open_tasks().contains(&2));
    });
}

#[test]
fn task_escrow_zero_value() {
    ExtBuilder::default().build().execute_with(|| {
        // WHEN: Create task with zero escrow
        assert_ok!(NsnTaskMarket::create_task_intent(
            RuntimeOrigin::signed(ALICE),
            0,
            100
        ));

        // THEN: Task created, no escrow reserved
        let task = NsnTaskMarket::tasks(0).expect("Task should exist");
        assert_eq!(task.escrow, 0);
        assert_eq!(Balances::reserved_balance(ALICE), 0);
        assert_eq!(Balances::free_balance(ALICE), 10_000);
    });
}

#[test]
fn complete_task_zero_escrow() {
    ExtBuilder::default().build().execute_with(|| {
        // GIVEN: Task with zero escrow
        assert_ok!(NsnTaskMarket::create_task_intent(
            RuntimeOrigin::signed(ALICE),
            0,
            100
        ));
        assert_ok!(NsnTaskMarket::accept_assignment(
            RuntimeOrigin::signed(BOB),
            0
        ));

        let bob_balance_before = Balances::free_balance(BOB);

        // WHEN: Complete task
        assert_ok!(NsnTaskMarket::complete_task(RuntimeOrigin::signed(BOB), 0));

        // THEN: No balance change
        assert_eq!(Balances::free_balance(BOB), bob_balance_before);
    });
}

#[test]
fn deadline_calculation_correct() {
    ExtBuilder::default().build().execute_with(|| {
        // GIVEN: Current block is 1
        assert_eq!(System::block_number(), 1);

        // WHEN: Create task with 50 block deadline
        assert_ok!(NsnTaskMarket::create_task_intent(
            RuntimeOrigin::signed(ALICE),
            100,
            50
        ));

        // THEN: Deadline is current_block + deadline_blocks
        let task = NsnTaskMarket::tasks(0).expect("Task should exist");
        assert_eq!(task.created_at, 1);
        assert_eq!(task.deadline, 51);

        // Advance to block 10, create another task
        roll_to(10);
        assert_ok!(NsnTaskMarket::create_task_intent(
            RuntimeOrigin::signed(ALICE),
            100,
            50
        ));

        let task2 = NsnTaskMarket::tasks(1).expect("Task should exist");
        assert_eq!(task2.created_at, 10);
        assert_eq!(task2.deadline, 60);
    });
}

#[test]
fn helper_functions_work() {
    ExtBuilder::default().build().execute_with(|| {
        // Initial state
        assert_eq!(NsnTaskMarket::open_task_count(), 0);
        assert!(!NsnTaskMarket::task_exists(0));

        // Create task
        assert_ok!(NsnTaskMarket::create_task_intent(
            RuntimeOrigin::signed(ALICE),
            100,
            100
        ));

        // After creation
        assert_eq!(NsnTaskMarket::open_task_count(), 1);
        assert!(NsnTaskMarket::task_exists(0));
        assert!(!NsnTaskMarket::task_exists(1));

        // After assignment
        assert_ok!(NsnTaskMarket::accept_assignment(
            RuntimeOrigin::signed(BOB),
            0
        ));
        assert_eq!(NsnTaskMarket::open_task_count(), 0);
        assert!(NsnTaskMarket::task_exists(0)); // Task still exists, just not open
    });
}

#[test]
fn self_assignment_works() {
    ExtBuilder::default().build().execute_with(|| {
        // GIVEN: Alice creates a task
        assert_ok!(NsnTaskMarket::create_task_intent(
            RuntimeOrigin::signed(ALICE),
            100,
            100
        ));

        // WHEN: Alice accepts her own task
        assert_ok!(NsnTaskMarket::accept_assignment(
            RuntimeOrigin::signed(ALICE),
            0
        ));

        // THEN: Alice is the executor
        let task = NsnTaskMarket::tasks(0).expect("Task should exist");
        assert_eq!(task.executor, Some(ALICE));
        assert_eq!(task.status, TaskStatus::Assigned);
    });
}

#[test]
fn self_completion_works() {
    ExtBuilder::default().build().execute_with(|| {
        // GIVEN: Alice creates and accepts her own task
        assert_ok!(NsnTaskMarket::create_task_intent(
            RuntimeOrigin::signed(ALICE),
            100,
            100
        ));
        assert_ok!(NsnTaskMarket::accept_assignment(
            RuntimeOrigin::signed(ALICE),
            0
        ));

        let alice_balance_before = Balances::free_balance(ALICE);
        let alice_reserved_before = Balances::reserved_balance(ALICE);

        // WHEN: Alice completes her own task
        assert_ok!(NsnTaskMarket::complete_task(
            RuntimeOrigin::signed(ALICE),
            0
        ));

        // THEN: Escrow is unreserved and "transferred" to self (net zero change)
        assert_eq!(Balances::reserved_balance(ALICE), 0);
        assert_eq!(
            Balances::free_balance(ALICE),
            alice_balance_before + alice_reserved_before
        );
    });
}

#[test]
fn fail_reasons_are_recorded() {
    ExtBuilder::default().build().execute_with(|| {
        let reasons = vec![
            FailReason::ExecutorFailed,
            FailReason::Cancelled,
            FailReason::DeadlineExceeded,
            FailReason::Other,
        ];

        for (i, reason) in reasons.into_iter().enumerate() {
            // Create and assign task
            assert_ok!(NsnTaskMarket::create_task_intent(
                RuntimeOrigin::signed(ALICE),
                100,
                100
            ));
            assert_ok!(NsnTaskMarket::accept_assignment(
                RuntimeOrigin::signed(BOB),
                i as u64
            ));

            // Fail with specific reason
            assert_ok!(NsnTaskMarket::fail_task(
                RuntimeOrigin::signed(BOB),
                i as u64,
                reason.clone()
            ));

            // Verify event contains correct reason
            let event = last_event();
            assert!(matches!(
                event,
                RuntimeEvent::NsnTaskMarket(Event::TaskFailed { reason: r, .. }) if r == reason
            ));
        }
    });
}

// ============================================================================
// Integration-Style Tests
// ============================================================================

#[test]
fn full_task_lifecycle_success() {
    ExtBuilder::default().build().execute_with(|| {
        let initial_alice_balance = Balances::free_balance(ALICE);
        let initial_bob_balance = Balances::free_balance(BOB);

        // Step 1: Create task
        assert_ok!(NsnTaskMarket::create_task_intent(
            RuntimeOrigin::signed(ALICE),
            500,
            100
        ));
        assert_eq!(Balances::free_balance(ALICE), initial_alice_balance - 500);
        assert_eq!(Balances::reserved_balance(ALICE), 500);

        // Step 2: Accept assignment
        assert_ok!(NsnTaskMarket::accept_assignment(
            RuntimeOrigin::signed(BOB),
            0
        ));

        // Step 3: Complete task
        assert_ok!(NsnTaskMarket::complete_task(RuntimeOrigin::signed(BOB), 0));

        // Final state verification
        assert_eq!(Balances::reserved_balance(ALICE), 0);
        assert_eq!(Balances::free_balance(ALICE), initial_alice_balance - 500);
        assert_eq!(Balances::free_balance(BOB), initial_bob_balance + 500);

        let task = NsnTaskMarket::tasks(0).expect("Task should exist");
        assert_eq!(task.status, TaskStatus::Completed);
    });
}

#[test]
fn full_task_lifecycle_failure() {
    ExtBuilder::default().build().execute_with(|| {
        let initial_alice_balance = Balances::free_balance(ALICE);
        let initial_bob_balance = Balances::free_balance(BOB);

        // Step 1: Create task
        assert_ok!(NsnTaskMarket::create_task_intent(
            RuntimeOrigin::signed(ALICE),
            500,
            100
        ));

        // Step 2: Accept assignment
        assert_ok!(NsnTaskMarket::accept_assignment(
            RuntimeOrigin::signed(BOB),
            0
        ));

        // Step 3: Fail task
        assert_ok!(NsnTaskMarket::fail_task(
            RuntimeOrigin::signed(BOB),
            0,
            FailReason::ExecutorFailed
        ));

        // Final state verification: escrow returned to Alice
        assert_eq!(Balances::reserved_balance(ALICE), 0);
        assert_eq!(Balances::free_balance(ALICE), initial_alice_balance);
        assert_eq!(Balances::free_balance(BOB), initial_bob_balance);

        let task = NsnTaskMarket::tasks(0).expect("Task should exist");
        assert_eq!(task.status, TaskStatus::Failed);
    });
}

#[test]
fn multiple_concurrent_tasks_lifecycle() {
    ExtBuilder::default().build().execute_with(|| {
        // Alice creates 3 tasks
        for _ in 0..3 {
            assert_ok!(NsnTaskMarket::create_task_intent(
                RuntimeOrigin::signed(ALICE),
                100,
                100
            ));
        }

        // Bob accepts task 0, Charlie accepts task 1
        assert_ok!(NsnTaskMarket::accept_assignment(
            RuntimeOrigin::signed(BOB),
            0
        ));
        assert_ok!(NsnTaskMarket::accept_assignment(
            RuntimeOrigin::signed(CHARLIE),
            1
        ));

        // Bob completes, Charlie fails
        assert_ok!(NsnTaskMarket::complete_task(RuntimeOrigin::signed(BOB), 0));
        assert_ok!(NsnTaskMarket::fail_task(
            RuntimeOrigin::signed(CHARLIE),
            1,
            FailReason::ExecutorFailed
        ));

        // Alice cancels task 2
        assert_ok!(NsnTaskMarket::fail_task(
            RuntimeOrigin::signed(ALICE),
            2,
            FailReason::Cancelled
        ));

        // Verify final states
        assert_eq!(
            NsnTaskMarket::tasks(0).unwrap().status,
            TaskStatus::Completed
        );
        assert_eq!(NsnTaskMarket::tasks(1).unwrap().status, TaskStatus::Failed);
        assert_eq!(NsnTaskMarket::tasks(2).unwrap().status, TaskStatus::Failed);

        // Verify open queue is empty
        assert_eq!(NsnTaskMarket::open_tasks().len(), 0);
    });
}
