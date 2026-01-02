// Copyright 2024 Neural Sovereign Network
// This file is part of NSN Chain.

//! Unit tests for pallet-nsn-task-market

use crate::{mock::*, Error, Event, FailReason, TaskLane, TaskPriority, TaskStatus};
use frame_support::{assert_noop, assert_ok, traits::Hooks, BoundedVec};

// ============================================================================
// Helper Functions
// ============================================================================

/// Create default model requirements for testing
fn default_model_requirements() -> BoundedVec<u8, MaxModelIdLen> {
    BoundedVec::try_from(b"flux-schnell".to_vec()).unwrap()
}

/// Create a default input_cid for testing
fn default_input_cid() -> BoundedVec<u8, MaxCidLen> {
    BoundedVec::try_from(b"QmInputCid123456789".to_vec()).unwrap()
}

/// Create a default output_cid for testing
fn default_output_cid() -> BoundedVec<u8, MaxCidLen> {
    BoundedVec::try_from(b"QmOutputCid123456789".to_vec()).unwrap()
}

/// Create a default attestation CID for testing
fn default_attestation_cid() -> BoundedVec<u8, MaxCidLen> {
    BoundedVec::try_from(b"QmAttestationCid123456789".to_vec()).unwrap()
}

/// Default compute budget for testing
const DEFAULT_COMPUTE_BUDGET: u64 = 1000;

/// Accept assignment and return the selected executor
fn accept_and_get_executor(task_id: u64) -> AccountId {
    assert_ok!(NsnTaskMarket::accept_assignment(
        RuntimeOrigin::signed(BOB),
        task_id
    ));
    let task = NsnTaskMarket::tasks(task_id).expect("Task should exist");
    task.executor.expect("Executor should be assigned")
}

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
            TaskLane::Lane1,
            TaskPriority::Normal,
            default_model_requirements(),
            default_input_cid(),
            DEFAULT_COMPUTE_BUDGET,
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
        assert_eq!(task.model_requirements, default_model_requirements());
        assert_eq!(task.input_cid, default_input_cid());
        assert_eq!(task.compute_budget, DEFAULT_COMPUTE_BUDGET);
        assert_eq!(task.output_cid, None);

        // Verify escrow reserved
        assert_eq!(Balances::free_balance(ALICE), 9_900);
        assert_eq!(Balances::reserved_balance(ALICE), 100);

        // Verify task is in open queue
        assert_eq!(NsnTaskMarket::open_lane1_tasks().len(), 1);
        assert!(NsnTaskMarket::open_lane1_tasks().contains(&0));

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
                deadline: 101,
                lane: TaskLane::Lane1,
                priority: TaskPriority::Normal,
                model_requirements: _
            })
        ));
    });
}

// ============================================================================
// Renderer Registry Enforcement
// ============================================================================

#[test]
fn create_task_intent_fails_for_unregistered_renderer() {
    ExtBuilder::default().build().execute_with(|| {
        let unregistered: BoundedVec<u8, MaxModelIdLen> =
            BoundedVec::try_from(b"unknown-renderer".to_vec()).unwrap();

        assert_noop!(
            NsnTaskMarket::create_task_intent(
                RuntimeOrigin::signed(ALICE),
                TaskLane::Lane1,
                TaskPriority::Normal,
                unregistered,
                default_input_cid(),
                DEFAULT_COMPUTE_BUDGET,
                100,
                100
            ),
            Error::<Test>::RendererNotRegistered
        );
    });
}

#[test]
fn create_task_intent_fails_for_lane_mismatch() {
    ExtBuilder::default().build().execute_with(|| {
        let lane0_renderer: BoundedVec<u8, MaxModelIdLen> =
            BoundedVec::try_from(b"lane0-renderer".to_vec()).unwrap();

        assert_ok!(NsnTaskMarket::register_renderer(
            RuntimeOrigin::root(),
            lane0_renderer.clone(),
            TaskLane::Lane0,
            true,
            10_000,
            6_000
        ));

        assert_noop!(
            NsnTaskMarket::create_task_intent(
                RuntimeOrigin::signed(ALICE),
                TaskLane::Lane1,
                TaskPriority::Normal,
                lane0_renderer,
                default_input_cid(),
                DEFAULT_COMPUTE_BUDGET,
                100,
                100
            ),
            Error::<Test>::RendererLaneMismatch
        );
    });
}

#[test]
fn register_renderer_rejects_lane0_nondeterministic() {
    ExtBuilder::default().build().execute_with(|| {
        let lane0_renderer: BoundedVec<u8, MaxModelIdLen> =
            BoundedVec::try_from(b"lane0-nondet".to_vec()).unwrap();

        assert_noop!(
            NsnTaskMarket::register_renderer(
                RuntimeOrigin::root(),
                lane0_renderer,
                TaskLane::Lane0,
                false,
                10_000,
                6_000
            ),
            Error::<Test>::RendererNotDeterministic
        );
    });
}

#[test]
fn lane0_priority_blocks_lane1_assignment() {
    ExtBuilder::default().build().execute_with(|| {
        let lane0_renderer: BoundedVec<u8, MaxModelIdLen> =
            BoundedVec::try_from(b"lane0-renderer".to_vec()).unwrap();

        assert_ok!(NsnTaskMarket::register_renderer(
            RuntimeOrigin::root(),
            lane0_renderer.clone(),
            TaskLane::Lane0,
            true,
            10_000,
            6_000
        ));

        // Create Lane 0 task
        assert_ok!(NsnTaskMarket::create_task_intent(
            RuntimeOrigin::signed(ALICE),
            TaskLane::Lane0,
            TaskPriority::Normal,
            lane0_renderer,
            default_input_cid(),
            DEFAULT_COMPUTE_BUDGET,
            100,
            100
        ));

        // Create Lane 1 task
        assert_ok!(NsnTaskMarket::create_task_intent(
            RuntimeOrigin::signed(ALICE),
            TaskLane::Lane1,
            TaskPriority::Normal,
            default_model_requirements(),
            default_input_cid(),
            DEFAULT_COMPUTE_BUDGET,
            100,
            100
        ));

        // Accepting Lane 1 assignment should be blocked
        assert_noop!(
            NsnTaskMarket::accept_assignment(RuntimeOrigin::signed(BOB), 1),
            Error::<Test>::Lane0Priority
        );
    });
}

#[test]
fn preempt_lane1_tasks_when_lane0_pending() {
    ExtBuilder::default().build().execute_with(|| {
        // Create two Lane 1 tasks and assign them
        assert_ok!(NsnTaskMarket::create_task_intent(
            RuntimeOrigin::signed(ALICE),
            TaskLane::Lane1,
            TaskPriority::Normal,
            default_model_requirements(),
            default_input_cid(),
            DEFAULT_COMPUTE_BUDGET,
            100,
            100
        ));
        assert_ok!(NsnTaskMarket::create_task_intent(
            RuntimeOrigin::signed(ALICE),
            TaskLane::Lane1,
            TaskPriority::Normal,
            default_model_requirements(),
            default_input_cid(),
            DEFAULT_COMPUTE_BUDGET,
            100,
            100
        ));

        let _executor = accept_and_get_executor(0);
        assert_ok!(NsnTaskMarket::accept_assignment(
            RuntimeOrigin::signed(BOB),
            1
        ));

        assert_eq!(NsnTaskMarket::assigned_lane1_tasks().len(), 2);

        // Create a Lane 0 renderer + task to trigger preemption
        let lane0_renderer: BoundedVec<u8, MaxModelIdLen> =
            BoundedVec::try_from(b"lane0-preempt".to_vec()).unwrap();
        assert_ok!(NsnTaskMarket::register_renderer(
            RuntimeOrigin::root(),
            lane0_renderer.clone(),
            TaskLane::Lane0,
            true,
            10_000,
            6_000
        ));
        assert_ok!(NsnTaskMarket::create_task_intent(
            RuntimeOrigin::signed(ALICE),
            TaskLane::Lane0,
            TaskPriority::Normal,
            lane0_renderer,
            default_input_cid(),
            DEFAULT_COMPUTE_BUDGET,
            100,
            100
        ));

        // Trigger on_initialize to preempt
        let current = System::block_number();
        NsnTaskMarket::on_initialize(current);

        let task0 = NsnTaskMarket::tasks(0).expect("task 0 exists");
        let task1 = NsnTaskMarket::tasks(1).expect("task 1 exists");
        assert_eq!(task0.status, TaskStatus::Failed);
        assert_eq!(task1.status, TaskStatus::Failed);
        assert!(NsnTaskMarket::assigned_lane1_tasks().is_empty());
    });
}

#[test]
fn renderer_deregister_does_not_cancel_inflight_task() {
    ExtBuilder::default().build().execute_with(|| {
        let renderer_id: BoundedVec<u8, MaxModelIdLen> =
            BoundedVec::try_from(b"inflight-renderer".to_vec()).unwrap();

        assert_ok!(NsnTaskMarket::register_renderer(
            RuntimeOrigin::root(),
            renderer_id.clone(),
            TaskLane::Lane1,
            false,
            60_000,
            4_000
        ));

        assert_ok!(NsnTaskMarket::create_task_intent(
            RuntimeOrigin::signed(ALICE),
            TaskLane::Lane1,
            TaskPriority::Normal,
            renderer_id.clone(),
            default_input_cid(),
            DEFAULT_COMPUTE_BUDGET,
            100,
            100
        ));

        let executor = accept_and_get_executor(0);

        assert_ok!(NsnTaskMarket::deregister_renderer(
            RuntimeOrigin::root(),
            renderer_id
        ));

        assert_ok!(NsnTaskMarket::submit_result(
            RuntimeOrigin::signed(executor),
            0,
            default_output_cid(),
            None
        ));
    });
}

#[test]
fn lane0_completion_requires_attestation() {
    ExtBuilder::default().build().execute_with(|| {
        let lane0_renderer: BoundedVec<u8, MaxModelIdLen> =
            BoundedVec::try_from(b"lane0-attest".to_vec()).unwrap();

        assert_ok!(NsnTaskMarket::register_renderer(
            RuntimeOrigin::root(),
            lane0_renderer.clone(),
            TaskLane::Lane0,
            true,
            10_000,
            6_000
        ));

        assert_ok!(NsnTaskMarket::create_task_intent(
            RuntimeOrigin::signed(ALICE),
            TaskLane::Lane0,
            TaskPriority::Normal,
            lane0_renderer,
            default_input_cid(),
            DEFAULT_COMPUTE_BUDGET,
            100,
            100
        ));

        let executor = accept_and_get_executor(0);

        assert_noop!(
            NsnTaskMarket::submit_result(
                RuntimeOrigin::signed(executor),
                0,
                default_output_cid(),
                None
            ),
            Error::<Test>::MissingAttestation
        );
    });
}

#[test]
fn accept_assignment_succeeds() {
    ExtBuilder::default().build().execute_with(|| {
        // GIVEN: Alice has created a task
        assert_ok!(NsnTaskMarket::create_task_intent(
            RuntimeOrigin::signed(ALICE),
            TaskLane::Lane1,
            TaskPriority::Normal,
            default_model_requirements(),
            default_input_cid(),
            DEFAULT_COMPUTE_BUDGET,
            100,
            100
        ));

        // WHEN: Assignment is accepted
        let executor = accept_and_get_executor(0);

        // THEN: Task is assigned to executor
        let task = NsnTaskMarket::tasks(0).expect("Task should exist");
        assert_eq!(task.status, TaskStatus::Assigned);
        assert_eq!(task.executor, Some(executor));

        // Verify task removed from open queue
        assert_eq!(NsnTaskMarket::open_lane1_tasks().len(), 0);

        // Verify event
        let event = last_event();
        assert!(matches!(
            event,
            RuntimeEvent::NsnTaskMarket(Event::TaskAssigned {
                task_id: 0,
                executor: e
            }) if e == executor
        ));
    });
}

#[test]
fn submit_result_requires_verification_before_payout() {
    ExtBuilder::default().build().execute_with(|| {
        // GIVEN: Alice created a task, Bob accepted it
        assert_ok!(NsnTaskMarket::create_task_intent(
            RuntimeOrigin::signed(ALICE),
            TaskLane::Lane1,
            TaskPriority::Normal,
            default_model_requirements(),
            default_input_cid(),
            DEFAULT_COMPUTE_BUDGET,
            100,
            100
        ));
        let executor = accept_and_get_executor(0);

        let alice_balance_before = Balances::free_balance(ALICE);
        let executor_balance_before = Balances::free_balance(executor);

        // WHEN: Bob submits the task result
        assert_ok!(NsnTaskMarket::submit_result(
            RuntimeOrigin::signed(executor),
            0,
            default_output_cid(),
            None
        ));

        // THEN: Task is submitted, escrow still reserved
        let task = NsnTaskMarket::tasks(0).expect("Task should exist");
        assert_eq!(task.status, TaskStatus::Submitted);
        assert_eq!(task.output_cid, Some(default_output_cid()));

        // Escrow still reserved, no payout yet
        assert_eq!(Balances::reserved_balance(ALICE), 100);
        assert_eq!(Balances::free_balance(ALICE), alice_balance_before);
        assert_eq!(Balances::free_balance(executor), executor_balance_before);

        // Verify event
        let event = last_event();
        assert!(matches!(
            event,
            RuntimeEvent::NsnTaskMarket(Event::TaskSubmitted {
                task_id: 0,
                executor: e,
                output_cid: _,
                attestation_cid: None,
                ..
            }) if e == executor
        ));
    });
}

#[test]
fn task_verified_after_quorum_pays_executor() {
    ExtBuilder::default().build().execute_with(|| {
        assert_ok!(NsnTaskMarket::create_task_intent(
            RuntimeOrigin::signed(ALICE),
            TaskLane::Lane1,
            TaskPriority::Normal,
            default_model_requirements(),
            default_input_cid(),
            DEFAULT_COMPUTE_BUDGET,
            100,
            100
        ));
        let executor = accept_and_get_executor(0);

        let executor_balance_before = Balances::free_balance(executor);

        assert_ok!(NsnTaskMarket::submit_result(
            RuntimeOrigin::signed(executor),
            0,
            default_output_cid(),
            None
        ));
        assert_ok!(NsnTaskMarket::submit_attestation(
            RuntimeOrigin::signed(ALICE),
            0,
            80,
            Some(default_attestation_cid())
        ));
        assert_ok!(NsnTaskMarket::submit_attestation(
            RuntimeOrigin::signed(CHARLIE),
            0,
            85,
            Some(default_attestation_cid())
        ));

        let task = NsnTaskMarket::tasks(0).expect("Task should exist");
        assert_eq!(task.status, TaskStatus::Verified);
        assert_eq!(Balances::reserved_balance(ALICE), 0);
        assert_eq!(Balances::free_balance(executor), executor_balance_before + 70);
    });
}

#[test]
fn invalid_attestation_rejected() {
    ExtBuilder::default().build().execute_with(|| {
        assert_ok!(NsnTaskMarket::create_task_intent(
            RuntimeOrigin::signed(ALICE),
            TaskLane::Lane1,
            TaskPriority::Normal,
            default_model_requirements(),
            default_input_cid(),
            DEFAULT_COMPUTE_BUDGET,
            100,
            100
        ));
        let executor = accept_and_get_executor(0);
        assert_ok!(NsnTaskMarket::submit_result(
            RuntimeOrigin::signed(executor),
            0,
            default_output_cid(),
            None
        ));

        // Non-validator cannot attest
        assert_noop!(
            NsnTaskMarket::submit_attestation(
                RuntimeOrigin::signed(DAVE),
                0,
                80,
                Some(default_attestation_cid())
            ),
            Error::<Test>::NotValidator
        );

        // Duplicate attestation rejected
        assert_ok!(NsnTaskMarket::submit_attestation(
            RuntimeOrigin::signed(ALICE),
            0,
            80,
            Some(default_attestation_cid())
        ));
        assert_noop!(
            NsnTaskMarket::submit_attestation(
                RuntimeOrigin::signed(ALICE),
                0,
                80,
                Some(default_attestation_cid())
            ),
            Error::<Test>::DuplicateAttestation
        );
    });
}

#[test]
fn verification_deadline_rejects_task_and_refunds() {
    ExtBuilder::default().build().execute_with(|| {
        assert_ok!(NsnTaskMarket::create_task_intent(
            RuntimeOrigin::signed(ALICE),
            TaskLane::Lane1,
            TaskPriority::Normal,
            default_model_requirements(),
            default_input_cid(),
            DEFAULT_COMPUTE_BUDGET,
            100,
            100
        ));
        let executor = accept_and_get_executor(0);

        assert_ok!(NsnTaskMarket::submit_result(
            RuntimeOrigin::signed(executor),
            0,
            default_output_cid(),
            None
        ));

        let alice_balance_before = Balances::free_balance(ALICE);
        roll_to(System::block_number() + VerificationPeriod::get() + 1);

        assert_ok!(NsnTaskMarket::finalize_task(
            RuntimeOrigin::signed(CHARLIE),
            0
        ));

        let task = NsnTaskMarket::tasks(0).expect("Task should exist");
        assert_eq!(task.status, TaskStatus::Rejected);
        assert_eq!(Balances::reserved_balance(ALICE), 0);
        assert_eq!(Balances::free_balance(ALICE), alice_balance_before + 100);
    });
}

#[test]
fn fail_task_by_executor_succeeds() {
    ExtBuilder::default().build().execute_with(|| {
        // GIVEN: Alice created a task, Bob accepted it
        assert_ok!(NsnTaskMarket::create_task_intent(
            RuntimeOrigin::signed(ALICE),
            TaskLane::Lane1,
            TaskPriority::Normal,
            default_model_requirements(),
            default_input_cid(),
            DEFAULT_COMPUTE_BUDGET,
            100,
            100
        ));
        let executor = accept_and_get_executor(0);

        let alice_balance_before = Balances::free_balance(ALICE);

        // WHEN: Bob fails the task
        assert_ok!(NsnTaskMarket::fail_task(
            RuntimeOrigin::signed(executor),
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
            TaskLane::Lane1,
            TaskPriority::Normal,
            default_model_requirements(),
            default_input_cid(),
            DEFAULT_COMPUTE_BUDGET,
            100,
            100
        ));
        let _executor = accept_and_get_executor(0);

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
            TaskLane::Lane1,
            TaskPriority::Normal,
            default_model_requirements(),
            default_input_cid(),
            DEFAULT_COMPUTE_BUDGET,
            100,
            100
        ));
        assert_eq!(NsnTaskMarket::open_lane1_tasks().len(), 1);

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
        assert_eq!(NsnTaskMarket::open_lane1_tasks().len(), 0);
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
                NsnTaskMarket::create_task_intent(
                    RuntimeOrigin::signed(ALICE),
                    TaskLane::Lane1,
                    TaskPriority::Normal,
                    default_model_requirements(),
                    default_input_cid(),
                    DEFAULT_COMPUTE_BUDGET,
                    100,
                    100
                ),
                Error::<Test>::InsufficientBalance
            );
        });
}

#[test]
fn create_task_insufficient_escrow_fails() {
    ExtBuilder::default().build().execute_with(|| {
        // WHEN: Alice tries to create a task with escrow below minimum (MinEscrow = 10)
        assert_noop!(
            NsnTaskMarket::create_task_intent(
                RuntimeOrigin::signed(ALICE),
                TaskLane::Lane1,
                TaskPriority::Normal,
                default_model_requirements(),
                default_input_cid(),
                DEFAULT_COMPUTE_BUDGET,
                100,
                5 // Below MinEscrow of 10
            ),
            Error::<Test>::InsufficientEscrow
        );
    });
}

#[test]
fn create_task_invalid_deadline_fails() {
    ExtBuilder::default().build().execute_with(|| {
        // WHEN: Alice tries to create a task with zero deadline
        assert_noop!(
            NsnTaskMarket::create_task_intent(
                RuntimeOrigin::signed(ALICE),
                TaskLane::Lane1,
                TaskPriority::Normal,
                default_model_requirements(),
                default_input_cid(),
                DEFAULT_COMPUTE_BUDGET,
                0,
                100
            ),
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
                    TaskLane::Lane1,
                    TaskPriority::Normal,
                    default_model_requirements(),
                    default_input_cid(),
                    DEFAULT_COMPUTE_BUDGET,
                    100,
                    10
                ));
                assert_eq!(NsnTaskMarket::next_task_id(), i + 1);
            }

            // WHEN: Try to create one more
            assert_noop!(
                NsnTaskMarket::create_task_intent(
                    RuntimeOrigin::signed(ALICE),
                    TaskLane::Lane1,
                    TaskPriority::Normal,
                    default_model_requirements(),
                    default_input_cid(),
                    DEFAULT_COMPUTE_BUDGET,
                    100,
                    10
                ),
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
            TaskLane::Lane1,
            TaskPriority::Normal,
            default_model_requirements(),
            default_input_cid(),
            DEFAULT_COMPUTE_BUDGET,
            100,
            100
        ));
        let _executor = accept_and_get_executor(0);

        // WHEN: Charlie tries to accept the same task
        assert_noop!(
            NsnTaskMarket::accept_assignment(RuntimeOrigin::signed(CHARLIE), 0),
            Error::<Test>::TaskNotOpen
        );
    });
}

#[test]
fn submit_result_not_found_fails() {
    ExtBuilder::default().build().execute_with(|| {
        // WHEN: Bob tries to complete a non-existent task
        assert_noop!(
            NsnTaskMarket::submit_result(
                RuntimeOrigin::signed(BOB),
                999,
                default_output_cid(),
                None
            ),
            Error::<Test>::TaskNotFound
        );
    });
}

#[test]
fn submit_result_not_assigned_fails() {
    ExtBuilder::default().build().execute_with(|| {
        // GIVEN: Task is open (not assigned)
        assert_ok!(NsnTaskMarket::create_task_intent(
            RuntimeOrigin::signed(ALICE),
            TaskLane::Lane1,
            TaskPriority::Normal,
            default_model_requirements(),
            default_input_cid(),
            DEFAULT_COMPUTE_BUDGET,
            100,
            100
        ));

        // WHEN: Bob tries to complete an open task
        assert_noop!(
            NsnTaskMarket::submit_result(
                RuntimeOrigin::signed(BOB),
                0,
                default_output_cid(),
                None
            ),
            Error::<Test>::TaskNotAssigned
        );
    });
}

#[test]
fn submit_result_not_executor_fails() {
    ExtBuilder::default().build().execute_with(|| {
        // GIVEN: Task is assigned to Bob
        assert_ok!(NsnTaskMarket::create_task_intent(
            RuntimeOrigin::signed(ALICE),
            TaskLane::Lane1,
            TaskPriority::Normal,
            default_model_requirements(),
            default_input_cid(),
            DEFAULT_COMPUTE_BUDGET,
            100,
            100
        ));
        let _executor = accept_and_get_executor(0);

        // WHEN: Dave tries to submit someone else's task
        assert_noop!(
            NsnTaskMarket::submit_result(
                RuntimeOrigin::signed(DAVE),
                0,
                default_output_cid(),
                None
            ),
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
            TaskLane::Lane1,
            TaskPriority::Normal,
            default_model_requirements(),
            default_input_cid(),
            DEFAULT_COMPUTE_BUDGET,
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
            TaskLane::Lane1,
            TaskPriority::Normal,
            default_model_requirements(),
            default_input_cid(),
            DEFAULT_COMPUTE_BUDGET,
            100,
            100
        ));
        let _executor = accept_and_get_executor(0);

        // WHEN: Dave (neither executor nor requester) tries to fail
        assert_noop!(
            NsnTaskMarket::fail_task(
                RuntimeOrigin::signed(DAVE),
                0,
                FailReason::ExecutorFailed
            ),
            Error::<Test>::NotExecutor
        );
    });
}

#[test]
fn submit_result_twice_fails() {
    ExtBuilder::default().build().execute_with(|| {
        // GIVEN: Task is submitted
        assert_ok!(NsnTaskMarket::create_task_intent(
            RuntimeOrigin::signed(ALICE),
            TaskLane::Lane1,
            TaskPriority::Normal,
            default_model_requirements(),
            default_input_cid(),
            DEFAULT_COMPUTE_BUDGET,
            100,
            100
        ));
        let executor = accept_and_get_executor(0);
        assert_ok!(NsnTaskMarket::submit_result(
            RuntimeOrigin::signed(executor),
            0,
            default_output_cid(),
            None
        ));

        // WHEN: Bob tries to submit again
        assert_noop!(
            NsnTaskMarket::submit_result(
                RuntimeOrigin::signed(executor),
                0,
                default_output_cid(),
                None
            ),
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
            TaskLane::Lane1,
            TaskPriority::Normal,
            default_model_requirements(),
            default_input_cid(),
            DEFAULT_COMPUTE_BUDGET,
            100,
            100
        ));
        let executor = accept_and_get_executor(0);
        assert_ok!(NsnTaskMarket::fail_task(
            RuntimeOrigin::signed(executor),
            0,
            FailReason::ExecutorFailed
        ));

        // WHEN: Bob tries to fail again
        assert_noop!(
            NsnTaskMarket::fail_task(
                RuntimeOrigin::signed(executor),
                0,
                FailReason::ExecutorFailed
            ),
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
                TaskLane::Lane1,
                TaskPriority::Normal,
                default_model_requirements(),
                default_input_cid(),
                DEFAULT_COMPUTE_BUDGET,
                100,
                100 + i as u128
            ));
        }

        // THEN: All tasks exist with correct IDs and escrows
        for i in 0..5 {
            let task = NsnTaskMarket::tasks(i).expect("Task should exist");
            assert_eq!(task.id, i);
            assert_eq!(task.escrow, 100 + i as u128);
        }

        assert_eq!(NsnTaskMarket::next_task_id(), 5);
        assert_eq!(NsnTaskMarket::open_lane1_tasks().len(), 5);
        assert_eq!(
            Balances::reserved_balance(ALICE),
            100 + 101 + 102 + 103 + 104
        );
    });
}

#[test]
fn open_lane1_tasks_queue_maintained_correctly() {
    ExtBuilder::default().build().execute_with(|| {
        // Create 3 tasks
        for _ in 0..3 {
            assert_ok!(NsnTaskMarket::create_task_intent(
                RuntimeOrigin::signed(ALICE),
                TaskLane::Lane1,
                TaskPriority::Normal,
                default_model_requirements(),
                default_input_cid(),
                DEFAULT_COMPUTE_BUDGET,
                100,
                100
            ));
        }
        assert_eq!(NsnTaskMarket::open_lane1_tasks().len(), 3);
        assert!(NsnTaskMarket::open_lane1_tasks().contains(&0));
        assert!(NsnTaskMarket::open_lane1_tasks().contains(&1));
        assert!(NsnTaskMarket::open_lane1_tasks().contains(&2));

        // Accept task 1 (middle one)
        assert_ok!(NsnTaskMarket::accept_assignment(
            RuntimeOrigin::signed(BOB),
            1
        ));
        assert_eq!(NsnTaskMarket::open_lane1_tasks().len(), 2);
        assert!(NsnTaskMarket::open_lane1_tasks().contains(&0));
        assert!(!NsnTaskMarket::open_lane1_tasks().contains(&1));
        assert!(NsnTaskMarket::open_lane1_tasks().contains(&2));

        // Cancel task 0
        assert_ok!(NsnTaskMarket::fail_task(
            RuntimeOrigin::signed(ALICE),
            0,
            FailReason::Cancelled
        ));
        assert_eq!(NsnTaskMarket::open_lane1_tasks().len(), 1);
        assert!(!NsnTaskMarket::open_lane1_tasks().contains(&0));
        assert!(NsnTaskMarket::open_lane1_tasks().contains(&2));
    });
}

#[test]
fn task_minimum_escrow() {
    ExtBuilder::default().build().execute_with(|| {
        // WHEN: Create task with exactly minimum escrow (MinEscrow = 10)
        assert_ok!(NsnTaskMarket::create_task_intent(
            RuntimeOrigin::signed(ALICE),
            TaskLane::Lane1,
            TaskPriority::Normal,
            default_model_requirements(),
            default_input_cid(),
            DEFAULT_COMPUTE_BUDGET,
            100,
            10 // Exactly MinEscrow
        ));

        // THEN: Task created with minimum escrow
        let task = NsnTaskMarket::tasks(0).expect("Task should exist");
        assert_eq!(task.escrow, 10);
        assert_eq!(Balances::reserved_balance(ALICE), 10);
    });
}

#[test]
fn verify_task_with_minimum_escrow() {
    ExtBuilder::default().build().execute_with(|| {
        // GIVEN: Task with minimum escrow
        assert_ok!(NsnTaskMarket::create_task_intent(
            RuntimeOrigin::signed(ALICE),
            TaskLane::Lane1,
            TaskPriority::Normal,
            default_model_requirements(),
            default_input_cid(),
            DEFAULT_COMPUTE_BUDGET,
            100,
            10
        ));
        let executor = accept_and_get_executor(0);

        let executor_balance_before = Balances::free_balance(executor);

        // WHEN: Submit result and reach verification quorum
        assert_ok!(NsnTaskMarket::submit_result(
            RuntimeOrigin::signed(executor),
            0,
            default_output_cid(),
            None
        ));
        assert_ok!(NsnTaskMarket::submit_attestation(
            RuntimeOrigin::signed(ALICE),
            0,
            90,
            Some(default_attestation_cid())
        ));
        assert_ok!(NsnTaskMarket::submit_attestation(
            RuntimeOrigin::signed(CHARLIE),
            0,
            90,
            Some(default_attestation_cid())
        ));

        // THEN: Balance change reflects minimum escrow payout
        assert_eq!(Balances::free_balance(executor), executor_balance_before + 7);
        assert_eq!(Balances::free_balance(TREASURY_ACCOUNT), 10_002);
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
            TaskLane::Lane1,
            TaskPriority::Normal,
            default_model_requirements(),
            default_input_cid(),
            DEFAULT_COMPUTE_BUDGET,
            50,
            100
        ));

        // THEN: Deadline is current_block + deadline_blocks
        let task = NsnTaskMarket::tasks(0).expect("Task should exist");
        assert_eq!(task.created_at, 1);
        assert_eq!(task.deadline, 51);

        // Advance to block 10, create another task
        roll_to(10);
        assert_ok!(NsnTaskMarket::create_task_intent(
            RuntimeOrigin::signed(ALICE),
            TaskLane::Lane1,
            TaskPriority::Normal,
            default_model_requirements(),
            default_input_cid(),
            DEFAULT_COMPUTE_BUDGET,
            50,
            100
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
            TaskLane::Lane1,
            TaskPriority::Normal,
            default_model_requirements(),
            default_input_cid(),
            DEFAULT_COMPUTE_BUDGET,
            100,
            100
        ));

        // After creation
        assert_eq!(NsnTaskMarket::open_task_count(), 1);
        assert!(NsnTaskMarket::task_exists(0));
        assert!(!NsnTaskMarket::task_exists(1));

        // After assignment
        let _executor = accept_and_get_executor(0);
        assert_eq!(NsnTaskMarket::open_task_count(), 0);
        assert!(NsnTaskMarket::task_exists(0)); // Task still exists, just not open
    });
}

#[test]
fn self_assignment_works() {
    ExtBuilder::default().build().execute_with(|| {
        // GIVEN: Alice creates a Lane 0 task (only Alice eligible)
        let lane0_renderer: BoundedVec<u8, MaxModelIdLen> =
            BoundedVec::try_from(b"lane0-self".to_vec()).unwrap();
        assert_ok!(NsnTaskMarket::register_renderer(
            RuntimeOrigin::root(),
            lane0_renderer.clone(),
            TaskLane::Lane0,
            true,
            10_000,
            6_000
        ));
        assert_ok!(NsnTaskMarket::create_task_intent(
            RuntimeOrigin::signed(ALICE),
            TaskLane::Lane0,
            TaskPriority::Normal,
            lane0_renderer,
            default_input_cid(),
            DEFAULT_COMPUTE_BUDGET,
            100,
            100
        ));

        // WHEN: Assignment is accepted
        let executor = accept_and_get_executor(0);

        // THEN: Alice is the executor
        let task = NsnTaskMarket::tasks(0).expect("Task should exist");
        assert_eq!(task.executor, Some(ALICE));
        assert_eq!(task.status, TaskStatus::Assigned);
        assert_eq!(executor, ALICE);
    });
}

#[test]
fn self_completion_works() {
    ExtBuilder::default().build().execute_with(|| {
        // GIVEN: Alice creates and accepts her own Lane 0 task
        let lane0_renderer: BoundedVec<u8, MaxModelIdLen> =
            BoundedVec::try_from(b"lane0-self-complete".to_vec()).unwrap();
        assert_ok!(NsnTaskMarket::register_renderer(
            RuntimeOrigin::root(),
            lane0_renderer.clone(),
            TaskLane::Lane0,
            true,
            10_000,
            6_000
        ));
        assert_ok!(NsnTaskMarket::create_task_intent(
            RuntimeOrigin::signed(ALICE),
            TaskLane::Lane0,
            TaskPriority::Normal,
            lane0_renderer,
            default_input_cid(),
            DEFAULT_COMPUTE_BUDGET,
            100,
            100
        ));
        let executor = accept_and_get_executor(0);

        let alice_balance_before = Balances::free_balance(ALICE);
        let alice_reserved_before = Balances::reserved_balance(ALICE);

        // WHEN: Alice submits her own task and it is verified
        assert_ok!(NsnTaskMarket::submit_result(
            RuntimeOrigin::signed(executor),
            0,
            default_output_cid(),
            Some(default_attestation_cid())
        ));
        assert_ok!(NsnTaskMarket::submit_attestation(
            RuntimeOrigin::signed(BOB),
            0,
            85,
            Some(default_attestation_cid())
        ));
        assert_ok!(NsnTaskMarket::submit_attestation(
            RuntimeOrigin::signed(CHARLIE),
            0,
            85,
            Some(default_attestation_cid())
        ));

        // THEN: Escrow is unreserved and "transferred" to self (net zero change)
        assert_eq!(Balances::reserved_balance(ALICE), 0);
        assert_eq!(
            Balances::free_balance(ALICE),
            alice_balance_before + alice_reserved_before - 30
        );
        assert_eq!(Balances::free_balance(TREASURY_ACCOUNT), 10_020);
    });
}

#[test]
fn fail_reasons_are_recorded() {
    ExtBuilder::default().build().execute_with(|| {
        let reasons = vec![
            FailReason::ExecutorFailed,
            FailReason::Cancelled,
            FailReason::DeadlineExceeded,
            FailReason::Preempted,
            FailReason::InvalidInput,
            FailReason::Other,
        ];

        for (i, reason) in reasons.into_iter().enumerate() {
            // Create and assign task
            assert_ok!(NsnTaskMarket::create_task_intent(
                RuntimeOrigin::signed(ALICE),
                TaskLane::Lane1,
                TaskPriority::Normal,
                default_model_requirements(),
                default_input_cid(),
                DEFAULT_COMPUTE_BUDGET,
                100,
                100
            ));
            let executor = accept_and_get_executor(i as u64);

            // Fail with specific reason
            assert_ok!(NsnTaskMarket::fail_task(
                RuntimeOrigin::signed(executor),
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

#[test]
fn task_model_requirements_and_input_cid_stored() {
    ExtBuilder::default().build().execute_with(|| {
        let model_requirements: BoundedVec<u8, MaxModelIdLen> =
            BoundedVec::try_from(b"custom-model-v2".to_vec()).unwrap();
        let input_cid: BoundedVec<u8, MaxCidLen> =
            BoundedVec::try_from(b"QmCustomInputCid".to_vec()).unwrap();
        let compute_budget = 5000u64;

        // WHEN: Create task with custom model requirements, input_cid, compute_budget
        assert_ok!(NsnTaskMarket::create_task_intent(
            RuntimeOrigin::signed(ALICE),
            TaskLane::Lane1,
            TaskPriority::Normal,
            model_requirements.clone(),
            input_cid.clone(),
            compute_budget,
            100,
            100
        ));

        // THEN: Task stores the custom values
        let task = NsnTaskMarket::tasks(0).expect("Task should exist");
        assert_eq!(task.model_requirements, model_requirements);
        assert_eq!(task.input_cid, input_cid);
        assert_eq!(task.compute_budget, compute_budget);
    });
}

#[test]
fn task_output_cid_stored_on_submission() {
    ExtBuilder::default().build().execute_with(|| {
        let output_cid: BoundedVec<u8, MaxCidLen> =
            BoundedVec::try_from(b"QmCustomOutputCid".to_vec()).unwrap();

        // GIVEN: Task created and assigned
        assert_ok!(NsnTaskMarket::create_task_intent(
            RuntimeOrigin::signed(ALICE),
            TaskLane::Lane1,
            TaskPriority::Normal,
            default_model_requirements(),
            default_input_cid(),
            DEFAULT_COMPUTE_BUDGET,
            100,
            100
        ));
        let executor = accept_and_get_executor(0);

        // Verify output_cid is None before completion
        let task_before = NsnTaskMarket::tasks(0).expect("Task should exist");
        assert_eq!(task_before.output_cid, None);

        // WHEN: Submit with custom output_cid
        assert_ok!(NsnTaskMarket::submit_result(
            RuntimeOrigin::signed(executor),
            0,
            output_cid.clone(),
            None
        ));

        // THEN: output_cid is stored
        let task_after = NsnTaskMarket::tasks(0).expect("Task should exist");
        assert_eq!(task_after.output_cid, Some(output_cid));
    });
}

// ============================================================================
// Integration-Style Tests
// ============================================================================

#[test]
fn full_task_lifecycle_success() {
    ExtBuilder::default().build().execute_with(|| {
        let initial_alice_balance = Balances::free_balance(ALICE);

        // Step 1: Create task
        assert_ok!(NsnTaskMarket::create_task_intent(
            RuntimeOrigin::signed(ALICE),
            TaskLane::Lane1,
            TaskPriority::Normal,
            default_model_requirements(),
            default_input_cid(),
            DEFAULT_COMPUTE_BUDGET,
            100,
            500
        ));
        assert_eq!(Balances::free_balance(ALICE), initial_alice_balance - 500);
        assert_eq!(Balances::reserved_balance(ALICE), 500);

        // Step 2: Accept assignment
        let executor = accept_and_get_executor(0);
        let initial_executor_balance = Balances::free_balance(executor);

        // Step 3: Submit result
        assert_ok!(NsnTaskMarket::submit_result(
            RuntimeOrigin::signed(executor),
            0,
            default_output_cid(),
            None
        ));

        // Step 4: Submit attestations (quorum reached)
        assert_ok!(NsnTaskMarket::submit_attestation(
            RuntimeOrigin::signed(ALICE),
            0,
            90,
            Some(default_attestation_cid())
        ));
        assert_ok!(NsnTaskMarket::submit_attestation(
            RuntimeOrigin::signed(CHARLIE),
            0,
            90,
            Some(default_attestation_cid())
        ));

        // Final state verification
        assert_eq!(Balances::reserved_balance(ALICE), 0);
        assert_eq!(Balances::free_balance(ALICE), initial_alice_balance - 500);
        assert_eq!(
            Balances::free_balance(executor),
            initial_executor_balance + 350
        );
        assert_eq!(Balances::free_balance(TREASURY_ACCOUNT), 10_100);

        let task = NsnTaskMarket::tasks(0).expect("Task should exist");
        assert_eq!(task.status, TaskStatus::Verified);
        assert_eq!(task.output_cid, Some(default_output_cid()));
    });
}

#[test]
fn full_task_lifecycle_failure() {
    ExtBuilder::default().build().execute_with(|| {
        let initial_alice_balance = Balances::free_balance(ALICE);

        // Step 1: Create task
        assert_ok!(NsnTaskMarket::create_task_intent(
            RuntimeOrigin::signed(ALICE),
            TaskLane::Lane1,
            TaskPriority::Normal,
            default_model_requirements(),
            default_input_cid(),
            DEFAULT_COMPUTE_BUDGET,
            100,
            500
        ));

        // Step 2: Accept assignment
        let executor = accept_and_get_executor(0);
        let initial_executor_balance = Balances::free_balance(executor);

        // Step 3: Fail task
        assert_ok!(NsnTaskMarket::fail_task(
            RuntimeOrigin::signed(executor),
            0,
            FailReason::ExecutorFailed
        ));

        // Final state verification: escrow returned to Alice
        assert_eq!(Balances::reserved_balance(ALICE), 0);
        assert_eq!(Balances::free_balance(ALICE), initial_alice_balance);
        assert_eq!(Balances::free_balance(executor), initial_executor_balance);

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
                TaskLane::Lane1,
                TaskPriority::Normal,
                default_model_requirements(),
                default_input_cid(),
                DEFAULT_COMPUTE_BUDGET,
                100,
                100
            ));
        }

        // Accept assignments
        let exec0 = accept_and_get_executor(0);
        let exec1 = accept_and_get_executor(1);

        // Bob submits and verifies task 0
        assert_ok!(NsnTaskMarket::submit_result(
            RuntimeOrigin::signed(exec0),
            0,
            default_output_cid(),
            None
        ));
        assert_ok!(NsnTaskMarket::submit_attestation(
            RuntimeOrigin::signed(ALICE),
            0,
            90,
            Some(default_attestation_cid())
        ));
        assert_ok!(NsnTaskMarket::submit_attestation(
            RuntimeOrigin::signed(CHARLIE),
            0,
            90,
            Some(default_attestation_cid())
        ));
        // Charlie fails task 1
        assert_ok!(NsnTaskMarket::fail_task(
            RuntimeOrigin::signed(exec1),
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
            TaskStatus::Verified
        );
        assert_eq!(NsnTaskMarket::tasks(1).unwrap().status, TaskStatus::Failed);
        assert_eq!(NsnTaskMarket::tasks(2).unwrap().status, TaskStatus::Failed);

        // Verify open queue is empty
        assert_eq!(NsnTaskMarket::open_lane1_tasks().len(), 0);
    });
}
