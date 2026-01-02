//! Redundant execution and consensus tracking for NSN tasks.
//!
//! This module coordinates multi-executor assignment, collects results,
//! computes consensus, and submits attestations once quorum is reached.

use crate::task_queue::{Lane, Task, TaskId};
use nsn_types::{FailureReason, NodeCapability};
use serde::Serialize;
use std::collections::{HashMap, HashSet};
use std::time::{Duration, Instant};

/// Unique identifier for an executor node.
pub type ExecutorId = String;

/// Configuration defaults for redundancy and consensus.
#[derive(Debug, Clone)]
pub struct RedundancyConfig {
    pub replicas: usize,
    pub quorum: usize,
    pub semantic_min_score: f32,
    pub semantic_epsilon: f32,
    pub consensus_timeout: Duration,
}

impl Default for RedundancyConfig {
    fn default() -> Self {
        Self {
            replicas: 3,
            quorum: 2,
            semantic_min_score: 0.75,
            semantic_epsilon: 0.05,
            consensus_timeout: Duration::from_secs(120),
        }
    }
}

impl RedundancyConfig {
    pub fn policy_for(&self, deterministic: bool) -> ConsensusPolicy {
        if deterministic {
            ConsensusPolicy::deterministic(self.replicas, self.quorum)
        } else {
            ConsensusPolicy::semantic(
                self.replicas,
                self.quorum,
                self.semantic_min_score,
                self.semantic_epsilon,
            )
        }
    }
}

/// Consensus mode selection.
#[derive(Debug, Clone)]
pub enum ConsensusMode {
    Deterministic,
    Semantic { min_score: f32, epsilon: f32 },
}

/// Per-task consensus policy.
#[derive(Debug, Clone)]
pub struct ConsensusPolicy {
    pub replicas: usize,
    pub quorum: usize,
    pub mode: ConsensusMode,
}

impl ConsensusPolicy {
    pub fn deterministic(replicas: usize, quorum: usize) -> Self {
        Self {
            replicas,
            quorum,
            mode: ConsensusMode::Deterministic,
        }
    }

    pub fn semantic(replicas: usize, quorum: usize, min_score: f32, epsilon: f32) -> Self {
        Self {
            replicas,
            quorum,
            mode: ConsensusMode::Semantic { min_score, epsilon },
        }
    }

    pub fn validate(&self) -> Result<(), RedundancyError> {
        if self.replicas == 0 {
            return Err(RedundancyError::InvalidPolicy(
                "replicas must be >= 1".to_string(),
            ));
        }
        if self.quorum == 0 || self.quorum > self.replicas {
            return Err(RedundancyError::InvalidPolicy(
                "quorum must be >= 1 and <= replicas".to_string(),
            ));
        }
        if let ConsensusMode::Semantic { min_score, epsilon } = self.mode {
            if !(0.0..=1.0).contains(&min_score) {
                return Err(RedundancyError::InvalidPolicy(
                    "semantic min_score must be in [0, 1]".to_string(),
                ));
            }
            if epsilon < 0.0 {
                return Err(RedundancyError::InvalidPolicy(
                    "semantic epsilon must be >= 0".to_string(),
                ));
            }
        }
        Ok(())
    }
}

/// Assignment returned to the scheduler/executor layer.
#[derive(Debug, Clone)]
pub struct RedundantAssignment {
    pub task_id: TaskId,
    pub executors: Vec<ExecutorId>,
    pub policy: ConsensusPolicy,
}

/// Executor registry entry used for selection.
#[derive(Debug, Clone)]
pub struct ExecutorInfo {
    pub id: ExecutorId,
    pub capability: NodeCapability,
    pub reputation: u64,
}

/// Registry interface for executor selection.
pub trait ExecutorRegistry {
    fn eligible_executors(&self, lane: Lane) -> Vec<ExecutorInfo>;
}

/// Static registry for tests or bootstrap environments.
#[derive(Debug, Clone)]
pub struct StaticExecutorRegistry {
    executors: Vec<ExecutorInfo>,
}

impl StaticExecutorRegistry {
    pub fn new(executors: Vec<ExecutorInfo>) -> Self {
        Self { executors }
    }
}

impl ExecutorRegistry for StaticExecutorRegistry {
    fn eligible_executors(&self, _lane: Lane) -> Vec<ExecutorInfo> {
        self.executors.clone()
    }
}

/// Execution result from a single executor.
#[derive(Debug, Clone)]
pub struct ExecutionResult {
    pub output_cid: String,
    pub output_hash: String,
    pub semantic_score: Option<f32>,
    pub duration_ms: u64,
    pub received_at: Instant,
}

impl ExecutionResult {
    pub fn new(output_cid: String, semantic_score: Option<f32>, duration_ms: u64) -> Self {
        let output_hash = hash_output_cid(&output_cid);
        Self {
            output_cid,
            output_hash,
            semantic_score,
            duration_ms,
            received_at: Instant::now(),
        }
    }
}

/// Consensus success record.
#[derive(Debug, Clone)]
pub struct ConsensusRecord {
    pub output_cid: String,
    pub output_hash: String,
    pub score: u8,
    pub semantic_score: Option<f32>,
    pub participants: Vec<ExecutorId>,
}

/// Consensus failure reasons.
#[derive(Debug, Clone)]
pub enum ConsensusFailureReason {
    InsufficientQuorum,
    HashMismatch,
    SemanticMismatch,
    Timeout,
    AttestationSubmitFailed(String),
}

/// Consensus evaluation outcome.
#[derive(Debug, Clone)]
pub enum ConsensusOutcome {
    Pending,
    Achieved(ConsensusRecord),
    Rejected(ConsensusFailureReason),
}

/// Attestation bundle ready for chain submission.
#[derive(Debug, Clone)]
pub struct AttestationBundle {
    pub task_id: TaskId,
    pub output_cid: String,
    pub score: u8,
    pub semantic_score: Option<f32>,
    pub attestation_cid: Option<String>,
    pub executors: Vec<ExecutorId>,
}

/// Attestation submission interface.
pub trait AttestationSubmitter {
    fn submit_attestation(
        &mut self,
        attestation: AttestationBundle,
    ) -> Result<(), AttestationError>;
}

impl<T: AttestationSubmitter + ?Sized> AttestationSubmitter for &mut T {
    fn submit_attestation(
        &mut self,
        attestation: AttestationBundle,
    ) -> Result<(), AttestationError> {
        (**self).submit_attestation(attestation)
    }
}

impl AttestationSubmitter for Box<dyn AttestationSubmitter> {
    fn submit_attestation(
        &mut self,
        attestation: AttestationBundle,
    ) -> Result<(), AttestationError> {
        (**self).submit_attestation(attestation)
    }
}

/// No-op submitter for environments without chain connectivity.
#[derive(Debug, Default)]
pub struct NoopAttestationSubmitter;

impl AttestationSubmitter for NoopAttestationSubmitter {
    fn submit_attestation(
        &mut self,
        _attestation: AttestationBundle,
    ) -> Result<(), AttestationError> {
        Ok(())
    }
}

/// Publish attestations over the P2P GossipSub topic.
#[derive(Debug, Clone)]
pub struct P2pAttestationSubmitter {
    command_tx: tokio::sync::mpsc::UnboundedSender<nsn_p2p::ServiceCommand>,
}

impl P2pAttestationSubmitter {
    pub fn new(command_tx: tokio::sync::mpsc::UnboundedSender<nsn_p2p::ServiceCommand>) -> Self {
        Self { command_tx }
    }
}

#[derive(Debug, Serialize)]
struct AttestationMessage {
    task_id: u64,
    output_cid: String,
    score: u8,
    semantic_score: Option<f32>,
    attestation_cid: Option<String>,
    executors: Vec<ExecutorId>,
}

impl AttestationSubmitter for P2pAttestationSubmitter {
    fn submit_attestation(
        &mut self,
        attestation: AttestationBundle,
    ) -> Result<(), AttestationError> {
        let message = AttestationMessage {
            task_id: attestation.task_id.0,
            output_cid: attestation.output_cid,
            score: attestation.score,
            semantic_score: attestation.semantic_score,
            attestation_cid: attestation.attestation_cid,
            executors: attestation.executors,
        };

        let payload = serde_json::to_vec(&message)
            .map_err(|err| AttestationError::SubmissionFailed(err.to_string()))?;
        let (tx, rx) = tokio::sync::oneshot::channel();
        self.command_tx
            .send(nsn_p2p::ServiceCommand::Publish(
                nsn_p2p::TopicCategory::Attestations,
                payload,
                tx,
            ))
            .map_err(|err| AttestationError::SubmissionFailed(err.to_string()))?;

        match rx.blocking_recv() {
            Ok(Ok(_)) => Ok(()),
            Ok(Err(err)) => Err(AttestationError::SubmissionFailed(err.to_string())),
            Err(err) => Err(AttestationError::SubmissionFailed(err.to_string())),
        }
    }
}

/// Submit attestations through two submitters, succeeding if either succeeds.
#[derive(Debug)]
pub struct DualAttestationSubmitter<A, B> {
    primary: A,
    secondary: B,
}

impl<A, B> DualAttestationSubmitter<A, B> {
    pub fn new(primary: A, secondary: B) -> Self {
        Self { primary, secondary }
    }
}

impl<A, B> AttestationSubmitter for DualAttestationSubmitter<A, B>
where
    A: AttestationSubmitter,
    B: AttestationSubmitter,
{
    fn submit_attestation(
        &mut self,
        attestation: AttestationBundle,
    ) -> Result<(), AttestationError> {
        let primary_result = self.primary.submit_attestation(attestation.clone());
        let secondary_result = self.secondary.submit_attestation(attestation);

        match (primary_result, secondary_result) {
            (Ok(_), _) | (_, Ok(_)) => Ok(()),
            (Err(err), Err(_)) => Err(err),
        }
    }
}

/// Aggregated redundancy metrics.
#[derive(Debug, Default, Clone)]
pub struct RedundancyMetrics {
    pub tasks_started: u64,
    pub tasks_resolved: u64,
    pub consensus_successes: u64,
    pub consensus_failures: u64,
    pub consensus_timeouts: u64,
    pub total_latency_ms: u128,
}

impl RedundancyMetrics {
    pub fn success_rate(&self) -> f32 {
        if self.tasks_resolved == 0 {
            return 0.0;
        }
        self.consensus_successes as f32 / self.tasks_resolved as f32
    }

    pub fn average_latency_ms(&self) -> Option<u128> {
        if self.tasks_resolved == 0 {
            return None;
        }
        Some(self.total_latency_ms / self.tasks_resolved as u128)
    }
}

/// Task tracking state for redundant execution.
#[derive(Debug, Clone)]
pub struct RedundantTask {
    pub task: Task,
    pub policy: ConsensusPolicy,
    pub assigned: Vec<ExecutorId>,
    pub results: HashMap<ExecutorId, ExecutionResult>,
    pub failures: HashSet<ExecutorId>,
    pub status: RedundantTaskStatus,
    pub created_at: Instant,
    pub resolved_at: Option<Instant>,
}

impl RedundantTask {
    pub fn new(task: Task, policy: ConsensusPolicy, assigned: Vec<ExecutorId>) -> Self {
        Self {
            task,
            policy,
            assigned,
            results: HashMap::new(),
            failures: HashSet::new(),
            status: RedundantTaskStatus::Assigned,
            created_at: Instant::now(),
            resolved_at: None,
        }
    }

    pub fn is_resolved(&self) -> bool {
        matches!(
            self.status,
            RedundantTaskStatus::ConsensusReached(_)
                | RedundantTaskStatus::Rejected(_)
                | RedundantTaskStatus::TimedOut
        )
    }
}

/// High-level task status for redundancy.
#[derive(Debug, Clone)]
pub enum RedundantTaskStatus {
    Assigned,
    InProgress,
    ConsensusReached(ConsensusRecord),
    Rejected(ConsensusFailureReason),
    TimedOut,
}

/// Errors for redundancy processing.
#[derive(Debug, thiserror::Error)]
pub enum RedundancyError {
    #[error("task not found: {0}")]
    TaskNotFound(TaskId),
    #[error("task already tracked: {0}")]
    TaskAlreadyTracked(TaskId),
    #[error("executor not assigned to task")]
    ExecutorNotAssigned,
    #[error("result already recorded for executor")]
    DuplicateResult,
    #[error("policy invalid: {0}")]
    InvalidPolicy(String),
    #[error("not enough executors for assignment")]
    NotEnoughExecutors,
    #[error("consensus already resolved")]
    AlreadyResolved,
    #[error("semantic score missing or invalid")]
    InvalidSemanticScore,
    #[error("attestation submission failed: {0}")]
    AttestationFailed(String),
}

/// Attestation submission errors.
#[derive(Debug, thiserror::Error)]
pub enum AttestationError {
    #[error("submission failed: {0}")]
    SubmissionFailed(String),
}

/// Redundant execution coordinator.
#[derive(Debug)]
pub struct RedundantScheduler<S: AttestationSubmitter> {
    config: RedundancyConfig,
    submitter: S,
    tasks: HashMap<TaskId, RedundantTask>,
    metrics: RedundancyMetrics,
}

impl<S: AttestationSubmitter> RedundantScheduler<S> {
    pub fn new(config: RedundancyConfig, submitter: S) -> Self {
        Self {
            config,
            submitter,
            tasks: HashMap::new(),
            metrics: RedundancyMetrics::default(),
        }
    }

    pub fn into_submitter(self) -> S {
        self.submitter
    }

    pub fn metrics(&self) -> &RedundancyMetrics {
        &self.metrics
    }

    pub fn task_status(&self, task_id: TaskId) -> Option<&RedundantTaskStatus> {
        self.tasks.get(&task_id).map(|task| &task.status)
    }

    pub fn assign_task(
        &mut self,
        task: Task,
        deterministic: bool,
        registry: &dyn ExecutorRegistry,
    ) -> Result<RedundantAssignment, RedundancyError> {
        let task_id = task.id;
        if self.tasks.contains_key(&task_id) {
            return Err(RedundancyError::TaskAlreadyTracked(task_id));
        }

        let policy = self.config.policy_for(deterministic);
        policy.validate()?;

        let executors = self.select_executors(task_id, task.lane, registry, policy.replicas)?;
        if executors.len() < policy.replicas {
            return Err(RedundancyError::NotEnoughExecutors);
        }

        let redundant_task = RedundantTask::new(task, policy.clone(), executors.clone());
        self.tasks.insert(task_id, redundant_task);
        self.metrics.tasks_started += 1;

        Ok(RedundantAssignment {
            task_id,
            executors,
            policy,
        })
    }

    pub fn record_success(
        &mut self,
        task_id: TaskId,
        executor: ExecutorId,
        output_cid: String,
        semantic_score: Option<f32>,
        duration_ms: u64,
    ) -> Result<ConsensusOutcome, RedundancyError> {
        {
            let task = self
                .tasks
                .get_mut(&task_id)
                .ok_or(RedundancyError::TaskNotFound(task_id))?;

            if task.is_resolved() {
                return Err(RedundancyError::AlreadyResolved);
            }

            if !task.assigned.contains(&executor) {
                return Err(RedundancyError::ExecutorNotAssigned);
            }

            if task.results.contains_key(&executor) || task.failures.contains(&executor) {
                return Err(RedundancyError::DuplicateResult);
            }

            if matches!(task.policy.mode, ConsensusMode::Semantic { .. }) {
                let score = semantic_score.ok_or(RedundancyError::InvalidSemanticScore)?;
                if !(0.0..=1.0).contains(&score) {
                    return Err(RedundancyError::InvalidSemanticScore);
                }
            }

            task.results.insert(
                executor,
                ExecutionResult::new(output_cid, semantic_score, duration_ms),
            );
            task.status = RedundantTaskStatus::InProgress;
        }

        let outcome = {
            let task = self
                .tasks
                .get(&task_id)
                .ok_or(RedundancyError::TaskNotFound(task_id))?;
            Self::evaluate_consensus(task)
        };

        self.finalize_outcome(task_id, outcome.clone())?;
        Ok(outcome)
    }

    pub fn record_failure(
        &mut self,
        task_id: TaskId,
        executor: ExecutorId,
        _reason: FailureReason,
    ) -> Result<ConsensusOutcome, RedundancyError> {
        {
            let task = self
                .tasks
                .get_mut(&task_id)
                .ok_or(RedundancyError::TaskNotFound(task_id))?;

            if task.is_resolved() {
                return Err(RedundancyError::AlreadyResolved);
            }

            if !task.assigned.contains(&executor) {
                return Err(RedundancyError::ExecutorNotAssigned);
            }

            if task.results.contains_key(&executor) || task.failures.contains(&executor) {
                return Err(RedundancyError::DuplicateResult);
            }

            task.failures.insert(executor);
            task.status = RedundantTaskStatus::InProgress;
        }

        let outcome = {
            let task = self
                .tasks
                .get(&task_id)
                .ok_or(RedundancyError::TaskNotFound(task_id))?;
            Self::evaluate_consensus(task)
        };

        self.finalize_outcome(task_id, outcome.clone())?;
        Ok(outcome)
    }

    pub fn expire_timeouts(&mut self, now: Instant) -> Vec<TaskId> {
        let mut expired = Vec::new();
        for (task_id, task) in self.tasks.iter_mut() {
            if task.is_resolved() {
                continue;
            }
            if now.duration_since(task.created_at) > self.config.consensus_timeout {
                task.status = RedundantTaskStatus::TimedOut;
                task.resolved_at = Some(now);
                self.metrics.tasks_resolved += 1;
                self.metrics.consensus_timeouts += 1;
                self.metrics.consensus_failures += 1;
                self.metrics.total_latency_ms += now.duration_since(task.created_at).as_millis();
                expired.push(*task_id);
            }
        }
        expired
    }

    fn select_executors(
        &mut self,
        task_id: TaskId,
        lane: Lane,
        registry: &dyn ExecutorRegistry,
        replicas: usize,
    ) -> Result<Vec<ExecutorId>, RedundancyError> {
        let mut candidates = registry
            .eligible_executors(lane)
            .into_iter()
            .filter(|executor| is_executor_capable(executor))
            .collect::<Vec<_>>();

        if candidates.is_empty() {
            return Err(RedundancyError::NotEnoughExecutors);
        }

        candidates.sort_by(|a, b| {
            b.reputation
                .cmp(&a.reputation)
                .then_with(|| a.id.cmp(&b.id))
        });

        let start = (task_id.0 as usize) % candidates.len();
        let mut selected = Vec::with_capacity(replicas);
        for offset in 0..candidates.len() {
            if selected.len() >= replicas {
                break;
            }
            let idx = (start + offset) % candidates.len();
            selected.push(candidates[idx].id.clone());
        }

        Ok(selected)
    }

    fn evaluate_consensus(task: &RedundantTask) -> ConsensusOutcome {
        let potential = task.assigned.len().saturating_sub(task.failures.len());

        if potential < task.policy.quorum {
            return ConsensusOutcome::Rejected(ConsensusFailureReason::InsufficientQuorum);
        }

        match task.policy.mode {
            ConsensusMode::Deterministic => Self::evaluate_deterministic(task, potential),
            ConsensusMode::Semantic { min_score, epsilon } => {
                Self::evaluate_semantic(task, potential, min_score, epsilon)
            }
        }
    }

    fn evaluate_deterministic(task: &RedundantTask, potential: usize) -> ConsensusOutcome {
        let mut groups: HashMap<String, Vec<ExecutorId>> = HashMap::new();
        for (executor, result) in &task.results {
            groups
                .entry(result.output_hash.clone())
                .or_default()
                .push(executor.clone());
        }

        let mut best: Option<(String, Vec<ExecutorId>)> = None;
        for (hash, members) in groups {
            if best
                .as_ref()
                .map(|(_, current)| members.len() > current.len())
                .unwrap_or(true)
            {
                best = Some((hash, members));
            }
        }

        if let Some((hash, members)) = best {
            if members.len() >= task.policy.quorum {
                let output_cid = task
                    .results
                    .values()
                    .find(|result| result.output_hash == hash)
                    .map(|result| result.output_cid.clone())
                    .unwrap_or_default();
                return ConsensusOutcome::Achieved(ConsensusRecord {
                    output_cid,
                    output_hash: hash,
                    score: 100,
                    semantic_score: None,
                    participants: members,
                });
            }
        }

        if task.results.len() >= potential {
            return ConsensusOutcome::Rejected(ConsensusFailureReason::HashMismatch);
        }

        ConsensusOutcome::Pending
    }

    fn evaluate_semantic(
        task: &RedundantTask,
        potential: usize,
        min_score: f32,
        epsilon: f32,
    ) -> ConsensusOutcome {
        let mut eligible: Vec<(ExecutorId, ExecutionResult)> = Vec::new();
        for (executor, result) in &task.results {
            if let Some(score) = result.semantic_score {
                if score >= min_score {
                    eligible.push((executor.clone(), result.clone()));
                }
            }
        }

        if eligible.len() >= task.policy.quorum {
            let mut min = f32::MAX;
            let mut max = f32::MIN;
            let mut sum = 0.0;
            let mut best_output: Option<String> = None;
            for (_, result) in &eligible {
                let score = result.semantic_score.unwrap_or(0.0);
                if score < min {
                    min = score;
                }
                if score > max {
                    max = score;
                    best_output = Some(result.output_cid.clone());
                }
                sum += score;
            }

            if (max - min) <= epsilon {
                let avg = sum / eligible.len() as f32;
                let score = ((avg * 100.0).round() as i32).clamp(0, 100) as u8;
                let output = best_output.unwrap_or_default();
                let hash = hash_output_cid(&output);
                let participants = eligible.into_iter().map(|(id, _)| id).collect();
                return ConsensusOutcome::Achieved(ConsensusRecord {
                    output_cid: output,
                    output_hash: hash,
                    score,
                    semantic_score: Some(avg),
                    participants,
                });
            }

            if task.results.len() >= potential {
                return ConsensusOutcome::Rejected(ConsensusFailureReason::SemanticMismatch);
            }
        }

        if task.results.len() >= potential {
            return ConsensusOutcome::Rejected(ConsensusFailureReason::SemanticMismatch);
        }

        ConsensusOutcome::Pending
    }

    fn finalize_outcome(
        &mut self,
        task_id: TaskId,
        outcome: ConsensusOutcome,
    ) -> Result<(), RedundancyError> {
        match outcome {
            ConsensusOutcome::Achieved(record) => {
                let attestation = AttestationBundle {
                    task_id,
                    output_cid: record.output_cid.clone(),
                    score: record.score,
                    semantic_score: record.semantic_score,
                    attestation_cid: None,
                    executors: record.participants.clone(),
                };

                if let Err(err) = self.submitter.submit_attestation(attestation) {
                    let failure = ConsensusFailureReason::AttestationSubmitFailed(err.to_string());
                    if let Some(task) = self.tasks.get_mut(&task_id) {
                        task.status = RedundantTaskStatus::Rejected(failure.clone());
                        let now = Instant::now();
                        task.resolved_at = Some(now);
                        self.metrics.total_latency_ms +=
                            now.duration_since(task.created_at).as_millis();
                    }
                    self.metrics.tasks_resolved += 1;
                    self.metrics.consensus_failures += 1;
                    return Err(RedundancyError::AttestationFailed(err.to_string()));
                }

                if let Some(task) = self.tasks.get_mut(&task_id) {
                    task.status = RedundantTaskStatus::ConsensusReached(record);
                    let now = Instant::now();
                    task.resolved_at = Some(now);
                    self.metrics.tasks_resolved += 1;
                    self.metrics.consensus_successes += 1;
                    self.metrics.total_latency_ms +=
                        now.duration_since(task.created_at).as_millis();
                }

                Ok(())
            }
            ConsensusOutcome::Rejected(reason) => {
                if let Some(task) = self.tasks.get_mut(&task_id) {
                    task.status = RedundantTaskStatus::Rejected(reason);
                    let now = Instant::now();
                    task.resolved_at = Some(now);
                    self.metrics.tasks_resolved += 1;
                    self.metrics.consensus_failures += 1;
                    self.metrics.total_latency_ms +=
                        now.duration_since(task.created_at).as_millis();
                }
                Ok(())
            }
            ConsensusOutcome::Pending => Ok(()),
        }
    }
}

fn hash_output_cid(output_cid: &str) -> String {
    blake3::hash(output_cid.as_bytes()).to_hex().to_string()
}

fn is_executor_capable(executor: &ExecutorInfo) -> bool {
    matches!(
        executor.capability,
        NodeCapability::SuperNode | NodeCapability::DirectorOnly
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::task_queue::{Lane, Task};

    #[derive(Default, Debug)]
    struct RecordingSubmitter {
        submissions: Vec<AttestationBundle>,
    }

    impl AttestationSubmitter for RecordingSubmitter {
        fn submit_attestation(
            &mut self,
            attestation: AttestationBundle,
        ) -> Result<(), AttestationError> {
            self.submissions.push(attestation);
            Ok(())
        }
    }

    fn make_task(id: u64) -> Task {
        Task::new(
            TaskId::new(id),
            "model".to_string(),
            "input".to_string(),
            Lane::Lane1,
        )
    }

    fn make_registry() -> StaticExecutorRegistry {
        StaticExecutorRegistry::new(vec![
            ExecutorInfo {
                id: "a".to_string(),
                capability: NodeCapability::SuperNode,
                reputation: 900,
            },
            ExecutorInfo {
                id: "b".to_string(),
                capability: NodeCapability::DirectorOnly,
                reputation: 800,
            },
            ExecutorInfo {
                id: "c".to_string(),
                capability: NodeCapability::DirectorOnly,
                reputation: 700,
            },
        ])
    }

    #[test]
    fn deterministic_consensus_succeeds_with_quorum() {
        let mut scheduler =
            RedundantScheduler::new(RedundancyConfig::default(), RecordingSubmitter::default());
        let registry = make_registry();

        let assignment = scheduler
            .assign_task(make_task(1), true, &registry)
            .unwrap();
        assert_eq!(assignment.executors.len(), 3);

        scheduler
            .record_success(
                TaskId::new(1),
                assignment.executors[0].clone(),
                "cidA".to_string(),
                None,
                100,
            )
            .unwrap();

        let outcome = scheduler
            .record_success(
                TaskId::new(1),
                assignment.executors[1].clone(),
                "cidA".to_string(),
                None,
                110,
            )
            .unwrap();

        match outcome {
            ConsensusOutcome::Achieved(record) => {
                assert_eq!(record.score, 100);
                assert_eq!(record.output_cid, "cidA");
            }
            _ => panic!("expected consensus achieved"),
        }

        let submitter = scheduler.into_submitter();
        assert_eq!(submitter.submissions.len(), 1);
        assert_eq!(submitter.submissions[0].score, 100);
    }

    #[test]
    fn deterministic_consensus_rejects_on_mismatch() {
        let mut scheduler =
            RedundantScheduler::new(RedundancyConfig::default(), RecordingSubmitter::default());
        let registry = make_registry();

        let assignment = scheduler
            .assign_task(make_task(2), true, &registry)
            .unwrap();

        scheduler
            .record_success(
                TaskId::new(2),
                assignment.executors[0].clone(),
                "cidA".to_string(),
                None,
                100,
            )
            .unwrap();
        scheduler
            .record_success(
                TaskId::new(2),
                assignment.executors[1].clone(),
                "cidB".to_string(),
                None,
                120,
            )
            .unwrap();
        let outcome = scheduler
            .record_success(
                TaskId::new(2),
                assignment.executors[2].clone(),
                "cidC".to_string(),
                None,
                130,
            )
            .unwrap();

        match outcome {
            ConsensusOutcome::Rejected(reason) => match reason {
                ConsensusFailureReason::HashMismatch => {}
                _ => panic!("expected hash mismatch"),
            },
            _ => panic!("expected consensus rejected"),
        }

        let submitter = scheduler.into_submitter();
        assert_eq!(submitter.submissions.len(), 0);
    }

    #[test]
    fn semantic_consensus_succeeds_within_epsilon() {
        let mut config = RedundancyConfig::default();
        config.semantic_min_score = 0.7;
        config.semantic_epsilon = 0.05;
        let mut scheduler = RedundantScheduler::new(config, RecordingSubmitter::default());
        let registry = make_registry();

        let assignment = scheduler
            .assign_task(make_task(3), false, &registry)
            .unwrap();

        scheduler
            .record_success(
                TaskId::new(3),
                assignment.executors[0].clone(),
                "cidA".to_string(),
                Some(0.8),
                100,
            )
            .unwrap();
        let outcome = scheduler
            .record_success(
                TaskId::new(3),
                assignment.executors[1].clone(),
                "cidB".to_string(),
                Some(0.82),
                120,
            )
            .unwrap();

        match outcome {
            ConsensusOutcome::Achieved(record) => {
                assert!(record.score >= 80);
                assert!(record.semantic_score.unwrap() >= 0.8);
            }
            _ => panic!("expected semantic consensus"),
        }

        let submitter = scheduler.into_submitter();
        assert_eq!(submitter.submissions.len(), 1);
    }

    #[test]
    fn semantic_consensus_rejects_when_all_results_diverge() {
        let mut config = RedundancyConfig::default();
        config.semantic_min_score = 0.7;
        config.semantic_epsilon = 0.01;
        let mut scheduler = RedundantScheduler::new(config, RecordingSubmitter::default());
        let registry = make_registry();

        let assignment = scheduler
            .assign_task(make_task(4), false, &registry)
            .unwrap();

        scheduler
            .record_success(
                TaskId::new(4),
                assignment.executors[0].clone(),
                "cidA".to_string(),
                Some(0.75),
                100,
            )
            .unwrap();
        scheduler
            .record_success(
                TaskId::new(4),
                assignment.executors[1].clone(),
                "cidB".to_string(),
                Some(0.85),
                110,
            )
            .unwrap();
        let outcome = scheduler
            .record_success(
                TaskId::new(4),
                assignment.executors[2].clone(),
                "cidC".to_string(),
                Some(0.95),
                120,
            )
            .unwrap();

        match outcome {
            ConsensusOutcome::Rejected(reason) => match reason {
                ConsensusFailureReason::SemanticMismatch => {}
                _ => panic!("expected semantic mismatch"),
            },
            _ => panic!("expected rejection"),
        }
    }

    #[test]
    fn timeout_marks_task_failed() {
        let mut config = RedundancyConfig::default();
        config.consensus_timeout = Duration::from_millis(0);
        let mut scheduler = RedundantScheduler::new(config, RecordingSubmitter::default());
        let registry = make_registry();

        scheduler
            .assign_task(make_task(5), true, &registry)
            .unwrap();

        let expired = scheduler.expire_timeouts(Instant::now());
        assert_eq!(expired.len(), 1);
        assert_eq!(scheduler.metrics().consensus_timeouts, 1);
    }

    #[test]
    fn load_test_metrics() {
        let mut scheduler =
            RedundantScheduler::new(RedundancyConfig::default(), RecordingSubmitter::default());
        let registry = make_registry();

        for id in 0..50 {
            let assignment = scheduler
                .assign_task(make_task(id), true, &registry)
                .unwrap();
            scheduler
                .record_success(
                    TaskId::new(id),
                    assignment.executors[0].clone(),
                    "cidA".to_string(),
                    None,
                    100,
                )
                .unwrap();
            scheduler
                .record_success(
                    TaskId::new(id),
                    assignment.executors[1].clone(),
                    "cidA".to_string(),
                    None,
                    110,
                )
                .unwrap();
        }

        assert_eq!(scheduler.metrics().consensus_successes, 50);
        let submitter = scheduler.into_submitter();
        assert_eq!(submitter.submissions.len(), 50);
    }

    #[test]
    fn selection_uses_registry_roles() {
        let mut scheduler =
            RedundantScheduler::new(RedundancyConfig::default(), RecordingSubmitter::default());
        let registry = StaticExecutorRegistry::new(vec![ExecutorInfo {
            id: "v1".to_string(),
            capability: NodeCapability::ValidatorOnly,
            reputation: 1000,
        }]);

        let result = scheduler.assign_task(make_task(6), true, &registry);
        assert!(matches!(result, Err(RedundancyError::NotEnoughExecutors)));
    }

    #[test]
    fn selection_prefers_high_reputation() {
        let mut scheduler =
            RedundantScheduler::new(RedundancyConfig::default(), RecordingSubmitter::default());
        let registry = StaticExecutorRegistry::new(vec![
            ExecutorInfo {
                id: "low".to_string(),
                capability: NodeCapability::DirectorOnly,
                reputation: 10,
            },
            ExecutorInfo {
                id: "high".to_string(),
                capability: NodeCapability::DirectorOnly,
                reputation: 100,
            },
            ExecutorInfo {
                id: "mid".to_string(),
                capability: NodeCapability::DirectorOnly,
                reputation: 50,
            },
        ]);

        let assignment = scheduler
            .assign_task(make_task(0), true, &registry)
            .unwrap();

        assert_eq!(assignment.executors[0], "high");
        assert_eq!(assignment.executors[1], "mid");
        assert_eq!(assignment.executors[2], "low");
    }
}
