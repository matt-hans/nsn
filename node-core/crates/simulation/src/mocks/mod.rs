//! Reusable mock implementations for NSN services.
//!
//! These mocks are extracted from existing test files and enhanced with
//! additional capabilities for simulation testing.
//!
//! ## Lane 0 Mocks (Video Generation)
//!
//! - [`MockVortexClient`] - AI video generation via sidecar
//! - [`MockBftParticipant`] - BFT consensus participation
//! - [`MockChunkPublisher`] - P2P video chunk distribution
//!
//! ## Lane 1 Mocks (Task Marketplace)
//!
//! - [`MockExecutionRunner`] - Task execution via sidecar
//! - [`MockResultSubmitter`] - Result submission to chain
//! - [`ChainListenerAdapter`] - Bridge MockChainClient to channel interface
//!
//! ## Common Mocks
//!
//! - [`MockChainClient`] - On-chain event injection and extrinsic tracking

pub mod bft;
pub mod chain;
pub mod listener_adapter;
pub mod p2p;
pub mod runner;
pub mod submitter;
pub mod vortex;

// Lane 0 mocks
pub use bft::MockBftParticipant;
pub use p2p::MockChunkPublisher;
pub use vortex::MockVortexClient;

// Lane 1 mocks
pub use listener_adapter::ChainListenerAdapter;
pub use runner::{ExecutionEvent, MockExecutionRunner};
pub use submitter::{MockResultSubmitter, SubmissionEvent};

// Common mocks
pub use chain::{MockChainClient, MockChainState, SubmittedExtrinsic};
