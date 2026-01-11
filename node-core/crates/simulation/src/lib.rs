//! # NSN Simulation Crate
//!
//! Deterministic network simulation harness for multi-node E2E testing of NSN's
//! distributed components. Enables testing of epoch elections, BFT consensus,
//! P2P message propagation, and fault tolerance without spawning actual processes.
//!
//! ## Features
//!
//! - **In-memory network simulation**: No real sockets or ports required
//! - **Deterministic timing**: Uses `tokio::time::pause()` for reproducible tests
//! - **Byzantine fault injection**: Configure nodes to exhibit Byzantine behavior
//! - **Network partition simulation**: Test consensus under network splits
//! - **Reusable scenarios**: Pre-built test scenarios for common patterns
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use nsn_simulation::{TestHarness, Scenario};
//!
//! #[tokio::test]
//! async fn test_consensus() {
//!     let mut harness = TestHarness::new();
//!
//!     // Add 5 directors
//!     for _ in 0..5 {
//!         harness.add_director();
//!     }
//!
//!     // Run baseline consensus scenario
//!     let result = Scenario::BaselineConsensus.run(&mut harness).await;
//!     Scenario::BaselineConsensus.verify(&result).unwrap();
//! }
//! ```
//!
//! ## Architecture
//!
//! The simulation crate provides:
//!
//! - [`SimulatedNetwork`]: In-memory message routing with configurable latency
//! - [`TestHarness`]: High-level orchestrator for multi-node scenarios
//! - [`mocks`]: Reusable mock implementations for NSN services
//! - [`Scenario`]: Pre-defined test scenarios with verification

pub mod harness;
pub mod mocks;
pub mod network;
pub mod scenarios;

pub use harness::TestHarness;
pub use network::{LatencyProfile, PendingMessage, SimulatedNetwork, SimulatedNode};
pub use scenarios::{Scenario, ScenarioConfig, ScenarioFailure, ScenarioResult};

/// Byzantine behavior configurations for simulated nodes.
#[derive(Debug, Clone, PartialEq)]
pub enum ByzantineBehavior {
    /// Node drops all messages (crash fault)
    DropMessages,
    /// Node delays messages by specified duration
    DelayMessages(std::time::Duration),
    /// Node produces divergent CLIP embeddings during BFT
    DivergentEmbeddings,
    /// Node produces invalid signatures
    InvalidSignatures,
}

/// Node role in the simulated network.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NodeRole {
    /// Lane 0 director (video generation)
    Director,
    /// Lane 1 executor (task marketplace)
    Executor,
    /// Storage provider
    Storage,
}

impl std::fmt::Display for NodeRole {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NodeRole::Director => write!(f, "Director"),
            NodeRole::Executor => write!(f, "Executor"),
            NodeRole::Storage => write!(f, "Storage"),
        }
    }
}

/// Message topic categories for routing.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum TopicCategory {
    /// Recipe distribution
    Recipe,
    /// BFT consensus messages
    Consensus,
    /// Video chunk distribution
    VideoChunk,
    /// Task marketplace events
    TaskMarket,
    /// Epoch coordination
    Epoch,
}

impl std::fmt::Display for TopicCategory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TopicCategory::Recipe => write!(f, "recipe"),
            TopicCategory::Consensus => write!(f, "consensus"),
            TopicCategory::VideoChunk => write!(f, "video-chunk"),
            TopicCategory::TaskMarket => write!(f, "task-market"),
            TopicCategory::Epoch => write!(f, "epoch"),
        }
    }
}

/// Re-export error types for convenience.
pub mod error {
    pub use super::harness::HarnessError;
    pub use super::network::NetworkError;
    pub use super::scenarios::ScenarioFailure;
}
