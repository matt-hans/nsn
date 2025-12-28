//! ICN Regional Relay Node (Tier 2 Distribution)
//!
//! Provides city-level content caching and distribution between Super-Nodes (Tier 1)
//! and Viewers (Tier 3). Key features:
//! - LRU cache with 1TB capacity and disk persistence
//! - QUIC server for viewer connections (WebTransport-compatible)
//! - QUIC client for upstream Super-Node fetching
//! - Latency-based region auto-detection
//! - Kademlia DHT for shard discovery

pub mod cache;
pub mod config;
pub mod dht_verification;
pub mod error;
pub mod health_check;
pub mod latency_detector;
pub mod merkle_proof;
pub mod metrics;
pub mod p2p_service;
pub mod quic_server;
pub mod relay_node;
pub mod upstream_client;

pub use config::Config;
pub use error::{RelayError, Result};
pub use relay_node::RelayNode;
