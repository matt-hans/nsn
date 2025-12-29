//! ICN Super-Node Library
//!
//! Tier 1 storage and relay infrastructure with:
//! - Reed-Solomon erasure coding (10+4)
//! - CID-based shard persistence
//! - Kademlia DHT for shard discovery
//! - QUIC transport for shard distribution
//! - On-chain audit response

pub mod audit_monitor;
pub mod chain_client;
pub mod config;
pub mod erasure;
pub mod error;
pub mod metrics;
pub mod p2p_service;
pub mod quic_server;
pub mod storage;
pub mod storage_cleanup;

pub use config::Config;
pub use error::{Result, SuperNodeError};
