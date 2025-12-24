//! ICN Common Library
//!
//! Shared components for off-chain nodes:
//! - P2P networking (libp2p + GossipSub + Kademlia)
//! - Chain client (subxt connection to Moonbeam)
//! - Shared types and protocols
//! - Reputation oracle

pub mod chain;
pub mod p2p;
pub mod types;

mod chain {
    //! Substrate chain client using subxt
}

mod p2p {
    //! libp2p networking layer
}

mod types {
    //! Shared types for ICN protocol
}
