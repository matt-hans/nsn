//! Kademlia helper functions
//!
//! Helper functions for building and configuring Kademlia DHT.

use libp2p::kad::{store::MemoryStore, Behaviour as KademliaBehaviour, Config as KademliaConfig};
use libp2p::{PeerId, StreamProtocol};

use super::kademlia::{
    K_VALUE, NSN_KAD_PROTOCOL_ID, PROVIDER_RECORD_TTL, PROVIDER_REPUBLISH_INTERVAL, QUERY_TIMEOUT,
};

/// Build Kademlia behaviour with NSN configuration
///
/// # Arguments
/// * `local_peer_id` - Local peer ID
///
/// # Returns
/// Configured Kademlia behaviour
pub fn build_kademlia(local_peer_id: PeerId) -> KademliaBehaviour<MemoryStore> {
    let mut config = KademliaConfig::default();

    // Set NSN protocol ID
    let protocol = StreamProtocol::try_from_owned(NSN_KAD_PROTOCOL_ID.to_string())
        .expect("NSN_KAD_PROTOCOL_ID is a valid protocol string");
    config.set_protocol_names(vec![protocol]);

    // Set query timeout
    config.set_query_timeout(QUERY_TIMEOUT);

    // Set replication factor (k-bucket size)
    config.set_replication_factor(
        K_VALUE
            .try_into()
            .expect("K_VALUE should fit in NonZeroUsize"),
    );

    // Set provider record TTL and publication interval
    config.set_provider_publication_interval(Some(PROVIDER_REPUBLISH_INTERVAL));
    config.set_provider_record_ttl(Some(PROVIDER_RECORD_TTL));

    // Set record TTL (for future record storage)
    config.set_record_ttl(Some(PROVIDER_RECORD_TTL));

    // Create memory store
    let store = MemoryStore::new(local_peer_id);

    // Create Kademlia behavior
    KademliaBehaviour::with_config(local_peer_id, store, config)
}
