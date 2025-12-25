// STUB: Election monitor for watching DirectorsElected events

use crate::types::{PeerId, SlotNumber};
use tracing::{debug, info};

/// Monitors chain for director election events
pub struct ElectionMonitor {
    _own_peer_id: PeerId,
}

impl ElectionMonitor {
    pub fn new(own_peer_id: PeerId) -> Self {
        info!("Election monitor initialized for {}", own_peer_id);
        Self {
            _own_peer_id: own_peer_id,
        }
    }

    #[cfg_attr(feature = "stub", allow(dead_code))]
    pub fn is_elected(&self, slot: SlotNumber, directors: &[PeerId]) -> bool {
        let elected = directors.contains(&self._own_peer_id);
        debug!("Slot {}: elected={}", slot, elected);
        elected
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test Case 5: Election monitor self-detection
    /// Must correctly identify when elected
    #[test]
    fn test_election_self_detection() {
        let monitor = ElectionMonitor::new("Alice".to_string());

        let directors = vec![
            "Alice".to_string(),
            "Bob".to_string(),
            "Charlie".to_string(),
        ];

        assert!(monitor.is_elected(100, &directors));

        let other_directors = vec!["Bob".to_string(), "Charlie".to_string()];

        assert!(!monitor.is_elected(101, &other_directors));
    }
}
