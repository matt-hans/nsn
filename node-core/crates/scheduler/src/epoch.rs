//! Epoch tracking and On-Deck handling for NSN scheduler
//!
//! Manages epoch transitions and director election notifications:
//! - On-Deck notification (2 minutes before epoch): Start draining Lane 1 tasks
//! - Epoch start: Switch to Lane 0, load video models if needed
//! - Epoch end: Switch back to Lane 1, resume general compute

use nsn_types::EpochInfo;
use serde::{Deserialize, Serialize};

/// On-Deck notification lead time in seconds (2 minutes)
pub const ON_DECK_LEAD_TIME_SECS: u64 = 120;

/// Tracks epoch state and director election status
#[derive(Debug, Clone)]
pub struct EpochTracker {
    /// Current active epoch (if any)
    current_epoch: Option<EpochInfo>,
    /// Next epoch we're preparing for (On-Deck)
    next_epoch: Option<EpochInfo>,
    /// Whether this node is director for current epoch
    am_director_current: bool,
    /// Whether this node is director for next epoch (On-Deck)
    am_director_next: bool,
    /// Timestamp when On-Deck notification was received
    on_deck_received_at: Option<std::time::Instant>,
}

impl Default for EpochTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl EpochTracker {
    /// Create a new epoch tracker
    pub fn new() -> Self {
        Self {
            current_epoch: None,
            next_epoch: None,
            am_director_current: false,
            am_director_next: false,
            on_deck_received_at: None,
        }
    }

    /// Get the current epoch info
    pub fn current_epoch(&self) -> Option<&EpochInfo> {
        self.current_epoch.as_ref()
    }

    /// Get the next epoch info (On-Deck)
    pub fn next_epoch(&self) -> Option<&EpochInfo> {
        self.next_epoch.as_ref()
    }

    /// Check if this node is currently a director
    pub fn is_director_current(&self) -> bool {
        self.am_director_current
    }

    /// Check if this node will be a director in the next epoch
    pub fn is_director_next(&self) -> bool {
        self.am_director_next
    }

    /// Handle On-Deck notification
    ///
    /// Called when this node is elected as director for an upcoming epoch.
    /// This triggers Lane 1 draining to prepare for video generation.
    pub fn on_deck(&mut self, epoch: EpochInfo, am_director: bool) {
        tracing::info!(
            epoch = epoch.epoch,
            slot = epoch.slot,
            am_director = am_director,
            "On-Deck notification received"
        );

        self.next_epoch = Some(epoch);
        self.am_director_next = am_director;
        self.on_deck_received_at = Some(std::time::Instant::now());
    }

    /// Handle epoch start
    ///
    /// Called when a new epoch begins. The pending On-Deck epoch becomes current.
    pub fn epoch_started(&mut self, epoch: EpochInfo) {
        tracing::info!(
            epoch = epoch.epoch,
            slot = epoch.slot,
            active_lane = epoch.active_lane,
            "Epoch started"
        );

        // Move On-Deck status to current
        self.current_epoch = Some(epoch);
        self.am_director_current = self.am_director_next;

        // Clear On-Deck state
        self.next_epoch = None;
        self.am_director_next = false;
        self.on_deck_received_at = None;
    }

    /// Handle epoch end
    ///
    /// Called when the current epoch ends.
    pub fn epoch_ended(&mut self) {
        if let Some(ref epoch) = self.current_epoch {
            tracing::info!(epoch = epoch.epoch, "Epoch ended");
        }

        self.current_epoch = None;
        self.am_director_current = false;
    }

    /// Check if we're currently in On-Deck state
    ///
    /// Returns true if we've received an On-Deck notification and haven't
    /// yet transitioned to the new epoch.
    pub fn is_on_deck(&self) -> bool {
        self.next_epoch.is_some() && self.am_director_next
    }

    /// Check if Lane 1 should be drained
    ///
    /// Returns true if:
    /// - We're On-Deck for the next epoch as a director
    /// - We need to prepare for Lane 0 (video generation)
    pub fn should_drain_lane1(&self) -> bool {
        self.is_on_deck()
    }

    /// Get the time elapsed since On-Deck notification
    pub fn time_since_on_deck(&self) -> Option<std::time::Duration> {
        self.on_deck_received_at.map(|t| t.elapsed())
    }

    /// Check if we're close to epoch transition
    ///
    /// Returns true if we're within the On-Deck window and should
    /// prioritize finishing current Lane 1 work.
    pub fn is_transition_imminent(&self) -> bool {
        if let Some(elapsed) = self.time_since_on_deck() {
            // If we've been On-Deck for more than 90 seconds,
            // transition is imminent (30 seconds remaining)
            elapsed.as_secs() > (ON_DECK_LEAD_TIME_SECS - 30)
        } else {
            false
        }
    }

    /// Get the current active lane
    pub fn active_lane(&self) -> Option<u8> {
        self.current_epoch.as_ref().map(|e| e.active_lane)
    }

    /// Check if a specific lane is currently active
    pub fn is_lane_active(&self, lane: u8) -> bool {
        self.active_lane() == Some(lane)
    }
}

/// Epoch transition event for state machine coordination
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum EpochEvent {
    /// On-Deck notification received
    OnDeck { epoch: EpochInfo, am_director: bool },
    /// Epoch has started
    EpochStarted { epoch: EpochInfo },
    /// Epoch has ended
    EpochEnded { epoch: u64 },
    /// Director election result
    DirectorElected {
        epoch: u64,
        directors: Vec<String>,
        is_self: bool,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_epoch(epoch: u64, slot: u64, active_lane: u8) -> EpochInfo {
        EpochInfo {
            epoch,
            slot,
            block_number: epoch * 100 + slot,
            active_lane,
        }
    }

    #[test]
    fn test_epoch_tracker_new() {
        let tracker = EpochTracker::new();

        assert!(tracker.current_epoch().is_none());
        assert!(tracker.next_epoch().is_none());
        assert!(!tracker.is_director_current());
        assert!(!tracker.is_director_next());
        assert!(!tracker.is_on_deck());
    }

    #[test]
    fn test_on_deck() {
        let mut tracker = EpochTracker::new();
        let epoch = make_epoch(1, 0, 0);

        tracker.on_deck(epoch.clone(), true);

        assert!(tracker.is_on_deck());
        assert!(tracker.is_director_next());
        assert!(tracker.should_drain_lane1());
        assert!(tracker.next_epoch().is_some());
        assert_eq!(tracker.next_epoch().unwrap().epoch, 1);
    }

    #[test]
    fn test_on_deck_not_director() {
        let mut tracker = EpochTracker::new();
        let epoch = make_epoch(1, 0, 0);

        tracker.on_deck(epoch.clone(), false);

        assert!(!tracker.is_on_deck()); // Not on-deck if not director
        assert!(!tracker.is_director_next());
        assert!(!tracker.should_drain_lane1());
    }

    #[test]
    fn test_epoch_started() {
        let mut tracker = EpochTracker::new();
        let epoch = make_epoch(1, 0, 0);

        // First, receive On-Deck notification
        tracker.on_deck(epoch.clone(), true);
        assert!(tracker.is_on_deck());

        // Then, epoch starts
        tracker.epoch_started(epoch.clone());

        assert!(!tracker.is_on_deck()); // No longer On-Deck
        assert!(tracker.is_director_current()); // Now current director
        assert!(!tracker.is_director_next()); // Next is cleared
        assert!(tracker.current_epoch().is_some());
        assert_eq!(tracker.current_epoch().unwrap().epoch, 1);
    }

    #[test]
    fn test_epoch_ended() {
        let mut tracker = EpochTracker::new();
        let epoch = make_epoch(1, 0, 0);

        tracker.on_deck(epoch.clone(), true);
        tracker.epoch_started(epoch);
        assert!(tracker.is_director_current());

        tracker.epoch_ended();

        assert!(!tracker.is_director_current());
        assert!(tracker.current_epoch().is_none());
    }

    #[test]
    fn test_full_epoch_transition() {
        let mut tracker = EpochTracker::new();

        // Epoch 1: Not a director
        let epoch1 = make_epoch(1, 0, 1);
        tracker.epoch_started(epoch1);
        assert!(!tracker.is_director_current());
        assert_eq!(tracker.active_lane(), Some(1));

        // On-Deck for Epoch 2
        let epoch2 = make_epoch(2, 0, 0);
        tracker.on_deck(epoch2.clone(), true);
        assert!(tracker.is_on_deck());
        assert!(tracker.should_drain_lane1());

        // Epoch 2 starts
        tracker.epoch_started(epoch2);
        assert!(tracker.is_director_current());
        assert_eq!(tracker.active_lane(), Some(0));
        assert!(!tracker.is_on_deck());

        // Epoch 2 ends
        tracker.epoch_ended();
        assert!(!tracker.is_director_current());
    }

    #[test]
    fn test_active_lane() {
        let mut tracker = EpochTracker::new();

        // No active lane initially
        assert!(tracker.active_lane().is_none());
        assert!(!tracker.is_lane_active(0));
        assert!(!tracker.is_lane_active(1));

        // Start epoch with Lane 0
        let epoch = make_epoch(1, 0, 0);
        tracker.epoch_started(epoch);

        assert_eq!(tracker.active_lane(), Some(0));
        assert!(tracker.is_lane_active(0));
        assert!(!tracker.is_lane_active(1));
    }

    #[test]
    fn test_time_since_on_deck() {
        let mut tracker = EpochTracker::new();

        // No On-Deck yet
        assert!(tracker.time_since_on_deck().is_none());

        // Receive On-Deck
        let epoch = make_epoch(1, 0, 0);
        tracker.on_deck(epoch, true);

        // Should have some elapsed time
        std::thread::sleep(std::time::Duration::from_millis(10));
        let elapsed = tracker.time_since_on_deck().unwrap();
        assert!(elapsed.as_millis() >= 10);
    }
}
