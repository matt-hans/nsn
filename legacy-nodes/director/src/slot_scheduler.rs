use crate::types::{BlockNumber, SlotNumber, SlotTask};
use std::collections::BTreeMap;
use tracing::{debug, warn};

/// Manages pipeline lookahead queue for upcoming slots
#[cfg_attr(feature = "stub", allow(dead_code))]
pub struct SlotScheduler {
    /// Pending slots ordered by slot number
    pending: BTreeMap<SlotNumber, SlotTask>,
    /// Maximum lookahead distance
    lookahead: u32,
}

#[cfg_attr(feature = "stub", allow(dead_code))]
impl SlotScheduler {
    pub fn new(lookahead: u32) -> Self {
        Self {
            pending: BTreeMap::new(),
            lookahead,
        }
    }

    /// Add a slot to the queue
    pub fn add_slot(&mut self, task: SlotTask) -> crate::error::Result<()> {
        debug!("Adding slot {} to scheduler", task.slot);
        self.pending.insert(task.slot, task);
        Ok(())
    }

    /// Get the next slot to process
    pub fn get_next_slot(&mut self) -> Option<SlotTask> {
        self.pending.iter().next().map(|(_, task)| task.clone())
    }

    /// Remove and return slot
    pub fn take_slot(&mut self, slot: SlotNumber) -> Option<SlotTask> {
        self.pending.remove(&slot)
    }

    /// Cancel a slot (e.g., deadline missed)
    pub fn cancel_slot(&mut self, slot: SlotNumber) -> crate::error::Result<()> {
        if self.pending.remove(&slot).is_some() {
            warn!("Canceled slot {}", slot);
            Ok(())
        } else {
            Err(
                crate::error::DirectorError::SlotScheduler(format!("Slot {} not found", slot))
                    .into(),
            )
        }
    }

    /// Check if slot deadline has passed
    pub fn is_deadline_passed(&self, slot: SlotNumber, current_block: BlockNumber) -> bool {
        if let Some(task) = self.pending.get(&slot) {
            current_block >= task.deadline_block
        } else {
            false
        }
    }

    /// Get all pending slots
    pub fn pending_slots(&self) -> Vec<SlotNumber> {
        self.pending.keys().copied().collect()
    }

    /// Clear all pending slots
    pub fn clear(&mut self) {
        self.pending.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test Case 2: Slot scheduler maintains lookahead queue
    /// Ensures pipeline queue maintains correct ordering
    #[test]
    fn test_slot_scheduler_lookahead() {
        let mut scheduler = SlotScheduler::new(2);

        let task100 = SlotTask {
            slot: 100,
            deadline_block: 1000,
            directors: vec!["Alice".to_string()],
        };
        let task101 = SlotTask {
            slot: 101,
            deadline_block: 1050,
            directors: vec!["Bob".to_string()],
        };
        let task102 = SlotTask {
            slot: 102,
            deadline_block: 1100,
            directors: vec!["Charlie".to_string()],
        };

        scheduler.add_slot(task100.clone()).unwrap();
        scheduler.add_slot(task102.clone()).unwrap(); // Add out of order
        scheduler.add_slot(task101.clone()).unwrap();

        // Should return in order: 100, 101, 102
        let next = scheduler.get_next_slot().unwrap();
        assert_eq!(next.slot, 100);

        scheduler.take_slot(100);
        let next = scheduler.get_next_slot().unwrap();
        assert_eq!(next.slot, 101);

        scheduler.take_slot(101);
        let next = scheduler.get_next_slot().unwrap();
        assert_eq!(next.slot, 102);
    }

    #[test]
    fn test_deadline_detection() {
        let mut scheduler = SlotScheduler::new(2);

        let task = SlotTask {
            slot: 50,
            deadline_block: 1200,
            directors: vec![],
        };

        scheduler.add_slot(task).unwrap();

        assert!(!scheduler.is_deadline_passed(50, 1199));
        assert!(scheduler.is_deadline_passed(50, 1200));
        assert!(scheduler.is_deadline_passed(50, 1201));
    }

    #[test]
    fn test_cancel_slot() {
        let mut scheduler = SlotScheduler::new(2);

        let task = SlotTask {
            slot: 100,
            deadline_block: 1000,
            directors: vec![],
        };

        scheduler.add_slot(task).unwrap();
        assert_eq!(scheduler.pending_slots().len(), 1);

        scheduler.cancel_slot(100).unwrap();
        assert_eq!(scheduler.pending_slots().len(), 0);

        // Canceling non-existent slot should error
        let result = scheduler.cancel_slot(999);
        assert!(result.is_err());
    }

    /// Test Case: Slot deadline missed - task cancelled
    /// Purpose: Verify generation task is cancelled when deadline reached
    /// Contract: No BFT result submitted after deadline
    /// Scenario 6 from task specification
    #[tokio::test]
    async fn test_slot_deadline_cancellation() {
        let mut scheduler = SlotScheduler::new(2);

        let task = SlotTask {
            slot: 50,
            deadline_block: 1200,
            directors: vec!["Alice".to_string()],
        };

        scheduler.add_slot(task).unwrap();

        // Simulate time passing - current block reaches deadline
        let current_block = 1200;

        // Check if deadline is passed
        assert!(
            scheduler.is_deadline_passed(50, current_block),
            "Deadline should be reached at block 1200"
        );

        // When deadline is detected, slot should be cancelled
        scheduler.cancel_slot(50).unwrap();

        // Verify slot is removed from queue
        assert_eq!(scheduler.pending_slots().len(), 0);

        // Attempt to get the slot should return None
        let removed_slot = scheduler.get_next_slot();
        assert!(removed_slot.is_none(), "Slot should be cancelled");

        // In real implementation, this would also:
        // 1. Stop ongoing Vortex generation task
        // 2. NOT submit BFT result to chain
        // 3. Emit SlotMissed event
        // 4. Increment metrics.missed_slots_total
    }

    /// Test Case: Multiple slots with different deadlines
    /// Purpose: Verify scheduler cancels only expired slots
    /// Contract: Non-expired slots remain in queue
    #[test]
    fn test_selective_deadline_cancellation() {
        let mut scheduler = SlotScheduler::new(3);

        let task1 = SlotTask {
            slot: 100,
            deadline_block: 1200,
            directors: vec![],
        };
        let task2 = SlotTask {
            slot: 101,
            deadline_block: 1250,
            directors: vec![],
        };
        let task3 = SlotTask {
            slot: 102,
            deadline_block: 1300,
            directors: vec![],
        };

        scheduler.add_slot(task1).unwrap();
        scheduler.add_slot(task2).unwrap();
        scheduler.add_slot(task3).unwrap();

        let current_block = 1225;

        // Slot 100 deadline passed (1200 < 1225)
        assert!(scheduler.is_deadline_passed(100, current_block));

        // Slot 101 deadline not passed (1250 > 1225)
        assert!(!scheduler.is_deadline_passed(101, current_block));

        // Slot 102 deadline not passed (1300 > 1225)
        assert!(!scheduler.is_deadline_passed(102, current_block));

        // Cancel expired slot
        scheduler.cancel_slot(100).unwrap();

        // Verify remaining slots still in queue
        let pending = scheduler.pending_slots();
        assert_eq!(pending.len(), 2);
        assert!(pending.contains(&101));
        assert!(pending.contains(&102));
        assert!(!pending.contains(&100));
    }

    /// Test Case: Deadline detection at exact boundary
    /// Purpose: Verify deadline detection is inclusive (>=)
    /// Contract: Deadline block itself is considered "passed"
    #[test]
    fn test_deadline_exact_boundary() {
        let mut scheduler = SlotScheduler::new(1);

        let task = SlotTask {
            slot: 50,
            deadline_block: 1000,
            directors: vec![],
        };

        scheduler.add_slot(task).unwrap();

        // Block before deadline
        assert!(!scheduler.is_deadline_passed(50, 999));

        // Exact deadline block - should be considered passed
        assert!(scheduler.is_deadline_passed(50, 1000));

        // Block after deadline
        assert!(scheduler.is_deadline_passed(50, 1001));
    }

    /// Test Case: Clear all pending slots
    /// Purpose: Verify clear() removes all slots
    /// Contract: Queue should be empty after clear
    #[test]
    fn test_clear_all_slots() {
        let mut scheduler = SlotScheduler::new(5);

        for i in 0..5 {
            let task = SlotTask {
                slot: i,
                deadline_block: 1000 + (i as u32 * 50),
                directors: vec![],
            };
            scheduler.add_slot(task).unwrap();
        }

        assert_eq!(scheduler.pending_slots().len(), 5);

        scheduler.clear();

        assert_eq!(scheduler.pending_slots().len(), 0);
        assert!(scheduler.get_next_slot().is_none());
    }

    /// Test Case: Take slot removes it from queue
    /// Purpose: Verify take_slot removes and returns the slot
    /// Contract: Slot should no longer be in queue after take
    #[test]
    fn test_take_slot_removal() {
        let mut scheduler = SlotScheduler::new(2);

        let task = SlotTask {
            slot: 100,
            deadline_block: 1000,
            directors: vec!["Alice".to_string()],
        };

        scheduler.add_slot(task.clone()).unwrap();
        assert_eq!(scheduler.pending_slots().len(), 1);

        // Take the slot
        let taken = scheduler.take_slot(100);
        assert!(taken.is_some());

        let taken_task = taken.unwrap();
        assert_eq!(taken_task.slot, 100);
        assert_eq!(taken_task.deadline_block, 1000);

        // Queue should now be empty
        assert_eq!(scheduler.pending_slots().len(), 0);

        // Taking again should return None
        let taken_again = scheduler.take_slot(100);
        assert!(taken_again.is_none());
    }
}
