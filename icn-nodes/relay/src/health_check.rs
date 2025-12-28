//! Super-Node health monitoring
//!
//! Periodically pings Super-Nodes to detect failures and update routing

use std::collections::HashMap;
use std::time::Duration;
use tokio::time;
use tracing::{debug, info, warn};

/// Health status for a Super-Node
#[derive(Debug, Clone)]
pub struct HealthStatus {
    pub address: String,
    pub healthy: bool,
    pub last_check: chrono::DateTime<chrono::Utc>,
    pub consecutive_failures: u32,
}

/// Health checker for Super-Nodes
pub struct HealthChecker {
    /// Super-Node addresses to monitor
    super_nodes: Vec<String>,
    /// Health status map
    statuses: HashMap<String, HealthStatus>,
    /// Check interval
    check_interval: Duration,
}

impl HealthChecker {
    /// Create new health checker
    pub fn new(super_nodes: Vec<String>, check_interval_secs: u64) -> Self {
        let mut statuses = HashMap::new();
        for addr in &super_nodes {
            statuses.insert(
                addr.clone(),
                HealthStatus {
                    address: addr.clone(),
                    healthy: true,
                    last_check: chrono::Utc::now(),
                    consecutive_failures: 0,
                },
            );
        }

        Self {
            super_nodes,
            statuses,
            check_interval: Duration::from_secs(check_interval_secs),
        }
    }

    /// Run health check loop
    pub async fn run(&mut self) {
        let mut interval = time::interval(self.check_interval);

        loop {
            interval.tick().await;
            self.check_all_nodes().await;
        }
    }

    /// Check health of all Super-Nodes
    async fn check_all_nodes(&mut self) {
        debug!(
            "Running health checks for {} Super-Nodes",
            self.super_nodes.len()
        );

        for addr in &self.super_nodes.clone() {
            let healthy = self.check_node(addr).await;

            if let Some(status) = self.statuses.get_mut(addr) {
                status.last_check = chrono::Utc::now();

                if healthy {
                    if !status.healthy {
                        info!("Super-Node {} recovered", addr);
                    }
                    status.healthy = true;
                    status.consecutive_failures = 0;
                } else {
                    status.consecutive_failures += 1;

                    if status.consecutive_failures >= 3 {
                        if status.healthy {
                            warn!("Super-Node {} marked unhealthy after 3 failures", addr);
                        }
                        status.healthy = false;
                    }
                }
            }
        }
    }

    /// Check health of single Super-Node (via TCP ping)
    async fn check_node(&self, addr: &str) -> bool {
        crate::latency_detector::ping_super_node(addr, Duration::from_secs(2))
            .await
            .is_some()
    }

    /// Get healthy Super-Node addresses
    pub fn get_healthy_nodes(&self) -> Vec<String> {
        self.statuses
            .values()
            .filter(|s| s.healthy)
            .map(|s| s.address.clone())
            .collect()
    }

    /// Get current health status
    pub fn get_status(&self) -> HashMap<String, HealthStatus> {
        self.statuses.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_health_checker_creation() {
        let super_nodes = vec!["127.0.0.1:9002".to_string(), "127.0.0.1:9003".to_string()];
        let checker = HealthChecker::new(super_nodes.clone(), 60);

        assert_eq!(checker.super_nodes.len(), 2);
        assert_eq!(checker.statuses.len(), 2);
    }

    #[tokio::test]
    async fn test_get_healthy_nodes() {
        let super_nodes = vec!["127.0.0.1:9002".to_string()];
        let checker = HealthChecker::new(super_nodes, 60);

        let healthy = checker.get_healthy_nodes();
        assert_eq!(healthy.len(), 1);
        assert_eq!(healthy[0], "127.0.0.1:9002");
    }
}
