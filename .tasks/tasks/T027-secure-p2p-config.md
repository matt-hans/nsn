---
id: T027
title: Secure P2P Configuration (Rate Limiting, DoS Protection)
status: pending
priority: 2
agent: backend
dependencies: [T021, T022]
blocked_by: []
created: 2025-12-24T00:00:00Z
updated: 2025-12-24T00:00:00Z
tags: [p2p, security, rate-limiting, off-chain, phase1]

context_refs:
  - context/project.md
  - context/architecture.md
  - context/acceptance-templates.md

docs_refs:
  - ../docs/prd.md (§17.1, §7.1)
  - ../docs/architecture.md (§7.3)

est_tokens: 9500
actual_tokens: null
---

## Description

Implement comprehensive security configuration for P2P networking layer including rate limiting, bandwidth throttling, connection timeouts, and DoS protection mechanisms. Ensures ICN nodes are resilient against network-level attacks.

**Technical Approach:**
- Rate limit incoming requests (100 requests/min per peer)
- Bandwidth throttling (100 Mbps max per connection)
- Connection timeout enforcement (30 seconds inactivity)
- Maximum connection limits (256 total, 2 per peer)
- TLS via rustls (pure Rust, no OpenSSL dependencies)
- Peer reputation-based connection prioritization
- Metrics for security event monitoring

**Integration Points:**
- Builds on T021 (libp2p core) and T022 (GossipSub)
- Used by all off-chain nodes for DoS protection
- Integrates with T026 (reputation oracle) for reputation-based policies

## Business Context

**User Story:** As an ICN node operator, I want protection against DoS attacks and resource exhaustion, so that my node remains stable and responsive under adversarial conditions.

**Why This Matters:**
- Prevents network-layer attacks (bandwidth exhaustion, connection flooding)
- Ensures fair resource allocation (no single peer monopolizes bandwidth)
- Protects against Sybil attacks (rate limiting + reputation)

**What It Unblocks:**
- Mainnet deployment (security is critical for production)
- High-value node operations (Directors, Super-Nodes)
- Economic security (prevents griefing attacks)

**Priority Justification:** Priority 2 (Important). Not critical for Phase 1 testnet (trusted environment), but essential for mainnet security.

## Acceptance Criteria

- [ ] **Rate Limiting**: Maximum 100 requests/min per peer enforced
- [ ] **Bandwidth Throttling**: Maximum 100 Mbps per connection enforced
- [ ] **Connection Timeout**: Idle connections closed after 30 seconds
- [ ] **Max Connections**: Maximum 256 total connections enforced
- [ ] **Max Connections Per Peer**: Maximum 2 connections per peer enforced
- [ ] **TLS Encryption**: rustls used for TLS (no OpenSSL)
- [ ] **Reputation-Based Prioritization**: High-reputation peers bypass rate limits (2× allowance)
- [ ] **Graylist Enforcement**: Peers exceeding limits are graylisted (temp ban for 1 hour)
- [ ] **Metrics Exposed**: Rate limit violations, bandwidth usage, connection attempts
- [ ] **DoS Detection**: Automatic detection of attack patterns (connection floods, message spam)
- [ ] **Graceful Degradation**: Node remains functional under attack (drops low-priority connections)

## Test Scenarios

**Test Case 1: Rate Limit Enforcement**
- Given: Peer A sending 150 requests/min to local node
- When: Request #101 arrives within 1-minute window
- Then:
  - Request rejected
  - Error logged: "Rate limit exceeded for peer A"
  - Metrics show rate_limit_violations +1
  - Peer A graylisted for 1 hour

**Test Case 2: Bandwidth Throttling**
- Given: Peer B attempting to send 150 Mbps
- When: Data transfer exceeds 100 Mbps threshold
- Then:
  - Connection throttled to 100 Mbps
  - Data queued or dropped (backpressure)
  - Metrics show bandwidth_throttled +1

**Test Case 3: Connection Timeout**
- Given: Connection to Peer C with no activity for 35 seconds
- When: Timeout check runs
- Then:
  - Connection closed
  - Event logged: "Connection timeout: Peer C"
  - Metrics show connection_timeouts +1

**Test Case 4: Max Connections Enforcement**
- Given: Local node with 256 active connections
- When: Peer D attempts new connection
- Then:
  - Connection rejected
  - Error logged: "Max connections reached (256/256)"
  - Metrics show connection_rejections +1

**Test Case 5: Per-Peer Connection Limit**
- Given: Peer E with 2 active connections
- When: Peer E attempts 3rd connection
- Then:
  - Connection rejected
  - Error logged: "Per-peer connection limit (2/2) for Peer E"
  - Metrics show per_peer_limit_violations +1

**Test Case 6: Reputation-Based Rate Limit Bypass**
- Given: Peer F with on-chain reputation 950 (high)
- When: Peer F sends 150 requests/min (would violate limit)
- Then:
  - Rate limit increased to 200 requests/min (2× bonus)
  - Requests accepted
  - Metrics show reputation_bypass_applied +1

**Test Case 7: Graylist Temporary Ban**
- Given: Peer G graylisted at T=0 (1-hour ban)
- When: Peer G attempts connection at T=30min
- Then:
  - Connection rejected
  - Error logged: "Peer G is graylisted (expires in 30min)"
  - Metrics show graylist_rejections +1
- When: Peer G attempts connection at T=61min
- Then:
  - Connection accepted (graylist expired)
  - Peer G removed from graylist

**Test Case 8: DoS Attack Detection (Connection Flood)**
- Given: 50 connection attempts from different IPs within 10 seconds
- When: DoS detection algorithm runs
- Then:
  - Pattern recognized: "Connection flood attack"
  - Alert logged: "DoS attack detected: connection flood"
  - Temporary global rate limit applied (50% reduction)
  - Metrics show dos_attacks_detected +1

## Reference Documentation
- [rustls Documentation](https://docs.rs/rustls/latest/rustls/)
- [libp2p Connection Limits](https://docs.rs/libp2p/latest/libp2p/swarm/struct.SwarmBuilder.html#method.with_connection_limits)
- [Governor Rate Limiter](https://docs.rs/governor/latest/governor/)

## Technical Implementation

**Required Components:**

```
off-chain/src/security/
├── mod.rs                  # Security configuration and orchestration
├── rate_limiter.rs         # Per-peer rate limiting
├── bandwidth.rs            # Bandwidth throttling
├── graylist.rs             # Temporary peer bans
├── dos_detection.rs        # Attack pattern detection
└── reputation_policy.rs    # Reputation-based policy adjustments

off-chain/tests/
└── integration_security.rs # Security integration tests
```

**Key Rust Modules:**

```rust
// src/security/mod.rs
use std::time::{Duration, Instant};
use libp2p::PeerId;

pub struct SecureP2pConfig {
    // Rate limiting
    pub max_requests_per_minute: u32,
    pub rate_limit_window: Duration,

    // Bandwidth
    pub max_bandwidth_mbps: u32,
    pub bandwidth_check_interval: Duration,

    // Connections
    pub max_connections: usize,
    pub max_connections_per_peer: usize,
    pub connection_timeout: Duration,

    // Reputation-based policies
    pub reputation_rate_limit_multiplier: f64,
    pub min_reputation_for_bypass: u64,

    // Graylist
    pub graylist_duration: Duration,
    pub graylist_threshold_violations: u32,

    // DoS detection
    pub dos_connection_flood_threshold: u32,
    pub dos_detection_window: Duration,
}

impl Default for SecureP2pConfig {
    fn default() -> Self {
        Self {
            max_requests_per_minute: 100,
            rate_limit_window: Duration::from_secs(60),
            max_bandwidth_mbps: 100,
            bandwidth_check_interval: Duration::from_secs(1),
            max_connections: 256,
            max_connections_per_peer: 2,
            connection_timeout: Duration::from_secs(30),
            reputation_rate_limit_multiplier: 2.0,
            min_reputation_for_bypass: 800,
            graylist_duration: Duration::from_secs(3600),  // 1 hour
            graylist_threshold_violations: 3,
            dos_connection_flood_threshold: 50,
            dos_detection_window: Duration::from_secs(10),
        }
    }
}

// src/security/rate_limiter.rs
use std::collections::HashMap;

pub struct RateLimiter {
    config: SecureP2pConfig,
    request_counts: Arc<RwLock<HashMap<PeerId, RequestCounter>>>,
    reputation_oracle: Option<Arc<ReputationOracle>>,
    metrics: Arc<SecurityMetrics>,
}

struct RequestCounter {
    count: u32,
    window_start: Instant,
}

impl RateLimiter {
    pub fn new(config: SecureP2pConfig, reputation_oracle: Option<Arc<ReputationOracle>>) -> Self {
        Self {
            config,
            request_counts: Arc::new(RwLock::new(HashMap::new())),
            reputation_oracle,
            metrics: Arc::new(SecurityMetrics::new()),
        }
    }

    pub fn check_rate_limit(&self, peer_id: &PeerId) -> Result<(), RateLimitError> {
        let mut counts = self.request_counts.write().unwrap();

        let now = Instant::now();

        let counter = counts.entry(peer_id.clone()).or_insert(RequestCounter {
            count: 0,
            window_start: now,
        });

        // Reset window if expired
        if now.duration_since(counter.window_start) > self.config.rate_limit_window {
            counter.count = 0;
            counter.window_start = now;
        }

        // Get rate limit (with reputation multiplier)
        let limit = self.get_rate_limit_for_peer(peer_id);

        // Check limit
        if counter.count >= limit {
            self.metrics.rate_limit_violations.inc_by(1, &[peer_id.to_string().as_str()]);
            return Err(RateLimitError::LimitExceeded {
                peer_id: peer_id.clone(),
                limit,
                actual: counter.count,
            });
        }

        counter.count += 1;
        Ok(())
    }

    fn get_rate_limit_for_peer(&self, peer_id: &PeerId) -> u32 {
        let base_limit = self.config.max_requests_per_minute;

        if let Some(oracle) = &self.reputation_oracle {
            let reputation = oracle.get_reputation(peer_id);

            // High-reputation peers get multiplier
            if reputation >= self.config.min_reputation_for_bypass {
                let multiplier = self.config.reputation_rate_limit_multiplier;
                let adjusted = (base_limit as f64 * multiplier) as u32;
                tracing::debug!(
                    "Peer {} has high reputation ({}), rate limit adjusted: {} -> {}",
                    peer_id, reputation, base_limit, adjusted
                );
                self.metrics.reputation_bypass_applied.inc();
                return adjusted;
            }
        }

        base_limit
    }
}

// src/security/graylist.rs
use std::collections::HashMap;

pub struct Graylist {
    config: SecureP2pConfig,
    graylisted: Arc<RwLock<HashMap<PeerId, GraylistEntry>>>,
    metrics: Arc<SecurityMetrics>,
}

struct GraylistEntry {
    banned_at: Instant,
    reason: String,
    violations: u32,
}

impl Graylist {
    pub fn new(config: SecureP2pConfig) -> Self {
        Self {
            config,
            graylisted: Arc::new(RwLock::new(HashMap::new())),
            metrics: Arc::new(SecurityMetrics::new()),
        }
    }

    pub fn is_graylisted(&self, peer_id: &PeerId) -> bool {
        let graylisted = self.graylisted.read().unwrap();

        if let Some(entry) = graylisted.get(peer_id) {
            let now = Instant::now();
            let elapsed = now.duration_since(entry.banned_at);

            if elapsed < self.config.graylist_duration {
                return true;
            }
        }

        false
    }

    pub fn add(&self, peer_id: PeerId, reason: String) {
        let mut graylisted = self.graylisted.write().unwrap();

        let entry = graylisted.entry(peer_id.clone()).or_insert(GraylistEntry {
            banned_at: Instant::now(),
            reason: reason.clone(),
            violations: 0,
        });

        entry.violations += 1;
        entry.banned_at = Instant::now();

        tracing::warn!(
            "Peer {} graylisted (violations: {}): {}",
            peer_id, entry.violations, reason
        );

        self.metrics.peers_graylisted.inc();
    }

    pub fn remove(&self, peer_id: &PeerId) {
        let mut graylisted = self.graylisted.write().unwrap();
        graylisted.remove(peer_id);
        tracing::info!("Peer {} removed from graylist", peer_id);
    }

    pub fn cleanup_expired(&self) {
        let mut graylisted = self.graylisted.write().unwrap();
        let now = Instant::now();

        graylisted.retain(|peer_id, entry| {
            let elapsed = now.duration_since(entry.banned_at);
            let keep = elapsed < self.config.graylist_duration;

            if !keep {
                tracing::info!("Graylist expired for peer {}", peer_id);
            }

            keep
        });
    }
}

// src/security/dos_detection.rs
pub struct DosDetector {
    config: SecureP2pConfig,
    connection_attempts: Arc<RwLock<VecDeque<Instant>>>,
    metrics: Arc<SecurityMetrics>,
}

impl DosDetector {
    pub fn detect_connection_flood(&self) -> bool {
        let attempts = self.connection_attempts.read().unwrap();
        let now = Instant::now();

        // Count attempts within detection window
        let recent_attempts = attempts.iter()
            .filter(|&&t| now.duration_since(t) < self.config.dos_detection_window)
            .count();

        if recent_attempts as u32 > self.config.dos_connection_flood_threshold {
            tracing::error!(
                "DoS attack detected: {} connection attempts in {}s",
                recent_attempts,
                self.config.dos_detection_window.as_secs()
            );
            self.metrics.dos_attacks_detected.inc();
            return true;
        }

        false
    }

    pub fn record_connection_attempt(&self) {
        let mut attempts = self.connection_attempts.write().unwrap();
        attempts.push_back(Instant::now());

        // Keep only recent attempts (sliding window)
        let cutoff = Instant::now() - self.config.dos_detection_window * 2;
        while let Some(&oldest) = attempts.front() {
            if oldest < cutoff {
                attempts.pop_front();
            } else {
                break;
            }
        }
    }
}

// Integration with P2pService
impl P2pService {
    pub async fn new_with_security(
        config: P2pConfig,
        security_config: SecureP2pConfig,
        reputation_oracle: Option<Arc<ReputationOracle>>,
    ) -> Result<Self, Error> {
        let rate_limiter = Arc::new(RateLimiter::new(security_config.clone(), reputation_oracle.clone()));
        let graylist = Arc::new(Graylist::new(security_config.clone()));
        let dos_detector = Arc::new(DosDetector::new(security_config.clone()));

        // Spawn graylist cleanup task
        let graylist_clone = graylist.clone();
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(60));
            loop {
                interval.tick().await;
                graylist_clone.cleanup_expired();
            }
        });

        // ... rest of P2pService initialization

        Ok(Self {
            // ...
            rate_limiter: Some(rate_limiter),
            graylist: Some(graylist),
            dos_detector: Some(dos_detector),
        })
    }

    fn handle_incoming_request(&mut self, peer_id: &PeerId, request: Request) -> Result<(), Error> {
        // Check graylist
        if let Some(graylist) = &self.graylist {
            if graylist.is_graylisted(peer_id) {
                tracing::debug!("Rejected request from graylisted peer {}", peer_id);
                return Err(Error::PeerGraylisted);
            }
        }

        // Check rate limit
        if let Some(rate_limiter) = &self.rate_limiter {
            if let Err(e) = rate_limiter.check_rate_limit(peer_id) {
                tracing::warn!("Rate limit exceeded for peer {}: {:?}", peer_id, e);

                // Graylist after repeated violations
                if let Some(graylist) = &self.graylist {
                    graylist.add(peer_id.clone(), "Rate limit violations".to_string());
                }

                return Err(Error::RateLimitExceeded);
            }
        }

        // Process request
        self.process_request(request)
    }

    fn handle_new_connection(&mut self, peer_id: &PeerId) -> Result<(), Error> {
        // Record connection attempt (DoS detection)
        if let Some(dos_detector) = &self.dos_detector {
            dos_detector.record_connection_attempt();

            if dos_detector.detect_connection_flood() {
                // Apply temporary global rate limit
                tracing::error!("DoS attack detected, applying global rate limit");
                // Reject new connections temporarily
                return Err(Error::DosAttackDetected);
            }
        }

        // Check graylist
        if let Some(graylist) = &self.graylist {
            if graylist.is_graylisted(peer_id) {
                return Err(Error::PeerGraylisted);
            }
        }

        Ok(())
    }
}
```

**Validation Commands:**

```bash
# Build with security features
cargo build --release -p icn-off-chain --features security

# Run unit tests
cargo test -p icn-off-chain security::

# Run integration tests
cargo test --test integration_security -- --nocapture

# Start node with security enabled
RUST_LOG=debug cargo run --release -- \
  --port 9000 \
  --max-connections 256 \
  --rate-limit 100

# Check security metrics
curl http://localhost:9100/metrics | grep security

# Stress test (simulate attack)
cargo run --example dos_attack_simulation
```

**Code Patterns:**
- Use Arc<RwLock<>> for concurrent access to security state
- Prometheus metrics for all security events
- Structured logging with peer_id and violation details
- Background tokio tasks for cleanup and monitoring

## Dependencies

**Hard Dependencies** (must be complete first):
- [T021] libp2p Core Setup - provides connection management
- [T022] GossipSub Configuration - protected by rate limiting

**Soft Dependencies:**
- [T026] Reputation Oracle - for reputation-based policies (optional)

**External Dependencies:**
- rustls (TLS)
- governor (rate limiting library, optional)
- prometheus (metrics)

## Design Decisions

**Decision 1: 100 Requests/Min Base Rate Limit**
- **Rationale:** Balance between legitimate high-activity nodes (directors during BFT) and DoS protection
- **Alternatives:**
  - 50/min: More restrictive but may limit legitimate use
  - 200/min: More permissive but weaker protection
- **Trade-offs:** 100/min is ~1.7 req/sec, reasonable for P2P coordination

**Decision 2: Reputation-Based Rate Limit Bypass**
- **Rationale:** Reward high-reputation nodes with more capacity, aligns incentives
- **Alternatives:**
  - No bypass: Simpler but punishes high-quality nodes
  - Stake-based bypass: Less fair (favors wealthy)
- **Trade-offs:** Requires reputation oracle integration (acceptable complexity)

**Decision 3: 1-Hour Graylist Duration**
- **Rationale:** Long enough to deter attacks, short enough to allow recovery after fixing issues
- **Alternatives:**
  - Permanent ban: Too harsh, no recovery path
  - 10 min: Too short, attackers can retry quickly
- **Trade-offs:** 1 hour is standard for temporary bans

**Decision 4: DoS Detection via Connection Flood Pattern**
- **Rationale:** Simple heuristic (50 connections in 10s) catches most flood attacks
- **Alternatives:**
  - Machine learning: More accurate but complex
  - No detection: Vulnerable to attacks
- **Trade-offs:** May have false positives during network events (acceptable)

## Risks & Mitigations

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| False positive rate limiting (legitimate burst traffic) | Medium | Low | Reputation bypass for high-rep nodes, monitor metrics, adjust thresholds based on data |
| Graylist circumvention (attacker uses new identities) | High | Medium | Combine with stake requirements (on-chain), reputation decay for new identities |
| DoS detection bypass (slow-rate attack) | Medium | Medium | Multiple detection heuristics (bandwidth, message spam), monitor long-term patterns |
| Rate limiter memory exhaustion | Low | Low | Periodic cleanup of old entries, LRU eviction for >10k peers |
| Reputation oracle lag (stale scores) | Low | Low | Accept eventual consistency, cache is updated every 60s |

## Progress Log

### [2025-12-24T00:00:00Z] - Task Created

**Created By:** task-creator agent
**Reason:** User request for P2P networking layer tasks (Phase 1)
**Dependencies:** T021 (libp2p core), T022 (GossipSub)
**Estimated Complexity:** Standard (rate limiting, DoS protection, reputation integration)

## Completion Checklist

**Code Complete:**
- [ ] All acceptance criteria met
- [ ] All test scenarios pass
- [ ] Code reviewed
- [ ] Documentation updated
- [ ] Clippy/linting passes
- [ ] Formatting applied
- [ ] No regression in existing tests

**Deployment Ready:**
- [ ] Integration tests pass on testnet
- [ ] Metrics verified in Grafana
- [ ] Logs structured and parseable
- [ ] Error paths tested
- [ ] Resource usage within limits
- [ ] Monitoring alerts configured

**Definition of Done:**
Task is complete when ALL acceptance criteria met, rate limiting enforces 100 req/min per peer, graylist bans violators for 1 hour, DoS detection alerts on attack patterns, integration tests demonstrate security enforcement, and production-ready with metrics and reputation-based policies.
