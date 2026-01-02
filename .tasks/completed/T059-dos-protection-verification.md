# T059: DoS Protection Integration Verification

## Priority: P2
## Complexity: 1 week
## Status: Pending
## Depends On: T027 (Secure P2P Config)

---

## Objective

Wire existing DoS detection and rate limiting modules into the active P2P service and validate they are enforced in production paths.

## Background

`dos_detection.rs` and `rate_limiter.rs` exist, but integration with libp2p service is unclear. Without enforcement, nodes remain vulnerable to message floods and resource exhaustion.

## Implementation

### Step 1: Locate Enforcement Hook Points

- Identify inbound message handlers for GossipSub and request/response
- Attach rate limiter checks before payload deserialization

### Step 2: Integrate Rate Limiter

- Enforce per-peer limits (messages/second, bytes/second)
- Apply penalties or temporary bans on violation

### Step 3: Integrate DoS Detection

- Enable anomaly detection hooks (sustained abuse, connection churn)
- Emit metrics and structured logs

### Step 4: Configuration

- Expose limits in config with safe defaults
- Add overrides for testing

## Acceptance Criteria

- [ ] All inbound paths enforce rate limiting
- [ ] Repeated violations trigger peer downscore or temporary ban
- [ ] Metrics exported for dropped/limited messages
- [ ] Configurable thresholds validated
- [ ] Integration tests simulate flooding and verify mitigation

## Testing

- Unit test: rate limiter blocks after N messages
- Integration test: simulated peer flood -> messages dropped
- Soak test: sustained traffic under limit passes

## Deliverables

1. P2P service integration changes
2. Config schema updates
3. Tests for flood scenarios
4. Metrics dashboard updates

---

**This task improves resilience against DoS and spam attacks.**
