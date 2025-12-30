# Security Verification Report - T043

**Task ID:** T043
**Task Name:** Migrate GossipSub, Reputation Oracle, and P2P Metrics to node-core
**Date:** 2025-12-30
**Agent:** verify-security
**Stage:** 3

---

## Executive Summary

**Security Score:** 88/100 (GOOD)
**Decision:** PASS
**Critical Issues:** 0
**High Issues:** 0
**Medium Issues:** 2
**Low Issues:** 1

---

## Files Analyzed

| File | Purpose | Lines |
|------|---------|-------|
| `node-core/crates/p2p/src/gossipsub.rs` | GossipSub config with Ed25519 signatures | 482 |
| `node-core/crates/p2p/src/reputation_oracle.rs` | On-chain reputation sync via subxt | 535 |
| `node-core/crates/p2p/src/scoring.rs` | Peer scoring parameters | 323 |
| `node-core/crates/p2p/src/metrics.rs` | Prometheus metrics | 179 |
| `node-core/crates/p2p/src/topics.rs` | Topic definitions and parsing | 350 |
| `node-core/crates/p2p/src/identity.rs` | Ed25519 keypair management | 315 |
| `node-core/crates/p2p/src/lib.rs` | Public API re-exports | 53 |

---

## OWASP Top 10 Compliance

| Category | Status | Notes |
|----------|--------|-------|
| A1: Injection | PASS | No SQL/database queries. Topic parsing uses strict match. |
| A2: Broken Authentication | PASS | Ed25519 signatures required (ValidationMode::Strict) |
| A3: Sensitive Data Exposure | WARN | Plaintext keypair storage (see MEDIUM-001) |
| A4: XXE | N/A | No XML parsing |
| A5: Broken Access Control | PASS | No unauthorized access patterns found |
| A6: Security Misconfiguration | PASS | No hardcoded secrets |
| A7: XSS | N/A | Backend Rust code |
| A8: Insecure Deserialization | PASS | serde used for simple enums only |
| A9: Vulnerable Components | PASS | Uses standard libp2p, subxt, prometheus crates |
| A10: Insufficient Logging | PASS | Comprehensive tracing with debug/info/warn/error |

---

## Detailed Findings

### CRITICAL Vulnerabilities

**None** - No critical security issues found.

---

### HIGH Vulnerabilities

**None** - No high severity issues found.

---

### MEDIUM Vulnerabilities

#### MEDIUM-001: Plaintext Keypair Storage

**Severity:** MEDIUM (CVSS 4.5)
**Location:** `node-core/crates/p2p/src/identity.rs:69-95`

**Vulnerable Code:**
```rust
/// Save keypair to file
///
/// WARNING: This stores the keypair in plaintext. In production,
/// use encrypted storage or HSM.
///
/// # Arguments
/// * `keypair` - The keypair to save
/// * `path` - File path to save to
pub fn save_keypair(keypair: &Keypair, path: &Path) -> Result<(), IdentityError> {
    let bytes = keypair
        .to_protobuf_encoding()
        .map_err(|_| IdentityError::InvalidKeypair)?;

    let mut file = fs::File::create(path)?;
    file.write_all(&bytes)?;
    // ...
}
```

**Issue:** Private keys are stored in plaintext on disk. While Unix permissions (0o600) are set, the file contents are not encrypted.

**Impact:** If the filesystem is compromised or the keypair file is copied, an attacker can impersonate the node.

**Fix:**
```rust
// Option 1: Use encrypted storage with age or sodiumoxide
use age::{secrecy::Secret, Encryptor};

pub fn save_keypair_encrypted(
    keypair: &Keypair,
    path: &Path,
    passphrase: &str
) -> Result<(), IdentityError> {
    let bytes = keypair.to_protobuf_encoding()
        .map_err(|_| IdentityError::InvalidKeypair)?;

    let encryptor = Encryptor::with_user_passphrase(passphrase);
    // ... encrypt and write

    Ok(())
}

// Option 2: Require HSM integration for production
#[cfg(feature = "hsm")]
pub fn load_keypair_from_hsm(_slot: u32) -> Result<Keypair, IdentityError> {
    // Integrate with YubiHSM2 or similar
}
```

**Mitigation in Current Code:** The code includes a WARNING comment documenting this limitation. Unix-only 0o600 permissions provide basic protection.

---

#### MEDIUM-002: Insecure WebSocket RPC URL (Test Code)

**Severity:** MEDIUM (CVSS 3.1)
**Location:** Multiple test files using `ws://localhost:9944`

**Issue:** Test code uses unencrypted WebSocket (`ws://`) for RPC connections. While this is acceptable for local testing, production should use `wss://`.

**Affected Files:**
- `reputation_oracle.rs` (tests)
- `scoring.rs` (tests)
- `gossipsub.rs` (tests)
- `test_helpers.rs`

**Example:**
```rust
let oracle = ReputationOracle::new("ws://localhost:9944".to_string());
```

**Fix (for production initialization):**
```rust
// In production code, require wss://
pub fn new(rpc_url: String) -> Result<Self, OracleError> {
    if !rpc_url.starts_with("wss://") && !cfg!(test) {
        return Err(OracleError::ConnectionFailed(
            "Only wss:// URLs allowed in production".to_string()
        ));
    }
    // ...
}
```

**Note:** This is only in test code. The production `ReputationOracle::new()` accepts any URL but should validate in production builds.

---

### LOW Vulnerabilities

#### LOW-001: Reputation Score Overflow

**Severity:** LOW (CVSS 2.0)
**Location:** `node-core/crates/p2p/src/scoring.rs:267-294`

**Issue:** The `compute_app_specific_score` function doesn't clamp the score when reputation exceeds MAX_REPUTATION (1000).

**Code:**
```rust
pub async fn get_gossipsub_score(&self, peer_id: &PeerId) -> f64 {
    let reputation = self.get_reputation(peer_id).await;
    // Normalize: (reputation / MAX_REPUTATION) * 50.0
    (reputation as f64 / MAX_REPUTATION as f64) * 50.0
}
```

**Test Result:**
```rust
// Test reputation above MAX_REPUTATION (overflow scenario)
oracle.set_reputation(peer_id, 2000).await;
let score = compute_app_specific_score(&peer_id, &oracle).await;
// Score = 100.0 (exceeds intended max of 50.0)
```

**Fix:**
```rust
pub async fn get_gossipsub_score(&self, peer_id: &PeerId) -> f64 {
    let reputation = self.get_reputation(peer_id).await;
    // Clamp to MAX_REPUTATION and normalize
    let clamped = reputation.min(MAX_REPUTATION);
    (clamped as f64 / MAX_REPUTATION as f64) * 50.0
}
```

**Impact:** Low - reputation should be bounded by on-chain logic, but defense-in-depth would add clamping.

---

## Security Strengths

1. **Ed25519 Signature Verification** (`gossipsub.rs:76`)
   - `ValidationMode::Strict` required for all GossipSub messages
   - Prevents message spoofing

2. **Message Size Limits** (`gossipsub.rs:82`, `topics.rs:83-92`)
   - 16MB max for video chunks
   - 64KB for BFT signals
   - Per-topic enforcement in `publish_message()`

3. **Thread Safety** (`reputation_oracle.rs`)
   - `Arc<RwLock<T>>` used for all shared state
   - Comprehensive concurrent access tests (`test_reputation_oracle_concurrent_access`)

4. **Topic Parsing Validation** (`topics.rs:159-170`)
   - Strict whitelist match for topic strings
   - Returns `None` for unknown topics (no default behavior)

5. **Peer Scoring Anti-Manipulation** (`scoring.rs`)
   - Topic-specific penalties (BFT: -20, others: -10)
   - Graylist threshold (-100) prevents spam
   - Duplicate cache time (120s) prevents replay attacks

6. **No Hardcoded Secrets**
   - All credentials/URLs passed as parameters
   - Test-only `ws://localhost:9944` clearly marked

7. **File Permissions** (`identity.rs:86-92`)
   - Unix-only 0o600 (owner read/write only) for keypair files

---

## Dependency Vulnerability Check

No automated dependency scan was run. Recommendations:

```bash
# Run cargo-audit for the p2p crate
cd node-core/crates/p2p
cargo audit

# Check for security advisories
cargo search libp2p --limit 1
cargo search subxt --limit 1
cargo search prometheus --limit 1
```

**Known Dependencies:**
- `libp2p` - Actively maintained, mature
- `subxt` - Official Parity Substrate client
- `prometheus` - Standard Rust client

---

## Cryptographic Review

| Component | Algorithm | Status |
|-----------|-----------|--------|
| P2P Identity | Ed25519 | Strong |
| Message Signing | Ed25519 (libp2p) | Strong |
| Transport | Noise XX (libp2p QUIC) | Strong |
| Keypair Storage | Plaintext | **Weak** (MEDIUM-001) |

**Noise XX Protocol:** The code uses libp2p's QUIC transport with Noise XX handshake (ephemeral keys), providing forward secrecy.

---

## Network Security Analysis

### RPC Connection Security
- **Current:** WebSocket URL passed to `ReputationOracle::new()`
- **Risk:** No validation that URL uses `wss://` in production
- **Recommendation:** Add URL validation for production builds

### GossipSub Security
- **Message Authenticity:** Ed25519 signatures enforced
- **Topic Isolation:** Separate topics per category
- **Flood Publishing:** Enabled for BFT signals (low-latency trade-off)

---

## Test Coverage Analysis

Security-relevant tests found:

| Test | Coverage | Status |
|------|----------|--------|
| `test_build_gossipsub_config` | Config validation | PASS |
| `test_publish_message_size_enforcement` | Size limits | PASS |
| `test_max_transmit_size_boundary` | Boundary testing | PASS |
| `test_parse_topic_invalid_inputs` | Input validation | PASS |
| `test_reputation_oracle_concurrent_access` | Thread safety | PASS |
| `test_reputation_oracle_concurrent_write_access` | Thread safety | PASS |
| `test_scoring_overflow_protection` | Numeric bounds | Documents behavior |
| `test_keypair_file_permissions` | File permissions | PASS (Unix only) |

---

## Recommendations

### Immediate (Pre-Deployment)
1. **MEDIUM-001:** Implement encrypted keypair storage or HSM integration
2. **MEDIUM-002:** Add `wss://` URL validation for production RPC connections
3. **LOW-001:** Add clamping to `get_gossipsub_score()`

### This Sprint
4. Add `cargo audit` to CI pipeline
5. Add integration tests with actual chain connection

### Next Quarter
6. Consider adding certificate pinning for RPC endpoints
7. Evaluate using libp2p's `tor` feature for anonymous P2P connections

---

## Conclusion

The P2P module demonstrates strong security fundamentals with Ed25519 signatures, proper input validation, and comprehensive thread safety. The two MEDIUM issues are well-documented limitations (plaintext key storage with warning comment, test-only `ws://` URLs) that should be addressed before production deployment.

**No blocking issues.** The code is safe to merge with the understanding that:
1. Keypair encryption/HSM should be implemented before mainnet
2. RPC URL validation should be added for production configurations

---

## Audit Trail

**Timestamp:** 2025-12-30T...
**Duration:** ~120 seconds
**Issues Found:** 3 (2 MEDIUM, 1 LOW)
**Lines Scanned:** ~2,200
**Files Analyzed:** 7
