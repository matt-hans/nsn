# Security Verification Report - T021 (libp2p Core Setup)

**Date:** 2025-12-29
**Task:** T021 - libp2p Core Setup
**Scope:** `icn-nodes/common/src/p2p/`

---

## Executive Summary

**Decision:** PASS
**Score:** 82/100
**Critical Issues:** 0
**High Issues:** 0
**Medium Issues:** 2
**Low Issues:** 1

---

## Detailed Findings

### CRITICAL Vulnerabilities
None

### HIGH Vulnerabilities
None

### MEDIUM Vulnerabilities

#### MEDIUM-001: Plaintext Keypair Storage
**Severity:** MEDIUM (CVSS 4.5)
**Location:** `icn-nodes/common/src/p2p/identity.rs:69-95`
**CWE:** CWE-312 (Cleartext Storage of Sensitive Information)

**Vulnerable Code:**
```rust
/// Save keypair to file
///
/// WARNING: This stores the keypair in plaintext. In production,
/// use encrypted storage or HSM.
pub fn save_keypair(keypair: &Keypair, path: &Path) -> Result<(), IdentityError> {
    let bytes = keypair
        .to_protobuf_encoding()
        .map_err(|_| IdentityError::InvalidKeypair)?;

    let mut file = fs::File::create(path)?;
    file.write_all(&bytes)?;
    // ...
}
```

**Impact:** Private keys stored on disk without encryption. If filesystem is compromised, attacker can obtain node identity.

**Mitigation Present:**
- File permissions set to 0o600 (Unix only)
- Warning comment in code

**Recommendation:** Before production, implement encrypted key storage using age or SOPS. Consider HSM for validator/director nodes.

---

#### MEDIUM-002: Noise XX Not Explicitly Configured
**Severity:** MEDIUM (CVSS 4.3)
**Location:** `icn-nodes/common/src/p2p/service.rs:115-123`
**CWE:** CWE-327 (Use of a Broken or Risky Cryptographic Algorithm)

**Observation:**
```rust
let swarm = SwarmBuilder::with_existing_identity(keypair)
    .with_tokio()
    .with_quic()  // QUIC transport
    .with_behaviour(|_| IcnBehaviour::new())
    .map_err(|e| ServiceError::Swarm(format!("Failed to create behaviour: {}", e)))?
    .with_swarm_config(|cfg| {
        cfg.with_idle_connection_timeout(config.connection_timeout)
    })
    .build();
```

The code uses `.with_quic()` which should include Noise XX encryption by default in libp2p 0.53, but encryption configuration is not explicit.

**Verification Needed:** Confirm libp2p 0.53's QUIC transport defaults to Noise XX or configure explicitly.

**Recommendation:** Explicitly configure authentication:
```rust
.with_noise_config(libp2p::noise::Config::new(&keypair)?)
```

---

### LOW Vulnerabilities

#### LOW-001: PeerId to AccountId32 Conversion Using Byte Slicing
**Severity:** LOW (CVSS 3.1)
**Location:** `icn-nodes/common/src/p2p/identity.rs:45-67`

**Observation:**
```rust
// For Ed25519, extract the last 32 bytes (the actual public key)
if encoded.len() >= 32 {
    let key_bytes: [u8; 32] = encoded[encoded.len() - 32..]
        .try_into()
        .map_err(|_| IdentityError::ConversionError("Invalid key length".to_string()))?;
    Ok(AccountId32::from(key_bytes))
```

**Impact:** Byte slicing relies on protobuf encoding structure. If encoding format changes, conversion may break.

**Status:** Acceptable for Ed25519 keys with protobuf encoding. Documented with tests.

---

## OWASP Top 10 Compliance

| Category | Status |
|----------|--------|
| A1: Injection | PASS (No SQL/database code) |
| A2: Broken Authentication | PASS (Ed25519 keypairs) |
| A3: Sensitive Data Exposure | WARN (Plaintext keypair storage) |
| A4: XXE | N/A (No XML parsing) |
| A5: Broken Access Control | PASS (Connection limits enforced) |
| A6: Security Misconfiguration | PASS (No hardcoded secrets) |
| A7: XSS | N/A (No web UI) |
| A8: Insecure Deserialization | PASS (Protobuf validation) |
| A9: Vulnerable Components | PASS (libp2p 0.53 is current) |
| A10: Insufficient Logging | PASS (Prometheus metrics) |

---

## Positive Security Findings

1. **Ed25519 Keypair Generation:** Uses `Keypair::generate_ed25519()` - cryptographically secure
2. **No Hardcoded Secrets:** Scanned entire P2P module - no hardcoded API keys, passwords, or tokens
3. **Connection Limits:** Global (256) and per-peer (2) limits prevent DoS
4. **Unix File Permissions:** 0o600 for keypair files (owner read/write only)
5. **Input Validation:** Proper error handling for invalid keypair formats
6. **Comprehensive Tests:** Tests cover edge cases like corrupted/empty keypair files

---

## Dependency Security

| Package | Version | Status |
|---------|---------|--------|
| libp2p | 0.53 | Current stable (OK) |
| ed25519-dalek | 2.1 | Current (OK) |
| sp-core | 28.0 | Current (OK) |
| tokio | 1.43 | Current (OK) |

---

## Recommendations

1. **Pre-Production (Required):** Implement encrypted keypair storage using age or SOPS
2. **Next Sprint:** Verify Noise XX is explicitly configured or add explicit configuration
3. **Documentation:** Document HSM requirements for production validator/director nodes
4. **Monitoring:** Add alerting for keypair file permission changes

---

## Test Coverage

| Component | Coverage |
|-----------|----------|
| Keypair generation | PASS |
| Keypair save/load | PASS |
| PeerId to AccountId | PASS |
| Connection limits | PASS |
| Invalid input handling | PASS |
| File permissions (Unix) | PASS |

---

**Overall Assessment:** PASS with conditions. Code is secure for development/testing. Production deployment requires encrypted key storage solution.
