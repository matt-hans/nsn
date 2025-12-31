# Security Audit Report - T024 P2P Kademlia DHT

**Date:** 2025-12-30
**Task:** T024 - Implement Kademlia DHT for P2P peer discovery
**Agent:** verify-security
**Scope:** node-core/crates/p2p/src/kademlia.rs, kademlia_helpers.rs, behaviour.rs, service.rs

---

## Executive Summary

| Metric | Value |
|--------|-------|
| **Security Score** | 78/100 |
| **Critical Issues** | 0 |
| **High Issues** | 1 |
| **Medium Issues** | 3 |
| **Low Issues** | 2 |
| **Recommendation** | PASS (with improvements recommended) |

---

## Detailed Findings

### CRITICAL Vulnerabilities

None.

---

### HIGH Vulnerabilities

#### VULN-001: Plaintext Keypair Storage
**Severity:** HIGH (CVSS 6.5)
**Location:** `identity.rs:69-94`
**CWE:** CWE-312 (Cleartext Storage of Sensitive Information)

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

    // Set restrictive permissions (Unix only)
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut perms = file.metadata()?.permissions();
        perms.set_mode(0o600); // Only owner can read/write
        fs::set_permissions(path, perms)?;
    }

    Ok(())
}
```

**Impact:** Private keys are stored in plaintext on disk. While 0o600 permissions restrict access on Unix, this provides no protection against:
- Root compromise
- Disk theft/imaging
- File system backup exposure
- Windows systems (no permission setting)

**Fix (Recommended):**
```rust
use chacha20poly1305::{
    aead::{Aead, AeadCore, KeyInit, OsRng},
    ChaCha20Poly1305,
};

const KEYPAIR_ENCRYPTION_VERSION: u8 = 1;

pub fn save_keypair_encrypted(
    keypair: &Keypair,
    path: &Path,
    encryption_password: &[u8],
) -> Result<(), IdentityError> {
    let bytes = keypair
        .to_protobuf_encoding()
        .map_err(|_| IdentityError::InvalidKeypair)?;

    // Derive key from password using Argon2
    let salt = ChaCha20Poly1305::generate_nonce(&mut OsRng);
    let key = argon2::hash_raw(encryption_password, &salt, &argon2::Argon2::default())
        .map_err(|e| IdentityError::ConversionError(format!("Key derivation: {}", e)))?;

    let cipher = ChaCha20Poly1305::new_from_slice(&key[..32])
        .map_err(|_| IdentityError::ConversionError("Invalid key length".to_string()))?;

    let nonce = ChaCha20Poly1305::generate_nonce(&mut OsRng);
    let ciphertext = cipher.encrypt(&nonce, bytes.as_ref())
        .map_err(|_| IdentityError::ConversionError("Encryption failed".to_string()))?;

    // Write: version(1) + salt + nonce + ciphertext
    let mut file = fs::File::create(path)?;
    file.write_all(&[KEYPAIR_ENCRYPTION_VERSION])?;
    file.write_all(&salt)?;
    file.write_all(&nonce)?;
    file.write_all(&ciphertext)?;

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut perms = file.metadata()?.permissions();
        perms.set_mode(0o600);
        fs::set_permissions(path, perms)?;
    }

    Ok(())
}
```

---

### MEDIUM Vulnerabilities

#### VULN-002: Insufficient Peer Validation Before Adding to Routing Table
**Severity:** MEDIUM (CVSS 5.3)
**Location:** `service.rs:310-317`
**CWE:** CWE-346 (Origin Validation Error)

**Vulnerable Code:**
```rust
SwarmEvent::ConnectionEstablished { peer_id, endpoint, .. } => {
    let addr = endpoint.get_remote_address().clone();
    self.swarm
        .behaviour_mut()
        .kademlia
        .add_address(peer_id, addr.clone());
    debug!("Added connected peer {} at {} to Kademlia routing table", peer_id, addr);
}
```

**Impact:** Every connected peer is immediately added to Kademlia routing table without validation. This enables:
- Eclipse attacks (adversary populates routing table with Sybil nodes)
- Routing table poisoning
- Accelerated network partition

**Fix:**
```rust
// Add connection manager validation
SwarmEvent::ConnectionEstablished { peer_id, endpoint, .. } => {
    // Only add to routing table if:
    // 1. Connection is outbound (we dialed them) OR
    // 2. Peer has minimum reputation score (from oracle)
    let addr = endpoint.get_remote_address().clone();

    // Check reputation before adding to routing table
    let should_add = if self.connection_manager.is_outbound_connection(peer_id) {
        true
    } else {
        // For inbound connections, verify minimum reputation
        match self.reputation_oracle.get_score(peer_id).await {
            Ok(score) if score >= MIN_KADEMLIA_REPUTATION => true,
            _ => false,
        }
    };

    if should_add {
        self.swarm
            .behaviour_mut()
            .kademlia
            .add_address(peer_id, addr.clone());
        debug!("Added peer {} to Kademlia routing table", peer_id);
    } else {
        debug!("Skipped adding peer {} to Kademlia (low reputation/no validation)", peer_id);
    }
}
```

#### VULN-003: Missing Size Bounds on `local_provided_shards`
**Severity:** MEDIUM (CVSS 5.0)
**Location:** `kademlia.rs:108`, `service.rs:150`
**CWE:** CWE-770 (Allocation of Resources Without Limits)

**Vulnerable Code:**
```rust
/// Local shards being provided (for republish)
local_provided_shards: Vec<[u8; 32]>,
```

**Impact:** Unbounded growth of shard tracking. A malicious or buggy client could publish unlimited shard records, causing:
- Memory exhaustion
- DoS through unbounded `republish_providers()` iterations

**Fix:**
```rust
const MAX_LOCAL_PROVIDED_SHARDS: usize = 10_000;

pub struct KademliaService {
    // ...
    /// Local shards being provided (for republish)
    /// Bounded to prevent resource exhaustion
    local_provided_shards: Vec<[u8; 32]>,
}

impl KademliaService {
    pub fn start_providing(
        &mut self,
        shard_hash: [u8; 32],
        result_tx: oneshot::Sender<Result<bool, KademliaError>>,
    ) -> QueryId {
        // Check capacity before adding
        if self.local_provided_shards.len() >= MAX_LOCAL_PROVIDED_SHARDS {
            let _ = result_tx.send(Err(KademliaError::ProviderPublishFailed(
                "Maximum provider records reached".to_string(),
            )));
            // Return a dummy query ID - we can't return QueryId from here
            // In practice, return error through different channel
            return self.kademlia.get_closest_peers(PeerId::random());
        }

        if !self.local_provided_shards.contains(&shard_hash) {
            self.local_provided_shards.push(shard_hash);
        }
        // ... rest of function
    }
}
```

#### VULN-004: Google STUN Servers Without Certificate Pinning
**Severity:** MEDIUM (CVSS 4.6)
**Location:** `config.rs:57-61`
**CWE:** CWE-295 (Improper Certificate Validation)

**Vulnerable Code:**
```rust
pub stun_servers: Vec<String>,
// Default:
stun_servers: vec![
    "stun.l.google.com:19302".to_string(),
    "stun1.l.google.com:19302".to_string(),
    "stun2.l.google.com:19302".to_string(),
],
```

**Impact:** STUN requests are sent without certificate pinning. While STUN is UDP-based and doesn't use TLS, the STUN server addresses could be:
- Modified by a MITM on local network (DNS hijacking)
- Redirected to malicious STUN servers revealing external IP

**Fix:**
1. Use multiple STUN server providers (not just Google)
2. Validate STUN responses (magic cookie, transaction ID matching)
3. Consider running own STUN infrastructure for production

---

### LOW Vulnerabilities

#### VULN-005: Excessive `unwrap()`/`expect()` in Production Code
**Severity:** LOW (CVSS 3.1)
**Location:** Multiple files
**CWE:** CWE-391 (Unchecked Error Condition)

**Instances in non-test code:**
```rust
// kademlia.rs:122 - Protocol ID parse
let protocol = StreamProtocol::try_from_owned(NSN_KAD_PROTOCOL_ID.to_string())
    .expect("NSN_KAD_PROTOCOL_ID is a valid protocol string");

// kademlia.rs:129 - Replication factor
kad_config.set_replication_factor(K_VALUE.try_into().expect("K_VALUE fits in NonZeroUsize"));

// kademlia.rs:211 - Start providing
.expect("start_providing should not fail immediately");
```

**Impact:** These `expect()` calls will panic on unexpected conditions. While the constants are compile-time validated, runtime failures could still occur due to:
- Memory allocation failure
- Unexpected libp2p state

**Fix:**
```rust
// Convert expect() to proper error handling
pub fn new(local_peer_id: PeerId, config: KademliaServiceConfig) -> Result<Self, KademliaError> {
    let protocol = StreamProtocol::try_from_owned(NSN_KAD_PROTOCOL_ID.to_string())
        .map_err(|e| KademliaError::QueryFailed(format!("Invalid protocol: {}", e)))?;

    let replication_factor = NonZeroUsize::try_from(K_VALUE)
        .map_err(|_| KademliaError::QueryFailed("Invalid K_VALUE".to_string()))?;

    // ... rest of code with proper error propagation
}
```

#### VULN-006: Metrics Endpoint on All Interfaces
**Severity:** LOW (CVSS 3.1)
**Location:** `service.rs:194`
**CWE:** CWE-215 (Information Exposure Through Debug Information)

**Vulnerable Code:**
```rust
let metrics_addr: SocketAddr = ([0, 0, 0, 0], config.metrics_port).into();
```

**Impact:** Prometheus metrics exposed on all interfaces (0.0.0.0). Could expose:
- Peer connection counts
- Network topology information
- Internal metrics that could aid reconnaissance

**Fix:**
```rust
// Bind to localhost by default, make interface configurable
let metrics_bind_addr = config.metrics_bind_addr.unwrap_or_else(|| {
    ([127, 0, 0, 1], config.metrics_port).into()
});
let metrics_addr: SocketAddr = metrics_bind_addr.into();
```

---

## Dependency Security Check

### Cargo.toml Dependencies

| Dependency | Version | Known CVEs | Status |
|------------|---------|------------|--------|
| libp2p | workspace | None (0.53.x) | OK |
| tokio | workspace | None | OK |
| sp-core | 28.0 | None | OK |
| prometheus | 0.13 | None | OK |
| subxt | 0.37 | None | OK |
| serde | workspace | None | OK |
| hex | 0.4 | None | OK |

**Recommendation:** Run `cargo audit` before deployment to verify latest CVE status.

---

## DHT-Specific Security Analysis

### Eclipse Attack Mitigation

**Current State:** Partially implemented

| Control | Implemented | Notes |
|---------|-------------|-------|
| Connection limit | Yes | `max_connections: 256` |
| Per-peer limit | Yes | `max_connections_per_peer: 2` |
| Reputation-based routing | Partial | ReputationOracle exists but not used in routing decisions |
| Sybil resistance | No | No stake-based peer gating |

### Kademlia Security Parameters

| Parameter | Value | Assessment |
|-----------|-------|------------|
| k-bucket size (K) | 20 | Standard, appropriate |
| Query timeout | 10s | Appropriate |
| Record TTL | 12h | Standard |
| Protocol ID | `/nsn/kad/1.0.0` | Prevents cross-network pollution |

---

## OWASP Top 10 Compliance

| Category | Status | Notes |
|----------|--------|-------|
| A1: Injection | PASS | No dynamic SQL/command construction |
| A2: Broken Authentication | PASS | Ed25519 keys properly used |
| A3: Sensitive Data Exposure | WARN | Keypair stored in plaintext |
| A4: XXE | N/A | No XML parsing |
| A5: Broken Access Control | WARN | Insufficient peer validation |
| A6: Security Misconfiguration | MINOR | Metrics on all interfaces |
| A7: XSS | N/A | Not web-facing |
| A8: Insecure Deserialization | PASS | Uses serde with standard types |
| A9: Vulnerable Components | PASS | No known CVEs in dependencies |
| A10: Logging | PASS | No sensitive data in logs (hex encoding used) |

---

## Positive Security Findings

1. **No hardcoded credentials** - No passwords, API keys, or secrets found
2. **No unsafe Rust** - Code review found no `unsafe` blocks
3. **Proper use of libp2p crypto** - Ed25519, Noise XX encryption
4. **File permissions** - Unix 0o600 on keypair files
5. **Test coverage** - Comprehensive unit and integration tests
6. **Hex encoding** - Sensitive data (shard hashes) logged as hex
7. **No SQL injection vectors** - No database queries

---

## Threat Model: Kademlia DHT

| Threat | Likelihood | Impact | Mitigation Status |
|--------|-----------|--------|-------------------|
| Eclipse Attack | Medium | High | Partial (connection limits exist, reputation filtering needed) |
| Sybil Attack | Medium | Medium | Open (no stake gating) |
| Routing Table Poisoning | Medium | Medium | Open (see VULN-002) |
| DHT Spam | Low | Low | Partial (query timeout exists) |
| Record Saturation | Low | Medium | Open (see VULN-003) |

---

## Recommendations

### Immediate (Pre-Production)
1. **Implement encrypted keypair storage** (VULN-001) - HIGH priority
2. **Add reputation-based routing table admission** (VULN-002) - HIGH priority
3. **Bound local_provided_shards size** (VULN-003) - MEDIUM priority

### This Sprint
4. Run `cargo audit` and fix any dependency vulnerabilities
5. Add peer reputation gating before Kademlia routing table insertion
6. Configure metrics endpoint binding address (localhost only for dev, configurable for prod)

### Next Quarter
7. Implement stake-based Sybil resistance for peer admission
8. Add STUN response validation
9. Consider operating your own STUN infrastructure

---

## Conclusion

The T024 P2P Kademlia implementation demonstrates good security practices with no critical vulnerabilities. The main concerns are:

1. **HIGH:** Plaintext keypair storage should use encryption
2. **MEDIUM:** Peer validation before routing table insertion needs improvement
3. **MEDIUM:** Unbounded shard tracking needs size limits

**Decision: PASS** - The code can proceed with the recommended improvements implemented before production deployment. The HIGH-severity issue (plaintext keys) should be addressed before mainnet launch.

---

**Report Generated:** 2025-12-30T22:30:00Z
**Agent:** verify-security
**Task:** T024
