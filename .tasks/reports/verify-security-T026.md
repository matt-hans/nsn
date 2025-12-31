# Security Verification Report - T026

**Task ID:** T026
**Task Name:** Reputation Oracle with chain RPC connection
**Files Analyzed:**
- `node-core/crates/p2p/src/reputation_oracle.rs`
- `node-core/crates/p2p/src/service.rs`
- `node-core/crates/p2p/src/identity.rs`
- `node-core/crates/p2p/src/config.rs`

**Date:** 2025-12-31
**Agent:** verify-security
**Stage:** 3 (STAGE 3 Verification)

---

## Executive Summary

### Security Score: 92/100 (PASS) ✅

### Critical Issues: 0
### High Issues: 0
### Medium Issues: 2
### Low Issues: 2

### Recommendation: **PASS** - No blocking vulnerabilities found. Code follows secure practices with room for minor improvements.

---

## Detailed Findings

### CRITICAL Vulnerabilities

**None** ✅

---

### HIGH Vulnerabilities

**None** ✅

---

### MEDIUM Vulnerabilities

#### M1: RPC URL Not Validated Before Connection
**Location:** `reputation_oracle.rs:314-318`

**Vulnerable Code:**
```rust
async fn connect(&self) -> Result<(), OracleError> {
    OnlineClient::<PolkadotConfig>::from_url(&self.rpc_url)
        .await
        .map(|_| ())
        .map_err(|e| OracleError::ConnectionFailed(e.to_string()))
}
```

**Issue:** The RPC URL is passed directly from user input without validation. A malicious URL could potentially:
- Point to a malicious endpoint serving crafted responses
- Cause unexpected connection behavior

**CVSS:** 4.3 (MEDIUM) - CWE-20

**Fix:**
```rust
pub fn validate_rpc_url(url: &str) -> Result<(), OracleError> {
    // Ensure URL starts with ws:// or wss://
    if !url.starts_with("ws://") && !url.starts_with("wss://") {
        return Err(OracleError::ConnectionFailed(
            "RPC URL must use ws:// or wss:// scheme".into()
        ));
    }

    // Prevent localhost in production builds
    #[cfg(not(feature = "test-helpers"))]
    {
        if url.contains("127.0.0.1") || url.contains("localhost") {
            return Err(OracleError::ConnectionFailed(
                "Localhost RPC not allowed in production".into()
            ));
        }
    }

    Ok(())
}

async fn connect(&self) -> Result<(), OracleError> {
    validate_rpc_url(&self.rpc_url)?;
    OnlineClient::<PolkadotConfig>::from_url(&self.rpc_url)
        .await
        .map(|_| ())
        .map_err(|e| OracleError::ConnectionFailed(e.to_string()))
}
```

---

#### M2: Test Helper Functions Without Feature Gate Protection
**Location:** `reputation_oracle.rs:389-398`

**Vulnerable Code:**
```rust
/// Manually set reputation for a peer (for testing)
#[cfg(any(test, feature = "test-helpers"))]
pub async fn set_reputation(&self, peer_id: PeerId, score: u64) {
    self.cache.write().await.insert(peer_id, score);
}
```

**Issue:** While `#[cfg(test)]` prevents compilation in production, the `test-helpers` feature could accidentally be enabled in release builds, allowing arbitrary reputation manipulation.

**CVSS:** 3.7 (MEDIUM) - CWE-433

**Fix:**
```rust
/// Manually set reputation for a peer (for testing ONLY)
///
/// # Security Warning
/// NEVER enable `test-helpers` feature in production builds.
#[cfg(test)]  // Remove "test-helpers" feature option
pub async fn set_reputation(&self, peer_id: PeerId, score: u64) {
    self.cache.write().await.insert(peer_id, score);
}
```

---

### LOW Vulnerabilities

#### L1: Keypair File Permissions Not Set
**Location:** `identity.rs:70-89`

**Issue:** When saving keypairs to disk, file permissions are not explicitly set. On Unix systems, files default to umask (often 0644), making private keys readable by other users.

**CVSS:** 2.1 (LOW) - CWE-732

**Current Code:**
```rust
pub fn save_keypair(keypair: &Keypair, path: &Path) -> Result<(), IdentityError> {
    let encoded = keypair.to_bytes();
    let mut file = File::create(path)?;
    file.write_all(&encoded)?;
    Ok(())
}
```

**Fix:**
```rust
#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;

pub fn save_keypair(keypair: &Keypair, path: &Path) -> Result<(), IdentityError> {
    let encoded = keypair.to_bytes();

    // Create file with restricted permissions first
    let mut options = File::options();
    options.write(true).create(true).truncate(true);

    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600); // Owner read/write only
    }

    let mut file = options.open(path)?;
    file.write_all(&encoded)?;

    #[cfg(not(unix))]
    {
        // On Windows, try to set restrictive permissions
        let _ = file.set_permissions(std::fs::Permissions::from_mode(0o600));
    }

    Ok(())
}
```

---

#### L2: Error Messages May Expose Internal State
**Location:** `reputation_oracle.rs:285-287`

**Issue:** Error messages include raw error strings from subxt which may contain internal paths or implementation details.

**CVSS:** 2.0 (LOW) - CWE-209

**Current:**
```rust
error!("Failed to connect to chain: {}. Retrying in 10s...", e);
```

**Fix:**
```rust
// Log detailed error at DEBUG level, generic at ERROR level
debug!("Chain connection failed (details): {:?}", e);
error!("Failed to connect to chain. Retrying in 10s...");
```

---

## Security Analysis by Category

### 1. Hardcoded Secrets ✅ PASS
- No hardcoded passwords, API keys, or tokens found
- RPC URL is passed as runtime parameter
- Keypairs are loaded from files or generated

### 2. Input Validation ⚠️ WARN
- RPC URL validation missing (MEDIUM)
- PeerId inputs use libp2p's built-in validation ✅
- Multiaddr parsing uses Rust's type system ✅

### 3. Cryptographic Operations ✅ PASS
- Ed25519 for signing via libp2p ✅
- No custom crypto implementations ✅
- Subtle usage of saturating arithmetic for reputation scoring ✅

### 4. Concurrent Access ✅ PASS
- RwLock protects cache and maps ✅
- Arc enables safe sharing across tasks ✅
- Comprehensive concurrent access tests (lines 559-663) ✅

### 5. Error Handling ✅ PASS
- Custom error types (OracleError, ServiceError, IdentityError) ✅
- No unwrap() in production code (all in tests only) ✅
- Graceful degradation on RPC failures ✅

### 6. Chain Security ✅ PASS
- Read-only storage queries ✅
- No extrinsic submission capability ✅
- Safe default values for unknown peers (100) ✅

---

## OWASP Top 10 Compliance

| Category | Status | Notes |
|----------|--------|-------|
| A1: Injection | ✅ PASS | No SQL/command injection. Storage queries use type-safe API. |
| A2: Broken Authentication | ✅ PASS | No auth required (read-only oracle). |
| A3: Sensitive Data Exposure | ⚠️ WARN | Keypair file permissions (LOW). |
| A4: XXE | ✅ PASS | No XML parsing. |
| A5: Broken Access Control | ✅ PASS | No access controls needed (read-only). |
| A6: Security Misconfiguration | ⚠️ WARN | RPC URL validation (MEDIUM). |
| A7: XSS | ✅ PASS | Not applicable (backend code). |
| A8: Insecure Deserialization | ✅ PASS | Uses SCALE codec, serde with controlled types. |
| A9: Vulnerable Components | ⚠️ CHECK | cargo-audit not available. |
| A10: Insufficient Logging | ✅ PASS | Comprehensive logging via tracing crate. |

---

## Dependency Vulnerabilities

**Status:** Unable to verify - cargo-audit not installed in environment.

**Recommendation:** Install and run `cargo audit` before production deployment.

---

## Positive Security Findings

1. **Safe Defaults:** `DEFAULT_REPUTATION = 100` prevents zero-reputation attacks
2. **Rate Limiting:** 60-second sync interval prevents chain spam
3. **Type Safety:** Extensive use of Rust's type system prevents invalid states
4. **Test Coverage:** Comprehensive tests including concurrent access scenarios
5. **Metrics Integration:** Prometheus metrics for observability
6. **Graceful Degradation:** System continues with cached data on RPC failure
7. **No Secret Leakage:** Error messages don't expose sensitive data

---

## Remediation Roadmap

### Immediate (Pre-Deployment)
1. **HIGH:** Run `cargo install cargo-audit && cargo audit`
2. **MEDIUM:** Add RPC URL validation (M1)

### This Sprint
1. **MEDIUM:** Remove `test-helpers` feature gate option (M2)
2. **LOW:** Set restrictive file permissions on keypair files (L1)

### Next Quarter
1. **LOW:** Sanitize error messages for production (L2)
2. Consider adding TLS certificate pinning for wss:// URLs

---

## Compilation Status

✅ **PASSES** - Code compiles successfully with `cargo check`

---

## Test Coverage Analysis

✅ **EXCELLENT** - 16 unit tests covering:
- Oracle creation and initialization
- Default reputation handling
- Reputation CRUD operations
- GossipSub score normalization
- Peer registration/unregistration
- Cache size tracking
- Connection failure handling
- **Concurrent access safety** (critical for thread safety)
- Metrics tracking

---

## Final Recommendation

**DECISION: PASS**

The Reputation Oracle implementation demonstrates strong security practices:
- No hardcoded secrets
- Safe concurrent access patterns
- Read-only chain operations
- Comprehensive error handling
- Type-safe storage queries

The two MEDIUM issues (RPC URL validation, test-helpers feature gate) are not blockers but should be addressed before mainnet deployment.

**Security Score: 92/100**

**Breakdown:**
- Secrets Management: 100/100
- Input Validation: 80/100 (M1)
- Cryptography: 100/100
- Concurrency: 100/100
- Error Handling: 90/100 (L2)
- File Operations: 80/100 (L1)

---

**Report Generated:** 2025-12-31T00:00:00Z
**Agent:** verify-security
**Version:** 1.0.0
