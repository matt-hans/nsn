# Security Audit Report - T013 (Viewer Client Application)

**Date:** 2025-12-28
**Task:** T013 - Viewer Client Application
**Scope:** viewer/ directory (Tauri 2.0 + React desktop app)
**Agent:** verify-security

---

## Executive Summary

- **Score:** 78/100
- **Critical:** 0
- **High:** 1
- **Medium:** 3
- **Low:** 2
- **Recommendation:** **WARN** - Address HIGH before production deployment

---

## CRITICAL Vulnerabilities

None

---

## HIGH Vulnerabilities

### VULN-001: Missing Content Security Policy (CSP)
**Severity:** HIGH (CVSS 7.5)
**Location:** `viewer/src-tauri/tauri.conf.json:23`
**CWE:** CWE-693

**Vulnerable Code:**
```json
"security": {
    "csp": null
}
```

**Impact:**
- No Content Security Policy configured
- Application is vulnerable to XSS attacks if malicious content is injected
- Missing inline script restrictions, form-action restrictions, and source whitelisting

**Fix:**
```json
"security": {
    "csp": "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data: https:; connect-src 'self' ws://localhost:* wss://*.icn.network; object-src 'none'; frame-ancestors 'none';"
}
```

**Notes:**
- Tauri's IPC (`invoke()`) provides some protection as frontend cannot call unregistered commands
- However, a comprehensive CSP defense-in-depth is recommended for production

---

## MEDIUM Vulnerabilities

### VULN-002: esbuild Development Vulnerability
**Severity:** MEDIUM (CVSS 5.3)
**Location:** `viewer/package.json` (dependencies)
**CWE:** CWE-346

**Issue:** `esbuild <=0.24.2` has SSRF vulnerability (GHSA-67mh-4wv8-2f99).

**Details:**
- Development server allows any website to send requests and read responses
- Affects Vite 0.11.0 - 6.1.6 (via esbuild dependency)
- Fix available: `pnpm update esbuild vite`

**Impact:** Development-only risk; does not affect production builds.

**Fix:**
```bash
pnpm update esbuild vite
```

### VULN-003: Mock Peer IDs in Production Code
**Severity:** MEDIUM (CVSS 5.3)
**Location:** `viewer/src-tauri/src/commands.rs:44-52`

**Vulnerable Code:**
```rust
RelayInfo {
    peer_id: "12D3KooWFakeRelay1".to_string(),
    multiaddr: "/ip4/127.0.0.1/udp/9003/quic/webtransport".to_string(),
    // ...
}
```

**Impact:**
- Mock/fake data hardcoded in relay discovery
- May mislead users or cause connection issues if deployed to production
- Development artifacts should not be present in release builds

**Fix:**
- Use feature flags: `#[cfg(debug_assertions)]` for mock data
- Return error when no real relays available in release mode
- Document the expected production relay discovery flow

### VULN-004: localStorage Without Encryption
**Severity:** MEDIUM (CVSS 4.6)
**Location:** `viewer/src/store/appStore.ts:114-122`

**Vulnerable Code:**
```typescript
{
    name: "icn-viewer-storage",
    partialize: (state) => ({
        volume: state.volume,
        quality: state.quality,
        seedingEnabled: state.seedingEnabled,
        currentSlot: state.currentSlot,
    }),
}
```

**Impact:**
- User settings stored in localStorage without encryption
- Data accessible to any process with browser data access
- While current data is non-sensitive, future sensitive data could be exposed

**Fix:**
- Use Tauri's secure storage API for sensitive data
- For current non-sensitive settings, document the data classification
- Consider adding encryption for `seedingEnabled` if it becomes privacy-relevant

---

## LOW Vulnerabilities

### VULN-005: DevTools Open in Debug Mode
**Severity:** LOW (CVSS 3.1)
**Location:** `viewer/src-tauri/src/main.rs:17-19`

**Code:**
```rust
#[cfg(debug_assertions)]
{
    let window = app.get_webview_window("main").unwrap();
    window.open_devtools();
}
```

**Impact:**
- DevTools automatically opens in debug builds
- Acceptable for development, but ensure this is NOT present in release builds
- The `#[cfg(debug_assertions)]` correctly guards this

**Fix:**
- Current implementation is correct - no action needed
- Ensure release builds do not include this (verified via cfg attribute)

### VULN-006: Unsanitized Console Error Messages
**Severity:** LOW (CVSS 3.1)
**Location:** `viewer/src/App.tsx:64`, `viewer/src/services/p2p.ts:24`

**Code:**
```typescript
console.error("Initialization error:", error);
```

**Impact:**
- Error messages may contain sensitive information
- In production, verbose logging should be disabled

**Fix:**
- Use a logging utility that respects build environment
- Sanitize error messages before logging in production

---

## Dependency Vulnerabilities

| Package | Version | CVE | Severity | Fix |
|---------|---------|-----|----------|-----|
| esbuild | <=0.24.2 | GHSA-67mh-4wv8-2f99 | MODERATE | `pnpm update esbuild` |
| vite | 0.11.0-6.1.6 | (via esbuild) | MODERATE | `pnpm update vite` |

**Note:** These are development-only dependencies and do not affect production builds.

---

## OWASP Top 10 Compliance

| Category | Status | Notes |
|----------|--------|-------|
| A1: Injection | PASS | No SQL/command injection vectors found |
| A2: Broken Authentication | N/A | No authentication in viewer client |
| A3: Sensitive Data Exposure | WARN | localStorage unencrypted, CSP missing |
| A4: XXE | N/A | No XML parsing |
| A5: Broken Access Control | N/A | No server-side access control |
| A6: Security Misconfiguration | WARN | CSP disabled, mock data in code |
| A7: XSS | PASS | React auto-escapes, no innerHTML usage |
| A8: Insecure Deserialization | PASS | serde_json used safely |
| A9: Vulnerable Components | WARN | esbuild vulnerability (dev only) |
| A10: Logging & Monitoring | WARN | Verbose console logging |

---

## Positive Security Findings

1. **No XSS vulnerabilities found:** React used correctly with JSX (no `innerHTML`, `dangerouslySetInnerHTML`, or `eval()`)

2. **No hardcoded secrets:** No API keys, passwords, tokens, or private keys found in codebase

3. **Type-safe IPC:** Tauri's command system provides type safety between Rust backend and JS frontend

4. **No unsafe Rust:** No `unsafe` blocks detected in the Rust code

5. **Input validation present:** Quality settings validated against whitelist in `App.tsx`

6. **Proper error handling:** Commands return `Result<T, String>` for proper error propagation

7. **Test coverage:** Unit tests present for storage and commands

---

## Threat Model for Viewer Client

| Threat | Likelihood | Mitigation |
|--------|-----------|------------|
| XSS via malicious content | Low | React auto-escapes; add CSP for defense-in-depth |
| Local file read | Low | Tauri enforces file system access; no fs commands exposed |
| IPC abuse | Low | Only registered commands callable via `invoke()` |
| Data tampering in localStorage | Medium | Consider cryptographic signatures for critical data |
| P2P relay impersonation | Medium | Future: implement peer ID verification with libp2p (T027) |
| Supply chain attack | Medium | Enable lockfile auditing, pin dependency versions |

---

## Deferred Implementation Tracking

| Issue | Tracked In | Severity |
|-------|------------|----------|
| Peer signature verification | T027 | HIGH |
| DHT response validation | T027 | HIGH |
| WebTransport implementation | T027 | - |

---

## Remediation Roadmap

### Immediate (Pre-Deployment)
1. **[HIGH]** Add CSP to `tauri.conf.json` (VULN-001)

### This Sprint
2. **[MEDIUM]** Update esbuild/vite to fix dev dependency vulnerability (VULN-002)
3. **[MEDIUM]** Remove or guard mock relay data with `#[cfg(debug_assertions)]` (VULN-003)
4. **[LOW]** Implement production-safe logging utility (VULN-006)

### Next Release
5. **[MEDIUM]** Evaluate need for encrypted storage (VULN-004)

### Post-T027 (Regional Relay Implementation)
6. **[HIGH]** Implement Ed25519 peer signature verification
7. **[HIGH]** Add DHT response validation with typed Multiaddr/PeerId

---

## Compliance Notes

- **GDPR:** localStorage usage should be documented in privacy policy
- **PCI-DSS:** N/A - no payment processing in viewer client
- **COPPA:** N/A - no data collection from children

---

## Conclusion

The ICN Viewer Client demonstrates **solid security fundamentals** with no critical vulnerabilities. The primary concern is the **missing Content Security Policy**, which is a defense-in-depth measure rather than a critical issue given Tauri's built-in IPC protections.

The codebase follows secure coding practices:
- No hardcoded secrets
- React used correctly (no XSS patterns)
- Type-safe IPC via Tauri
- Proper error handling

**Recommendation: WARN** - Address the HIGH severity CSP issue and update dev dependencies before production deployment.

---

**Report Generated:** 2025-12-28
**Agent:** verify-security
**Task:** T013 (Viewer Client Application)
