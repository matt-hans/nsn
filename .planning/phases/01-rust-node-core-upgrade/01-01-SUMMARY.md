---
phase: 01-rust-node-core-upgrade
plan: 01
subsystem: p2p-networking
tags: [webrtc, libp2p, certificate, config]

dependency_graph:
  requires: []
  provides:
    - libp2p-webrtc dependency (0.9.0-alpha.1)
    - CertificateManager for WebRTC certificate persistence
    - P2pConfig WebRTC configuration fields
  affects:
    - 01-02 (WebRTC transport setup will use these foundations)
    - 02-01 (Discovery endpoint may expose certhash)

tech_stack:
  added:
    - libp2p-webrtc v0.9.0-alpha.1
  patterns:
    - Certificate persistence with PEM format
    - Config extension for feature flags

key_files:
  created:
    - node-core/crates/p2p/src/cert.rs
  modified:
    - node-core/Cargo.toml
    - node-core/crates/p2p/Cargo.toml
    - node-core/crates/p2p/src/config.rs
    - node-core/crates/p2p/src/lib.rs

decisions:
  - id: webrtc-alpha-version
    decision: Use libp2p-webrtc v0.9.0-alpha.1 instead of planned 0.9.0-alpha.2
    rationale: v0.9.0-alpha.2 does not exist on crates.io; alpha.1 is latest available
    impact: None - compiles without version conflicts with libp2p 0.53

metrics:
  duration: 4m 30s
  completed: 2026-01-18
  tasks: 3/3
  tests_added: 8
  tests_total_passed: 191
---

# Phase 01 Plan 01: WebRTC Dependencies and Certificate Persistence Summary

**One-liner:** libp2p-webrtc dependency added with CertificateManager for stable certhash persistence across restarts

## What Was Built

### Task 1: Add libp2p-webrtc dependency
Added `libp2p-webrtc = { version = "0.9.0-alpha.1", features = ["tokio", "pem"] }` to workspace and nsn-p2p crate.

**Note:** The plan specified version 0.9.0-alpha.2, but this version does not exist on crates.io. Used 0.9.0-alpha.1 (the latest available) which compiles cleanly with libp2p 0.53.

### Task 2: Create certificate persistence module
Created `cert.rs` module with:
- `CertificateManager` struct for managing WebRTC certificates
- `load_or_generate()` method that loads existing PEM certificate or generates new one
- Certificate persistence to `{data_dir}/webrtc_cert.pem`
- Unix file permissions set to 0o600 for security
- `CertError` enum with Io, Generation, Parse variants

Exports added to `lib.rs`:
- `pub mod cert;`
- `pub use cert::{CertError, CertificateManager};`

### Task 3: Extend P2pConfig with WebRTC fields
Added configuration fields:
- `enable_webrtc: bool` - Enable WebRTC transport (default: false)
- `webrtc_port: u16` - UDP port for WebRTC (default: 9003)
- `data_dir: Option<PathBuf>` - Directory for certificate persistence
- `external_address: Option<String>` - External multiaddr for NAT/Docker

## Tests Added

| Module | Test | Purpose |
|--------|------|---------|
| cert | test_generate_and_load_certificate | Verify generate then load same fingerprint |
| cert | test_certificate_persistence_across_instances | Verify fingerprint persists across manager instances |
| cert | test_load_nonexistent_generates_new | Verify auto-generation when no cert exists |
| cert | test_certificate_file_permissions | Verify 0o600 permissions on Unix |
| cert | test_cert_error_display | Verify error message formatting |
| config | test_config_defaults (updated) | Verify WebRTC defaults |
| config | test_config_serialization (updated) | Include WebRTC fields |
| config | test_webrtc_config_fields (new) | Verify WebRTC field serialization |

## Commits

| Hash | Type | Description |
|------|------|-------------|
| f4dd233 | feat | Add libp2p-webrtc dependency for WebRTC transport |
| dc36778 | feat | Add WebRTC certificate persistence module |
| dbd2e1f | feat | Extend P2pConfig with WebRTC transport fields |

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] libp2p-webrtc version correction**
- **Found during:** Task 1
- **Issue:** Plan specified v0.9.0-alpha.2 but this version does not exist on crates.io
- **Fix:** Used v0.9.0-alpha.1 (latest available version)
- **Files modified:** node-core/Cargo.toml
- **Commit:** f4dd233

**2. [Rule 1 - Bug] Fingerprint Display trait not implemented**
- **Found during:** Task 2 (test verification)
- **Issue:** Test used `fingerprint.to_string()` but `Fingerprint` type does not implement Display
- **Fix:** Changed to use Debug format: `format!("{:?}", fingerprint)`
- **Files modified:** node-core/crates/p2p/src/cert.rs
- **Commit:** dc36778

## Verification Results

```
cargo check -p nsn-p2p: SUCCESS (no errors)
cargo test -p nsn-p2p cert: 5 passed
cargo test -p nsn-p2p config: 3 passed (including new test_webrtc_config_fields)
Module exports: CertError, CertificateManager exported from lib.rs
Config fields: enable_webrtc, webrtc_port, data_dir, external_address present
```

## Next Phase Readiness

**Ready for 01-02 (WebRTC Transport Setup):**
- libp2p-webrtc dependency available
- CertificateManager ready to provide certificates to transport builder
- P2pConfig has all fields needed for WebRTC transport configuration

**No blockers identified.**
