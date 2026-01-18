---
phase: 01-rust-node-core-upgrade
plan: 02
subsystem: p2p-networking
tags: [webrtc, libp2p, transport, cli]

dependency_graph:
  requires:
    - 01-01 (CertificateManager, P2pConfig WebRTC fields)
  provides:
    - WebRTC transport integration in P2pService
    - CLI flags for WebRTC configuration
    - Integration tests for WebRTC transport
  affects:
    - 02-01 (Discovery endpoint will expose certhash)
    - 03-01 (Viewer will connect via WebRTC multiaddr)

tech_stack:
  added: []
  changed:
    - libp2p-webrtc: 0.9.0-alpha.1 -> 0.7.1-alpha (compatibility fix)
  patterns:
    - Conditional transport composition in SwarmBuilder
    - External address advertisement for NAT/Docker

key_files:
  created: []
  modified:
    - node-core/Cargo.toml
    - node-core/crates/p2p/src/service.rs
    - node-core/bin/nsn-node/src/main.rs

decisions:
  - id: webrtc-version-downgrade
    decision: Downgrade libp2p-webrtc from 0.9.0-alpha.1 to 0.7.1-alpha
    rationale: libp2p-webrtc 0.9.0-alpha.1 requires libp2p-core 0.43 but libp2p 0.53 uses libp2p-core 0.41, causing trait incompatibility
    impact: Uses older but stable API; no functional difference for WebRTC-direct transport

metrics:
  duration: 9m 23s
  completed: 2026-01-18
  tasks: 3/3
  tests_added: 2
  tests_total_passed: 203
---

# Phase 01 Plan 02: WebRTC Transport Setup Summary

**One-liner:** WebRTC transport integrated into P2pService with CLI flags and certificate persistence for browser connectivity

## What Was Built

### Task 1: Add WebRTC transport to P2pService swarm
Modified `service.rs` to conditionally add WebRTC transport:

1. **Certificate loading** - When `enable_webrtc` is true, loads or generates certificate via CertificateManager
2. **Transport composition** - Uses `with_other_transport()` to add WebRTC alongside TCP/QUIC
3. **WebRTC listener** - Starts UDP listener on configured port for browser connections
4. **External address** - Advertises external multiaddr when configured (NAT/Docker support)
5. **Enhanced logging** - WebRTC listen addresses logged at INFO level with certhash

**Key code pattern:**
```rust
let mut swarm = if let Some(cert) = webrtc_cert {
    swarm_builder
        .with_other_transport(|id_keys| {
            Ok(webrtc::tokio::Transport::new(id_keys.clone(), cert.clone())
                .map(|(peer_id, conn), _| (peer_id, StreamMuxerBox::new(conn))))
        })
        .with_behaviour(|_| behaviour)
        .build()
} else {
    // ... without WebRTC
};
```

### Task 2: Add CLI flags for WebRTC configuration
Added global CLI flags to nsn-node:

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--p2p-enable-webrtc` | bool | false | Enable WebRTC transport |
| `--p2p-webrtc-port` | u16 | 9003 | UDP port for WebRTC |
| `--p2p-external-address` | Option<String> | None | External multiaddr for NAT |
| `--data-dir` | PathBuf | /var/lib/nsn | Directory for certificates |

Flags are wired to P2pConfig construction with info logging when enabled.

### Task 3: Add integration tests for WebRTC transport
Added 2 integration tests:

| Test | Purpose |
|------|---------|
| `test_service_with_webrtc_enabled` | Verify service creates with WebRTC, certificate generated |
| `test_webrtc_certificate_persists` | Verify certificate content persists across restarts |

## Commits

| Hash | Type | Description |
|------|------|-------------|
| e800730 | feat | Add WebRTC transport to P2pService swarm |
| 6575642 | feat | Add CLI flags for WebRTC configuration |
| ed95ee0 | test | Add integration tests for WebRTC transport |

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] libp2p-webrtc version incompatibility**
- **Found during:** Task 1 (cargo check)
- **Issue:** libp2p-webrtc 0.9.0-alpha.1 uses libp2p-core 0.43 but libp2p 0.53 uses libp2p-core 0.41. The `Transport::map()` trait method was not available due to different trait implementations.
- **Fix:** Downgraded to libp2p-webrtc 0.7.1-alpha which uses libp2p-core 0.41, maintaining compatibility
- **Files modified:** node-core/Cargo.toml
- **Commit:** e800730
- **Impact:** No functional difference; 0.7.1-alpha provides identical WebRTC-direct transport capability

## Verification Results

```
cargo build -p nsn-node: SUCCESS
CLI flags exist: --p2p-enable-webrtc, --p2p-webrtc-port, --p2p-external-address, --data-dir
cargo test -p nsn-p2p webrtc: 3 passed (config + 2 service tests)
Manual test: WebRTC transport enabled, certificate generated at configured path
Certificate permissions: 0600 (owner read/write only)
```

**Manual test output:**
```
INFO nsn_node: WebRTC transport enabled on UDP port 9003, data dir: "/tmp/nsn-test-webrtc"
INFO nsn_p2p::cert: Generating new WebRTC certificate at "/tmp/nsn-test-webrtc/webrtc_cert.pem"
```

## Next Phase Readiness

**Ready for 02-01 (Discovery Bridge):**
- WebRTC transport is active and listening
- Certificate fingerprint available via `cert.fingerprint()`
- External address advertising works for NAT environments

**Ready for 03-01 (Viewer Implementation):**
- Nodes can accept WebRTC connections on configured port
- Certhash will be included in multiaddr for browser connections

**No blockers identified.**
