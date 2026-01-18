---
phase: 02-discovery-bridge
plan: 01
subsystem: p2p-networking
tags: [http, discovery, webrtc, cors]

dependencies:
  requires: [01-02]
  provides: [http-discovery-endpoint]
  affects: [03-viewer-implementation]

tech-stack:
  added: []
  patterns: [http-state-sharing, atomic-swarm-ready-flag]

key-files:
  created:
    - node-core/crates/p2p/src/discovery.rs
  modified:
    - node-core/crates/p2p/src/service.rs
    - node-core/crates/p2p/src/lib.rs

decisions:
  - id: discovery-response-format
    description: "JSON envelope with success/data/error pattern matching CONTEXT.md spec"
  - id: filter-external-override
    description: "When external_address configured, return ONLY that address"
  - id: swarm-ready-via-atomic
    description: "Use AtomicBool for swarm_ready flag, set on first NewListenAddr"

metrics:
  duration: 10m35s
  completed: 2026-01-18
---

# Phase 2 Plan 01: P2P Discovery HTTP Endpoint Summary

HTTP discovery endpoint for browsers to discover WebRTC connection details via `/p2p/info`.

## One-liner

Discovery module with P2pInfoResponse types, address filtering, HTTP `/p2p/info` endpoint with CORS, and 503 handling for swarm startup.

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Create discovery module with P2pInfo types | 2a588bf | discovery.rs, lib.rs |
| 2 | Add /p2p/info endpoint with CORS and 503 | ff07ab1 | service.rs |
| 3 | Add integration tests for discovery endpoint | c764283 | service.rs |

## Key Deliverables

### Discovery Module (`discovery.rs`)
- `P2pInfoResponse` envelope with success/data/error pattern
- `P2pInfoData` with peer_id, multiaddrs, protocols, features
- `P2pFeatures` with webrtc_enabled and role flags
- `filter_addresses()` with external override and IPv6 link-local filtering
- `is_webrtc_address()` and `default_protocols()` utilities
- 10 unit tests for response formats and filtering logic

### HTTP Server Updates (`service.rs`)
- `HttpState` struct for shared swarm state with HTTP server
- `http_listeners`, `http_external_addrs`, `swarm_ready` fields in P2pService
- `serve_http()` function handling both `/metrics` and `/p2p/info`
- 503 response with `Retry-After: 5` header when swarm not ready
- CORS headers (`Access-Control-Allow-Origin: *`)
- `Cache-Control: no-store, max-age=0` for freshness
- SwarmEvent handlers update shared listener/external state

### Integration Tests
- `test_discovery_endpoint_returns_valid_json` - verifies JSON structure
- `test_discovery_endpoint_cors_preflight` - verifies OPTIONS handling
- `test_discovery_endpoint_503_before_ready` - verifies ready state polling

## Test Results

```
running 13 tests (10 discovery + 3 integration)
test discovery::tests::test_default_protocols ... ok
test discovery::tests::test_error_response_format ... ok
test discovery::tests::test_filter_deduplicates ... ok
test discovery::tests::test_filter_external_override ... ok
test discovery::tests::test_filter_keeps_rfc1918 ... ok
test discovery::tests::test_filter_removes_link_local_ipv6 ... ok
test discovery::tests::test_is_webrtc_address ... ok
test discovery::tests::test_node_initializing_error_format ... ok
test discovery::tests::test_success_response_format ... ok
test discovery::tests::test_success_response_has_protocols_field ... ok
test service::tests::test_discovery_endpoint_503_before_ready ... ok
test service::tests::test_discovery_endpoint_cors_preflight ... ok
test service::tests::test_discovery_endpoint_returns_valid_json ... ok
```

## Response Format

**Success (200):**
```json
{
  "success": true,
  "data": {
    "peer_id": "12D3KooW...",
    "multiaddrs": ["/ip4/.../udp/9003/webrtc-direct/certhash/..."],
    "protocols": ["/nsn/video/1.0.0", "/ipfs/id/1.0.0", "/ipfs/ping/1.0.0"],
    "features": {
      "webrtc_enabled": true,
      "role": "node"
    }
  }
}
```

**Error (503 - Swarm not ready):**
```json
{
  "success": false,
  "error": {
    "code": "NODE_INITIALIZING",
    "message": "Swarm not ready, please retry"
  }
}
```
Headers: `Retry-After: 5`

## Deviations from Plan

None - plan executed exactly as written.

## Next Phase Readiness

Phase 3 (Viewer Implementation) can now:
1. Fetch `/p2p/info` from discovery endpoint
2. Parse JSON to extract peer_id and multiaddrs
3. Handle 503 with retry logic using `Retry-After` header
4. Check `features.webrtc_enabled` before connecting
5. Use multiaddrs to establish WebRTC-direct connection

## Commits

- `2a588bf` feat(02-01): add P2P discovery module with types and address filtering
- `ff07ab1` feat(02-01): add /p2p/info HTTP endpoint with CORS and 503 handling
- `c764283` test(02-01): add integration tests for discovery endpoint
