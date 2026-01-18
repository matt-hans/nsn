# Phase 2 Context: Discovery Bridge (HTTP Sidecar)

**Created:** 2026-01-18
**Phase Goal:** Provide HTTP endpoint for browsers to discover WebRTC address

---

## Decision 1: Discovery URL Configuration

**Pattern:** Client-side bootstrapping (no central gateway)

### Development
- Default URL: `http://127.0.0.1:9615`
- Docker Compose maps port `9615:9615` to host

### Production
- Hardcoded bootstrap list shipped with viewer
- List of trusted public nodes: `["https://node1.nsn.network:9615", ...]`
- Shuffle list before iterating (prevent thundering herd)
- Sequential try with 2s timeout per node
- Once connected to mesh, DHT provides additional peers

### Node Selection Logic
- Client-side active probing (no centralized load balancer)
- Algorithm: shuffle → iterate → first successful response wins

### Configuration Tiers (Priority Order)
1. `localStorage.user_custom_node` — Power user override
2. `import.meta.env.VITE_NSN_BOOTSTRAP_URL` — CI/CD builds
3. Hardcoded defaults in source — Standard users

### Requirements
- CORS enabled on Rust HTTP server (allow `*` or specific origins)
- HTTPS required for web-hosted viewer (mixed content)
- HTTP acceptable for Tauri desktop app

---

## Decision 2: Error Responses

**Pattern:** Standardized JSON envelope with explicit feature flags

### Response Schema

**Success (200):**
```json
{
  "success": true,
  "data": {
    "peer_id": "12D3Koo...",
    "multiaddrs": ["/ip4/.../udp/9003/webrtc/certhash/..."],
    "features": {
      "webrtc_enabled": true,
      "role": "director"
    }
  }
}
```

**Error (4xx/5xx):**
```json
{
  "success": false,
  "error": {
    "code": "SNAKE_CASE_CODE",
    "message": "Human readable description",
    "details": {}
  }
}
```

### Specific Cases

| Condition | Status | Code | Behavior |
|-----------|--------|------|----------|
| Swarm not initialized | 503 | `NODE_INITIALIZING` | Include `Retry-After: 5` header |
| WebRTC not bound yet | 200 | - | Return partial data, client retries if `webrtc_enabled: true` but no `/webrtc/` address |
| WebRTC disabled | 200 | - | `features.webrtc_enabled: false`, client tries different node |

### Client Logic
- 503 → RetryableError (back off, retry)
- 200 + `webrtc_enabled: false` → FatalError (try different node)
- 200 + `webrtc_enabled: true` + no address → RetryableError (race condition, retry in 1s)
- 200 + valid address → Connect

---

## Decision 3: Response Caching

**Pattern:** No caching — freshness over efficiency

### Server-Side
- **No caching** — Always query swarm directly
- Rationale: `swarm.listeners()` is instant in-memory operation
- Risk of caching: stale certhash after node restart

### HTTP Headers
```
Cache-Control: no-store, max-age=0
```
- Prevents browser and proxy caching
- Critical: certhash can change on restart

### Client-Side Persistence
- **Optimistic last-known-good** strategy
- On successful connection: save HTTP URL to `localStorage.last_connected_node`
- On app launch:
  1. Try `last_connected_node` first (2s timeout)
  2. If fails, fall back to bootstrap list
- Creates "sticky" sessions, reduces bootstrap load

### Address Change Detection
- **Reactive, not proactive** — No polling
- Rely on libp2p `connection:close` event
- On disconnect: re-fetch `/p2p/info`, get new address, reconnect
- HTTP endpoint is "signaling server" — only consulted when transport fails

---

## Decision 4: Address Filtering Logic

**Pattern:** Reachability-focused filtering, external address replaces internals

### Localhost (`127.0.0.1`, `::1`)
- **If `--p2p-external-address` set:** Filter out localhost
- **If no external address:** Keep localhost (dev mode)

### Private Networks (RFC1918)
- **Do not filter** `10.x`, `172.16-31.x`, `192.168.x`
- Rationale: Valid for home lab / corporate intranet deployments
- Docker issues solved by `--p2p-external-address`, not code filtering

### External Address Override
- **Replacement mode** — If set, return ONLY external address
- Discard all auto-detected internal IPs
- Rationale: Browsers may race/timeout on unreachable internal IPs

### IPv6
- **Filter:** `fe80::/10` (link-local) — Requires scope ID, useless to remote clients
- **Keep:** `2000::/3` (global unicast), `fc00::/7` (unique local)

### Filtering Algorithm
```
if external_address configured:
    return [external_address + certhash]  # ONLY external
else:
    return swarm.listeners()
        .filter(not link_local_ipv6)
        # Keep localhost, keep RFC1918
```

---

## Implementation Checklist

From these decisions, Phase 2 must implement:

- [ ] `GET /p2p/info` endpoint on existing HTTP server (port 9615)
- [ ] JSON response with `success`, `data.peer_id`, `data.multiaddrs`, `data.features`
- [ ] `features.webrtc_enabled` boolean flag
- [ ] `features.role` field (director/validator)
- [ ] 503 response when swarm not ready (with `Retry-After` header)
- [ ] `Cache-Control: no-store, max-age=0` header
- [ ] CORS headers (`Access-Control-Allow-Origin: *`)
- [ ] Address filtering: external replaces internals, filter IPv6 link-local
- [ ] Integration with swarm state to get current listeners

---

## Out of Scope (Viewer-Side)

These items are captured but belong to Phase 3 (Viewer Implementation):
- Bootstrap list management
- Configuration tier logic
- Last-known-good persistence
- Reconnection on disconnect

---

## Deferred Ideas

None captured during this discussion.

---

*Context v1.0 — Ready for Phase 2 planning*
