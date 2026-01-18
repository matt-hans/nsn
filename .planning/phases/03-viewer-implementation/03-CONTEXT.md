# Phase 3: Viewer Implementation — Context

**Phase:** 3 of 7
**Goal:** Transform viewer into a lightweight P2P node that connects directly to mesh
**Created:** 2026-01-18
**Status:** Ready for research/planning

---

## Connection UX

### Bootstrap Experience (Two-Phase)

**Phase 1 — Full-Screen Bootstrapping Overlay:**
- Terminal-style or sleek progress bar aesthetic
- Progressive messaging:
  - 0-5s: "Connecting to Swarm..."
  - 5-15s: "Negotiating NAT traversal..."
  - 15-30s: "Finding alternative routes..."
- Hard error at 30s: "Unable to join the Neural Sovereign Network" with "Try Manual Bootstrap" and "Retry" buttons
- Treat like a video game loading screen, not a website spinner

**Phase 2 — Content Loading:**
- Once mesh joined, overlay vanishes
- Skeleton loader (gray pulse) in video container when selecting a slot
- Text: "Streaming Slot #12345"

### Persistent Network Health Widget

**Location:** Top-right corner, always visible

**States:**
| State | Visual | Text |
|-------|--------|------|
| Healthy | Green dot | "Mesh Active (5 peers)" |
| Degraded | Yellow dot | "Low Peers" |
| Disconnected | Red dot | "Disconnected" |

**Hover Tooltip (expanded details):**
```
Connected Node: 12D3K... (Validator)
Latency: 45ms
Protocol: WebRTC
```

### Reconnection Behavior

**Visual Treatment:**
- Last valid frame remains visible
- Frame desaturates (black & white) or blurs slightly
- Semi-transparent modal overlay in center

**Messaging:**
- Text: "Signal Lost. Re-routing through Mesh..."
- After 5s of failed reconnection: Show "Select different node" button

**Rationale:** Freezing confirms app is running; clearing to black feels like a crash.

---

## Discovery Behavior

### Tiered Configuration

Discovery URLs checked in priority order:

| Priority | Source | Purpose |
|----------|--------|---------|
| 1 | `localStorage.last_known_node` | Persistence/anti-censorship |
| 2 | User settings (custom node) | Sovereignty/advanced users |
| 3 | `import.meta.env.VITE_BOOTSTRAP_NODES` | Dev/CI configuration |
| 4 | Hardcoded defaults | Foundation fallback |

### Parallel Race Pattern

**Algorithm:**
1. Build candidate list from tiered sources
2. Shuffle hardcoded defaults (avoid hammering first node)
3. Take batch of 3 random candidates
4. Fire `fetch('/p2p/info')` to all 3 simultaneously
5. First valid response wins, cancel others
6. If all 3 fail, take next batch of 3
7. Repeat until success or exhausted

**Timeout:** 3 seconds per request

### WebRTC Requirement

**If discovery returns no WebRTC addresses:**
- Treat as failure (equivalent to 500 error)
- Log warning: "Node X is online (TCP) but WebRTC is disabled/not ready"
- Immediately try next batch
- Do NOT retry same node with backoff

### Last Known Good Persistence

**On successful connection:**
- Save node URL to `localStorage.last_known_node`

**On next launch:**
- Insert last known good at top of candidate list
- Enables recovery if hardcoded defaults become unreachable

---

## Error Presentation

### Connection Errors (Total Failure)

**Primary UI:**
- Clean, centered message: "Unable to join the Neural Sovereign Network. Please check your internet connection."
- "Retry" button
- "Try Manual Bootstrap" button (opens settings)

**Expandable Diagnostics:**
- Small link below: `[+] View Network Diagnostics`
- Expands to show raw log:
```
[Error] Bootstrap Node A (US-East): Connection Refused (TCP 9615)
[Error] Bootstrap Node B (EU-West): Timeout (3000ms)
[Info]  Local Network: Online
[Fail]  Critical: No entry nodes reachable.
```

**Rationale:** Respects user intelligence without cluttering interface. Enables easy screenshot for support.

### Actionable Diagnostics (Heuristic-Based)

**Known Pattern — Firewall/VPN Detection:**
- Trigger: HTTP `/p2p/info` succeeds, but WebRTC connection fails/timeouts
- Diagnosis: UDP blocking (corporate firewall, VPN)
- Message: "Network Firewall Detected."
- Tip: "Your internet provider seems to be blocking P2P (UDP) traffic. Try disabling your VPN or switching networks."

**Rationale:** Specific advice based on failure type solves 80% of support tickets.

### Transient Errors (Streaming Glitches)

**Default Behavior (Silent Recovery):**
- Dropped chunks or buffer <2s: Show nothing
- App self-heals using buffer
- No user disruption for minor network blips

**Power User View — "Stats for Nerds":**
- Access: Right-click context menu → "Stats for Nerds"
- Semi-transparent overlay showing:
  - Buffer Health: `4.2s`
  - Dropped Frames: `12`
  - GossipSub Mesh: `6 peers`
  - Throughput: `2.4 MB/s`

### Connected But No Data ("The Waiting Room")

**Use chain context to explain why:**

| Network State | Message |
|---------------|---------|
| Generating | "Director is synthesizing video... (Epoch 105, Slot 3)" |
| Idle | "Network Idle. Waiting for next proposal." |
| Consensus | "Validators verifying output... (3/5 signatures)" |

**Rationale:** Shifts anxiety from "app broken" to "network busy working" — positive "alive" feeling.

---

## Mock Removal Strategy

**Approach:** Direct replacement, no feature flags

- Remove all mock video generators
- Remove hardcoded video chunks
- Remove fake peer connections
- If no real data arrives, show "Waiting Room" state with chain context (not mock data)

---

## Deferred Ideas

None captured during discussion.

---

## Technical Constraints (From Roadmap)

These are fixed by Phase 3 deliverables:

- **Dependencies:** js-libp2p, @libp2p/webrtc, @chainsafe/libp2p-noise, @chainsafe/libp2p-gossipsub, @polkadot/types-codec, @multiformats/multiaddr
- **P2P Client Location:** `viewer/src/services/p2pClient.ts`
- **Video Topic:** `/nsn/video/1.0.0`
- **SCALE Decoding:** VideoChunk type via @polkadot/types registry

---

## Summary for Downstream Agents

| Area | Decision |
|------|----------|
| Bootstrap UX | Two-phase: full-screen overlay → skeleton loader |
| Timeout progression | 5s → 15s → 30s with changing messages |
| Network indicator | Persistent top-right widget (green/yellow/red) |
| Reconnection | Freeze + desaturate frame, "Signal Lost" overlay |
| Discovery priority | localStorage → settings → env → hardcoded |
| Discovery algorithm | Parallel race, batch of 3, first wins |
| No WebRTC response | Treat as failure, try next batch |
| Persistence | Save last known good to localStorage |
| Error display | Friendly + expandable diagnostics |
| Actionable errors | Heuristic-based (firewall detection) |
| Transient errors | Silent recovery, "Stats for Nerds" on right-click |
| No data state | Chain context ("Director synthesizing...", etc.) |
| Mock removal | Direct replacement, no flags |
