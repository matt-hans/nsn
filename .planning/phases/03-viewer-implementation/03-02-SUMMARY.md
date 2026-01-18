---
phase: 03-viewer-implementation
plan: 02
subsystem: p2p-discovery
tags: [webrtc, libp2p, scale-codec, http-discovery, @polkadot/types]

# Dependency graph
requires:
  - phase: 03-01
    provides: P2PClient with WebRTC-Direct transport
  - phase: 02-01
    provides: HTTP /p2p/info endpoint (P2pInfoResponse format)
provides:
  - HTTP-based node discovery service with parallel race pattern
  - SCALE VideoChunk decoder matching Rust struct definition
  - Tiered configuration priority (localStorage → settings → env → hardcoded)
  - WebRTC multiaddr extraction with certhash validation
affects:
  - 03-03 (Video Streaming Protocol)
  - 03-04 (Chain RPC Integration)
  - 03-05 (Integration Testing)

# Tech tracking
tech-stack:
  added:
    - @polkadot/types TypeRegistry for SCALE decoding
  patterns:
    - Parallel race pattern for discovery (3-node batches)
    - TypeRegistry singleton (module-level, not per-decode)
    - localStorage persistence for last known good node
    - 503 retry handling for node initialization

key-files:
  created:
    - viewer/src/services/discovery.ts (HTTP discovery, 258 lines)
    - viewer/src/services/videoCodec.ts (SCALE decoder, 208 lines)

key-decisions:
  - "TypeRegistry created once at module level - NOT per decode (prevents memory leak)"
  - "Non-WebRTC responses treated as failure (per CONTEXT.md - immediately try next batch)"
  - "Hardcoded defaults shuffled to avoid hammering first node"

patterns-established:
  - "Pattern: Discovery tiered priority - localStorage → settings → env → hardcoded"
  - "Pattern: Parallel batch processing with race and timeout (default 3000ms)"
  - "Pattern: SCALE field order must match Rust struct exactly (position-based encoding)"

# Metrics
duration: ~3min
completed: 2026-01-18
---

# Phase 3 Plan 02: Discovery Client Summary

**HTTP-based node discovery with parallel race pattern and SCALE VideoChunk decoder matching Rust struct**

## Performance

- **Duration:** ~3 minutes
- **Started:** 2026-01-18T06:42:39Z
- **Completed:** 2026-01-18T06:45:41Z
- **Tasks:** 2
- **Files created:** 2

## Accomplishments

- Created `discovery.ts` service with HTTP-based node lookup via `/p2p/info` endpoint
- Implemented parallel race pattern with 3-node batches and configurable timeout
- Created `videoCodec.ts` with TypeRegistry-based SCALE decoder for VideoChunk
- WebRTC multiaddr extraction with certhash validation and peer ID appending
- Tiered configuration priority with localStorage persistence for last known good node

## Task Commits

Each task was committed atomically:

1. **Task 1: Create discovery service** - `2b77b61` (feat)
2. **Task 2: Create SCALE VideoChunk codec** - `2b77b61` (feat)

**Plan metadata:** (not yet created)

_Note: Both tasks committed in single commit as they were completed together_

## Files Created/Modified

### Created

- `viewer/src/services/discovery.ts` - HTTP-based node discovery service
  - `discoverNode(baseUrl)`: Fetch `/p2p/info`, extract WebRTC multiaddr with certhash
  - `discoverWithRace(candidates, batchSize, timeoutMs)`: Parallel batch discovery
  - `buildCandidateList()`: Tiered configuration (localStorage → settings → env → hardcoded)
  - Handles 503 for node initialization retry
  - Shuffles hardcoded defaults to avoid hammering first node
  - Saves successful node to `localStorage.last_known_node`

- `viewer/src/services/videoCodec.ts` - SCALE VideoChunk decoder
  - TypeRegistry created once at module level (prevents memory leak)
  - `decodeVideoChunk(data)`: Decodes SCALE binary to TypeScript object
  - Field order matches Rust VideoChunk struct exactly (version, slot, content_id, etc.)
  - `validateVideoChunk(chunk)`: Optional helper for chunk validation
  - Handles Bytes to Uint8Array conversion and hex parsing

### Key Types

- `DiscoveryCandidate` interface: `{ url: string, source: "localStorage" | "settings" | "env" | "hardcoded" }`
- `DecodedVideoChunk` interface: Full chunk metadata with camelCase fields
- `P2pInfoResponse` interface: Matches Rust response format

## Decisions Made

1. **TypeRegistry singleton pattern**: Created once at module level, NOT per decode. Creating per decode would cause memory leak and performance issues.

2. **Non-WebRTC = failure**: Per CONTEXT.md, nodes without WebRTC addresses are treated as failure (equivalent to 500 error) and caller immediately tries next batch. This prevents wasting time on TCP-only nodes.

3. **Hardcoded shuffling**: To avoid hammering the first hardcoded node, hardcoded defaults are shuffled before being appended to candidate list. Non-hardcoded sources (localStorage, env) preserve priority order.

4. **503 retry handling**: Node returns 503 while swarm is initializing (~5 seconds). Discovery returns `null` (not throwing) so caller can retry with backoff.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

1. **pnpm not in PATH**: pnpm was installed but not available in PATH during execution. Worked around by using `npx pnpm` for all commands.

2. **Pre-existing TypeScript errors**: Build fails due to errors in existing `p2p.ts` (simple-peer dependency removed) and `signaling.test.ts`. These are pre-existing issues not caused by this plan's changes. The new files themselves compile correctly.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

**Ready for Phase 3 Plan 03 (Video Streaming Protocol):**

- Discovery service can find WebRTC nodes via HTTP
- SCALE codec can decode VideoChunk messages from GossipSub
- P2PClient from 03-01 can dial WebRTC multiaddrs
- Next step: Integrate discovery + P2PClient + videoCodec for video streaming

**Blockers/Concerns:**

- None

**Ready for Integration:**

The three services (discovery.ts, videoCodec.ts, p2pClient.ts) can now be wired together:
1. Use `buildCandidateList()` + `discoverWithRace()` to find node
2. Use `P2PClient.initialize()` + `dial(multiaddr)` to connect
3. Use `subscribe()` + `decodeVideoChunk()` to receive video chunks

---
*Phase: 03-viewer-implementation*
*Completed: 2026-01-18*
