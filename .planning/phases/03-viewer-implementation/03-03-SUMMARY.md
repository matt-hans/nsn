---
phase: 03-viewer-implementation
plan: 03
subsystem: Viewer P2P Integration
tags: [libp2p, GossipSub, video-streaming, TypeScript, SCALE-codec]

requires:
  - "03-02"  # Discovery Client

provides:
  - Video topic subscription via GossipSub
  - P2P-to-video pipeline adapter
  - Bitrate calculation from chunk stats

affects:
  - "03-04"  # Chain RPC Integration - will use this subscription

tech-stack:
  added:
    - "@libp2p/webrtc (already present)"
  patterns:
    - GossipSub pub/sub messaging
    - SCALE codec integration
    - Event-driven pipeline architecture

key-files:
  created:
    - "viewer/src/services/types.ts"
  modified:
    - "viewer/src/services/p2pClient.ts"
    - "viewer/src/services/videoPipeline.ts"
    - "viewer/src/services/p2p.ts"
    - "viewer/src/services/videoPipeline.test.ts"
    - "viewer/vite.config.ts"

decisions:
  - "Created shared types.ts to decouple VideoChunkMessage from legacy p2p.ts"
  - "Converted p2p.ts to legacy stub with deprecation warnings"
  - "Disabled legacy tests for p2p-service and signaling modules"

deviations:
  - "**Rule 3 - Blocking Issues Fixed:**"
    - "Created types.ts for shared VideoChunkMessage type - p2p.ts had broken simple-peer import"
    - "Converted p2p.ts to legacy stub - removed SimplePeer dependency references"
    - "Disabled p2p-service.test.ts and signaling.test.ts - test for removed P2PService class"
    - "Updated vite.config.ts manual chunks - removed simple-peer reference, added libp2p and polkadot chunks"
    - "Fixed import in videoPipeline.test.ts - changed from ./p2p to ./types"

metrics:
  duration: "1 session"
  completed: "2026-01-18"
  tests: "0 (existing tests disabled, new tests to be added in later plans)"
---

# Phase 3 Plan 03: Video Streaming Protocol Summary

## One-Liner
GossipSub video subscription with SCALE chunk decoding feeding into existing video pipeline adapter.

## Deliverables

### Task 1: Add video subscription to P2PClient
- **VIDEO_TOPIC constant**: `/nsn/video/1.0.0` exported from p2pClient.ts
- **subscribeToVideoTopic()**: Subscribes to GossipSub topic, delivers Uint8Array to handler
- **unsubscribeFromVideoTopic()**: Cleanup method for subscription
- **stop() updated**: Calls unsubscribe before stopping node

### Task 2: Create P2P chunk adapter for video pipeline
- **types.ts created**: Shared VideoChunkMessage interface
- **connectP2PToPipeline()**: Wires P2PClient to VideoPipeline
  - Decodes SCALE VideoChunk using decodeVideoChunk()
  - Adapts DecodedVideoChunk to VideoChunkMessage format
  - Handles errors silently (logs, doesn't crash pipeline)
- **chunkStats tracking**: Records receivedAt and size for each chunk
- **getBitrateMbps()**: Calculates bitrate over 5-second window
- **getLatencyMs()**: Placeholder (will be implemented in Plan 04)

## Deviations from Plan

### Auto-fixed Issues (Rule 3 - Blocking)

**1. Missing shared type causing import cycle**
- **Found during:** Task 2
- **Issue:** videoPipeline.ts imported VideoChunkMessage from p2p.ts, which had broken simple-peer import
- **Fix:** Created types.ts with shared VideoChunkMessage interface
- **Files modified:** videoPipeline.ts, videoPipeline.test.ts

**2. Broken simple-peer import in p2p.ts**
- **Found during:** Verification build
- **Issue:** p2p.ts still imported simple-peer (removed in Plan 01), blocking compilation
- **Fix:** Converted p2p.ts to legacy stub with deprecation warnings
- **Files modified:** p2p.ts (447 lines → 126 lines)

**3. vite.config.ts referencing simple-peer**
- **Found during:** Build step
- **Issue:** manualChunks had `p2p: ["simple-peer"]` causing build failure
- **Fix:** Replaced with libp2p and polkadot chunks
- **Files modified:** vite.config.ts

**4. Tests for removed P2PService class**
- **Found during:** Compilation
- **Issue:** p2p-service.test.ts and signaling.test.ts tested removed classes
- **Fix:** Disabled both test files with comment blocks
- **Files modified:** p2p-service.test.ts, signaling.test.ts

## Next Phase Readiness

✅ **P2PClient subscribes to /nsn/video/1.0.0**  
✅ **connectP2PToPipeline() decodes and feeds chunks**  
✅ **VideoPipeline tracks chunk stats for bitrate**  
⚠️ **Latency calculation pending (Plan 04)**  
⚠️ **Tests disabled (to be re-enabled in Plan 04)**

### Known Issues
- Legacy p2p.ts functions are stubs with deprecation warnings
- UI components (App.tsx, VideoPlayer) still call legacy discoverRelays/connectToRelay
- Test files disabled pending migration to P2PClient

### Recommendations
- Plan 04 should migrate UI components to use P2PClient + discovery.ts
- Consider adding integration tests for GossipSub subscription
- Add end-to-end test for SCALE codec → pipeline flow
