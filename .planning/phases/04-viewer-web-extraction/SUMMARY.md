# Phase 4, Plan 1: Viewer Web Extraction - Summary

**Completed:** 2026-01-09
**Duration:** 1 session
**Tests:** 136 total (19 new for signaling/P2P integration)

## Objective

Extract the React viewer frontend from the Tauri desktop shell into a standalone web application with WebRTC-based P2P video chunk delivery.

## Deliverables

### Core Changes

1. **Package Configuration** - Removed Tauri dependencies, added simple-peer for WebRTC
2. **Browser API Migration** - Replaced Tauri IPC with native browser APIs and localStorage
3. **WebRTC Signaling Client** - WebSocket-based peer discovery with state machine
4. **WebRTC P2P Service** - simple-peer integration for DataChannel video delivery
5. **Development Signaling Server** - Node.js WebSocket server for local testing
6. **Test Infrastructure** - Mock WebSocket, RTCPeerConnection, WebCodecs for jsdom
7. **Vite Build Configuration** - Standalone web deployment with chunk splitting
8. **Integration Tests** - Signaling and P2P service unit tests

### New Files

| File | Purpose |
|------|---------|
| `viewer/src/services/signaling.ts` | WebSocket signaling client (~150 lines) |
| `viewer/src/services/__tests__/signaling.test.ts` | Signaling unit tests (7 tests) |
| `viewer/src/services/__tests__/p2p-service.test.ts` | P2P service unit tests (12 tests) |
| `viewer/scripts/signaling-server.js` | Development signaling server (~140 lines) |

### Modified Files

| File | Changes |
|------|---------|
| `viewer/package.json` | -@tauri-apps/*, +simple-peer, +ws, +signal script |
| `viewer/src/App.tsx` | Removed Tauri invoke, browser API init |
| `viewer/src/components/SettingsModal/index.tsx` | Removed Tauri IPC, Zustand persistence |
| `viewer/src/services/p2p.ts` | Full rewrite: simple-peer + signaling integration |
| `viewer/src/services/p2p.test.ts` | Updated for WebRTC behavior |
| `viewer/src/test/setup.ts` | WebSocket, RTC, WebCodecs mocks |
| `viewer/vite.config.ts` | Web-only build config, chunk splitting |
| `viewer/tsconfig.json` | Added vite/client types |
| `viewer/biome.json` | Ignore src-tauri/gen |

## Technical Details

### WebRTC Architecture

- **Signaling Protocol**: WebSocket with JSON messages (join/leave/offer/answer/ice-candidate)
- **Peer Connections**: simple-peer library for WebRTC DataChannel abstraction
- **NAT Traversal**: Public STUN servers (stun.l.google.com:19302, stun1-4)
- **Video Chunk Format**: 17-byte header [slot:4][chunk_index:4][timestamp:8][is_keyframe:1] + data
- **Max Chunk Size**: 64KB per RFC 8831 recommendation

### Build Output

- **Bundle Size**: ~284KB uncompressed, ~90KB gzipped
- **Chunk Strategy**: vendor (react, react-dom, zustand), p2p (simple-peer)
- **Target**: ES2021 for modern browser compatibility
- **Source Maps**: Enabled for production debugging

## Commits

| Hash | Type | Description |
|------|------|-------------|
| `2c2653f` | chore | Remove Tauri dependencies, add simple-peer for WebRTC |
| `9d535c5` | refactor | Remove Tauri IPC, use browser APIs and fallback relays |
| `42f9feb` | feat | Add WebRTC signaling client for peer discovery |
| `da019f0` | feat | Implement WebRTC P2P service with simple-peer |
| `d1e802e` | feat | Add development signaling server for WebRTC |
| `cfde70b` | test | Update test mocks for web environment |
| `be6a8e4` | chore | Configure Vite for standalone web deployment |
| `70f694b` | test | Add integration tests for signaling and P2P services |

## Verification

- [x] All 136 tests passing
- [x] TypeScript compilation clean
- [x] Biome linting clean
- [x] Production build successful (dist/)
- [x] Signaling server starts and responds to /health
- [x] No Tauri dependencies remain in viewer

## Usage

```bash
# Development
cd viewer
pnpm install
node scripts/signaling-server.js &  # Terminal 1
pnpm dev                              # Terminal 2

# Production build
pnpm build  # Output: dist/
```

## Next Phase

Phase 5: Multi-Node E2E Simulation - Network simulation testing infrastructure
