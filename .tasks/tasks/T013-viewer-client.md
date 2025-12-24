---
id: T013
title: Viewer Client Application (Tauri Desktop App)
status: pending
priority: 2
agent: fullstack
dependencies: [T009, T011, T012]
blocked_by: []
created: 2025-12-24T00:00:00Z
updated: 2025-12-24T00:00:00Z
tags: [off-chain, viewer, tauri, react, ui, phase2]

context_refs:
  - context/project.md
  - context/architecture.md
  - context/acceptance-templates.md

docs_refs:
  - docs/prd.md#section-4.1
  - docs/architecture.md#section-5.1

est_tokens: 14000
actual_tokens: null
---

## Description

Implement the Viewer Client Application, ICN's end-user interface for consuming AI-generated video streams. Built with Tauri 2.0 (Rust backend + React 18 frontend), the viewer provides a native desktop experience with hardware-accelerated video playback, P2P connectivity, and optional content seeding for ICN token rewards.

The Viewer Client performs six core functions:
1. **Stream Discovery**: Discover available streams via Kademlia DHT or hardcoded bootstrap
2. **Content Playback**: Decode and play video streams using WebCodecs (hardware-accelerated)
3. **P2P Connectivity**: Connect to Regional Relays or Super-Nodes via libp2p-js (QUIC/WebTransport)
4. **Adaptive Bitrate**: Adjust quality based on bandwidth (1080p → 720p → 480p)
5. **Optional Seeding**: Re-share watched content to other viewers (earn small ICN rewards)
6. **Wallet Integration**: MetaMask-style wallet for ICN token display and staking (future)

Viewers operate permissionlessly (no stake required for consumption), represent the largest node type by count (target: 10,000+ viewers at scale), and are the primary interface for ICN's "endless stream" UX vision.

**Technical Approach:**
- Tauri 2.0 for cross-platform native app (macOS, Windows, Linux)
- React 18.x with TypeScript for UI
- Zustand 4.x for state management (lightweight Redux alternative)
- WebCodecs API for hardware-accelerated video decode (AV1, VP9, H.264)
- libp2p-js 1.x for browser-compatible P2P (WebTransport, WebRTC fallback)
- Wasm-based crypto (ed25519, sha256) for optional seeding signatures

**Integration Points:**
- Queries Kademlia DHT for relay/Super-Node multiaddrs (via libp2p-js)
- Connects to Regional Relay (T012) or Super-Node (T011) via WebTransport
- Fetches video chunks in HLS-like manifest format
- Optional: publishes seeding availability to DHT

## Business Context

**User Story:** As a viewer, I want a desktop application that automatically discovers video streams, plays them with low latency (<45s glass-to-glass), and optionally lets me earn ICN tokens by seeding content, so that I can enjoy endless AI-generated content without needing to understand blockchain or P2P networking.

**Why This Matters:** Viewers are ICN's end users and primary growth metric. A polished, low-friction UX is critical for mainstream adoption. Unlike crypto-native dApps, ICN must hide complexity (P2P, staking, consensus) behind a simple "Watch" button.

**What It Unblocks:**
- User acceptance testing with non-technical audiences
- Viral growth through word-of-mouth (compelling UX = organic sharing)
- Revenue generation (future: premium features, in-app ICN purchasing)
- Community-driven content curation (upvoting, playlists)

**Priority Justification:** Priority 2 (Important) - Required for public mainnet launch but not critical for Moonriver MVP. Initial testnet can use CLI tools or browser DevTools. Viewer app is the "front door" for mainstream users.

## Acceptance Criteria

- [ ] Tauri app compiles and runs on macOS, Windows, Linux (`npm run tauri build`)
- [ ] DHT discovery finds at least one relay or Super-Node within 5 seconds
- [ ] WebTransport connection established to relay (fallback to WebRTC if QUIC blocked)
- [ ] Video manifest fetched (list of chunk CIDs and shard IDs)
- [ ] Video chunks downloaded and buffered (minimum 10 seconds ahead)
- [ ] WebCodecs decoder initialized and decodes first frame within 2 seconds
- [ ] Video playback starts within 10 seconds of app launch (cold start)
- [ ] Playback continues smoothly for 5 minutes without buffering (99% uptime)
- [ ] Adaptive bitrate switches quality based on bandwidth (detect via download speed)
- [ ] UI displays current slot, director info, reputation scores, and playback stats
- [ ] Optional seeding toggle (checkbox in settings) enables re-uploading watched chunks
- [ ] Seeding uploads chunks to other viewers via WebRTC (limited to 10 Mbps upload)
- [ ] App state persists across restarts (last watched slot, volume, quality preference)
- [ ] Graceful shutdown on window close with cleanup of P2P connections
- [ ] Unit tests for state management (Zustand store), chunk buffer logic
- [ ] E2E test: launches app, connects to mock relay, plays 30s of video

## Test Scenarios

**Test Case 1: Cold Start and Stream Discovery**
- Given: Viewer app launched for first time (no cached state)
  And: DHT has 3 relays in viewer's region
- When: App queries DHT for nearby relays
  And: DHT returns relay multiaddrs: [/ip4/1.2.3.4/udp/9003/quic/webtransport]
- Then: App selects closest relay (latency-based ping)
  And: WebTransport connection established within 2 seconds
  And: Connection status in UI: "Connected to NA-WEST relay"

**Test Case 2: Video Playback Start**
- Given: Connected to relay
  And: Relay provides video manifest for current slot 100
- When: App requests chunks: [chunk_0, chunk_1, chunk_2, ...]
  And: Chunks downloaded and added to buffer
- Then: WebCodecs decoder decodes first frame
  And: Video playback starts in <canvas> element
  And: UI displays: "Playing Slot 100 | Director: 0x1234...abcd | FPS: 24"
  And: Audio synced with video

**Test Case 3: Adaptive Bitrate (Bandwidth Degradation)**
- Given: Playback in progress at 1080p (5 Mbps)
  And: Available bandwidth drops to 2 Mbps (simulated network throttle)
- When: Chunk download time exceeds buffer threshold (>3 seconds for 2s chunk)
- Then: App detects slow download
  And: Switches to 720p quality
  And: Requests 720p chunks from relay
  And: UI notification: "Quality: 1080p → 720p (low bandwidth)"
  And: Playback continues without buffering

**Test Case 4: Relay Disconnection and Failover**
- Given: Playback via relay A
  And: Relay A crashes (disconnects WebTransport)
- When: App detects connection loss (keepalive timeout)
- Then: App queries DHT for alternative relays
  And: Connects to relay B or Super-Node directly
  And: Resumes playback from last buffered chunk
  And: Total downtime <5 seconds (transparent to user)

**Test Case 5: Optional Seeding Enabled**
- Given: Viewer has watched 50 chunks (total 250MB)
  And: Seeding enabled in settings
- When: Other viewer requests chunk_5 (app has cached)
  And: Request received via WebRTC
- Then: App uploads chunk_5 to requesting peer
  And: Upload bandwidth capped at 10 Mbps
  And: UI displays: "Seeding: 250 MB uploaded | Earned: 0.05 ICN"

**Test Case 6: App State Persistence**
- Given: Viewer watching slot 200, volume 70%, quality 720p
  And: User closes app window
- When: App shutdown triggered
  And: State persisted to local storage: `{ last_slot: 200, volume: 70, quality: 720 }`
- Then: After restart, app resumes slot 200 at 70% volume, 720p quality

**Test Case 7: WebCodecs Decode Failure (Unsupported Codec)**
- Given: Relay sends video chunk encoded with AV1
  And: User's browser/OS doesn't support AV1 hardware decode
- When: WebCodecs decoder initialization attempts AV1
  And: Initialization fails
- Then: App falls back to VP9 (request different codec from relay)
  Or: If VP9 also unsupported, display error: "Your device doesn't support required video codecs. Please update your browser/OS."

**Test Case 8: DHT Discovery Timeout (No Relays Found)**
- Given: DHT query for relays returns no results (network isolation or early testnet)
- When: 5-second timeout expires
- Then: App falls back to hardcoded Super-Node list (bootstrap fallback)
  And: Connects directly to Super-Node
  And: Warning in UI: "P2P discovery unavailable, using fallback"

## Technical Implementation

**Required Components:**

**Tauri Backend (Rust):**
- `src-tauri/src/main.rs` - Tauri app entrypoint, window setup, IPC handlers
- `src-tauri/src/p2p.rs` - libp2p-rust bridge (if needed for native performance), or defer to JS
- `src-tauri/src/storage.rs` - Local storage for app state (SQLite or JSON file)
- `src-tauri/src/commands.rs` - Tauri commands for frontend IPC (get_relays, fetch_chunk, etc.)

**React Frontend (TypeScript):**
- `src/App.tsx` - Main React component, routing (if multi-page)
- `src/components/VideoPlayer.tsx` - WebCodecs-based video player component
- `src/components/StreamInfo.tsx` - Display slot, director, reputation, stats
- `src/components/Settings.tsx` - User preferences (seeding, quality, volume)
- `src/store/appStore.ts` - Zustand state management (playback state, P2P peers, settings)
- `src/services/p2p.ts` - libp2p-js integration (DHT, WebTransport, WebRTC)
- `src/services/videoBuffer.ts` - Chunk buffering logic, adaptive bitrate
- `src/services/webcodecs.ts` - WebCodecs decoder wrapper, frame rendering

**Validation Commands:**
```bash
# Install dependencies
npm install

# Run dev mode (hot reload)
npm run tauri dev

# Build production app (creates .dmg, .exe, .AppImage)
npm run tauri build

# Run frontend tests (Jest + React Testing Library)
npm test

# Type checking
npx tsc --noEmit

# Linting
npm run lint

# E2E tests (Playwright)
npm run test:e2e
```

**Code Patterns:**
```typescript
// Zustand store for app state
import create from 'zustand';

interface AppState {
  currentSlot: number;
  playbackState: 'idle' | 'buffering' | 'playing' | 'paused';
  quality: '1080p' | '720p' | '480p';
  connectedRelay: string | null;
  seedingEnabled: boolean;

  setCurrentSlot: (slot: number) => void;
  setPlaybackState: (state: AppState['playbackState']) => void;
  setQuality: (quality: AppState['quality']) => void;
}

export const useAppStore = create<AppState>((set) => ({
  currentSlot: 0,
  playbackState: 'idle',
  quality: '1080p',
  connectedRelay: null,
  seedingEnabled: false,

  setCurrentSlot: (slot) => set({ currentSlot: slot }),
  setPlaybackState: (state) => set({ playbackState: state }),
  setQuality: (quality) => set({ quality }),
}));

// WebCodecs video decoder
export class VideoDecoder {
  private decoder: any;
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d')!;

    this.decoder = new window.VideoDecoder({
      output: (frame: VideoFrame) => this.renderFrame(frame),
      error: (e: Error) => console.error('Decode error:', e),
    });
  }

  async init(codec: string) {
    const config = {
      codec, // e.g., 'av01.0.05M.08' for AV1
      optimizeForLatency: true,
    };

    const support = await window.VideoDecoder.isConfigSupported(config);
    if (!support.supported) {
      throw new Error(`Codec ${codec} not supported`);
    }

    this.decoder.configure(config);
  }

  decode(chunk: EncodedVideoChunk) {
    this.decoder.decode(chunk);
  }

  private renderFrame(frame: VideoFrame) {
    this.ctx.drawImage(frame, 0, 0, this.canvas.width, this.canvas.height);
    frame.close();
  }
}

// libp2p-js DHT discovery
import { createLibp2p } from 'libp2p';
import { kadDHT } from '@libp2p/kad-dht';
import { webTransport } from '@libp2p/webtransport';

export async function discoverRelays(): Promise<Multiaddr[]> {
  const node = await createLibp2p({
    transports: [webTransport()],
    dht: kadDHT(),
  });

  await node.start();

  // Query DHT for relays in viewer's region
  const region = 'NA-WEST'; // Could detect via GeoIP
  const key = `/icn/relays/${region}`;

  const results = [];
  for await (const event of node.dht.get(new TextEncoder().encode(key))) {
    if (event.name === 'VALUE') {
      const addrs = JSON.parse(new TextDecoder().decode(event.value));
      results.push(...addrs);
    }
  }

  await node.stop();
  return results.map((a: string) => multiaddr(a));
}

// Adaptive bitrate logic
export class AdaptiveBitrateController {
  private downloadSpeeds: number[] = [];
  private currentQuality: '1080p' | '720p' | '480p' = '1080p';

  recordDownloadSpeed(bytes: number, durationMs: number) {
    const mbps = (bytes * 8) / (durationMs / 1000) / 1_000_000;
    this.downloadSpeeds.push(mbps);

    // Keep last 10 samples
    if (this.downloadSpeeds.length > 10) {
      this.downloadSpeeds.shift();
    }

    this.adjustQuality();
  }

  private adjustQuality() {
    const avgSpeed = this.downloadSpeeds.reduce((a, b) => a + b, 0) / this.downloadSpeeds.length;

    if (avgSpeed < 2 && this.currentQuality !== '480p') {
      this.currentQuality = '480p';
      console.log('Switched to 480p (low bandwidth)');
    } else if (avgSpeed >= 2 && avgSpeed < 5 && this.currentQuality !== '720p') {
      this.currentQuality = '720p';
      console.log('Switched to 720p (medium bandwidth)');
    } else if (avgSpeed >= 5 && this.currentQuality !== '1080p') {
      this.currentQuality = '1080p';
      console.log('Switched to 1080p (high bandwidth)');
    }
  }

  getCurrentQuality(): string {
    return this.currentQuality;
  }
}
```

## Dependencies

**Hard Dependencies** (must be complete first):
- [T009] Director Node - Directors generate content for viewers to watch
- [T011] Super-Node - Fallback if relays unavailable
- [T012] Regional Relay - Primary content source for viewers

**Soft Dependencies** (nice to have):
- [T007] pallet-icn-treasury - For displaying ICN rewards earned from seeding
- Wallet integration (MetaMask API) - For future ICN token interaction

**External Dependencies:**
- Tauri 2.0 (macOS 11+, Windows 10+, Ubuntu 20.04+)
- Node.js 18+ and npm
- Browser with WebCodecs support (Chrome 94+, Edge 94+, Safari 16.4+)
- libp2p-js 1.x and dependencies

## Design Decisions

**Decision 1: Tauri instead of Electron**
- **Rationale:** Tauri uses native WebView (WebKit/WebView2/GTK) instead of bundling Chromium. Result: 10× smaller binary (~5MB vs ~50MB), lower memory usage, faster startup.
- **Alternatives:**
  - Electron: Larger binary, higher resource usage, but more mature ecosystem
  - Native (Swift/Kotlin/C++): Best performance, but 3× development effort
- **Trade-offs:** (+) Lightweight, fast. (-) WebView differences across platforms (more testing needed).

**Decision 2: WebCodecs instead of HTML5 `<video>` element**
- **Rationale:** WebCodecs provides direct access to hardware decoders, lower latency (no browser buffering overhead), frame-level control for adaptive bitrate.
- **Alternatives:**
  - HTML5 `<video>` + HLS.js: Simpler but higher latency, less control
  - Canvas + manual decode (ffmpeg.wasm): CPU-intensive, no hardware acceleration
- **Trade-offs:** (+) Best latency and quality. (-) Requires modern browser (Chrome 94+).

**Decision 3: libp2p-js instead of native Rust libp2p in Tauri**
- **Rationale:** libp2p-js supports WebTransport and WebRTC (browser-friendly). Rust libp2p has better performance but harder to bridge to browser context.
- **Alternatives:**
  - Rust libp2p via Tauri IPC: Better performance, but complex IPC layer
  - Custom WebSocket relay: Centralization, defeats P2P purpose
- **Trade-offs:** (+) Browser-compatible, standard P2P. (-) Slightly higher CPU usage than native.

**Decision 4: Zustand instead of Redux for state management**
- **Rationale:** Zustand is lightweight (2KB), simpler API (no actions/reducers boilerplate), TypeScript-first. Perfect for app-level state (not complex like Redux DevTools scenarios).
- **Alternatives:**
  - Redux: More features, larger ecosystem, but overkill for viewer app
  - React Context: No persistence, harder to optimize re-renders
- **Trade-offs:** (+) Simple, fast. (-) Less tooling (no time-travel debugging).

**Decision 5: Optional seeding instead of mandatory**
- **Rationale:** Some users on metered connections or mobile hotspots can't afford upload bandwidth. Making seeding optional maximizes accessibility while still incentivizing contribution.
- **Alternatives:**
  - Mandatory seeding (BitTorrent-style): Better network health, but excludes users
  - No seeding: Simpler, but relays bear all distribution cost
- **Trade-offs:** (+) Inclusive, ethical. (-) Tragedy of the commons (freeloaders).

## Risks & Mitigations

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| WebCodecs not supported (older browsers) | High (no video playback) | Medium | Detect at startup, show error with browser upgrade link. Fallback to HLS.js + HTML5 `<video>` for degraded mode. |
| P2P blocked by corporate firewall | Medium (can't discover relays) | High | Hardcoded Super-Node fallback (centralized but functional). Provide "Direct Connect" input for manual relay address. |
| High memory usage (buffering too much) | Medium (app crashes on low-end devices) | Medium | Limit buffer to 30 seconds (vs 60s). Monitor memory in DevTools, alert user if >500MB. Reduce quality to 480p on <4GB RAM devices. |
| Seeding copyright concerns | Critical (legal risk) | Low | Content policy disclaimer in UI. Allow users to disable seeding. Future: DMCA compliance tools (takedown notifications). |
| DHT spam/Sybil attacks | Medium (viewer connects to malicious nodes) | Medium | Verify relay stake on-chain before connecting (query `pallet-icn-stake`). Implement reputation-based relay selection. |
| Cross-platform WebView inconsistencies | Medium (bugs on specific OS) | High | Extensive testing on macOS, Windows, Linux. Use Tauri's webview abstractions. Document known issues per platform. |

## Progress Log

### [2025-12-24] - Task Created

**Created By:** task-creator agent
**Reason:** User request to create comprehensive off-chain node tasks for ICN project
**Dependencies:** T009 (Director Node), T011 (Super-Node), T012 (Regional Relay)
**Estimated Complexity:** Complex (14,000 tokens) - Full-featured desktop app with video playback, P2P, adaptive bitrate

## Completion Checklist

**Code Complete:**
- [ ] All acceptance criteria met and verified
- [ ] Unit tests pass (Zustand store, buffer logic)
- [ ] E2E tests pass (Playwright: app launch, playback, settings)
- [ ] TypeScript type checking passes (`tsc --noEmit`)
- [ ] Linting passes (ESLint)
- [ ] Cross-platform builds succeed (macOS .dmg, Windows .exe, Linux .AppImage)

**Integration Ready:**
- [ ] App successfully discovers relays via DHT
- [ ] Video playback starts within 10 seconds (cold start)
- [ ] Adaptive bitrate tested (manual bandwidth throttle)
- [ ] Seeding tested (upload to mock peer)
- [ ] App state persists across restarts

**Production Ready:**
- [ ] Tested on macOS 11+, Windows 10+, Ubuntu 20.04+
- [ ] Resource usage profiled (<300MB RAM, <5% CPU idle)
- [ ] Error paths tested (DHT timeout, codec unsupported, relay disconnect)
- [ ] User documentation written (installation, usage, troubleshooting)
- [ ] Privacy policy added (seeding disclosure, data collection)
- [ ] App signed and notarized (macOS), code-signed (Windows)
- [ ] Auto-updater configured (Tauri built-in updater)

**Definition of Done:**
Task is complete when viewer app is installed and used by 10+ beta testers for 7 days, achieves >95% playback success rate (no buffering), <10s cold start time, and receives positive UX feedback (NPS >8) indicating readiness for public launch.
