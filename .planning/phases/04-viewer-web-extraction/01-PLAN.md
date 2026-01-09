# Phase 4, Plan 1: Viewer Web Extraction

## Objective

Extract the React viewer frontend from the Tauri desktop shell into a standalone web application with WebRTC-based P2P video chunk delivery. This enables browser-based access for testnet users without requiring desktop app installation.

## Execution Context

**Files to read:**
- `viewer/src/App.tsx` (Tauri invoke calls)
- `viewer/src/services/p2p.ts` (P2P mock implementation)
- `viewer/src/components/SettingsModal/index.tsx` (Tauri save_settings invoke)
- `viewer/src/store/appStore.ts` (Zustand state with localStorage persist)
- `viewer/package.json` (dependencies)
- `viewer/vite.config.ts` (build configuration)
- `viewer/src-tauri/tauri.conf.json` (Tauri configuration)

**Build commands:**
```bash
cd viewer && pnpm install
cd viewer && pnpm dev           # Standalone web dev server
cd viewer && pnpm build         # Production web build
cd viewer && pnpm test          # Unit tests
cd viewer && pnpm lint          # Biome linting
```

## Context

**Current State:**
- Viewer is a Tauri desktop app (React + Rust backend)
- Only 3 files use `@tauri-apps/api`: App.tsx, SettingsModal/index.tsx, services/p2p.ts
- Zustand store already persists settings to localStorage
- P2P implementation is mocked (returns hardcoded relays, mock success on connect)
- Video pipeline uses WebCodecs (browser-native) and Canvas2D
- All UI components are pure React with no Tauri dependencies

**Tauri Dependencies to Replace:**
1. `invoke('load_settings')` in App.tsx → Read from localStorage (already done by Zustand)
2. `invoke('save_settings', {...})` in SettingsModal → localStorage (already done by Zustand)
3. `invoke('get_relays')` in p2p.ts → Hardcoded fallback list or fetch from signaling server

**WebRTC Integration Approach:**
Based on research, the recommended approach for browser P2P video streaming:
- Use [simple-peer](https://github.com/feross/simple-peer) for WebRTC DataChannel abstraction
- Implement a minimal WebSocket signaling server for peer discovery
- Video chunks delivered via DataChannel binary messages (64KB max chunk size)
- Fallback: Direct HTTP/WebSocket to relay node if P2P fails

**Architecture Summary:**

```
Current (Tauri Desktop)              Target (Web + WebRTC)
┌────────────────────────┐           ┌────────────────────────────────┐
│ React Frontend         │           │ React Frontend (unchanged)     │
│ ↓                     │           │ ↓                              │
│ @tauri-apps/api       │    →      │ localStorage + WebRTC P2P      │
│ ↓                     │           │ ↓                              │
│ Rust Backend (IPC)    │           │ Signaling WebSocket Server     │
│ ↓                     │           │ ↓                              │
│ libp2p networking     │           │ WebRTC DataChannel Mesh        │
└────────────────────────┘           └────────────────────────────────┘
```

**Scope Decision:**
This plan focuses on **extraction and basic WebRTC P2P** (MVP). Advanced features deferred:
- Full libp2p-webrtc integration (defer to Phase 5+ when relay infrastructure ready)
- STUN/TURN server deployment (use free public STUN servers for MVP)
- Advanced peer selection algorithms (simple first-available for MVP)

## Tasks

### Task 1: Create Web-Only Package Configuration

Create package.json for standalone web build without Tauri dependencies.

**Files to modify:**
- `viewer/package.json` - Remove Tauri dependencies, keep React/Vite/Zustand
- `viewer/vite.config.ts` - Simplify for web-only build
- `viewer/tsconfig.json` - Ensure web target

**Implementation:**

package.json changes:
```json
{
  "name": "icn-viewer-web",
  "version": "0.1.0",
  "description": "ICN Viewer Client - Web app for watching interdimensional cable",
  "dependencies": {
    "react": "^18.3.0",
    "react-dom": "^18.3.0",
    "zustand": "^4.5.0",
    "simple-peer": "^9.11.1"
  },
  "devDependencies": {
    "@types/simple-peer": "^9.11.8",
    // ... keep existing dev deps, remove @tauri-apps/cli
  }
}
```

vite.config.ts changes:
- Remove Tauri dev server integration
- Set base path for deployment
- Keep existing test configuration

**Acceptance criteria:**
- [ ] `pnpm install` succeeds without Tauri
- [ ] `pnpm dev` starts standalone Vite dev server on :5173
- [ ] `pnpm build` produces `dist/` for static hosting
- [ ] No TypeScript errors with simple-peer types

**Checkpoint:** `pnpm dev` starts without errors

### Task 2: Replace Tauri IPC with Browser APIs

Remove `@tauri-apps/api` imports and replace with localStorage operations.

**Files to modify:**
- `viewer/src/App.tsx` - Remove invoke, use Zustand directly
- `viewer/src/components/SettingsModal/index.tsx` - Remove invoke, Zustand handles persistence
- `viewer/src/services/p2p.ts` - Remove invoke, return hardcoded relays

**Implementation:**

App.tsx changes:
```typescript
// REMOVE: import { invoke } from "@tauri-apps/api/core";

// BEFORE: const settings = await invoke<{...}>("load_settings");
// AFTER: Settings already loaded from localStorage by Zustand persist middleware

// Simplify init to just relay discovery:
useEffect(() => {
  const init = async () => {
    setConnectionStatus("connecting");
    const relays = await discoverRelays();
    if (relays.length > 0) {
      setConnectedRelay(relays[0].peer_id, relays[0].region);
      setConnectionStatus("connected");
    } else {
      setConnectionStatus("error");
    }
  };
  init();
}, [setConnectionStatus, setConnectedRelay]);
```

SettingsModal changes:
```typescript
// REMOVE: import { invoke } from "@tauri-apps/api/core";

// BEFORE: await invoke("save_settings", { settings: {...} });
// AFTER: Remove - Zustand persist middleware auto-saves to localStorage
const handleSave = () => {
  // Settings already persisted by Zustand
  toggleSettings();
};
```

p2p.ts changes:
```typescript
// REMOVE: import { invoke } from "@tauri-apps/api/core";

// BEFORE: const relays = await invoke<RelayInfo[]>("get_relays");
// AFTER: Return hardcoded fallback or fetch from signaling server

const FALLBACK_RELAYS: RelayInfo[] = [
  {
    peer_id: "12D3KooWDpJ7As7BWAwRMfu1VU2WCqNjvq387JEYKDBj4kx6nXTN",
    multiaddr: "/dns4/relay1.icn.network/tcp/4001/wss",
    region: "us-east",
    is_fallback: true,
  },
  // ... more fallback relays
];

export async function discoverRelays(): Promise<RelayInfo[]> {
  // Try signaling server first (when available)
  try {
    const response = await fetch('https://signal.icn.network/relays');
    if (response.ok) {
      return await response.json();
    }
  } catch {
    console.warn("Signaling server unavailable, using fallback relays");
  }
  return FALLBACK_RELAYS;
}
```

**Acceptance criteria:**
- [ ] No imports from `@tauri-apps/api` in codebase
- [ ] App loads settings from localStorage on startup
- [ ] Settings persist across page refresh
- [ ] Relay discovery works with fallback list
- [ ] All existing tests pass

**Checkpoint:** `pnpm test` passes, app loads in browser

### Task 3: Implement WebRTC Signaling Client

Create signaling client for WebRTC peer connection establishment.

**Files to create:**
- `viewer/src/services/signaling.ts` - WebSocket signaling client

**Implementation:**
```typescript
// Signaling protocol for WebRTC peer discovery
// Connects to signaling server, exchanges SDP offers/answers

export interface SignalingMessage {
  type: 'offer' | 'answer' | 'ice-candidate' | 'peer-list' | 'join' | 'leave';
  from?: string;
  to?: string;
  payload?: unknown;
}

export class SignalingClient {
  private ws: WebSocket | null = null;
  private peerId: string;
  private onMessage: (msg: SignalingMessage) => void;

  constructor(peerId: string, onMessage: (msg: SignalingMessage) => void) {
    this.peerId = peerId;
    this.onMessage = onMessage;
  }

  async connect(serverUrl: string): Promise<void> {
    return new Promise((resolve, reject) => {
      this.ws = new WebSocket(serverUrl);

      this.ws.onopen = () => {
        // Announce ourselves to the signaling server
        this.send({ type: 'join', from: this.peerId });
        resolve();
      };

      this.ws.onerror = (error) => reject(error);

      this.ws.onmessage = (event) => {
        const msg = JSON.parse(event.data) as SignalingMessage;
        this.onMessage(msg);
      };

      this.ws.onclose = () => {
        console.log('Signaling connection closed');
      };
    });
  }

  send(message: SignalingMessage): void {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(message));
    }
  }

  disconnect(): void {
    if (this.ws) {
      this.send({ type: 'leave', from: this.peerId });
      this.ws.close();
      this.ws = null;
    }
  }
}
```

**Acceptance criteria:**
- [ ] SignalingClient connects to WebSocket server
- [ ] Handles offer/answer/ice-candidate message types
- [ ] Proper cleanup on disconnect
- [ ] Unit tests with mock WebSocket

**Checkpoint:** `pnpm test -- signaling` passes

### Task 4: Implement WebRTC P2P Service

Replace mock P2P with real WebRTC DataChannel implementation using simple-peer.

**Files to modify:**
- `viewer/src/services/p2p.ts` - Full WebRTC implementation

**Implementation:**
```typescript
import SimplePeer from 'simple-peer';
import { SignalingClient, SignalingMessage } from './signaling';

// P2P mesh for video chunk delivery via WebRTC DataChannel

interface Peer {
  id: string;
  connection: SimplePeer.Instance;
  isConnected: boolean;
}

export class P2PService {
  private peerId: string;
  private peers: Map<string, Peer> = new Map();
  private signaling: SignalingClient;
  private videoChunkHandler: ((msg: VideoChunkMessage) => void) | null = null;

  constructor() {
    this.peerId = crypto.randomUUID();
    this.signaling = new SignalingClient(this.peerId, this.handleSignalingMessage.bind(this));
  }

  async connect(signalingUrl: string): Promise<boolean> {
    try {
      await this.signaling.connect(signalingUrl);
      return true;
    } catch (error) {
      console.error('Failed to connect to signaling server:', error);
      return false;
    }
  }

  private handleSignalingMessage(msg: SignalingMessage): void {
    switch (msg.type) {
      case 'peer-list':
        this.handlePeerList(msg.payload as string[]);
        break;
      case 'offer':
        this.handleOffer(msg.from!, msg.payload);
        break;
      case 'answer':
        this.handleAnswer(msg.from!, msg.payload);
        break;
      case 'ice-candidate':
        this.handleIceCandidate(msg.from!, msg.payload);
        break;
    }
  }

  private handlePeerList(peerIds: string[]): void {
    // Initiate connections to discovered peers
    for (const peerId of peerIds) {
      if (peerId !== this.peerId && !this.peers.has(peerId)) {
        this.initiatePeerConnection(peerId);
      }
    }
  }

  private initiatePeerConnection(peerId: string): void {
    const peer = new SimplePeer({
      initiator: true,
      trickle: true,
      config: {
        iceServers: [
          { urls: 'stun:stun.l.google.com:19302' },
          { urls: 'stun:stun1.l.google.com:19302' },
        ],
      },
    });

    this.setupPeerHandlers(peer, peerId);
    this.peers.set(peerId, { id: peerId, connection: peer, isConnected: false });
  }

  private handleOffer(fromPeerId: string, signal: unknown): void {
    let existingPeer = this.peers.get(fromPeerId);

    if (!existingPeer) {
      const peer = new SimplePeer({
        initiator: false,
        trickle: true,
        config: {
          iceServers: [
            { urls: 'stun:stun.l.google.com:19302' },
            { urls: 'stun:stun1.l.google.com:19302' },
          ],
        },
      });
      this.setupPeerHandlers(peer, fromPeerId);
      existingPeer = { id: fromPeerId, connection: peer, isConnected: false };
      this.peers.set(fromPeerId, existingPeer);
    }

    existingPeer.connection.signal(signal);
  }

  private handleAnswer(fromPeerId: string, signal: unknown): void {
    const peer = this.peers.get(fromPeerId);
    if (peer) {
      peer.connection.signal(signal);
    }
  }

  private handleIceCandidate(fromPeerId: string, candidate: unknown): void {
    const peer = this.peers.get(fromPeerId);
    if (peer) {
      peer.connection.signal(candidate);
    }
  }

  private setupPeerHandlers(peer: SimplePeer.Instance, peerId: string): void {
    peer.on('signal', (signal) => {
      const msgType = signal.type === 'offer' ? 'offer' :
                      signal.type === 'answer' ? 'answer' : 'ice-candidate';
      this.signaling.send({
        type: msgType,
        from: this.peerId,
        to: peerId,
        payload: signal,
      });
    });

    peer.on('connect', () => {
      console.log(`Connected to peer: ${peerId}`);
      const peerRecord = this.peers.get(peerId);
      if (peerRecord) {
        peerRecord.isConnected = true;
      }
    });

    peer.on('data', (data: Uint8Array) => {
      // Parse video chunk from binary data
      const chunk = this.parseVideoChunk(data);
      if (chunk && this.videoChunkHandler) {
        this.videoChunkHandler(chunk);
      }
    });

    peer.on('close', () => {
      console.log(`Peer disconnected: ${peerId}`);
      this.peers.delete(peerId);
    });

    peer.on('error', (err) => {
      console.error(`Peer error (${peerId}):`, err);
      this.peers.delete(peerId);
    });
  }

  private parseVideoChunk(data: Uint8Array): VideoChunkMessage | null {
    // Binary format: [slot:4][chunk_index:4][timestamp:8][is_keyframe:1][data:rest]
    if (data.length < 17) return null;

    const view = new DataView(data.buffer);
    return {
      slot: view.getUint32(0),
      chunk_index: view.getUint32(4),
      timestamp: Number(view.getBigUint64(8)),
      is_keyframe: data[16] === 1,
      data: data.slice(17),
    };
  }

  onVideoChunk(handler: (msg: VideoChunkMessage) => void): void {
    this.videoChunkHandler = handler;
  }

  getConnectedPeerCount(): number {
    let count = 0;
    for (const peer of this.peers.values()) {
      if (peer.isConnected) count++;
    }
    return count;
  }

  disconnect(): void {
    for (const peer of this.peers.values()) {
      peer.connection.destroy();
    }
    this.peers.clear();
    this.signaling.disconnect();
  }
}

// Export singleton for backward compatibility with existing code
let p2pService: P2PService | null = null;

export function getP2PService(): P2PService {
  if (!p2pService) {
    p2pService = new P2PService();
  }
  return p2pService;
}

// Legacy API for backward compatibility
export async function connectToRelay(relay: RelayInfo): Promise<boolean> {
  const service = getP2PService();
  // In WebRTC mode, we connect to signaling server instead of relay directly
  const signalingUrl = relay.multiaddr.includes('wss')
    ? relay.multiaddr.replace('/tcp/4001/wss', '/signal')
    : 'wss://signal.icn.network';
  return service.connect(signalingUrl);
}

export function onVideoChunk(handler: (msg: VideoChunkMessage) => void): void {
  getP2PService().onVideoChunk(handler);
}

export function disconnect(): void {
  getP2PService().disconnect();
}

export function getConnectionStatus(): boolean {
  return getP2PService().getConnectedPeerCount() > 0;
}
```

**Acceptance criteria:**
- [ ] WebRTC peer connections established via simple-peer
- [ ] Signaling via WebSocket for offer/answer exchange
- [ ] Video chunks received via DataChannel
- [ ] Binary chunk parsing matches existing VideoChunkMessage interface
- [ ] Backward-compatible API for existing VideoPlayer usage
- [ ] Unit tests with mock simple-peer

**Checkpoint:** `pnpm test -- p2p` passes

### Task 5: Add Mock Signaling Server for Development

Create minimal signaling server for local development and testing.

**Files to create:**
- `viewer/scripts/signaling-server.js` - Node.js WebSocket signaling server

**Implementation:**
```javascript
// Minimal signaling server for WebRTC peer discovery
// Usage: node scripts/signaling-server.js

import { WebSocketServer } from 'ws';

const PORT = 8080;
const wss = new WebSocketServer({ port: PORT });

const peers = new Map(); // peerId -> WebSocket

wss.on('connection', (ws) => {
  let peerId = null;

  ws.on('message', (data) => {
    const msg = JSON.parse(data.toString());

    switch (msg.type) {
      case 'join':
        peerId = msg.from;
        peers.set(peerId, ws);
        console.log(`Peer joined: ${peerId}`);

        // Send current peer list to new peer
        const peerList = Array.from(peers.keys()).filter(id => id !== peerId);
        ws.send(JSON.stringify({ type: 'peer-list', payload: peerList }));

        // Notify existing peers
        for (const [id, peer] of peers) {
          if (id !== peerId) {
            peer.send(JSON.stringify({ type: 'peer-list', payload: [peerId] }));
          }
        }
        break;

      case 'offer':
      case 'answer':
      case 'ice-candidate':
        // Relay to target peer
        const targetWs = peers.get(msg.to);
        if (targetWs) {
          targetWs.send(JSON.stringify(msg));
        }
        break;

      case 'leave':
        peers.delete(peerId);
        console.log(`Peer left: ${peerId}`);
        break;
    }
  });

  ws.on('close', () => {
    if (peerId) {
      peers.delete(peerId);
      console.log(`Peer disconnected: ${peerId}`);
    }
  });
});

console.log(`Signaling server running on ws://localhost:${PORT}`);
```

**Add npm script to package.json:**
```json
{
  "scripts": {
    "signal": "node scripts/signaling-server.js"
  }
}
```

**Acceptance criteria:**
- [ ] Signaling server starts with `pnpm signal`
- [ ] Handles join/leave/offer/answer/ice-candidate messages
- [ ] Broadcasts peer list on join
- [ ] Relays signaling messages to target peers
- [ ] Clean disconnection handling

**Checkpoint:** `pnpm signal` starts server, logs connections

### Task 6: Update Test Mocks for Web Environment

Update test setup to mock WebRTC and WebSocket instead of Tauri.

**Files to modify:**
- `viewer/src/test/setup.ts` - Replace Tauri mocks with WebRTC/WebSocket mocks

**Implementation:**
```typescript
// Test setup for web environment

// Mock WebSocket
class MockWebSocket {
  static CONNECTING = 0;
  static OPEN = 1;
  static CLOSING = 2;
  static CLOSED = 3;

  readyState = MockWebSocket.OPEN;
  onopen: (() => void) | null = null;
  onclose: (() => void) | null = null;
  onmessage: ((event: { data: string }) => void) | null = null;
  onerror: ((error: Event) => void) | null = null;

  constructor(url: string) {
    setTimeout(() => this.onopen?.(), 0);
  }

  send(data: string): void {
    // Mock implementation - can be extended for specific tests
  }

  close(): void {
    this.readyState = MockWebSocket.CLOSED;
    this.onclose?.();
  }
}

global.WebSocket = MockWebSocket as unknown as typeof WebSocket;

// Mock RTCPeerConnection for simple-peer
class MockRTCPeerConnection {
  localDescription: RTCSessionDescription | null = null;
  remoteDescription: RTCSessionDescription | null = null;
  iceConnectionState = 'new';

  createOffer = vi.fn().mockResolvedValue({ type: 'offer', sdp: 'mock-sdp' });
  createAnswer = vi.fn().mockResolvedValue({ type: 'answer', sdp: 'mock-sdp' });
  setLocalDescription = vi.fn().mockResolvedValue(undefined);
  setRemoteDescription = vi.fn().mockResolvedValue(undefined);
  addIceCandidate = vi.fn().mockResolvedValue(undefined);
  close = vi.fn();

  onicecandidate: ((event: { candidate: RTCIceCandidate | null }) => void) | null = null;
  ondatachannel: ((event: { channel: RTCDataChannel }) => void) | null = null;
  onconnectionstatechange: (() => void) | null = null;
}

global.RTCPeerConnection = MockRTCPeerConnection as unknown as typeof RTCPeerConnection;

// Mock RTCSessionDescription
global.RTCSessionDescription = class {
  type: string;
  sdp: string;
  constructor(init: { type: string; sdp: string }) {
    this.type = init.type;
    this.sdp = init.sdp;
  }
} as unknown as typeof RTCSessionDescription;

// Keep existing WebCodecs and Canvas mocks
// ... (existing VideoDecoder, EncodedVideoChunk, VideoFrame mocks)
```

**Acceptance criteria:**
- [ ] WebSocket mock available in tests
- [ ] RTCPeerConnection mock available in tests
- [ ] Existing video pipeline tests still pass
- [ ] New P2P service tests can mock WebRTC

**Checkpoint:** `pnpm test` passes with new mocks

### Task 7: Create Standalone Vite Configuration

Configure Vite for standalone web deployment without Tauri.

**Files to modify:**
- `viewer/vite.config.ts` - Web-only configuration
- `viewer/index.html` - Update if needed

**Implementation:**

vite.config.ts:
```typescript
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],

  // Base path for deployment (can be customized via env)
  base: process.env.VITE_BASE_PATH || '/',

  server: {
    port: 5173,
    strictPort: true,
  },

  build: {
    target: 'es2021',
    outDir: 'dist',
    sourcemap: true,
    rollupOptions: {
      output: {
        manualChunks: {
          vendor: ['react', 'react-dom', 'zustand'],
          p2p: ['simple-peer'],
        },
      },
    },
  },

  test: {
    globals: true,
    environment: 'jsdom',
    setupFiles: ['./src/test/setup.ts'],
    include: ['src/**/*.{test,spec}.{js,ts,jsx,tsx}'],
    coverage: {
      reporter: ['text', 'json', 'html'],
      exclude: ['node_modules/', 'src/test/'],
    },
  },

  define: {
    // Environment variables for signaling server URL
    'import.meta.env.VITE_SIGNALING_URL': JSON.stringify(
      process.env.VITE_SIGNALING_URL || 'ws://localhost:8080'
    ),
  },
});
```

**Acceptance criteria:**
- [ ] `pnpm build` produces optimized dist/ folder
- [ ] Vendor chunks properly split for caching
- [ ] Source maps generated for debugging
- [ ] Environment variables configurable for deployment

**Checkpoint:** `pnpm build && ls -la dist/` shows built assets

### Task 8: Integration Test - Full Web Flow

Create integration test that validates end-to-end web functionality.

**Files to create:**
- `viewer/src/services/__tests__/p2p.test.ts` - P2P service tests
- `viewer/src/services/__tests__/signaling.test.ts` - Signaling client tests

**Test scenarios:**

p2p.test.ts:
```typescript
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { P2PService, VideoChunkMessage } from '../p2p';

describe('P2PService', () => {
  let service: P2PService;

  beforeEach(() => {
    service = new P2PService();
  });

  afterEach(() => {
    service.disconnect();
  });

  it('connects to signaling server', async () => {
    const result = await service.connect('ws://localhost:8080');
    expect(result).toBe(true);
  });

  it('handles peer discovery from signaling', async () => {
    await service.connect('ws://localhost:8080');

    // Simulate receiving peer list
    // ... mock signaling message

    expect(service.getConnectedPeerCount()).toBeGreaterThanOrEqual(0);
  });

  it('receives video chunks via DataChannel', async () => {
    const chunks: VideoChunkMessage[] = [];
    service.onVideoChunk((chunk) => chunks.push(chunk));

    await service.connect('ws://localhost:8080');

    // Simulate receiving chunk data
    // ... mock DataChannel message

    // Verify chunk parsed correctly
  });

  it('handles peer disconnection gracefully', async () => {
    await service.connect('ws://localhost:8080');
    service.disconnect();

    expect(service.getConnectedPeerCount()).toBe(0);
  });
});
```

signaling.test.ts:
```typescript
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { SignalingClient, SignalingMessage } from '../signaling';

describe('SignalingClient', () => {
  it('connects and sends join message', async () => {
    const onMessage = vi.fn();
    const client = new SignalingClient('test-peer', onMessage);

    await client.connect('ws://localhost:8080');

    // Verify join message was sent
    // (check via mock WebSocket)
  });

  it('handles incoming signaling messages', async () => {
    const onMessage = vi.fn();
    const client = new SignalingClient('test-peer', onMessage);

    await client.connect('ws://localhost:8080');

    // Simulate incoming message
    // ... trigger mock WebSocket onmessage

    expect(onMessage).toHaveBeenCalled();
  });

  it('sends offer/answer messages', async () => {
    const onMessage = vi.fn();
    const client = new SignalingClient('test-peer', onMessage);

    await client.connect('ws://localhost:8080');

    client.send({
      type: 'offer',
      from: 'test-peer',
      to: 'other-peer',
      payload: { type: 'offer', sdp: 'test-sdp' },
    });

    // Verify message sent via mock WebSocket
  });
});
```

**Acceptance criteria:**
- [ ] P2P service unit tests pass
- [ ] Signaling client unit tests pass
- [ ] Mock WebRTC connections work in tests
- [ ] Video chunk parsing verified
- [ ] Error handling tested

**Checkpoint:** `pnpm test -- --coverage` shows adequate coverage

## Verification

**Build and test:**
```bash
cd viewer && pnpm install
cd viewer && pnpm lint
cd viewer && pnpm typecheck
cd viewer && pnpm test
cd viewer && pnpm build
```

**Manual verification:**
```bash
# Terminal 1: Start signaling server
cd viewer && pnpm signal

# Terminal 2: Start web app
cd viewer && pnpm dev

# Open http://localhost:5173 in browser
# Verify: App loads, settings persist on refresh
# Verify: Connection status updates
# Verify: No console errors about Tauri
```

**Expected output:**
- All TypeScript types resolve
- No Tauri imports in compiled output
- Web app loads in browser without Tauri
- Settings persist via localStorage
- WebRTC signaling connects (with local server)

## Success Criteria

- [ ] No `@tauri-apps/api` imports in codebase
- [ ] `pnpm dev` starts standalone web server
- [ ] `pnpm build` produces deployable dist/
- [ ] Settings persist via Zustand localStorage
- [ ] WebRTC P2P service implemented with simple-peer
- [ ] Signaling client connects to WebSocket server
- [ ] Mock signaling server for development
- [ ] All unit tests pass
- [ ] App functions in Chrome/Firefox/Safari

## Output

**Artifacts:**
- Standalone web application in `viewer/dist/`
- WebRTC P2P service with signaling client
- Development signaling server
- Updated test suite for web environment

**Dependencies for next phases:**
- Phase 5 (Multi-Node E2E Simulation) will need signaling server deployment
- Phase 6 (Testnet Deployment) will configure production signaling infrastructure

**Deferred to future work:**
- Production signaling server deployment (Phase 6)
- TURN server for NAT traversal (evaluate based on testnet feedback)
- libp2p-webrtc full integration (when relay infrastructure ready)
- Peer selection optimization (latency-based, geography-aware)

## Research Notes

**WebRTC DataChannel:**
- Max chunk size: 64KB recommended ([RFC 8831](https://datatracker.ietf.org/doc/html/rfc8831))
- Binary data via ArrayBuffer
- SCTP over DTLS for transport

**simple-peer:**
- Abstracts WebRTC complexity ([GitHub](https://github.com/feross/simple-peer))
- Handles ICE candidate exchange
- Trickle ICE supported
- Used by WebTorrent and CDNBye

**Signaling Options:**
- WebSocket (chosen for simplicity)
- Firebase Realtime Database ([firepeer](https://github.com/nicktomlin/firepeer))
- WebTorrent trackers ([P2PT](https://github.com/nicktomlin/p2pt))

**Public STUN Servers:**
- `stun:stun.l.google.com:19302`
- `stun:stun1.l.google.com:19302`
