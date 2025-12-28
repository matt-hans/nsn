# ICN Viewer Client - Component Specifications

> **Design Philosophy**: Retro-futuristic cosmic portal aesthetic - VHS static meets quantum physics
> **Visual Identity**: CRT warmth + neon portals + dimensional distortion

---

## Typography System

### Font Stack (Load Order)
```css
/* Google Fonts Import */
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;600;700;900&family=Rajdhani:wght@400;500;600;700&family=IBM+Plex+Sans:wght@300;400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap');
```

| Role | Font | Weights | Usage |
|------|------|---------|-------|
| Display | Orbitron | 400-900 | Slot numbers, hero moments, brand |
| Heading | Rajdhani | 400-700 | Section headers, labels, UI chrome |
| Body | IBM Plex Sans | 300-600 | Descriptions, paragraphs |
| Mono | IBM Plex Mono | 400-600 | Peer IDs, stats, timestamps |

---

## Component Hierarchy

```
App.tsx (Root)
├── AppShell (Layout wrapper with textures)
│   ├── NoiseOverlay (SVG noise texture)
│   ├── ScanlineOverlay (CRT effect)
│   └── VignetteOverlay (Edge darkening)
├── TopBar.tsx (Gradient header)
│   ├── Logo (Orbitron + glow)
│   ├── ConnectionStatus (Animated indicator)
│   └── SettingsButton (Icon button)
├── VideoPlayer.tsx (Main canvas + controls)
│   ├── VideoCanvas (WebCodecs + CRT filters)
│   ├── SlotDisplay (Large Orbitron number + glitch)
│   ├── ControlsOverlay (Gradient bar)
│   │   ├── PlayPauseButton
│   │   ├── SeekBar (Portal gradient)
│   │   ├── VolumeControl
│   │   ├── QualitySelector
│   │   └── FullscreenButton
│   └── LoadingPortal (Multi-ring animation)
├── Sidebar.tsx (Slide-in panel)
│   ├── SlotInfo (Current slot details)
│   ├── DirectorInfo (Peer ID + reputation)
│   ├── NetworkStats (Bitrate, latency, peers)
│   └── SeedingPanel (Contribution stats)
└── SettingsModal.tsx (Overlay)
    ├── SeedingToggle
    ├── QualityPreference
    └── VolumeDefault
```

---

## Core Components

### AppShell (Layout Wrapper)

**Purpose**: Provides base layout with atmospheric textures

```tsx
interface AppShellProps {
  children: React.ReactNode;
}

// CSS Layers (bottom to top)
// 1. Void gradient background
// 2. Grid pattern (subtle)
// 3. Content
// 4. Scanline overlay
// 5. Noise overlay
// 6. Vignette overlay
```

**Styling**:
```css
.app-shell {
  position: relative;
  width: 100vw;
  height: 100vh;
  background: radial-gradient(ellipse at center, #1A0533 0%, #0D0221 50%, #050110 100%);
  overflow: hidden;
}

/* Subtle grid pattern */
.app-shell::before {
  content: '';
  position: absolute;
  inset: 0;
  background:
    linear-gradient(rgba(0, 255, 198, 0.03) 1px, transparent 1px),
    linear-gradient(90deg, rgba(0, 255, 198, 0.03) 1px, transparent 1px);
  background-size: 40px 40px;
  opacity: 0.5;
  pointer-events: none;
}

/* Scanline overlay */
.scanlines {
  position: fixed;
  inset: 0;
  background: repeating-linear-gradient(
    0deg,
    rgba(0, 0, 0, 0.15) 0px,
    rgba(0, 0, 0, 0.15) 1px,
    transparent 1px,
    transparent 2px
  );
  opacity: 0.06;
  pointer-events: none;
  z-index: 9998;
}

/* Noise texture */
.noise {
  position: fixed;
  inset: 0;
  background-image: url("data:image/svg+xml,..."); /* noise SVG */
  opacity: 0.025;
  mix-blend-mode: overlay;
  pointer-events: none;
  z-index: 9999;
}

/* Vignette */
.vignette {
  position: fixed;
  inset: 0;
  background: radial-gradient(ellipse 100% 100% at 50% 50%, transparent 40%, rgba(0, 0, 0, 0.5) 100%);
  pointer-events: none;
  z-index: 9997;
}
```

---

### TopBar

**Purpose**: App-wide status and navigation

**Layout**:
```
┌──────────────────────────────────────────────────────────────────────┐
│ [ICN Logo]                [● Connected to NA-WEST]           [⚙️]  │
└──────────────────────────────────────────────────────────────────────┘
```

**Styling**:
```css
.topbar {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  height: 56px;
  background: linear-gradient(180deg, rgba(13, 2, 33, 0.95) 0%, transparent 100%);
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 24px;
  z-index: 100;
}

.topbar-logo {
  font-family: 'Orbitron', sans-serif;
  font-weight: 700;
  font-size: 1.25rem;
  letter-spacing: 0.15em;
  text-transform: uppercase;
  color: #00FFC6;
  filter: drop-shadow(0 0 10px rgba(0, 255, 198, 0.4));
}

.connection-status {
  display: flex;
  align-items: center;
  gap: 10px;
  font-family: 'Rajdhani', sans-serif;
  font-weight: 500;
  font-size: 0.875rem;
  letter-spacing: 0.05em;
  text-transform: uppercase;
}

.status-dot {
  width: 10px;
  height: 10px;
  border-radius: 50%;
  background: #00FFC6;
  box-shadow: 0 0 12px rgba(0, 255, 198, 0.6);
  animation: pulse-glow 2s ease-in-out infinite;
}

.status-dot.error {
  background: #FF3366;
  box-shadow: 0 0 12px rgba(255, 51, 102, 0.6);
}

.status-dot.buffering {
  background: #FFEE00;
  animation: spin 1s linear infinite;
}
```

---

### VideoPlayer

**Purpose**: Main video playback with WebCodecs integration

**Props**:
```typescript
interface VideoPlayerProps {
  currentSlot: number;
  quality: '1080p' | '720p' | '480p' | 'auto';
  onSlotChange?: (slot: number) => void;
}
```

**State**:
```typescript
interface PlayerState {
  playbackState: 'idle' | 'buffering' | 'playing' | 'paused';
  currentTime: number;
  duration: number;
  volume: number;
  isMuted: boolean;
  isFullscreen: boolean;
  showControls: boolean;
  bufferedPercent: number;
}
```

**Styling**:
```css
.video-player {
  position: relative;
  width: 100%;
  height: 100%;
  background: #000;
}

.video-canvas {
  width: 100%;
  height: 100%;
  object-fit: contain;
  /* CRT warmth filter */
  filter: brightness(1.02) contrast(1.02) saturate(1.05);
}

/* Slot number display */
.slot-display {
  position: absolute;
  top: 80px;
  left: 32px;
  font-family: 'Orbitron', sans-serif;
  font-weight: 900;
  font-size: 4rem;
  letter-spacing: 0.15em;
  color: #00FFC6;
  text-shadow: 0 0 40px rgba(0, 255, 198, 0.4);
  opacity: 0;
  transition: opacity 300ms;
}

.slot-display.visible {
  opacity: 1;
}

.slot-display.changing {
  animation: glitch 400ms steps(4, end);
}

@keyframes glitch {
  0%, 100% {
    transform: translate(0);
    text-shadow: 2px 0 #00FFC6, -2px 0 #FF3399;
  }
  10% {
    transform: translate(-2px, 1px);
    text-shadow: -2px 0 #00FFC6, 2px 0 #FF3399;
  }
  20% {
    transform: translate(2px, -1px);
    text-shadow: 2px 0 #7B61FF, -2px 0 #00FFC6;
  }
  30% {
    transform: translate(0);
    text-shadow: -2px 0 #FF3399, 2px 0 #7B61FF;
  }
}
```

---

### ControlsOverlay

**Purpose**: Video playback controls bar

**Styling**:
```css
.controls-overlay {
  position: absolute;
  bottom: 0;
  left: 0;
  right: 0;
  height: 100px;
  background: linear-gradient(180deg, transparent 0%, rgba(13, 2, 33, 0.95) 100%);
  backdrop-filter: blur(12px);
  padding: 20px 28px;
  display: flex;
  flex-direction: column;
  gap: 16px;
  opacity: 1;
  transition: opacity 300ms, transform 300ms;
}

.controls-overlay.hidden {
  opacity: 0;
  transform: translateY(20px);
  pointer-events: none;
}

/* Seek bar container */
.seekbar-container {
  width: 100%;
  height: 24px;
  display: flex;
  align-items: center;
  cursor: pointer;
}

.seekbar {
  position: relative;
  width: 100%;
  height: 4px;
  background: rgba(255, 255, 255, 0.15);
  border-radius: 9999px;
  overflow: hidden;
  transition: height 150ms;
}

.seekbar:hover {
  height: 6px;
}

.seekbar-buffered {
  position: absolute;
  left: 0;
  top: 0;
  height: 100%;
  background: rgba(255, 255, 255, 0.3);
  border-radius: 9999px;
}

.seekbar-progress {
  position: absolute;
  left: 0;
  top: 0;
  height: 100%;
  background: linear-gradient(90deg, #00FFC6 0%, #7B61FF 50%, #FF3399 100%);
  border-radius: 9999px;
  box-shadow: 0 0 10px rgba(0, 255, 198, 0.5);
}

.seekbar-thumb {
  position: absolute;
  top: 50%;
  transform: translate(-50%, -50%) scale(0);
  width: 16px;
  height: 16px;
  background: #00FFC6;
  border-radius: 50%;
  box-shadow: 0 0 15px rgba(0, 255, 198, 0.6);
  transition: transform 150ms;
}

.seekbar:hover .seekbar-thumb {
  transform: translate(-50%, -50%) scale(1);
}

/* Control buttons row */
.controls-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.controls-left {
  display: flex;
  align-items: center;
  gap: 16px;
}

.controls-right {
  display: flex;
  align-items: center;
  gap: 12px;
}

/* Icon button */
.icon-button {
  width: 44px;
  height: 44px;
  display: flex;
  align-items: center;
  justify-content: center;
  background: rgba(255, 255, 255, 0.05);
  border: none;
  border-radius: 12px;
  color: #FFFFFF;
  cursor: pointer;
  transition: background 150ms, box-shadow 150ms;
}

.icon-button:hover {
  background: rgba(0, 255, 198, 0.15);
  box-shadow: 0 0 15px rgba(0, 255, 198, 0.2);
}

.icon-button:focus-visible {
  outline: 2px solid #00FFC6;
  outline-offset: 3px;
}

/* Volume slider */
.volume-control {
  display: flex;
  align-items: center;
  gap: 8px;
}

.volume-slider {
  width: 80px;
  height: 4px;
  background: rgba(255, 255, 255, 0.2);
  border-radius: 9999px;
  appearance: none;
}

.volume-slider::-webkit-slider-thumb {
  appearance: none;
  width: 14px;
  height: 14px;
  background: #FFFFFF;
  border-radius: 50%;
  cursor: pointer;
}
```

---

### LoadingPortal

**Purpose**: Cosmic loading animation during buffering

**Implementation**:
```tsx
const LoadingPortal = () => {
  const messages = [
    "Scanning dimensional frequencies...",
    "Tuning interdimensional receiver...",
    "Stabilizing portal connection...",
    "Aligning quantum bandwidth...",
    "Locking onto dimension C-137...",
    "Bypassing galactic federation...",
    "Calibrating reality anchor..."
  ];

  const [messageIndex, setMessageIndex] = useState(0);

  useEffect(() => {
    const interval = setInterval(() => {
      setMessageIndex(i => (i + 1) % messages.length);
    }, 2500);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="loading-portal">
      <div className="portal-rings">
        <div className="ring ring-1" />
        <div className="ring ring-2" />
        <div className="ring ring-3" />
      </div>
      <p className="loading-message">{messages[messageIndex]}</p>
    </div>
  );
};
```

**Styling**:
```css
.loading-portal {
  position: absolute;
  inset: 0;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 32px;
  background: rgba(13, 2, 33, 0.8);
  backdrop-filter: blur(8px);
}

.portal-rings {
  position: relative;
  width: 120px;
  height: 120px;
}

.ring {
  position: absolute;
  inset: 0;
  border-radius: 50%;
  border: 2px solid transparent;
  animation: portal-spin 3s linear infinite;
}

.ring-1 {
  border-top-color: #00FFC6;
  animation-duration: 2s;
}

.ring-2 {
  inset: 15px;
  border-right-color: #7B61FF;
  animation-duration: 2.5s;
  animation-direction: reverse;
}

.ring-3 {
  inset: 30px;
  border-bottom-color: #FF3399;
  animation-duration: 3s;
}

@keyframes portal-spin {
  0% {
    transform: rotate(0deg);
    filter: hue-rotate(0deg);
  }
  100% {
    transform: rotate(360deg);
    filter: hue-rotate(360deg);
  }
}

.loading-message {
  font-family: 'IBM Plex Sans', sans-serif;
  font-size: 1rem;
  color: #B8A8D1;
  animation: fade-pulse 2.5s ease-in-out infinite;
}

@keyframes fade-pulse {
  0%, 100% { opacity: 0.6; }
  50% { opacity: 1; }
}
```

---

### Sidebar

**Purpose**: Slide-in panel with slot metadata and stats

**Behavior**:
- Slides from left edge on hover or button click
- Width: 320px
- Glassmorphic background with portal glow border

**Styling**:
```css
.sidebar {
  position: fixed;
  left: 0;
  top: 0;
  bottom: 0;
  width: 320px;
  background: rgba(13, 2, 33, 0.95);
  backdrop-filter: blur(24px);
  border-right: 1px solid rgba(0, 255, 198, 0.1);
  box-shadow: 20px 0 60px rgba(0, 0, 0, 0.5);
  padding: 80px 24px 24px;
  transform: translateX(-100%);
  transition: transform 400ms cubic-bezier(0.16, 1, 0.3, 1);
  z-index: 50;
}

.sidebar.open {
  transform: translateX(0);
}

/* Section styling */
.sidebar-section {
  margin-bottom: 32px;
}

.sidebar-section-title {
  font-family: 'Rajdhani', sans-serif;
  font-weight: 600;
  font-size: 0.75rem;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: #7A6B94;
  margin-bottom: 16px;
}

/* Stat row */
.stat-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 10px 0;
  border-bottom: 1px solid rgba(255, 255, 255, 0.05);
}

.stat-label {
  font-family: 'IBM Plex Sans', sans-serif;
  font-size: 0.875rem;
  color: #B8A8D1;
}

.stat-value {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.875rem;
  color: #FFFFFF;
}

.stat-value.highlight {
  color: #00FFC6;
}

/* Director info card */
.director-card {
  background: rgba(0, 255, 198, 0.05);
  border: 1px solid rgba(0, 255, 198, 0.15);
  border-radius: 12px;
  padding: 16px;
}

.director-id {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.75rem;
  color: #B8A8D1;
  word-break: break-all;
}

.reputation-bar {
  height: 4px;
  background: rgba(255, 255, 255, 0.1);
  border-radius: 9999px;
  margin-top: 12px;
  overflow: hidden;
}

.reputation-fill {
  height: 100%;
  background: linear-gradient(90deg, #00FFC6 0%, #7B61FF 100%);
  border-radius: 9999px;
}
```

---

### SettingsModal

**Purpose**: User preferences overlay

**Styling**:
```css
.settings-backdrop {
  position: fixed;
  inset: 0;
  background: rgba(13, 2, 33, 0.9);
  backdrop-filter: blur(12px);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 1000;
  animation: fade-in 200ms ease-out;
}

.settings-modal {
  width: 480px;
  max-height: 80vh;
  background: rgba(26, 5, 51, 0.95);
  backdrop-filter: blur(20px);
  border: 1px solid rgba(0, 255, 198, 0.15);
  border-radius: 24px;
  padding: 32px;
  overflow-y: auto;
  animation: modal-enter 300ms cubic-bezier(0.16, 1, 0.3, 1);
}

@keyframes modal-enter {
  0% {
    opacity: 0;
    transform: scale(0.95) translateY(20px);
  }
  100% {
    opacity: 1;
    transform: scale(1) translateY(0);
  }
}

.settings-title {
  font-family: 'Orbitron', sans-serif;
  font-weight: 700;
  font-size: 1.5rem;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: #FFFFFF;
  margin-bottom: 32px;
}

/* Setting row */
.setting-row {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  padding: 20px 0;
  border-bottom: 1px solid rgba(255, 255, 255, 0.05);
}

.setting-info {
  flex: 1;
  padding-right: 24px;
}

.setting-label {
  font-family: 'Rajdhani', sans-serif;
  font-weight: 600;
  font-size: 1rem;
  color: #FFFFFF;
  margin-bottom: 4px;
}

.setting-description {
  font-family: 'IBM Plex Sans', sans-serif;
  font-size: 0.875rem;
  color: #7A6B94;
  line-height: 1.5;
}

/* Toggle switch */
.toggle {
  position: relative;
  width: 52px;
  height: 28px;
  background: rgba(255, 255, 255, 0.1);
  border: 1px solid rgba(255, 255, 255, 0.2);
  border-radius: 9999px;
  cursor: pointer;
  transition: background 200ms, border-color 200ms, box-shadow 200ms;
}

.toggle.on {
  background: linear-gradient(135deg, #00FFC6 0%, #00D9A6 100%);
  border-color: transparent;
  box-shadow: 0 0 20px rgba(0, 255, 198, 0.4);
}

.toggle-indicator {
  position: absolute;
  top: 3px;
  left: 3px;
  width: 22px;
  height: 22px;
  background: #6B7280;
  border-radius: 50%;
  transition: transform 200ms, background 200ms;
}

.toggle.on .toggle-indicator {
  transform: translateX(24px);
  background: #FFFFFF;
}

/* Dropdown */
.dropdown-trigger {
  min-width: 140px;
  padding: 12px 16px;
  background: rgba(0, 0, 0, 0.4);
  border: 1px solid rgba(255, 255, 255, 0.15);
  border-radius: 12px;
  color: #FFFFFF;
  font-family: 'IBM Plex Sans', sans-serif;
  font-size: 0.875rem;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
}

.dropdown-menu {
  position: absolute;
  top: 100%;
  right: 0;
  margin-top: 8px;
  min-width: 160px;
  background: rgba(26, 5, 51, 0.95);
  backdrop-filter: blur(20px);
  border: 1px solid rgba(0, 255, 198, 0.15);
  border-radius: 12px;
  padding: 8px;
  box-shadow: 0 16px 48px rgba(0, 0, 0, 0.5);
}

.dropdown-item {
  padding: 12px 16px;
  border-radius: 8px;
  color: #B8A8D1;
  cursor: pointer;
  transition: background 150ms, color 150ms;
}

.dropdown-item:hover {
  background: rgba(0, 255, 198, 0.1);
  color: #FFFFFF;
}

.dropdown-item.active {
  background: rgba(0, 255, 198, 0.15);
  color: #00FFC6;
}
```

---

## State Management (Zustand)

```typescript
// store/appStore.ts
import { create } from 'zustand';
import { persist } from 'zustand/middleware';

interface AppState {
  // Playback
  currentSlot: number;
  playbackState: 'idle' | 'buffering' | 'playing' | 'paused';
  volume: number;
  isMuted: boolean;
  quality: '1080p' | '720p' | '480p' | 'auto';

  // Connection
  connectionStatus: 'disconnected' | 'connecting' | 'connected' | 'error';
  connectedRelay: string | null;
  relayRegion: string | null;

  // Stats
  bitrate: number;
  latency: number;
  connectedPeers: number;
  bufferSeconds: number;

  // Settings
  seedingEnabled: boolean;

  // UI
  showSidebar: boolean;
  showSettings: boolean;
  showControls: boolean;

  // Actions
  setCurrentSlot: (slot: number) => void;
  setPlaybackState: (state: AppState['playbackState']) => void;
  setVolume: (volume: number) => void;
  toggleMute: () => void;
  setQuality: (quality: AppState['quality']) => void;
  setConnectionStatus: (status: AppState['connectionStatus']) => void;
  toggleSidebar: () => void;
  toggleSettings: () => void;
  setSeedingEnabled: (enabled: boolean) => void;
}

export const useAppStore = create<AppState>()(
  persist(
    (set) => ({
      // Initial state
      currentSlot: 0,
      playbackState: 'idle',
      volume: 80,
      isMuted: false,
      quality: 'auto',
      connectionStatus: 'disconnected',
      connectedRelay: null,
      relayRegion: null,
      bitrate: 0,
      latency: 0,
      connectedPeers: 0,
      bufferSeconds: 0,
      seedingEnabled: false,
      showSidebar: false,
      showSettings: false,
      showControls: true,

      // Actions
      setCurrentSlot: (slot) => set({ currentSlot: slot }),
      setPlaybackState: (state) => set({ playbackState: state }),
      setVolume: (volume) => set({ volume, isMuted: volume === 0 }),
      toggleMute: () => set((s) => ({ isMuted: !s.isMuted })),
      setQuality: (quality) => set({ quality }),
      setConnectionStatus: (status) => set({ connectionStatus: status }),
      toggleSidebar: () => set((s) => ({ showSidebar: !s.showSidebar })),
      toggleSettings: () => set((s) => ({ showSettings: !s.showSettings })),
      setSeedingEnabled: (enabled) => set({ seedingEnabled: enabled }),
    }),
    {
      name: 'icn-viewer-storage',
      partialize: (state) => ({
        volume: state.volume,
        quality: state.quality,
        seedingEnabled: state.seedingEnabled,
      }),
    }
  )
);
```

---

## Accessibility

### Focus Management
- All interactive elements have visible focus rings (2px solid #00FFC6, 3px offset)
- Focus trap in modals
- Escape key closes modals and panels

### Keyboard Shortcuts
| Key | Action |
|-----|--------|
| Space | Play/Pause |
| M | Toggle mute |
| F | Toggle fullscreen |
| I | Toggle sidebar |
| Left/Right | Seek ±5s |
| Up/Down | Volume ±10% |
| Esc | Close modal/panel |

### Screen Reader
- ARIA labels on all icon buttons
- Live region for status changes
- Semantic heading structure

### Reduced Motion
```css
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after {
    animation-duration: 0.01ms !important;
    transition-duration: 0.01ms !important;
  }
}
```

---

## File Structure

```
viewer/
├── src/
│   ├── App.tsx
│   ├── main.tsx
│   ├── components/
│   │   ├── AppShell.tsx
│   │   ├── TopBar.tsx
│   │   ├── VideoPlayer/
│   │   │   ├── index.tsx
│   │   │   ├── ControlsOverlay.tsx
│   │   │   ├── SeekBar.tsx
│   │   │   ├── VolumeControl.tsx
│   │   │   ├── QualitySelector.tsx
│   │   │   └── LoadingPortal.tsx
│   │   ├── Sidebar/
│   │   │   ├── index.tsx
│   │   │   ├── SlotInfo.tsx
│   │   │   ├── DirectorInfo.tsx
│   │   │   └── NetworkStats.tsx
│   │   ├── SettingsModal/
│   │   │   ├── index.tsx
│   │   │   ├── Toggle.tsx
│   │   │   └── Dropdown.tsx
│   │   └── ui/
│   │       ├── Button.tsx
│   │       ├── IconButton.tsx
│   │       └── Tooltip.tsx
│   ├── store/
│   │   └── appStore.ts
│   ├── services/
│   │   ├── p2p.ts
│   │   ├── videoBuffer.ts
│   │   └── webcodecs.ts
│   ├── hooks/
│   │   ├── useKeyboardShortcuts.ts
│   │   ├── useControlsVisibility.ts
│   │   └── usePersistedState.ts
│   ├── styles/
│   │   ├── global.css
│   │   ├── tokens.css
│   │   └── animations.css
│   └── utils/
│       ├── formatters.ts
│       └── constants.ts
└── src-tauri/
    ├── src/
    │   ├── main.rs
    │   ├── commands.rs
    │   └── storage.rs
    └── Cargo.toml
```

---

## Animation Choreography

### Page Load Sequence (staggered 80ms)
1. Void background fades in (0ms)
2. TopBar slides down (80ms)
3. Video canvas fades in (160ms)
4. Loading portal appears (240ms)

### Slot Change
1. Current video blurs + scales up (portal-transition)
2. Glitch effect on slot number
3. New video fades in
4. Stats update with number animation

### Sidebar Open
1. Backdrop fades in (200ms)
2. Panel slides from left (400ms, spring easing)
3. Content staggers in (80ms delay per section)

### Settings Modal
1. Backdrop blurs in (200ms)
2. Modal scales up from 95% (300ms, spring)
3. Content fades up (staggered 80ms)

---

**Design System Version**: 2.0
**Last Updated**: 2025-12-28
**Aesthetic**: Retro-futuristic CRT + Cosmic Portals
