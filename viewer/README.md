# ICN Viewer Client

> **VHS Quantum Desktop App** - Watch interdimensional cable streams with style

Built with Tauri 2.0 + React 18 + Zustand + WebCodecs

## Features

- **Hardware-Accelerated Playback**: WebCodecs API for low-latency video decode
- **P2P Discovery**: Connect to relays via libp2p-js Kademlia DHT
- **Adaptive Bitrate**: Automatic quality switching (1080p/720p/480p)
- **Optional Seeding**: Earn ICN tokens by re-sharing content
- **VHS Quantum UI**: Retro-futuristic cosmic portal aesthetic
- **Keyboard Shortcuts**: Full keyboard control (Space, M, F, I, Arrows)
- **State Persistence**: Resume playback across app restarts

## Prerequisites

- **Node.js** 20+ and pnpm 8+
- **Rust** 1.75+ (for Tauri backend)
- **macOS** 11+ / **Windows** 10+ / **Ubuntu** 20.04+

## Installation

```bash
# Install dependencies
pnpm install

# Run development mode (hot reload)
pnpm tauri:dev

# Build production app
pnpm tauri:build
```

## Development

### Project Structure

```
viewer/
├── src/                      # React frontend
│   ├── components/           # UI components
│   │   ├── AppShell.tsx      # Layout wrapper
│   │   ├── TopBar.tsx        # Connection status
│   │   ├── VideoPlayer/      # Playback components
│   │   ├── Sidebar/          # Stream metadata
│   │   └── SettingsModal/    # User preferences
│   ├── store/
│   │   └── appStore.ts       # Zustand state management
│   ├── services/
│   │   ├── p2p.ts            # libp2p-js DHT discovery
│   │   ├── videoBuffer.ts    # Chunk buffering
│   │   └── webcodecs.ts      # Hardware decoder
│   ├── hooks/
│   │   └── useKeyboardShortcuts.ts
│   └── styles/
│       ├── tokens.css        # Design system tokens
│       ├── animations.css    # Keyframes
│       └── global.css        # Component styles
├── src-tauri/                # Rust backend
│   └── src/
│       ├── main.rs           # Tauri entrypoint
│       ├── commands.rs       # IPC handlers
│       └── storage.rs        # Settings persistence
└── package.json
```

### Type Checking

```bash
pnpm typecheck
```

### Linting

```bash
pnpm lint
pnpm lint:fix
```

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| **Space** | Play/Pause |
| **M** | Toggle mute |
| **F** | Toggle fullscreen |
| **I** | Toggle sidebar |
| **←/→** | Seek ±5s |
| **↑/↓** | Volume ±10% |
| **Esc** | Close modal/panel |

## Design System

Uses **VHS Quantum** aesthetic:
- **Fonts**: Orbitron (display), Rajdhani (heading), IBM Plex Sans (body), IBM Plex Mono (mono)
- **Colors**: Portal Cyan (#00FFC6), Dimensional Magenta (#FF3399), Void (#0D0221)
- **Effects**: Scanlines, noise overlay, portal transitions, glitch animations

See `.tasks/design-system/` for full specifications.

## Configuration

### Settings Location

Settings are persisted to:
- **macOS**: `~/Library/Application Support/icn-viewer/settings.json`
- **Windows**: `%APPDATA%\icn-viewer\settings.json`
- **Linux**: `~/.config/icn-viewer/settings.json`

### Tauri Config

Edit `src-tauri/tauri.conf.json` to change:
- Window dimensions
- App identifier
- Build settings

## Troubleshooting

### WebCodecs not supported

Requires Chrome 94+, Edge 94+, or Safari 16.4+. Update your browser/OS.

### P2P discovery fails

App falls back to hardcoded relays. Check firewall settings if DHT queries are blocked.

### High memory usage

Reduce buffer size or quality in settings. Check browser DevTools for leaks.

## License

MIT

## Related

- [ICN Chain](../icn-chain/) - Polkadot SDK runtime
- [Director Node](../icn-nodes/director/) - Video generation
- [Regional Relay](../icn-nodes/relay/) - Content distribution
