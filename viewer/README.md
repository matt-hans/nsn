# ICN Viewer

Desktop application for watching Interdimensional Cable, built with Tauri 2.0 + React.

## Features

- Stream video from ICN network via WebCodecs
- Optional P2P seeding to help distribute content
- Wallet integration for staking (future)

## Development

```bash
# Install dependencies
pnpm install

# Run in development mode
pnpm tauri:dev

# Build for production
pnpm tauri:build
```

## Architecture

- **Frontend:** React 18 + Zustand for state
- **Backend:** Tauri (Rust) with optional libp2p for seeding
- **Video:** WebCodecs API for low-latency playback

## Task

- T013: Viewer client implementation (Phase 2)
