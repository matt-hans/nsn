// ICN Viewer Client - Zustand State Management
// Centralized app state with localStorage persistence

import { create } from "zustand";
import { persist } from "zustand/middleware";

export interface AppState {
	// Playback state
	currentSlot: number;
	playbackState: "idle" | "buffering" | "playing" | "paused";
	volume: number;
	isMuted: boolean;
	quality: "1080p" | "720p" | "480p" | "auto";

	// Connection state
	connectionStatus: "disconnected" | "connecting" | "connected" | "error";
	connectedRelay: string | null;
	relayRegion: string | null;

	// Playback stats
	bitrate: number; // Mbps
	latency: number; // ms
	connectedPeers: number;
	bufferSeconds: number;
	currentTime: number;
	duration: number;
	isFullscreen: boolean;

	// Director info
	directorPeerId: string | null;
	directorReputation: number;

	// Settings
	seedingEnabled: boolean;
	uploadedBytes: number;

	// UI state
	showSidebar: boolean;
	showSettings: boolean;
	showControls: boolean;

	// Actions
	setCurrentSlot: (slot: number) => void;
	setPlaybackState: (state: AppState["playbackState"]) => void;
	setVolume: (volume: number) => void;
	toggleMute: () => void;
	setQuality: (quality: AppState["quality"]) => void;
	setConnectionStatus: (status: AppState["connectionStatus"]) => void;
	setConnectedRelay: (relay: string, region: string) => void;
	updateStats: (
		stats: Partial<
			Pick<AppState, "bitrate" | "latency" | "connectedPeers" | "bufferSeconds">
		>,
	) => void;
	updatePlaybackTime: (time: number, duration: number) => void;
	setDirectorInfo: (peerId: string, reputation: number) => void;
	toggleSidebar: () => void;
	toggleSettings: () => void;
	toggleFullscreen: () => void;
	setSeedingEnabled: (enabled: boolean) => void;
	incrementUploadedBytes: (bytes: number) => void;
	setShowControls: (show: boolean) => void;
}

export const useAppStore = create<AppState>()(
	persist(
		(set) => ({
			// Initial state
			currentSlot: 0,
			playbackState: "idle",
			volume: 80,
			isMuted: false,
			quality: "auto",
			connectionStatus: "disconnected",
			connectedRelay: null,
			relayRegion: null,
			bitrate: 0,
			latency: 0,
			connectedPeers: 0,
			bufferSeconds: 0,
			currentTime: 0,
			duration: 0,
			isFullscreen: false,
			directorPeerId: null,
			directorReputation: 0,
			seedingEnabled: false,
			uploadedBytes: 0,
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
			setConnectedRelay: (relay, region) =>
				set({ connectedRelay: relay, relayRegion: region }),
			updateStats: (stats) => set((s) => ({ ...s, ...stats })),
			updatePlaybackTime: (time, duration) =>
				set({ currentTime: time, duration }),
			setDirectorInfo: (peerId, reputation) =>
				set({ directorPeerId: peerId, directorReputation: reputation }),
			toggleSidebar: () => set((s) => ({ showSidebar: !s.showSidebar })),
			toggleSettings: () => set((s) => ({ showSettings: !s.showSettings })),
			toggleFullscreen: () => set((s) => ({ isFullscreen: !s.isFullscreen })),
			setSeedingEnabled: (enabled) => set({ seedingEnabled: enabled }),
			incrementUploadedBytes: (bytes) =>
				set((s) => ({ uploadedBytes: s.uploadedBytes + bytes })),
			setShowControls: (show) => set({ showControls: show }),
		}),
		{
			name: "icn-viewer-storage",
			partialize: (state) => ({
				volume: state.volume,
				quality: state.quality,
				seedingEnabled: state.seedingEnabled,
				currentSlot: state.currentSlot,
			}),
		},
	),
);
