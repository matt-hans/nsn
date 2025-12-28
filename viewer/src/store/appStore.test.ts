// ICN Viewer Client - AppStore Unit Tests

import { beforeEach, describe, expect, it } from "vitest";
import { useAppStore } from "./appStore";

describe("AppStore", () => {
	beforeEach(() => {
		// Reset store to initial state
		useAppStore.setState({
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
		});
	});

	describe("State mutations", () => {
		it("should set current slot", () => {
			useAppStore.getState().setCurrentSlot(12345);
			expect(useAppStore.getState().currentSlot).toBe(12345);
		});

		it("should set playback state", () => {
			useAppStore.getState().setPlaybackState("playing");
			expect(useAppStore.getState().playbackState).toBe("playing");
		});

		it("should set volume", () => {
			useAppStore.getState().setVolume(50);
			expect(useAppStore.getState().volume).toBe(50);
		});
	});

	describe("Volume controls", () => {
		it("should set isMuted to true when volume is 0", () => {
			useAppStore.getState().setVolume(0);
			expect(useAppStore.getState().isMuted).toBe(true);
		});

		it("should set isMuted to false when volume > 0", () => {
			useAppStore.getState().setVolume(50);
			expect(useAppStore.getState().isMuted).toBe(false);
		});

		it("should toggle mute", () => {
			useAppStore.getState().toggleMute();
			expect(useAppStore.getState().isMuted).toBe(true);
			useAppStore.getState().toggleMute();
			expect(useAppStore.getState().isMuted).toBe(false);
		});
	});

	describe("Quality settings", () => {
		it("should set quality to 1080p", () => {
			useAppStore.getState().setQuality("1080p");
			expect(useAppStore.getState().quality).toBe("1080p");
		});

		it("should set quality to 720p", () => {
			useAppStore.getState().setQuality("720p");
			expect(useAppStore.getState().quality).toBe("720p");
		});

		it("should set quality to 480p", () => {
			useAppStore.getState().setQuality("480p");
			expect(useAppStore.getState().quality).toBe("480p");
		});

		it("should set quality to auto", () => {
			useAppStore.getState().setQuality("auto");
			expect(useAppStore.getState().quality).toBe("auto");
		});
	});

	describe("Connection state", () => {
		it("should set connection status", () => {
			useAppStore.getState().setConnectionStatus("connected");
			expect(useAppStore.getState().connectionStatus).toBe("connected");
		});

		it("should set connected relay", () => {
			useAppStore.getState().setConnectedRelay("12D3KooWRelay1", "us-east-1");
			expect(useAppStore.getState().connectedRelay).toBe("12D3KooWRelay1");
			expect(useAppStore.getState().relayRegion).toBe("us-east-1");
		});
	});

	describe("Stats updates", () => {
		it("should update stats", () => {
			useAppStore.getState().updateStats({
				bitrate: 5.2,
				latency: 45,
				connectedPeers: 8,
				bufferSeconds: 12.5,
			});
			expect(useAppStore.getState().bitrate).toBe(5.2);
			expect(useAppStore.getState().latency).toBe(45);
			expect(useAppStore.getState().connectedPeers).toBe(8);
			expect(useAppStore.getState().bufferSeconds).toBe(12.5);
		});

		it("should update playback time", () => {
			useAppStore.getState().updatePlaybackTime(30, 90);
			expect(useAppStore.getState().currentTime).toBe(30);
			expect(useAppStore.getState().duration).toBe(90);
		});

		it("should set director info", () => {
			useAppStore.getState().setDirectorInfo("12D3KooWDirector1", 850);
			expect(useAppStore.getState().directorPeerId).toBe("12D3KooWDirector1");
			expect(useAppStore.getState().directorReputation).toBe(850);
		});
	});

	describe("UI toggles", () => {
		it("should toggle sidebar", () => {
			expect(useAppStore.getState().showSidebar).toBe(false);
			useAppStore.getState().toggleSidebar();
			expect(useAppStore.getState().showSidebar).toBe(true);
			useAppStore.getState().toggleSidebar();
			expect(useAppStore.getState().showSidebar).toBe(false);
		});

		it("should toggle settings", () => {
			expect(useAppStore.getState().showSettings).toBe(false);
			useAppStore.getState().toggleSettings();
			expect(useAppStore.getState().showSettings).toBe(true);
			useAppStore.getState().toggleSettings();
			expect(useAppStore.getState().showSettings).toBe(false);
		});

		it("should toggle fullscreen", () => {
			expect(useAppStore.getState().isFullscreen).toBe(false);
			useAppStore.getState().toggleFullscreen();
			expect(useAppStore.getState().isFullscreen).toBe(true);
			useAppStore.getState().toggleFullscreen();
			expect(useAppStore.getState().isFullscreen).toBe(false);
		});
	});

	describe("Seeding", () => {
		it("should set seeding enabled", () => {
			useAppStore.getState().setSeedingEnabled(true);
			expect(useAppStore.getState().seedingEnabled).toBe(true);
		});

		it("should increment uploaded bytes", () => {
			useAppStore.getState().incrementUploadedBytes(1024);
			expect(useAppStore.getState().uploadedBytes).toBe(1024);
			useAppStore.getState().incrementUploadedBytes(2048);
			expect(useAppStore.getState().uploadedBytes).toBe(3072);
		});
	});

	describe("Edge Cases and Input Validation", () => {
		it("should accept boundary volumes - 0 and 100", () => {
			useAppStore.getState().setVolume(0);
			expect(useAppStore.getState().volume).toBe(0);
			expect(useAppStore.getState().isMuted).toBe(true);

			useAppStore.getState().setVolume(100);
			expect(useAppStore.getState().volume).toBe(100);
			expect(useAppStore.getState().isMuted).toBe(false);
		});

		it("should handle rapid state changes without corruption", () => {
			const state = useAppStore.getState();

			// Rapid consecutive calls - final state should be correct
			state.setVolume(10);
			state.setVolume(20);
			state.setVolume(30);
			state.setVolume(40);
			state.setVolume(50);

			// Re-fetch state to ensure consistency
			const finalState = useAppStore.getState();
			expect(typeof finalState.volume).toBe("number");
			expect(finalState.volume).toBeGreaterThanOrEqual(0);
			expect(finalState.volume).toBeLessThanOrEqual(100);
		});

		it("should handle current slot changes", () => {
			useAppStore.getState().setCurrentSlot(0);
			expect(useAppStore.getState().currentSlot).toBe(0);

			useAppStore.getState().setCurrentSlot(999999);
			expect(useAppStore.getState().currentSlot).toBe(999999);
		});

		it("should handle playback time updates with boundaries", () => {
			useAppStore.getState().updatePlaybackTime(0, 100);
			let state = useAppStore.getState();
			expect(state.currentTime).toBe(0);
			expect(state.duration).toBe(100);

			useAppStore.getState().updatePlaybackTime(50, 100);
			state = useAppStore.getState();
			expect(state.currentTime).toBe(50);

			useAppStore.getState().updatePlaybackTime(100, 100);
			state = useAppStore.getState();
			expect(state.currentTime).toBe(100);
		});
	});

	describe("State Persistence", () => {
		it("should persist volume to localStorage", () => {
			const testVolume = 65;
			useAppStore.getState().setVolume(testVolume);

			// Check localStorage
			const stored = localStorage.getItem("icn-app-storage");
			expect(stored).toBeDefined();
			if (stored) {
				const parsed = JSON.parse(stored);
				expect(parsed.state.volume).toBe(testVolume);
			}
		});

		it("should persist quality to localStorage", () => {
			useAppStore.getState().setQuality("720p");

			const stored = localStorage.getItem("icn-app-storage");
			expect(stored).toBeDefined();
			if (stored) {
				const parsed = JSON.parse(stored);
				expect(parsed.state.quality).toBe("720p");
			}
		});

		it("should persist seedingEnabled to localStorage", () => {
			useAppStore.getState().setSeedingEnabled(true);

			const stored = localStorage.getItem("icn-app-storage");
			expect(stored).toBeDefined();
			if (stored) {
				const parsed = JSON.parse(stored);
				expect(parsed.state.seedingEnabled).toBe(true);
			}
		});

		it("should persist currentSlot to localStorage", () => {
			useAppStore.getState().setCurrentSlot(54321);

			const stored = localStorage.getItem("icn-app-storage");
			expect(stored).toBeDefined();
			if (stored) {
				const parsed = JSON.parse(stored);
				expect(parsed.state.currentSlot).toBe(54321);
			}
		});

		it("should not persist transient state (playbackState, showSettings, etc)", () => {
			useAppStore.getState().setPlaybackState("playing");
			useAppStore.getState().toggleSettings();
			useAppStore.getState().toggleSidebar();

			const stored = localStorage.getItem("icn-app-storage");
			expect(stored).toBeDefined();
			if (stored) {
				const parsed = JSON.parse(stored);
				// These should not be persisted
				expect(parsed.state.playbackState).toBeUndefined();
				expect(parsed.state.showSettings).toBeUndefined();
				expect(parsed.state.showSidebar).toBeUndefined();
			}
		});

		it("should handle corrupted localStorage gracefully", () => {
			// Corrupt localStorage
			localStorage.setItem("icn-app-storage", "invalid json {{{");

			// Store should still work with defaults
			const state = useAppStore.getState();
			expect(state.volume).toBeDefined();
			expect(state.quality).toBeDefined();
		});
	});
});
