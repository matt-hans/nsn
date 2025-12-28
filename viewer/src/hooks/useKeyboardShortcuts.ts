// ICN Viewer Client - Keyboard Shortcuts Hook

import { useEffect } from "react";
import { useAppStore } from "../store/appStore";

export function useKeyboardShortcuts() {
	useEffect(() => {
		const handleKeyPress = (e: KeyboardEvent) => {
			const state = useAppStore.getState();

			// Ignore if typing in input
			if (
				e.target instanceof HTMLInputElement ||
				e.target instanceof HTMLTextAreaElement
			) {
				return;
			}

			switch (e.key.toLowerCase()) {
				case " ":
					e.preventDefault();
					state.setPlaybackState(
						state.playbackState === "playing" ? "paused" : "playing",
					);
					break;
				case "m":
					state.toggleMute();
					break;
				case "f":
					state.toggleFullscreen();
					break;
				case "i":
					state.toggleSidebar();
					break;
				case "escape":
					if (state.showSettings) state.toggleSettings();
					if (state.showSidebar) state.toggleSidebar();
					if (state.isFullscreen) state.toggleFullscreen();
					break;
				case "arrowleft":
					// Seek backward 5s
					state.updatePlaybackTime(
						Math.max(0, state.currentTime - 5),
						state.duration,
					);
					break;
				case "arrowright":
					// Seek forward 5s
					state.updatePlaybackTime(
						Math.min(state.duration, state.currentTime + 5),
						state.duration,
					);
					break;
				case "arrowup":
					// Volume up 10%
					state.setVolume(Math.min(100, state.volume + 10));
					break;
				case "arrowdown":
					// Volume down 10%
					state.setVolume(Math.max(0, state.volume - 10));
					break;
			}
		};

		window.addEventListener("keydown", handleKeyPress);
		return () => window.removeEventListener("keydown", handleKeyPress);
	}, []);
}
