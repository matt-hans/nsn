// ICN Viewer Client - Main App Component

import { invoke } from "@tauri-apps/api/core";
import { useEffect } from "react";
import AppShell from "./components/AppShell";
import SettingsModal from "./components/SettingsModal";
import Sidebar from "./components/Sidebar";
import TopBar from "./components/TopBar";
import VideoPlayer from "./components/VideoPlayer";
import { useKeyboardShortcuts } from "./hooks/useKeyboardShortcuts";
import { discoverRelays } from "./services/p2p";
import { useAppStore } from "./store/appStore";

function App() {
	const { showSidebar, showSettings, setConnectionStatus, setConnectedRelay } =
		useAppStore();

	// Keyboard shortcuts
	useKeyboardShortcuts();

	// Initialize app on mount
	useEffect(() => {
		const init = async () => {
			try {
				// Load persisted settings from Tauri backend
				try {
					const settings = await invoke<{
						volume: number;
						quality: string;
						seeding_enabled: boolean;
						last_slot: number | null;
					}>("load_settings");

					useAppStore.getState().setVolume(settings.volume);
					// Validate quality value from backend before setting
					const validQualities = ["1080p", "720p", "480p", "auto"] as const;
					if (
						validQualities.includes(
							settings.quality as (typeof validQualities)[number],
						)
					) {
						useAppStore
							.getState()
							.setQuality(settings.quality as (typeof validQualities)[number]);
					} else {
						useAppStore.getState().setQuality("auto");
					}
					useAppStore.getState().setSeedingEnabled(settings.seeding_enabled);
					if (settings.last_slot) {
						useAppStore.getState().setCurrentSlot(settings.last_slot);
					}
				} catch (error) {
					console.warn("Failed to load settings from backend:", error);
					// Continue with default settings from Zustand store
				}

				// Discover relays via P2P
				setConnectionStatus("connecting");
				const relays = await discoverRelays();

				if (relays.length > 0) {
					const relay = relays[0]; // Select first relay (could optimize by latency)
					setConnectedRelay(relay.peer_id, relay.region);
					setConnectionStatus("connected");
				} else {
					setConnectionStatus("error");
				}
			} catch (error) {
				console.error("Initialization error:", error);
				setConnectionStatus("error");
			}
		};

		init();
	}, [setConnectionStatus, setConnectedRelay]);

	return (
		<AppShell>
			<TopBar />
			<VideoPlayer />
			{showSidebar && <Sidebar />}
			{showSettings && <SettingsModal />}
		</AppShell>
	);
}

export default App;
