// ICN Viewer Client - Main App Component

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
	// Settings are automatically loaded from localStorage by Zustand persist middleware
	useEffect(() => {
		const init = async () => {
			try {
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
