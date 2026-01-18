// ICN Viewer Client - Main App Component

import { useEffect } from "react";
import AppShell from "./components/AppShell";
import { BootstrapOverlay } from "./components/BootstrapOverlay";
import SettingsModal from "./components/SettingsModal";
import Sidebar from "./components/Sidebar";
import TopBar from "./components/TopBar";
import VideoPlayer from "./components/VideoPlayer";
import { useKeyboardShortcuts } from "./hooks/useKeyboardShortcuts";
import { useP2PConnection } from "./hooks/useP2PConnection";
import { useAppStore } from "./store/appStore";

function App() {
	const { showSidebar, showSettings, connectionStatus } = useAppStore();
	const { connect } = useP2PConnection();

	// Keyboard shortcuts
	useKeyboardShortcuts();

	// Auto-connect to P2P mesh on mount
	useEffect(() => {
		connect();
	}, [connect]);

	// Show full-screen overlay during bootstrap (per CONTEXT.md)
	// Only show overlay during connection phases, not when explicitly disconnected
	const showBootstrapOverlay =
		connectionStatus !== "connected" && connectionStatus !== "disconnected";

	return (
		<>
			{showBootstrapOverlay && <BootstrapOverlay />}
			<AppShell>
				<TopBar />
				<VideoPlayer />
				{showSidebar && <Sidebar />}
				{showSettings && <SettingsModal />}
			</AppShell>
		</>
	);
}

export default App;
