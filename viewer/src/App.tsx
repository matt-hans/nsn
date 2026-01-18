// ICN Viewer Client - Main App Component

import { useEffect } from "react";
import AppShell from "./components/AppShell";
import SettingsModal from "./components/SettingsModal";
import Sidebar from "./components/Sidebar";
import TopBar from "./components/TopBar";
import VideoPlayer from "./components/VideoPlayer";
import { useKeyboardShortcuts } from "./hooks/useKeyboardShortcuts";
import { useP2PConnection } from "./hooks/useP2PConnection";
import { useAppStore } from "./store/appStore";

function App() {
	const { showSidebar, showSettings } = useAppStore();
	const { connect } = useP2PConnection();

	// Keyboard shortcuts
	useKeyboardShortcuts();

	// Auto-connect to P2P mesh on mount
	useEffect(() => {
		connect();
	}, [connect]);

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
