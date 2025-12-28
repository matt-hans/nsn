// ICN Viewer Client - Top Bar Component

import { useAppStore } from "../store/appStore";

export default function TopBar() {
	const { connectionStatus, relayRegion, toggleSettings } = useAppStore();

	const getStatusColor = () => {
		switch (connectionStatus) {
			case "connected":
				return "status-connected";
			case "connecting":
				return "status-buffering";
			case "error":
			case "disconnected":
				return "status-error";
			default:
				return "";
		}
	};

	const getStatusText = () => {
		switch (connectionStatus) {
			case "connected":
				return `Connected to ${relayRegion || "Relay"}`;
			case "connecting":
				return "Connecting...";
			case "error":
				return "Connection Failed";
			case "disconnected":
				return "Disconnected";
			default:
				return "";
		}
	};

	return (
		<div className="topbar">
			<div className="topbar-logo">ICN</div>

			<div className="connection-status">
				<div className={`status-dot ${getStatusColor()}`} />
				<span>{getStatusText()}</span>
			</div>

			<button
				type="button"
				className="icon-button"
				onClick={toggleSettings}
				aria-label="Settings"
			>
				<svg
					width="22"
					height="22"
					viewBox="0 0 24 24"
					fill="none"
					stroke="currentColor"
					strokeWidth="2"
				>
					<title>Settings</title>
					<circle cx="12" cy="12" r="3" />
					<path d="M12 1v6m0 6v6m-5-13L8.5 9m2.5 6l-1.5 2.5m10-10.5L17 9m-2.5 6l1.5 2.5m2.5-10l-6 3m-6 0l-6-3m6 6l-6 3m12 0l6-3" />
				</svg>
			</button>
		</div>
	);
}
