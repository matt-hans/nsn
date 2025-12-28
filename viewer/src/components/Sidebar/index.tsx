// ICN Viewer Client - Sidebar Component

import { useAppStore } from "../../store/appStore";

export default function Sidebar() {
	const {
		showSidebar,
		currentSlot,
		directorPeerId,
		directorReputation,
		bitrate,
		latency,
		connectedPeers,
		bufferSeconds,
		uploadedBytes,
		seedingEnabled,
		toggleSidebar,
	} = useAppStore();

	const formatBytes = (bytes: number) => {
		if (bytes < 1024) return `${bytes} B`;
		if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
		if (bytes < 1024 * 1024 * 1024)
			return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
		return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GB`;
	};

	return (
		<>
			<div
				className={`sidebar ${showSidebar ? "open" : ""}`}
				role="complementary"
				aria-label="Stream information"
			>
				{/* Slot Info */}
				<section className="sidebar-section">
					<h2 className="sidebar-section-title">Current Slot</h2>
					<div className="slot-number-large">
						{currentSlot.toString().padStart(5, "0")}
					</div>
				</section>

				{/* Director Info */}
				<section className="sidebar-section">
					<h2 className="sidebar-section-title">Director</h2>
					<div className="director-card">
						<div className="director-id">{directorPeerId || "Unknown"}</div>
						<div className="stat-row">
							<span className="stat-label">Reputation</span>
							<span className="stat-value highlight">{directorReputation}</span>
						</div>
						<div className="reputation-bar">
							<div
								className="reputation-fill"
								style={{ width: `${Math.min(directorReputation / 10, 100)}%` }}
							/>
						</div>
					</div>
				</section>

				{/* Network Stats */}
				<section className="sidebar-section">
					<h2 className="sidebar-section-title">Network Stats</h2>
					<div className="stat-row">
						<span className="stat-label">Bitrate</span>
						<span className="stat-value">{bitrate.toFixed(1)} Mbps</span>
					</div>
					<div className="stat-row">
						<span className="stat-label">Latency</span>
						<span className="stat-value">{latency} ms</span>
					</div>
					<div className="stat-row">
						<span className="stat-label">Connected Peers</span>
						<span className="stat-value">{connectedPeers}</span>
					</div>
					<div className="stat-row">
						<span className="stat-label">Buffer</span>
						<span className="stat-value">{bufferSeconds.toFixed(1)}s</span>
					</div>
				</section>

				{/* Seeding Stats */}
				{seedingEnabled && (
					<section className="sidebar-section">
						<h2 className="sidebar-section-title">Seeding</h2>
						<div className="stat-row">
							<span className="stat-label">Uploaded</span>
							<span className="stat-value highlight">
								{formatBytes(uploadedBytes)}
							</span>
						</div>
						<div className="stat-row">
							<span className="stat-label">Earned</span>
							<span className="stat-value highlight">
								{((uploadedBytes / (1024 * 1024 * 1024)) * 0.001).toFixed(4)}{" "}
								ICN
							</span>
						</div>
					</section>
				)}
			</div>

			{/* Backdrop */}
			{showSidebar && (
				<div
					className="sidebar-backdrop"
					onClick={toggleSidebar}
					onKeyDown={(e) => {
						if (e.key === "Enter" || e.key === " " || e.key === "Escape") {
							e.preventDefault();
							toggleSidebar();
						}
					}}
					role="button"
					tabIndex={0}
					aria-label="Close sidebar"
				/>
			)}
		</>
	);
}
