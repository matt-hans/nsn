// ICN Viewer Client - Network Status Widget
// Persistent network health indicator with hover details

import React from "react";
import { useAppStore } from "../store/appStore";

export interface NetworkStatusProps {
	className?: string;
}

/**
 * NetworkStatus widget showing connection health
 *
 * Per CONTEXT.md spec:
 * - Green dot: Healthy (3+ peers)
 * - Yellow dot: Degraded (low peers, connecting)
 * - Red dot: Disconnected/error
 * - Hover tooltip with node ID, latency, protocol
 *
 * Location: Top-right corner, always visible
 */
export function NetworkStatus({ className = "" }: NetworkStatusProps) {
	const {
		connectionStatus,
		meshPeerCount,
		connectedPeerId,
		latency,
		bootstrapProgress,
	} = useAppStore();

	// Determine status color and text
	let statusColor: string;
	let statusText: string;

	switch (connectionStatus) {
		case "connected":
			if (meshPeerCount >= 3) {
				statusColor = "bg-green-500";
				statusText = `Mesh Active (${meshPeerCount} peers)`;
			} else if (meshPeerCount > 0) {
				statusColor = "bg-yellow-500";
				statusText = `Low Peers (${meshPeerCount})`;
			} else {
				statusColor = "bg-yellow-500";
				statusText = "Connected (0 peers)";
			}
			break;
		case "connecting":
			statusColor = "bg-yellow-500";
			statusText = bootstrapProgress.message || "Connecting...";
			break;
		case "error":
			statusColor = "bg-red-500";
			statusText = "Disconnected";
			break;
		default:
			statusColor = "bg-gray-500";
			statusText = "Disconnected";
			break;
	}

	// Truncate peer ID for display
	const shortPeerId = connectedPeerId
		? `${connectedPeerId.slice(0, 8)}...`
		: null;

	return (
		<div className={`relative group ${className}`}>
			{/* Main indicator */}
			<div className="flex items-center gap-2 px-3 py-1.5 bg-black/60 rounded-lg cursor-default">
				<div className={`w-2 h-2 rounded-full ${statusColor} animate-pulse`} />
				<span className="text-white text-sm">{statusText}</span>
			</div>

			{/* Hover tooltip - expanded details */}
			{connectionStatus === "connected" && (
				<div className="absolute right-0 top-full mt-1 hidden group-hover:block z-50">
					<div className="bg-black/90 text-white text-xs rounded-lg p-3 min-w-48 shadow-lg">
						<div className="space-y-1">
							<div>
								<span className="text-gray-400">Node: </span>
								<span className="font-mono">{shortPeerId || "Unknown"}</span>
							</div>
							<div>
								<span className="text-gray-400">Latency: </span>
								<span>{latency > 0 ? `${latency}ms` : "Measuring..."}</span>
							</div>
							<div>
								<span className="text-gray-400">Protocol: </span>
								<span>WebRTC-Direct</span>
							</div>
						</div>
					</div>
				</div>
			)}
		</div>
	);
}
