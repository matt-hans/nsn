// ICN Viewer Client - Bootstrap Overlay Component
// Full-screen loading screen during P2P mesh connection

import { useEffect, useState } from "react";
import { useAppStore } from "../store/appStore";

/**
 * BootstrapOverlay - Full-screen connection bootstrap UI
 *
 * Per CONTEXT.md spec:
 * - Terminal-style aesthetic with green-on-black
 * - Progressive messaging during connection phases
 * - 30s timeout with manual bootstrap option
 * - Expandable diagnostics on error
 * - Retry and manual bootstrap buttons
 */
export function BootstrapOverlay() {
	const { bootstrapProgress, connectionError } = useAppStore();
	const [showManual, setShowManual] = useState(false);

	// After 30s in error/connecting state, show manual option
	useEffect(() => {
		if (bootstrapProgress.startedAt) {
			const elapsed = Date.now() - bootstrapProgress.startedAt;
			if (elapsed > 30000 && bootstrapProgress.phase === "error") {
				setShowManual(true);
			}
		}
	}, [bootstrapProgress.startedAt, bootstrapProgress.phase]);

	// Progressive messaging per CONTEXT.md
	const getMessage = (): string => {
		switch (bootstrapProgress.phase) {
			case "discovering":
				return "Connecting to Swarm...";
			case "connecting":
				return bootstrapProgress.message || "Negotiating NAT traversal...";
			case "subscribing":
				return "Joining video channel...";
			case "error":
				return connectionError || "Unable to join the Neural Sovereign Network";
			default:
				return "Initializing...";
		}
	};

	const isLoading =
		bootstrapProgress.phase !== "ready" && bootstrapProgress.phase !== "idle";

	return (
		<>
			{isLoading && (
				<div className="fixed inset-0 z-50 bg-black flex flex-col items-center justify-center">
					{/* Terminal-style aesthetic */}
					<div className="text-center space-y-6 max-w-md px-4">
						{/* Animated loader */}
						<div className="w-16 h-16 mx-auto border-4 border-green-500 border-t-transparent rounded-full animate-spin" />

						{/* Status message */}
						<p className="text-green-500 font-mono text-lg">{getMessage()}</p>

						{/* Error state buttons */}
						{bootstrapProgress.phase === "error" && (
							<div className="space-y-3 mt-8">
								<button
									type="button"
									onClick={() => window.location.reload()}
									className="w-full px-6 py-3 bg-green-600 hover:bg-green-500 text-white rounded-lg font-medium transition"
								>
									Retry
								</button>
								{showManual && (
									<button
										type="button"
										onClick={() => {
											// Will be connected to settings modal later
											console.log("Open manual bootstrap");
										}}
										className="w-full px-6 py-3 bg-gray-700 hover:bg-gray-600 text-white rounded-lg font-medium transition"
									>
										Try Manual Bootstrap
									</button>
								)}
							</div>
						)}

						{/* Expandable diagnostics for errors */}
						{bootstrapProgress.phase === "error" && connectionError && (
							<details className="text-left text-gray-400 text-sm mt-4">
								<summary className="cursor-pointer hover:text-gray-300">
									[+] View Network Diagnostics
								</summary>
								<pre className="mt-2 p-3 bg-gray-900 rounded font-mono text-xs overflow-auto">
									{connectionError}
								</pre>
							</details>
						)}
					</div>
				</div>
			)}
		</>
	);
}
