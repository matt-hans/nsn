// ICN Viewer Client - P2P Connection Hook
// React hook for P2P lifecycle management with reconnection

import { useCallback, useEffect, useRef } from "react";
import { buildCandidateList, discoverWithRace } from "../services/discovery";
import { P2PClient } from "../services/p2pClient";
import {
	connectP2PToPipeline,
	getVideoPipeline,
} from "../services/videoPipeline";
import { useAppStore } from "../store/appStore";

export interface P2PConnectionState {
	connect: () => Promise<void>;
	disconnect: () => void;
	isConnected: boolean;
	isConnecting: boolean;
	client: P2PClient | null;
}

/**
 * React hook for managing P2P connection lifecycle
 *
 * Features:
 * - Automatic reconnection with exponential backoff
 * - Bootstrap progress tracking for UX
 * - Discovery with parallel race pattern
 * - Automatic cleanup on unmount
 *
 * @returns P2P connection state and control functions
 */
export function useP2PConnection(): P2PConnectionState {
	const p2pClientRef = useRef<P2PClient | null>(null);
	const reconnectTimeoutRef = useRef<number | null>(null);
	const reconnectAttemptRef = useRef(0);

	const {
		connectionStatus,
		setConnectionStatus,
		setConnectedPeerId,
		setMeshPeerCount,
		setConnectionError,
		setBootstrapProgress,
		setLastConnectedNodeUrl,
	} = useAppStore();

	/**
	 * Main connection function
	 * 1. Initialize P2P client
	 * 2. Discover mesh node via HTTP
	 * 3. Dial discovered node
	 * 4. Subscribe to video topic
	 * 5. Handle errors with exponential backoff
	 */
	const connect = useCallback(async () => {
		// Clear any pending reconnect
		if (reconnectTimeoutRef.current) {
			clearTimeout(reconnectTimeoutRef.current);
			reconnectTimeoutRef.current = null;
		}

		setConnectionStatus("connecting");
		setConnectionError(null);
		setBootstrapProgress("discovering", "Connecting to Swarm...");

		try {
			// 1. Initialize P2P client
			const client = new P2PClient();
			await client.initialize();
			p2pClientRef.current = client;
			setConnectedPeerId(client.getPeerId());

			// 2. Discover a mesh node
			const candidates = buildCandidateList();
			setBootstrapProgress("connecting", "Finding mesh nodes...");
			const multiaddrs = await discoverWithRace(candidates);
			if (multiaddrs.length === 0) {
				throw new Error("No WebRTC addresses returned from discovery");
			}

			// Extract node URL from multiaddr for persistence
			const urlMatch = multiaddrs[0].match(/ip4\/([^/]+)\//);
			if (urlMatch) {
				const nodeUrl = `http://${urlMatch[1]}:9615`;
				setLastConnectedNodeUrl(nodeUrl);
			}

			// 3. Dial the discovered node
			setBootstrapProgress("connecting", "Negotiating NAT traversal...");
			await client.dialAny(multiaddrs);

			// 4. Subscribe to video topic
			setBootstrapProgress("subscribing", "Joining video channel...");
			const pipeline = getVideoPipeline();
			if (pipeline) {
				connectP2PToPipeline(client, pipeline);
			}

			// 5. Success
			setConnectionStatus("connected");
			setBootstrapProgress("ready", "Mesh Active");
			setMeshPeerCount(client.getConnectedPeers());
			reconnectAttemptRef.current = 0;

			console.log(
				`[P2P] Connected successfully, peer count: ${client.getConnectedPeers()}`,
			);
		} catch (error) {
			const message =
				error instanceof Error ? error.message : "Connection failed";
			console.error("[P2P] Connection failed:", message);
			setConnectionStatus("error");
			setConnectionError(message);
			setBootstrapProgress("error", message);

			// Schedule reconnection with exponential backoff
			scheduleReconnect();
		}
	}, [
		setConnectionStatus,
		setConnectedPeerId,
		setMeshPeerCount,
		setConnectionError,
		setBootstrapProgress,
		setLastConnectedNodeUrl,
	]);

	/**
	 * Schedule reconnection with exponential backoff
	 * Delay: 1s, 2s, 4s, 8s, 16s, 30s (max)
	 */
	const scheduleReconnect = useCallback(() => {
		const attempt = reconnectAttemptRef.current;
		const delay = Math.min(1000 * 2 ** attempt, 30000); // Max 30s
		reconnectAttemptRef.current = attempt + 1;

		console.log(
			`[P2P] Scheduling reconnect in ${delay}ms (attempt ${attempt + 1})`,
		);
		setBootstrapProgress(
			"error",
			`Reconnecting in ${Math.ceil(delay / 1000)}s...`,
		);

		reconnectTimeoutRef.current = window.setTimeout(() => {
			connect();
		}, delay);
	}, [connect, setBootstrapProgress]);

	/**
	 * Disconnect from P2P network
	 * Stops client, clears refs, resets state
	 */
	const disconnect = useCallback(() => {
		if (reconnectTimeoutRef.current) {
			clearTimeout(reconnectTimeoutRef.current);
			reconnectTimeoutRef.current = null;
		}

		if (p2pClientRef.current) {
			p2pClientRef.current.stop();
			p2pClientRef.current = null;
		}

		setConnectionStatus("disconnected");
		setConnectedPeerId(null);
		setMeshPeerCount(0);
		setBootstrapProgress("idle", "");
	}, [
		setConnectionStatus,
		setConnectedPeerId,
		setMeshPeerCount,
		setBootstrapProgress,
	]);

	// Cleanup on unmount
	useEffect(() => {
		return () => {
			disconnect();
		};
	}, [disconnect]);

	return {
		connect,
		disconnect,
		isConnected: connectionStatus === "connected",
		isConnecting: connectionStatus === "connecting",
		client: p2pClientRef.current,
	};
}
