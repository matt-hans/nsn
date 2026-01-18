// ICN Viewer Client - P2P Service (Legacy Stub)
// This file provides backward compatibility during migration to libp2p
// TODO: Remove after migration is complete

// Re-export types from shared types file
export type { VideoChunkMessage } from "./types";

// Stub implementation to allow compilation
// These functions will be replaced by new P2PClient implementation

export interface RelayInfo {
	peer_id: string;
	multiaddr: string;
	region: string;
	latency_ms?: number;
	is_fallback: boolean;
}

// Fallback relays for when signaling server is unavailable
const FALLBACK_RELAYS: RelayInfo[] = [
	{
		peer_id: "12D3KooWDpJ7As7BWAwRMfu1VU2WCqNjvq387JEYKDBj4kx6nXTN",
		multiaddr: "/dns4/relay1.icn.network/tcp/4001/wss",
		region: "us-east",
		is_fallback: true,
	},
	{
		peer_id: "12D3KooWQYV9dGMFoRzNSJ4s3qXHGsF7hYVmMpYu4zUQ9jZv3p2N",
		multiaddr: "/dns4/relay2.icn.network/tcp/4001/wss",
		region: "eu-west",
		is_fallback: true,
	},
];

// Module-level state for legacy API
let videoChunkHandler: ((msg: import("./types").VideoChunkMessage) => void) | null = null;
let mockStreamInterval: number | null = null;
let isConnected = false;

/**
 * Discover relays via signaling server or fallback to hardcoded list
 * @deprecated Use discovery.ts instead
 */
export async function discoverRelays(): Promise<RelayInfo[]> {
	console.warn("[p2p.ts] discoverRelays is deprecated. Use discovery.ts instead.");
	return FALLBACK_RELAYS;
}

/**
 * Connect to relay (legacy API stub)
 * @deprecated Use P2PClient instead
 */
export async function connectToRelay(_relay: RelayInfo): Promise<boolean> {
	console.warn("[p2p.ts] connectToRelay is deprecated. Use P2PClient instead.");
	return false;
}

/**
 * Subscribe to video chunks from relay (legacy API stub)
 * @deprecated Use P2PClient.subscribeToVideoTopic instead
 */
export function onVideoChunk(handler: (msg: import("./types").VideoChunkMessage) => void): void {
	console.warn("[p2p.ts] onVideoChunk is deprecated. Use P2PClient.subscribeToVideoTopic instead.");
	videoChunkHandler = handler;
}

/**
 * Disconnect from relay (legacy API stub)
 * @deprecated Use P2PClient.stop instead
 */
export function disconnect(): void {
	console.warn("[p2p.ts] disconnect is deprecated. Use P2PClient.stop instead.");
	isConnected = false;

	if (mockStreamInterval !== null) {
		clearInterval(mockStreamInterval);
		mockStreamInterval = null;
	}

	videoChunkHandler = null;
}

/**
 * Get connection status (legacy API stub)
 * @deprecated Use P2PClient.isStarted instead
 */
export function getConnectionStatus(): boolean {
	return isConnected;
}

/**
 * Start mock video stream for development/testing
 * Generates synthetic 24fps chunks
 * @deprecated Use connectP2PToPipeline instead
 */
export function startMockVideoStream(slotNumber: number): void {
	console.warn("[p2p.ts] startMockVideoStream is deprecated.");
	if (!videoChunkHandler) {
		console.warn("No video chunk handler registered");
		return;
	}

	let chunkIndex = 0;
	const fps = 24;

	mockStreamInterval = window.setInterval(() => {
		if (videoChunkHandler && isConnected) {
			// Generate mock chunk (empty data for testing)
			const chunk: import("./types").VideoChunkMessage = {
				slot: slotNumber,
				chunk_index: chunkIndex,
				data: new Uint8Array(1024), // 1KB mock data
				timestamp: Date.now() * 1000, // microseconds
				is_keyframe: chunkIndex % 24 === 0, // Keyframe every second
			};

			videoChunkHandler(chunk);
			chunkIndex++;
		}
	}, 1000 / fps);
}
