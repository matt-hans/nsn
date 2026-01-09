// ICN Viewer Client - P2P Service
// Web-native implementation using WebRTC DataChannel for video chunk delivery

export interface RelayInfo {
	peer_id: string;
	multiaddr: string;
	region: string;
	latency_ms?: number;
	is_fallback: boolean;
}

export interface VideoChunkMessage {
	slot: number;
	chunk_index: number;
	data: Uint8Array;
	timestamp: number;
	is_keyframe: boolean;
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

// Video chunk subscription handlers
let videoChunkHandler: ((msg: VideoChunkMessage) => void) | null = null;
let mockStreamInterval: number | null = null;
let isConnected = false;

/**
 * Discover relays via signaling server or fallback to hardcoded list
 */
export async function discoverRelays(): Promise<RelayInfo[]> {
	// Try signaling server first (when available)
	const signalingUrl =
		import.meta.env.VITE_SIGNALING_URL || "ws://localhost:8080";
	try {
		const httpUrl = signalingUrl
			.replace("ws://", "http://")
			.replace("wss://", "https://");
		const response = await fetch(`${httpUrl}/relays`);
		if (response.ok) {
			return await response.json();
		}
	} catch {
		console.warn("Signaling server unavailable, using fallback relays");
	}
	return FALLBACK_RELAYS;
}

/**
 * Connect to relay via WebTransport (libp2p-js)
 *
 * IMPLEMENTATION STATUS: DEFERRED to T027 (Regional Relay Node)
 * - Requires libp2p-js with WebTransport support
 * - Awaits relay infrastructure deployment (T027)
 * - Currently returns mock success for UI integration testing
 *
 * Future implementation will use:
 * - createLibp2p() with webTransport() transport
 * - GossipSub for /icn/video/1.0.0 topic subscription
 * - Connection to multiaddr from relay discovery
 *
 * Tracked in: T027-regional-relay-node.md
 */
export async function connectToRelay(relay: RelayInfo): Promise<boolean> {
	try {
		console.log("Connecting to relay:", relay.peer_id);
		// DEFERRED: Full libp2p-js implementation in T027
		// const node = await createLibp2p({
		//   transports: [webTransport()],
		// });
		// await node.dial(multiaddr(relay.multiaddr));
		isConnected = true;
		return true; // Mock success for now
	} catch (error) {
		console.error("Failed to connect to relay:", error);
		isConnected = false;
		return false;
	}
}

/**
 * Subscribe to video chunks from relay
 */
export function onVideoChunk(handler: (msg: VideoChunkMessage) => void): void {
	videoChunkHandler = handler;
}

/**
 * Disconnect from relay
 */
export function disconnect(): void {
	isConnected = false;
	if (mockStreamInterval !== null) {
		clearInterval(mockStreamInterval);
		mockStreamInterval = null;
	}
	videoChunkHandler = null;
}

/**
 * Get connection status
 */
export function getConnectionStatus(): boolean {
	return isConnected;
}

/**
 * Start mock video stream for development/testing
 * Generates synthetic 24fps chunks
 */
export function startMockVideoStream(slotNumber: number): void {
	if (!videoChunkHandler) {
		console.warn("No video chunk handler registered");
		return;
	}

	let chunkIndex = 0;
	const fps = 24;

	mockStreamInterval = window.setInterval(() => {
		if (videoChunkHandler && isConnected) {
			// Generate mock chunk (empty data for testing)
			const chunk: VideoChunkMessage = {
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
