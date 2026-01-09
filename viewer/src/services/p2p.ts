// ICN Viewer Client - P2P Service
// Web-native implementation using WebRTC DataChannel for video chunk delivery

import SimplePeer from "simple-peer";
import { SignalingClient, type SignalingMessage } from "./signaling";

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

// ICE servers for STUN/TURN
const ICE_SERVERS: RTCIceServer[] = [
	{ urls: "stun:stun.l.google.com:19302" },
	{ urls: "stun:stun1.l.google.com:19302" },
];

// Video chunk binary format header size
// [slot:4][chunk_index:4][timestamp:8][is_keyframe:1] = 17 bytes
const CHUNK_HEADER_SIZE = 17;

interface PeerConnection {
	id: string;
	peer: SimplePeer.Instance;
	isConnected: boolean;
}

/**
 * P2P Service for WebRTC-based video chunk delivery.
 * Uses simple-peer for WebRTC abstraction and a signaling server for peer discovery.
 */
export class P2PService {
	private peerId: string;
	private peers: Map<string, PeerConnection> = new Map();
	private signaling: SignalingClient | null = null;
	private videoChunkHandler: ((msg: VideoChunkMessage) => void) | null = null;
	private isConnectedToSignaling = false;

	constructor() {
		this.peerId = crypto.randomUUID();
	}

	/**
	 * Get the local peer ID
	 */
	getPeerId(): string {
		return this.peerId;
	}

	/**
	 * Connect to the signaling server and join the P2P network
	 */
	async connect(signalingUrl: string): Promise<boolean> {
		try {
			this.signaling = new SignalingClient(
				this.peerId,
				this.handleSignalingMessage.bind(this),
			);
			await this.signaling.connect(signalingUrl);
			this.isConnectedToSignaling = true;
			return true;
		} catch (error) {
			console.error("Failed to connect to signaling server:", error);
			this.isConnectedToSignaling = false;
			return false;
		}
	}

	/**
	 * Handle incoming signaling messages
	 */
	private handleSignalingMessage(msg: SignalingMessage): void {
		switch (msg.type) {
			case "peer-list":
				this.handlePeerList(msg.payload as string[]);
				break;
			case "offer":
				if (msg.from) {
					this.handleOffer(msg.from, msg.payload as SimplePeer.SignalData);
				}
				break;
			case "answer":
				if (msg.from) {
					this.handleAnswer(msg.from, msg.payload as SimplePeer.SignalData);
				}
				break;
			case "ice-candidate":
				if (msg.from) {
					this.handleIceCandidate(
						msg.from,
						msg.payload as SimplePeer.SignalData,
					);
				}
				break;
		}
	}

	/**
	 * Handle peer list from signaling server
	 */
	private handlePeerList(peerIds: string[]): void {
		// Initiate connections to discovered peers
		for (const peerId of peerIds) {
			if (peerId !== this.peerId && !this.peers.has(peerId)) {
				this.initiatePeerConnection(peerId);
			}
		}
	}

	/**
	 * Initiate a new peer connection as the initiator
	 */
	private initiatePeerConnection(peerId: string): void {
		const peer = new SimplePeer({
			initiator: true,
			trickle: true,
			config: { iceServers: ICE_SERVERS },
		});

		this.setupPeerHandlers(peer, peerId);
		this.peers.set(peerId, { id: peerId, peer, isConnected: false });
	}

	/**
	 * Handle incoming SDP offer
	 */
	private handleOffer(fromPeerId: string, signal: SimplePeer.SignalData): void {
		let peerConn = this.peers.get(fromPeerId);

		if (!peerConn) {
			const peer = new SimplePeer({
				initiator: false,
				trickle: true,
				config: { iceServers: ICE_SERVERS },
			});
			this.setupPeerHandlers(peer, fromPeerId);
			peerConn = { id: fromPeerId, peer, isConnected: false };
			this.peers.set(fromPeerId, peerConn);
		}

		peerConn.peer.signal(signal);
	}

	/**
	 * Handle incoming SDP answer
	 */
	private handleAnswer(
		fromPeerId: string,
		signal: SimplePeer.SignalData,
	): void {
		const peerConn = this.peers.get(fromPeerId);
		if (peerConn) {
			peerConn.peer.signal(signal);
		}
	}

	/**
	 * Handle incoming ICE candidate
	 */
	private handleIceCandidate(
		fromPeerId: string,
		signal: SimplePeer.SignalData,
	): void {
		const peerConn = this.peers.get(fromPeerId);
		if (peerConn) {
			peerConn.peer.signal(signal);
		}
	}

	/**
	 * Set up event handlers for a peer connection
	 */
	private setupPeerHandlers(peer: SimplePeer.Instance, peerId: string): void {
		peer.on("signal", (signal) => {
			// Determine message type based on signal content
			const msgType =
				(signal as { type?: string }).type === "offer"
					? "offer"
					: (signal as { type?: string }).type === "answer"
						? "answer"
						: "ice-candidate";

			this.signaling?.send({
				type: msgType,
				from: this.peerId,
				to: peerId,
				payload: signal,
			});
		});

		peer.on("connect", () => {
			console.log(`Connected to peer: ${peerId}`);
			const peerConn = this.peers.get(peerId);
			if (peerConn) {
				peerConn.isConnected = true;
			}
		});

		peer.on("data", (data: Uint8Array) => {
			// Parse video chunk from binary data
			const chunk = this.parseVideoChunk(data);
			if (chunk && this.videoChunkHandler) {
				this.videoChunkHandler(chunk);
			}
		});

		peer.on("close", () => {
			console.log(`Peer disconnected: ${peerId}`);
			this.peers.delete(peerId);
		});

		peer.on("error", (err) => {
			console.error(`Peer error (${peerId}):`, err);
			this.peers.delete(peerId);
		});
	}

	/**
	 * Parse a video chunk from binary DataChannel message
	 * Binary format: [slot:4][chunk_index:4][timestamp:8][is_keyframe:1][data:rest]
	 */
	private parseVideoChunk(data: Uint8Array): VideoChunkMessage | null {
		if (data.length < CHUNK_HEADER_SIZE) {
			return null;
		}

		const view = new DataView(data.buffer, data.byteOffset, data.byteLength);
		return {
			slot: view.getUint32(0),
			chunk_index: view.getUint32(4),
			timestamp: Number(view.getBigUint64(8)),
			is_keyframe: data[16] === 1,
			data: data.slice(CHUNK_HEADER_SIZE),
		};
	}

	/**
	 * Register a handler for incoming video chunks
	 */
	onVideoChunk(handler: (msg: VideoChunkMessage) => void): void {
		this.videoChunkHandler = handler;
	}

	/**
	 * Get the number of connected peers
	 */
	getConnectedPeerCount(): number {
		let count = 0;
		for (const peerConn of this.peers.values()) {
			if (peerConn.isConnected) {
				count++;
			}
		}
		return count;
	}

	/**
	 * Disconnect from all peers and the signaling server
	 */
	disconnect(): void {
		for (const peerConn of this.peers.values()) {
			peerConn.peer.destroy();
		}
		this.peers.clear();
		this.signaling?.disconnect();
		this.signaling = null;
		this.isConnectedToSignaling = false;
	}

	/**
	 * Check if connected to signaling server
	 */
	isConnected(): boolean {
		return this.isConnectedToSignaling;
	}
}

// ============================================================================
// Legacy API for backward compatibility with existing code
// ============================================================================

// Module-level state for legacy API
let p2pService: P2PService | null = null;
let videoChunkHandler: ((msg: VideoChunkMessage) => void) | null = null;
let mockStreamInterval: number | null = null;
let isConnected = false;

/**
 * Get the singleton P2P service instance
 */
export function getP2PService(): P2PService {
	if (!p2pService) {
		p2pService = new P2PService();
	}
	return p2pService;
}

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
 * Connect to relay (legacy API - now connects via signaling server)
 */
export async function connectToRelay(relay: RelayInfo): Promise<boolean> {
	try {
		console.log("Connecting to relay:", relay.peer_id);
		const service = getP2PService();

		// In WebRTC mode, we connect to signaling server instead of relay directly
		const signalingUrl =
			import.meta.env.VITE_SIGNALING_URL || "ws://localhost:8080";
		const connected = await service.connect(signalingUrl);

		if (connected) {
			// Forward video chunks to legacy handler
			service.onVideoChunk((chunk) => {
				if (videoChunkHandler) {
					videoChunkHandler(chunk);
				}
			});
		}

		isConnected = connected;
		return connected;
	} catch (error) {
		console.error("Failed to connect to relay:", error);
		isConnected = false;
		return false;
	}
}

/**
 * Subscribe to video chunks from relay (legacy API)
 */
export function onVideoChunk(handler: (msg: VideoChunkMessage) => void): void {
	videoChunkHandler = handler;

	// If P2P service exists, also register there
	if (p2pService) {
		p2pService.onVideoChunk(handler);
	}
}

/**
 * Disconnect from relay (legacy API)
 */
export function disconnect(): void {
	isConnected = false;

	if (mockStreamInterval !== null) {
		clearInterval(mockStreamInterval);
		mockStreamInterval = null;
	}

	videoChunkHandler = null;

	if (p2pService) {
		p2pService.disconnect();
		p2pService = null;
	}
}

/**
 * Get connection status (legacy API)
 */
export function getConnectionStatus(): boolean {
	if (p2pService) {
		return p2pService.isConnected() || p2pService.getConnectedPeerCount() > 0;
	}
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
