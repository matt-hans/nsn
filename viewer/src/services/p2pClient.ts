// ICN Viewer Client - libp2p-based P2P Client
// WebRTC-Direct transport for browser-to-mesh connectivity

import { gossipsub } from "@chainsafe/libp2p-gossipsub";

/**
 * Error codes for P2P dial failures
 */
export type P2PDialErrorCode =
	| "NOT_INITIALIZED"
	| "INVALID_TRANSPORT"
	| "MISSING_CERTHASH"
	| "MISSING_PEER_ID"
	| "ADDRESS_DENIED"
	| "CONNECTION_TIMEOUT"
	| "CERTIFICATE_MISMATCH"
	| "CONNECTION_REFUSED"
	| "NETWORK_UNREACHABLE"
	| "DIAL_FAILED";

/**
 * Structured error for P2P dial failures with diagnostic information
 */
export class P2PDialError extends Error {
	public readonly code: P2PDialErrorCode;
	public readonly multiaddr: string;
	public readonly cause?: unknown;

	constructor(code: P2PDialErrorCode, message: string, multiaddr: string, cause?: unknown) {
		super(`[${code}] ${message}`);
		this.name = "P2PDialError";
		this.code = code;
		this.multiaddr = multiaddr;
		this.cause = cause;
	}

	/**
	 * Get a user-friendly diagnostic message with remediation hints
	 */
	getDiagnostic(): string {
		switch (this.code) {
			case "NOT_INITIALIZED":
				return "P2P client not ready. Please wait for initialization.";
			case "INVALID_TRANSPORT":
				return "Invalid address format. Browser requires WebRTC-Direct transport.";
			case "MISSING_CERTHASH":
				return "Node may have WebRTC disabled. Ensure node is started with --p2p-enable-webrtc flag.";
			case "MISSING_PEER_ID":
				return "Address format error. Contact node operator.";
			case "ADDRESS_DENIED":
				return "Dial denied. Node may be advertising an undialable address (0.0.0.0/localhost). Check node external address config.";
			case "CONNECTION_TIMEOUT":
				return "Connection timed out. Check if node is running and firewall allows UDP traffic.";
			case "CERTIFICATE_MISMATCH":
				return "Certificate mismatch. Node certificate may have changed - try refreshing.";
			case "CONNECTION_REFUSED":
				return "Connection refused. Verify node is running with WebRTC enabled on the correct port.";
			case "NETWORK_UNREACHABLE":
				return "Network unreachable. Check your internet connection and firewall settings.";
			case "DIAL_FAILED":
				return "Connection failed. See browser console for details.";
			default:
				return this.message;
		}
	}
}
import type { GossipSub } from "@chainsafe/libp2p-gossipsub";
import { noise } from "@chainsafe/libp2p-noise";
import { yamux } from "@chainsafe/libp2p-yamux";
import { identify } from "@libp2p/identify";
import { webRTCDirect } from "@libp2p/webrtc";
import { multiaddr } from "@multiformats/multiaddr";
import type { Multiaddr } from "@multiformats/multiaddr";
import { type Libp2p, createLibp2p } from "libp2p";

/**
 * Video topic for GossipSub
 * Matches Rust node topic for video chunk distribution
 */
export const VIDEO_TOPIC = "/nsn/video/1.0.0";

const DEFAULT_DIAL_TIMEOUT_MS = 8000;

function getDialTimeoutMs(): number {
	const envTimeout = import.meta.env.VITE_P2P_DIAL_TIMEOUT_MS;
	if (envTimeout) {
		const parsed = parseInt(envTimeout, 10);
		if (!isNaN(parsed) && parsed > 0) {
			return parsed;
		}
	}
	return DEFAULT_DIAL_TIMEOUT_MS;
}

/**
 * P2PClient: Web-native P2P client for direct browser-to-mesh connectivity
 * Uses libp2p with WebRTC-Direct transport for outbound connections
 */
export class P2PClient {
	private node: Libp2p | null = null;
	private messageHandlers: Map<string, (data: Uint8Array) => void> = new Map();
	private gossipSub: GossipSub | null = null;
	private static readonly webRtcOnlyGater = (addr: Multiaddr): boolean => {
		const addrStr = addr.toString();
		if (addrStr.includes("/ip4/0.0.0.0/") || addrStr.includes("/ip6/::/")) {
			return true;
		}
		// Allow private/LAN IPs for WebRTC-Direct in local testnets.
		if (addrStr.includes("/webrtc-direct/") || addrStr.includes("/webrtc/")) {
			return false;
		}
		// Deny non-WebRTC transports in the browser.
		return true;
	};

	/**
	 * Initialize the libp2p node with WebRTC-Direct transport
	 * Configures: webRTCDirect, noise encryption, yamux muxing, gossipsub, identify
	 */
	async initialize(): Promise<void> {
		if (this.node) {
			throw new Error("P2PClient already initialized");
		}

		try {
			this.node = await createLibp2p<any>({
				// No listen addresses - browsers cannot accept incoming connections
				addresses: {
					listen: [],
				},
				connectionGater: {
					denyDialMultiaddr: P2PClient.webRtcOnlyGater,
				},
				transports: [webRTCDirect()],
				connectionEncrypters: [noise()],
				streamMuxers: [yamux()],
				services: {
					pubsub: gossipsub({
						emitSelf: false, // Don't receive our own messages
						fallbackToFloodsub: false, // Use GossipSub only
					}),
					identify: identify(),
				},
			});

			this.gossipSub = this.node.services.pubsub as GossipSub;

			// Set up message handler for GossipSub
			this.gossipSub.addEventListener("message", (evt: any) => {
				const message = evt.detail;
				const topic = message.topic;
				const data = message.data;
				const handler = this.messageHandlers.get(topic);
				if (handler) {
					handler(data);
				}
			});

			await this.node.start();
		} catch (error) {
			throw new Error(`Failed to initialize libp2p node: ${error}`);
		}
	}

	/**
	 * Dial a peer at the given multiaddr
	 * Parses the multiaddr string and establishes outbound connection
	 *
	 * @throws P2PDialError with detailed diagnostic information
	 */
	async dial(
		multiaddrString: string,
		options?: { signal?: AbortSignal }
	): Promise<void> {
		if (!this.node) {
			throw new P2PDialError("NOT_INITIALIZED", "P2PClient not initialized. Call initialize() first.", multiaddrString);
		}

		// Validate multiaddr format
		if (!multiaddrString.includes("/webrtc-direct/")) {
			throw new P2PDialError(
				"INVALID_TRANSPORT",
				"Multiaddr must use /webrtc-direct/ transport for browser connections",
				multiaddrString
			);
		}

		if (!multiaddrString.includes("/certhash/")) {
			throw new P2PDialError(
				"MISSING_CERTHASH",
				"Multiaddr missing /certhash/ - node may have WebRTC disabled or certificate not generated",
				multiaddrString
			);
		}

		if (!multiaddrString.includes("/p2p/")) {
			throw new P2PDialError(
				"MISSING_PEER_ID",
				"Multiaddr missing /p2p/<peer_id> suffix",
				multiaddrString
			);
		}

			try {
				console.log(`[P2P] Dialing: ${multiaddrString.slice(0, 80)}...`);
				const ma = multiaddr(multiaddrString);
				await this.node.dial(ma, options);
				console.log(`[P2P] Successfully connected to peer`);
			} catch (error) {
				// Log full error details for debugging
				const errorMsg = String(error);
				const errorLower = errorMsg.toLowerCase();
			console.error(`[P2P] Dial failed:`, {
				multiaddr: multiaddrString,
				error: error,
				errorMessage: errorMsg,
				errorType: error?.constructor?.name,
				// Try to extract more details if available
				cause: (error as any)?.cause,
				code: (error as any)?.code,
			});

			// Classify the error for better diagnostics
			if (errorLower.includes("timeout") || (error as any)?.name === "AbortError") {
				throw new P2PDialError(
					"CONNECTION_TIMEOUT",
					`Connection timed out - check if node is reachable at the specified address`,
					multiaddrString,
					error
				);
			}

			if (errorLower.includes("connection gater") || errorLower.includes("gater denied")) {
				throw new P2PDialError(
					"ADDRESS_DENIED",
					`Dial rejected by connection gater - address may be non-routable (0.0.0.0/localhost)`,
					multiaddrString,
					error
				);
			}

			if (errorLower.includes("certificate") || errorLower.includes("dtls") || errorLower.includes("fingerprint")) {
				throw new P2PDialError(
					"CERTIFICATE_MISMATCH",
					`WebRTC certificate validation failed - certhash may be outdated`,
					multiaddrString,
					error
				);
			}

			if (errorLower.includes("refused") || errorLower.includes("econnrefused")) {
				throw new P2PDialError(
					"CONNECTION_REFUSED",
					`Connection refused - check if node is running and WebRTC port is open`,
					multiaddrString,
					error
				);
			}

			if (errorLower.includes("unreachable") || errorLower.includes("enetunreach")) {
				throw new P2PDialError(
					"NETWORK_UNREACHABLE",
					`Network unreachable - check firewall and network connectivity`,
					multiaddrString,
					error
				);
			}

			// Check for ICE/STUN failures
			if (errorLower.includes("ice") || errorLower.includes("stun") || errorLower.includes("candidate")) {
				throw new P2PDialError(
					"DIAL_FAILED",
					`WebRTC ICE negotiation failed - may be firewall or NAT issue`,
					multiaddrString,
					error
				);
			}

			// Check for noise protocol failures
			if (errorLower.includes("noise") || errorLower.includes("handshake") || errorLower.includes("decrypt")) {
				throw new P2PDialError(
					"DIAL_FAILED",
					`Noise protocol handshake failed - possible version mismatch`,
					multiaddrString,
					error
				);
			}

			throw new P2PDialError(
				"DIAL_FAILED",
				`Failed to establish WebRTC connection: ${errorMsg.slice(0, 200)}`,
				multiaddrString,
				error
			);
		}
	}

	/**
	 * Dial a list of candidate multiaddrs in priority order
	 * Tries each address until a connection is established
	 */
	async dialAny(multiaddrs: string[]): Promise<void> {
		if (multiaddrs.length === 0) {
			throw new P2PDialError(
				"INVALID_TRANSPORT",
				"No WebRTC addresses to dial",
				"",
			);
		}

		const timeoutMs = getDialTimeoutMs();
		let lastError: unknown;
		for (let i = 0; i < multiaddrs.length; i++) {
			const addr = multiaddrs[i];
			try {
				console.log(
					`[P2P] Dial attempt ${i + 1}/${multiaddrs.length}: ${addr.slice(0, 80)}...`
				);
				const controller = new AbortController();
				const timeoutId = window.setTimeout(() => controller.abort(), timeoutMs);
				try {
					await this.dial(addr, { signal: controller.signal });
				} finally {
					clearTimeout(timeoutId);
				}
				return;
			} catch (error) {
				lastError = error;
				console.warn(`[P2P] Dial attempt ${i + 1} failed`, error);
			}
		}

		if (lastError instanceof P2PDialError) {
			throw lastError;
		}

		throw new P2PDialError(
			"DIAL_FAILED",
			`All ${multiaddrs.length} dial attempts failed`,
			multiaddrs[0],
			lastError
		);
	}

	/**
	 * Subscribe to a GossipSub topic
	 * Registers a handler that will be called for each message on the topic
	 */
	subscribe(topic: string, handler: (data: Uint8Array) => void): void {
		if (!this.node) {
			throw new Error("P2PClient not initialized. Call initialize() first.");
		}

		if (!this.gossipSub) {
			throw new Error("PubSub service not available");
		}

		this.messageHandlers.set(topic, handler);
		this.gossipSub.subscribe(topic);
	}

	/**
	 * Unsubscribe from a GossipSub topic
	 */
	unsubscribe(topic: string): void {
		if (!this.node) {
			throw new Error("P2PClient not initialized. Call initialize() first.");
		}

		if (!this.gossipSub) {
			throw new Error("PubSub service not available");
		}

		this.messageHandlers.delete(topic);
		this.gossipSub.unsubscribe(topic);
	}

	/**
	 * Publish data to a GossipSub topic
	 */
	async publish(topic: string, data: Uint8Array): Promise<void> {
		if (!this.node) {
			throw new Error("P2PClient not initialized. Call initialize() first.");
		}

		if (!this.gossipSub) {
			throw new Error("PubSub service not available");
		}

		await this.gossipSub.publish(topic, data);
	}

	/**
	 * Subscribe to video topic for chunk delivery
	 * Handler receives raw Uint8Array of SCALE-encoded VideoChunk
	 *
	 * @param handler - Callback for video chunk data
	 * @throws Error if node not initialized
	 */
	subscribeToVideoTopic(handler: (data: Uint8Array) => void): void {
		if (!this.node) {
			throw new Error("P2PClient not initialized. Call initialize() first.");
		}

		if (!this.gossipSub) {
			throw new Error("PubSub service not available");
		}

		// Subscribe to video topic
		this.gossipSub.subscribe(VIDEO_TOPIC);

		// Add message listener for video topic
		const videoHandler = (evt: any) => {
			const message = evt.detail;
			if (message.topic === VIDEO_TOPIC) {
				// Ensure data is Uint8Array
				const data =
					message.data instanceof Uint8Array
						? message.data
						: new Uint8Array(message.data);
				handler(data);
			}
		};

		this.gossipSub.addEventListener("message", videoHandler);

		// Store listener reference for cleanup
		this.messageHandlers.set(VIDEO_TOPIC, videoHandler as any);
	}

	/**
	 * Unsubscribe from video topic
	 */
	unsubscribeFromVideoTopic(): void {
		if (!this.node || !this.gossipSub) {
			return;
		}

		// Remove event listener
		const handler = this.messageHandlers.get(VIDEO_TOPIC);
		if (handler) {
			this.gossipSub.removeEventListener("message", handler as any);
			this.messageHandlers.delete(VIDEO_TOPIC);
		}

		// Unsubscribe from topic
		this.gossipSub.unsubscribe(VIDEO_TOPIC);
	}

	/**
	 * Stop the libp2p node and cleanup resources
	 */
	async stop(): Promise<void> {
		// Unsubscribe from video topic first
		this.unsubscribeFromVideoTopic();

		if (this.node) {
			await this.node.stop();
			this.node = null;
			this.gossipSub = null;
			this.messageHandlers.clear();
		}
	}

	/**
	 * Check if the libp2p node is running
	 */
	isStarted(): boolean {
		return this.node?.status === "started";
	}

	/**
	 * Get the local peer ID
	 */
	getPeerId(): string | null {
		if (!this.node) {
			return null;
		}
		return this.node.peerId?.toString() ?? null;
	}

	/**
	 * Get the number of connected peers
	 */
	getConnectedPeers(): number {
		if (!this.node) {
			return 0;
		}

		let count = 0;
		for (const peer of this.node.getPeers()) {
			// Check if we have an active connection to this peer
			const connections = this.node.getConnections(peer);
			if (connections && connections.length > 0) {
				count++;
			}
		}
		return count;
	}

	/**
	 * Get all peers (connected and known)
	 */
	getPeers(): string[] {
		if (!this.node) {
			return [];
		}

		return this.node.getPeers().map((peer) => peer.toString());
	}

	/**
	 * Get the libp2p node instance (for advanced use cases)
	 */
	getNode(): Libp2p | null {
		return this.node;
	}
}

// ============================================================================
// Types
// ============================================================================

export type { Libp2p } from "libp2p";
export type { GossipSub } from "@chainsafe/libp2p-gossipsub";
