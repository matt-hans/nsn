// ICN Viewer Client - libp2p-based P2P Client
// WebRTC-Direct transport for browser-to-mesh connectivity

import { gossipsub } from "@chainsafe/libp2p-gossipsub";
import type { GossipSub } from "@chainsafe/libp2p-gossipsub";
import { noise } from "@chainsafe/libp2p-noise";
import { yamux } from "@chainsafe/libp2p-yamux";
import { identify } from "@libp2p/identify";
import { webRTCDirect } from "@libp2p/webrtc";
import { multiaddr } from "@multiformats/multiaddr";
import { type Libp2p, createLibp2p } from "libp2p";

/**
 * P2PClient: Web-native P2P client for direct browser-to-mesh connectivity
 * Uses libp2p with WebRTC-Direct transport for outbound connections
 */
export class P2PClient {
	private node: Libp2p | null = null;
	private messageHandlers: Map<string, (data: Uint8Array) => void> = new Map();
	private gossipSub: GossipSub | null = null;

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
	 */
	async dial(multiaddrString: string): Promise<void> {
		if (!this.node) {
			throw new Error("P2PClient not initialized. Call initialize() first.");
		}

		try {
			const ma = multiaddr(multiaddrString);
			await this.node.dial(ma);
		} catch (error) {
			throw new Error(`Failed to dial peer: ${error}`);
		}
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
	 * Stop the libp2p node and cleanup resources
	 */
	async stop(): Promise<void> {
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
