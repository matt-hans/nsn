// ICN Viewer Client - WebRTC Signaling Client
// Handles WebSocket communication for WebRTC peer discovery and signaling

export type SignalingMessageType =
	| "offer"
	| "answer"
	| "ice-candidate"
	| "peer-list"
	| "join"
	| "leave";

export interface SignalingMessage {
	type: SignalingMessageType;
	from?: string;
	to?: string;
	payload?: unknown;
}

export type SignalingMessageHandler = (msg: SignalingMessage) => void;

export type SignalingState =
	| "disconnected"
	| "connecting"
	| "connected"
	| "error";

/**
 * WebSocket-based signaling client for WebRTC peer connection establishment.
 * Handles SDP offer/answer exchange and ICE candidate relay between peers.
 */
export class SignalingClient {
	private ws: WebSocket | null = null;
	private peerId: string;
	private onMessage: SignalingMessageHandler;
	private state: SignalingState = "disconnected";

	constructor(peerId: string, onMessage: SignalingMessageHandler) {
		this.peerId = peerId;
		this.onMessage = onMessage;
	}

	/**
	 * Get current connection state
	 */
	getState(): SignalingState {
		return this.state;
	}

	/**
	 * Get the peer ID for this client
	 */
	getPeerId(): string {
		return this.peerId;
	}

	/**
	 * Connect to the signaling server
	 * @param serverUrl - WebSocket URL of the signaling server
	 */
	async connect(serverUrl: string): Promise<void> {
		if (this.ws?.readyState === WebSocket.OPEN) {
			return;
		}

		this.state = "connecting";

		return new Promise((resolve, reject) => {
			try {
				this.ws = new WebSocket(serverUrl);

				this.ws.onopen = () => {
					this.state = "connected";
					// Announce ourselves to the signaling server
					this.send({ type: "join", from: this.peerId });
					resolve();
				};

				this.ws.onerror = () => {
					this.state = "error";
					reject(new Error("WebSocket connection failed"));
				};

				this.ws.onmessage = (event) => {
					try {
						const msg = JSON.parse(event.data as string) as SignalingMessage;
						this.onMessage(msg);
					} catch (error) {
						console.error("Failed to parse signaling message:", error);
					}
				};

				this.ws.onclose = () => {
					this.state = "disconnected";
					this.handleDisconnect();
				};
			} catch (error) {
				this.state = "error";
				reject(error);
			}
		});
	}

	/**
	 * Handle disconnection with optional reconnection
	 */
	private handleDisconnect(): void {
		this.ws = null;
		// Optionally attempt reconnection for transient failures
		// Currently disabled for MVP - caller should handle reconnection
	}

	/**
	 * Send a signaling message to the server
	 */
	send(message: SignalingMessage): boolean {
		if (this.ws?.readyState === WebSocket.OPEN) {
			this.ws.send(JSON.stringify(message));
			return true;
		}
		return false;
	}

	/**
	 * Send an SDP offer to a specific peer
	 */
	sendOffer(toPeerId: string, sdp: RTCSessionDescriptionInit): boolean {
		return this.send({
			type: "offer",
			from: this.peerId,
			to: toPeerId,
			payload: sdp,
		});
	}

	/**
	 * Send an SDP answer to a specific peer
	 */
	sendAnswer(toPeerId: string, sdp: RTCSessionDescriptionInit): boolean {
		return this.send({
			type: "answer",
			from: this.peerId,
			to: toPeerId,
			payload: sdp,
		});
	}

	/**
	 * Send an ICE candidate to a specific peer
	 */
	sendIceCandidate(toPeerId: string, candidate: RTCIceCandidate): boolean {
		return this.send({
			type: "ice-candidate",
			from: this.peerId,
			to: toPeerId,
			payload: candidate.toJSON(),
		});
	}

	/**
	 * Disconnect from the signaling server
	 */
	disconnect(): void {
		if (this.ws) {
			this.send({ type: "leave", from: this.peerId });
			this.ws.close();
			this.ws = null;
		}
		this.state = "disconnected";
	}

	/**
	 * Check if connected to signaling server
	 */
	isConnected(): boolean {
		return this.ws?.readyState === WebSocket.OPEN;
	}
}
