// ICN Viewer Client - Test Setup
// Mock WebRTC, WebSocket, and WebCodecs for testing

import "@testing-library/jest-dom";
import { vi } from "vitest";

// ============================================================================
// WebSocket Mock
// ============================================================================

class MockWebSocket {
	static CONNECTING = 0;
	static OPEN = 1;
	static CLOSING = 2;
	static CLOSED = 3;

	readyState = MockWebSocket.CONNECTING;
	onopen: (() => void) | null = null;
	onclose: ((event: { code: number; reason: string }) => void) | null = null;
	onmessage: ((event: { data: string }) => void) | null = null;
	onerror: ((error: Event) => void) | null = null;

	constructor(_url: string) {
		// Simulate async connection - by default fail (no server)
		setTimeout(() => {
			this.readyState = MockWebSocket.CLOSED;
			this.onerror?.(new Event("error"));
		}, 0);
	}

	send(_data: string): void {
		if (this.readyState !== MockWebSocket.OPEN) {
			throw new Error("WebSocket is not open");
		}
		// Mock implementation - messages are dropped in tests
	}

	close(code = 1000, reason = ""): void {
		this.readyState = MockWebSocket.CLOSED;
		this.onclose?.({ code, reason });
	}
}

// @ts-expect-error - Mock WebSocket for testing
globalThis.WebSocket = MockWebSocket;

// ============================================================================
// RTCPeerConnection Mock (for simple-peer)
// ============================================================================

class MockRTCPeerConnection {
	localDescription: RTCSessionDescription | null = null;
	remoteDescription: RTCSessionDescription | null = null;
	iceConnectionState: RTCIceConnectionState = "new";
	connectionState: RTCPeerConnectionState = "new";
	signalingState: RTCSignalingState = "stable";

	onicecandidate:
		| ((event: { candidate: RTCIceCandidate | null }) => void)
		| null = null;
	ondatachannel: ((event: { channel: RTCDataChannel }) => void) | null = null;
	onconnectionstatechange: (() => void) | null = null;
	oniceconnectionstatechange: (() => void) | null = null;

	createOffer = vi.fn().mockResolvedValue({ type: "offer", sdp: "mock-sdp" });
	createAnswer = vi.fn().mockResolvedValue({ type: "answer", sdp: "mock-sdp" });
	setLocalDescription = vi.fn().mockResolvedValue(undefined);
	setRemoteDescription = vi.fn().mockResolvedValue(undefined);
	addIceCandidate = vi.fn().mockResolvedValue(undefined);
	createDataChannel = vi.fn().mockReturnValue({
		readyState: "open",
		send: vi.fn(),
		close: vi.fn(),
		onopen: null,
		onclose: null,
		onmessage: null,
		onerror: null,
	});
	close = vi.fn();
}

// @ts-expect-error - Mock RTCPeerConnection for testing
globalThis.RTCPeerConnection = MockRTCPeerConnection;

// Mock RTCSessionDescription
globalThis.RTCSessionDescription = class {
	type: RTCSdpType;
	sdp: string;
	constructor(init: { type: RTCSdpType; sdp: string }) {
		this.type = init.type;
		this.sdp = init.sdp;
	}
	toJSON() {
		return { type: this.type, sdp: this.sdp };
	}
} as unknown as typeof RTCSessionDescription;

// Mock RTCIceCandidate
globalThis.RTCIceCandidate = class {
	candidate: string;
	sdpMid: string | null;
	sdpMLineIndex: number | null;

	constructor(init: RTCIceCandidateInit) {
		this.candidate = init.candidate || "";
		this.sdpMid = init.sdpMid || null;
		this.sdpMLineIndex = init.sdpMLineIndex ?? null;
	}

	toJSON() {
		return {
			candidate: this.candidate,
			sdpMid: this.sdpMid,
			sdpMLineIndex: this.sdpMLineIndex,
		};
	}
} as unknown as typeof RTCIceCandidate;

// ============================================================================
// Canvas Mock
// ============================================================================

HTMLCanvasElement.prototype.getContext = vi.fn(() => ({
	drawImage: vi.fn(),
	clearRect: vi.fn(),
	fillRect: vi.fn(),
	canvas: { width: 1920, height: 1080 },
})) as unknown as typeof HTMLCanvasElement.prototype.getContext;

// ============================================================================
// WebCodecs Mock
// ============================================================================

globalThis.VideoDecoder = class VideoDecoderMock {
	static async isConfigSupported(config: VideoDecoderConfig) {
		return { supported: config.codec.startsWith("vp09") };
	}

	constructor(
		private callbacks: {
			output: (frame: VideoFrame) => void;
			error: (e: DOMException) => void;
		},
	) {}

	configure(_config: VideoDecoderConfig) {
		// Mock configure
	}

	decode(_chunk: EncodedVideoChunk) {
		// Mock decode - immediately call output with mock frame
		const mockFrame = {
			close: vi.fn(),
		} as unknown as VideoFrame;
		this.callbacks.output(mockFrame);
	}

	close() {
		// Mock close
	}
} as unknown as typeof VideoDecoder;

globalThis.EncodedVideoChunk = class EncodedVideoChunkMock {
	type: "key" | "delta";
	timestamp: number;
	data: AllowSharedBufferSource;

	constructor(init: EncodedVideoChunkInit) {
		this.type = init.type;
		this.timestamp = init.timestamp;
		this.data = init.data;
	}
} as unknown as typeof EncodedVideoChunk;

globalThis.VideoFrame = class VideoFrameMock {
	close() {
		// Mock close
	}
} as unknown as typeof VideoFrame;

// ============================================================================
// Crypto Mock (for crypto.randomUUID)
// ============================================================================

if (!globalThis.crypto?.randomUUID) {
	Object.defineProperty(globalThis, "crypto", {
		value: {
			...globalThis.crypto,
			randomUUID: () =>
				"xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, (c) => {
					const r = (Math.random() * 16) | 0;
					const v = c === "x" ? r : (r & 0x3) | 0x8;
					return v.toString(16);
				}),
		},
	});
}
