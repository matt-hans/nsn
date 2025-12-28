// ICN Viewer Client - Test Setup
// Mock Tauri API and WebCodecs for testing

import "@testing-library/jest-dom";
import { vi } from "vitest";

// Mock Tauri API
vi.mock("@tauri-apps/api/core", () => ({
	invoke: vi.fn((cmd: string) => {
		if (cmd === "get_relays") {
			return Promise.resolve([
				{
					peer_id: "12D3KooWMockRelay1",
					multiaddr: "/ip4/127.0.0.1/tcp/30333",
					region: "us-east-1",
					latency_ms: 45,
					is_fallback: false,
				},
			]);
		}
		if (cmd === "save_settings") {
			return Promise.resolve();
		}
		return Promise.resolve();
	}),
}));

// Mock HTMLCanvasElement.getContext
HTMLCanvasElement.prototype.getContext = vi.fn(() => ({
	drawImage: vi.fn(),
	clearRect: vi.fn(),
	fillRect: vi.fn(),
	canvas: { width: 1920, height: 1080 },
})) as unknown as typeof HTMLCanvasElement.prototype.getContext;

// Mock WebCodecs API
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
