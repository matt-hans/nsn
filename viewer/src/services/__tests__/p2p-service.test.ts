// ICN Viewer Client - P2P Service Unit Tests
// DISABLED: Legacy P2PService removed during libp2p migration
// Tests for new P2PClient should be added separately

/*
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { P2PService, type VideoChunkMessage } from "../p2p";

describe("P2PService", () => {
	let service: P2PService;

	beforeEach(() => {
		service = new P2PService();
	});

	afterEach(() => {
		service.disconnect();
	});

	describe("construction", () => {
		it("should generate unique peer ID", () => {
			const peerId = service.getPeerId();
			expect(peerId).toBeDefined();
			expect(typeof peerId).toBe("string");
			expect(peerId.length).toBeGreaterThan(0);
		});

		it("should generate different IDs for different instances", () => {
			const service2 = new P2PService();
			expect(service.getPeerId()).not.toBe(service2.getPeerId());
			service2.disconnect();
		});

		it("should start with zero connected peers", () => {
			expect(service.getConnectedPeerCount()).toBe(0);
		});

		it("should start disconnected", () => {
			expect(service.isConnected()).toBe(false);
		});
	});

	describe("connect", () => {
		it("should return false when signaling server unavailable", async () => {
			// Mock WebSocket fails by default in test setup
			const result = await service.connect("ws://localhost:8080");
			expect(result).toBe(false);
		});

		it("should set connected state to false on failure", async () => {
			await service.connect("ws://localhost:8080");
			expect(service.isConnected()).toBe(false);
		});
	});

	describe("disconnect", () => {
		it("should reset connection state", async () => {
			// Attempt connection first
			await service.connect("ws://localhost:8080");

			// Disconnect
			service.disconnect();

			expect(service.isConnected()).toBe(false);
			expect(service.getConnectedPeerCount()).toBe(0);
		});

		it("should be safe to call multiple times", () => {
			expect(() => {
				service.disconnect();
				service.disconnect();
				service.disconnect();
			}).not.toThrow();
		});
	});

	describe("onVideoChunk", () => {
		it("should register video chunk handler", () => {
			const handler = vi.fn();
			service.onVideoChunk(handler);

			// Handler is registered but won't be called without actual peer data
			expect(handler).not.toHaveBeenCalled();
		});

		it("should allow replacing handler", () => {
			const handler1 = vi.fn();
			const handler2 = vi.fn();

			service.onVideoChunk(handler1);
			service.onVideoChunk(handler2);

			// Both handlers registered (second replaces first)
			expect(handler1).not.toHaveBeenCalled();
			expect(handler2).not.toHaveBeenCalled();
		});
	});

	describe("video chunk parsing", () => {
		it("should parse valid video chunk binary data", () => {
			// Create test binary data
			// Format: [slot:4][chunk_index:4][timestamp:8][is_keyframe:1][data:rest]
			const data = new Uint8Array(20);
			const view = new DataView(data.buffer);

			view.setUint32(0, 42); // slot
			view.setUint32(4, 100); // chunk_index
			view.setBigUint64(8, BigInt(1704067200000)); // timestamp
			data[16] = 1; // is_keyframe = true
			data[17] = 0xaa; // data byte 1
			data[18] = 0xbb; // data byte 2
			data[19] = 0xcc; // data byte 3

			// Access private method through any cast for testing
			const service2 = service as unknown as {
				parseVideoChunk: (data: Uint8Array) => VideoChunkMessage | null;
			};

			const chunk = service2.parseVideoChunk(data);

			expect(chunk).not.toBeNull();
			expect(chunk?.slot).toBe(42);
			expect(chunk?.chunk_index).toBe(100);
			expect(chunk?.timestamp).toBe(1704067200000);
			expect(chunk?.is_keyframe).toBe(true);
			expect(chunk?.data.length).toBe(3);
			expect(chunk?.data[0]).toBe(0xaa);
		});

		it("should return null for data smaller than header size", () => {
			const data = new Uint8Array(10); // Less than 17 bytes

			const service2 = service as unknown as {
				parseVideoChunk: (data: Uint8Array) => VideoChunkMessage | null;
			};

			const chunk = service2.parseVideoChunk(data);
			expect(chunk).toBeNull();
		});
	});
});
*/
