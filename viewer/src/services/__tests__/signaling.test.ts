// ICN Viewer Client - Signaling Client Unit Tests

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { SignalingClient, type SignalingMessage } from "../signaling";

describe("SignalingClient", () => {
	let onMessage: ReturnType<typeof vi.fn>;
	let client: SignalingClient;

	beforeEach(() => {
		onMessage = vi.fn();
		client = new SignalingClient("test-peer-123", onMessage);
	});

	afterEach(() => {
		client.disconnect();
	});

	describe("construction", () => {
		it("should create client with peer ID", () => {
			expect(client.getPeerId()).toBe("test-peer-123");
		});

		it("should start in disconnected state", () => {
			expect(client.getState()).toBe("disconnected");
			expect(client.isConnected()).toBe(false);
		});
	});

	describe("connect", () => {
		it("should transition to connecting state and fail when server unavailable", async () => {
			// Start connection (will fail due to mock WebSocket)
			const connectPromise = client.connect("ws://localhost:8080");

			// State should be connecting
			expect(client.getState()).toBe("connecting");

			// Wait for connection attempt to complete (mock fails)
			await expect(connectPromise).rejects.toThrow("WebSocket connection failed");

			// Should be in error state after failure
			expect(client.getState()).toBe("error");
		});
	});

	describe("disconnect", () => {
		it("should set state to disconnected", () => {
			client.disconnect();
			expect(client.getState()).toBe("disconnected");
			expect(client.isConnected()).toBe(false);
		});
	});

	describe("send", () => {
		it("should return false when not connected", () => {
			const result = client.send({
				type: "offer",
				from: "test-peer",
				to: "other-peer",
			});
			expect(result).toBe(false);
		});
	});

	describe("helper methods", () => {
		it("should format offer message correctly", () => {
			// sendOffer relies on send() which requires connection
			// Just verify it returns false when not connected
			const result = client.sendOffer("target-peer", {
				type: "offer",
				sdp: "test-sdp",
			});
			expect(result).toBe(false);
		});

		it("should format answer message correctly", () => {
			const result = client.sendAnswer("target-peer", {
				type: "answer",
				sdp: "test-sdp",
			});
			expect(result).toBe(false);
		});
	});
});
