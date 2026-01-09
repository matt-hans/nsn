// ICN Viewer Client - P2P Service Unit Tests

import { describe, expect, it } from "vitest";
import { connectToRelay, discoverRelays } from "./p2p";

describe("P2P Service", () => {
	describe("discoverRelays", () => {
		it("should return fallback relays when signaling server unavailable", async () => {
			// Without a signaling server, discoverRelays returns fallback relays
			const relays = await discoverRelays();
			expect(relays.length).toBeGreaterThan(0);
			// Fallback relays have is_fallback: true
			expect(relays[0].is_fallback).toBe(true);
			expect(relays[0].region).toBeDefined();
		});

		it("should return array of relays", async () => {
			const relays = await discoverRelays();
			expect(Array.isArray(relays)).toBe(true);
			// Should have at least one fallback relay
			expect(relays.length).toBeGreaterThanOrEqual(1);
		});
	});

	describe("connectToRelay", () => {
		it("should return true for mock connection", async () => {
			const relay = {
				peer_id: "12D3KooWRelay1",
				multiaddr: "/ip4/127.0.0.1/tcp/30333",
				region: "us-east-1",
				is_fallback: false,
			};
			const result = await connectToRelay(relay);
			expect(result).toBe(true);
		});
	});
});
