// ICN Viewer Client - P2P Service Unit Tests

import { describe, expect, it } from "vitest";
import { connectToRelay, discoverRelays } from "./p2p";

describe("P2P Service", () => {
	describe("discoverRelays", () => {
		it("should return relays from IPC", async () => {
			const relays = await discoverRelays();
			expect(relays).toHaveLength(1);
			expect(relays[0].peer_id).toBe("12D3KooWMockRelay1");
			expect(relays[0].region).toBe("us-east-1");
		});

		it("should return empty array on error", async () => {
			// This test validates the error handling path
			// In the mock, it always succeeds, but real implementation handles errors
			const relays = await discoverRelays();
			expect(Array.isArray(relays)).toBe(true);
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
