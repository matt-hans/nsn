// ICN Viewer Client - P2P Service (Compatibility Layer)
// Re-exports from p2pClient.ts for backward compatibility

export { P2PClient, VIDEO_TOPIC } from "./p2pClient";
export type { VideoChunkMessage } from "./types";

import { buildCandidateList, discoverWithRace } from "./discovery";
// Legacy exports - these functions delegate to P2PClient
import { P2PClient } from "./p2pClient";

let sharedClient: P2PClient | null = null;

export function getP2PService(): P2PClient {
	if (!sharedClient) {
		sharedClient = new P2PClient();
	}
	return sharedClient;
}

export async function connectToMesh(): Promise<boolean> {
	try {
		const client = getP2PService();
		await client.initialize();
		const candidates = buildCandidateList();
		const multiaddr = await discoverWithRace(candidates);
		await client.dial(multiaddr);
		return true;
	} catch (error) {
		console.error("Failed to connect to mesh:", error);
		return false;
	}
}

export function disconnect(): void {
	if (sharedClient) {
		sharedClient.stop();
		sharedClient = null;
	}
}

export function getConnectionStatus(): boolean {
	return sharedClient?.isStarted() ?? false;
}

// REMOVED: startMockVideoStream - no more mocks
// REMOVED: All simple-peer and signaling related code
