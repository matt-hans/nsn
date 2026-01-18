// ICN Viewer Client - Shared Types
// Common types used across P2P and video services

/**
 * Video chunk message format
 * Used for delivering video chunks from P2P network to pipeline
 */
export interface VideoChunkMessage {
	slot: number;
	chunk_index: number;
	data: Uint8Array;
	timestamp: number;
	is_keyframe: boolean;
}
