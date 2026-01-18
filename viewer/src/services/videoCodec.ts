// ICN Viewer Client - SCALE VideoChunk Codec
// Decodes SCALE-encoded VideoChunk messages from Rust nodes

import { TypeRegistry } from "@polkadot/types";

/**
 * VideoChunk struct matching Rust definition
 * Source: /home/matt/nsn/node-core/crates/types/src/lib.rs:233-242
 *
 * Rust struct:
 * ```rust
 * pub struct VideoChunk {
 *     pub header: VideoChunkHeader,
 *     pub payload: Vec<u8>,
 *     pub signer: Vec<u8>,
 *     pub signature: Vec<u8>,
 * }
 * ```
 *
 * VideoChunkHeader (lines 212-229):
 * ```rust
 * pub struct VideoChunkHeader {
 *     pub version: u16,
 *     pub slot: u64,
 *     pub content_id: String,
 *     pub chunk_index: u32,
 *     pub total_chunks: u32,
 *     pub timestamp_ms: u64,
 *     pub is_keyframe: bool,
 *     pub payload_hash: [u8; 32],
 * }
 * ```
 */

/**
 * Create TypeRegistry ONCE at module level
 * CRITICAL: Do NOT create per decode - causes memory leak and performance issues
 * Field order MUST match Rust struct order exactly (SCALE is position-based)
 */
const registry = new TypeRegistry();

registry.register({
	VideoChunkHeader: {
		version: "u16",
		slot: "u64",
		content_id: "Text",
		chunk_index: "u32",
		total_chunks: "u32",
		timestamp_ms: "u64",
		is_keyframe: "bool",
		payload_hash: "[u8; 32]",
	},
	VideoChunk: {
		header: "VideoChunkHeader",
		payload: "Bytes",
		signer: "Bytes",
		signature: "Bytes",
	},
});

/**
 * Decoded VideoChunk - TypeScript representation
 * Fields use camelCase for JavaScript convention, but match Rust semantics
 */
export interface DecodedVideoChunk {
	/** Slot number for the video */
	slot: bigint;
	/** Zero-based chunk index */
	chunkIndex: number;
	/** Total number of chunks in the stream */
	totalChunks: number;
	/** Timestamp when chunk was published (ms since Unix epoch) */
	timestampMs: bigint;
	/** Whether this is a keyframe boundary */
	isKeyframe: boolean;
	/** Raw video payload bytes */
	payload: Uint8Array;
	/** Content identifier (IPFS CID or other) */
	contentId: string;
	/** Schema version */
	version: number;
	/** Blake3 hash of payload */
	payloadHash: Uint8Array;
	/** Signer public key bytes */
	signer: Uint8Array;
	/** Signature over header + payload hash */
	signature: Uint8Array;
}

/**
 * Decode a SCALE-encoded VideoChunk binary message
 *
 * @param data - Raw SCALE-encoded bytes from GossipSub message
 * @returns Decoded VideoChunk object
 * @throws Error if decode fails (malformed data, wrong type, etc.)
 */
export function decodeVideoChunk(data: Uint8Array): DecodedVideoChunk {
	try {
		// Create type from registry using binary data
		const chunk = registry.createType("VideoChunk", data);

		// Convert to JSON for field extraction
		// Note: toJSON() returns plain JS objects with proper type conversion
		const json = chunk.toJSON() as any;

		if (!json.header) {
			throw new Error("Missing header in VideoChunk");
		}

		const header = json.header;

		// Convert Bytes fields to Uint8Array
		const payload = bytesToUint8Array(json.payload);
		const signer = bytesToUint8Array(json.signer);
		const signature = bytesToUint8Array(json.signature);
		const payloadHash = hexToUint8Array(header.payload_hash);

		return {
			slot: BigInt(header.slot),
			chunkIndex: header.chunk_index,
			totalChunks: header.total_chunks,
			timestampMs: BigInt(header.timestamp_ms),
			isKeyframe: header.is_keyframe,
			payload,
			contentId: header.content_id,
			version: header.version,
			payloadHash,
			signer,
			signature,
		};
	} catch (error) {
		throw new Error(
			`Failed to decode VideoChunk: ${error instanceof Error ? error.message : String(error)}`,
		);
	}
}

/**
 * Convert @polkadot/types Bytes to Uint8Array
 * Handles both string hex and array formats
 */
function bytesToUint8Array(value: unknown): Uint8Array {
	if (typeof value === "string") {
		return hexToUint8Array(value);
	}
	if (Array.isArray(value)) {
		return new Uint8Array(value);
	}
	if (value instanceof Uint8Array) {
		return value;
	}
	throw new Error(`Cannot convert to Uint8Array: ${typeof value}`);
}

/**
 * Convert hex string (with or without 0x prefix) to Uint8Array
 */
function hexToUint8Array(hex: string): Uint8Array {
	// Remove 0x prefix if present
	const cleanHex = hex.startsWith("0x") ? hex.slice(2) : hex;

	// Handle empty array case
	if (cleanHex.length === 0) {
		return new Uint8Array(0);
	}

	// Check even length
	if (cleanHex.length % 2 !== 0) {
		throw new Error(`Invalid hex string: odd length (${cleanHex.length})`);
	}

	const array = new Uint8Array(cleanHex.length / 2);
	for (let i = 0; i < array.length; i++) {
		const byte = Number.parseInt(cleanHex.slice(i * 2, i * 2 + 2), 16);
		if (Number.isNaN(byte)) {
			throw new Error(
				`Invalid hex string at position ${i}: ${cleanHex.slice(i * 2, i * 2 + 2)}`,
			);
		}
		array[i] = byte;
	}
	return array;
}

/**
 * Validate VideoChunk structure (optional helper)
 * Checks that all required fields are present and valid
 *
 * @param chunk - Decoded VideoChunk to validate
 * @returns true if valid, throws if invalid
 */
export function validateVideoChunk(chunk: DecodedVideoChunk): boolean {
	if (chunk.chunkIndex < 0 || chunk.chunkIndex >= chunk.totalChunks) {
		throw new Error(
			`Invalid chunk index: ${chunk.chunkIndex} / ${chunk.totalChunks}`,
		);
	}
	if (chunk.totalChunks <= 0) {
		throw new Error(`Invalid total chunks: ${chunk.totalChunks}`);
	}
	if (chunk.payload.length === 0) {
		throw new Error("Empty payload");
	}
	if (chunk.version !== 1) {
		throw new Error(`Unsupported version: ${chunk.version}`);
	}
	return true;
}
