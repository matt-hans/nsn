// ICN Viewer Client - Video Pipeline Orchestration
// Coordinates buffer, decoder, and P2P for playback

import type { P2PClient } from "./p2pClient";
import type { VideoChunkMessage } from "./types";
import { AdaptiveBitrateController, VideoBuffer } from "./videoBuffer";
import { decodeVideoChunk } from "./videoCodec";
import { VideoDecoderService } from "./webcodecs";

export class VideoPipeline {
	private buffer: VideoBuffer;
	private decoder: VideoDecoderService;
	private abrController: AdaptiveBitrateController;
	private running = false;
	private decodeLoopId: number | null = null;
	private currentQuality: "1080p" | "720p" | "480p" = "1080p";
	private qualityChangeCallback:
		| ((quality: "1080p" | "720p" | "480p" | "auto") => void)
		| null = null;
	private chunkStats: { receivedAt: number; size: number }[] = [];

	constructor(canvas: HTMLCanvasElement, fps = 24) {
		this.buffer = new VideoBuffer(fps);
		this.decoder = new VideoDecoderService(canvas);
		this.abrController = new AdaptiveBitrateController();
	}

	/**
	 * Initialize pipeline with codec
	 */
	async init(codec = "vp09.00.10.08"): Promise<void> {
		await this.decoder.init(codec);
	}

	/**
	 * Start decoding and rendering
	 */
	start(): void {
		this.running = true;
		this.startDecodeLoop();
		this.setupABRMonitoring();
	}

	/**
	 * Set callback for quality changes
	 */
	onQualityChange(
		callback: (quality: "1080p" | "720p" | "480p" | "auto") => void,
	): void {
		this.qualityChangeCallback = callback;
	}

	/**
	 * Monitor ABR controller and sync quality changes
	 */
	private setupABRMonitoring(): void {
		const checkQuality = () => {
			if (!this.running) return;

			const newQuality = this.abrController.getCurrentQuality();
			if (newQuality !== this.currentQuality) {
				this.currentQuality = newQuality;
				console.log(`ABR quality changed to ${newQuality}`);

				// Notify callback (typically updates Zustand store)
				if (this.qualityChangeCallback) {
					this.qualityChangeCallback(newQuality);
				}
			}

			// Check every 2 seconds
			setTimeout(checkQuality, 2000);
		};

		checkQuality();
	}

	/**
	 * Stop pipeline
	 */
	stop(): void {
		this.running = false;
		if (this.decodeLoopId !== null) {
			cancelAnimationFrame(this.decodeLoopId);
			this.decodeLoopId = null;
		}
	}

	/**
	 * Handle incoming video chunk from P2P
	 */
	handleIncomingChunk(message: VideoChunkMessage): void {
		const startTime = Date.now();

		// Record chunk stats for bitrate calculation
		this.chunkStats.push({ receivedAt: Date.now(), size: message.data.length });
		// Prune stats older than 10 seconds
		const cutoff = Date.now() - 10000;
		this.chunkStats = this.chunkStats.filter((s) => s.receivedAt > cutoff);

		this.buffer.addChunk({
			slot: message.slot,
			chunk_index: message.chunk_index,
			data: message.data,
			timestamp: message.timestamp,
			is_keyframe: message.is_keyframe,
		});

		// Record download speed for ABR
		const durationMs = Date.now() - startTime;
		this.abrController.recordDownloadSpeed(message.data.length, durationMs);
	}

	/**
	 * Calculate current bitrate in Mbps from recent chunks
	 */
	getBitrateMbps(): number {
		if (this.chunkStats.length < 2) return 0;
		const now = Date.now();
		const windowMs = 5000; // 5 second window
		const recentStats = this.chunkStats.filter(
			(s) => now - s.receivedAt < windowMs,
		);
		if (recentStats.length < 2) return 0;

		const totalBytes = recentStats.reduce((sum, s) => sum + s.size, 0);
		const totalBits = totalBytes * 8;
		const mbps = totalBits / windowMs / 1000; // bits/ms = kbps, /1000 = Mbps
		return Math.round(mbps * 100) / 100;
	}

	/**
	 * Calculate latency from chunk timestamp to now
	 * Uses most recent chunk
	 */
	getLatencyMs(): number {
		// This will be populated when we track chunk timestamps
		// For now return 0 - will be wired in Plan 04
		return 0;
	}

	/**
	 * Main decode loop - runs at ~60fps
	 */
	private startDecodeLoop(): void {
		const loop = () => {
			if (!this.running) return;

			// Wait for buffer to be ready
			if (this.buffer.isBufferReady()) {
				const chunk = this.buffer.getNextChunk();
				if (chunk) {
					this.decodeChunk(chunk);
				}
			}

			this.decodeLoopId = requestAnimationFrame(loop);
		};

		loop();
	}

	/**
	 * Decode a single chunk
	 */
	private decodeChunk(chunk: {
		slot: number;
		chunk_index: number;
		data: Uint8Array;
		timestamp: number;
		is_keyframe: boolean;
	}): void {
		// Calculate timestamp from frame index for consistent timing
		// WebCodecs expects microseconds
		const frameDurationUs = 1_000_000 / 24; // 24 fps
		const timestamp = chunk.chunk_index * frameDurationUs;

		const encodedChunk = new EncodedVideoChunk({
			type: chunk.is_keyframe ? "key" : "delta",
			timestamp: timestamp,
			data: chunk.data,
		});

		this.decoder.decode(encodedChunk);
	}

	/**
	 * Get current ABR quality
	 */
	getCurrentQuality(): string {
		return this.abrController.getCurrentQuality();
	}

	/**
	 * Get buffered seconds
	 */
	getBufferedSeconds(): number {
		return this.buffer.getBufferedSeconds();
	}

	/**
	 * Cleanup resources
	 */
	destroy(): void {
		this.stop();
		this.decoder.destroy();
		this.buffer.clear();
	}
}

// Singleton instance
let pipeline: VideoPipeline | null = null;

export function getVideoPipeline(
	canvas?: HTMLCanvasElement,
): VideoPipeline | null {
	if (canvas && !pipeline) {
		pipeline = new VideoPipeline(canvas);
	}
	return pipeline;
}

export function destroyVideoPipeline(): void {
	if (pipeline) {
		pipeline.destroy();
		pipeline = null;
	}
}

/**
 * Connect P2P client to video pipeline for chunk delivery.
 * Decodes SCALE-encoded chunks and feeds to pipeline buffer.
 *
 * @param p2pClient - P2PClient instance
 * @param pipeline - VideoPipeline instance
 */
export function connectP2PToPipeline(
	p2pClient: P2PClient,
	pipeline: VideoPipeline,
): void {
	p2pClient.subscribeToVideoTopic((data: Uint8Array) => {
		try {
			const chunk = decodeVideoChunk(data);

			// Adapt DecodedVideoChunk to VideoChunkMessage format
			pipeline.handleIncomingChunk({
				slot: Number(chunk.slot),
				chunk_index: chunk.chunkIndex,
				data: chunk.payload,
				timestamp: Number(chunk.timestampMs) * 1000, // ms to microseconds for WebCodecs
				is_keyframe: chunk.isKeyframe,
			});
		} catch (err) {
			console.error("Failed to decode video chunk:", err);
			// Don't crash pipeline on single bad chunk - silently drop
		}
	});
}
