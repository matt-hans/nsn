// ICN Viewer Client - Video Pipeline Orchestration
// Coordinates buffer, decoder, and P2P for playback

import type { VideoChunkMessage } from "./p2p";
import { AdaptiveBitrateController, VideoBuffer } from "./videoBuffer";
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

		this.buffer.addChunk({
			slot: message.slot,
			chunk_index: message.chunk_index,
			data: message.data,
			timestamp: message.timestamp,
		});

		// Record download speed for ABR
		const durationMs = Date.now() - startTime;
		this.abrController.recordDownloadSpeed(message.data.length, durationMs);
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
	}): void {
		const encodedChunk = new EncodedVideoChunk({
			type: chunk.chunk_index === 0 ? "key" : "delta",
			timestamp: chunk.timestamp,
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
