// ICN Viewer Client - Video Buffer and Adaptive Bitrate Logic

export interface VideoChunk {
	slot: number;
	chunk_index: number;
	data: Uint8Array;
	timestamp: number;
}

export class VideoBuffer {
	private chunks: VideoChunk[] = [];
	private readonly minBufferSeconds = 5;
	private readonly maxBufferSize = 300; // ~12.5 seconds at 24fps
	private readonly fps: number;

	constructor(fps = 24) {
		this.fps = fps;
	}

	/**
	 * Add chunk to buffer
	 */
	addChunk(chunk: VideoChunk): void {
		// Prevent unbounded growth
		if (this.chunks.length >= this.maxBufferSize) {
			// Remove oldest chunks when buffer full
			this.chunks.shift();
		}
		this.chunks.push(chunk);
		this.chunks.sort((a, b) => a.chunk_index - b.chunk_index);
	}

	/**
	 * Get next chunk to decode
	 */
	getNextChunk(): VideoChunk | null {
		return this.chunks.shift() || null;
	}

	/**
	 * Check if buffer has minimum content for playback
	 */
	isBufferReady(): boolean {
		return this.chunks.length >= this.minBufferSeconds * this.fps;
	}

	/**
	 * Get buffered duration in seconds
	 */
	getBufferedSeconds(): number {
		// Guard against division by zero
		if (this.fps <= 0) return 0;
		return this.chunks.length / this.fps;
	}

	/**
	 * Clear buffer
	 */
	clear(): void {
		this.chunks = [];
	}
}

export class AdaptiveBitrateController {
	private downloadSpeeds: number[] = [];
	private currentQuality: "1080p" | "720p" | "480p" = "1080p";

	/**
	 * Record download speed for chunk
	 */
	recordDownloadSpeed(bytes: number, durationMs: number): void {
		const mbps = (bytes * 8) / (durationMs / 1000) / 1_000_000;
		this.downloadSpeeds.push(mbps);

		// Keep last 10 samples
		if (this.downloadSpeeds.length > 10) {
			this.downloadSpeeds.shift();
		}

		this.adjustQuality();
	}

	/**
	 * Adjust quality based on average download speed
	 */
	private adjustQuality(): void {
		const avgSpeed =
			this.downloadSpeeds.reduce((a, b) => a + b, 0) /
			this.downloadSpeeds.length;

		if (avgSpeed < 2 && this.currentQuality !== "480p") {
			this.currentQuality = "480p";
			console.log("Switched to 480p (low bandwidth)");
		} else if (
			avgSpeed >= 2 &&
			avgSpeed < 5 &&
			this.currentQuality !== "720p"
		) {
			this.currentQuality = "720p";
			console.log("Switched to 720p (medium bandwidth)");
		} else if (avgSpeed >= 5 && this.currentQuality !== "1080p") {
			this.currentQuality = "1080p";
			console.log("Switched to 1080p (high bandwidth)");
		}
	}

	/**
	 * Get current quality recommendation
	 */
	getCurrentQuality(): "1080p" | "720p" | "480p" {
		return this.currentQuality;
	}
}
