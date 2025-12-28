// ICN Viewer Client - WebCodecs Decoder Service
// Hardware-accelerated video decoding

export class VideoDecoderService {
	private decoder: VideoDecoder | null = null;
	private canvas: HTMLCanvasElement;
	private ctx: CanvasRenderingContext2D;
	private isConfigured = false;

	constructor(canvas: HTMLCanvasElement) {
		this.canvas = canvas;
		// biome-ignore lint/style/noNonNullAssertion: 2d context always available
		this.ctx = canvas.getContext("2d")!;
	}

	/**
	 * Initialize decoder with codec
	 */
	async init(codec: string): Promise<void> {
		const config: VideoDecoderConfig = {
			codec, // e.g., 'vp09.00.10.08' for VP9
			optimizeForLatency: true,
		};

		// Check if codec is supported
		const support = await VideoDecoder.isConfigSupported(config);
		if (!support.supported) {
			throw new Error(`Codec ${codec} not supported`);
		}

		this.decoder = new VideoDecoder({
			output: (frame: VideoFrame) => this.renderFrame(frame),
			error: (e: DOMException) => {
				console.error("Decode error:", e);
			},
		});

		this.decoder.configure(config);
		this.isConfigured = true;
	}

	/**
	 * Decode video chunk
	 */
	decode(chunk: EncodedVideoChunk): void {
		if (!this.decoder || !this.isConfigured) {
			console.error("Decoder not initialized");
			return;
		}

		this.decoder.decode(chunk);
	}

	/**
	 * Render frame to canvas
	 */
	private renderFrame(frame: VideoFrame): void {
		this.ctx.drawImage(frame, 0, 0, this.canvas.width, this.canvas.height);
		frame.close();
	}

	/**
	 * Cleanup decoder
	 */
	destroy(): void {
		if (this.decoder) {
			this.decoder.close();
			this.decoder = null;
		}
		this.isConfigured = false;
	}
}
