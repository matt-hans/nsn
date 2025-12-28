// ICN Viewer Client - Video Pipeline Tests
// Comprehensive test coverage for video pipeline orchestration

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { VideoChunkMessage } from "./p2p";
import {
	VideoPipeline,
	destroyVideoPipeline,
	getVideoPipeline,
} from "./videoPipeline";

// Mock dependencies
vi.mock("./videoBuffer", () => {
	class MockVideoBuffer {
		addChunk = vi.fn();
		getNextChunk = vi.fn();
		isBufferReady = vi.fn();
		getBufferedSeconds = vi.fn(() => 3.5);
		clear = vi.fn();
	}

	class MockAdaptiveBitrateController {
		recordDownloadSpeed = vi.fn();
		getCurrentQuality = vi.fn(() => "720p");
	}

	return {
		VideoBuffer: MockVideoBuffer,
		AdaptiveBitrateController: MockAdaptiveBitrateController,
	};
});

vi.mock("./webcodecs", () => {
	class MockVideoDecoderService {
		init = vi.fn().mockResolvedValue(undefined);
		decode = vi.fn();
		destroy = vi.fn();
	}

	return {
		VideoDecoderService: MockVideoDecoderService,
	};
});

describe("VideoPipeline", () => {
	let canvas: HTMLCanvasElement;
	let pipeline: VideoPipeline;

	beforeEach(() => {
		// Create mock canvas
		canvas = document.createElement("canvas");
		pipeline = new VideoPipeline(canvas);
	});

	afterEach(() => {
		vi.clearAllMocks();
		destroyVideoPipeline();
	});

	describe("Initialization", () => {
		it("should initialize with canvas and create dependencies", () => {
			expect(pipeline).toBeDefined();
			expect((pipeline as any).buffer).toBeDefined();
			expect((pipeline as any).decoder).toBeDefined();
			expect((pipeline as any).abrController).toBeDefined();
		});

		it("should initialize decoder with default codec", async () => {
			const decoder = (pipeline as any).decoder;
			await pipeline.init();
			expect(decoder.init).toHaveBeenCalledWith("vp09.00.10.08");
		});

		it("should initialize decoder with custom codec", async () => {
			const decoder = (pipeline as any).decoder;
			await pipeline.init("av01.0.05M.08");
			expect(decoder.init).toHaveBeenCalledWith("av01.0.05M.08");
		});
	});

	describe("Lifecycle Management", () => {
		it("should start decode loop when start() is called", () => {
			const requestAnimationFrameSpy = vi.spyOn(
				globalThis,
				"requestAnimationFrame",
			);

			pipeline.start();

			expect(requestAnimationFrameSpy).toHaveBeenCalled();
			expect((pipeline as any).running).toBe(true);
		});

		it("should stop decode loop when stop() is called", () => {
			const cancelAnimationFrameSpy = vi.spyOn(
				globalThis,
				"cancelAnimationFrame",
			);

			// Start then stop
			pipeline.start();
			const loopId = (pipeline as any).decodeLoopId;
			pipeline.stop();

			expect((pipeline as any).running).toBe(false);
			expect((pipeline as any).decodeLoopId).toBe(null);
			if (loopId !== null) {
				expect(cancelAnimationFrameSpy).toHaveBeenCalledWith(loopId);
			}
		});

		it("should cleanup resources on destroy", () => {
			const decoder = (pipeline as any).decoder;
			const buffer = (pipeline as any).buffer;

			pipeline.start();
			pipeline.destroy();

			expect((pipeline as any).running).toBe(false);
			expect(decoder.destroy).toHaveBeenCalled();
			expect(buffer.clear).toHaveBeenCalled();
		});
	});

	describe("Chunk Processing", () => {
		it("should handle incoming video chunk correctly", () => {
			const buffer = (pipeline as any).buffer;
			const abrController = (pipeline as any).abrController;

			const message: VideoChunkMessage = {
				slot: 12345,
				chunk_index: 0,
				data: new Uint8Array([1, 2, 3, 4]),
				timestamp: 1000000,
				is_keyframe: true,
			};

			pipeline.handleIncomingChunk(message);

			expect(buffer.addChunk).toHaveBeenCalledWith({
				slot: 12345,
				chunk_index: 0,
				data: message.data,
				timestamp: 1000000,
			});

			expect(abrController.recordDownloadSpeed).toHaveBeenCalledWith(
				4, // data.length
				expect.any(Number), // duration in ms
			);
		});

		it("should measure download speed for ABR", () => {
			const abrController = (pipeline as any).abrController;

			const message: VideoChunkMessage = {
				slot: 12345,
				chunk_index: 5,
				data: new Uint8Array(1024 * 50), // 50KB
				timestamp: 2000000,
				is_keyframe: false,
			};

			const startTime = Date.now();
			pipeline.handleIncomingChunk(message);
			const endTime = Date.now();

			const callArgs = abrController.recordDownloadSpeed.mock.calls[0];
			expect(callArgs[0]).toBe(1024 * 50);
			expect(callArgs[1]).toBeGreaterThanOrEqual(0);
			expect(callArgs[1]).toBeLessThanOrEqual(endTime - startTime + 10);
		});
	});

	describe("Decode Loop", () => {
		it("should decode chunks when buffer is ready", async () => {
			const buffer = (pipeline as any).buffer;
			const decoder = (pipeline as any).decoder;

			// Mock buffer ready with chunk
			buffer.isBufferReady.mockReturnValue(true);
			buffer.getNextChunk.mockReturnValue({
				slot: 12345,
				chunk_index: 0,
				data: new Uint8Array([1, 2, 3]),
				timestamp: 1000000,
			});

			pipeline.start();

			// Wait for animation frame
			await new Promise((resolve) => requestAnimationFrame(resolve));

			expect(buffer.isBufferReady).toHaveBeenCalled();
			expect(buffer.getNextChunk).toHaveBeenCalled();
			expect(decoder.decode).toHaveBeenCalled();

			pipeline.stop();
		});

		it("should not decode when buffer is not ready", async () => {
			const buffer = (pipeline as any).buffer;
			const decoder = (pipeline as any).decoder;

			buffer.isBufferReady.mockReturnValue(false);

			pipeline.start();

			await new Promise((resolve) => requestAnimationFrame(resolve));

			expect(buffer.isBufferReady).toHaveBeenCalled();
			expect(buffer.getNextChunk).not.toHaveBeenCalled();
			expect(decoder.decode).not.toHaveBeenCalled();

			pipeline.stop();
		});

		it("should create key frame for chunk_index 0", async () => {
			const buffer = (pipeline as any).buffer;
			const decoder = (pipeline as any).decoder;

			buffer.isBufferReady.mockReturnValue(true);
			buffer.getNextChunk.mockReturnValue({
				slot: 12345,
				chunk_index: 0,
				data: new Uint8Array([1, 2, 3]),
				timestamp: 1000000,
			});

			pipeline.start();
			await new Promise((resolve) => requestAnimationFrame(resolve));

			const encodedChunk = decoder.decode.mock.calls[0][0];
			expect(encodedChunk.type).toBe("key");

			pipeline.stop();
		});

		it("should create delta frame for chunk_index > 0", async () => {
			const buffer = (pipeline as any).buffer;
			const decoder = (pipeline as any).decoder;

			buffer.isBufferReady.mockReturnValue(true);
			buffer.getNextChunk.mockReturnValue({
				slot: 12345,
				chunk_index: 5,
				data: new Uint8Array([1, 2, 3]),
				timestamp: 2000000,
			});

			pipeline.start();
			await new Promise((resolve) => requestAnimationFrame(resolve));

			const encodedChunk = decoder.decode.mock.calls[0][0];
			expect(encodedChunk.type).toBe("delta");

			pipeline.stop();
		});
	});

	describe("Status Queries", () => {
		it("should return current ABR quality", () => {
			const quality = pipeline.getCurrentQuality();
			expect(quality).toBe("720p");
		});

		it("should return buffered seconds", () => {
			const buffered = pipeline.getBufferedSeconds();
			expect(buffered).toBe(3.5);
		});
	});
});

describe("Singleton Pattern", () => {
	let canvas: HTMLCanvasElement;

	beforeEach(() => {
		canvas = document.createElement("canvas");
		destroyVideoPipeline();
	});

	afterEach(() => {
		destroyVideoPipeline();
	});

	it("should create singleton instance with canvas", () => {
		const instance1 = getVideoPipeline(canvas);
		expect(instance1).not.toBeNull();
		expect(instance1).toBeInstanceOf(VideoPipeline);
	});

	it("should return same instance on subsequent calls", () => {
		const instance1 = getVideoPipeline(canvas);
		const instance2 = getVideoPipeline();
		expect(instance2).toBe(instance1);
	});

	it("should return null when no canvas provided and not initialized", () => {
		const instance = getVideoPipeline();
		expect(instance).toBeNull();
	});

	it("should destroy singleton instance", () => {
		const instance = getVideoPipeline(canvas);
		expect(instance).not.toBeNull();

		destroyVideoPipeline();

		const newInstance = getVideoPipeline();
		expect(newInstance).toBeNull();
	});

	it("should allow recreation after destroy", () => {
		const instance1 = getVideoPipeline(canvas);
		destroyVideoPipeline();

		const canvas2 = document.createElement("canvas");
		const instance2 = getVideoPipeline(canvas2);

		expect(instance2).not.toBeNull();
		expect(instance2).not.toBe(instance1);
	});
});
