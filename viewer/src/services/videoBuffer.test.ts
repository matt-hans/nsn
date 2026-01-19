// ICN Viewer Client - VideoBuffer Unit Tests

import { beforeEach, describe, expect, it } from "vitest";
import {
	AdaptiveBitrateController,
	VideoBuffer,
	type VideoChunk,
} from "./videoBuffer";

describe("VideoBuffer", () => {
	let buffer: VideoBuffer;

	beforeEach(() => {
		buffer = new VideoBuffer();
	});

	it("should add chunk to buffer", () => {
		const chunk: VideoChunk = {
			slot: 1,
			chunk_index: 0,
			data: new Uint8Array([1, 2, 3]),
			timestamp: Date.now(),
			is_keyframe: true,
		};
		buffer.addChunk(chunk);
		expect(buffer.getNextChunk()).toEqual(chunk);
	});

	it("should sort chunks by index", () => {
		const chunk1: VideoChunk = {
			slot: 1,
			chunk_index: 2,
			data: new Uint8Array([1]),
			timestamp: Date.now(),
			is_keyframe: false,
		};
		const chunk2: VideoChunk = {
			slot: 1,
			chunk_index: 0,
			data: new Uint8Array([2]),
			timestamp: Date.now(),
			is_keyframe: true,
		};
		const chunk3: VideoChunk = {
			slot: 1,
			chunk_index: 1,
			data: new Uint8Array([3]),
			timestamp: Date.now(),
			is_keyframe: false,
		};

		buffer.addChunk(chunk1);
		buffer.addChunk(chunk2);
		buffer.addChunk(chunk3);

		expect(buffer.getNextChunk()?.chunk_index).toBe(0);
		expect(buffer.getNextChunk()?.chunk_index).toBe(1);
		expect(buffer.getNextChunk()?.chunk_index).toBe(2);
	});

	it("should return null when buffer is empty", () => {
		expect(buffer.getNextChunk()).toBeNull();
	});

	it("should check if buffer is ready", () => {
		expect(buffer.isBufferReady()).toBe(false);

		// Add minimum chunks (5 seconds * 24 fps = 120 chunks)
		for (let i = 0; i < 120; i++) {
			buffer.addChunk({
				slot: 1,
				chunk_index: i,
				data: new Uint8Array([i]),
				timestamp: Date.now(),
				is_keyframe: i === 0, // First frame is keyframe
			});
		}

		expect(buffer.isBufferReady()).toBe(true);
	});

	it("should calculate buffered seconds", () => {
		// Add 48 chunks = 2 seconds @ 24fps
		for (let i = 0; i < 48; i++) {
			buffer.addChunk({
				slot: 1,
				chunk_index: i,
				data: new Uint8Array([i]),
				timestamp: Date.now(),
				is_keyframe: i === 0, // First frame is keyframe
			});
		}

		expect(buffer.getBufferedSeconds()).toBe(2);
	});

	it("should clear buffer", () => {
		buffer.addChunk({
			slot: 1,
			chunk_index: 0,
			data: new Uint8Array([1]),
			timestamp: Date.now(),
			is_keyframe: true,
		});
		buffer.clear();
		expect(buffer.getNextChunk()).toBeNull();
		expect(buffer.getBufferedSeconds()).toBe(0);
	});
});

describe("AdaptiveBitrateController", () => {
	let controller: AdaptiveBitrateController;

	beforeEach(() => {
		controller = new AdaptiveBitrateController();
	});

	it("should start with 1080p quality", () => {
		expect(controller.getCurrentQuality()).toBe("1080p");
	});

	it("should switch to 480p on low bandwidth", () => {
		// Simulate low bandwidth (1.5 Mbps)
		for (let i = 0; i < 5; i++) {
			controller.recordDownloadSpeed(187_500, 1000); // 1.5 Mbps
		}
		expect(controller.getCurrentQuality()).toBe("480p");
	});

	it("should switch to 720p on medium bandwidth", () => {
		// Simulate medium bandwidth (3.5 Mbps)
		for (let i = 0; i < 5; i++) {
			controller.recordDownloadSpeed(437_500, 1000); // 3.5 Mbps
		}
		expect(controller.getCurrentQuality()).toBe("720p");
	});

	it("should stay at 1080p on high bandwidth", () => {
		// Simulate high bandwidth (8 Mbps)
		for (let i = 0; i < 5; i++) {
			controller.recordDownloadSpeed(1_000_000, 1000); // 8 Mbps
		}
		expect(controller.getCurrentQuality()).toBe("1080p");
	});

	it("should adapt quality based on rolling average", () => {
		// Start high
		controller.recordDownloadSpeed(1_000_000, 1000); // 8 Mbps
		expect(controller.getCurrentQuality()).toBe("1080p");

		// Degrade to low
		for (let i = 0; i < 10; i++) {
			controller.recordDownloadSpeed(187_500, 1000); // 1.5 Mbps
		}
		expect(controller.getCurrentQuality()).toBe("480p");

		// Recover to medium
		for (let i = 0; i < 10; i++) {
			controller.recordDownloadSpeed(437_500, 1000); // 3.5 Mbps
		}
		expect(controller.getCurrentQuality()).toBe("720p");
	});
});
