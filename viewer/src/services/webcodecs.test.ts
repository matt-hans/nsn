// ICN Viewer Client - WebCodecs Decoder Unit Tests

import { beforeEach, describe, expect, it, vi } from "vitest";
import { VideoDecoderService } from "./webcodecs";

describe("VideoDecoderService", () => {
	let canvas: HTMLCanvasElement;
	let decoder: VideoDecoderService;

	beforeEach(() => {
		canvas = document.createElement("canvas");
		decoder = new VideoDecoderService(canvas);
	});

	it("should initialize with VP9 codec", async () => {
		await expect(decoder.init("vp09.00.10.08")).resolves.not.toThrow();
	});

	it("should reject unsupported codec", async () => {
		await expect(decoder.init("unsupported_codec")).rejects.toThrow(
			"not supported",
		);
	});

	it("should log error when decoding before init", () => {
		const consoleSpy = vi.spyOn(console, "error");
		const mockChunk = new EncodedVideoChunk({
			type: "key",
			timestamp: 0,
			data: new Uint8Array([0, 1, 2]),
		});

		decoder.decode(mockChunk);
		expect(consoleSpy).toHaveBeenCalledWith("Decoder not initialized");
	});

	it("should decode after init", async () => {
		await decoder.init("vp09.00.10.08");

		const mockChunk = new EncodedVideoChunk({
			type: "key",
			timestamp: 0,
			data: new Uint8Array([0, 1, 2]),
		});

		expect(() => decoder.decode(mockChunk)).not.toThrow();
	});

	it("should clean up decoder on destroy", async () => {
		await decoder.init("vp09.00.10.08");
		decoder.destroy();
		// After destroy, decode should fail
		const consoleSpy = vi.spyOn(console, "error");
		const mockChunk = new EncodedVideoChunk({
			type: "key",
			timestamp: 0,
			data: new Uint8Array([0, 1, 2]),
		});
		decoder.decode(mockChunk);
		expect(consoleSpy).toHaveBeenCalledWith("Decoder not initialized");
	});
});
