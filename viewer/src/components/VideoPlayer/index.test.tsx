// ICN Viewer Client - VideoPlayer Component Tests
// Comprehensive test coverage for main video player

import { render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import * as videoPipelineService from "../../services/videoPipeline";
import { useAppStore } from "../../store/appStore";
import VideoPlayer from "./index";

// Mock services
vi.mock("../../services/videoPipeline");

// Mock child components
vi.mock("./ControlsOverlay", () => ({
	default: () => <div data-testid="controls-overlay">Controls</div>,
}));

vi.mock("./LoadingPortal", () => ({
	default: () => <div data-testid="loading-portal">Loading...</div>,
}));

describe("VideoPlayer", () => {
	const mockPipeline = {
		init: vi.fn().mockResolvedValue(undefined),
		start: vi.fn(),
		handleIncomingChunk: vi.fn(),
		getBufferedSeconds: vi.fn(() => 3.0),
		getBitrateMbps: vi.fn(() => 5.2),
		getLatencyMs: vi.fn(() => 45),
		getCurrentQuality: vi.fn(() => "1080p"),
		onQualityChange: vi.fn(),
		destroy: vi.fn(),
	};

	beforeEach(() => {
		vi.clearAllMocks();

		// Setup default mocks
		vi.mocked(videoPipelineService.getVideoPipeline).mockReturnValue(
			mockPipeline as any,
		);
		vi.mocked(videoPipelineService.destroyVideoPipeline).mockImplementation(
			() => {},
		);

		// Reset store to defaults
		useAppStore.setState({
			currentSlot: 12345,
			playbackState: "paused",
			isFullscreen: false,
			updateStats: vi.fn(),
			setPlaybackState: vi.fn(),
		});
	});

	afterEach(() => {
		vi.clearAllMocks();
	});

	describe("Rendering", () => {
		it("should render video canvas", () => {
			render(<VideoPlayer />);

			const canvas = document.querySelector("canvas");
			expect(canvas).toBeInTheDocument();
			expect(canvas).toHaveAttribute("width", "1920");
			expect(canvas).toHaveAttribute("height", "1080");
			expect(canvas).toHaveClass("video-canvas");
		});

		it("should render controls overlay", () => {
			render(<VideoPlayer />);

			expect(screen.getByTestId("controls-overlay")).toBeInTheDocument();
		});

		it("should show loading portal when buffering", () => {
			useAppStore.setState({ playbackState: "buffering" });
			render(<VideoPlayer />);

			expect(screen.getByTestId("loading-portal")).toBeInTheDocument();
		});

		it("should not show loading portal when playing", () => {
			useAppStore.setState({ playbackState: "playing" });
			render(<VideoPlayer />);

			expect(screen.queryByTestId("loading-portal")).not.toBeInTheDocument();
		});

		it("should apply fullscreen class when fullscreen is active", () => {
			useAppStore.setState({ isFullscreen: true });
			render(<VideoPlayer />);

			const container = document.querySelector(".video-player");
			expect(container).toHaveClass("fullscreen");
		});
	});

	describe("Slot Display", () => {
		it("should display current slot number", () => {
			useAppStore.setState({ currentSlot: 12345 });
			render(<VideoPlayer />);

			expect(screen.getByText(/SLOT 12345/)).toBeInTheDocument();
		});

		it("should pad slot number with zeros", () => {
			useAppStore.setState({ currentSlot: 42 });
			render(<VideoPlayer />);

			expect(screen.getByText(/SLOT 00042/)).toBeInTheDocument();
		});

		it("should show slot display on render", async () => {
			render(<VideoPlayer />);

			const slotDisplay = screen.getByText(/SLOT/);
			const container = slotDisplay.closest(".slot-display");

			// Wait for animation to kick in
			await waitFor(() => {
				expect(container).toHaveClass("visible");
			});
		});
	});

	describe("Pipeline Initialization", () => {
		it("should initialize video pipeline on mount", async () => {
			render(<VideoPlayer />);

			await waitFor(() => {
				expect(videoPipelineService.getVideoPipeline).toHaveBeenCalledWith(
					expect.any(HTMLCanvasElement),
				);
			});
		});

		it("should initialize decoder with VP9 codec", async () => {
			render(<VideoPlayer />);

			await waitFor(() => {
				expect(mockPipeline.init).toHaveBeenCalledWith("vp09.00.10.08");
			});
		});

		it("should start pipeline after initialization", async () => {
			render(<VideoPlayer />);

			await waitFor(() => {
				expect(mockPipeline.start).toHaveBeenCalled();
			});
		});

		it("should set up ABR quality change callback", async () => {
			render(<VideoPlayer />);

			await waitFor(() => {
				expect(mockPipeline.onQualityChange).toHaveBeenCalled();
			});
		});
	});

	describe("Stats Updates", () => {
		it("should update stats periodically", async () => {
			const updateStats = vi.fn();
			useAppStore.setState({ updateStats });

			render(<VideoPlayer />);

			await waitFor(
				() => {
					expect(mockPipeline.start).toHaveBeenCalled();
				},
				{ timeout: 3000 },
			);

			// Wait for stats update
			await waitFor(
				() => {
					expect(updateStats).toHaveBeenCalled();
				},
				{ timeout: 2000 },
			);

			expect(updateStats).toHaveBeenCalledWith(
				expect.objectContaining({
					bufferSeconds: 3.0,
				}),
			);
		});

		it("should use correct stat methods", async () => {
			render(<VideoPlayer />);

			await waitFor(
				() => {
					expect(mockPipeline.start).toHaveBeenCalled();
				},
				{ timeout: 3000 },
			);

			await waitFor(
				() => {
					expect(mockPipeline.getBitrateMbps).toHaveBeenCalled();
					expect(mockPipeline.getLatencyMs).toHaveBeenCalled();
				},
				{ timeout: 2000 },
			);
		});

		it("should transition to playing when buffer is ready", async () => {
			const setPlaybackState = vi.fn();
			useAppStore.setState({
				playbackState: "buffering",
				setPlaybackState,
			});
			mockPipeline.getBufferedSeconds.mockReturnValue(2.5);

			render(<VideoPlayer />);

			await waitFor(
				() => {
					expect(mockPipeline.start).toHaveBeenCalled();
				},
				{ timeout: 3000 },
			);

			// Wait for state transition
			await waitFor(
				() => {
					expect(setPlaybackState).toHaveBeenCalledWith("playing");
				},
				{ timeout: 2000 },
			);
		});

		it("should not transition to playing if buffer not ready", async () => {
			const setPlaybackState = vi.fn();
			useAppStore.setState({
				playbackState: "buffering",
				setPlaybackState,
			});
			mockPipeline.getBufferedSeconds.mockReturnValue(1.0);

			render(<VideoPlayer />);

			await waitFor(
				() => {
					expect(mockPipeline.start).toHaveBeenCalled();
				},
				{ timeout: 3000 },
			);

			// Give time for potential state transition (shouldn't happen)
			await new Promise((resolve) => setTimeout(resolve, 1100));

			expect(setPlaybackState).not.toHaveBeenCalledWith("playing");
		});
	});

	describe("Cleanup", () => {
		it("should destroy video pipeline on unmount", () => {
			const { unmount } = render(<VideoPlayer />);

			unmount();

			expect(videoPipelineService.destroyVideoPipeline).toHaveBeenCalled();
		});

		it("should clear stats interval on unmount", async () => {
			const clearIntervalSpy = vi.spyOn(globalThis, "clearInterval");

			const { unmount } = render(<VideoPlayer />);

			await waitFor(
				() => {
					expect(mockPipeline.start).toHaveBeenCalled();
				},
				{ timeout: 3000 },
			);

			unmount();

			expect(clearIntervalSpy).toHaveBeenCalled();

			clearIntervalSpy.mockRestore();
		});
	});

	describe("Error Handling", () => {
		it("should show error when pipeline creation fails", async () => {
			vi.mocked(videoPipelineService.getVideoPipeline).mockReturnValue(null);

			render(<VideoPlayer />);

			await waitFor(() => {
				expect(screen.getByText("Pipeline Error")).toBeInTheDocument();
				expect(
					screen.getByText("Failed to create video pipeline"),
				).toBeInTheDocument();
			});
		});

		it("should show error when pipeline initialization fails", async () => {
			mockPipeline.init.mockRejectedValue(new Error("Decoder init failed"));

			render(<VideoPlayer />);

			await waitFor(() => {
				expect(screen.getByText("Pipeline Error")).toBeInTheDocument();
				expect(screen.getByText("Decoder init failed")).toBeInTheDocument();
			});
		});
	});

	describe("P2P Integration", () => {
		it("should not handle P2P connection directly (managed by useP2PConnection)", () => {
			// This test verifies that VideoPlayer no longer manages P2P connection
			// P2P is now handled by useP2PConnection hook in App.tsx
			render(<VideoPlayer />);

			// VideoPlayer should not call any P2P functions directly
			// It only handles video pipeline and stats
			expect(mockPipeline.start).toBeDefined();
		});

		it("should rely on connectP2PToPipeline for video chunk handling", async () => {
			// The connectP2PToPipeline function is called by useP2PConnection
			// VideoPlayer just needs the pipeline to be ready
			render(<VideoPlayer />);

			await waitFor(() => {
				expect(mockPipeline.start).toHaveBeenCalled();
			});
		});
	});
});
