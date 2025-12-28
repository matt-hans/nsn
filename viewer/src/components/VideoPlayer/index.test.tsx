// ICN Viewer Client - VideoPlayer Component Tests
// Comprehensive test coverage for main video player

import { render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import * as p2pService from "../../services/p2p";
import * as videoPipelineService from "../../services/videoPipeline";
import { useAppStore } from "../../store/appStore";
import VideoPlayer from "./index";

// Mock services
vi.mock("../../services/p2p");
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
		getCurrentQuality: vi.fn(() => "1080p"),
		onQualityChange: vi.fn(),
		destroy: vi.fn(),
	};

	const mockRelay = {
		peer_id: "12D3KooWMockRelay",
		multiaddr: "/ip4/127.0.0.1/tcp/30333",
		region: "us-east-1",
		latency_ms: 45,
		is_fallback: false,
	};

	beforeEach(() => {
		vi.clearAllMocks();

		// Setup default mocks
		vi.mocked(videoPipelineService.getVideoPipeline).mockReturnValue(
			mockPipeline as any,
		);
		vi.mocked(p2pService.discoverRelays).mockResolvedValue([mockRelay]);
		vi.mocked(p2pService.connectToRelay).mockResolvedValue(true);
		vi.mocked(p2pService.onVideoChunk).mockImplementation(() => {});
		vi.mocked(p2pService.startMockVideoStream).mockImplementation(() => {});
		vi.mocked(p2pService.disconnect).mockImplementation(() => {});
		vi.mocked(videoPipelineService.destroyVideoPipeline).mockImplementation(
			() => {},
		);

		// Reset store to defaults
		useAppStore.setState({
			currentSlot: 12345,
			playbackState: "paused",
			isFullscreen: false,
			connectionStatus: "disconnected",
			connectedRelay: null,
			relayRegion: null,
			setConnectionStatus: vi.fn(),
			setConnectedRelay: vi.fn(),
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

		it("should set up video chunk handler", async () => {
			render(<VideoPlayer />);

			await waitFor(() => {
				expect(p2pService.onVideoChunk).toHaveBeenCalled();
			});
		});

		it("should start pipeline after successful connection", async () => {
			render(<VideoPlayer />);

			await waitFor(() => {
				expect(mockPipeline.start).toHaveBeenCalled();
			});
		});
	});

	describe("Relay Connection", () => {
		it("should discover relays on mount", async () => {
			render(<VideoPlayer />);

			await waitFor(() => {
				expect(p2pService.discoverRelays).toHaveBeenCalled();
			});
		});

		it("should set connecting status before connection", async () => {
			const setConnectionStatus = vi.fn();
			useAppStore.setState({ setConnectionStatus });

			render(<VideoPlayer />);

			await waitFor(() => {
				expect(setConnectionStatus).toHaveBeenCalledWith("connecting");
			});
		});

		it("should connect to first available relay", async () => {
			render(<VideoPlayer />);

			await waitFor(() => {
				expect(p2pService.connectToRelay).toHaveBeenCalledWith(mockRelay);
			});
		});

		it("should update connection status on successful connection", async () => {
			const setConnectionStatus = vi.fn();
			const setConnectedRelay = vi.fn();
			useAppStore.setState({ setConnectionStatus, setConnectedRelay });

			render(<VideoPlayer />);

			await waitFor(() => {
				expect(setConnectionStatus).toHaveBeenCalledWith("connected");
				expect(setConnectedRelay).toHaveBeenCalledWith(
					mockRelay.peer_id,
					mockRelay.region,
				);
			});
		});

		it("should start mock video stream for development", async () => {
			useAppStore.setState({ currentSlot: 12345 });
			render(<VideoPlayer />);

			await waitFor(() => {
				expect(p2pService.startMockVideoStream).toHaveBeenCalledWith(12345);
			});
		});
	});

	describe("Error Handling", () => {
		it("should show error when no relays available", async () => {
			vi.mocked(p2pService.discoverRelays).mockResolvedValue([]);

			render(<VideoPlayer />);

			await waitFor(() => {
				expect(screen.getByText("Connection Error")).toBeInTheDocument();
				expect(screen.getByText("No relays available")).toBeInTheDocument();
			});
		});

		it("should show error when connection fails", async () => {
			vi.mocked(p2pService.connectToRelay).mockResolvedValue(false);

			render(<VideoPlayer />);

			await waitFor(() => {
				expect(screen.getByText("Connection Error")).toBeInTheDocument();
				expect(
					screen.getByText("Failed to connect to relay"),
				).toBeInTheDocument();
			});
		});

		it("should show error when pipeline creation fails", async () => {
			vi.mocked(videoPipelineService.getVideoPipeline).mockReturnValue(null);

			render(<VideoPlayer />);

			await waitFor(() => {
				expect(screen.getByText("Connection Error")).toBeInTheDocument();
				expect(
					screen.getByText("Failed to create video pipeline"),
				).toBeInTheDocument();
			});
		});

		it("should set error status when initialization fails", async () => {
			const setConnectionStatus = vi.fn();
			useAppStore.setState({ setConnectionStatus });
			vi.mocked(videoPipelineService.getVideoPipeline).mockReturnValue(null);

			render(<VideoPlayer />);

			await waitFor(() => {
				expect(setConnectionStatus).toHaveBeenCalledWith("error");
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

		it("should transition to playing when buffer is ready", async () => {
			const setPlaybackState = vi.fn();
			useAppStore.setState({
				playbackState: "buffering",
				setPlaybackState,
			});
			mockPipeline.getBufferedSeconds.mockReturnValue(5.5);

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
			mockPipeline.getBufferedSeconds.mockReturnValue(2.0);

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
		it("should disconnect P2P on unmount", () => {
			const { unmount } = render(<VideoPlayer />);

			unmount();

			expect(p2pService.disconnect).toHaveBeenCalled();
		});

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

	describe("Video Chunk Handling", () => {
		it("should process incoming video chunks", async () => {
			let chunkHandler: ((chunk: any) => void) | undefined;
			vi.mocked(p2pService.onVideoChunk).mockImplementation((handler) => {
				chunkHandler = handler;
			});

			render(<VideoPlayer />);

			await waitFor(
				() => {
					expect(p2pService.onVideoChunk).toHaveBeenCalled();
				},
				{ timeout: 3000 },
			);

			const mockChunk = {
				slot: 12345,
				chunk_index: 0,
				data: new Uint8Array([1, 2, 3]),
				timestamp: 1000000,
			};

			chunkHandler?.(mockChunk);

			expect(mockPipeline.handleIncomingChunk).toHaveBeenCalledWith(mockChunk);
		});

		it("should update buffer stats when chunk is received", async () => {
			let chunkHandler: ((chunk: any) => void) | undefined;
			vi.mocked(p2pService.onVideoChunk).mockImplementation((handler) => {
				chunkHandler = handler;
			});

			const updateStats = vi.fn();
			useAppStore.setState({ updateStats });

			render(<VideoPlayer />);

			await waitFor(
				() => {
					expect(p2pService.onVideoChunk).toHaveBeenCalled();
				},
				{ timeout: 3000 },
			);

			const mockChunk = {
				slot: 12345,
				chunk_index: 0,
				data: new Uint8Array([1, 2, 3]),
				timestamp: 1000000,
			};

			// Get calls before chunk handling
			const callsBefore = updateStats.mock.calls.length;

			chunkHandler?.(mockChunk);

			// Check that a new call was made
			expect(updateStats.mock.calls.length).toBeGreaterThan(callsBefore);

			// Check the most recent call has bufferSeconds
			const mostRecentCall =
				updateStats.mock.calls[updateStats.mock.calls.length - 1][0];
			expect(mostRecentCall).toHaveProperty("bufferSeconds");
			// The buffer returns the mocked value from mockPipeline
			expect(typeof mostRecentCall.bufferSeconds).toBe("number");
		});
	});
});
