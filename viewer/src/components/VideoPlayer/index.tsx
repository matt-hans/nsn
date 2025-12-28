// ICN Viewer Client - Video Player Component

import { useEffect, useRef, useState } from "react";
import {
	connectToRelay,
	disconnect,
	discoverRelays,
	onVideoChunk,
	startMockVideoStream,
} from "../../services/p2p";
import {
	destroyVideoPipeline,
	getVideoPipeline,
} from "../../services/videoPipeline";
import { useAppStore } from "../../store/appStore";
import ControlsOverlay from "./ControlsOverlay";
import LoadingPortal from "./LoadingPortal";

export default function VideoPlayer() {
	const canvasRef = useRef<HTMLCanvasElement>(null);
	const [slotVisible, setSlotVisible] = useState(false);
	const [slotChanging, setSlotChanging] = useState(false);
	const [connectionError, setConnectionError] = useState<string | null>(null);

	const {
		currentSlot,
		playbackState,
		isFullscreen,
		setConnectionStatus,
		setConnectedRelay,
		updateStats,
		setPlaybackState,
	} = useAppStore();

	// Show slot number on slot change
	// biome-ignore lint/correctness/useExhaustiveDependencies: intentionally triggers on currentSlot change
	useEffect(() => {
		setSlotChanging(true);
		setSlotVisible(true);

		const timer = setTimeout(() => {
			setSlotChanging(false);
			setTimeout(() => setSlotVisible(false), 2000);
		}, 400);

		return () => clearTimeout(timer);
	}, [currentSlot]);

	// Initialize video pipeline when canvas is ready
	// biome-ignore lint/correctness/useExhaustiveDependencies: only initialize once on mount
	useEffect(() => {
		if (!canvasRef.current) return;

		const initPipeline = async () => {
			try {
				// Create pipeline
				// biome-ignore lint/style/noNonNullAssertion: checked above
				const pipeline = getVideoPipeline(canvasRef.current!);
				if (!pipeline) {
					throw new Error("Failed to create video pipeline");
				}

				// Initialize decoder
				await pipeline.init("vp09.00.10.08");

				// Set up video chunk handler
				onVideoChunk((chunk) => {
					pipeline.handleIncomingChunk(chunk);

					// Update buffer stats
					updateStats({
						bufferSeconds: pipeline.getBufferedSeconds(),
					});
				});

				// Discover and connect to relays
				setConnectionStatus("connecting");
				const relays = await discoverRelays();

				if (relays.length === 0) {
					throw new Error("No relays available");
				}

				// Connect to first relay
				const connected = await connectToRelay(relays[0]);

				if (!connected) {
					throw new Error("Failed to connect to relay");
				}

				setConnectionStatus("connected");
				setConnectedRelay(relays[0].peer_id, relays[0].region);
				setPlaybackState("buffering");

				// Set up ABR quality change callback
				pipeline.onQualityChange((quality) => {
					useAppStore.getState().setQuality(quality);
				});

				// Start mock video stream (only in development/testing)
				// Always start in browser environment (Vite dev or E2E tests)
				startMockVideoStream(currentSlot);

				// Start pipeline
				pipeline.start();

				// Update stats periodically
				const statsInterval = setInterval(() => {
					updateStats({
						bufferSeconds: pipeline.getBufferedSeconds(),
						bitrate: 5.2, // Mock bitrate
						latency: 45, // Mock latency
						connectedPeers: 8, // Mock peers
					});

					// Transition to playing once buffered
					if (
						pipeline.getBufferedSeconds() >= 5 &&
						playbackState === "buffering"
					) {
						setPlaybackState("playing");
					}
				}, 1000);

				return () => {
					clearInterval(statsInterval);
				};
			} catch (error) {
				console.error("Pipeline initialization failed:", error);
				setConnectionError(
					error instanceof Error ? error.message : "Connection failed",
				);
				setConnectionStatus("error");
			}
		};

		initPipeline();

		return () => {
			disconnect();
			destroyVideoPipeline();
		};
	}, []); // Only run once on mount

	return (
		<div className={`video-player ${isFullscreen ? "fullscreen" : ""}`}>
			<canvas
				ref={canvasRef}
				className="video-canvas"
				width={1920}
				height={1080}
			/>

			<div
				className={`slot-display ${slotVisible ? "visible" : ""} ${slotChanging ? "changing" : ""}`}
			>
				SLOT {currentSlot.toString().padStart(5, "0")}
			</div>

			{playbackState === "buffering" && <LoadingPortal />}

			{connectionError && (
				<div className="error-overlay">
					<div className="error-message">
						<h2>Connection Error</h2>
						<p>{connectionError}</p>
					</div>
				</div>
			)}

			<ControlsOverlay />
		</div>
	);
}
