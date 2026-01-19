// ICN Viewer Client - Video Player Component

import { useEffect, useRef, useState } from "react";
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
		setPlaybackState,
		updateStats,
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

				// Initialize decoder with VP9 Level 4.1 (supports 3Mbps, 4K)
				await pipeline.init("vp09.00.41.08");

				// P2P connection and video chunk handling is now managed by useP2PConnection hook
				// The hook calls connectP2PToPipeline() which subscribes to video topic
				// and handles incoming chunks via the pipeline

				// Set up ABR quality change callback
				pipeline.onQualityChange((quality) => {
					useAppStore.getState().setQuality(quality);
				});

				// Start pipeline
				pipeline.start();

				// Update stats periodically
				const statsInterval = setInterval(() => {
					updateStats({
						bufferSeconds: pipeline.getBufferedSeconds(),
						bitrate: pipeline.getBitrateMbps(),
						latency: pipeline.getLatencyMs(),
					});

					// Transition to playing once buffered
					if (
						pipeline.getBufferedSeconds() >= 2 &&
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
					error instanceof Error
						? error.message
						: "Pipeline initialization failed",
				);
			}
		};

		initPipeline();

		return () => {
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
						<h2>Pipeline Error</h2>
						<p>{connectionError}</p>
					</div>
				</div>
			)}

			<ControlsOverlay />
		</div>
	);
}
