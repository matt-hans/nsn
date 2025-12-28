// ICN Viewer Client - Video Controls Overlay

import { useAppStore } from "../../store/appStore";

export default function ControlsOverlay() {
	const {
		playbackState,
		volume,
		isMuted,
		quality,
		currentTime,
		duration,
		showControls,
		setPlaybackState,
		setVolume,
		toggleMute,
		setQuality,
		toggleFullscreen,
	} = useAppStore();

	const isPlaying = playbackState === "playing";
	const progressPercent = duration > 0 ? (currentTime / duration) * 100 : 0;

	const handlePlayPause = () => {
		setPlaybackState(isPlaying ? "paused" : "playing");
	};

	const handleSeek = (e: React.MouseEvent<HTMLDivElement>) => {
		const rect = e.currentTarget.getBoundingClientRect();
		const percent = (e.clientX - rect.left) / rect.width;
		const newTime = percent * duration;
		useAppStore.getState().updatePlaybackTime(newTime, duration);
	};

	const formatTime = (seconds: number) => {
		const mins = Math.floor(seconds / 60);
		const secs = Math.floor(seconds % 60);
		return `${mins}:${secs.toString().padStart(2, "0")}`;
	};

	if (!showControls) return null;

	return (
		<div className="controls-overlay">
			{/* Seek bar */}
			{/* biome-ignore lint/a11y/useKeyWithClickEvents: seekbar interaction via keyboard volume controls */}
			<div className="seekbar-container" onClick={handleSeek}>
				<div className="seekbar">
					<div className="seekbar-buffered" style={{ width: "80%" }} />
					<div
						className="seekbar-progress"
						style={{ width: `${progressPercent}%` }}
					/>
					<div
						className="seekbar-thumb"
						style={{ left: `${progressPercent}%` }}
					/>
				</div>
			</div>

			{/* Controls row */}
			<div className="controls-row">
				<div className="controls-left">
					<button
						type="button"
						className="icon-button"
						onClick={handlePlayPause}
						aria-label={isPlaying ? "Pause" : "Play"}
					>
						{isPlaying ? (
							<svg
								width="22"
								height="22"
								viewBox="0 0 24 24"
								fill="currentColor"
							>
								<title>Pause</title>
								<path d="M6 4h4v16H6V4zm8 0h4v16h-4V4z" />
							</svg>
						) : (
							<svg
								width="22"
								height="22"
								viewBox="0 0 24 24"
								fill="currentColor"
							>
								<title>Play</title>
								<path d="M8 5v14l11-7z" />
							</svg>
						)}
					</button>

					<div className="volume-control">
						<button
							type="button"
							className="icon-button"
							onClick={toggleMute}
							aria-label={isMuted ? "Unmute" : "Mute"}
						>
							<svg
								width="22"
								height="22"
								viewBox="0 0 24 24"
								fill="none"
								stroke="currentColor"
								strokeWidth="2"
							>
								<title>{isMuted ? "Unmute" : "Mute"}</title>
								{isMuted ? (
									<>
										<path d="M11 5L6 9H2v6h4l5 4V5zM22 9l-6 6m0-6l6 6" />
									</>
								) : (
									<>
										<path d="M11 5L6 9H2v6h4l5 4V5zM15.54 8.46a5 5 0 010 7.07M19.07 4.93a10 10 0 010 14.14" />
									</>
								)}
							</svg>
						</button>

						<input
							type="range"
							className="volume-slider"
							min="0"
							max="100"
							value={isMuted ? 0 : volume}
							onChange={(e) => setVolume(Number(e.target.value))}
							aria-label="Volume"
						/>
					</div>

					<span className="time-display">
						{formatTime(currentTime)} / {formatTime(duration)}
					</span>
				</div>

				<div className="controls-right">
					<select
						className="quality-select"
						value={quality}
						onChange={(e) =>
							setQuality(e.target.value as "1080p" | "720p" | "480p" | "auto")
						}
						aria-label="Quality"
					>
						<option value="auto">Auto</option>
						<option value="1080p">1080p</option>
						<option value="720p">720p</option>
						<option value="480p">480p</option>
					</select>

					<button
						type="button"
						className="icon-button"
						onClick={toggleFullscreen}
						aria-label="Fullscreen"
					>
						<svg
							width="22"
							height="22"
							viewBox="0 0 24 24"
							fill="none"
							stroke="currentColor"
							strokeWidth="2"
						>
							<title>Fullscreen</title>
							<path d="M8 3H5a2 2 0 00-2 2v3m18 0V5a2 2 0 00-2-2h-3m0 18h3a2 2 0 002-2v-3M3 16v3a2 2 0 002 2h3" />
						</svg>
					</button>
				</div>
			</div>
		</div>
	);
}
