// ICN Viewer Client - Settings Modal

import { invoke } from "@tauri-apps/api/core";
import { useAppStore } from "../../store/appStore";

export default function SettingsModal() {
	const {
		quality,
		volume,
		seedingEnabled,
		toggleSettings,
		setQuality,
		setSeedingEnabled,
	} = useAppStore();

	const handleSave = async () => {
		try {
			await invoke("save_settings", {
				settings: {
					volume,
					quality,
					seeding_enabled: seedingEnabled,
					last_slot: useAppStore.getState().currentSlot,
				},
			});
		} catch (error) {
			console.warn("Failed to save settings to backend:", error);
			// Continue anyway - settings are persisted in localStorage via Zustand
		}
		toggleSettings();
	};

	return (
		<div
			className="settings-backdrop"
			onClick={(e) => e.target === e.currentTarget && toggleSettings()}
			onKeyDown={(e) => {
				if (
					e.target === e.currentTarget &&
					(e.key === "Enter" || e.key === " " || e.key === "Escape")
				) {
					e.preventDefault();
					toggleSettings();
				}
			}}
			// biome-ignore lint/a11y/useSemanticElements: backdrop div requires role for accessibility
			role="button"
			tabIndex={0}
		>
			<div
				className="settings-modal"
				// biome-ignore lint/a11y/useSemanticElements: styled div modal, not native dialog element
				role="dialog"
				aria-labelledby="settings-title"
			>
				<h1 id="settings-title" className="settings-title">
					Settings
				</h1>

				<div className="setting-row">
					<div className="setting-info">
						<div className="setting-label">Seeding</div>
						<div className="setting-description">
							Re-share watched content to help the network (earn small ICN
							rewards)
						</div>
					</div>
					<button
						type="button"
						className={`toggle ${seedingEnabled ? "on" : ""}`}
						onClick={() => setSeedingEnabled(!seedingEnabled)}
						aria-label="Toggle seeding"
						aria-pressed={seedingEnabled}
					>
						<div className="toggle-indicator" />
					</button>
				</div>

				<div className="setting-row">
					<div className="setting-info">
						<div className="setting-label">Quality Preference</div>
						<div className="setting-description">
							Default video quality (can be overridden per session)
						</div>
					</div>
					<select
						className="dropdown-trigger"
						value={quality}
						onChange={(e) =>
							setQuality(e.target.value as "1080p" | "720p" | "480p" | "auto")
						}
						aria-label="Quality preference"
					>
						<option value="auto">Auto</option>
						<option value="1080p">1080p</option>
						<option value="720p">720p</option>
						<option value="480p">480p</option>
					</select>
				</div>

				<div className="settings-actions">
					<button
						type="button"
						className="button-secondary"
						onClick={toggleSettings}
					>
						Cancel
					</button>
					<button type="button" className="button-primary" onClick={handleSave}>
						Save Settings
					</button>
				</div>
			</div>
		</div>
	);
}
