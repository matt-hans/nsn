// ICN Viewer Client - Keyboard Shortcuts Tests
// Comprehensive test coverage for keyboard shortcut handling

import { renderHook } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { useAppStore } from "../store/appStore";
import { useKeyboardShortcuts } from "./useKeyboardShortcuts";

describe("useKeyboardShortcuts", () => {
	// Mock state methods
	const mockSetPlaybackState = vi.fn();
	const mockToggleMute = vi.fn();
	const mockToggleFullscreen = vi.fn();
	const mockToggleSidebar = vi.fn();
	const mockToggleSettings = vi.fn();
	const mockUpdatePlaybackTime = vi.fn();
	const mockSetVolume = vi.fn();

	beforeEach(() => {
		// Reset mocks
		vi.clearAllMocks();

		// Set initial state
		useAppStore.setState({
			playbackState: "paused",
			volume: 50,
			isMuted: false,
			isFullscreen: false,
			showSidebar: false,
			showSettings: false,
			currentTime: 10,
			duration: 100,
			setPlaybackState: mockSetPlaybackState,
			toggleMute: mockToggleMute,
			toggleFullscreen: mockToggleFullscreen,
			toggleSidebar: mockToggleSidebar,
			toggleSettings: mockToggleSettings,
			updatePlaybackTime: mockUpdatePlaybackTime,
			setVolume: mockSetVolume,
		});
	});

	afterEach(() => {
		vi.clearAllMocks();
	});

	it("should toggle play/pause when space is pressed", () => {
		renderHook(() => useKeyboardShortcuts());

		const event = new KeyboardEvent("keydown", { key: " " });
		window.dispatchEvent(event);

		expect(mockSetPlaybackState).toHaveBeenCalledWith("playing");
	});

	it("should toggle play/pause from playing to paused", () => {
		useAppStore.setState({ playbackState: "playing" });
		renderHook(() => useKeyboardShortcuts());

		const event = new KeyboardEvent("keydown", { key: " " });
		window.dispatchEvent(event);

		expect(mockSetPlaybackState).toHaveBeenCalledWith("paused");
	});

	it("should toggle mute when 'm' is pressed", () => {
		renderHook(() => useKeyboardShortcuts());

		const event = new KeyboardEvent("keydown", { key: "m" });
		window.dispatchEvent(event);

		expect(mockToggleMute).toHaveBeenCalledTimes(1);
	});

	it("should toggle fullscreen when 'f' is pressed", () => {
		renderHook(() => useKeyboardShortcuts());

		const event = new KeyboardEvent("keydown", { key: "f" });
		window.dispatchEvent(event);

		expect(mockToggleFullscreen).toHaveBeenCalledTimes(1);
	});

	it("should toggle sidebar when 'i' is pressed", () => {
		renderHook(() => useKeyboardShortcuts());

		const event = new KeyboardEvent("keydown", { key: "i" });
		window.dispatchEvent(event);

		expect(mockToggleSidebar).toHaveBeenCalledTimes(1);
	});

	it("should close modals when escape is pressed", () => {
		useAppStore.setState({
			showSettings: true,
			showSidebar: true,
			isFullscreen: true,
		});
		renderHook(() => useKeyboardShortcuts());

		const event = new KeyboardEvent("keydown", { key: "Escape" });
		window.dispatchEvent(event);

		expect(mockToggleSettings).toHaveBeenCalledTimes(1);
		expect(mockToggleSidebar).toHaveBeenCalledTimes(1);
		expect(mockToggleFullscreen).toHaveBeenCalledTimes(1);
	});

	it("should not close modals when escape is pressed if they are not open", () => {
		useAppStore.setState({
			showSettings: false,
			showSidebar: false,
			isFullscreen: false,
		});
		renderHook(() => useKeyboardShortcuts());

		const event = new KeyboardEvent("keydown", { key: "Escape" });
		window.dispatchEvent(event);

		expect(mockToggleSettings).not.toHaveBeenCalled();
		expect(mockToggleSidebar).not.toHaveBeenCalled();
		expect(mockToggleFullscreen).not.toHaveBeenCalled();
	});

	it("should seek backward 5s when left arrow is pressed", () => {
		useAppStore.setState({ currentTime: 10, duration: 100 });
		renderHook(() => useKeyboardShortcuts());

		const event = new KeyboardEvent("keydown", { key: "ArrowLeft" });
		window.dispatchEvent(event);

		expect(mockUpdatePlaybackTime).toHaveBeenCalledWith(5, 100);
	});

	it("should not seek below 0 when left arrow is pressed near start", () => {
		useAppStore.setState({ currentTime: 2, duration: 100 });
		renderHook(() => useKeyboardShortcuts());

		const event = new KeyboardEvent("keydown", { key: "ArrowLeft" });
		window.dispatchEvent(event);

		expect(mockUpdatePlaybackTime).toHaveBeenCalledWith(0, 100);
	});

	it("should seek forward 5s when right arrow is pressed", () => {
		useAppStore.setState({ currentTime: 10, duration: 100 });
		renderHook(() => useKeyboardShortcuts());

		const event = new KeyboardEvent("keydown", { key: "ArrowRight" });
		window.dispatchEvent(event);

		expect(mockUpdatePlaybackTime).toHaveBeenCalledWith(15, 100);
	});

	it("should not seek beyond duration when right arrow is pressed near end", () => {
		useAppStore.setState({ currentTime: 97, duration: 100 });
		renderHook(() => useKeyboardShortcuts());

		const event = new KeyboardEvent("keydown", { key: "ArrowRight" });
		window.dispatchEvent(event);

		expect(mockUpdatePlaybackTime).toHaveBeenCalledWith(100, 100);
	});

	it("should increase volume by 10% when up arrow is pressed", () => {
		useAppStore.setState({ volume: 50 });
		renderHook(() => useKeyboardShortcuts());

		const event = new KeyboardEvent("keydown", { key: "ArrowUp" });
		window.dispatchEvent(event);

		expect(mockSetVolume).toHaveBeenCalledWith(60);
	});

	it("should not increase volume above 100 when up arrow is pressed at high volume", () => {
		useAppStore.setState({ volume: 95 });
		renderHook(() => useKeyboardShortcuts());

		const event = new KeyboardEvent("keydown", { key: "ArrowUp" });
		window.dispatchEvent(event);

		expect(mockSetVolume).toHaveBeenCalledWith(100);
	});

	it("should decrease volume by 10% when down arrow is pressed", () => {
		useAppStore.setState({ volume: 50 });
		renderHook(() => useKeyboardShortcuts());

		const event = new KeyboardEvent("keydown", { key: "ArrowDown" });
		window.dispatchEvent(event);

		expect(mockSetVolume).toHaveBeenCalledWith(40);
	});

	it("should not decrease volume below 0 when down arrow is pressed at low volume", () => {
		useAppStore.setState({ volume: 5 });
		renderHook(() => useKeyboardShortcuts());

		const event = new KeyboardEvent("keydown", { key: "ArrowDown" });
		window.dispatchEvent(event);

		expect(mockSetVolume).toHaveBeenCalledWith(0);
	});

	it("should handle case-insensitive keys", () => {
		renderHook(() => useKeyboardShortcuts());

		const event = new KeyboardEvent("keydown", { key: "M" });
		window.dispatchEvent(event);

		expect(mockToggleMute).toHaveBeenCalledTimes(1);
	});

	it("should ignore keyboard events when typing in input", () => {
		renderHook(() => useKeyboardShortcuts());

		const input = document.createElement("input");
		document.body.appendChild(input);

		const event = new KeyboardEvent("keydown", {
			key: " ",
			bubbles: true,
		});
		Object.defineProperty(event, "target", {
			value: input,
			writable: false,
		});

		window.dispatchEvent(event);

		expect(mockSetPlaybackState).not.toHaveBeenCalled();

		document.body.removeChild(input);
	});

	it("should ignore keyboard events when typing in textarea", () => {
		renderHook(() => useKeyboardShortcuts());

		const textarea = document.createElement("textarea");
		document.body.appendChild(textarea);

		const event = new KeyboardEvent("keydown", {
			key: "m",
			bubbles: true,
		});
		Object.defineProperty(event, "target", {
			value: textarea,
			writable: false,
		});

		window.dispatchEvent(event);

		expect(mockToggleMute).not.toHaveBeenCalled();

		document.body.removeChild(textarea);
	});

	it("should cleanup event listener on unmount", () => {
		const removeEventListenerSpy = vi.spyOn(window, "removeEventListener");

		const { unmount } = renderHook(() => useKeyboardShortcuts());

		unmount();

		expect(removeEventListenerSpy).toHaveBeenCalledWith(
			"keydown",
			expect.any(Function),
		);

		removeEventListenerSpy.mockRestore();
	});

	it("should prevent default for space key", () => {
		renderHook(() => useKeyboardShortcuts());

		const event = new KeyboardEvent("keydown", { key: " " });
		const preventDefaultSpy = vi.spyOn(event, "preventDefault");

		window.dispatchEvent(event);

		expect(preventDefaultSpy).toHaveBeenCalled();
	});
});
