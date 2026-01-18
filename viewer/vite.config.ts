/// <reference types="vitest" />
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// https://vitejs.dev/config/
export default defineConfig({
	plugins: [react()],

	// Clear screen on rebuild
	clearScreen: false,

	// Development server configuration
	server: {
		port: 5173,
		strictPort: true,
		// Allow access from local network for mobile testing
		host: true,
		// Allow all hostnames (for Tailscale, SSH, etc.)
		allowedHosts: true,
	},

	// Preview server configuration
	preview: {
		port: 4173,
		strictPort: true,
	},

	// Environment variable prefix
	envPrefix: "VITE_",

	// Build configuration for standalone web deployment
	build: {
		target: "es2021",
		outDir: "dist",
		sourcemap: true,
		// Chunk splitting for optimal caching
		rollupOptions: {
			output: {
				manualChunks: {
					vendor: ["react", "react-dom", "zustand"],
					libp2p: [
						"libp2p",
						"@libp2p/webrtc",
						"@chainsafe/libp2p-noise",
						"@chainsafe/libp2p-yamux",
						"@chainsafe/libp2p-gossipsub",
					],
					polkadot: ["@polkadot/types", "@polkadot/types-codec"],
				},
			},
		},
		// Minimum chunk size for splitting
		chunkSizeWarningLimit: 500,
	},

	// Base path for deployment (can be customized via VITE_BASE_PATH)
	base: process.env.VITE_BASE_PATH || "/",

	// Define environment variables
	define: {
		// Default signaling server URL (can be overridden via .env)
		"import.meta.env.VITE_SIGNALING_URL": JSON.stringify(
			process.env.VITE_SIGNALING_URL || "ws://localhost:8080",
		),
	},

	// Test configuration
	test: {
		globals: true,
		environment: "jsdom",
		setupFiles: ["./src/test/setup.ts"],
		include: ["src/**/*.{test,spec}.{js,ts,jsx,tsx}"],
		exclude: ["**/node_modules/**", "**/dist/**", "**/e2e/**"],
		coverage: {
			reporter: ["text", "json", "html"],
			exclude: ["node_modules/", "src/test/", "**/*.d.ts"],
		},
	},
});
