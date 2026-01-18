// ICN Viewer Client - Discovery Service
// HTTP-based node discovery for WebRTC bootstrap

/**
 * P2P info response envelope matching Rust P2pInfoResponse
 * Source: /home/matt/nsn/node-core/crates/p2p/src/discovery.rs
 */
interface P2pInfoResponse {
	success: boolean;
	data?: {
		peer_id: string;
		multiaddrs: string[];
		protocols: string[];
		features: {
			webrtc_enabled: boolean;
			role: string;
		};
	};
	error?: {
		code: string;
		message: string;
	};
}

/**
 * Discovery candidate with source tracking
 * Per CONTEXT.md: Tiered configuration priority
 */
export interface DiscoveryCandidate {
	url: string;
	source: "localStorage" | "settings" | "env" | "hardcoded";
}

/**
 * Default hardcoded bootstrap nodes (testnet only)
 * Per CONTEXT.md: Foundation fallback, will be shuffled to avoid hammering
 */
const HARDCODED_DEFAULTS: DiscoveryCandidate[] = [
	{ url: "http://localhost:9615", source: "hardcoded" },
];

/**
 * Discover a single node and extract its WebRTC multiaddr
 *
 * Process:
 * 1. Fetch `${baseUrl}/p2p/info`
 * 2. Handle 503 (return null for retry - node initializing)
 * 3. Parse JSON response
 * 4. Find multiaddr containing `/webrtc-direct/` AND `/certhash/`
 * 5. If WebRTC multiaddr found but missing `/p2p/<peer_id>`, append it
 * 6. Return null if no WebRTC address available (caller treats as failure per CONTEXT.md)
 *
 * @param baseUrl - Base URL of the node to discover
 * @returns WebRTC multiaddr string, or null if unavailable/temporary failure
 * @throws Error if response is non-successful (not 503) or malformed
 */
export async function discoverNode(baseUrl: string): Promise<string | null> {
	try {
		const response = await fetch(`${baseUrl}/p2p/info`, {
			headers: {
				Accept: "application/json",
			},
		});

		// 503 = Node still initializing, can retry
		if (response.status === 503) {
			console.log(`[Discovery] Node at ${baseUrl} is initializing (503)`);
			return null;
		}

		if (!response.ok) {
			throw new Error(`HTTP ${response.status}: ${response.statusText}`);
		}

		const info: P2pInfoResponse = await response.json();

		if (!info.success || !info.data) {
			const errorMsg = info.error?.message || "Unknown error";
			throw new Error(`Discovery failed: ${errorMsg}`);
		}

		// Check if WebRTC is enabled
		if (!info.data.features.webrtc_enabled) {
			console.warn(`[Discovery] Node at ${baseUrl} has WebRTC disabled`);
			return null;
		}

		// Find WebRTC-Direct multiaddr with certhash
		const webrtcAddr = info.data.multiaddrs.find(
			(addr) => addr.includes("/webrtc-direct/") && addr.includes("/certhash/"),
		);

		if (!webrtcAddr) {
			console.warn(
				`[Discovery] Node at ${baseUrl} has no WebRTC address available`,
			);
			return null;
		}

		// Append peer ID if not present (required for WebRTC-Direct)
		if (!webrtcAddr.includes("/p2p/")) {
			return `${webrtcAddr}/p2p/${info.data.peer_id}`;
		}

		return webrtcAddr;
	} catch (error) {
		if (error instanceof TypeError && error.message.includes("fetch")) {
			// Network error - treat as null for retry
			console.warn(`[Discovery] Network error fetching ${baseUrl}:`, error);
			return null;
		}
		throw error;
	}
}

/**
 * Shuffle array in place (Fisher-Yates)
 * Used for hardcoded defaults to avoid hammering first node
 */
function shuffle<T>(array: T[]): T[] {
	const result = [...array];
	for (let i = result.length - 1; i > 0; i--) {
		const j = Math.floor(Math.random() * (i + 1));
		[result[i], result[j]] = [result[j], result[i]];
	}
	return result;
}

/**
 * Discover a node using parallel race pattern
 *
 * Per CONTEXT.md Discovery Behavior:
 * - Process in batches of 3
 * - Race each batch with timeout (default 3000ms)
 * - First valid response wins
 * - On success, save to localStorage
 * - Shuffle hardcoded entries to avoid hammering first node
 *
 * @param candidates - List of discovery candidates
 * @param batchSize - Number of candidates to try in parallel (default: 3)
 * @param timeoutMs - Timeout for each batch in milliseconds (default: 3000)
 * @returns WebRTC multiaddr string of first successful discovery
 * @throws Error if all candidates fail
 */
export async function discoverWithRace(
	candidates: DiscoveryCandidate[],
	batchSize = 3,
	timeoutMs = 3000,
): Promise<string> {
	if (candidates.length === 0) {
		throw new Error("No discovery candidates provided");
	}

	// Separate hardcoded entries for shuffling (avoid hammering first node)
	const hardcoded = candidates.filter((c) => c.source === "hardcoded");
	const nonHardcoded = candidates.filter((c) => c.source !== "hardcoded");

	// Build final list: non-hardcoded first (preserves priority), then shuffled hardcoded
	const shuffledCandidates = [...nonHardcoded, ...shuffle(hardcoded)];

	// Process in batches
	for (let i = 0; i < shuffledCandidates.length; i += batchSize) {
		const batch = shuffledCandidates.slice(i, i + batchSize);

		console.log(
			`[Discovery] Trying batch ${Math.floor(i / batchSize) + 1}:`,
			batch.map((c) => c.url),
		);

		try {
			// Race: first success wins, or timeout
			const result = await Promise.race([
				// Try all candidates in batch
				...batch.map(async (candidate) => {
					const multiaddr = await discoverNode(candidate.url);
					if (multiaddr) {
						// Save to localStorage on success
						localStorage.setItem("last_known_node", candidate.url);
						console.log(
							`[Discovery] Found node at ${candidate.url}: ${multiaddr.slice(0, 50)}...`,
						);
						return { multiaddr, url: candidate.url };
					}
					return null;
				}),
				// Timeout fallback
				new Promise<null>((_, reject) =>
					setTimeout(() => reject(new Error("Batch timeout")), timeoutMs),
				),
			]);

			if (result?.multiaddr) {
				return result.multiaddr;
			}
		} catch (error) {
			// Batch timeout or all failed - continue to next batch
			console.log(
				`[Discovery] Batch ${Math.floor(i / batchSize) + 1} failed or timed out`,
			);
		}
	}

	throw new Error("All discovery candidates failed");
}

/**
 * Build candidate list in priority order per CONTEXT.md
 *
 * Priority:
 * 1. localStorage.getItem('last_known_node') (if exists)
 * 2. User settings (placeholder for future)
 * 3. import.meta.env.VITE_BOOTSTRAP_NODES (comma-separated)
 * 4. Hardcoded defaults (testnet only)
 *
 * @returns Array of discovery candidates in priority order
 */
export function buildCandidateList(): DiscoveryCandidate[] {
	const candidates: DiscoveryCandidate[] = [];

	// Priority 1: Last known good node from localStorage
	const lastKnown = localStorage.getItem("last_known_node");
	if (lastKnown) {
		candidates.push({ url: lastKnown, source: "localStorage" });
	}

	// Priority 2: User settings (placeholder for future UI setting)
	// const userSetting = getUserSetting('bootstrap_node');
	// if (userSetting) {
	//   candidates.push({ url: userSetting, source: 'settings' });
	// }

	// Priority 3: Environment variable
	const envNodes = import.meta.env.VITE_BOOTSTRAP_NODES;
	if (envNodes) {
		const urls = envNodes
			.split(",")
			.map((s: string) => s.trim())
			.filter(Boolean);
		for (const url of urls) {
			candidates.push({ url, source: "env" });
		}
	}

	// Priority 4: Hardcoded defaults (testnet only)
	candidates.push(...HARDCODED_DEFAULTS);

	// Remove duplicates (preserving priority order)
	const seen = new Set<string>();
	const unique: DiscoveryCandidate[] = [];
	for (const candidate of candidates) {
		if (!seen.has(candidate.url)) {
			seen.add(candidate.url);
			unique.push(candidate);
		}
	}

	return unique;
}
