// ICN Viewer Client - Discovery Service
// HTTP-based node discovery for WebRTC bootstrap

/**
 * Error codes for discovery failures
 */
export type DiscoveryErrorCode =
	| "NETWORK_ERROR"
	| "HTTP_ERROR"
	| "NODE_INITIALIZING"
	| "WEBRTC_DISABLED"
	| "NO_WEBRTC_ADDRESS"
	| "PARSE_ERROR"
	| "ALL_CANDIDATES_FAILED";

/**
 * Structured error for discovery failures with diagnostic information
 */
export class DiscoveryError extends Error {
	public readonly code: DiscoveryErrorCode;
	public readonly url?: string;
	public readonly httpStatus?: number;
	public readonly cause?: unknown;

	constructor(
		code: DiscoveryErrorCode,
		message: string,
		options?: { url?: string; httpStatus?: number; cause?: unknown }
	) {
		super(`[${code}] ${message}`);
		this.name = "DiscoveryError";
		this.code = code;
		this.url = options?.url;
		this.httpStatus = options?.httpStatus;
		this.cause = options?.cause;
	}

	/**
	 * Get a user-friendly diagnostic message with remediation hints
	 */
	getDiagnostic(): string {
		switch (this.code) {
			case "NETWORK_ERROR":
				return `Cannot reach discovery endpoint${this.url ? ` at ${this.url}` : ""}. Check if node is running and port is correct.`;
			case "HTTP_ERROR":
				return `Discovery endpoint returned HTTP ${this.httpStatus || "error"}. Check node logs for details.`;
			case "NODE_INITIALIZING":
				return "Node is still starting up. Please wait and try again.";
			case "WEBRTC_DISABLED":
				return "Node has WebRTC disabled. Node operator must start with --p2p-enable-webrtc flag.";
			case "NO_WEBRTC_ADDRESS":
				return "Node has no WebRTC address available. Check node configuration and certificate generation.";
			case "PARSE_ERROR":
				return "Invalid response from discovery endpoint. Check node version compatibility.";
			case "ALL_CANDIDATES_FAILED":
				return "All bootstrap nodes unreachable. Check network connectivity and node availability.";
			default:
				return this.message;
		}
	}
}

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
 * Default discovery port for P2P info endpoint
 * Must match Rust node's metrics_port default (9100)
 * Can be overridden via VITE_P2P_DISCOVERY_PORT environment variable
 */
const DEFAULT_DISCOVERY_PORT = 9100;

/**
 * Get the discovery port from environment or use default
 */
function getDiscoveryPort(): number {
	const envPort = import.meta.env.VITE_P2P_DISCOVERY_PORT;
	if (envPort) {
		const parsed = parseInt(envPort, 10);
		if (!isNaN(parsed) && parsed > 0 && parsed < 65536) {
			return parsed;
		}
		console.warn(
			`[Discovery] Invalid VITE_P2P_DISCOVERY_PORT: ${envPort}, using default ${DEFAULT_DISCOVERY_PORT}`,
		);
	}
	return DEFAULT_DISCOVERY_PORT;
}

/**
 * Get the discovery protocol - always HTTP since Rust node doesn't serve HTTPS
 * Note: This may cause mixed-content issues if viewer is served over HTTPS,
 * but the alternative (HTTPS) won't work since the Rust node only serves HTTP.
 * For production, consider running the metrics server behind an HTTPS reverse proxy.
 */
function getDiscoveryProtocol(): string {
	// Check for explicit protocol override via environment variable
	const envProtocol = import.meta.env.VITE_P2P_DISCOVERY_PROTOCOL;
	if (envProtocol === "https" || envProtocol === "http") {
		return envProtocol;
	}

	// Default to HTTP since Rust node only serves HTTP on the metrics port
	// If browser is on HTTPS, this will cause mixed-content warnings
	// but HTTPS won't work at all since the node doesn't support it
	return "http";
}

function isLoopbackHost(hostname: string): boolean {
	return hostname === "localhost" || hostname === "127.0.0.1" || hostname === "0.0.0.0";
}

function isLoopbackAddr(addr: string): boolean {
	return addr.includes("/ip4/127.") || addr.includes("/ip4/0.0.0.0/") || addr.includes("/ip6/::1/");
}

/**
 * Default hardcoded bootstrap nodes (testnet only)
 * Per CONTEXT.md: Foundation fallback, will be shuffled to avoid hammering
 *
 * Derives from current window.location to support:
 * - Local development (localhost)
 * - Tailscale access (hostname like "pc")
 * - LAN access (IP addresses)
 *
 * Note: Uses HTTP protocol explicitly since Rust node only serves HTTP.
 * If you need HTTPS, set VITE_P2P_DISCOVERY_PROTOCOL=https and run a reverse proxy.
 */
const HARDCODED_DEFAULTS: DiscoveryCandidate[] = [
	{
		// Use same hostname as viewer, with HTTP protocol and configurable port
		// HTTP is required since Rust node's metrics server only supports HTTP
		url: `${getDiscoveryProtocol()}://${window.location.hostname}:${getDiscoveryPort()}`,
		source: "hardcoded",
	},
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
 * @returns Ordered list of WebRTC multiaddrs, or null if unavailable/temporary failure
 * @throws Error if response is non-successful (not 503) or malformed
 */
export async function discoverNode(baseUrl: string): Promise<string[] | null> {
	try {
		const response = await fetch(`${baseUrl}/p2p/info`, {
			headers: {
				Accept: "application/json",
			},
		});

		// 503 = Node still initializing, can retry
		if (response.status === 503) {
			console.log(`[Discovery] Node at ${baseUrl} is initializing (503)`);
			// Return null to allow retry - this is a temporary state
			return null;
		}

		if (!response.ok) {
			throw new DiscoveryError(
				"HTTP_ERROR",
				`Discovery endpoint returned HTTP ${response.status}: ${response.statusText}`,
				{ url: baseUrl, httpStatus: response.status }
			);
		}

		let info: P2pInfoResponse;
		try {
			info = await response.json();
		} catch (parseError) {
			throw new DiscoveryError(
				"PARSE_ERROR",
				`Invalid JSON response from discovery endpoint`,
				{ url: baseUrl, cause: parseError }
			);
		}

		if (!info.success || !info.data) {
			const errorMsg = info.error?.message || "Unknown error";
			throw new DiscoveryError(
				"HTTP_ERROR",
				`Discovery failed: ${errorMsg}`,
				{ url: baseUrl }
			);
		}

		// Check if WebRTC is enabled - this is a configuration issue on the node side
		if (!info.data.features.webrtc_enabled) {
			console.warn(
				`[Discovery] Node at ${baseUrl} has WebRTC disabled. ` +
				`Node must be started with --p2p-enable-webrtc flag.`
			);
			// Throw structured error so caller knows why this node failed
			throw new DiscoveryError(
				"WEBRTC_DISABLED",
				`Node has WebRTC transport disabled`,
				{ url: baseUrl }
			);
		}

		// Find WebRTC-Direct multiaddr with certhash
		// Prefer addresses that match the current hostname (for Tailscale/LAN access)
		const currentHostname = window.location.hostname;

		// Get all WebRTC addresses with certhash
		const webrtcAddrs = info.data.multiaddrs.filter(
			(addr) => addr.includes("/webrtc-direct/") && addr.includes("/certhash/"),
		);

		// Also check for WebRTC addresses without certhash (configuration issue)
		const webrtcAddrsNoCerthash = info.data.multiaddrs.filter(
			(addr) => addr.includes("/webrtc-direct/") && !addr.includes("/certhash/"),
		);

		if (webrtcAddrs.length === 0) {
			if (webrtcAddrsNoCerthash.length > 0) {
				console.warn(
					`[Discovery] Node at ${baseUrl} has WebRTC addresses but missing /certhash/. ` +
					`This may indicate a certificate generation issue. Addresses: ${webrtcAddrsNoCerthash.join(", ")}`
				);
			}
			throw new DiscoveryError(
				"NO_WEBRTC_ADDRESS",
				`Node has no WebRTC address with certhash available`,
				{ url: baseUrl }
			);
		}

		let baseHost: string | undefined;
		try {
			baseHost = new URL(baseUrl).hostname;
		} catch {
			baseHost = undefined;
		}
		const baseHostIsIp =
			baseHost != null && /^\d{1,3}(\.\d{1,3}){3}$/.test(baseHost);

		const appendPeerId = (addr: string): string =>
			addr.includes("/p2p/") ? addr : `${addr}/p2p/${info.data.peer_id}`;

		const prioritized: string[] = [];
		const pushAddr = (addr?: string): void => {
			if (!addr) {
				return;
			}
			const withPeer = appendPeerId(addr);
			if (!prioritized.includes(withPeer)) {
				prioritized.push(withPeer);
			}
		};

		// Try to find address matching current hostname unless it's loopback
		if (!isLoopbackHost(currentHostname)) {
			pushAddr(webrtcAddrs.find((addr) => addr.includes(currentHostname)));
		}

		// Prefer the discovery host if it is an explicit IP
		if (baseHostIsIp) {
			pushAddr(webrtcAddrs.find((addr) => addr.includes(`/ip4/${baseHost}/`)));
		}

		// Prefer LAN IP (192.168.x.x) for local network WebRTC
		// Tailscale doesn't forward UDP, so we need LAN IP for WebRTC
		pushAddr(
			webrtcAddrs.find((addr) => addr.match(/\/ip4\/192\.168\.\d+\.\d+\//)),
		);

		// Prefer Tailscale IP (100.x.x.x) if available
		// Note: WebRTC over Tailscale UDP may not work without ACL configuration
		pushAddr(
			webrtcAddrs.find((addr) => addr.match(/\/ip4\/100\.\d+\.\d+\.\d+\//)),
		);

		// Prefer any non-loopback address before falling back to localhost
		pushAddr(webrtcAddrs.find((addr) => !isLoopbackAddr(addr)));

		// Final fallback: allow loopback (localhost testing)
		pushAddr(webrtcAddrs.find((addr) => isLoopbackAddr(addr)));

		// Append any remaining addresses as last-resort fallbacks
		for (const addr of webrtcAddrs) {
			pushAddr(addr);
		}

		if (prioritized.length === 0) {
			throw new DiscoveryError(
				"NO_WEBRTC_ADDRESS",
				`Node has no WebRTC address with certhash available`,
				{ url: baseUrl }
			);
		}

		console.log(
			`[Discovery] Found ${prioritized.length} WebRTC address candidates. ` +
			`Primary: ${prioritized[0].slice(0, 60)}...`
		);

		return prioritized;
	} catch (error) {
		// Re-throw DiscoveryError as-is
		if (error instanceof DiscoveryError) {
			throw error;
		}

		const errorMsg = String(error).toLowerCase();

		// Detect mixed content errors (HTTPS page trying to fetch HTTP)
		if (
			errorMsg.includes("mixed content") ||
			errorMsg.includes("insecure") ||
			errorMsg.includes("blocked") ||
			(error instanceof TypeError && errorMsg.includes("failed to fetch") && baseUrl.startsWith("http://"))
		) {
			console.error(
				`[Discovery] Mixed content error: Browser may be blocking HTTP request from HTTPS page. ` +
				`Discovery URL: ${baseUrl}. Consider using HTTP for the viewer or setting up HTTPS proxy.`
			);
			throw new DiscoveryError(
				"NETWORK_ERROR",
				`Mixed content blocked: Cannot make HTTP request from HTTPS page`,
				{ url: baseUrl, cause: error }
			);
		}

		// Handle network errors (fetch failures - connection refused, timeout, etc.)
		if (error instanceof TypeError) {
			console.warn(`[Discovery] Network error fetching ${baseUrl}:`, error);
			throw new DiscoveryError(
				"NETWORK_ERROR",
				`Cannot reach discovery endpoint: ${errorMsg.includes("timeout") ? "connection timed out" : "connection failed"}`,
				{ url: baseUrl, cause: error }
			);
		}

		// Wrap unknown errors
		throw new DiscoveryError(
			"HTTP_ERROR",
			`Discovery failed: ${error}`,
			{ url: baseUrl, cause: error }
		);
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
 * @returns WebRTC multiaddr candidates of first successful discovery
 * @throws Error if all candidates fail
 */
export async function discoverWithRace(
	candidates: DiscoveryCandidate[],
	batchSize = 3,
	timeoutMs = 3000,
): Promise<string[]> {
	if (candidates.length === 0) {
		throw new DiscoveryError(
			"ALL_CANDIDATES_FAILED",
			"No discovery candidates provided"
		);
	}

	// Track errors from all attempts for diagnostic reporting
	const collectedErrors: DiscoveryError[] = [];

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
				// Try all candidates in batch, collecting errors
				...batch.map(async (candidate) => {
					try {
						console.log(`[Discovery] Attempting: ${candidate.url}`);
						const multiaddrs = await discoverNode(candidate.url);
						if (multiaddrs && multiaddrs.length > 0) {
							// Save to localStorage on success
							localStorage.setItem("last_known_node", candidate.url);
							console.log(
								`[Discovery] Found node at ${candidate.url}: ${multiaddrs[0].slice(0, 50)}...`,
							);
							return { multiaddrs, url: candidate.url };
						}
						// discoverNode returned null (e.g., 503 or no WebRTC address found)
						console.warn(`[Discovery] ${candidate.url}: returned null (node may be initializing)`);
						// Track this as NODE_INITIALIZING so it shows in summary
						collectedErrors.push(new DiscoveryError(
							"NODE_INITIALIZING",
							"Node returned null response (may be starting up)",
							{ url: candidate.url }
						));
						return null;
					} catch (error) {
						// Collect error for diagnostic reporting
						if (error instanceof DiscoveryError) {
							collectedErrors.push(error);
							console.warn(`[Discovery] ${candidate.url}: ${error.code} - ${error.getDiagnostic()}`);
						} else {
							const wrapped = new DiscoveryError(
								"HTTP_ERROR",
								`Discovery failed: ${error}`,
								{ url: candidate.url, cause: error }
							);
							collectedErrors.push(wrapped);
							console.warn(`[Discovery] ${candidate.url}: ${error}`);
						}
						return null;
					}
				}),
				// Timeout fallback
				new Promise<null>((_, reject) =>
					setTimeout(() => reject(new Error("Batch timeout")), timeoutMs),
				),
			]);

			if (result?.multiaddrs) {
				return result.multiaddrs;
			}
		} catch (error) {
			// Batch timeout - continue to next batch
			console.log(
				`[Discovery] Batch ${Math.floor(i / batchSize) + 1} timed out`,
			);
		}
	}

	// All candidates failed - provide detailed diagnostic
	const errorSummary = summarizeDiscoveryErrors(collectedErrors);
	throw new DiscoveryError(
		"ALL_CANDIDATES_FAILED",
		`All ${candidates.length} discovery candidates failed. ${errorSummary}`
	);
}

/**
 * Summarize collected discovery errors for user-friendly reporting
 */
function summarizeDiscoveryErrors(errors: DiscoveryError[]): string {
	if (errors.length === 0) {
		return "No specific errors collected (may have timed out).";
	}

	// Count errors by type and collect URLs
	const counts: Record<string, number> = {};
	const urlsByCode: Record<string, string[]> = {};
	for (const err of errors) {
		counts[err.code] = (counts[err.code] || 0) + 1;
		if (!urlsByCode[err.code]) {
			urlsByCode[err.code] = [];
		}
		if (err.url) {
			urlsByCode[err.code].push(err.url);
		}
	}

	const parts: string[] = [];
	if (counts.NETWORK_ERROR) {
		const urls = urlsByCode.NETWORK_ERROR?.join(", ") || "";
		parts.push(`${counts.NETWORK_ERROR} unreachable${urls ? ` (${urls})` : ""}`);
	}
	if (counts.WEBRTC_DISABLED) {
		parts.push(`${counts.WEBRTC_DISABLED} have WebRTC disabled (need --p2p-enable-webrtc)`);
	}
	if (counts.NO_WEBRTC_ADDRESS) {
		parts.push(`${counts.NO_WEBRTC_ADDRESS} missing certhash`);
	}
	if (counts.HTTP_ERROR) {
		parts.push(`${counts.HTTP_ERROR} HTTP errors`);
	}
	if (counts.PARSE_ERROR) {
		parts.push(`${counts.PARSE_ERROR} parse errors`);
	}
	if (counts.NODE_INITIALIZING) {
		parts.push(`${counts.NODE_INITIALIZING} still initializing`);
	}

	// Log detailed errors for debugging
	console.error("[Discovery] Error details:", errors.map(e => ({
		code: e.code,
		url: e.url,
		message: e.message,
		diagnostic: e.getDiagnostic()
	})));

	return parts.length > 0 ? parts.join(", ") : `Unknown errors: ${Object.keys(counts).join(", ")}`;
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
/**
 * Validate and potentially fix a localStorage URL
 * Returns null if the URL should be discarded (stale/invalid)
 */
function validateLocalStorageUrl(url: string): string | null {
	try {
		const parsed = new URL(url);
		const expectedPort = getDiscoveryPort();
		const expectedProtocol = getDiscoveryProtocol();

		// Check for stale port (e.g., old 9101 vs new 9100)
		if (parsed.port && parseInt(parsed.port, 10) !== expectedPort) {
			console.warn(
				`[Discovery] Clearing stale localStorage URL with wrong port: ${url} ` +
				`(expected port ${expectedPort})`
			);
			localStorage.removeItem("last_known_node");
			return null;
		}

		// Check for protocol mismatch (e.g., https when we need http)
		if (parsed.protocol !== `${expectedProtocol}:`) {
			console.warn(
				`[Discovery] Clearing stale localStorage URL with wrong protocol: ${url} ` +
				`(expected ${expectedProtocol})`
			);
			localStorage.removeItem("last_known_node");
			return null;
		}

		return url;
	} catch {
		// Invalid URL format
		console.warn(`[Discovery] Clearing invalid localStorage URL: ${url}`);
		localStorage.removeItem("last_known_node");
		return null;
	}
}

export function buildCandidateList(): DiscoveryCandidate[] {
	const candidates: DiscoveryCandidate[] = [];

	// Priority 1: Last known good node from localStorage (with validation)
	const lastKnown = localStorage.getItem("last_known_node");
	if (lastKnown) {
		const validatedUrl = validateLocalStorageUrl(lastKnown);
		if (validatedUrl) {
			candidates.push({ url: validatedUrl, source: "localStorage" });
		}
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

	// Log the candidate list for debugging
	console.log(
		`[Discovery] Built candidate list (${unique.length} candidates):`,
		unique.map(c => `${c.source}: ${c.url}`)
	);

	return unique;
}
