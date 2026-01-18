// ICN Viewer - Development Signaling Server
// Minimal WebSocket signaling server for WebRTC peer discovery
// Usage: node scripts/signaling-server.js [port]

import { createServer } from "http";
import { WebSocketServer } from "ws";

const PORT = Number.parseInt(process.argv[2] || "8080", 10);

// Create HTTP server for both WebSocket and REST endpoints
const httpServer = createServer((req, res) => {
	// Enable CORS for development
	res.setHeader("Access-Control-Allow-Origin", "*");
	res.setHeader("Access-Control-Allow-Methods", "GET, OPTIONS");
	res.setHeader("Access-Control-Allow-Headers", "Content-Type");

	if (req.method === "OPTIONS") {
		res.writeHead(204);
		res.end();
		return;
	}

	// REST endpoint for relay discovery
	if (req.url === "/relays" && req.method === "GET") {
		const relays = [
			{
				peer_id: "12D3KooWSignalDev",
				multiaddr: `/ip4/127.0.0.1/tcp/${PORT}/ws`,
				region: "local-dev",
				is_fallback: false,
			},
		];
		res.writeHead(200, { "Content-Type": "application/json" });
		res.end(JSON.stringify(relays));
		return;
	}

	// Health check
	if (req.url === "/health" && req.method === "GET") {
		res.writeHead(200, { "Content-Type": "application/json" });
		res.end(JSON.stringify({ status: "ok", peers: peers.size }));
		return;
	}

	res.writeHead(404);
	res.end("Not Found");
});

// Create WebSocket server
const wss = new WebSocketServer({ server: httpServer });

// Track connected peers: peerId -> WebSocket
const peers = new Map();

wss.on("connection", (ws) => {
	let peerId = null;

	ws.on("message", (data) => {
		try {
			const msg = JSON.parse(data.toString());

			switch (msg.type) {
				case "join":
					peerId = msg.from;
					peers.set(peerId, ws);
					console.log(`[JOIN] Peer joined: ${peerId} (total: ${peers.size})`);

					// Send current peer list to new peer
					const peerList = Array.from(peers.keys()).filter(
						(id) => id !== peerId,
					);
					ws.send(JSON.stringify({ type: "peer-list", payload: peerList }));

					// Notify existing peers about the new peer
					for (const [id, peerWs] of peers) {
						if (id !== peerId && peerWs.readyState === 1) {
							peerWs.send(
								JSON.stringify({ type: "peer-list", payload: [peerId] }),
							);
						}
					}
					break;

				case "offer":
				case "answer":
				case "ice-candidate": {
					// Relay signaling message to target peer
					const targetWs = peers.get(msg.to);
					if (targetWs && targetWs.readyState === 1) {
						targetWs.send(JSON.stringify(msg));
						console.log(`[RELAY] ${msg.type}: ${msg.from} -> ${msg.to}`);
					} else {
						console.log(`[WARN] Target peer not found: ${msg.to}`);
					}
					break;
				}

				case "leave":
					if (peerId) {
						peers.delete(peerId);
						console.log(`[LEAVE] Peer left: ${peerId} (total: ${peers.size})`);
					}
					break;

				default:
					console.log(`[WARN] Unknown message type: ${msg.type}`);
			}
		} catch (error) {
			console.error("[ERROR] Failed to parse message:", error);
		}
	});

	ws.on("close", () => {
		if (peerId) {
			peers.delete(peerId);
			console.log(
				`[DISCONNECT] Peer disconnected: ${peerId} (total: ${peers.size})`,
			);
		}
	});

	ws.on("error", (error) => {
		console.error(`[ERROR] WebSocket error for ${peerId}:`, error.message);
	});
});

httpServer.listen(PORT, () => {
	console.log(`
╔════════════════════════════════════════════════════════════╗
║           ICN Viewer - Development Signaling Server        ║
╠════════════════════════════════════════════════════════════╣
║  WebSocket: ws://localhost:${PORT}                            ║
║  HTTP:      http://localhost:${PORT}                          ║
║  Relays:    http://localhost:${PORT}/relays                   ║
║  Health:    http://localhost:${PORT}/health                   ║
╚════════════════════════════════════════════════════════════╝
  `);
});

// Graceful shutdown
process.on("SIGINT", () => {
	console.log("\n[SHUTDOWN] Closing signaling server...");
	wss.close();
	httpServer.close();
	process.exit(0);
});
