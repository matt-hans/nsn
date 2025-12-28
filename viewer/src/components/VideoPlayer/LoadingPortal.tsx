// ICN Viewer Client - Loading Portal Animation

import { useEffect, useState } from "react";

const messages = [
	"Scanning dimensional frequencies...",
	"Tuning interdimensional receiver...",
	"Stabilizing portal connection...",
	"Aligning quantum bandwidth...",
	"Locking onto dimension C-137...",
	"Bypassing galactic federation...",
	"Calibrating reality anchor...",
];

export default function LoadingPortal() {
	const [messageIndex, setMessageIndex] = useState(0);

	useEffect(() => {
		const interval = setInterval(() => {
			setMessageIndex((i) => (i + 1) % messages.length);
		}, 2500);

		return () => clearInterval(interval);
	}, []);

	return (
		<div className="loading-portal">
			<div className="portal-rings">
				<div className="ring ring-1" />
				<div className="ring ring-2" />
				<div className="ring ring-3" />
			</div>
			<p className="loading-message">{messages[messageIndex]}</p>
		</div>
	);
}
