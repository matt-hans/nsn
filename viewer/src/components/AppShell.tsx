// ICN Viewer Client - App Shell with VHS Quantum atmosphere

import type { ReactNode } from "react";

interface AppShellProps {
	children: ReactNode;
}

export default function AppShell({ children }: AppShellProps) {
	return (
		<div className="app-shell">
			{children}
			<div className="scanlines" />
			<div className="noise" />
			<div className="vignette" />
		</div>
	);
}
