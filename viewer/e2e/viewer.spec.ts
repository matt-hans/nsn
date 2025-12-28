import { expect, test } from "@playwright/test";

test.describe("ICN Viewer Client E2E", () => {
	test.beforeEach(async ({ page }) => {
		await page.goto("/");
	});

	test("app shell displays on launch", async ({ page }) => {
		await expect(page.locator(".topbar")).toBeVisible();
		await expect(page.locator(".video-player")).toBeVisible();
		await expect(page.locator(".topbar-logo")).toHaveText("ICN");
	});

	test("connecting status shows initially", async ({ page }) => {
		const status = page.locator(".connection-status");
		await expect(status).toContainText(/Connecting|Connected/);
	});

	test("connect to mock relay within 5 seconds", async ({ page }) => {
		await expect(page.locator(".connection-status")).toContainText(
			"Connected",
			{ timeout: 5000 },
		);
	});

	test("display slot number", async ({ page }) => {
		const slotDisplay = page.locator(".slot-display");
		await expect(slotDisplay).toBeVisible();
		await expect(slotDisplay).toContainText(/SLOT \d{5}/);
	});

	test("toggle sidebar with 'i' key", async ({ page }) => {
		// Press 'i' to open sidebar (it renders conditionally)
		await page.keyboard.press("i");
		await page.waitForTimeout(200); // Wait for state update and render

		// Sidebar should now be visible with 'open' class
		const sidebar = page.locator(".sidebar");
		await expect(sidebar).toBeVisible();
		await expect(sidebar).toHaveClass(/open/);

		// Press 'i' to close
		await page.keyboard.press("i");
		await page.waitForTimeout(200); // Wait for state update
		await expect(sidebar).not.toBeVisible();
	});

	test("toggle settings modal", async ({ page }) => {
		// Click settings button
		await page.click('button[aria-label="Settings"]');

		// Settings modal should be visible
		const modal = page.locator(".settings-modal");
		await expect(modal).toBeVisible();
		await expect(modal).toContainText("Settings");

		// Click cancel to close
		await page.click("text=Cancel");
		await expect(modal).not.toBeVisible();
	});

	test("play video for 30 seconds", async ({ page }, testInfo) => {
		// Increase timeout for this specific test
		testInfo.setTimeout(60000);

		// Wait for connection
		await expect(page.locator(".connection-status")).toContainText(
			"Connected",
			{ timeout: 5000 },
		);

		// Wait for buffering to complete and playback to start
		await page.waitForTimeout(10000); // 10 seconds for buffering

		// Check that canvas is present
		const canvas = page.locator("canvas.video-canvas");
		await expect(canvas).toBeVisible();

		// Wait additional 20 seconds (total 30)
		await page.waitForTimeout(20000);

		// Verify still playing (no error overlay)
		await expect(page.locator(".error-overlay")).not.toBeVisible();
	});

	test("toggle mute with 'm' key", async ({ page }) => {
		// Get initial state from store (assume volume > 0)
		await page.keyboard.press("m");
		// Wait a bit for state update
		await page.waitForTimeout(100);

		// Toggle back
		await page.keyboard.press("m");
		await page.waitForTimeout(100);

		// Verify no errors
		await expect(page.locator(".error-overlay")).not.toBeVisible();
	});

	test("volume adjustment with arrow keys", async ({ page }) => {
		// Up arrow increases volume
		await page.keyboard.press("ArrowUp");
		await page.waitForTimeout(100);

		// Down arrow decreases volume
		await page.keyboard.press("ArrowDown");
		await page.waitForTimeout(100);

		// Verify no errors
		await expect(page.locator(".error-overlay")).not.toBeVisible();
	});

	test("network stats in sidebar", async ({ page }) => {
		// Open sidebar
		await page.keyboard.press("i");

		const sidebar = page.locator(".sidebar");
		await expect(sidebar).toBeVisible();

		// Check for network stats
		await expect(sidebar.locator("text=Bitrate")).toBeVisible();
		await expect(sidebar.locator("text=Latency")).toBeVisible();
		await expect(sidebar.locator("text=Connected Peers")).toBeVisible();
		await expect(sidebar.locator("text=Buffer")).toBeVisible();
	});

	test("close settings with Escape", async ({ page }) => {
		// Open settings
		await page.click('button[aria-label="Settings"]');
		await expect(page.locator(".settings-modal")).toBeVisible();

		// Close with Escape
		await page.keyboard.press("Escape");
		await expect(page.locator(".settings-modal")).not.toBeVisible();
	});

	test("toggle seeding in settings", async ({ page }) => {
		// Open settings
		await page.click('button[aria-label="Settings"]');

		// Find seeding toggle
		const toggle = page.locator('button[aria-label="Toggle seeding"]');
		await expect(toggle).toBeVisible();

		// Toggle seeding on
		await toggle.click();

		// Save settings
		await page.click("text=Save Settings");

		// Verify modal closed
		await expect(page.locator(".settings-modal")).not.toBeVisible();
	});

	test("change quality preference", async ({ page }) => {
		// Open settings
		await page.click('button[aria-label="Settings"]');

		// Find quality dropdown
		const dropdown = page.locator('select[aria-label="Quality preference"]');
		await expect(dropdown).toBeVisible();

		// Change to 720p
		await dropdown.selectOption("720p");

		// Save settings
		await page.click("text=Save Settings");

		// Verify modal closed
		await expect(page.locator(".settings-modal")).not.toBeVisible();
	});

	test("settings persist across reload", async ({ page }) => {
		// Open settings and change quality
		await page.click('button[aria-label="Settings"]');
		const dropdown = page.locator('select[aria-label="Quality preference"]');
		await dropdown.selectOption("480p");
		await page.click("text=Save Settings");

		// Reload page
		await page.reload();

		// Open settings again
		await page.click('button[aria-label="Settings"]');

		// Verify quality is still 480p
		const newDropdown = page.locator('select[aria-label="Quality preference"]');
		await expect(newDropdown).toHaveValue("480p");
	});
});
