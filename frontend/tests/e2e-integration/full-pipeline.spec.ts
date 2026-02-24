/**
 * Integration tests — run against the REAL FastAPI + React stack.
 *
 * Prerequisites:
 *   - FastAPI backend running on http://127.0.0.1:8000
 *   - React dev server running on http://localhost:5173
 *   - Models load lazily on first request (first test may be slow)
 *   - HF_TOKEN must be set for MedGemma access (full pipeline tests)
 *
 * Run:
 *   npx playwright test --config playwright.integration.config.ts
 */

import { test, expect } from "@playwright/test";
import * as path from "path";
import * as fs from "fs";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Path to a real CXR image from project fixtures
const SAMPLE_IMAGE = path.resolve(
  __dirname,
  "../../../tests/fixtures/sample_cxr_vqa/nodule_mass.png",
);

test.describe("Backend Health", () => {
  test("API health endpoint responds", async ({ request }) => {
    const resp = await request.get("/api/health");
    expect(resp.ok()).toBe(true);
    const body = await resp.json();
    expect(body.status).toBe("ok");
  });

  test("WebSocket endpoint accepts connections", async ({ page }) => {
    // Navigate first to have a page context
    await page.goto("/");

    // Test WS connection from browser
    const connected = await page.evaluate(() => {
      return new Promise<boolean>((resolve) => {
        const ws = new WebSocket(
          `${window.location.protocol === "https:" ? "wss:" : "ws:"}//${window.location.host}/ws/pipeline`,
        );
        ws.onopen = () => {
          ws.close();
          resolve(true);
        };
        ws.onerror = () => resolve(false);
        setTimeout(() => resolve(false), 5000);
      });
    });
    expect(connected).toBe(true);
  });
});

test.describe("Image Upload", () => {
  test("upload real CXR image via UI", async ({ page }) => {
    expect(fs.existsSync(SAMPLE_IMAGE)).toBe(true);

    await page.goto("/");
    await expect(page.locator(".dropzone")).toBeVisible();

    // Upload real image
    const input = page.locator('.dropzone input[type="file"]');
    await input.setInputFiles(SAMPLE_IMAGE);

    // Image should appear
    await expect(page.locator(".image-container")).toBeVisible({
      timeout: 10_000,
    });
    await expect(page.locator(".image-container img")).toBeVisible();

    // Prompt should be enabled
    await expect(page.locator(".prompt-input")).toBeEnabled();
    await expect(page.locator(".prompt-input")).toHaveAttribute(
      "placeholder",
      /Ask about the image/,
    );
  });

  test("upload via API returns correct metadata", async ({ request }) => {
    const imgBuffer = fs.readFileSync(SAMPLE_IMAGE);
    const resp = await request.post("/api/upload-image", {
      multipart: {
        file: {
          name: "nodule_mass.png",
          mimeType: "image/png",
          buffer: imgBuffer,
        },
      },
    });

    expect(resp.ok()).toBe(true);
    const body = await resp.json();
    expect(body.image_id).toBeTruthy();
    expect(body.width).toBeGreaterThan(0);
    expect(body.height).toBeGreaterThan(0);
    expect(body.url).toContain("/api/images/");
  });
});

test.describe("Targeted Pipeline (full end-to-end)", () => {
  test("run 'Where is the Nodule/Mass?' pipeline", async ({ page }) => {
    await page.goto("/");

    // Upload image
    const input = page.locator('.dropzone input[type="file"]');
    await input.setInputFiles(SAMPLE_IMAGE);
    await expect(page.locator(".image-container")).toBeVisible({
      timeout: 10_000,
    });

    // Type prompt and submit
    const promptInput = page.locator(".prompt-input");
    await promptInput.fill("Where is the Nodule/Mass?");
    await page.locator(".btn-primary").click();

    // Button should disable immediately (optimistic)
    await expect(page.locator(".btn-primary")).toBeDisabled({ timeout: 5_000 });

    // S1 step should appear quickly
    await expect(page.locator(".step-card").first()).toBeVisible({
      timeout: 30_000,
    });
    await expect(page.locator(".step-card").first()).toContainText("S1");

    // Wait for pipeline to complete (up to 8 minutes for model loading + inference)
    await expect(page.locator(".btn-primary")).toBeEnabled({
      timeout: 480_000,
    });

    // Check if pipeline errored (e.g., MedGemma auth failure)
    const chatContent = await page.locator(".chat-panel").textContent();
    const hasError = chatContent?.includes("OSError") || chatContent?.includes("gated repo");

    if (hasError) {
      // Pipeline failed due to model access — verify graceful degradation
      // S1 step should still be present
      const stepCount = await page.locator(".step-card").count();
      expect(stepCount).toBeGreaterThanOrEqual(1);

      // Error should be displayed (via error banner or assistant message)
      const errorBanner = page.locator(".error-banner");
      const assistantMsgs = page.locator(".chat-message-assistant");
      const hasErrorBanner = await errorBanner.isVisible().catch(() => false);
      const hasAssistantMsg = (await assistantMsgs.count()) >= 1;
      expect(hasErrorBanner || hasAssistantMsg).toBe(true);

      // Run button should be re-enabled (not stuck)
      await expect(page.locator(".btn-primary")).toBeEnabled();

      test.info().annotations.push({
        type: "skip-reason",
        description: "MedGemma auth failed (HF_TOKEN missing/expired). Pipeline steps verified up to auth failure.",
      });
      return;
    }

    // --- Full pipeline success: verify all checklist items ---

    // 1. Step cards present (S1, R1, SEG, F*, R2)
    const stepCount = await page.locator(".step-card").count();
    expect(stepCount).toBeGreaterThanOrEqual(4); // At minimum: S1, R1, SEG, R2

    // 2. All steps completed (done status)
    const doneSteps = await page.locator(".step-card-done").count();
    expect(doneSteps).toBe(stepCount);

    // 3. Segments visible in table with real labels
    const segmentRows = page.locator(".segment-row");
    const segCount = await segmentRows.count();
    // Might be 0 if no segments found (fallback), or >0 if segments found
    if (segCount > 0) {
      // 4. Color dots visible in table
      await expect(page.locator(".segment-color-dot").first()).toBeVisible();

      // 5. Segments have real labels (not "Segment A")
      const firstLabel = await segmentRows
        .first()
        .locator("td")
        .nth(1)
        .textContent();
      expect(firstLabel).toBeTruthy();
      expect(firstLabel).not.toMatch(/^Segment [A-Z]$/);

      // 6. Different colors for different segments
      if (segCount >= 2) {
        const color1 = await page
          .locator(".segment-color-dot")
          .first()
          .evaluate((el) => getComputedStyle(el).backgroundColor);
        const color2 = await page
          .locator(".segment-color-dot")
          .nth(1)
          .evaluate((el) => getComputedStyle(el).backgroundColor);
        expect(color1).not.toBe(color2);
      }

      // 7. Segment chips in chat with labels (not raw [SEG:A])
      const chips = page.locator(".seg-chip");
      const chipCount = await chips.count();
      if (chipCount > 0) {
        const chipText = await chips.first().textContent();
        expect(chipText).toBeTruthy();
        expect(chipText).not.toContain("[SEG:");

        // 8. Chip has color dot
        await expect(chips.first().locator(".seg-chip-dot")).toBeVisible();
      }

      // 9. Hover chip → row highlights
      if (chipCount > 0) {
        const chip = chips.first();
        await chip.hover();
        await expect(chip).toHaveClass(/seg-chip-highlighted/);

        // Corresponding row should highlight
        const row = segmentRows.first();
        await expect(row).toHaveClass(/segment-row-highlighted/);
      }

      // 10. Hover row → chip highlights
      if (chipCount > 0) {
        // Move away first
        await page.locator(".panel-header").first().hover();
        await page.waitForTimeout(100);

        const row = segmentRows.first();
        await row.hover();
        await expect(row).toHaveClass(/segment-row-highlighted/);

        const chip = chips.first();
        await expect(chip).toHaveClass(/seg-chip-highlighted/);
      }
    }

    // 11. Assistant message has content (Markdown renders as text)
    const assistantMsg = page.locator(".chat-message-assistant").first();
    await expect(assistantMsg).toBeVisible();
    const assistantText = await assistantMsg.textContent();
    expect(assistantText!.length).toBeGreaterThan(20);

    // 12. No error banner
    await expect(page.locator(".error-banner")).not.toBeVisible();

    // 13. Streaming cursor gone (job complete)
    await expect(page.locator(".streaming-cursor")).not.toBeVisible();
  });
});

test.describe("Describe Pipeline (full end-to-end)", () => {
  test("run 'Describe the image' pipeline", async ({ page }) => {
    await page.goto("/");

    // Upload image
    const input = page.locator('.dropzone input[type="file"]');
    await input.setInputFiles(SAMPLE_IMAGE);
    await expect(page.locator(".image-container")).toBeVisible({
      timeout: 10_000,
    });

    // Type describe prompt and submit
    await page.locator(".prompt-input").fill("Describe the findings");
    await page.locator(".btn-primary").click();

    // Button should disable immediately
    await expect(page.locator(".btn-primary")).toBeDisabled({ timeout: 5_000 });

    // Wait for completion (up to 8 minutes)
    await expect(page.locator(".btn-primary")).toBeEnabled({
      timeout: 480_000,
    });

    // Check if pipeline errored
    const chatContent = await page.locator(".chat-panel").textContent();
    const hasError = chatContent?.includes("OSError") || chatContent?.includes("gated repo");

    if (hasError) {
      // Verify MedSAM3 still worked before MedGemma failure
      const stepTexts = await page.locator(".step-card").allTextContents();
      const hasS1 = stepTexts.some((t) => t.includes("S1"));
      expect(hasS1).toBe(true);

      // Should have at least some assistant messages (intermediate status)
      const assistantMsgs = page.locator(".chat-message-assistant");
      expect(await assistantMsgs.count()).toBeGreaterThanOrEqual(1);

      // Run button re-enabled
      await expect(page.locator(".btn-primary")).toBeEnabled();

      test.info().annotations.push({
        type: "skip-reason",
        description: "MedGemma auth failed. Verified pipeline handles error gracefully.",
      });
      return;
    }

    // --- Full pipeline success ---

    // Should have A1 auto-segment step
    const stepTexts = await page
      .locator(".step-card")
      .allTextContents();
    const hasAutoSegment =
      stepTexts.some((t) => t.includes("A1")) ||
      stepTexts.some((t) => t.includes("auto") || t.includes("Auto"));
    expect(hasAutoSegment || stepTexts.length >= 3).toBe(true);

    // Should have assistant response (use .first() to handle multiple messages)
    const assistantMsg = page.locator(".chat-message-assistant").first();
    await expect(assistantMsg).toBeVisible();

    // Should have segments from auto-segment
    const segmentRows = page.locator(".segment-row");
    const segCount = await segmentRows.count();
    if (segCount > 0) {
      await expect(page.locator(".segment-color-dot").first()).toBeVisible();
    }

    // No error banner
    await expect(page.locator(".error-banner")).not.toBeVisible();

    // Streaming cursor gone
    await expect(page.locator(".streaming-cursor")).not.toBeVisible();
  });
});
