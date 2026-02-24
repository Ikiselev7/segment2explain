import { test, expect } from "@playwright/test";
import { MockBackend } from "../fixtures/mock-ws";
import { parallelPipelineMessages } from "../fixtures/sample-messages";

/** Helper: set up mock, navigate, upload image */
async function setupWithImage(
  page: import("@playwright/test").Page,
  mock: MockBackend,
) {
  await mock.setup();
  await mock.goto();

  const input = page.locator('.dropzone input[type="file"]');
  await input.setInputFiles({
    name: "test.png",
    mimeType: "image/png",
    buffer: Buffer.from(
      "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg==",
      "base64",
    ),
  });

  await expect(page.locator(".image-container")).toBeVisible({ timeout: 5000 });
}

test.describe("Parallel Pipeline (Quick mode)", () => {
  let mock: MockBackend;

  test.beforeEach(async ({ page }) => {
    mock = new MockBackend(page);
    await setupWithImage(page, mock);
  });

  test("mode toggle buttons are visible", async ({ page }) => {
    const evidenceBtn = page.locator(".mode-btn", { hasText: "Evidence" });
    const quickBtn = page.locator(".mode-btn", { hasText: "Quick" });
    await expect(evidenceBtn).toBeVisible();
    await expect(quickBtn).toBeVisible();
  });

  test("Evidence mode is selected by default", async ({ page }) => {
    const evidenceBtn = page.locator(".mode-btn-active");
    await expect(evidenceBtn).toHaveText("Evidence");
  });

  test("clicking Quick switches the active mode", async ({ page }) => {
    const quickBtn = page.locator(".mode-btn", { hasText: "Quick" });
    await quickBtn.click();

    await expect(quickBtn).toHaveClass(/mode-btn-active/);
    const evidenceBtn = page.locator(".mode-btn", { hasText: "Evidence" });
    await expect(evidenceBtn).not.toHaveClass(/mode-btn-active/);
  });

  test("Quick mode sends mode=parallel in start_job", async ({ page }) => {
    // Switch to Quick mode
    await page.locator(".mode-btn", { hasText: "Quick" }).click();

    await page.locator(".prompt-input").fill("Is there cardiomegaly?");
    await page.locator(".btn-primary").click();

    const sent = await mock.getSentMessages();
    const startMsg = sent.find((s) => {
      try {
        return JSON.parse(s).type === "start_job";
      } catch {
        return false;
      }
    });

    expect(startMsg).toBeDefined();
    const parsed = JSON.parse(startMsg!);
    expect(parsed.mode).toBe("parallel");
  });

  test("Evidence mode sends mode=sequential in start_job", async ({
    page,
  }) => {
    // Evidence is default — just submit
    await page.locator(".prompt-input").fill("Where is the nodule?");
    await page.locator(".btn-primary").click();

    const sent = await mock.getSentMessages();
    const startMsg = sent.find((s) => {
      try {
        return JSON.parse(s).type === "start_job";
      } catch {
        return false;
      }
    });

    expect(startMsg).toBeDefined();
    const parsed = JSON.parse(startMsg!);
    expect(parsed.mode).toBe("sequential");
  });

  test("parallel pipeline shows ANSWER, SELECT, SEG, and LINK steps", async ({
    page,
  }) => {
    await page.locator(".mode-btn", { hasText: "Quick" }).click();
    await page.locator(".prompt-input").fill("Is there cardiomegaly?");
    await page.locator(".btn-primary").click();
    await mock.replayMessages(parallelPipelineMessages);

    // Should have S1, ANSWER, SELECT, SEG, LINK steps
    await expect(page.locator(".step-card")).toHaveCount(5);
    await expect(
      page.locator(".step-card", { hasText: "MedGemma: answer" }),
    ).toBeVisible();
    await expect(
      page.locator(".step-card", { hasText: "Select concepts" }),
    ).toBeVisible();
  });

  test("streaming answer text appears in chat", async ({ page }) => {
    await page.locator(".mode-btn", { hasText: "Quick" }).click();
    await page.locator(".prompt-input").fill("Is there cardiomegaly?");
    await page.locator(".btn-primary").click();
    await mock.replayMessages(parallelPipelineMessages);

    // After job_completed, streaming text is finalized into a chat message.
    // The streamed answer is the LAST non-reasoning assistant message.
    const assistant = page.locator(
      ".chat-message-assistant:not(.chat-message-reasoning)",
    );
    await expect(assistant.last()).toContainText("heart appears mildly enlarged");
  });

  test("segments appear after concept extraction", async ({ page }) => {
    await page.locator(".mode-btn", { hasText: "Quick" }).click();
    await page.locator(".prompt-input").fill("Is there cardiomegaly?");
    await page.locator(".btn-primary").click();
    await mock.replayMessages(parallelPipelineMessages);

    // Segments table should show 2 segments
    const segRows = page.locator(".segment-row");
    await expect(segRows).toHaveCount(2);
  });

  test("concept highlights appear in answer text after linking", async ({
    page,
  }) => {
    await page.locator(".mode-btn", { hasText: "Quick" }).click();
    await page.locator(".prompt-input").fill("Is there cardiomegaly?");
    await page.locator(".btn-primary").click();
    await mock.replayMessages(parallelPipelineMessages);

    // Concept highlights should be present (heart, lungs terms in answer)
    const highlights = page.locator(".concept-highlight");
    await expect(highlights.first()).toBeVisible({ timeout: 5000 });
    // At least "heart" and "cardiomegaly" should be highlighted
    const count = await highlights.count();
    expect(count).toBeGreaterThanOrEqual(2);
  });

  test("concept highlight has correct border color", async ({ page }) => {
    await page.locator(".mode-btn", { hasText: "Quick" }).click();
    await page.locator(".prompt-input").fill("Is there cardiomegaly?");
    await page.locator(".btn-primary").click();
    await mock.replayMessages(parallelPipelineMessages);

    const firstHighlight = page.locator(".concept-highlight").first();
    await expect(firstHighlight).toBeVisible({ timeout: 5000 });

    const borderBottom = await firstHighlight.evaluate(
      (el) => getComputedStyle(el).borderBottomStyle,
    );
    expect(borderBottom).toBe("solid");
  });

  test("hovering concept highlight activates hover state", async ({
    page,
  }) => {
    await page.locator(".mode-btn", { hasText: "Quick" }).click();
    await page.locator(".prompt-input").fill("Is there cardiomegaly?");
    await page.locator(".btn-primary").click();
    await mock.replayMessages(parallelPipelineMessages);

    const highlight = page.locator(".concept-highlight").first();
    await expect(highlight).toBeVisible({ timeout: 5000 });

    await highlight.hover();
    await expect(highlight).toHaveClass(/concept-highlight-active/);
  });

  test("mode toggle is disabled during running job", async ({ page }) => {
    await page.locator(".prompt-input").fill("test");
    await page.locator(".btn-primary").click();

    // After clicking Run, beginJob sets isRunning=true
    const modeButtons = page.locator(".mode-btn");
    await expect(modeButtons.first()).toBeDisabled();
    await expect(modeButtons.last()).toBeDisabled();
  });
});
