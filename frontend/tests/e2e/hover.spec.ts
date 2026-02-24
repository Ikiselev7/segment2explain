import { test, expect } from "@playwright/test";
import { MockBackend } from "../fixtures/mock-ws";
import { pipelineMessages } from "../fixtures/sample-messages";

/** Helper: set up, navigate, upload image, run targeted pipeline */
async function setupWithPipeline(
  page: import("@playwright/test").Page,
  mock: MockBackend,
) {
  await mock.setup();
  await mock.goto();

  // Upload image
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

  // Run pipeline
  await page.locator(".prompt-input").fill("Where is the nodule?");
  await page.locator(".btn-primary").click();
  await mock.replayMessages(pipelineMessages);

  await expect(page.locator(".chat-message-assistant:not(.chat-message-reasoning)")).toBeVisible();
}

test.describe("Hover Cross-Referencing", () => {
  let mock: MockBackend;

  test.beforeEach(async ({ page }) => {
    mock = new MockBackend(page);
    await setupWithPipeline(page, mock);
  });

  test("hovering segment chip highlights it", async ({ page }) => {
    const chip = page.locator(".seg-chip").first();
    await expect(chip).toBeVisible();

    await chip.hover();
    await expect(chip).toHaveClass(/seg-chip-highlighted/);
  });

  test("leaving segment chip removes highlight", async ({ page }) => {
    const chip = page.locator(".seg-chip").first();
    await chip.hover();
    await expect(chip).toHaveClass(/seg-chip-highlighted/);

    await page.locator(".panel-header").first().hover();
    await expect(chip).not.toHaveClass(/seg-chip-highlighted/);
  });

  test("hovering segment row in table highlights row", async ({ page }) => {
    const row = page.locator(".segment-row").first();
    await expect(row).toBeVisible();

    await row.hover();
    await expect(row).toHaveClass(/segment-row-highlighted/);
  });

  test("hovering segment row highlights corresponding chip in chat", async ({
    page,
  }) => {
    const row = page.locator(".segment-row").first();
    await row.hover();

    const chip = page.locator(".seg-chip").first();
    await expect(chip).toHaveClass(/seg-chip-highlighted/);
  });

  test("hovering chip highlights corresponding row in table", async ({
    page,
  }) => {
    const chip = page.locator(".seg-chip").first();
    await chip.hover();

    const row = page.locator(".segment-row").first();
    await expect(row).toHaveClass(/segment-row-highlighted/);
  });

  test("segment color dot is visible in table", async ({ page }) => {
    const dot = page.locator(".segment-color-dot").first();
    await expect(dot).toBeVisible();
  });
});
