import { test, expect } from "@playwright/test";
import { MockBackend } from "../fixtures/mock-ws";
import { pipelineMessages } from "../fixtures/sample-messages";

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

test.describe("Chat Interactions", () => {
  let mock: MockBackend;

  test.beforeEach(async ({ page }) => {
    mock = new MockBackend(page);
    await setupWithImage(page, mock);
  });

  test("Enter key submits prompt", async ({ page }) => {
    await page.locator(".prompt-input").fill("Where is the nodule?");
    await page.locator(".prompt-input").press("Enter");

    // Check that a WS message was sent
    const sent = await mock.getSentMessages();
    expect(sent.length).toBeGreaterThanOrEqual(1);
    const parsed = JSON.parse(sent[sent.length - 1]);
    expect(parsed.type).toBe("start_job");
    expect(parsed.prompt).toBe("Where is the nodule?");
  });

  test("Shift+Enter does not submit (allows newline)", async ({ page }) => {
    await page.locator(".prompt-input").fill("Line 1");
    await page.locator(".prompt-input").press("Shift+Enter");
    await page.locator(".prompt-input").type("Line 2");

    const value = await page.locator(".prompt-input").inputValue();
    expect(value).toContain("Line 1");
    expect(value).toContain("Line 2");

    const sent = await mock.getSentMessages();
    expect(sent.length).toBe(0);
  });

  test("Run button is disabled during pipeline execution", async ({
    page,
  }) => {
    await page.locator(".prompt-input").fill("Where is the nodule?");
    await page.locator(".btn-primary").click();

    await mock.replayMessages([{ type: "job_started", job_id: "test-1" }]);

    await expect(page.locator(".btn-primary")).toBeDisabled();
  });

  test("Run button re-enables after pipeline completes", async ({ page }) => {
    await page.locator(".prompt-input").fill("Where is the nodule?");
    await page.locator(".btn-primary").click();
    await mock.replayMessages(pipelineMessages);

    await expect(page.locator(".btn-primary")).toBeEnabled();
  });

  test("Clear button resets chat", async ({ page }) => {
    await page.locator(".prompt-input").fill("Where is the nodule?");
    await page.locator(".btn-primary").click();
    await mock.replayMessages(pipelineMessages);

    await expect(page.locator(".step-card")).toHaveCount(6);

    await page.locator(".btn-secondary").click();

    await expect(page.locator(".step-card")).toHaveCount(0);
    await expect(page.locator(".chat-message-assistant")).toHaveCount(0);
    await expect(page.locator(".empty-state")).toBeVisible();
  });

  test("streaming text shows cursor animation", async ({ page }) => {
    await page.locator(".prompt-input").fill("Where is the nodule?");
    await page.locator(".btn-primary").click();

    await mock.replayMessages([
      { type: "job_started", job_id: "test-1" },
      { type: "chat_delta", text: "Analyzing the image..." },
    ]);

    await expect(page.locator(".streaming-cursor")).toBeVisible();
    await expect(page.locator(".chat-message-assistant")).toContainText(
      "Analyzing the image...",
    );
  });

  test("assistant message contains formatted text with chips", async ({
    page,
  }) => {
    await page.locator(".prompt-input").fill("Where is the nodule?");
    await page.locator(".btn-primary").click();
    await mock.replayMessages(pipelineMessages);

    const assistant = page.locator(".chat-message-assistant:not(.chat-message-reasoning)");
    await expect(assistant).toContainText("round nodular opacity");
    await expect(assistant.locator(".seg-chip")).toContainText("Nodule");
  });

  test("auto-scrolls on new messages", async ({ page }) => {
    await page.locator(".prompt-input").fill("Where is the nodule?");
    await page.locator(".btn-primary").click();
    await mock.replayMessages(pipelineMessages);

    const scrollTop = await page
      .locator(".message-list")
      .evaluate((el) => el.scrollTop);
    const scrollHeight = await page
      .locator(".message-list")
      .evaluate((el) => el.scrollHeight);
    const clientHeight = await page
      .locator(".message-list")
      .evaluate((el) => el.clientHeight);

    if (scrollHeight > clientHeight) {
      expect(scrollTop + clientHeight).toBeGreaterThanOrEqual(
        scrollHeight - 50,
      );
    }
  });
});
