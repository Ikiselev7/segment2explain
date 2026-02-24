import { test, expect } from "@playwright/test";
import { MockBackend } from "../fixtures/mock-ws";
import {
  pipelineMessages,
  multiSegmentPipelineMessages,
  errorPipelineMessages,
  segmentRemovalMessages,
} from "../fixtures/sample-messages";

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

test.describe("Targeted Pipeline", () => {
  let mock: MockBackend;

  test.beforeEach(async ({ page }) => {
    mock = new MockBackend(page);
    await setupWithImage(page, mock);
  });

  test("step cards appear during pipeline execution", async ({ page }) => {
    await page.locator(".prompt-input").fill("Where is the nodule?");
    await page.locator(".btn-primary").click();
    await mock.replayMessages(pipelineMessages);

    await expect(page.locator(".step-card")).toHaveCount(6);
    await expect(page.locator(".step-card").first()).toContainText("S1");
    await expect(page.locator(".step-card").last()).toContainText("R2");
  });

  test("step cards show correct status icons", async ({ page }) => {
    await page.locator(".prompt-input").fill("Where is the nodule?");
    await page.locator(".btn-primary").click();
    await mock.replayMessages(pipelineMessages);

    const doneSteps = page.locator(".step-card-done");
    await expect(doneSteps).toHaveCount(6);
  });

  test("streaming text appears and finalizes", async ({ page }) => {
    await page.locator(".prompt-input").fill("Where is the nodule?");
    await page.locator(".btn-primary").click();
    await mock.replayMessages(pipelineMessages);

    const assistant = page.locator(".chat-message-assistant:not(.chat-message-reasoning)");
    await expect(assistant).toBeVisible();
    await expect(assistant).toContainText("round nodular opacity");
  });

  test("segment chip appears in final answer", async ({ page }) => {
    await page.locator(".prompt-input").fill("Where is the nodule?");
    await page.locator(".btn-primary").click();
    await mock.replayMessages(pipelineMessages);

    await expect(page.locator(".seg-chip")).toBeVisible();
    await expect(page.locator(".seg-chip")).toContainText("Nodule");
  });

  test("segments table shows segments", async ({ page }) => {
    await page.locator(".prompt-input").fill("Where is the nodule?");
    await page.locator(".btn-primary").click();
    await mock.replayMessages(pipelineMessages);

    await expect(page.locator(".segments-list")).toBeVisible();
    await expect(page.locator(".segment-row")).toHaveCount(1);
    await expect(page.locator(".segment-row").first()).toContainText("Nodule");
  });

  test("segment canvas renders filled overlays", async ({ page }) => {
    await page.locator(".prompt-input").fill("Where is the nodule?");
    await page.locator(".btn-primary").click();
    await mock.replayMessages(pipelineMessages);

    // Wait for the canvas to be present
    const canvas = page.locator(".image-container canvas");
    await expect(canvas).toBeVisible();

    // Verify the canvas has non-zero dimensions and contains drawn pixels
    const hasContent = await page.evaluate(() => {
      const cvs = document.querySelector(
        ".image-container canvas",
      ) as HTMLCanvasElement | null;
      if (!cvs || cvs.width === 0 || cvs.height === 0) return false;
      const ctx = cvs.getContext("2d");
      if (!ctx) return false;
      const data = ctx.getImageData(0, 0, cvs.width, cvs.height).data;
      // Check for any non-transparent pixel (alpha > 0)
      for (let i = 3; i < data.length; i += 4) {
        if (data[i] > 0) return true;
      }
      return false;
    });

    expect(hasContent).toBe(true);
  });
});

test.describe("Describe Pipeline", () => {
  let mock: MockBackend;

  test.beforeEach(async ({ page }) => {
    mock = new MockBackend(page);
    await setupWithImage(page, mock);
  });

  test("describe pipeline shows multiple segments", async ({ page }) => {
    await page.locator(".prompt-input").fill("Describe the image");
    await page.locator(".btn-primary").click();
    await mock.replayMessages(multiSegmentPipelineMessages);

    await expect(page.locator(".segment-row")).toHaveCount(2);
    await expect(page.locator(".segment-row").first()).toContainText("Heart");
    await expect(page.locator(".segment-row").last()).toContainText(
      "Left lung",
    );
  });

  test("describe pipeline shows multiple segment chips", async ({ page }) => {
    await page.locator(".prompt-input").fill("Describe the image");
    await page.locator(".btn-primary").click();
    await mock.replayMessages(multiSegmentPipelineMessages);

    await expect(page.locator(".seg-chip")).toHaveCount(2);
  });
});

test.describe("Error Handling", () => {
  let mock: MockBackend;

  test.beforeEach(async ({ page }) => {
    mock = new MockBackend(page);
    await setupWithImage(page, mock);
  });

  test("shows error banner on pipeline failure", async ({ page }) => {
    await page.locator(".prompt-input").fill("Where is the nodule?");
    await page.locator(".btn-primary").click();
    await mock.replayMessages(errorPipelineMessages);

    await expect(page.locator(".error-banner")).toBeVisible();
    await expect(page.locator(".error-banner")).toContainText(
      "CUDA out of memory",
    );
  });

  test("re-enables prompt input after error", async ({ page }) => {
    await page.locator(".prompt-input").fill("Where is the nodule?");
    await page.locator(".btn-primary").click();
    await mock.replayMessages(errorPipelineMessages);

    await expect(page.locator(".prompt-input")).toBeEnabled();
  });
});

test.describe("Segment Lifecycle", () => {
  let mock: MockBackend;

  test.beforeEach(async ({ page }) => {
    mock = new MockBackend(page);
    await setupWithImage(page, mock);
  });

  test("segment added then removed", async ({ page }) => {
    await page.locator(".prompt-input").fill("Where is the nodule?");
    await page.locator(".btn-primary").click();
    await mock.replayMessages(segmentRemovalMessages);

    await expect(page.locator(".segment-row")).toHaveCount(0);
  });
});
