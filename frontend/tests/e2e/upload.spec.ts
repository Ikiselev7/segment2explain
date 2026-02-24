import { test, expect } from "@playwright/test";
import { MockBackend } from "../fixtures/mock-ws";

test.describe("Image Upload", () => {
  let mock: MockBackend;

  test.beforeEach(async ({ page }) => {
    mock = new MockBackend(page);
    await mock.setup();
    await mock.goto();
  });

  test("shows dropzone when no image is uploaded", async ({ page }) => {
    await expect(page.locator(".dropzone")).toBeVisible();
    await expect(page.locator(".dropzone-text")).toContainText(
      "Drag & drop a chest X-ray",
    );
  });

  test("upload image via file input shows image panel", async ({ page }) => {
    const input = page.locator('.dropzone input[type="file"]');
    await input.setInputFiles({
      name: "test.png",
      mimeType: "image/png",
      buffer: Buffer.from(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg==",
        "base64",
      ),
    });

    await expect(page.locator(".image-container")).toBeVisible({
      timeout: 5000,
    });
    await expect(page.locator(".dropzone")).not.toBeVisible();
    await expect(page.locator(".image-container img")).toBeVisible();
  });

  test("prompt input is disabled before image upload", async ({ page }) => {
    await expect(page.locator(".prompt-input")).toBeDisabled();
    await expect(page.locator(".prompt-input")).toHaveAttribute(
      "placeholder",
      "Upload an image first",
    );
  });

  test("prompt input is enabled after image upload", async ({ page }) => {
    const input = page.locator('.dropzone input[type="file"]');
    await input.setInputFiles({
      name: "test.png",
      mimeType: "image/png",
      buffer: Buffer.from(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg==",
        "base64",
      ),
    });

    await expect(page.locator(".prompt-input")).toBeEnabled({ timeout: 5000 });
    await expect(page.locator(".prompt-input")).toHaveAttribute(
      "placeholder",
      /Ask about the image/,
    );
  });

  test("shows empty state message in chat", async ({ page }) => {
    await expect(page.locator(".empty-state")).toContainText(
      "Upload an image and ask a question to start",
    );
  });

  test("panel headers are visible", async ({ page }) => {
    await expect(page.locator(".panel-header").first()).toContainText("Image");
    await expect(page.locator(".panel-header").last()).toContainText("Chat");
  });
});
