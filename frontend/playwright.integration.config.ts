/**
 * Playwright config for integration tests.
 *
 * Unlike the unit tests (playwright.config.ts) which mock the backend,
 * these tests run against the REAL FastAPI + React stack.
 * Both servers must be running before executing these tests.
 */

import { defineConfig, devices } from "@playwright/test";

export default defineConfig({
  testDir: "./tests/e2e-integration",
  fullyParallel: false, // Sequential — tests share backend state
  retries: 0,
  workers: 1,
  reporter: "html",
  timeout: 600_000, // 10 min per test (model loading + pipeline)
  use: {
    baseURL: "http://localhost:5173",
    trace: "on-first-retry",
  },
  projects: [
    {
      name: "chromium",
      use: { ...devices["Desktop Chrome"] },
    },
  ],
  // Don't start servers — they must already be running
  // (use /start-app command or run manually)
  expect: {
    timeout: 30_000,
  },
});
