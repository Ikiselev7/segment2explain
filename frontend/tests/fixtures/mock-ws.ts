/**
 * Mock WebSocket + HTTP server for Playwright tests.
 *
 * Uses page.addInitScript() to replace the global WebSocket constructor
 * BEFORE any application JavaScript runs. This ensures the React app's
 * useWebSocket hook connects to a MockWebSocket instead of a real one.
 */

import { type Page } from "@playwright/test";
import type { ServerMessage } from "../../src/types/messages";

const INTER_MESSAGE_DELAY = 30; // ms between messages

export class MockBackend {
  private page: Page;

  constructor(page: Page) {
    this.page = page;
  }

  /**
   * Set up HTTP route mocks + WebSocket mock.
   * Must be called BEFORE page.goto().
   */
  async setup() {
    // === HTTP Route Mocks ===

    // Mock health endpoint
    await this.page.route("**/api/health", (route) =>
      route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({
          status: "ok",
          medgemma_loaded: false,
          medsam3_loaded: false,
        }),
      }),
    );

    // Mock image upload
    await this.page.route("**/api/upload-image", (route) => {
      const imageId = `img-${Date.now()}`;
      route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({
          image_id: imageId,
          width: 512,
          height: 512,
          url: `/api/images/${imageId}`,
        }),
      });
    });

    // Mock image serving — return a 100x100 gray PNG (large enough for canvas rendering)
    await this.page.route("**/api/images/**", (route) => {
      const pngBytes = Buffer.from(
        "iVBORw0KGgoAAAANSUhEUgAAAGQAAABkCAAAAABVicqIAAAAY0lEQVR4nO3QMQ0AIADAMEA50tHQAw6yGliyucd960FjFCHtIu0i7SLtIu0i7SLtIu0i7SLtIu0i7SLtIu0i7SLtIu0i7SLtIu0i7SLtIu0i7SLtIu0i7SLtIu0i7SLtIv/sOkxRAUhRxwhhAAAAAElFTkSuQmCC",
        "base64",
      );
      route.fulfill({
        status: 200,
        contentType: "image/png",
        body: pngBytes,
      });
    });

    // === WebSocket Mock (injected before page JS runs) ===
    await this.page.addInitScript(() => {
      class MockWebSocket extends EventTarget {
        static CONNECTING = 0;
        static OPEN = 1;
        static CLOSING = 2;
        static CLOSED = 3;

        readyState = MockWebSocket.CONNECTING;
        url: string;
        protocol = "";
        bufferedAmount = 0;
        extensions = "";
        binaryType: BinaryType = "blob";

        onopen: ((this: WebSocket, ev: Event) => unknown) | null = null;
        onmessage: ((this: WebSocket, ev: MessageEvent) => unknown) | null =
          null;
        onclose: ((this: WebSocket, ev: CloseEvent) => unknown) | null = null;
        onerror: ((this: WebSocket, ev: Event) => unknown) | null = null;

        constructor(url: string | URL, _protocols?: string | string[]) {
          super();
          this.url = typeof url === "string" ? url : url.toString();

          // Store reference for test access
          const win = window as Record<string, unknown>;
          win.__mockWs = this;
          win.__mockWsSent = win.__mockWsSent || [];

          // Simulate async connection
          setTimeout(() => {
            this.readyState = MockWebSocket.OPEN;
            const evt = new Event("open");
            this.onopen?.call(this as unknown as WebSocket, evt);
            this.dispatchEvent(evt);
          }, 10);
        }

        send(data: string | ArrayBufferLike | Blob | ArrayBufferView) {
          const win = window as Record<string, unknown>;
          (win.__mockWsSent as string[]).push(data as string);
          window.dispatchEvent(
            new CustomEvent("mock-ws-send", { detail: data }),
          );
        }

        close(_code?: number, _reason?: string) {
          this.readyState = MockWebSocket.CLOSED;
          const evt = new CloseEvent("close", { code: 1000 });
          this.onclose?.call(this as unknown as WebSocket, evt);
          this.dispatchEvent(evt);
        }

        /** Test helper: simulate receiving a message from "server" */
        _receive(data: string) {
          const evt = new MessageEvent("message", { data });
          this.onmessage?.call(this as unknown as WebSocket, evt);
          this.dispatchEvent(evt);
        }
      }

      // Assign static constants matching WebSocket
      Object.defineProperty(MockWebSocket, "CONNECTING", { value: 0 });
      Object.defineProperty(MockWebSocket, "OPEN", { value: 1 });
      Object.defineProperty(MockWebSocket, "CLOSING", { value: 2 });
      Object.defineProperty(MockWebSocket, "CLOSED", { value: 3 });

      // Replace global WebSocket
      (window as Record<string, unknown>).WebSocket = MockWebSocket;
    });
  }

  /** Navigate to the app. Must be called after setup(). */
  async goto(path = "/") {
    await this.page.goto(path);
    // Wait for the mock WS to be connected
    await this.page.waitForFunction(
      () => {
        const ws = (window as Record<string, unknown>).__mockWs as
          | { readyState: number }
          | undefined;
        return ws && ws.readyState === 1; // OPEN
      },
      { timeout: 5000 },
    );
  }

  /** Send a sequence of mock WS messages to the app */
  async replayMessages(
    messages: ServerMessage[],
    delayMs = INTER_MESSAGE_DELAY,
  ) {
    for (const msg of messages) {
      await this.page.evaluate((data) => {
        const ws = (window as Record<string, unknown>).__mockWs as
          | { _receive: (data: string) => void }
          | undefined;
        ws?._receive(JSON.stringify(data));
      }, msg);
      if (delayMs > 0) {
        await this.page.waitForTimeout(delayMs);
      }
    }
  }

  /** Get messages sent by the client over the mock WS */
  async getSentMessages(): Promise<string[]> {
    return this.page.evaluate(
      () =>
        ((window as Record<string, unknown>).__mockWsSent || []) as string[],
    );
  }
}
