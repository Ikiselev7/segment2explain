import type React from "react";
import { useEffect, useRef } from "react";
import { useJobStore } from "../../stores/jobStore";
import { ChatMessage } from "./ChatMessage";
import { StepCard } from "./StepCard";
import { StreamingText } from "./StreamingText";

export function MessageList() {
  const chatMessages = useJobStore((s) => s.chatMessages);
  const steps = useJobStore((s) => s.steps);
  const isRunning = useJobStore((s) => s.isRunning);
  const error = useJobStore((s) => s.error);
  const endRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom on new content
  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [chatMessages, steps, isRunning]);

  // Interleave chat messages and step cards
  // Steps appear in order, chat messages appear in order
  // We show steps first (as they happen), then the final answer
  const items: React.ReactNode[] = [];

  // Add steps
  for (const step of steps) {
    items.push(<StepCard key={`step-${step.id}`} step={step} />);
  }

  // Add chat messages
  for (let i = 0; i < chatMessages.length; i++) {
    items.push(
      <ChatMessage key={`msg-${i}`} message={chatMessages[i]} />
    );
  }

  return (
    <div className="message-list">
      {items.length === 0 && !isRunning && (
        <div className="empty-state">
          Upload an image and ask a question to start.
        </div>
      )}
      {items}
      {isRunning && <StreamingText />}
      {error && <div className="error-banner">{error}</div>}
      <div ref={endRef} />
    </div>
  );
}
