import { useJobStore } from "../../stores/jobStore";
import { MarkdownWithChips } from "./MarkdownWithChips";

export function StreamingText() {
  const streamingText = useJobStore((s) => s.streamingText);

  if (!streamingText) return null;

  return (
    <div className="chat-message chat-message-assistant">
      <MarkdownWithChips content={streamingText} />
      <span className="streaming-cursor" />
    </div>
  );
}
