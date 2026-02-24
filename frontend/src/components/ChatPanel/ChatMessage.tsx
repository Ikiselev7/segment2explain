import type { ChatMessage as ChatMessageType } from "../../types/pipeline";
import { MarkdownWithChips } from "./MarkdownWithChips";

interface Props {
  message: ChatMessageType;
}

export function ChatMessage({ message }: Props) {
  const isUser = message.role === "user";
  const isReasoning = message.reasoning === true;

  return (
    <div
      className={`chat-message ${
        isUser ? "chat-message-user" : "chat-message-assistant"
      }${isReasoning ? " chat-message-reasoning" : ""}`}
    >
      {isUser ? message.content : <MarkdownWithChips content={message.content} />}
    </div>
  );
}
