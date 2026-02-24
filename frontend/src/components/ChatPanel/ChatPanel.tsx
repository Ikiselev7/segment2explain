import type { ClientMessage } from "../../types/messages";
import { MessageList } from "./MessageList";
import { PromptInput } from "./PromptInput";

interface Props {
  onSend: (msg: ClientMessage) => void;
}

export function ChatPanel({ onSend }: Props) {
  return (
    <div className="panel chat-panel">
      <div className="panel-header">Chat</div>
      <MessageList />
      <PromptInput onSend={onSend} />
    </div>
  );
}
