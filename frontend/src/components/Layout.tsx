import type { ClientMessage } from "../types/messages";
import { ChatPanel } from "./ChatPanel/ChatPanel";
import { ImagePanel } from "./ImagePanel/ImagePanel";

interface Props {
  onSend: (msg: ClientMessage) => void;
}

export function Layout({ onSend }: Props) {
  return (
    <div className="app-layout">
      <ImagePanel />
      <ChatPanel onSend={onSend} />
    </div>
  );
}
