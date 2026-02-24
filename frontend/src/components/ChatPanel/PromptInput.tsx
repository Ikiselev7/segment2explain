import { useState } from "react";
import { useJobStore } from "../../stores/jobStore";
import type { ClientMessage } from "../../types/messages";

interface Props {
  onSend: (msg: ClientMessage) => void;
}

export function PromptInput({ onSend }: Props) {
  const [prompt, setPrompt] = useState("");
  const imageId = useJobStore((s) => s.imageId);
  const isRunning = useJobStore((s) => s.isRunning);
  const pipelineMode = useJobStore((s) => s.pipelineMode);
  const beginJob = useJobStore((s) => s.beginJob);
  const clearJob = useJobStore((s) => s.clearJob);
  const setPipelineMode = useJobStore((s) => s.setPipelineMode);

  const canSubmit = !!imageId && !!prompt.trim() && !isRunning;

  const handleSubmit = () => {
    if (!canSubmit) return;
    beginJob();
    onSend({
      type: "start_job",
      image_id: imageId!,
      prompt: prompt.trim(),
      mode: pipelineMode,
    });
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  return (
    <div className="prompt-bar">
      <textarea
        className="prompt-input"
        value={prompt}
        onChange={(e) => setPrompt(e.target.value)}
        onKeyDown={handleKeyDown}
        placeholder={
          imageId
            ? "Ask about the image\u2026 (e.g., Where is the nodule?)"
            : "Upload an image first"
        }
        rows={2}
        disabled={!imageId || isRunning}
      />
      <div className="prompt-controls">
        <div className="mode-toggle">
          <button
            className={`mode-btn ${pipelineMode === "sequential" ? "mode-btn-active" : ""}`}
            onClick={() => setPipelineMode("sequential")}
            disabled={isRunning}
            title="Evidence-grounded analysis: segments first, then answer"
          >
            Evidence
          </button>
          <button
            className={`mode-btn ${pipelineMode === "parallel" ? "mode-btn-active" : ""}`}
            onClick={() => setPipelineMode("parallel")}
            disabled={isRunning}
            title="Quick answer first, then segments appear progressively"
          >
            Quick
          </button>
        </div>
        <div className="prompt-actions">
          <button
            className="btn btn-primary"
            onClick={handleSubmit}
            disabled={!canSubmit}
          >
            Run
          </button>
          <button
            className="btn btn-secondary"
            onClick={clearJob}
            disabled={isRunning}
          >
            Clear
          </button>
        </div>
      </div>
    </div>
  );
}
