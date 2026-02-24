import type { Step } from "../../types/pipeline";

interface Props {
  step: Step;
}

const STATUS_ICONS: Record<string, string> = {
  done: "\u2713",
  running: "\u25B6",
  failed: "\u2717",
  queued: "\u25CB",
};

export function StepCard({ step }: Props) {
  const icon = STATUS_ICONS[step.status] || "\u25CB";

  return (
    <div className={`step-card step-card-${step.status}`}>
      <div className="step-card-name">
        <span className="step-card-icon">{icon}</span>
        {step.id}: {step.name}
      </div>
      {step.detail && <div className="step-card-detail">{step.detail}</div>}
    </div>
  );
}
