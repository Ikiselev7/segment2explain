import { useJobStore } from "../../stores/jobStore";
import { ImageUpload } from "./ImageUpload";
import { SegmentCanvas } from "./SegmentCanvas";
import { SegmentList } from "./SegmentList";

export function ImagePanel() {
  const imageUrl = useJobStore((s) => s.imageUrl);

  return (
    <div className="panel image-panel">
      <div className="panel-header">Image</div>
      {imageUrl ? <SegmentCanvas /> : <ImageUpload />}
      <SegmentList />
    </div>
  );
}
