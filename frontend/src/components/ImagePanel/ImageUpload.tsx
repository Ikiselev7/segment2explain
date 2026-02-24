import { useCallback } from "react";
import { useDropzone } from "react-dropzone";
import { useJobStore } from "../../stores/jobStore";

const API_BASE = "";

export function ImageUpload() {
  const setImage = useJobStore((s) => s.setImage);

  const onDrop = useCallback(
    async (files: File[]) => {
      if (files.length === 0) return;
      const file = files[0];
      const formData = new FormData();
      formData.append("file", file);

      const resp = await fetch(`${API_BASE}/api/upload-image`, {
        method: "POST",
        body: formData,
      });

      if (!resp.ok) {
        console.error("Upload failed:", resp.statusText);
        return;
      }

      const data = await resp.json();
      setImage(data.image_id, `${API_BASE}${data.url}`, data.width, data.height);
    },
    [setImage]
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { "image/*": [".png", ".jpg", ".jpeg", ".dcm"] },
    maxFiles: 1,
  });

  return (
    <div
      {...getRootProps()}
      className={`dropzone ${isDragActive ? "dropzone-active" : ""}`}
    >
      <input {...getInputProps()} />
      <div className="dropzone-text">
        {isDragActive
          ? "Drop image here…"
          : "Drag & drop a chest X-ray, or click to select"}
      </div>
      <div className="dropzone-hint">PNG, JPEG supported</div>
    </div>
  );
}
