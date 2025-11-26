"use client";

import { CameraDetection } from "@/components/camera-detection";

export default function Home() {
  return (
    <div className="min-h-screen flex flex-col items-center justify-center bg-background text-foreground p-4">
      <CameraDetection />
    </div>
  );
}
