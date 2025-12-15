"use client";

import { useRef, useState } from "react";
import { Button } from "@/components/ui/button";

interface Detection {
  // Adjust based on your actual Roboflow response structure
  [key: string]: any;
}

export function CameraDetection() {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [outputImage, setOutputImage] = useState<string | null>(null);
  const [isDetecting, setIsDetecting] = useState(false);
  const [detections, setDetections] = useState<Detection | null>(null);
  const [error, setError] = useState<string>("");

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file && file.type.startsWith("image/")) {
      const reader = new FileReader();
      reader.onload = (e) => {
        const imageUrl = e.target?.result as string;
        setSelectedImage(imageUrl);
        setOutputImage(null);
        setDetections(null);
        setError("");
      };
      reader.readAsDataURL(file);
    } else {
      setError("Please select a valid image file.");
    }
  };

  const handleUploadClick = () => {
    fileInputRef.current?.click();
  };

  const runDetection = async () => {
  if (!selectedImage) {
    setError("Please upload an image first.");
    return;
  }

  setIsDetecting(true);
  setError("");

  try {
    // STEP 1: Check pipeline status
    const statusRes = await fetch("/api/detect");
    const status = await statusRes.json();

    // STEP 2: Initialize if needed
    if (!status.initialized) {
      const initRes = await fetch("/api/detect?mode=init&debug=true", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          imageBase64: selectedImage,
        }),
      });

      if (!initRes.ok) {
        throw new Error("Initialization failed");
      }

      const initResult = await initRes.json();
      console.log("Initialization result:", initResult);

      // Optional: show classical CV debug image
      if (initResult.debugImage) {
        setOutputImage(initResult.debugImage);
      }
    }

    // STEP 3: Run vehicle detection
    const detectRes = await fetch("/api/detect?mode=detect", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        imageBase64: selectedImage,
      }),
    });

    if (!detectRes.ok) {
      throw new Error("Detection failed");
    }

    const detectResult = await detectRes.json();
    console.log("Detection result:", detectResult);

    setDetections(detectResult);

  } catch (err) {
    console.error(err);
    setError("Detection error. Please try again.");
  } finally {
    setIsDetecting(false);
  }
};


  return (
    <div className="flex flex-col items-center gap-6 w-full max-w-4xl mx-auto p-4">
      <h1 className="text-4xl font-bold">Parking Lot Detector</h1>

      {error && (
        <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded">
          {error}
        </div>
      )}

      <input
        ref={fileInputRef}
        type="file"
        accept="image/*"
        onChange={handleFileSelect}
        className="hidden"
      />

      <div className="flex gap-4">
        <Button
          onClick={handleUploadClick}
          size="lg"
          className="bg-black hover:bg-gray-800 text-white font-semibold px-8 py-6 rounded-lg shadow-lg transition-all"
        >
          Upload Image
        </Button>
        {selectedImage && (
          <Button
            onClick={runDetection}
            size="lg"
            disabled={isDetecting}
            className="bg-black hover:bg-gray-800 text-white font-semibold px-8 py-6 rounded-lg shadow-lg transition-all disabled:bg-gray-600 disabled:cursor-not-allowed"
          >
            {isDetecting ? "Detecting..." : "Run Detection"}
          </Button>
        )}
      </div>

      {selectedImage && (
        <div className="w-full grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="flex flex-col gap-2">
            <h2 className="text-xl font-semibold">Input Image:</h2>
            <div className="relative w-full aspect-video bg-black rounded-lg overflow-hidden">
              <img
                src={selectedImage}
                alt="Selected"
                className="w-full h-full object-contain"
              />
            </div>
          </div>

          {outputImage && (
            <div className="flex flex-col gap-2">
              <h2 className="text-xl font-semibold">Detection Output:</h2>
              <div className="relative w-full aspect-video bg-black rounded-lg overflow-hidden">
                <img
                  src={outputImage}
                  alt="Detection output"
                  className="w-full h-full object-contain"
                />
              </div>
            </div>
          )}
        </div>
      )}

      {detections && (
        <div className="w-full bg-slate-100 dark:bg-slate-800 rounded-lg p-4">
          <h2 className="text-xl font-semibold mb-2">Detection Results:</h2>
          <pre className="text-sm overflow-auto max-h-96">
            {JSON.stringify(detections, null, 2)}
          </pre>
        </div>
      )}
    </div>
  );
}
