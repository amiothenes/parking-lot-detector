import cv2
import numpy as np
import onnxruntime as ort
from dataclasses import dataclass
from typing import List, Optional
import time
import os

@dataclass
class VehicleDetection:
    class_name: str
    class_id: int
    confidence: float
    bbox: dict

@dataclass
class VehicleResult:
    vehicles: List[VehicleDetection]
    timing: dict
    frame_size: tuple

VEHICLE_CLASSES = {3: "car", 4: "motorcycle", 6: "bus", 8: "truck"}

class VehicleDetector:
    def __init__(self, model_path: str = None, config: dict = None):
        default_config = {
            "confidence_threshold": 0.15,
        }
        self.config = {**default_config, **(config or {})}
        
        if model_path is None:
            if os.path.exists("ssd_mobilenet_v1_12.onnx"): model_path = "ssd_mobilenet_v1_12.onnx"
            elif os.path.exists("models/ssd_mobilenet_v1_12.onnx"): model_path = "models/ssd_mobilenet_v1_12.onnx"
        
        self.session = None
        if model_path and os.path.exists(model_path):
            print(f"[VehicleDetector] Loading model: {model_path}")
            self.session = ort.InferenceSession(model_path)
            self.input_name = self.session.get_inputs()[0].name
            
            self.output_map = {}
            for i, node in enumerate(self.session.get_outputs()):
                if "box" in node.name.lower(): self.output_map["boxes"] = i
                elif "score" in node.name.lower(): self.output_map["scores"] = i
                elif "class" in node.name.lower(): self.output_map["classes"] = i
        else:
            print("[VehicleDetector] WARNING: No model found. Using Fallback mode only.")

    def detect(self, image_bytes: bytes) -> VehicleResult:
        timing = {}
        start_total = time.time()
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        orig_h, orig_w = image.shape[:2]

        vehicles = []
        
        if self.session:
            start = time.time()
            input_w, input_h = 300, 300
            resized = cv2.resize(image, (input_w, input_h))
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            input_tensor = np.expand_dims(rgb, axis=0).astype(np.uint8)
            
            outputs = self.session.run(None, {self.input_name: input_tensor})
            
            idx_boxes = self.output_map.get("boxes", 0)
            idx_classes = self.output_map.get("classes", 1)
            idx_scores = self.output_map.get("scores", 2)
            
            boxes = outputs[idx_boxes][0]
            classes = outputs[idx_classes][0]
            scores = outputs[idx_scores][0]

            for i, score in enumerate(scores):
                if score > self.config["confidence_threshold"]:
                    class_id = int(classes[i])
                    if class_id in VEHICLE_CLASSES:
                        y1, x1, y2, x2 = boxes[i]
                        bbox = {
                            "x1": float(max(0, x1 * orig_w)), "y1": float(max(0, y1 * orig_h)),
                            "x2": float(min(orig_w, x2 * orig_w)), "y2": float(min(orig_h, y2 * orig_h))
                        }
                        bbox["width"] = bbox["x2"] - bbox["x1"]
                        bbox["height"] = bbox["y2"] - bbox["y1"]
                        bbox["centerX"] = (bbox["x1"] + bbox["x2"]) / 2
                        bbox["centerY"] = (bbox["y1"] + bbox["y2"]) / 2
                        
                        vehicles.append(VehicleDetection(
                            class_name=VEHICLE_CLASSES[class_id], class_id=class_id, confidence=float(score), bbox=bbox
                        ))
            timing["inference_ms"] = round((time.time() - start) * 1000, 2)

        if len(vehicles) == 0:
            print("[VehicleDetector] AI found nothing. Running Fallback Blob Scan...")
            fallback_vehicles = self._fallback_blob_detect(image)
            vehicles.extend(fallback_vehicles)
            timing["fallback_ms"] = "Run"

        timing["total_ms"] = round((time.time() - start_total) * 1000, 2)
        return VehicleResult(vehicles=vehicles, timing=timing, frame_size=(orig_w, orig_h))

    def _fallback_blob_detect(self, image) -> List[VehicleDetection]:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, (0, 100, 100), (10, 255, 255))
        mask2 = cv2.inRange(hsv, (170, 100, 100), (180, 255, 255))
        mask3 = cv2.inRange(hsv, (0, 0, 200), (180, 50, 255))
        mask = mask1 | mask2 | mask3
        
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=2)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        blobs = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 2000 and area < 100000: # Filter noise
                x, y, w, h = cv2.boundingRect(cnt)
                bbox = {
                    "x1": float(x), "y1": float(y), "x2": float(x+w), "y2": float(y+h),
                    "width": w, "height": h, "centerX": x+w/2, "centerY": y+h/2
                }
                blobs.append(VehicleDetection("car", 3, 0.5, bbox))
        return blobs

# Helpers
_detector_instance = None
def get_vehicle_detector(model_path: str = None) -> VehicleDetector:
    global _detector_instance
    if _detector_instance is None: _detector_instance = VehicleDetector(model_path)
    return _detector_instance

def warmup_detector():
    try:
        det = get_vehicle_detector()
        det.detect(cv2.imencode(".jpg", np.zeros((300,300,3), np.uint8))[1].tobytes())
        print("[VehicleDetector] Warmup complete")
    except: pass

def vehicle_to_dict(v: VehicleDetection) -> dict:
    return {
        "class": v.class_name, "classId": v.class_id, "confidence": round(v.confidence, 3),
        "bbox": { "x1": round(v.bbox["x1"], 1), "y1": round(v.bbox["y1"], 1), "x2": round(v.bbox["x2"], 1), "y2": round(v.bbox["y2"], 1), "width": round(v.bbox["width"], 1), "height": round(v.bbox["height"], 1), "centerX": round(v.bbox["centerX"], 1), "centerY": round(v.bbox["centerY"], 1) }
    }