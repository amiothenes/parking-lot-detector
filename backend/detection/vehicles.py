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
            "confidence_threshold": 0.3,
            "min_vehicle_area_percent": 0.005,
            "max_vehicle_area_percent": 0.15,
            "edge_margin_percent": 0.12,
        }
        self.config = {**default_config, **(config or {})}
        
        if model_path is None:
            if os.path.exists("ssd_mobilenet_v1_12.onnx"): 
                model_path = "ssd_mobilenet_v1_12.onnx"
            elif os.path.exists("models/ssd_mobilenet_v1_12.onnx"): 
                model_path = "models/ssd_mobilenet_v1_12.onnx"
        
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
                            "x1": float(max(0, x1 * orig_w)), 
                            "y1": float(max(0, y1 * orig_h)),
                            "x2": float(min(orig_w, x2 * orig_w)), 
                            "y2": float(min(orig_h, y2 * orig_h))
                        }
                        bbox["width"] = bbox["x2"] - bbox["x1"]
                        bbox["height"] = bbox["y2"] - bbox["y1"]
                        bbox["centerX"] = (bbox["x1"] + bbox["x2"]) / 2
                        bbox["centerY"] = (bbox["y1"] + bbox["y2"]) / 2
                        
                        vehicles.append(VehicleDetection(
                            class_name=VEHICLE_CLASSES[class_id], 
                            class_id=class_id, 
                            confidence=float(score), 
                            bbox=bbox
                        ))
            timing["inference_ms"] = round((time.time() - start) * 1000, 2)
            print(f"[VehicleDetector] MobileNet found {len(vehicles)} vehicles")

        # Fallback: Color-based detection for toy cars
        if len(vehicles) == 0:
            print("[VehicleDetector] MobileNet found nothing. Running color-based detection...")
            start = time.time()
            fallback_vehicles = self._color_based_detect(image)
            vehicles.extend(fallback_vehicles)
            timing["fallback_ms"] = round((time.time() - start) * 1000, 2)
            print(f"[VehicleDetector] Fallback found {len(fallback_vehicles)} vehicles")

        timing["total_ms"] = round((time.time() - start_total) * 1000, 2)
        return VehicleResult(vehicles=vehicles, timing=timing, frame_size=(orig_w, orig_h))

    def _color_based_detect(self, image: np.ndarray) -> List[VehicleDetection]:
        """
        Detect vehicles by color (works well for toy cars).
        Detects: Red, Blue, Yellow, White, Black cars
        """
        h, w = image.shape[:2]
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        margin = int(h * self.config["edge_margin_percent"])
        
        color_masks = []
        
        mask_red1 = cv2.inRange(hsv, (0, 100, 100), (10, 255, 255))
        mask_red2 = cv2.inRange(hsv, (160, 100, 100), (180, 255, 255))
        color_masks.append(mask_red1 | mask_red2)
        
        mask_blue = cv2.inRange(hsv, (100, 100, 100), (130, 255, 255))
        color_masks.append(mask_blue)
        
        mask_yellow = cv2.inRange(hsv, (20, 100, 100), (35, 255, 255))
        color_masks.append(mask_yellow)
        
        mask_white = cv2.inRange(hsv, (0, 0, 200), (180, 30, 255))
        color_masks.append(mask_white)

        combined_mask = np.zeros_like(color_masks[0])
        for mask in color_masks:
            combined_mask = combined_mask | mask
        
        combined_mask[:margin, :] = 0
        combined_mask[h - margin:, :] = 0

        kernel = np.ones((7, 7), np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        vehicles = []
        total_area = w * h
        min_area = total_area * self.config["min_vehicle_area_percent"]
        max_area = total_area * self.config["max_vehicle_area_percent"]
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            
            if area < min_area or area > max_area:
                continue
            
            x, y, bw, bh = cv2.boundingRect(cnt)
            
            if y < margin or y + bh > h - margin:
                continue
            
            aspect = max(bw, bh) / (min(bw, bh) + 1)
            if aspect > 5:
                continue
            
            # Check rectangularity
            rect_area = bw * bh
            rectangularity = area / rect_area if rect_area > 0 else 0
            if rectangularity < 0.4:
                continue
            
            bbox = {
                "x1": float(x), 
                "y1": float(y),
                "x2": float(x + bw), 
                "y2": float(y + bh),
                "width": float(bw), 
                "height": float(bh),
                "centerX": float(x + bw / 2), 
                "centerY": float(y + bh / 2)
            }
            
            confidence = min(0.8, rectangularity + 0.2)
            
            vehicles.append(VehicleDetection(
                class_name="car",
                class_id=3,
                confidence=confidence,
                bbox=bbox
            ))
        
        vehicles = self._apply_nms(vehicles, iou_threshold=0.3)
        
        return vehicles
    
    def _apply_nms(self, detections: List[VehicleDetection], iou_threshold: float) -> List[VehicleDetection]:
        """Non-maximum suppression"""
        if len(detections) == 0:
            return []
        
        detections = sorted(detections, key=lambda d: d.confidence, reverse=True)
        
        keep = []
        while detections:
            best = detections.pop(0)
            keep.append(best)
            
            detections = [
                d for d in detections
                if self._compute_iou(best.bbox, d.bbox) < iou_threshold
            ]
        
        return keep
    
    def _compute_iou(self, b1: dict, b2: dict) -> float:
        x1 = max(b1["x1"], b2["x1"])
        y1 = max(b1["y1"], b2["y1"])
        x2 = min(b1["x2"], b2["x2"])
        y2 = min(b1["y2"], b2["y2"])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        
        area1 = b1["width"] * b1["height"]
        area2 = b2["width"] * b2["height"]
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0


# Helpers
_detector_instance = None

def get_vehicle_detector(model_path: str = None) -> VehicleDetector:
    global _detector_instance
    if _detector_instance is None: 
        _detector_instance = VehicleDetector(model_path)
    return _detector_instance

def warmup_detector():
    try:
        det = get_vehicle_detector()
        det.detect(cv2.imencode(".jpg", np.zeros((300, 300, 3), np.uint8))[1].tobytes())
        print("[VehicleDetector] Warmup complete")
    except Exception as e:
        print(f"[VehicleDetector] Warmup error: {e}")

def vehicle_to_dict(v: VehicleDetection) -> dict:
    return {
        "class": v.class_name, 
        "classId": v.class_id, 
        "confidence": round(v.confidence, 3),
        "bbox": {
            "x1": round(v.bbox["x1"], 1), 
            "y1": round(v.bbox["y1"], 1),
            "x2": round(v.bbox["x2"], 1), 
            "y2": round(v.bbox["y2"], 1),
            "width": round(v.bbox["width"], 1), 
            "height": round(v.bbox["height"], 1),
            "centerX": round(v.bbox["centerX"], 1), 
            "centerY": round(v.bbox["centerY"], 1)
        }
    }