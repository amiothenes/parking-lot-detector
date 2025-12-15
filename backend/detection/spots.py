import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
import time

@dataclass
class ParkingSpot:
    id: str
    corners: List[Tuple[int, int]]
    bbox: dict
    angle: float
    area: float
    confidence: float
    occupied: bool = False

@dataclass
class DetectionResult:
    spots: List[ParkingSpot]
    lines: List[dict]
    edges: np.ndarray
    image_size: Tuple[int, int]
    timing: dict
    orientation: str

class SpotDetector:
    def __init__(self, config: dict = None):
        default_config = {
            "white_thresh": 160,
            "saturation_max": 50,
            "peak_threshold": 0.5,
            "min_spot_gap": 35,
            "min_spot_width": 60,
            "edge_margin_percent": 0.15,
            
            "occupancy_edge_threshold": 0.04,
            "padding": 15               
        }
        self.config = {**default_config, **(config or {})}

    def detect(self, image_bytes: bytes) -> DetectionResult:
        timing = {}
        start_total = time.time()

        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        height, width = image.shape[:2]

        start = time.time()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        saturation = hsv[:, :, 1]
        
        bright_mask = blur > self.config["white_thresh"]
        low_sat_mask = saturation < self.config["saturation_max"]
        mask = (bright_mask & low_sat_mask).astype(np.uint8) * 255
        
        timing["masking_ms"] = round((time.time() - start) * 1000, 2)

        edge_margin = int(height * self.config["edge_margin_percent"])
        
        mask_filtered = mask.copy()
        mask_filtered[:edge_margin, :] = 0
        mask_filtered[height - edge_margin:, :] = 0

        start = time.time()
        v_proj = np.sum(mask_filtered, axis=0)
        h_proj = np.sum(mask_filtered, axis=1)
        
        v_norm = v_proj / (np.max(v_proj) + 1e-5)
        h_norm = h_proj / (np.max(h_proj) + 1e-5)
        
        v_var = np.var(v_norm)
        h_var = np.var(h_norm)
        
        spots = []
        orientation = "unknown"

        if h_var > v_var: 
            orientation = "vertical"
            spine_pos = self._find_peak_center(v_proj, width)
            dividers = self._find_peaks(h_proj, height, edge_margin, height - edge_margin)
            spots = self._make_vertical_spots(spine_pos, dividers, width, height, edge_margin)
        else:
            orientation = "horizontal"
            spine_pos = self._find_peak_center(h_proj, height)
            dividers = self._find_peaks(v_proj, width)
            spots = self._make_horizontal_spots(spine_pos, dividers, width, height)

        timing["construct_ms"] = round((time.time() - start) * 1000, 2)

        start = time.time()
        for spot in spots:
            spot.occupied = self._check_occupancy(hsv, spot.bbox)
        timing["occupancy_ms"] = round((time.time() - start) * 1000, 2)

        timing["total_ms"] = round((time.time() - start_total) * 1000, 2)
        print(f"[SpotDetector] Detected {orientation.upper()} layout ({len(spots)} spots)")

        return DetectionResult(
            spots=spots, lines=[], edges=mask,
            image_size=(width, height), timing=timing, orientation=orientation
        )

    def _find_peak_center(self, hist: np.ndarray, length: int) -> int:
        mx = np.max(hist)
        if mx == 0: return length // 2
        valid = hist > (mx * 0.5) 
        indices = np.where(valid)[0]
        if len(indices) == 0: return length // 2
        return int(np.mean(indices))

    def _find_peaks(self, hist: np.ndarray, length: int, min_pos: int = 0, max_pos: int = None) -> List[int]:
        if max_pos is None:
            max_pos = length
            
        mx = np.max(hist)
        if mx == 0: return [min_pos, max_pos]
        thresh = mx * self.config["peak_threshold"]
        indices = np.where(hist > thresh)[0]
        
        indices = indices[(indices >= min_pos) & (indices <= max_pos)]
        
        peaks = []
        if len(indices) > 0:
            cluster = [indices[0]]
            for i in range(1, len(indices)):
                if indices[i] - indices[i-1] < 15:
                    cluster.append(indices[i])
                else:
                    peaks.append(int(np.mean(cluster)))
                    cluster = [indices[i]]
            peaks.append(int(np.mean(cluster)))
        
        return peaks

    def _make_vertical_spots(self, spine_x, y_divs, w, h, edge_margin):
        spots = []
        if len(y_divs) < 2: return []
        
        filtered_divs = [d for d in y_divs if edge_margin <= d <= h - edge_margin]
        if len(filtered_divs) < 2:
            return []
        
        min_height = self.config.get("min_spot_width", 60)
        final_divs = [filtered_divs[0]]
        for i in range(1, len(filtered_divs)):
            if filtered_divs[i] - final_divs[-1] >= min_height:
                final_divs.append(filtered_divs[i])
        
        row_char = 65  # 'A'
        for i in range(len(final_divs) - 1):
            y1, y2 = final_divs[i], final_divs[i+1]
            if (y2 - y1) < self.config["min_spot_gap"]: 
                continue
            spots.append(self._create_spot(f"{chr(row_char)}1", 0, y1, spine_x, y2))
            spots.append(self._create_spot(f"{chr(row_char)}2", spine_x, y1, w, y2))
            row_char += 1
        return spots

    def _make_horizontal_spots(self, spine_y, x_divs, w, h):
        spots = []
        if len(x_divs) < 2: return []
        
        filtered_divs = [x_divs[0]]
        min_width = self.config.get("min_spot_width", 60)
        
        for i in range(1, len(x_divs)):
            if x_divs[i] - filtered_divs[-1] >= min_width:
                filtered_divs.append(x_divs[i])
        
        for i in range(len(filtered_divs) - 1):
            x1, x2 = filtered_divs[i], filtered_divs[i+1]
            if (x2 - x1) < self.config["min_spot_gap"]: continue
            spots.append(self._create_spot(f"A{i+1}", x1, 0, x2, spine_y))
            spots.append(self._create_spot(f"B{i+1}", x1, spine_y, x2, h))
        return spots

    def _create_spot(self, label, x1, y1, x2, y2) -> ParkingSpot:
        corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        bbox = {
            "x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2),
            "width": int(x2-x1), "height": int(y2-y1),
            "centerX": float((x1+x2)/2), "centerY": float((y1+y2)/2)
        }
        return ParkingSpot(id=label, corners=corners, bbox=bbox, angle=0.0, area=float((x2-x1)*(y2-y1)), confidence=1.0)

    def _check_occupancy(self, hsv_image: np.ndarray, bbox: dict) -> bool:
        pad = self.config["padding"]
        x1 = max(0, int(bbox["x1"]) + pad)
        y1 = max(0, int(bbox["y1"]) + pad)
        x2 = min(hsv_image.shape[1], int(bbox["x2"]) - pad)
        y2 = min(hsv_image.shape[0], int(bbox["y2"]) - pad)
        
        if x2 <= x1 or y2 <= y1: 
            return False
        
        roi_bgr = cv2.cvtColor(hsv_image[y1:y2, x1:x2], cv2.COLOR_HSV2BGR)
        roi_gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        
        edges = cv2.Canny(roi_gray, 50, 150)
        
        edge_pixels = np.count_nonzero(edges)
        total_pixels = edges.size
        edge_density = edge_pixels / total_pixels if total_pixels > 0 else 0
        
        return edge_density > self.config.get("occupancy_edge_threshold", 0.04)

    def generate_debug_image(self, image_bytes: bytes, result, occupancy_results: list = None, vehicles: list = None) -> bytes:
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        occ_lookup = {}
        if occupancy_results:
            print(f"[DEBUG] Building occupancy lookup from {len(occupancy_results)} results")
            for occ in occupancy_results:
                spot_id = occ.spot.id if hasattr(occ, 'spot') else occ.get('id')
                is_occupied = occ.occupied if hasattr(occ, 'occupied') else occ.get('occupied', False)
                occ_lookup[spot_id] = is_occupied
                print(f"[DEBUG] Spot {spot_id}: occupied={is_occupied}")
        else:
            print("[DEBUG] No occupancy_results provided - using edge-based fallback")
        
        for spot in result.spots:
            pts = np.array(spot.corners, np.int32).reshape((-1, 1, 2))
            
            is_occupied = occ_lookup.get(spot.id, spot.occupied)
            
            if is_occupied:
                cv2.polylines(image, [pts], True, (0, 0, 255), 4)  # Red
                cv2.putText(image, "OCCUPIED", (spot.bbox["x1"]+10, int(spot.bbox["centerY"])), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            else:
                cv2.polylines(image, [pts], True, (0, 255, 0), 3)  # Green
            
            cv2.putText(image, spot.id, (spot.bbox["x1"]+10, spot.bbox["y1"]+30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        if vehicles:
            for v in vehicles:
                bbox = v.bbox if hasattr(v, 'bbox') else v.get('bbox')
                if bbox:
                    x1, y1 = int(bbox["x1"]), int(bbox["y1"])
                    x2, y2 = int(bbox["x2"]), int(bbox["y2"])
                    cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 0), 3)
                    conf = v.confidence if hasattr(v, 'confidence') else v.get('confidence', 0)
                    cv2.putText(image, f"Car: {conf:.2f}", (x1, y1-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        _, buffer = cv2.imencode(".png", image)
        return buffer.tobytes()


def spot_to_dict(spot: ParkingSpot) -> dict:
    return {
        "id": spot.id, 
        "occupied": spot.occupied, 
        "corners": [[int(c[0]), int(c[1])] for c in spot.corners],
        "bbox": {
            "x1": int(spot.bbox["x1"]), 
            "y1": int(spot.bbox["y1"]), 
            "x2": int(spot.bbox["x2"]), 
            "y2": int(spot.bbox["y2"]), 
            "width": int(spot.bbox["width"]), 
            "height": int(spot.bbox["height"]), 
            "centerX": float(spot.bbox["centerX"]), 
            "centerY": float(spot.bbox["centerY"])
        },
        "angle": float(round(spot.angle, 2)), 
        "area": float(round(spot.area, 2)), 
        "confidence": float(round(spot.confidence, 3))
    }