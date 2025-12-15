from dataclasses import dataclass
from typing import List, Optional

from .spots import ParkingSpot
from .vehicles import VehicleDetection

@dataclass
class SpotOccupancy:
    spot: ParkingSpot
    occupied: bool
    vehicle: Optional[VehicleDetection]
    iou_score: float
    coverage_score: float
    containment_score: float

@dataclass
class OccupancyResult:
    spots: List[SpotOccupancy]
    summary: dict
    vehicle_count: int

class OccupancyAnalyzer:
    def __init__(self, config: dict = None):
        default_config = {
            "min_iou_threshold": 0.10,    
            "min_coverage_threshold": 0.10,
            "min_containment_threshold": 0.50
        }
        self.config = {**default_config, **(config or {})}

    def analyze(self, spots: List[ParkingSpot], vehicles: List[VehicleDetection]) -> OccupancyResult:
        spot_occupancies = []

        for spot in spots:
            best_match = None
            best_score = 0
            
            # metrics for the best match
            match_iou = 0
            match_cov = 0
            match_cont = 0

            for vehicle in vehicles:
                intersection = self._get_intersection_area(spot.bbox, vehicle.bbox)
                if intersection == 0: continue
                
                spot_area = spot.bbox["width"] * spot.bbox["height"]
                veh_area = vehicle.bbox["width"] * vehicle.bbox["height"]
                union = spot_area + veh_area - intersection
                
                iou = intersection / union if union > 0 else 0
                coverage = intersection / spot_area if spot_area > 0 else 0
                containment = intersection / veh_area if veh_area > 0 else 0 #

                is_occupied = (
                    iou >= self.config["min_iou_threshold"] or
                    coverage >= self.config["min_coverage_threshold"] or
                    containment >= self.config["min_containment_threshold"]
                )

                if is_occupied:
                    if containment > best_score:
                        best_score = containment
                        best_match = vehicle
                        match_iou = iou
                        match_cov = coverage
                        match_cont = containment

            spot_occupancies.append(SpotOccupancy(
                spot=spot,
                occupied=best_match is not None,
                vehicle=best_match,
                iou_score=match_iou,
                coverage_score=match_cov,
                containment_score=match_cont
            ))

        occupied_count = sum(1 for s in spot_occupancies if s.occupied)
        total = len(spots)

        summary = {
            "total": total,
            "occupied": occupied_count,
            "available": total - occupied_count,
            "occupancyRate": occupied_count / total if total > 0 else 0,
        }

        return OccupancyResult(
            spots=spot_occupancies,
            summary=summary,
            vehicle_count=len(vehicles),
        )

    def _get_intersection_area(self, b1: dict, b2: dict) -> float:
        x1 = max(b1["x1"], b2["x1"])
        y1 = max(b1["y1"], b2["y1"])
        x2 = min(b1["x2"], b2["x2"])
        y2 = min(b1["y2"], b2["y2"])
        return max(0, x2 - x1) * max(0, y2 - y1)


def format_summary(result: OccupancyResult) -> str:
    s = result.summary
    rate = round(s["occupancyRate"] * 100, 1)
    return f"Parking Status: {s['available']}/{s['total']} spots available ({rate}% occupied)"


def occupancy_to_dict(occ: SpotOccupancy) -> dict:
    return {
        "id": occ.spot.id,
        "occupied": occ.occupied,
        "iouScore": round(occ.iou_score, 3),
        "coverageScore": round(occ.coverage_score, 3),
        "containmentScore": round(occ.containment_score, 3),
        "bbox": occ.spot.bbox,
    }