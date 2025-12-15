from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import base64
import requests
import time
import threading

from detection import (
    SpotDetector,
    spot_to_dict,
    get_vehicle_detector,
    warmup_detector,
    vehicle_to_dict,
    OccupancyAnalyzer,
    format_summary,
    occupancy_to_dict,
)

app = FastAPI(title="Parking Detector API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

parking_spots = []
is_initialized = False
image_size = {"width": 0, "height": 0}
debug_data = {}
last_cv_result = None

spot_detector = SpotDetector()
occupancy_analyzer = OccupancyAnalyzer()

threading.Thread(target=warmup_detector, daemon=True).start()


async def get_image_bytes(request: Request) -> bytes:
    """Extract image bytes from various request formats"""
    content_type = request.headers.get("content-type", "")

    if "application/json" in content_type:
        body = await request.json()

        if "imageBase64" in body:
            b64 = body["imageBase64"]
            b64 = b64.split(",", 1)[-1]
            return base64.b64decode(b64)

        if "image" in body:
            r = requests.get(body["image"], timeout=10)
            if r.status_code != 200:
                raise HTTPException(400, "Failed to fetch image from URL")
            return r.content

        raise HTTPException(400, "Provide imageBase64 or image URL")

    if "multipart/form-data" in content_type:
        form = await request.form()
        file = form.get("image")
        if file is None:
            raise HTTPException(400, "Image file required")
        return await file.read()

    raise HTTPException(400, "Unsupported content type")


@app.post("/api/detect")
async def detect(request: Request):
    """
    Main detection endpoint
    
    Query params:
        ?mode=init    - Force re-initialization (detect spots)
        ?mode=detect  - Run vehicle detection only (default if initialized)
        ?debug=true   - Return debug visualization
    """
    global parking_spots, is_initialized, image_size, debug_data, last_cv_result

    mode = request.query_params.get("mode", "auto")
    debug = request.query_params.get("debug", "false").lower() == "true"

    try:
        img_bytes = await get_image_bytes(request)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(400, f"Failed to read image: {str(e)}")

    start_total = time.time()

    if mode == "init" or (not is_initialized and mode == "auto"):
        print("[API] Running initialization - Classical CV spot detection")

        cv_result = spot_detector.detect(img_bytes)
        last_cv_result = cv_result

        parking_spots = cv_result.spots
        image_size = {"width": cv_result.image_size[0], "height": cv_result.image_size[1]}
        is_initialized = True

        debug_data = {
            "lines_count": len(cv_result.lines),
            "timing": cv_result.timing,
        }

        response = {
            "success": True,
            "mode": "initialization",
            "message": f"Detected {len(parking_spots)} parking spots using Classical CV",
            "pipeline": {
                "step1": "Grayscale conversion (Y = 0.299R + 0.587G + 0.114B)",
                "step2": "Gaussian blur (5x5 kernel, noise reduction)",
                "step3": "Canny edge detection (gradient + non-max suppression + hysteresis)",
                "step4": "Hough line transform (polar voting: ρ = x·cos(θ) + y·sin(θ))",
                "step5": "Line grouping by angle similarity",
                "step6": "Parallel line pairing + rectangle fitting",
                "step7": "Non-maximum suppression (remove overlaps)",
            },
            "spots": [spot_to_dict(s) for s in parking_spots],
            "stats": {
                "totalSpots": len(parking_spots),
                "linesDetected": len(cv_result.lines),
            },
            "timing": cv_result.timing,
            "imageSize": image_size,
        }

        if debug:
            debug_image = spot_detector.generate_debug_image(
                img_bytes,
                cv_result,
                occupancy_results=None,
                vehicles=None
            )
            response["debugImage"] = f"data:image/png;base64,{base64.b64encode(debug_image).decode()}"

        return JSONResponse(response)

    if not is_initialized:
        return JSONResponse(
            {
                "error": "Pipeline not initialized",
                "hint": "Send an image with ?mode=init to detect parking spots first",
            },
            status_code=400,
        )

    print("[API] Running detection - Vehicle detection")

    try:
        vehicle_detector = get_vehicle_detector()
        vehicle_result = vehicle_detector.detect(img_bytes)
        vehicles_list = vehicle_result.vehicles
    except FileNotFoundError as e:
        return JSONResponse(
            {"error": str(e)},
            status_code=500,
        )

    occupancy_result = occupancy_analyzer.analyze(parking_spots, vehicles_list)

    total_time = round((time.time() - start_total) * 1000, 2)

    response = {
        "success": True,
        "mode": "detection",
        "summary": format_summary(occupancy_result),
        "occupancy": {
            "total": occupancy_result.summary["total"],
            "occupied": occupancy_result.summary["occupied"],
            "available": occupancy_result.summary["available"],
            "occupancyRate": round(occupancy_result.summary["occupancyRate"] * 100, 1),
        },
        "spots": [occupancy_to_dict(s) for s in occupancy_result.spots],
        "vehicles": [vehicle_to_dict(v) for v in vehicles_list],
        "timing": {
            "vehicleDetection": vehicle_result.timing,
            "totalMs": total_time,
        },
    }

    if debug:
        class MinimalResult:
            def __init__(self, spots):
                self.spots = spots
        
        debug_image = spot_detector.generate_debug_image(
            img_bytes,
            MinimalResult(parking_spots),
            occupancy_results=occupancy_result.spots,
            vehicles=vehicles_list
        )
        response["debugImage"] = f"data:image/png;base64,{base64.b64encode(debug_image).decode()}"

    return JSONResponse(response)


@app.get("/api/detect")
def status():
    """Get current pipeline status"""
    return {
        "initialized": is_initialized,
        "totalSpots": len(parking_spots),
        "imageSize": image_size,
        "spots": [spot_to_dict(s) for s in parking_spots] if is_initialized else [],
        "usage": {
            "initialize": "POST /api/detect?mode=init with image",
            "detect": "POST /api/detect with image",
            "debug": "POST /api/detect?mode=init&debug=true for visualization",
        },
    }


@app.delete("/api/detect")
def reset():
    """Reset the pipeline"""
    global parking_spots, is_initialized, image_size, debug_data, last_cv_result

    parking_spots = []
    is_initialized = False
    image_size = {"width": 0, "height": 0}
    debug_data = {}
    last_cv_result = None

    return {"success": True, "message": "Pipeline reset. Send a new image to reinitialize."}


@app.get("/health")
def health():
    """Health check endpoint"""
    return {"status": "ok", "initialized": is_initialized}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)