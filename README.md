# Parking Lot Detector

Finds open parking spots from a lot image. Two-stage pipeline — classical computer vision maps out where the spots
are, then a vehicle detector checks which ones are occupied.

## How it works

**Spot detection:** grayscale → Gaussian blur → Canny edge detection → Hough line transform → line grouping and
parallel pairing → rectangle fitting → non-max suppression. This finds the actual parking line geometry in the image.

**Occupancy check:** runs a vehicle detector over the same frame, then matches detections against the spot rectangles
to return which are occupied, which are open, and the overall occupancy rate.

## Stack

Next.js (frontend) · FastAPI (backend) · OpenCV for the CV pipeline · ONNX Runtime for vehicle detection · Roboflow
for the model/data pipeline.

## Running it

Backend: `cd backend && pip install -r requirements.txt && uvicorn main:app --reload`
Frontend: `npm install && npm run dev`

## Team

Built with [Diego](https://github.com/diegonsm) and Dorian.
