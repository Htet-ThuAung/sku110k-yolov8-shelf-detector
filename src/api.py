from fastapi import APIRouter, UploadFile, File
from fastapi.responses import HTMLResponse
from ultralytics import YOLO
import shutil
import os
import csv
import cv2
import time
from src.counting import count_boxes

router = APIRouter()

# Load YOLO model
model = YOLO("outputs/sku110k_continue/weights/best.pt")
CONF_THRESHOLD = 0.4

# Directories
OUTPUT_DIR = "outputs/inference_api_results"
ANNOTATED_DIR = os.path.join(OUTPUT_DIR, "annotated")
CSV_DIR = os.path.join(OUTPUT_DIR, "csvfile")
CSV_PATH = os.path.join(CSV_DIR, "results.csv")

# Ensure directories exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(ANNOTATED_DIR, exist_ok=True)
os.makedirs(CSV_DIR, exist_ok=True)


@router.post("/upload/", response_class=HTMLResponse)
async def predict_shelf(file: UploadFile = File(...)):
    try:
        # Save uploaded image
        input_path = os.path.join(OUTPUT_DIR, file.filename)
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Start timing
        start_time = time.time()

        # Run detection
        results = model(input_path, conf=CONF_THRESHOLD)
        result = results[0]
        count = count_boxes(result, conf_threshold=CONF_THRESHOLD)

        # Duration
        duration = time.time() - start_time

        # Read original image
        image = cv2.imread(input_path)
        if image is None:
            raise RuntimeError(f"Failed to load image from {input_path}")

        # Draw bounding boxes (no labels)
        boxes = result.boxes
        if boxes is not None and boxes.xyxy is not None:
            for box in boxes.xyxy.cpu().numpy().astype(int):
                if len(box) >= 4:
                    x1, y1, x2, y2 = box[:4]
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Overlay count and duration
        cv2.putText(
            image,
            f"Count: {count}",
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            image,
            f"Time: {duration:.2f}s",
            (10, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )

        # Save annotated image
        annotated_filename = f"annotated_{file.filename}"
        annotated_path = os.path.join(ANNOTATED_DIR, annotated_filename)
        cv2.imwrite(annotated_path, image)

        # CSV record
        image_url = f"/static/annotated/{annotated_filename}"
        row = [file.filename, count, f"{duration:.2f}", image_url]

        # Append to CSV
        file_exists = os.path.exists(CSV_PATH)
        with open(CSV_PATH, mode="a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["filename", "count", "duration(s)", "image"])
            writer.writerow(row)

        # Return HTML
        return HTMLResponse(
            f"""
            <h2>Prediction for {file.filename}</h2>
            <p><strong>Count:</strong> {count}</p>
            <p><strong>Processing Time:</strong> {duration:.2f}s</p>
            <img src="{image_url}" width="500"><br><br>
            <a href="/">← Back</a>
        """
        )

    except Exception as e:
        import traceback

        traceback.print_exc()
        return HTMLResponse(
            f"<h1>Internal Server Error</h1><p>{str(e)}</p>", status_code=500
        )
