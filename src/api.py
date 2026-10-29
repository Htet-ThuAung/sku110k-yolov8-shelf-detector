from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from ultralytics import YOLO
import shutil
import os
import csv
import cv2
import numpy as np
import time
from src.counting import count_boxes

router = APIRouter()

# Load YOLO model
model = YOLO("outputs/sku110k_continue/weights/best.pt")
CONF_THRESHOLD = 0.4
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

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
    """Validate an uploaded image, run detection, and save its result."""
    try:
        if not file.filename or not file.filename.strip():
            raise HTTPException(status_code=400, detail="A filename is required.")

        # Treat both slash styles as separators and reject path-like filenames.
        normalized_name = file.filename.replace("\\", "/")
        safe_filename = os.path.basename(normalized_name)
        if safe_filename != normalized_name or safe_filename in {"", ".", ".."}:
            raise HTTPException(status_code=400, detail="Invalid filename.")

        extension = os.path.splitext(safe_filename)[1].lower()
        if extension not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=415,
                detail="Unsupported image type. Upload JPG, JPEG, PNG, or WEBP.",
            )

        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="The uploaded file is empty.")
        if len(contents) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="Image exceeds the 10 MB limit.")

        # Decode before saving or running the model; extensions alone are not enough.
        image_array = np.frombuffer(contents, dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        if image is None:
            raise HTTPException(status_code=400, detail="The uploaded file is not a valid image.")

        input_path = os.path.join(OUTPUT_DIR, safe_filename)
        with open(input_path, "wb") as buffer:
            buffer.write(contents)

        start_time = time.time()
        results = model(input_path, conf=CONF_THRESHOLD)
        if not results:
            raise RuntimeError("The model returned no prediction results.")
        result = results[0]
        count = count_boxes(result, conf_threshold=CONF_THRESHOLD)
        duration = time.time() - start_time

        boxes = result.boxes
        if boxes is not None and boxes.xyxy is not None:
            for box in boxes.xyxy.cpu().numpy().astype(int):
                if len(box) >= 4:
                    x1, y1, x2, y2 = box[:4]
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.putText(
            image, f"Count: {count}", (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2,
        )
        cv2.putText(
            image, f"Time: {duration:.2f}s", (10, 80),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
        )

        annotated_filename = f"annotated_{safe_filename}"
        annotated_path = os.path.join(ANNOTATED_DIR, annotated_filename)
        if not cv2.imwrite(annotated_path, image):
            raise RuntimeError("Could not save the annotated image.")

        image_url = f"/static/annotated/{annotated_filename}"
        row = [safe_filename, count, f"{duration:.2f}", image_url]
        file_exists = os.path.exists(CSV_PATH)
        with open(CSV_PATH, mode="a", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            if not file_exists:
                writer.writerow(["filename", "count", "duration(s)", "image"])
            writer.writerow(row)

        return HTMLResponse(
            f"""
            <h2>Prediction for {safe_filename}</h2>
            <p><strong>Count:</strong> {count}</p>
            <p><strong>Processing Time:</strong> {duration:.2f}s</p>
            <img src="{image_url}" width="500"><br><br>
            <a href="/">← Back</a>
        """
        )
    except HTTPException as exc:
        return HTMLResponse(
            f"<h1>{exc.status_code} Error</h1><p>{exc.detail}</p>",
            status_code=exc.status_code,
        )
    except Exception:
        # Do not expose local paths, model details, or stack traces to clients.
        return HTMLResponse(
            "<h1>Internal Server Error</h1><p>Unable to process this image.</p>",
            status_code=500,
        )
