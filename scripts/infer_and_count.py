import argparse
from ultralytics import YOLO
from src.counting import count_boxes
import os
import cv2
import csv


def main():
    parser = argparse.ArgumentParser(description="Run inference and count items")

    parser.add_argument("--model", default="yolov8n.pt", help="Path to trained model")
    parser.add_argument(
        "--source",
        default="datasets/sku-110k/test/images",
        help="Path to image or directory",
    )
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument(
        "--output",
        default="outputs/sku_110k_inference",
        help="Folder to save annotated images",
    )

    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)

    model = YOLO(args.model)
    results = model(args.source, conf=args.conf)

    # Logging CSV
    log_path = os.path.join(args.output, "detection_counts.csv")
    file_exists = os.path.isfile(log_path)
    with open(log_path, mode="a", newline="") as log_file:
        writer = csv.writer(log_file)

        # Write header only if file is new
        if not file_exists:
            writer.writerow(["filename", "count"])

        for i, r in enumerate(results):
            count = count_boxes(r, conf_threshold=args.conf)
            print(f"[{i}] → Found {count} items")

            filename = os.path.basename(r.path)
            writer.writerow([filename, count])

            # Save annotated image
            annotated = r.plot(conf=False, labels=False)
            cv2.imwrite(os.path.join(args.output, filename), annotated)
            print(f"Results saved to {args.output} and counts logged to {log_path}")


if __name__ == "__main__":
    main()
