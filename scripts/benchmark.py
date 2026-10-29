"""Reproducible single-image inference benchmark for ShelfSense.

Example:
    python scripts/benchmark.py --model outputs/sku110k_continue/weights/best.pt --source datasets/SKU-110K/test --device 0 --imgsz 640 --warmup 5 --runs 30 --max-images 50 --output-dir reports/benchmark

This measures Ultralytics prediction latency (preprocessing + inference +
postprocessing) for each image. It excludes model loading, upload/network time,
image annotation, disk writes, and CSV history writes.
"""
import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_version(package_name):
    try:
        return metadata.version(package_name)
    except metadata.PackageNotFoundError:
        return "not installed"


def git_revision():
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=3,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def collect_images(source, max_images):
    source = Path(source)
    if not source.exists():
        raise FileNotFoundError(f"Benchmark source does not exist: {source}")

    if source.is_file():
        candidates = [source] if source.suffix.lower() in IMAGE_EXTENSIONS else []
    elif source.is_dir():
        candidates = sorted(
            path for path in source.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
    else:
        candidates = []

    valid_images = []
    invalid_images = []
    for path in candidates:
        image = cv2.imread(str(path))
        if image is None:
            invalid_images.append(str(path))
        else:
            valid_images.append(path)
        if max_images and len(valid_images) >= max_images:
            break

    if not valid_images:
        raise ValueError(f"No readable images found at: {source}")
    return valid_images, invalid_images


def synchronize_if_cuda(torch, use_cuda):
    if use_cuda:
        torch.cuda.synchronize()


def summarize_latencies(latencies_ms):
    values = np.asarray(latencies_ms, dtype=np.float64)
    if values.size == 0:
        raise ValueError("At least one measured inference is required.")
    mean_ms = float(np.mean(values))
    return {
        "runs": int(values.size),
        "mean_ms": mean_ms,
        "median_ms": float(np.median(values)),
        "p95_ms": float(np.percentile(values, 95)),
        "min_ms": float(np.min(values)),
        "max_ms": float(np.max(values)),
        "throughput_images_per_second": float(1000.0 / mean_ms) if mean_ms > 0 else None,
    }


def get_hardware_info(torch, requested_device):
    cuda_available = bool(torch.cuda.is_available())
    if requested_device == "cpu":
        selected_device = "cpu"
    elif requested_device in {"auto", "0", "cuda"} and cuda_available:
        selected_device = "cuda:0"
    else:
        selected_device = requested_device

    gpu_name = None
    gpu_memory_total_mb = None
    if cuda_available and selected_device != "cpu":
        try:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory_total_mb = round(
                torch.cuda.get_device_properties(0).total_memory / (1024 * 1024), 1
            )
        except Exception:
            pass

    return {
        "platform": platform.platform(),
        "python_version": sys.version.split()[0],
        "processor": platform.processor() or "not reported by platform",
        "cpu_logical_cores": os.cpu_count(),
        "torch_version": package_version("torch"),
        "ultralytics_version": package_version("ultralytics"),
        "opencv_version": cv2.__version__,
        "numpy_version": np.__version__,
        "cuda_available": cuda_available,
        "torch_cuda_version": getattr(torch.version, "cuda", None),
        "gpu_name": gpu_name,
        "gpu_memory_total_mb": gpu_memory_total_mb,
        "requested_device": requested_device,
        "selected_device": selected_device,
    }


def write_report(output_dir, summary, per_image_rows):
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "benchmark_summary.json"
    csv_path = output_dir / "benchmark_per_image.csv"
    markdown_path = output_dir / "benchmark_report.md"

    with json_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    fieldnames = [
        "image", "image_sha256", "width", "height", "measured_runs",
        "mean_ms", "median_ms", "p95_ms", "min_ms", "max_ms",
        "throughput_images_per_second", "mean_detections",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(per_image_rows)

    overall = summary["overall_latency"]
    hardware = summary["hardware"]
    lines = [
        "# ShelfSense Inference Benchmark",
        "",
        f"- Generated (UTC): {summary['generated_at_utc']}",
        f"- Model: {summary['model_path']}",
        f"- Model SHA-256: {summary['model_sha256']}",
        f"- Git revision: {summary['git_revision'] or 'unavailable'}",
        f"- Source: {summary['source']}",
        f"- Readable images: {summary['image_count']}",
        f"- Unreadable image files skipped: {len(summary['unreadable_images'])}",
        f"- Image size: {summary['settings']['imgsz']} (Ultralytics inference size)",
        f"- Confidence threshold: {summary['settings']['confidence']}",
        f"- Warm-up predictions per image: {summary['settings']['warmup']}",
        f"- Measured predictions per image: {summary['settings']['runs']}",
        f"- Maximum images: {summary['settings']['max_images'] or 'all readable images'}",
        f"- Device requested: {hardware['requested_device']}",
        f"- Device selected: {hardware['selected_device']}",
        f"- GPU: {hardware['gpu_name'] or 'not used / not detected'}",
        "",
        "## Latency summary",
        "",
        "| Measure | Result |",
        "|---|---:|",
        f"| Mean prediction latency | {overall['mean_ms']:.2f} ms |",
        f"| Median prediction latency | {overall['median_ms']:.2f} ms |",
        f"| 95th percentile latency | {overall['p95_ms']:.2f} ms |",
        f"| Minimum latency | {overall['min_ms']:.2f} ms |",
        f"| Maximum latency | {overall['max_ms']:.2f} ms |",
        f"| Sequential throughput estimate | {overall['throughput_images_per_second']:.2f} images/s |",
        "",
        "Latency statistics are calculated across all measured image predictions after warm-up. Throughput is the reciprocal of mean per-image prediction latency; it is not a concurrent-server load test.",
        "",
        "## Measurement scope",
        "",
        "The timed region includes the Ultralytics prediction call (image preprocessing, model inference, and postprocessing). Model loading, file upload/network time, annotation, saving output images, and CSV writes are excluded. GPU execution is synchronized around timed calls when CUDA is selected.",
        "",
        "## Environment",
        "",
        "| Property | Value |",
        "|---|---|",
        f"| OS/platform | {hardware['platform']} |",
        f"| Python | {hardware['python_version']} |",
        f"| Processor | {hardware['processor']} |",
        f"| Logical CPU cores | {hardware['cpu_logical_cores']} |",
        f"| PyTorch | {hardware['torch_version']} |",
        f"| Ultralytics | {hardware['ultralytics_version']} |",
        f"| OpenCV | {hardware['opencv_version']} |",
        f"| NumPy | {hardware['numpy_version']} |",
        f"| PyTorch CUDA runtime | {hardware['torch_cuda_version'] or 'unavailable'} |",
        f"| GPU memory | {str(hardware['gpu_memory_total_mb']) + ' MB' if hardware['gpu_memory_total_mb'] else 'not reported'} |",
        "",
        "## Reproduction",
        "",
        "Run the same command with the same checkpoint, input images, software versions, device, image size, confidence threshold, warm-up count, and measured run count. Compare the model SHA-256 and per-image SHA-256 values to confirm the artifacts match.",
        "",
        "~~~bash",
        "python scripts/benchmark.py --model outputs/sku110k_continue/weights/best.pt --source datasets/SKU-110K/test --device 0 --imgsz 640 --conf 0.4 --warmup 5 --runs 30 --output-dir reports/benchmark",
        "~~~",
        "",
        "## Interpretation and limitations",
        "",
        "- This is a local single-image prediction benchmark, not an end-to-end FastAPI latency or multi-user load test.",
        "- Results depend on device, drivers, thermal/power state, image size, and library versions.",
        "- Mean latency and estimated throughput should not be compared across different hardware as if they were controlled experiments.",
        "- Detection counts are included for context only; this benchmark does not measure accuracy.",
        "- The report records input and checkpoint hashes so a later run can be checked for artifact consistency.",
        "",
    ]
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, csv_path, markdown_path


def run_benchmark(args):
    model_path = Path(args.model)
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Model checkpoint not found: {model_path}. Provide the trained best.pt path."
        )
    if args.imgsz < 1 or args.warmup < 0 or args.runs < 1 or args.max_images < 0:
        raise ValueError("imgsz and runs must be positive; warmup and max-images must be zero or greater.")
    if not 0.0 <= args.conf <= 1.0:
        raise ValueError("Confidence threshold must be between 0 and 1.")

    images, unreadable = collect_images(args.source, args.max_images)

    import torch
    from ultralytics import YOLO

    hardware = get_hardware_info(torch, args.device)
    model = YOLO(str(model_path))
    device_arg = None if args.device == "auto" else args.device
    use_cuda = device_arg != "cpu" and bool(torch.cuda.is_available())

    per_image_rows = []
    all_latencies = []
    total_detections = 0
    benchmark_started = time.perf_counter()

    for image_path in images:
        image = cv2.imread(str(image_path))
        if image is None:
            raise RuntimeError(f"Image became unreadable during benchmark: {image_path}")
        image_latencies = []
        result = None

        for _ in range(args.warmup):
            result = model.predict(
                source=image, imgsz=args.imgsz, conf=args.conf,
                device=device_arg, verbose=False,
            )
        synchronize_if_cuda(torch, use_cuda)

        for _ in range(args.runs):
            synchronize_if_cuda(torch, use_cuda)
            start = time.perf_counter()
            results = model.predict(
                source=image, imgsz=args.imgsz, conf=args.conf,
                device=device_arg, verbose=False,
            )
            synchronize_if_cuda(torch, use_cuda)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            image_latencies.append(elapsed_ms)
            result = results[0]

        stats = summarize_latencies(image_latencies)
        detections = len(result.boxes) if result is not None and result.boxes is not None else 0
        total_detections += detections
        all_latencies.extend(image_latencies)
        try:
            relative_image = str(image_path.resolve().relative_to(Path.cwd().resolve()))
        except ValueError:
            relative_image = str(image_path.resolve())

        per_image_rows.append({
            "image": relative_image,
            "image_sha256": sha256_file(image_path),
            "width": int(image.shape[1]),
            "height": int(image.shape[0]),
            "measured_runs": stats["runs"],
            "mean_ms": round(stats["mean_ms"], 4),
            "median_ms": round(stats["median_ms"], 4),
            "p95_ms": round(stats["p95_ms"], 4),
            "min_ms": round(stats["min_ms"], 4),
            "max_ms": round(stats["max_ms"], 4),
            "throughput_images_per_second": round(stats["throughput_images_per_second"], 4),
            "mean_detections": detections,
        })
        print(
            f"{image_path.name}: mean={stats['mean_ms']:.2f} ms, "
            f"median={stats['median_ms']:.2f} ms, p95={stats['p95_ms']:.2f} ms"
        )

    elapsed_seconds = time.perf_counter() - benchmark_started
    summary = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_path": str(model_path.resolve()),
        "model_sha256": sha256_file(model_path),
        "git_revision": git_revision(),
        "source": str(Path(args.source).resolve()),
        "image_count": len(images),
        "unreadable_images": unreadable,
        "settings": {
            "imgsz": args.imgsz,
            "confidence": args.conf,
            "warmup": args.warmup,
            "runs": args.runs,
            "max_images": args.max_images,
            "timing_scope": "Ultralytics per-image predict call; excludes model loading and application I/O",
        },
        "hardware": hardware,
        "overall_latency": summarize_latencies(all_latencies),
        "total_detections_in_final_run_per_image": total_detections,
        "benchmark_elapsed_seconds_including_setup": round(elapsed_seconds, 3),
        "per_image_results": per_image_rows,
    }
    output_dir = Path(args.output_dir)
    paths = write_report(output_dir, summary, per_image_rows)
    print("\nOverall benchmark:")
    print(json.dumps(summary["overall_latency"], indent=2))
    print("\nReports written:")
    for path in paths:
        print(f"- {path}")
    print("No performance values have been assumed; these values come from this run.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark ShelfSense YOLO inference and generate reproducible JSON, CSV, and Markdown reports."
    )
    parser.add_argument("--model", default="outputs/sku110k_continue/weights/best.pt")
    parser.add_argument("--source", required=True, help="One image or a directory of images (searched recursively).")
    parser.add_argument("--device", choices=["auto", "cpu", "0"], default="auto", help='Use "auto", "cpu", or "0" for the first CUDA GPU (default: auto).')
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size (default: 640).")
    parser.add_argument("--conf", type=float, default=0.4, help="Detection confidence threshold (default: 0.4).")
    parser.add_argument("--warmup", type=int, default=5, help="Warm-up predictions per image (default: 5).")
    parser.add_argument("--runs", type=int, default=30, help="Measured predictions per image (default: 30).")
    parser.add_argument("--max-images", type=int, default=50, help="Maximum sorted images to benchmark; use 0 for all (default: 50).")
    parser.add_argument("--output-dir", default="reports/benchmark")
    return parser.parse_args()


if __name__ == "__main__":
    try:
        run_benchmark(parse_args())
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(f"Benchmark error: {error}") from error
