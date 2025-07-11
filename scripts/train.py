import argparse
from src.train import train_model


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8 model")

    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--data", default="configs/sku110k.yaml")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=2)  # safer default for your GPU
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--project", default="outputs")
    parser.add_argument("--name", default="sku110k_run")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--cache", action="store_true", help="Enable image caching (requires RAM)"
    )

    args = parser.parse_args()

    print(f"Starting training with config: {args}")

    train_model(
        args.model,
        args.data,
        args.epochs,
        args.batch,
        args.imgsz,
        args.project,
        args.name,
        workers=args.workers,
        cache=args.cache,
    )


if __name__ == "__main__":
    main()
