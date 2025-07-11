from ultralytics import YOLO


def run_inference(weights_path, img_path, imgsz=640, conf=0.25):
    model = YOLO(weights_path)
    results = model(img_path, imgsz=imgsz, conf=conf)
    return results
