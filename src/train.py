from ultralytics import YOLO


def train_model(
    model_name,
    data_yaml,
    epochs,
    batch_size,
    img_size,
    project,
    run_name,
    workers=2,
    cache=False,
):
    model = YOLO(model_name)

    model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        project=project,
        name=run_name,
        cache=cache,
        exist_ok=True,
        workers=workers,
    )
