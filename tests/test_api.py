"""API tests use a fake YOLO model; no checkpoint or GPU is required."""
import sys
import types

import cv2
import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


# src.api loads YOLO at import time. Stub the constructor before importing it
# so test collection does not require a downloaded model checkpoint.
fake_ultralytics = types.ModuleType("ultralytics")
fake_ultralytics.YOLO = lambda *args, **kwargs: None
sys.modules.setdefault("ultralytics", fake_ultralytics)

from src import api as shelf_api  # noqa: E402


class FakeTensor:
    def __init__(self, values):
        self.values = np.asarray(values)

    def cpu(self):
        return self

    def numpy(self):
        return self.values


class FakeBoxes:
    def __init__(self, confidences, coordinates):
        self.conf = FakeTensor(confidences)
        self.xyxy = FakeTensor(coordinates)


class FakeResult:
    def __init__(self, confidences=(0.9, 0.7), coordinates=((5, 5, 20, 20), (25, 5, 40, 20))):
        self.boxes = FakeBoxes(confidences, coordinates)


class FakeModel:
    def __init__(self, result=None, error=None):
        self.result = result if result is not None else FakeResult()
        self.error = error
        self.calls = 0

    def __call__(self, image_path, conf):
        self.calls += 1
        if self.error:
            raise self.error
        return [self.result]


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(shelf_api, "OUTPUT_DIR", str(tmp_path / "uploads"))
    monkeypatch.setattr(shelf_api, "ANNOTATED_DIR", str(tmp_path / "annotated"))
    monkeypatch.setattr(shelf_api, "CSV_DIR", str(tmp_path / "csv"))
    monkeypatch.setattr(shelf_api, "CSV_PATH", str(tmp_path / "csv" / "results.csv"))
    for directory in (shelf_api.OUTPUT_DIR, shelf_api.ANNOTATED_DIR, shelf_api.CSV_DIR):
        import os
        os.makedirs(directory, exist_ok=True)

    app = FastAPI()
    app.include_router(shelf_api.router)
    return TestClient(app)


@pytest.fixture()
def png_bytes():
    image = np.zeros((48, 64, 3), dtype=np.uint8)
    success, encoded = cv2.imencode(".png", image)
    assert success
    return encoded.tobytes()


def test_rejects_unsupported_file_extension(client, monkeypatch, png_bytes):
    model = FakeModel()
    monkeypatch.setattr(shelf_api, "model", model)

    response = client.post(
        "/upload/",
        files={"file": ("shelf.gif", png_bytes, "image/gif")},
    )

    assert response.status_code == 415
    assert "Unsupported image type" in response.text
    assert model.calls == 0


def test_rejects_empty_upload(client, monkeypatch):
    model = FakeModel()
    monkeypatch.setattr(shelf_api, "model", model)

    response = client.post(
        "/upload/",
        files={"file": ("empty.png", b"", "image/png")},
    )

    assert response.status_code == 400
    assert "empty" in response.text.lower()
    assert model.calls == 0


def test_rejects_invalid_image_bytes(client, monkeypatch):
    model = FakeModel()
    monkeypatch.setattr(shelf_api, "model", model)

    response = client.post(
        "/upload/",
        files={"file": ("not-really.png", b"not an image", "image/png")},
    )

    assert response.status_code == 400
    assert "valid image" in response.text
    assert model.calls == 0


def test_successful_inference_returns_count_and_saves_outputs(client, monkeypatch, png_bytes):
    model = FakeModel(FakeResult(confidences=(0.9, 0.7, 0.2),
                                 coordinates=((5, 5, 20, 20), (25, 5, 40, 20), (1, 1, 3, 3))))
    monkeypatch.setattr(shelf_api, "model", model)

    response = client.post(
        "/upload/",
        files={"file": ("shelf.png", png_bytes, "image/png")},
    )

    assert response.status_code == 200
    assert "Count:</strong> 2" in response.text
    assert "/static/annotated/annotated_shelf.png" in response.text
    assert (tmp_path_path(client, "annotated_shelf.png")).exists()
    assert (tmp_path_path(client, "results.csv")).exists()
    assert model.calls == 1


def tmp_path_path(client, filename):
    """Resolve output paths from the fixture's monkeypatched module constants."""
    if filename == "results.csv":
        return __import__("pathlib").Path(shelf_api.CSV_PATH)
    return __import__("pathlib").Path(shelf_api.ANNOTATED_DIR) / filename


def test_empty_detection_output_returns_zero_count(client, monkeypatch, png_bytes):
    model = FakeModel(FakeResult(confidences=(), coordinates=()))
    monkeypatch.setattr(shelf_api, "model", model)

    response = client.post(
        "/upload/",
        files={"file": ("empty-shelf.png", png_bytes, "image/png")},
    )

    assert response.status_code == 200
    assert "Count:</strong> 0" in response.text
    assert model.calls == 1


def test_model_exception_returns_generic_500_without_internal_details(client, monkeypatch, png_bytes):
    model = FakeModel(error=RuntimeError("private model path /secret/checkpoint.pt"))
    monkeypatch.setattr(shelf_api, "model", model)

    response = client.post(
        "/upload/",
        files={"file": ("shelf.png", png_bytes, "image/png")},
    )

    assert response.status_code == 500
    assert "Unable to process this image" in response.text
    assert "/secret/checkpoint.pt" not in response.text
