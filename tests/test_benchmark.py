"""Unit tests for benchmark helpers; no model checkpoint or GPU is required."""
import cv2
import numpy as np

from scripts.benchmark import collect_images, summarize_latencies


def test_latency_summary_reports_expected_statistics():
    summary = summarize_latencies([10.0, 20.0, 30.0])

    assert summary["runs"] == 3
    assert summary["mean_ms"] == 20.0
    assert summary["median_ms"] == 20.0
    assert summary["min_ms"] == 10.0
    assert summary["max_ms"] == 30.0
    assert summary["throughput_images_per_second"] == 50.0


def test_collect_images_uses_sorted_order_and_maximum(tmp_path):
    for name in ("c.png", "a.png", "b.png"):
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        ok, encoded = cv2.imencode(".png", image)
        assert ok
        (tmp_path / name).write_bytes(encoded.tobytes())

    paths, unreadable = collect_images(tmp_path, max_images=2)

    assert [path.name for path in paths] == ["a.png", "b.png"]
    assert unreadable == []


def test_collect_images_skips_unreadable_files(tmp_path):
    (tmp_path / "bad.png").write_bytes(b"not an image")
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    (tmp_path / "good.png").write_bytes(encoded.tobytes())

    paths, unreadable = collect_images(tmp_path, max_images=0)

    assert [path.name for path in paths] == ["good.png"]
    assert unreadable == [str(tmp_path / "bad.png")]
