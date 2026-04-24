import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "leafseveritycalculator")))

from tests._support import build_app


def _build_app():
    app = build_app()
    app.ui_inicial = -0.03365811811
    app.ub_inicial = 185
    app.cache = {}
    return app


def test_opencv_processing_returns_image_and_severity_range():
    app = _build_app()

    arr = np.full((300, 400, 3), 255, dtype=np.uint8)
    arr[50:200, 100:250] = [34, 200, 34]
    arr[100:250, 200:350] = [200, 34, 34]
    app.img_original = Image.fromarray(arr)

    processed_img, severity = app._process_image_opencv()
    assert isinstance(processed_img, Image.Image)
    assert 0.0 <= float(severity) <= 1.0

    processed_arr = np.array(processed_img)
    assert np.array_equal(processed_arr[120, 220], np.array([0, 255, 0], dtype=np.uint8))
    assert np.array_equal(processed_arr[440, 600], np.array([255, 0, 0], dtype=np.uint8))
    assert np.array_equal(processed_arr[20, 20], np.array([0, 0, 0], dtype=np.uint8))


def test_opencv_resize_respects_max_dimension():
    app = _build_app()
    arr = np.zeros((300, 400, 3), dtype=np.uint8)

    small = app._resize_image(arr, 100)
    assert max(small.shape[:2]) <= 100


def test_detailed_processing_uses_cache():
    app = _build_app()
    app.img_original = Image.fromarray(np.full((10, 12, 3), 128, dtype=np.uint8))

    calls = {"count": 0}

    def fake_process():
        calls["count"] += 1
        return "ok", 0.5

    app._process_image_opencv = fake_process

    first = app._process_image_detailed()
    second = app._process_image_detailed()

    assert first == second
    assert calls["count"] == 1
