import importlib
import os
import sys
import types

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def _install_test_stubs():
    if "toga" not in sys.modules:
        toga = types.ModuleType("toga")

        class DummyApp:
            pass

        toga.App = DummyApp
        style_mod = types.ModuleType("toga.style")
        style_mod.Pack = lambda *args, **kwargs: None
        style_pack_mod = types.ModuleType("toga.style.pack")
        style_pack_mod.COLUMN = "COLUMN"
        style_pack_mod.ROW = "ROW"
        style_pack_mod.BOTTOM = "BOTTOM"
        style_pack_mod.CENTER = "CENTER"
        sys.modules["toga"] = toga
        sys.modules["toga.style"] = style_mod
        sys.modules["toga.style.pack"] = style_pack_mod

    if "tatogalib.uri_io.urifilebrowser" not in sys.modules:
        tatogalib = types.ModuleType("tatogalib")
        uri_io = types.ModuleType("tatogalib.uri_io")
        urifilebrowser = types.ModuleType("tatogalib.uri_io.urifilebrowser")
        urifile = types.ModuleType("tatogalib.uri_io.urifile")

        class UriFileBrowser:
            async def open_file_dialog(self, *args, **kwargs):
                return []

        class UriFile:
            def __init__(self, *args, **kwargs):
                pass

        urifilebrowser.UriFileBrowser = UriFileBrowser
        urifile.UriFile = UriFile
        sys.modules["tatogalib"] = tatogalib
        sys.modules["tatogalib.uri_io"] = uri_io
        sys.modules["tatogalib.uri_io.urifilebrowser"] = urifilebrowser
        sys.modules["tatogalib.uri_io.urifile"] = urifile

    if "cv2_rolling_ball" not in sys.modules:
        cv2_rolling_ball = types.ModuleType("cv2_rolling_ball")

        def subtract_background_rolling_ball(arr, *args, **kwargs):
            return arr, np.zeros_like(arr)

        cv2_rolling_ball.subtract_background_rolling_ball = subtract_background_rolling_ball
        sys.modules["cv2_rolling_ball"] = cv2_rolling_ball

    if "cv2" not in sys.modules:
        cv2 = types.ModuleType("cv2")
        cv2.COLOR_RGB2BGR = 1
        cv2.COLOR_BGR2RGB = 2
        cv2.INTER_AREA = 3
        cv2.INTER_CUBIC = 4

        def cvtColor(img, code):
            if code in (cv2.COLOR_RGB2BGR, cv2.COLOR_BGR2RGB):
                return img[..., ::-1]
            return img

        def split(img):
            return img[..., 0], img[..., 1], img[..., 2]

        def divide(a, b):
            return np.divide(a, b)

        def resize(img, dsize=None, fx=None, fy=None, interpolation=None):
            if dsize is not None:
                new_w, new_h = dsize
            else:
                new_w = max(int(img.shape[1] * fx), 1)
                new_h = max(int(img.shape[0] * fy), 1)
            pil = Image.fromarray(img)
            return np.array(pil.resize((new_w, new_h), resample=Image.Resampling.BILINEAR))

        def merge(channels):
            return np.stack(channels, axis=-1)

        def subtract(a, b):
            return np.clip(a.astype(np.int16) - b.astype(np.int16), 0, 255).astype(np.uint8)

        def bitwise_not(a):
            return 255 - a

        cv2.cvtColor = cvtColor
        cv2.split = split
        cv2.divide = divide
        cv2.resize = resize
        cv2.merge = merge
        cv2.subtract = subtract
        cv2.bitwise_not = bitwise_not

        sys.modules["cv2"] = cv2


def _build_app():
    _install_test_stubs()
    module = importlib.import_module("leafseveritycalculator.src.leafseveritycalculator.app")
    cls = module.LeafSeverityCalculator
    app = cls.__new__(cls)
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
