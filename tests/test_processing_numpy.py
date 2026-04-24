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

    if "numpy_rolling_ball" not in sys.modules:
        numpy_rolling_ball = types.ModuleType("numpy_rolling_ball")

        def subtract_background_rolling_ball(arr, *args, **kwargs):
            return arr, np.zeros_like(arr)

        numpy_rolling_ball.subtract_background_rolling_ball = subtract_background_rolling_ball
        sys.modules["numpy_rolling_ball"] = numpy_rolling_ball


def _build_app():
    _install_test_stubs()
    module = importlib.import_module("leafseveritycalculator.src.leafseveritycalculator.app")
    cls = module.LeafSeverityCalculator
    app = cls.__new__(cls)
    app.ui_inicial = -0.03365811811
    app.ub_inicial = 185
    app.cache = {}
    return app


def test_numpy_processing_returns_image_and_severity_range():
    app = _build_app()

    arr = np.zeros((300, 400, 3), dtype=np.uint8)
    arr[50:200, 100:250] = [34, 200, 34]
    arr[100:250, 200:350] = [200, 34, 34]
    app.img_original = Image.fromarray(arr)

    processed_img, severity = app._process_image_opencv()
    assert isinstance(processed_img, Image.Image)
    assert 0.0 <= float(severity) <= 1.0


def test_numpy_resize_respects_max_dimension():
    app = _build_app()
    arr = np.zeros((300, 400, 3), dtype=np.uint8)

    small = app._resize_image_numpy(arr, 100)
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
