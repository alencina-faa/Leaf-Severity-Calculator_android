import importlib
import asyncio
import builtins
import io
import sys
import types

import numpy as np
from PIL import Image as PILImage


def _install_test_stubs():
    if "toga" not in sys.modules:
        toga = types.ModuleType("toga")

        class DummyApp:
            pass

        class DummyMainWindow:
            def __init__(self, title=None):
                self.title = title
                self.content = None
                self.shown = False
                self.info_calls = []
                self.dialog_calls = []

            def show(self):
                self.shown = True

            def info_dialog(self, title, message):
                self.info_calls.append((title, message))

            async def dialog(self, dialog):
                self.dialog_calls.append(dialog)
                return dialog

        class DummyBox:
            def __init__(self, *args, **kwargs):
                self.children = []
                self.style = kwargs.get("style")

            def add(self, child):
                self.children.append(child)

        class DummyScrollContainer:
            def __init__(self, content=None, **kwargs):
                self.content = content

        class DummyButton:
            def __init__(self, text=None, on_press=None, style=None, enabled=True, icon=None, **kwargs):
                self.text = text
                self.on_press = on_press
                self.style = style
                self.enabled = enabled
                self.icon = icon

        class DummyImageView:
            def __init__(self, image=None, style=None, **kwargs):
                self.image = image
                self.style = style

        class DummyLabel:
            def __init__(self, text="", style=None, **kwargs):
                self.text = text
                self.style = style

        class DummyImage:
            def __init__(self, src=None, **kwargs):
                self.src = src

        class DummyInfoDialog:
            def __init__(self, title, message):
                self.title = title
                self.message = message

        toga.App = DummyApp
        toga.MainWindow = DummyMainWindow
        toga.Box = DummyBox
        toga.ScrollContainer = DummyScrollContainer
        toga.Button = DummyButton
        toga.ImageView = DummyImageView
        toga.Label = DummyLabel
        toga.Image = DummyImage
        toga.InfoDialog = DummyInfoDialog

        style_mod = types.ModuleType("toga.style")
        style_mod.Pack = lambda *args, **kwargs: {"args": args, "kwargs": kwargs}

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
            pil = PILImage.fromarray(img)
            return np.array(pil.resize((new_w, new_h), resample=PILImage.Resampling.BILINEAR))

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
    return app


def test_startup_initializes_state_without_real_ui():
    app = _build_app()

    app.startup()

    assert app.img_original is None
    assert app.img_procesada is None
    assert app.severidad == 0
    assert app.processing is False
    assert app.cache == {}
    assert app.ui_inicial == -0.03365811811
    assert app.ub_inicial == 185
    assert app.main_window.title == "Calculadora de Severidad de Hojas"
    assert app.main_window.shown is True
    assert app.severity_button.enabled is False
    assert app.progress_label.text == ""
    assert app.lbl_severidad.text == ""


def test_inicio_resets_visual_state():
    app = _build_app()
    app.startup()

    app.photo.image = object()
    app.progress_label.text = "Procesando"
    app.severity_button.enabled = True
    app.result.image = object()
    app.lbl_severidad.text = "Severidad: 30%"

    app.inicio(None)

    assert app.photo.image is None
    assert app.progress_label.text == ""
    assert app.severity_button.enabled is False
    assert app.result.image is None
    assert app.lbl_severidad.text == ""


def test_mostrar_ayuda_displays_expected_copy():
    app = _build_app()
    app.startup()

    app.mostrar_ayuda(None)

    assert len(app.main_window.info_calls) == 1
    title, message = app.main_window.info_calls[0]
    assert title == "About This App"
    assert "calculates the leaf severity" in message
    assert "healthy leaf portion (green)" in message


def test_procesar_imagen_updates_result_state():
    app = _build_app()
    app.startup()
    module = importlib.import_module("leafseveritycalculator.src.leafseveritycalculator.app")

    processed_image = PILImage.new("RGB", (8, 8), color=(255, 0, 0))

    class FakeLoop:
        async def run_in_executor(self, executor, func):
            return func()

    app._process_image_detailed = lambda: (processed_image, 0.25)
    original_get_event_loop = module.asyncio.get_event_loop
    module.asyncio.get_event_loop = lambda: FakeLoop()
    try:
        asyncio.run(app.procesar_imagen(None))
    finally:
        module.asyncio.get_event_loop = original_get_event_loop

    assert app.img_procesada is processed_image
    assert app.result.image.src is processed_image
    assert app.severidad == 0.25
    assert app.lbl_severidad.text == "Severidad: 25.00%"
    assert app.severity_button.enabled is False
    assert app.processing is False


def test_procesar_imagen_reports_processing_errors():
    app = _build_app()
    app.startup()
    module = importlib.import_module("leafseveritycalculator.src.leafseveritycalculator.app")

    class FakeLoop:
        async def run_in_executor(self, executor, func):
            raise RuntimeError("fallo controlado")

    original_get_event_loop = module.asyncio.get_event_loop
    module.asyncio.get_event_loop = lambda: FakeLoop()
    try:
        asyncio.run(app.procesar_imagen(None))
    finally:
        module.asyncio.get_event_loop = original_get_event_loop

    assert app.processing is False
    assert len(app.main_window.dialog_calls) == 1
    dialog = app.main_window.dialog_calls[0]
    assert dialog.title == "Error"
    assert "fallo controlado" in dialog.message


def test_guardar_imagen_warns_when_there_is_no_processed_image():
    app = _build_app()
    app.startup()

    asyncio.run(app.guardar_imagen(None))

    assert len(app.main_window.dialog_calls) == 1
    dialog = app.main_window.dialog_calls[0]
    assert dialog.title == "Advertencia"
    assert "No hay imagen procesada" in dialog.message


def test_guardar_imagen_writes_png_and_reports_success():
    app = _build_app()
    app.startup()
    module = importlib.import_module("leafseveritycalculator.src.leafseveritycalculator.app")
    app.img_procesada = PILImage.new("RGB", (4, 4), color=(0, 255, 0))
    app.severidad = 0.25

    written = {"path": None, "data": b"", "makedirs": None}

    class FakeFile:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def write(self, data):
            written["data"] += data

    original_makedirs = module.os.makedirs
    original_strftime = module.time.strftime
    original_open = builtins.open

    def fake_makedirs(path, exist_ok=False):
        written["makedirs"] = (path, exist_ok)

    def fake_open(path, mode):
        written["path"] = path
        assert mode == "wb"
        return FakeFile()

    module.os.makedirs = fake_makedirs
    module.time.strftime = lambda fmt: "20260424"
    builtins.open = fake_open
    try:
        asyncio.run(app.guardar_imagen(None))
    finally:
        module.os.makedirs = original_makedirs
        module.time.strftime = original_strftime
        builtins.open = original_open

    assert written["makedirs"] == ("/sdcard/Download/LeafSeverityImages", True)
    assert written["path"].endswith("20260424_Severidad_25.00%.png")
    assert written["data"].startswith(b"\x89PNG")
    assert len(app.main_window.dialog_calls) == 1
    dialog = app.main_window.dialog_calls[0]
    assert dialog.title == "Éxito"
    assert written["path"] in dialog.message
