import importlib
import sys
import types

import numpy as np
from PIL import Image as PILImage


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


class FakeExecutorLoop:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error

    async def run_in_executor(self, executor, func):
        if self.error is not None:
            raise self.error
        if callable(self.result):
            return self.result()
        if self.result is not None:
            return self.result
        return func()


def install_test_stubs():
    if "toga" not in sys.modules:
        toga = types.ModuleType("toga")
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


def load_app_module():
    install_test_stubs()
    candidates = [
        "leafseveritycalculator.src.leafseveritycalculator.app",
        "src.leafseveritycalculator.app",
        "leafseveritycalculator.app",
    ]
    for module_name in candidates:
        try:
            return importlib.import_module(module_name)
        except ModuleNotFoundError:
            continue
    raise ModuleNotFoundError(
        "Could not import app module from any known path: " + ", ".join(candidates)
    )


def build_app():
    module = load_app_module()
    cls = module.LeafSeverityCalculator
    app = cls.__new__(cls)
    return app
