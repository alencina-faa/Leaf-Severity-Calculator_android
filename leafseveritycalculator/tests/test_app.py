import asyncio
import builtins
from PIL import Image as PILImage
from ._support import FakeExecutorLoop, build_app, load_app_module


def test_startup_initializes_state_without_real_ui():
    app = build_app()

    app.startup()

    assert app.img_original is None
    assert app.img_procesada is None
    assert app.severidad == 0
    assert app.processing is False
    assert app.cache == {}
    assert app.ui_inicial == -0.03365811811
    assert app.ub_inicial == 185
    assert app.main_window.title == "Leaf Severity Calculator"
    assert app.main_window.shown is True
    assert app.severity_button.enabled is False
    assert app.progress_label.text == ""
    assert app.lbl_severidad.text == ""


def test_go_home_resets_visual_state():
    app = build_app()
    app.startup()

    app.photo.image = object()
    app.progress_label.text = "Processing"
    app.severity_button.enabled = True
    app.result.image = object()
    app.lbl_severidad.text = "Severity: 30%"

    app.go_home(None)

    assert app.photo.image is None
    assert app.progress_label.text == ""
    assert app.severity_button.enabled is False
    assert app.result.image is None
    assert app.lbl_severidad.text == ""


def test_show_help_displays_expected_copy():
    app = build_app()
    app.startup()

    app.show_help(None)

    assert len(app.main_window.info_calls) == 1
    title, message = app.main_window.info_calls[0]
    assert title == "About This App"
    assert "calculates the leaf severity" in message
    assert "healthy leaf portion (green)" in message


def test_process_image_updates_result_state():
    app = build_app()
    app.startup()
    module = load_app_module()

    processed_image = PILImage.new("RGB", (8, 8), color=(255, 0, 0))

    app._process_image_detailed = lambda: (processed_image, 0.25)
    original_get_event_loop = module.asyncio.get_event_loop
    module.asyncio.get_event_loop = lambda: FakeExecutorLoop()
    try:
        asyncio.run(app.process_image(None))
    finally:
        module.asyncio.get_event_loop = original_get_event_loop

    assert app.img_procesada is processed_image
    assert app.result.image.src is processed_image
    assert app.severidad == 0.25
    assert app.lbl_severidad.text == "Severity: 25.00%"
    assert app.severity_button.enabled is False
    assert app.processing is False


def test_process_image_reports_processing_errors():
    app = build_app()
    app.startup()
    module = load_app_module()

    original_get_event_loop = module.asyncio.get_event_loop
    module.asyncio.get_event_loop = lambda: FakeExecutorLoop(error=RuntimeError("controlled failure"))
    try:
        asyncio.run(app.process_image(None))
    finally:
        module.asyncio.get_event_loop = original_get_event_loop

    assert app.processing is False
    assert len(app.main_window.dialog_calls) == 1
    dialog = app.main_window.dialog_calls[0]
    assert dialog.title == "Error"
    assert "controlled failure" in dialog.message


def test_save_image_warns_when_there_is_no_processed_image():
    app = build_app()
    app.startup()

    asyncio.run(app.save_image(None))

    assert len(app.main_window.dialog_calls) == 1
    dialog = app.main_window.dialog_calls[0]
    assert dialog.title == "Warning"
    assert "No processed image" in dialog.message


def test_save_image_writes_png_and_reports_success():
    app = build_app()
    app.startup()
    module = load_app_module()
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
        asyncio.run(app.save_image(None))
    finally:
        module.os.makedirs = original_makedirs
        module.time.strftime = original_strftime
        builtins.open = original_open

    assert written["makedirs"] == ("/sdcard/Download/LeafSeverityImages", True)
    assert written["path"].endswith("20260424_Severity_25.00%.png")
    assert written["data"].startswith(b"\x89PNG")
    assert len(app.main_window.dialog_calls) == 1
    dialog = app.main_window.dialog_calls[0]
    assert dialog.title == "Success"
    assert written["path"] in dialog.message
