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
    assert app.main_window.title == "Calculadora de Severidad de Hojas"
    assert app.main_window.shown is True
    assert app.severity_button.enabled is False
    assert app.progress_label.text == ""
    assert app.lbl_severidad.text == ""


def test_inicio_resets_visual_state():
    app = build_app()
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
    app = build_app()
    app.startup()

    app.mostrar_ayuda(None)

    assert len(app.main_window.info_calls) == 1
    title, message = app.main_window.info_calls[0]
    assert title == "About This App"
    assert "calculates the leaf severity" in message
    assert "healthy leaf portion (green)" in message


def test_procesar_imagen_updates_result_state():
    app = build_app()
    app.startup()
    module = load_app_module()

    processed_image = PILImage.new("RGB", (8, 8), color=(255, 0, 0))

    app._process_image_detailed = lambda: (processed_image, 0.25)
    original_get_event_loop = module.asyncio.get_event_loop
    module.asyncio.get_event_loop = lambda: FakeExecutorLoop()
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
    app = build_app()
    app.startup()
    module = load_app_module()

    original_get_event_loop = module.asyncio.get_event_loop
    module.asyncio.get_event_loop = lambda: FakeExecutorLoop(error=RuntimeError("fallo controlado"))
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
    app = build_app()
    app.startup()

    asyncio.run(app.guardar_imagen(None))

    assert len(app.main_window.dialog_calls) == 1
    dialog = app.main_window.dialog_calls[0]
    assert dialog.title == "Advertencia"
    assert "No hay imagen procesada" in dialog.message


def test_guardar_imagen_writes_png_and_reports_success():
    app = build_app()
    app.startup()
    module = load_app_module()
    import io as _io
    _buf = _io.BytesIO()
    PILImage.new("RGB", (4, 4), color=(0, 255, 0)).save(_buf, format="PNG")
    app.img_procesada = _buf.getvalue()
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
