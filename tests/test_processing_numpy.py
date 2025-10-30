import sys
import os
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from leafseveritycalculator.src.leafseveritycalculator.app import LeafSeverityCalculator


def test_resize_and_processing():
    # create instance without initializing Toga internals
    app = LeafSeverityCalculator.__new__(LeafSeverityCalculator)
    app.ui_inicial = -0.03365811811
    app.ub_inicial = 185

    # Build a synthetic image with a green and red region
    h, w = 300, 400
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    arr[50:200, 100:250] = [34, 200, 34]  # green
    arr[100:250, 200:350] = [200, 34, 34]  # red
    img = Image.fromarray(arr)

    app.img_original = img

    # Call processing
    result = app._process_image_opencv()
    assert result is not None
    processed_img, severity = result
    assert isinstance(processed_img, Image.Image)
    assert 0.0 <= float(severity) <= 1.0

    # Test resize: ensure that resizing to a smaller dimension reduces size
    small = app._resize_image_numpy(arr, 100)
    assert small.shape[0] <= arr.shape[0]
    assert small.shape[1] <= arr.shape[1]
