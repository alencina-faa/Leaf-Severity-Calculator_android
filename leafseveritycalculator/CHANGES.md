Image-processing refactor (package)
===================================

This package has been updated to use NumPy for image processing steps and
`numpy-rolling-ball` for background subtraction. Pillow remains required for
reading and writing image files and for resizing with Lanczos resampling.

Minimal recommended dependencies for the package:

- numpy
- pillow
- numpy-rolling-ball

If your deployment requires OpenCV (for other features), re-add the OpenCV wheel
or dependency to `pyproject.toml` and/or your packaging pipeline.
