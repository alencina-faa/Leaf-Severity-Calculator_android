Image-processing refactor
=========================

Recent changes (numpy_only branch):

- Removed runtime dependency on OpenCV for the core image-processing pipeline.
  All channel arithmetic, masks and background-subtraction orchestration now use
  NumPy and the `numpy-rolling-ball` package.

- Pillow is retained for image input/output (JPEG/PNG decoding/encoding) and for
  high-quality resizing using Lanczos resampling.

Minimal recommended dependencies:

- Python 3.8+
- numpy
- pillow
- numpy-rolling-ball
- toga

Notes:
- If you rely on OpenCV-specific features, you can re-add OpenCV to `pyproject.toml`.
- The Android packaging previously included an OpenCV wheel; that entry was
  removed on this branch to reduce native dependency size. Add it back if needed.
