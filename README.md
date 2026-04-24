# Leaf Severity Calculator - Main Branch

**Version**: 0.0.9a1 | **Stack**: Python 3.12 + Briefcase 0.3.22 | **Status**: ✅ Production Ready

## Overview

A cross-platform Android application that analyzes barley leaf images to calculate disease severity. The app segments leaves into background, healthy tissue (green), and diseased tissue (red), then computes the severity percentage.

### Key Features

- 📸 **Capture & Upload**: Take photos with device camera or select images from gallery
- 🔬 **Intelligent Segmentation**: Uses Otsu algorithm (blue band) and K-means ((red-green)/(red+green) index)
- 📊 **Severity Calculation**: Real-time calculation of disease percentage
- 💾 **Save Results**: Export processed images with severity metrics to device storage
- 🌐 **Bilingual Ready**: Full English USA UI (with Spanish versions available in other branches)
- ⚡ **Modern Stack**: Python 3.12, latest PyPI dependencies

## Technology Stack

| Component | Version | Source |
|-----------|---------|--------|
| Python | 3.12 | System |
| Briefcase | 0.3.22 | PyPI |
| toga-android | ~0.4.5 | PyPI |
| NumPy | Latest | PyPI |
| Pillow | Latest | PyPI |
| opencv-rolling-ball | Latest | PyPI |
| numpy-rolling-ball | >=1,<2 | PyPI |
| tatogalib | Latest | Local wheel |

## Project Structure

```
leafseveritycalculator/
├── src/leafseveritycalculator/
│   ├── __init__.py
│   ├── app.py                      # Main application logic
│   ├── __main__.py
│   └── resources/                  # App resources
├── android/
│   └── app_template/               # Android-specific templates
│       └── src/main/
│           ├── AndroidManifest.xml
│           └── res/                # Drawable, layout, strings, etc.
├── tests/                          # Test suite
├── wheels/                         # Universal wheels (tatogalib)
└── pyproject.toml                  # Project configuration

```

## Getting Started

### Prerequisites

- **Python 3.12** (Windows, macOS, or Linux)
- **Android SDK** (API level 24+)
- **Android NDK**
- **Java Development Kit (JDK)** 11+
- **Briefcase** 0.3.22

### Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/alencina-faa/Leaf-Severity-Calculator_android.git
   cd Leaf-Severity-Calculator_android
   git checkout main  # Ensure you're on the main branch
   ```

2. **Create a virtual environment**:
   ```bash
   python3.12 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install --upgrade pip setuptools wheel
   pip install briefcase==0.3.22
   ```

### Building for Android

#### Development/Debug Build

```bash
cd leafseveritycalculator

# Initial setup (one-time)
python -m briefcase create android --no-input

# Build debug APK
python -m briefcase build android

# Package debug APK
python -m briefcase package android
```

**Output**: `dist/LeafSeverityCalculator-0.0.9a1.apk`

#### Release Build

```bash
cd leafseveritycalculator

# Update dependencies
python -m briefcase update android -r

# Package release AAB (for Google Play)
python -m briefcase package android
```

**Output**: `dist/LeafSeverityCalculator-0.0.9a1.aab`

### Running on Device

1. **Transfer APK to Android device** (or deploy directly if connected):
   ```bash
   adb install dist/LeafSeverityCalculator-0.0.9a1.apk
   ```

2. **Grant permissions**:
   - Camera access (prompted on first use)
   - Storage access (for saving results)

3. **Launch the app** from device app drawer

## Key Dependencies Explained

### toga-android
BeeWare's native Android UI toolkit. Provides cross-platform widgets that render as native Android components.

### numpy & Pillow
Used for image processing:
- NumPy: Array operations, channel extraction, mathematical operations
- Pillow: Image I/O, resizing, format conversion

### numpy-rolling-ball
Background illumination correction using rolling-ball algorithm (subtracts background light to normalize images).

### opencv-rolling-ball
OpenCV-based rolling ball filter implementation (used in some processing pipelines).

### tatogalib
URI file browser and Android file system integration (allows selecting images from device gallery).

## Development Workflow

### Running Tests

```bash
cd leafseveritycalculator
python -m pytest tests/
```

### Code Structure

**Main Application** (`src/leafseveritycalculator/app.py`):
- `LeafSeverityCalculator`: Main app class
- `startup()`: Initialize UI and handlers
- `take_photo()`: Camera integration
- `open_image()`: Gallery file picker
- `extract_background_color()`: Illumination correction
- `procesar_imagen()`: Image segmentation and severity calculation
- `guardar_imagen()`: Save results to device storage

### UI Components

- **Buttons**: Take Photo, Select Image, Calculate Severity
- **Image Viewers**: Input photo, processed result
- **Progress Label**: Status updates during processing
- **Result Label**: Displays severity percentage
- **Icon Buttons**: Home, Save, Help, Exit
- **Institutional Logos**: UCEVA, FAA, CIC

## Android Permissions

Defined in `android/app_template/src/main/AndroidManifest.xml`:

```xml
<uses-permission android:name="android.permission.CAMERA" />
<uses-permission android:name="android.permission.READ_EXTERNAL_STORAGE" />
<uses-permission android:name="android.permission.WRITE_EXTERNAL_STORAGE" />
```

## Building Troubleshooting

### Issue: "Version number not valid"
**Solution**: Ensure version in `pyproject.toml` is PEP 440 compliant (e.g., `0.0.9a1`, not `0.0.9a`)

### Issue: Module not found errors
**Solution**: Ensure all dependencies are installed via pip:
```bash
pip install briefcase toga-android numpy pillow opencv-rolling-ball numpy-rolling-ball tatogalib
```

### Issue: Android SDK/NDK not found
**Solution**: Set environment variables:
```bash
export ANDROID_HOME=$HOME/Android/Sdk
export ANDROID_SDK_ROOT=$ANDROID_HOME
export PATH=$PATH:$ANDROID_HOME/tools:$ANDROID_HOME/platform-tools
```

## Deployment to Google Play

1. **Prepare AAB** (already done):
   ```bash
   python -m briefcase package android
   ```

2. **Sign release** (configured in `pyproject.toml`)

3. **Upload to Play Console**:
   - Navigate to [Google Play Console](https://play.google.com/console)
   - Select LeafSeverityCalculator app
   - Go to **Release** → **Production**
   - Upload `dist/LeafSeverityCalculator-0.0.9a1.aab`
   - Add release notes
   - Review and publish

## Environment Variables

For CI/CD workflows:

```bash
ANDROID_HOME         # Path to Android SDK
ANDROID_SDK_ROOT     # Alternative Android SDK path
ANDROID_NDK_ROOT     # Path to Android NDK
JAVA_HOME            # Path to JDK installation
```

## Version Information

- **Current**: 0.0.9a1 (Alpha Release)
- **Branch**: main (Production-ready modern stack)
- **Python**: 3.12.x
- **Briefcase**: 0.3.22
- **Last Updated**: April 24, 2026

## Related Branches

- **OpenCV-based**: Legacy Python 3.8 stack for broader device compatibility
  - Better support for older Android devices
  - Uses pre-built cp38 wheels
  - See `OpenCV-based` branch README

## Contributing

1. Create feature branch from `main`
2. Make changes and test locally
3. Update version in `pyproject.toml` (PEP 440 format)
4. Commit with descriptive messages
5. Create pull request with details

## License

See [LICENSE](../LICENSE) file.

## Authors

- Emiliano David
- Alberto Lencina
- Luisa Cabezas

**Institution**: Facultad de Agronomía, Universidad Nacional del Centro de la Provincia de Buenos Aires (UNICEN)

**Email**: alencina@azul.faa.unicen.edu.ar

## Additional Resources

- [BeeWare Briefcase Documentation](https://briefcase.readthedocs.io/)
- [Toga UI Toolkit](https://toga.readthedocs.io/)
- [Android Developer Guide](https://developer.android.com/guide)
- [Python for Android](https://python-for-android.readthedocs.io/)
- [Chaquopy (Python-Android build system)](https://chaquo.com/chaquopy/)

---

**Status**: ✅ Production Ready | **Last Build**: LeafSeverityCalculator-0.0.9a1.aab (43.21 MB)
