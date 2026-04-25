#!/usr/bin/env python3
"""Regenerate all icon files in this folder from logoapp-1920.png.

This script preserves each destination file's key characteristics by reading the
existing file before overwriting it:
- File name and extension
- Pixel size
- DPI metadata (for PNG files that currently contain it)
- Shape/transparency mask for files that already have alpha transparency
- Container format for ICO/ICNS
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

from PIL import Image, ImageOps

ICON_EXTENSIONS = {".png", ".ico", ".icns"}
DEFAULT_SOURCE = "logoapp-1920.png"


def _iter_targets(icon_dir: Path, source_name: str) -> Iterable[Path]:
    for path in sorted(icon_dir.iterdir()):
        if not path.is_file():
            continue
        if path.name == source_name:
            continue
        if path.suffix.lower() in ICON_EXTENSIONS:
            yield path


def _fit_rgba(src_rgba: Image.Image, size: Tuple[int, int]) -> Image.Image:
    return ImageOps.fit(src_rgba, size, method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))


def _extract_alpha_mask(template: Image.Image) -> Optional[Image.Image]:
    rgba = template.convert("RGBA")
    alpha = rgba.getchannel("A")
    extrema = alpha.getextrema()
    if extrema == (255, 255):
        return None
    return alpha


def _save_png_from_template(src_rgba: Image.Image, target: Path) -> None:
    with Image.open(target) as template:
        size = template.size
        dpi = template.info.get("dpi")
        mask = _extract_alpha_mask(template)

    output = _fit_rgba(src_rgba, size)
    if mask is not None:
        output.putalpha(mask)

    save_kwargs = {"format": "PNG", "optimize": True}
    if dpi is not None:
        save_kwargs["dpi"] = dpi
    output.save(target, **save_kwargs)


def _ico_sizes(template_path: Path) -> Sequence[Tuple[int, int]]:
    with Image.open(template_path) as ico_img:
        sizes = []
        if hasattr(ico_img, "ico") and ico_img.ico is not None:
            sizes = sorted(ico_img.ico.sizes())
        if not sizes:
            sizes = [ico_img.size]
        return sizes


def _save_ico_from_template(src_rgba: Image.Image, target: Path) -> None:
    sizes = _ico_sizes(target)
    largest = max(sizes)
    base = _fit_rgba(src_rgba, largest)
    base.save(target, format="ICO", sizes=sizes)


def _save_icns_from_template(src_rgba: Image.Image, target: Path) -> None:
    # Pillow's ICNS writer generates the container representation. We preserve
    # output container format and use a high-resolution source frame.
    # Existing files in this project use 256x256 as primary display size.
    with Image.open(target) as template:
        base_size = template.size

    frame = _fit_rgba(src_rgba, base_size)
    frame.save(target, format="ICNS")


def regenerate(icon_dir: Path, source_name: str, dry_run: bool) -> None:
    source_path = icon_dir / source_name
    if not source_path.exists():
        raise FileNotFoundError(f"Source image not found: {source_path}")

    with Image.open(source_path) as src:
        src_rgba = src.convert("RGBA")

    targets = list(_iter_targets(icon_dir, source_name))
    if not targets:
        print("No target icon files found.")
        return

    print(f"Source: {source_path.name} ({src_rgba.size[0]}x{src_rgba.size[1]})")
    print(f"Targets: {len(targets)}")

    for target in targets:
        suffix = target.suffix.lower()
        print(f"- {'Would update' if dry_run else 'Updating'} {target.name}")
        if dry_run:
            continue

        if suffix == ".png":
            _save_png_from_template(src_rgba, target)
        elif suffix == ".ico":
            _save_ico_from_template(src_rgba, target)
        elif suffix == ".icns":
            _save_icns_from_template(src_rgba, target)

    print("Done.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate icon assets from logoapp-1920.png")
    parser.add_argument(
        "--icons-dir",
        default=".",
        help="Directory containing icon files (default: current directory)",
    )
    parser.add_argument(
        "--source",
        default=DEFAULT_SOURCE,
        help=f"Source image file name (default: {DEFAULT_SOURCE})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be updated without writing files",
    )
    args = parser.parse_args()

    regenerate(Path(args.icons_dir).resolve(), args.source, args.dry_run)


if __name__ == "__main__":
    main()
