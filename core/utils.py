# core/utils.py
"""
Shared utilities — image listing, path helpers.
No model logic. No pipeline loops.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Set


IMAGE_EXTS: Set[str] = {".jpg", ".jpeg", ".png", ".bmp"}


def list_images(directory: Path, exts: Set[str] = IMAGE_EXTS) -> List[Path]:
    """List image files in a single directory (non-recursive)."""
    if not directory.exists():
        return []
    return sorted(
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in exts
    )


def list_images_recursive(directory: Path, exts: Set[str] = IMAGE_EXTS) -> List[Path]:
    """Recursively list all image files under directory."""
    if not directory.exists():
        return []
    return sorted(
        p for p in directory.rglob("*")
        if p.is_file() and p.suffix.lower() in exts
    )
