"""
Dataset Manager for MediScan-AI
================================
จัดการโครงสร้างไฟล์ dataset 2 รูปแบบ:
  1. Nested:  {root}/{main_class}/{sub_class}/train/good
  2. Flat:    {root}/{main_class}_{sub_class}/train/good

Supports:
  • Create folders (both formats)
  • List images / subclasses
  • Resolve path from class name
  • Auto-detect structure type
"""
from __future__ import annotations
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum, auto


class StructureType(Enum):
    """Dataset folder structure type."""
    NESTED = auto()    # {root}/{main}/{sub}/...
    FLAT = auto()      # {root}/{main_sub}/...
    AUTO = auto()      # Detect from existing folders


@dataclass
class ClassPath:
    """Represents a resolved class path with metadata."""
    main_class: str
    sub_class: str
    path: Path
    structure: StructureType
    train_good: Path = field(init=False)
    train_bad: Path = field(init=False)
    test_good: Path = field(init=False)
    test_bad: Path = field(init=False)
    
    def __post_init__(self):
        """Auto-resolve standard subfolders."""
        self.train_good = self.path / "train" / "good"
        self.train_bad  = self.path / "train" / "bad"
        self.test_good  = self.path / "test" / "good"
        self.test_bad   = self.path / "test" / "bad"
    
    def exists(self) -> bool:
        """Check if class folder exists."""
        return self.path.exists()
    
    def count_images(self, split: str = "train", label: str = "good", 
                     exts: Optional[set] = None) -> int:
        """Count images in a specific split/label folder."""
        exts = exts or {".jpg", ".jpeg", ".png", ".bmp"}
        folder = {
            ("train", "good"): self.train_good,
            ("train", "bad"):  self.train_bad,
            ("test", "good"):  self.test_good,
            ("test", "bad"):   self.test_bad,
        }.get((split, label))
        
        if not folder or not folder.exists():
            return 0
        return sum(1 for f in folder.iterdir() 
                   if f.is_file() and f.suffix.lower() in exts)


class DatasetManager:
    """
    Unified manager for pill dataset in nested or flat structure.
    
    Usage
    -----
    >>> mgr = DatasetManager(root="data_train_defection")
    
    # Create nested structure
    >>> mgr.create("paracap", "front", structure="nested")
    
    # Create flat structure  
    >>> mgr.create("vitaminc", "back", structure="flat")
    
    # Auto-detect and list
    >>> mgr.list_classes()                    # [('paracap', 'front'), ...]
    >>> mgr.get_class_path("paracap", "front")  # ClassPath object
    
    # List images
    >>> mgr.list_images("paracap", "front", split="train", label="good")
    """
    
    IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    NESTED_PATTERN = "{main}/{sub}"
    FLAT_PATTERN = "{main}_{sub}"
    
    def __init__(
        self,
        root: Union[str, Path],
        image_exts: Optional[set] = None,
        auto_create: bool = False,
    ):
        """
        Parameters
        ----------
        root : str | Path
            Base directory for dataset.
        image_exts : set, optional
            Valid image extensions (default: common formats).
        auto_create : bool
            Auto-create parent folders when accessing paths.
        """
        self.root = Path(root).resolve()
        self.image_exts = image_exts or self.IMAGE_EXTS
        self.auto_create = auto_create
        
        # Cache for detected structure
        self._detected_structure: Optional[StructureType] = None
        self._class_cache: Dict[Tuple[str, str], ClassPath] = {}
    
    # ─────────────────── create folders ───────────────────
    def create(
        self,
        main_class: str,
        sub_class: str,
        structure: Union[str, StructureType] = "auto",
        with_splits: bool = True,
    ) -> ClassPath:
        """
        Create class folder in specified structure.
        
        Parameters
        ----------
        main_class : str
            Parent class name (e.g., "paracap").
        sub_class : str
            Subclass name (e.g., "front", "back").
        structure : str | StructureType
            "nested", "flat", or "auto" (detect from existing).
        with_splits : bool
            Create train/{good,bad} and test/{good,bad} subfolders.
            
        Returns
        -------
        ClassPath : resolved path object
        """
        if isinstance(structure, str):
            structure = StructureType[structure.upper()]
        
        # Auto-detect if needed
        if structure == StructureType.AUTO:
            structure = self._detect_structure(main_class, sub_class)
        
        # Build path
        if structure == StructureType.NESTED:
            class_path = self.root / main_class / sub_class
        else:  # FLAT
            class_path = self.root / f"{main_class}_{sub_class}"
        
        # Create folders
        class_path.mkdir(parents=True, exist_ok=True)
        
        if with_splits:
            for split in ["train", "test"]:
                for label in ["good", "bad"]:
                    (class_path / split / label).mkdir(parents=True, exist_ok=True)
        
        # Cache and return
        result = ClassPath(
            main_class=main_class,
            sub_class=sub_class,
            path=class_path,
            structure=structure,
        )
        self._class_cache[(main_class, sub_class)] = result
        return result
    
    # ─────────────────── access / resolve ───────────────────
    def get_class_path(
        self,
        main_class: str,
        sub_class: str,
        structure: Union[str, StructureType] = "auto",
        create_if_missing: bool = False,
    ) -> Optional[ClassPath]:
        """
        Get ClassPath object for a class, creating if needed.
        
        Parameters
        ----------
        create_if_missing : bool
            If True, call create() when path doesn't exist.
            
        Returns
        -------
        ClassPath or None if not found and create_if_missing=False.
        """
        cache_key = (main_class, sub_class)
        if cache_key in self._class_cache:
            return self._class_cache[cache_key]
        
        if isinstance(structure, str):
            structure = StructureType[structure.upper()]
        
        if structure == StructureType.AUTO:
            structure = self._detect_structure(main_class, sub_class)
        
        if structure == StructureType.NESTED:
            class_path = self.root / main_class / sub_class
        else:
            class_path = self.root / f"{main_class}_{sub_class}"
        
        if not class_path.exists():
            if create_if_missing and self.auto_create:
                return self.create(main_class, sub_class, structure)
            return None
        
        result = ClassPath(
            main_class=main_class,
            sub_class=sub_class,
            path=class_path,
            structure=structure,
        )
        self._class_cache[cache_key] = result
        return result
    
    # ─────────────────── list / discover ───────────────────
    def list_classes(
        self,
        structure: Union[str, StructureType] = "auto",
        require_train_good: bool = True,
    ) -> List[Tuple[str, str]]:
        """
        List all discovered (main_class, sub_class) pairs.
        
        Parameters
        ----------
        require_train_good : bool
            Only include classes that have train/good/ folder with images.
            
        Returns
        -------
        List of (main, sub) tuples.
        """
        if isinstance(structure, str):
            structure = StructureType[structure.upper()]
        
        if structure == StructureType.AUTO:
            structure = self._detect_structure()
        
        results: List[Tuple[str, str]] = []
        
        if structure == StructureType.NESTED:
            # Scan {root}/*/* pattern
            for main_dir in sorted(self.root.iterdir()):
                if not main_dir.is_dir():
                    continue
                for sub_dir in sorted(main_dir.iterdir()):
                    if not sub_dir.is_dir():
                        continue
                    if require_train_good:
                        tg = sub_dir / "train" / "good"
                        if not self._has_images(tg):
                            continue
                    results.append((main_dir.name, sub_dir.name))
        else:  # FLAT
            # Scan {root}/*_{*} pattern
            for item in sorted(self.root.iterdir()):
                if not item.is_dir() or "_" not in item.name:
                    continue
                # Split on first underscore only
                parts = item.name.split("_", 1)
                if len(parts) != 2:
                    continue
                main, sub = parts
                if require_train_good:
                    tg = item / "train" / "good"
                    if not self._has_images(tg):
                        continue
                results.append((main, sub))
        
        return results
    
    def list_images(
        self,
        main_class: str,
        sub_class: str,
        split: str = "train",
        label: str = "good",
        structure: Union[str, StructureType] = "auto",
        sorted: bool = True,
    ) -> List[Path]:
        """
        List image files for a specific class/split/label.
        
        Returns
        -------
        List of Path objects to image files.
        """
        cp = self.get_class_path(main_class, sub_class, structure)
        if not cp:
            return []
        
        folder = {
            ("train", "good"): cp.train_good,
            ("train", "bad"):  cp.train_bad,
            ("test", "good"):  cp.test_good,
            ("test", "bad"):   cp.test_bad,
        }.get((split, label))
        
        if not folder or not folder.exists():
            return []
        
        images = [
            f for f in folder.iterdir()
            if f.is_file() and f.suffix.lower() in self.image_exts
        ]
        
        return sorted(images) if sorted else images
    
    def count_images(
        self,
        main_class: str,
        sub_class: str,
        split: str = "train",
        label: str = "good",
        structure: Union[str, StructureType] = "auto",
    ) -> int:
        """Quick count without listing all files."""
        cp = self.get_class_path(main_class, sub_class, structure)
        if not cp:
            return 0
        return cp.count_images(split, label, self.image_exts)
    
    # ─────────────────── utilities ───────────────────
    def _detect_structure(
        self,
        main_class: Optional[str] = None,
        sub_class: Optional[str] = None,
    ) -> StructureType:
        """
        Auto-detect structure from existing folders.
        Priority: nested > flat > default to nested.
        """
        if not self.root.exists():
            return StructureType.NESTED
        
        # If specific class requested, check that first
        if main_class and sub_class:
            nested = self.root / main_class / sub_class
            flat = self.root / f"{main_class}_{sub_class}"
            if nested.exists():
                return StructureType.NESTED
            if flat.exists():
                return StructureType.FLAT
            return StructureType.NESTED  # Default
        
        # Scan root for pattern detection
        nested_count = 0
        flat_count = 0
        
        for item in self.root.iterdir():
            if not item.is_dir():
                continue
            # Check if it's a main class (has subdirs)
            subdirs = [d for d in item.iterdir() if d.is_dir()]
            if subdirs:
                # Likely nested structure
                nested_count += 1
            elif "_" in item.name:
                # Likely flat structure
                flat_count += 1
        
        if nested_count >= flat_count:
            return StructureType.NESTED
        return StructureType.FLAT
    
    def _has_images(self, folder: Path) -> bool:
        """Check if folder contains any valid images."""
        if not folder.exists():
            return False
        return any(
            f.suffix.lower() in self.image_exts
            for f in folder.iterdir() if f.is_file()
        )
    
    def copy_images(
        self,
        source: Union[str, Path],
        main_class: str,
        sub_class: str,
        split: str = "train",
        label: str = "good",
        structure: Union[str, StructureType] = "auto",
        overwrite: bool = False,
    ) -> int:
        """
        Copy images from source folder to dataset structure.
        
        Returns
        -------
        Number of files copied.
        """
        cp = self.create(main_class, sub_class, structure)
        
        target = {
            ("train", "good"): cp.train_good,
            ("train", "bad"):  cp.train_bad,
            ("test", "good"):  cp.test_good,
            ("test", "bad"):   cp.test_bad,
        }.get((split, label))
        
        if not target:
            raise ValueError(f"Invalid split/label: {split}/{label}")
        
        source = Path(source)
        if not source.exists():
            raise FileNotFoundError(f"Source not found: {source}")
        
        copied = 0
        for src_file in source.iterdir():
            if not src_file.is_file() or src_file.suffix.lower() not in self.image_exts:
                continue
            
            dst_file = target / src_file.name
            if dst_file.exists() and not overwrite:
                continue
            
            shutil.copy2(src_file, dst_file)
            copied += 1
        
        return copied
    
    def summary(self) -> Dict:
        """Get dataset summary statistics."""
        classes = self.list_classes()
        total_images = 0
        by_class: Dict[str, Dict] = {}
        
        for main, sub in classes:
            cp = self.get_class_path(main, sub)
            if not cp:
                continue
            
            stats = {
                "train_good": cp.count_images("train", "good", self.image_exts),
                "train_bad":  cp.count_images("train", "bad", self.image_exts),
                "test_good":  cp.count_images("test", "good", self.image_exts),
                "test_bad":   cp.count_images("test", "bad", self.image_exts),
            }
            class_total = sum(stats.values())
            total_images += class_total
            
            key = f"{main}/{sub}" if cp.structure == StructureType.NESTED else f"{main}_{sub}"
            by_class[key] = {
                "structure": cp.structure.name,
                "path": str(cp.path),
                "counts": stats,
                "total": class_total,
            }
        
        return {
            "root": str(self.root),
            "total_classes": len(classes),
            "total_images": total_images,
            "classes": by_class,
        }
    
    # ─────────────────── magic methods ───────────────────
    def __repr__(self) -> str:
        return f"DatasetManager(root='{self.root}')"
    
    def __getitem__(self, key: Tuple[str, str]) -> Optional[ClassPath]:
        """Allow mgr[('main', 'sub')] syntax."""
        if isinstance(key, tuple) and len(key) == 2:
            return self.get_class_path(key[0], key[1])
        return None