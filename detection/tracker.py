# detection/tracker.py
"""
Centroid + IoU Tracker
=======================
Frame-to-frame object tracking by distance and box overlap.
No model logic. No pipeline loops.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Set


class CentroidIoUTracker:
    """
    Greedy centroid + IoU tracker for consecutive frames.

    Each detection info dict must contain:
        - ``"center"`` : ``(cx, cy)``
        - ``"bbox"``   : ``(x1, y1, x2, y2)``

    After ``update(infos)``, each info dict gets ``"track_id"`` assigned.
    """

    def __init__(
        self,
        max_distance: float = 80.0,
        iou_threshold: float = 0.80,
        max_age: int = 10,
    ):
        self._max_dist = max_distance
        self._iou_thr = iou_threshold
        self._max_age = max_age

        self._tracks: Dict[int, Dict[str, Any]] = {}
        self._next_id: int = 1

    # ───────── Main update ─────────
    def update(self, infos: List[Dict[str, Any]]) -> None:
        """Assign track IDs to current frame detections (mutates infos)."""
        if not infos:
            self._age_tracks()
            return

        candidates: list = []
        for di, info in enumerate(infos):
            for tid, track in self._tracks.items():
                dist = math.hypot(
                    info["center"][0] - track["center"][0],
                    info["center"][1] - track["center"][1],
                )
                iou = self._iou(info["bbox"], track["bbox"])

                if iou >= self._iou_thr:
                    candidates.append((0, dist, -iou, di, tid))
                elif dist <= self._max_dist:
                    candidates.append((1, dist, -iou, di, tid))

        candidates.sort()

        used_det: Set[int] = set()
        used_trk: Set[int] = set()

        for _, _, _, di, tid in candidates:
            if di in used_det or tid in used_trk:
                continue
            infos[di]["track_id"] = tid
            self._tracks[tid]["bbox"] = infos[di]["bbox"]
            self._tracks[tid]["center"] = infos[di]["center"]
            self._tracks[tid]["age"] = 0
            used_det.add(di)
            used_trk.add(tid)

        # new tracks for unmatched detections
        for di in range(len(infos)):
            if di not in used_det:
                tid = self._next_id
                self._next_id += 1
                infos[di]["track_id"] = tid
                self._tracks[tid] = {
                    "bbox": infos[di]["bbox"],
                    "center": infos[di]["center"],
                    "age": 0,
                }

        self._age_tracks(used_trk)

    # ───────── Internal ─────────
    def _age_tracks(self, keep: Optional[Set[int]] = None) -> None:
        keep = keep or set()
        for tid in list(self._tracks):
            if tid not in keep:
                self._tracks[tid]["age"] += 1
                if self._tracks[tid]["age"] > self._max_age:
                    del self._tracks[tid]

    @staticmethod
    def _iou(a: tuple, b: tuple) -> float:
        ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
        ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        if inter == 0:
            return 0.0
        aa = (a[2] - a[0]) * (a[3] - a[1])
        ab = (b[2] - b[0]) * (b[3] - b[1])
        return inter / (aa + ab - inter)

    # ───────── Public helpers ─────────
    def reset(self) -> None:
        """Clear all tracks and reset ID counter."""
        self._tracks.clear()
        self._next_id = 1

    @property
    def tracks(self) -> Dict[int, Dict[str, Any]]:
        return dict(self._tracks)
