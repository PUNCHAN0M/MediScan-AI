import math
from typing import Dict, List, Any


class DistanceTracker:
    """
    Lightweight centroid-distance tracker.
    Optimized for small/medium object counts.
    """

    def __init__(
        self,
        max_distance: float = 80.0,
        max_age: int = 10,
    ):
        self.max_distance = max_distance
        self.max_age = max_age
        self._tracks: Dict[int, Dict[str, Any]] = {}
        self._next_id = 1

    # ─────────────────────────────────────────────

    def update(self, detections: List[Dict]) -> List[Dict]:

        if not detections:
            self._age_tracks()
            return detections

        # Stable order (reduce ID switching)
        # detections = sorted(detections, key=lambda d: d["bbox"][0])

        # Compute centers
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            det["center"] = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
            det["track_id"] = -1

        used_tracks = set()
        used_dets = set()

        # Matching
        for di, det in enumerate(detections):

            best_tid = None
            best_dist = float("inf")

            for tid, track in self._tracks.items():
                if tid in used_tracks:
                    continue

                dx = det["center"][0] - track["center"][0]
                dy = det["center"][1] - track["center"][1]
                dist = math.hypot(dx, dy)

                if dist < best_dist and dist <= self.max_distance:
                    best_dist = dist
                    best_tid = tid

            if best_tid is not None:
                det["track_id"] = best_tid
                self._tracks[best_tid]["center"] = det["center"]
                self._tracks[best_tid]["age"] = 0
                used_tracks.add(best_tid)
                used_dets.add(di)

        # Create new tracks
        for di, det in enumerate(detections):
            if di not in used_dets:
                tid = self._next_id
                self._next_id += 1
                det["track_id"] = tid
                self._tracks[tid] = {
                    "center": det["center"],
                    "age": 0,
                }

        self._age_tracks(used_tracks)
        return detections

    # ─────────────────────────────────────────────

    def _age_tracks(self, active_ids=None):
        active_ids = active_ids or set()

        for tid in list(self._tracks.keys()):
            if tid not in active_ids:
                self._tracks[tid]["age"] += 1
                if self._tracks[tid]["age"] > self.max_age:
                    del self._tracks[tid]

    def reset(self):
        self._tracks.clear()
        self._next_id = 1