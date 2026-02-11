# core_predict/yolo_detector.py
"""
YOLO Pill Detector with Tracking.

Single Responsibility:
- Detect pills using YOLO segmentation
- Track pills across frames using center distance + IoU
- Return cropped pill images with tracking info
"""
from ultralytics import YOLO
import cv2
import numpy as np
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass


# =========================================================
# TRACKING UTILITIES
# =========================================================
def _center_xy(bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
    """Calculate center (x, y) of bounding box."""
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _center_distance(c1: Tuple[float, float], c2: Tuple[float, float]) -> float:
    """Euclidean distance between two centers."""
    dx = c1[0] - c2[0]
    dy = c1[1] - c2[1]
    return float((dx * dx + dy * dy) ** 0.5)


def _bbox_iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    """Calculate IoU between two bboxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter <= 0:
        return 0.0

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / (union + 1e-6)) if union > 0 else 0.0


@dataclass
class _Track:
    """Track data for a single pill."""
    track_id: int
    bbox: Tuple[int, int, int, int]
    center: Tuple[float, float]
    age: int = 0


class PillYOLODetector:
    """
    YOLO-based pill detector with tracking.
    
    Pipeline: YOLO → Box NMS → Mask NMS → Merge → Crop → Track
    Supports both detection (bbox) and segmentation (mask) modes.
    """

    def __init__(
        self,
        seg_model_path: str = "model/yolo26-segmentation.pt",
        det_model_path: str = "model/yolo26-detection.pt",
        img_size: int = 768,
        conf: float = 0.38,
        iou: float = 0.6,
        pad: int = 5,
        merge_dist_px: int = 60,
        enable_tracking: bool = True,
        track_max_distance: float = 80.0,
        track_iou_threshold: float = 0.80,
        track_max_age: int = 10,
        start_with_detection: bool = True,
    ):
        print("Loading YOLO models...")
        self.seg_model = YOLO(seg_model_path)
        print(f"✓ Loaded segmentation model: {seg_model_path}")
        
        # Try to load detection model, fallback to segmentation if fails
        self.det_model = None
        self.has_separate_det_model = False
        
        if start_with_detection and det_model_path:
            try:
                self.det_model = YOLO(det_model_path)
                self.has_separate_det_model = True
                print(f"✓ Loaded detection model: {det_model_path}")
            except Exception as e:
                print(f"⚠ Failed to load detection model: {e}")
                print(f"⚠ Using segmentation model for both modes")
                self.det_model = None
        
        # Set initial model
        if self.det_model is None:
            self.det_model = self.seg_model
            self.has_separate_det_model = False
        
        self.use_detection = start_with_detection and self.has_separate_det_model
        self.model = self.det_model if self.use_detection else self.seg_model
        
        if self.has_separate_det_model:
            print(f"Starting in {'DETECTION' if start_with_detection else 'SEGMENTATION'} mode")
        else:
            print(f"Starting in SEGMENTATION mode (no separate detection model)")

        self.img_size = img_size
        self.conf = conf
        self.iou = iou
        self.pad = pad
        self.merge_dist_px = merge_dist_px
        
        # Tracking state
        self.enable_tracking = enable_tracking
        self.track_max_distance = float(track_max_distance)
        self.track_iou_threshold = float(track_iou_threshold)
        self.track_max_age = int(track_max_age)
        self._next_id = 1
        self._tracks: Dict[int, _Track] = {}

    # =========================================================
    # TRACKING
    # =========================================================
    def _assign_track_ids(self, infos: List[Dict]) -> List[Dict]:
        """Assign track IDs to detections."""
        if not self.enable_tracking or not infos:
            for info in infos:
                info["track_id"] = -1
                info["center"] = _center_xy(info["padded_region"])
                info["match_distance"] = -1.0
                info["match_iou"] = 0.0
            return infos

        # Prepare detections
        dets: List[Tuple[int, Tuple[int, int, int, int], Tuple[float, float]]] = []
        for i, info in enumerate(infos):
            bbox = tuple(map(int, info["padded_region"]))
            center = _center_xy(bbox)
            dets.append((i, bbox, center))

        # Matching
        track_items = list(self._tracks.items())
        assigned_det_idx: set = set()
        assigned_track_ids: set = set()
        assignments: Dict[int, int] = {}
        match_meta: Dict[int, Dict[str, float]] = {}

        candidates: List[Tuple[int, float, float, int, int]] = []
        
        for det_idx, det_bbox, det_center in dets:
            for track_id, track in track_items:
                iou = _bbox_iou(det_bbox, track.bbox)
                dist = _center_distance(det_center, track.center)

                if iou >= self.track_iou_threshold:
                    candidates.append((0, dist, -iou, det_idx, track_id))
                elif dist <= self.track_max_distance:
                    candidates.append((1, dist, -iou, det_idx, track_id))

        candidates.sort(key=lambda t: (t[0], t[1], t[2]))
        
        for priority, dist, neg_iou, det_idx, track_id in candidates:
            if det_idx in assigned_det_idx or track_id in assigned_track_ids:
                continue
            assigned_det_idx.add(det_idx)
            assigned_track_ids.add(track_id)
            assignments[det_idx] = track_id
            match_meta[det_idx] = {"distance": float(dist), "iou": float(-neg_iou)}

        # Create new tracks
        for det_idx, det_bbox, det_center in dets:
            if det_idx not in assignments:
                new_id = self._next_id
                self._next_id += 1
                self._tracks[new_id] = _Track(
                    track_id=new_id, bbox=det_bbox, center=det_center, age=0
                )
                assignments[det_idx] = new_id
                match_meta[det_idx] = {"distance": -1.0, "iou": 0.0}

        # Update matched tracks
        for det_idx, det_bbox, det_center in dets:
            track_id = assignments.get(det_idx)
            if track_id is None:
                continue
            track = self._tracks.get(track_id)
            if track is None:
                self._tracks[track_id] = _Track(
                    track_id=track_id, bbox=det_bbox, center=det_center, age=0
                )
            else:
                track.bbox = det_bbox
                track.center = det_center
                track.age = 0

        # Age unmatched tracks
        for track_id in list(self._tracks.keys()):
            if track_id not in assigned_track_ids:
                self._tracks[track_id].age += 1
                if self._tracks[track_id].age > self.track_max_age:
                    del self._tracks[track_id]

        # Add tracking info to results
        for det_idx, info in enumerate(infos):
            tid = assignments.get(det_idx, -1)
            bbox = tuple(map(int, info["padded_region"]))
            cx, cy = _center_xy(bbox)
            info["track_id"] = int(tid)
            info["center"] = (float(cx), float(cy))
            meta = match_meta.get(det_idx, {"distance": -1.0, "iou": 0.0})
            info["match_distance"] = float(meta["distance"])
            info["match_iou"] = float(meta["iou"])

        return infos

    # =========================================================
    # NMS
    # =========================================================
    @staticmethod
    def _box_nms(boxes, scores, thr=0.7):
        x1, y1, x2, y2 = boxes.T
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        keep = []

        while order.size > 0:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)

            inter = w * h
            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)

            inds = np.where(iou <= thr)[0]
            order = order[inds + 1]

        return np.array(keep, dtype=int)

    @staticmethod
    def _mask_nms(masks, scores, thr=0.75):
        masks = (masks > 0.5).astype(bool)
        order = scores.argsort()[::-1]
        keep = []

        while len(order) > 0:
            i = order[0]
            keep.append(i)

            if len(order) == 1:
                break

            cur = masks[i]
            rest = masks[order[1:]]

            inter = (rest & cur).sum((1, 2))
            union = (rest | cur).sum((1, 2))
            iou = inter / (union + 1e-6)

            inds = np.where(iou <= thr)[0]
            order = order[inds + 1]

        return np.array(keep, dtype=int)

    def _detect_bbox_mode(
        self, frame: np.ndarray, results
    ) -> Tuple[List[np.ndarray], List[Dict[str, Any]]]:
        """Detection mode: crop using bounding boxes only."""
        if results.boxes is None or len(results.boxes) == 0:
            return [], []
        
        h, w = frame.shape[:2]
        boxes = results.boxes.xyxy.cpu().numpy()
        scores = results.boxes.conf.cpu().numpy()
        
        # Box NMS only
        keep = self._box_nms(boxes, scores, thr=self.iou)
        boxes, scores = boxes[keep], scores[keep]
        
        cropped = []
        infos = []
        
        for box, conf in zip(boxes, scores):
            x1, y1, x2, y2 = map(int, box)
            
            # Add padding
            x1 = max(0, x1 - self.pad)
            y1 = max(0, y1 - self.pad)
            x2 = min(w, x2 + self.pad)
            y2 = min(h, y2 + self.pad)
            
            if x2 <= x1 or y2 <= y1:
                continue
            
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            
            cropped.append(crop)
            infos.append({
                "bbox": (int(x1), int(y1), int(x2), int(y2)),
                "padded_region": (int(x1), int(y1), int(x2), int(y2)),
                "confidence": float(conf),
                "mode": "detection",
            })
        
        infos = self._assign_track_ids(infos)
        return cropped, infos
    
    def _merge_instances(self, masks, boxes, scores):
        N = len(masks)
        used = np.zeros(N, dtype=bool)

        new_masks = []
        new_boxes = []
        new_scores = []

        centers = np.stack([
            (boxes[:, 0] + boxes[:, 2]) / 2,
            (boxes[:, 1] + boxes[:, 3]) / 2
        ], axis=1)

        for i in range(N):
            if used[i]:
                continue

            cur_mask = masks[i].copy()
            cur_box = boxes[i].copy()
            cur_score = scores[i]
            used[i] = True

            for j in range(i + 1, N):
                if used[j]:
                    continue

                dist = np.linalg.norm(centers[i] - centers[j])
                if dist > self.merge_dist_px:
                    continue

                inter = (cur_mask & masks[j]).sum()
                union = (cur_mask | masks[j]).sum()
                iou = inter / (union + 1e-6)

                if iou > 0.3:
                    cur_mask |= masks[j]
                    cur_box = [
                        min(cur_box[0], boxes[j][0]),
                        min(cur_box[1], boxes[j][1]),
                        max(cur_box[2], boxes[j][2]),
                        max(cur_box[3], boxes[j][3]),
                    ]
                    cur_score = max(cur_score, scores[j])
                    used[j] = True

            new_masks.append(cur_mask)
            new_boxes.append(cur_box)
            new_scores.append(cur_score)

        return np.array(new_masks), np.array(new_boxes), np.array(new_scores)

    # =========================================================
    # MAIN DETECTION
    # =========================================================
    def _detect_bbox_mode(
        self, frame: np.ndarray, results
    ) -> Tuple[List[np.ndarray], List[Dict[str, Any]]]:
        """Detection mode: crop using bounding boxes only."""
        if results.boxes is None or len(results.boxes) == 0:
            return [], []
        
        h, w = frame.shape[:2]
        boxes = results.boxes.xyxy.cpu().numpy()
        scores = results.boxes.conf.cpu().numpy()
        
        # Box NMS only
        keep = self._box_nms(boxes, scores, thr=self.iou)
        boxes, scores = boxes[keep], scores[keep]
        
        cropped = []
        infos = []
        
        for box, conf in zip(boxes, scores):
            x1, y1, x2, y2 = map(int, box)
            
            # Add padding
            x1 = max(0, x1 - self.pad)
            y1 = max(0, y1 - self.pad)
            x2 = min(w, x2 + self.pad)
            y2 = min(h, y2 + self.pad)
            
            if x2 <= x1 or y2 <= y1:
                continue
            
            crop = frame[y1:y2, x1:x2].copy()
            if crop.size == 0:
                continue
            
            cropped.append(crop)
            infos.append({
                "bbox": (int(x1), int(y1), int(x2), int(y2)),
                "padded_region": (int(x1), int(y1), int(x2), int(y2)),
                "conf": float(conf),
                "mode": "detection",
            })
        
        infos = self._assign_track_ids(infos)
        return cropped, infos
    
    def detect_and_crop(
        self, frame: np.ndarray
    ) -> Tuple[List[np.ndarray], List[Dict[str, Any]]]:
        """Detect pills and return cropped images with info."""
        results = self.model(
            frame,
            imgsz=self.img_size,
            conf=self.conf,
            iou=self.iou,
            retina_masks=False,
            verbose=False
        )[0]

        # Detection mode: use bboxes only
        if self.use_detection:
            return self._detect_bbox_mode(frame, results)
        
        # Segmentation mode: use masks
        if results.masks is None:
            return [], []

        h, w = frame.shape[:2]
        
        masks_raw = results.masks.data.cpu().numpy()
        boxes = results.boxes.xyxy.cpu().numpy()
        scores = results.boxes.conf.cpu().numpy()
        
        # Resize masks
        masks_resized = []
        for mask in masks_raw:
            mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
            masks_resized.append(mask_resized > 0.5)
        masks = np.array(masks_resized, dtype=bool)

        # NMS
        keep = self._box_nms(boxes, scores)
        masks, boxes, scores = masks[keep], boxes[keep], scores[keep]

        keep = self._mask_nms(masks, scores)
        masks, boxes, scores = masks[keep], boxes[keep], scores[keep]

        # Merge
        masks, boxes, scores = self._merge_instances(masks, boxes, scores)

        cropped = []
        infos = []

        for mask, box, conf in zip(masks, boxes, scores):
            ys, xs = np.where(mask)
            if len(xs) == 0:
                continue

            x1 = max(0, xs.min() - self.pad)
            y1 = max(0, ys.min() - self.pad)
            x2 = min(w, xs.max() + self.pad)
            y2 = min(h, ys.max() + self.pad)

            crop = frame[y1:y2, x1:x2].copy()
            mask_region = mask[y1:y2, x1:x2]
            crop[~mask_region] = 0

            cropped.append(crop)
            infos.append({
                "bbox": box.astype(int).tolist(),
                "conf": float(conf),
                "padded_region": (x1, y1, x2, y2),
                "mode": "segmentation",
            })

        infos = self._assign_track_ids(infos)
        return cropped, infos

    def reset_tracking(self) -> None:
        """Reset all tracking state."""
        self._tracks.clear()
        self._next_id = 1
    
    def switch_to_segmentation(self) -> None:
        """Switch to segmentation mode for mask-based cropping."""
        if not self.use_detection:
            return
        self.use_detection = False
        self.model = self.seg_model
        print("[MODE] Switched to SEGMENTATION mode")
    
    def switch_to_detection(self) -> None:
        """Switch to detection mode for bbox-based cropping."""
        if self.use_detection or not self.has_separate_det_model:
            return
        self.use_detection = True
        self.model = self.det_model
        print("[MODE] Switched to DETECTION mode")
    
    @property
    def current_mode(self) -> str:
        """Get current detection mode."""
        return "DETECTION" if self.use_detection else "SEGMENTATION"
