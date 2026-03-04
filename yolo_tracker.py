from yolo_engine import YOLOEngine
from yolo_distance_tracker import DistanceTracker


class YOLOTracker:
    """
    Clean production pipeline:
    YOLOEngine → DistanceTracker
    """

    def __init__(
        self,
        model_path: str,
        mode: str = "segment",
        device: str = "cuda",
        conf: float = 0.5,
        iou: float = 0.6,
        input_size: int = 640,
        enable_tracking: bool = True,
    ):

        self.engine = YOLOEngine(
            model_path=model_path,
            mode=mode,
            device=device,
            input_size=input_size,
            conf=conf,
            iou=iou,
        )

        self.tracker = DistanceTracker() if enable_tracking else None

    def process(self, frame):

        detections = self.engine.inference(frame)

        if self.tracker is not None:
            detections = self.tracker.update(detections)

        return detections