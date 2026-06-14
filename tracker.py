
from ultralytics import YOLO
import numpy as np

class Tracker:
    """
    Wraps Ultralytics YOLOv8 built-in ByteTrack.
    ByteTrack uses:
      - Kalman filter for motion prediction
      - Hungarian algorithm for assignment
      - High/low confidence detection separation
      - Track buffer for occlusion handling
    """

    def __init__(self, model_path="yolov8n.pt", confidence_threshold=0.50):
        self.model = YOLO(model_path)
        self.conf  = confidence_threshold
        self.CLASS_PERSON = 0

    def update(self, frame, frame_id):
        """
        Run detection + tracking on a single frame.

        Args:
            frame    : BGR numpy array
            frame_id : current frame number (unused, ByteTrack handles internally)

        Returns:
            tracks: list of dicts, each containing:
                    track_id, bbox [x1,y1,x2,y2], confidence, last_bbox
        """
        results = self.model.track(
            frame,
            classes=[self.CLASS_PERSON],
            conf=self.conf,
            tracker="bytetrack.yaml",
            persist=True,       # keeps track state between frames
            verbose=False
        )

        tracks = []
        for result in results:
            if result.boxes is None:
                continue
            boxes = result.boxes
            if boxes.id is None:
                continue

            for i in range(len(boxes)):
                track_id   = int(boxes.id[i].item())
                bbox       = boxes.xyxy[i].tolist()
                confidence = float(boxes.conf[i].item())

                # Simple Track-like object so rest of pipeline
                # does not need to change
                track = SimpleTrack(track_id, bbox, confidence)
                tracks.append(track)

        return tracks


class SimpleTrack:
    """
    Mimics the Track object interface from tracker.py
    so detector, fusion, visualization all work unchanged.
    """
    def __init__(self, track_id, bbox, confidence):
        self.track_id   = track_id
        self.last_bbox  = bbox
        self.confidence = confidence
        self.hits       = 1
        self.history    = []

        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        self.history.append((0, cx, cy))

    def get_center(self):
        x1, y1, x2, y2 = self.last_bbox
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0
