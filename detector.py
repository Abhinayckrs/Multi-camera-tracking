
from ultralytics import YOLO
import numpy as np

class Detector:
    """
    Wraps YOLOv8n for person detection.
    Input  : a single BGR frame (numpy array from cv2)
    Output : list of detections, each as [x1, y1, x2, y2, confidence]
    """

    def __init__(self, model_path='yolov8n.pt', confidence_threshold=0.4):
        """
        model_path          : YOLOv8 weights. Downloads automatically on first run.
        confidence_threshold: ignore detections below this score (0 to 1)
        """
        self.model = YOLO(model_path)
        self.conf  = confidence_threshold
        self.CLASS_PERSON = 0   # in COCO dataset, class 0 = person

    def detect(self, frame):
        """
        Run detection on a single frame.

        Args:
            frame: numpy array, BGR, shape (H, W, 3)

        Returns:
            detections: numpy array of shape (N, 5)
                        each row = [x1, y1, x2, y2, confidence]
                        N = number of persons detected
                        Empty array if no persons found.
        """
        # Run YOLO — classes=[0] means only detect persons
        results = self.model(
            frame,
            classes=[self.CLASS_PERSON],
            conf=self.conf,
            verbose=False   # suppress per-frame console output
        )

        detections = []

        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue

            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                confidence      = box.conf[0].item()

                # Basic sanity filter: ignore tiny boxes (likely false positives)
                width  = x2 - x1
                height = y2 - y1
                if width < 20 or height < 40:
                    continue

                detections.append([x1, y1, x2, y2, confidence])

        if len(detections) == 0:
            return np.empty((0, 5), dtype=np.float32)

        return np.array(detections, dtype=np.float32)


# ----- Quick test (runs only when this file is executed directly) -----
if __name__ == "__main__":
    import cv2

    print("Testing Detector...")
    detector = Detector()

    # Create a blank frame to confirm model loads without error
    blank = np.zeros((640, 640, 3), dtype=np.uint8)
    result = detector.detect(blank)
    print(f"Blank frame detections: {len(result)}  (expected 0)")
    print("Detector initialized successfully.")
