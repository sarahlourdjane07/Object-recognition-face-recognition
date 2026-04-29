from ultralytics import YOLO
import random
import cv2

class YoloDetector2:
    def __init__(self, weights = "yolov8n.pt", conf = 0.25, imgsz = 640):
        self.model = None
        self.weights = weights
        self.conf = conf
        self.imgsz = imgsz
        self.colors = {}

    def _ensure_model(self):
        if self.model is None:
            self.model = YOLO(self.weights)

    def _color_for(self, cls_id):
        if cls_id not in self.colors:
            self.colors[cls_id] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

        return self.colors[cls_id]

    def draw_detections(self, frame_bgr):
        self._ensure_model()
        result = self.model(frame_bgr, imgsz = self.imgsz, conf = self.conf, verbose = False)[0]
        names = result.names
        if result.boxes is not None and len(result.boxes) > 0:
            for box in result.boxes:
                xyxy = box.xyxy[0].tolist()
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, xyxy)
                label = f"{names.get(cls_id, str(cls_id))} {conf * 100:1f}%"
                color = self._color_for(cls_id)
                cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)
                (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame_bgr, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
                cv2.putText(frame_bgr, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        return frame_bgr