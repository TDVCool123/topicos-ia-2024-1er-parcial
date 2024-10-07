from ultralytics import YOLO
import numpy as np
import cv2
from shapely.geometry import Polygon, box
from src.models import Detection, PredictionType, Segmentation
from src.config import get_settings

SETTINGS = get_settings()


def match_gun_bbox(segment: list[list[int]], bboxes: list[list[int]], max_distance: int = 10) -> list[int] | None:
    matched_box = None
    min_distance = float('inf')

    segment_polygon = Polygon(segment)

    for bbox in bboxes:
        bbox_polygon = Polygon([
            [bbox[0], bbox[1]],  # (x1, y1)
            [bbox[2], bbox[1]],  # (x2, y1)
            [bbox[2], bbox[3]],  # (x2, y2)
            [bbox[0], bbox[3]],  # (x1, y2)
        ])

        distance = segment_polygon.distance(bbox_polygon)
        if distance < min_distance and distance <= max_distance:
            min_distance = distance
            matched_box = bbox

    return matched_box


def annotate_detection(image_array: np.ndarray, detection: Detection) -> np.ndarray:
    ann_color = (255, 0, 0)
    annotated_img = image_array.copy()
    for label, conf, box in zip(detection.labels, detection.confidences, detection.boxes):
        x1, y1, x2, y2 = box
        cv2.rectangle(annotated_img, (x1, y1), (x2,y2), ann_color, 3)
        cv2.putText(
            annotated_img,
            f"{label}: {conf:.1f}",
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            ann_color,
            2,
        )
    return annotated_img

def annotate_segmentation(image_array: np.ndarray, segmentation: Segmentation, draw_boxes: bool = True) -> np.ndarray:
    annotated_img = image_array.copy()

    for label, polygon, box in zip(segmentation.labels, segmentation.polygons, segmentation.boxes):
        mask = np.zeros(image_array.shape, dtype=np.uint8)

        if label=="safe":
            color = (0, 255, 0) 
        elif label == 'danger':
            color = (255, 0, 0)

        pts = np.array(polygon, np.int32)
        pts = pts.reshape((-1, 1, 2))

        cv2.fillPoly(mask, [pts], color)

        #transparencia (50%)
        alpha = 0.5
        annotated_img = cv2.addWeighted(annotated_img, 1, mask, alpha, 0)

        if draw_boxes:
            x1, y1, x2, y2 = box
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                annotated_img,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                color,
                2,
            )

    return annotated_img


class GunDetector:
    def __init__(self) -> None:
        print(f"loading od model: {SETTINGS.od_model_path}")
        self.od_model = YOLO(SETTINGS.od_model_path)
        print(f"loading seg model: {SETTINGS.seg_model_path}")
        self.seg_model = YOLO(SETTINGS.seg_model_path)

    def detect_guns(self, image_array: np.ndarray, threshold: float = 0.5):
        results = self.od_model(image_array, conf=threshold)[0]
        labels = results.boxes.cls.tolist()
        indexes = [
            i for i in range(len(labels)) if labels[i] in [3, 4]
        ]  # 0 = "person"
        boxes = [
            [int(v) for v in box]
            for i, box in enumerate(results.boxes.xyxy.tolist())
            if i in indexes
        ]
        confidences = [
            c for i, c in enumerate(results.boxes.conf.tolist()) if i in indexes
        ]
        labels_txt = [
            results.names[labels[i]] for i in indexes
        ]
        return Detection(
            pred_type=PredictionType.object_detection,
            n_detections=len(boxes),
            boxes=boxes,
            labels=labels_txt,
            confidences=confidences,
        )
    
    def segment_people(self, image_array: np.ndarray, threshold: float = 0.5, max_distance: int = 10):
        results = self.seg_model(image_array, conf=threshold)[0]
        labels = results.boxes.cls.tolist()
        indexes = [
            i for i in range(len(labels)) if labels[i] == 0  # 0 = "person"
        ]
        boxes = [
            [int(v) for v in box]
            for i, box in enumerate(results.boxes.xyxy.tolist())
            if i in indexes
        ]

        polygons = []
        segment_labels = []
        detection = self.detect_guns(image_array, threshold)

        for i, box in enumerate(boxes):
            segment_polygon = [
                [box[0], box[1]],
                [box[2], box[1]],
                [box[2], box[3]],
                [box[0], box[3]],
            ]

            closest_gun = match_gun_bbox(segment_polygon, detection.boxes, max_distance)
            label = 'danger' if closest_gun else 'safe'

            polygons.append(segment_polygon)
            segment_labels.append(label)

        return Segmentation(
            pred_type=PredictionType.segmentation,
            n_detections=len(polygons),
            polygons=polygons,
            boxes=boxes,
            labels=segment_labels
        )
