from __future__ import annotations

from pathlib import Path

import cv2
from ultralytics import YOLO

from paddle_attr import PaddleAttributeExtractor
from reid_utils import ReIDEmbeddingExtractor


def clamp_bbox(
    bbox_xyxy: list[float], image_w: int, image_h: int
) -> tuple[int, int, int, int] | None:
    x1 = max(0, min(int(round(bbox_xyxy[0])), image_w - 1))
    y1 = max(0, min(int(round(bbox_xyxy[1])), image_h - 1))
    x2 = max(0, min(int(round(bbox_xyxy[2])), image_w))
    y2 = max(0, min(int(round(bbox_xyxy[3])), image_h))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def analyze_image(
    source_path: Path,
    model: YOLO,
    attr_extractor: PaddleAttributeExtractor,
    reid_extractor: ReIDEmbeddingExtractor,
    det_conf: float = 0.25,
) -> dict[str, object]:
    image_bgr = cv2.imread(str(source_path))
    if image_bgr is None:
        raise ValueError(f"failed to load image: {source_path}")

    image_h, image_w = image_bgr.shape[:2]
    result = model.predict(
        source=str(source_path), conf=det_conf, classes=[0], verbose=False
    )[0]

    detections: list[dict[str, object]] = []
    if result.boxes is not None and len(result.boxes) > 0:
        person_boxes = result.boxes.xyxy.cpu().tolist()
        confidences = result.boxes.conf.cpu().tolist()
        for raw_box, raw_confidence in zip(person_boxes, confidences):
            person_box = [float(v) for v in raw_box]
            clamped_bbox = clamp_bbox(person_box, image_w, image_h)
            if clamped_bbox is None:
                continue

            x1, y1, x2, y2 = clamped_bbox
            crop_bgr = image_bgr[y1:y2, x1:x2]
            crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            detections.append(
                {
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "bbox_area": float((x2 - x1) * (y2 - y1)),
                    "yolo_confidence": float(raw_confidence),
                    "feature": reid_extractor.extract(crop_bgr),
                    "attributes": attr_extractor.predict_attributes(crop_rgb),
                }
            )

    return {
        "image_bgr": image_bgr,
        "image_width": image_w,
        "image_height": image_h,
        "annotated_image": result.plot(),
        "detections": detections,
    }
