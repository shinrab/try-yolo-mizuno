from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import cv2
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simple person detection on one image")
    parser.add_argument("--source", default="bus.jpg", help="Input image path")
    parser.add_argument("--model", default="yolo26n.pt", help="YOLO model path")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument(
        "--output-json",
        default="runs/detect/person_output/reid_output.json",
        help="Output JSON path",
    )
    parser.add_argument(
        "--output-image-dir",
        default="runs/detect/person_output",
        help="Output directory for annotated detection image",
    )
    return parser.parse_args()


def calc_iou(box_a: list[float], box_b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter_area
    return 0.0 if denom <= 0.0 else inter_area / denom


def build_attributes(person_box: list[float], backpack_boxes: list[list[float]], handbag_boxes: list[list[float]]) -> dict:
    return {
        "hat": None,
        "glasses": None,
        "long_sleeve": None,
        "trousers": None,
        "handbag": any(calc_iou(person_box, b) > 0.05 for b in handbag_boxes),
        "shoulderbag": None,
        "backpack": any(calc_iou(person_box, b) > 0.05 for b in backpack_boxes),
    }


def run() -> None:
    args = parse_args()
    source_path = Path(args.source)
    if not source_path.is_file():
        raise FileNotFoundError(f"image not found: {source_path}")

    model = YOLO(args.model)
    result = model.predict(source=str(source_path), conf=args.conf, classes=[0, 24, 26], verbose=False)[0]

    output_image_dir = Path(args.output_image_dir)
    output_image_dir.mkdir(parents=True, exist_ok=True)
    output_image_path = output_image_dir / source_path.name
    annotated = result.plot()
    cv2.imwrite(str(output_image_path), annotated)

    person_boxes: list[list[float]] = []
    person_confs: list[float] = []
    backpack_boxes: list[list[float]] = []
    handbag_boxes: list[list[float]] = []

    if result.boxes is not None and len(result.boxes) > 0:
        boxes_xyxy = result.boxes.xyxy.cpu().tolist()
        boxes_cls = result.boxes.cls.cpu().tolist()
        boxes_conf = result.boxes.conf.cpu().tolist()

        for i, raw_box in enumerate(boxes_xyxy):
            box = [float(v) for v in raw_box]
            cls_id = int(boxes_cls[i])
            conf = float(boxes_conf[i])

            if cls_id == 0:
                person_boxes.append(box)
                person_confs.append(conf)
            elif cls_id == 24:
                backpack_boxes.append(box)
            elif cls_id == 26:
                handbag_boxes.append(box)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    persons: list[dict] = []
    for idx, person_box in enumerate(person_boxes, start=1):
        persons.append(
            {
                "person_id": idx,
                "bbox": [round(v, 2) for v in person_box],
                "confidence": round(person_confs[idx - 1], 4),
                "attributes": build_attributes(person_box, backpack_boxes, handbag_boxes),
                "timestamp": timestamp,
            }
        )

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source": str(source_path),
        "persons": persons,
        "summary": {"person_count": len(persons)},
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"done: persons={len(persons)}")
    print(f"json saved: {output_path}")
    print(f"image saved: {output_image_path}")


if __name__ == "__main__":
    run()
