from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import cv2
from ultralytics import YOLO

from paddle_attr import PaddleAttributeExtractor


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_SOURCE = BASE_DIR / "bus.jpg"
DEFAULT_MODEL = BASE_DIR / "yolo26n.pt"
DEFAULT_PDMODEL = BASE_DIR / "inference.pdmodel"
DEFAULT_PDIPARAMS = BASE_DIR / "inference.pdiparams"
DEFAULT_OUTPUT_JSON = BASE_DIR / "runs/detect/person_output/response.json"
DEFAULT_OUTPUT_IMAGE_DIR = BASE_DIR / "runs/detect/person_output"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect people and extract attributes")
    parser.add_argument("--source", default=str(DEFAULT_SOURCE), help="Input image path")
    parser.add_argument("--pdmodel", default=str(DEFAULT_PDMODEL), help="Paddle pdmodel path")
    parser.add_argument("--pdiparams", default=str(DEFAULT_PDIPARAMS), help="Paddle pdiparams path")
    parser.add_argument(
        "--output-json",
        default=str(DEFAULT_OUTPUT_JSON),
        help="Output JSON path (response list format)",
    )
    return parser.parse_args()


def run() -> None:
    args = parse_args()
    source_path = Path(args.source)
    if not source_path.is_file():
        raise FileNotFoundError(f"image not found: {source_path}")

    image_bgr = cv2.imread(str(source_path))
    if image_bgr is None:
        raise ValueError(f"failed to load image: {source_path}")
    image_h, image_w = image_bgr.shape[:2]

    attr_extractor = PaddleAttributeExtractor(
        pdmodel_path=str(args.pdmodel),
        pdiparams_path=str(args.pdiparams),
    )

    model = YOLO(str(DEFAULT_MODEL))
    result = model.predict(source=str(source_path), conf=0.25, classes=[0], verbose=False)[0]

    output_image_dir = DEFAULT_OUTPUT_IMAGE_DIR
    output_image_dir.mkdir(parents=True, exist_ok=True)
    output_image_path = output_image_dir / source_path.name
    annotated = result.plot()
    cv2.imwrite(str(output_image_path), annotated)

    person_boxes: list[list[float]] = []
    if result.boxes is not None and len(result.boxes) > 0:
        person_boxes = [[float(v) for v in raw_box] for raw_box in result.boxes.xyxy.cpu().tolist()]

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    persons: list[dict[str, object]] = []
    for idx, person_box in enumerate(person_boxes, start=1):
        x1 = max(0, min(int(round(person_box[0])), image_w - 1))
        y1 = max(0, min(int(round(person_box[1])), image_h - 1))
        x2 = max(0, min(int(round(person_box[2])), image_w))
        y2 = max(0, min(int(round(person_box[3])), image_h))
        if x2 <= x1 or y2 <= y1:
            continue

        crop_bgr = image_bgr[y1:y2, x1:x2]
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        attributes = attr_extractor.predict_attributes(crop_rgb)

        persons.append(
            {
                "person_id": idx,
                "attributes": attributes,
                "timestamp": timestamp,
            }
        )

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(persons, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"done: persons={len(persons)}")
    print(f"json saved: {output_path}")
    print(f"image saved: {output_image_path}")


if __name__ == "__main__":
    run()
