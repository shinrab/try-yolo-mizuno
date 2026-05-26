from __future__ import annotations

import argparse
import glob
import json
from datetime import datetime
from pathlib import Path

import cv2
from ultralytics import YOLO

from paddle_attr import PaddleAttributeExtractor
from person_pipeline import analyze_image
from reid_utils import ReIDEmbeddingExtractor, SimpleReIDMatcher

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_SOURCE = BASE_DIR / "bus.jpg"
DEFAULT_MODEL = BASE_DIR / "yolo26n.pt"
DEFAULT_PDMODEL = BASE_DIR / "inference.pdmodel"
DEFAULT_PDIPARAMS = BASE_DIR / "inference.pdiparams"
DEFAULT_OUTPUT_JSON = BASE_DIR / "runs/detect/person_output/response.json"
DEFAULT_OUTPUT_IMAGE_DIR = BASE_DIR / "runs/detect/person_output"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect people and extract attributes")
    parser.add_argument(
        "--source",
        default=str(DEFAULT_SOURCE),
        help="Input image path, image directory, or glob pattern",
    )
    parser.add_argument(
        "--pdmodel", default=str(DEFAULT_PDMODEL), help="Paddle pdmodel path"
    )
    parser.add_argument(
        "--pdiparams", default=str(DEFAULT_PDIPARAMS), help="Paddle pdiparams path"
    )
    parser.add_argument(
        "--output-json",
        default=str(DEFAULT_OUTPUT_JSON),
        help="Output JSON path (frame list format)",
    )
    parser.add_argument(
        "--output-image-dir",
        default=str(DEFAULT_OUTPUT_IMAGE_DIR),
        help="Directory for annotated images",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.70,
        help="Cosine similarity threshold for ReID matching",
    )
    parser.add_argument(
        "--max-inactive-frames",
        type=int,
        default=15,
        help="How many frames to keep unmatched tracks",
    )
    parser.add_argument(
        "--reid-model-name",
        default="osnet_x1_0",
        help="ReID embedding model name used by torchreid",
    )
    parser.add_argument(
        "--reid-model-path",
        default="",
        help="Optional local path to ReID model weights",
    )
    parser.add_argument(
        "--reid-device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device for ReID embedding extraction",
    )
    parser.add_argument(
        "--reid-verbose",
        action="store_true",
        help="Enable verbose logs for ReID model initialization",
    )
    return parser.parse_args()


def resolve_source_paths(source: str) -> list[Path]:
    source_path = Path(source)
    if source_path.is_file():
        return [source_path]

    if source_path.is_dir():
        return sorted(
            path
            for path in source_path.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )

    matched_paths = sorted(
        Path(path)
        for path in glob.glob(source)
        if Path(path).is_file() and Path(path).suffix.lower() in IMAGE_SUFFIXES
    )
    return matched_paths


def annotate_person_ids(
    image_bgr: cv2.typing.MatLike, persons: list[dict[str, object]]
) -> None:
    for person in persons:
        bbox = person["bbox"]
        x1, y1, _, _ = [int(value) for value in bbox]
        person_id = person["person_id"]
        cv2.putText(
            image_bgr,
            f"id:{person_id}",
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )


def process_frame(
    source_path: Path,
    frame_index: int,
    model: YOLO,
    attr_extractor: PaddleAttributeExtractor,
    reid_extractor: ReIDEmbeddingExtractor,
    matcher: SimpleReIDMatcher,
    output_image_dir: Path,
) -> dict[str, object]:
    analysis = analyze_image(
        source_path=source_path,
        model=model,
        attr_extractor=attr_extractor,
        reid_extractor=reid_extractor,
    )

    output_image_path = output_image_dir / source_path.name
    image_h = int(analysis["image_height"])
    image_w = int(analysis["image_width"])
    annotated = analysis["annotated_image"]

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    detections = analysis["detections"]

    matcher.assign(
        detections,
        frame_index=frame_index,
        frame_width=image_w,
        frame_height=image_h,
    )

    persons: list[dict[str, object]] = []
    for detection in detections:
        persons.append(
            {
                "person_id": detection["person_id"],
                "bbox": detection["bbox"],
                "yolo_confidence": detection["yolo_confidence"],
                "attributes": detection["attributes"],
                "timestamp": timestamp,
            }
        )

    annotate_person_ids(annotated, persons)
    cv2.imwrite(str(output_image_path), annotated)

    return {
        "frame_index": frame_index,
        "image_name": source_path.name,
        "image_path": str(source_path),
        "annotated_image_path": str(output_image_path),
        "persons": persons,
    }


def run() -> None:
    args = parse_args()
    source_paths = resolve_source_paths(args.source)
    if not source_paths:
        raise FileNotFoundError(f"no images found for source: {args.source}")

    attr_extractor = PaddleAttributeExtractor(
        pdmodel_path=str(args.pdmodel),
        pdiparams_path=str(args.pdiparams),
    )
    model = YOLO(str(DEFAULT_MODEL))
    matcher = SimpleReIDMatcher(
        similarity_threshold=args.similarity_threshold,
        max_inactive_frames=args.max_inactive_frames,
        use_center_distance=False,
    )
    reid_extractor = ReIDEmbeddingExtractor(
        model_name=args.reid_model_name,
        model_path=args.reid_model_path,
        device=args.reid_device,
        verbose=args.reid_verbose,
    )

    output_image_dir = Path(args.output_image_dir)
    output_image_dir.mkdir(parents=True, exist_ok=True)

    frames: list[dict[str, object]] = []
    for frame_index, source_path in enumerate(source_paths):
        frames.append(
            process_frame(
                source_path=source_path,
                frame_index=frame_index,
                model=model,
                attr_extractor=attr_extractor,
                reid_extractor=reid_extractor,
                matcher=matcher,
                output_image_dir=output_image_dir,
            )
        )

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(frames, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    person_count = sum(len(frame["persons"]) for frame in frames)
    print(f"done: frames={len(frames)} persons={person_count}")
    print(f"json saved: {output_path}")
    print(f"images saved: {output_image_dir}")


if __name__ == "__main__":
    run()
