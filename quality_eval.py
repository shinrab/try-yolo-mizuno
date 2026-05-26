from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

from paddle_attr import PaddleAttributeExtractor
from person_pipeline import analyze_image
from reid_utils import ReIDEmbeddingExtractor, cosine_similarity

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_SOURCE_DIR = BASE_DIR / "datasets/resized2"
DEFAULT_MODEL = BASE_DIR / "yolo26n.pt"
DEFAULT_PDMODEL = BASE_DIR / "inference.pdmodel"
DEFAULT_PDIPARAMS = BASE_DIR / "inference.pdiparams"
DEFAULT_OUTPUT_DIR = BASE_DIR / "runs/detect/person_output/quality_eval"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
RESOLUTION_PATTERN = re.compile(r"(?P<width>\d+)x(?P<height>\d+)", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the minimum resolution needed for person ReID extraction"
    )
    parser.add_argument(
        "--source-dir",
        default=str(DEFAULT_SOURCE_DIR),
        help="Directory containing resolution-variant images",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for quality evaluation outputs",
    )
    parser.add_argument(
        "--pdmodel", default=str(DEFAULT_PDMODEL), help="Paddle pdmodel path"
    )
    parser.add_argument(
        "--pdiparams", default=str(DEFAULT_PDIPARAMS), help="Paddle pdiparams path"
    )
    parser.add_argument(
        "--det-conf",
        type=float,
        default=0.25,
        help="YOLO confidence threshold for person detection",
    )
    parser.add_argument(
        "--reid-threshold",
        type=float,
        default=0.70,
        help="Minimum cosine similarity required to pass",
    )
    parser.add_argument(
        "--target-policy",
        choices=["max_bbox"],
        default="max_bbox",
        help="Rule to select the reference person when multiple people are detected",
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


def parse_resolution(path: Path) -> dict[str, int | str]:
    match = RESOLUTION_PATTERN.search(path.stem)
    if match is None:
        raise ValueError(f"failed to parse resolution from filename: {path.name}")

    width = int(match.group("width"))
    height = int(match.group("height"))
    return {
        "width": width,
        "height": height,
        "resolution_label": f"{width}x{height}",
        "sort_key": width * height,
    }


def resolve_source_paths(source_dir: Path) -> list[Path]:
    if not source_dir.is_dir():
        raise FileNotFoundError(f"source directory not found: {source_dir}")

    paths = [
        path
        for path in source_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]
    if not paths:
        raise FileNotFoundError(f"no images found in source directory: {source_dir}")

    return sorted(
        paths,
        key=lambda path: (
            int(parse_resolution(path)["sort_key"]),
            int(parse_resolution(path)["width"]),
            int(parse_resolution(path)["height"]),
            path.name,
        ),
        reverse=True,
    )


def select_reference_detection(
    detections: list[dict[str, object]], target_policy: str
) -> dict[str, object]:
    if not detections:
        raise ValueError("reference image does not contain any detected person")

    if target_policy != "max_bbox":
        raise ValueError(f"unsupported target policy: {target_policy}")

    return max(detections, key=lambda detection: float(detection["bbox_area"]))


def choose_best_detection(
    detections: list[dict[str, object]], reference_feature: object
) -> tuple[dict[str, object] | None, float | None]:
    best_detection: dict[str, object] | None = None
    best_similarity: float | None = None

    for detection in detections:
        similarity = cosine_similarity(detection["feature"], reference_feature)
        if best_similarity is None or similarity > best_similarity:
            best_detection = detection
            best_similarity = float(similarity)

    return best_detection, best_similarity


def bbox_area_ratio(detection: dict[str, object], width: int, height: int) -> float:
    image_area = max(width * height, 1)
    return float(float(detection["bbox_area"]) / float(image_area))


def annotate_selected_detection(
    image_bgr: cv2.typing.MatLike,
    detection: dict[str, object] | None,
    output_path: Path,
    label: str,
) -> None:
    annotated = image_bgr.copy()
    if detection is not None:
        x1, y1, x2, y2 = [int(value) for value in detection["bbox"]]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(
            annotated,
            label,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
    cv2.imwrite(str(output_path), annotated)


def build_result_record(
    image_path: Path,
    resolution: dict[str, int | str],
    detection: dict[str, object] | None,
    similarity: float | None,
    reid_threshold: float,
    annotated_image_path: Path,
) -> dict[str, object]:
    detection_found = detection is not None
    passes = bool(
        detection_found and similarity is not None and similarity >= reid_threshold
    )

    failure_reason = ""
    if not detection_found:
        failure_reason = "no_detection"
    elif similarity is None:
        failure_reason = "similarity_unavailable"
    elif similarity < reid_threshold:
        failure_reason = "similarity_below_threshold"

    return {
        "image_name": image_path.name,
        "image_path": str(image_path),
        "width": int(resolution["width"]),
        "height": int(resolution["height"]),
        "resolution_label": str(resolution["resolution_label"]),
        "sort_key": int(resolution["sort_key"]),
        "detection_found": detection_found,
        "selected_bbox": detection["bbox"] if detection_found else None,
        "selected_bbox_area_ratio": (
            bbox_area_ratio(
                detection, int(resolution["width"]), int(resolution["height"])
            )
            if detection_found
            else None
        ),
        "yolo_confidence": (
            float(detection["yolo_confidence"]) if detection_found else None
        ),
        "reid_similarity_to_reference": similarity,
        "attributes": detection["attributes"] if detection_found else None,
        "pass_fail": passes,
        "failure_reason": failure_reason,
        "annotated_image_path": str(annotated_image_path),
    }


def write_csv(results: list[dict[str, object]], output_path: Path) -> None:
    fieldnames = [
        "image_name",
        "resolution_label",
        "detection_found",
        "yolo_confidence",
        "reid_similarity_to_reference",
        "pass_fail",
        "failure_reason",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow({fieldname: result[fieldname] for fieldname in fieldnames})


def write_graph(results: list[dict[str, object]], output_path: Path) -> None:
    labels = [str(result["resolution_label"]) for result in results]
    similarities = [
        (
            float(result["reid_similarity_to_reference"])
            if result["reid_similarity_to_reference"] is not None
            else None
        )
        for result in results
    ]
    confidences = [
        float(result["yolo_confidence"])
        if result["yolo_confidence"] is not None
        else None
        for result in results
    ]
    pass_flags = [bool(result["pass_fail"]) for result in results]
    positions = list(range(len(results)))

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    def compute_axis_limits(values: list[float | None]) -> tuple[float, float]:
        valid_values = [value for value in values if value is not None]
        if not valid_values:
            return 0.0, 1.0

        min_value = min(valid_values)
        max_value = max(valid_values)
        if abs(max_value - min_value) < 1e-6:
            padding = max(abs(max_value) * 0.05, 0.02)
        else:
            padding = (max_value - min_value) * 0.12

        lower = max(0.0, min_value - padding)
        upper = min(1.05, max_value + padding)
        if upper - lower < 0.05:
            midpoint = (upper + lower) / 2.0
            lower = max(0.0, midpoint - 0.025)
            upper = min(1.05, midpoint + 0.025)
        return lower, upper

    similarity_values = [
        value if value is not None else float("nan") for value in similarities
    ]
    axes[0].plot(positions, similarity_values, marker="o", linewidth=2, color="#1f77b4")
    for pos, value, passed in zip(positions, similarities, pass_flags):
        if value is None:
            axes[0].scatter(pos, 0.0, color="#7f7f7f", marker="x", s=70)
            continue
        axes[0].scatter(
            pos,
            value,
            color="#2ca02c" if passed else "#ff7f0e",
            s=70,
            zorder=3,
        )
    axes[0].set_ylabel("ReID similarity")
    similarity_min, similarity_max = compute_axis_limits(similarities)
    axes[0].set_ylim(similarity_min, similarity_max)
    axes[0].grid(True, axis="y", alpha=0.3)
    axes[0].set_title("Quality Evaluation Summary")

    confidence_values = [
        value if value is not None else float("nan") for value in confidences
    ]
    axes[1].plot(positions, confidence_values, marker="s", linewidth=2, color="#9467bd")
    for pos, value, passed in zip(positions, confidences, pass_flags):
        if value is None:
            axes[1].scatter(pos, 0.0, color="#7f7f7f", marker="x", s=70)
            continue
        axes[1].scatter(
            pos,
            value,
            color="#2ca02c" if passed else "#ff7f0e",
            s=70,
            zorder=3,
        )
    axes[1].set_ylabel("YOLO confidence")
    axes[1].set_xlabel("Resolution")
    confidence_min, confidence_max = compute_axis_limits(confidences)
    axes[1].set_ylim(confidence_min, confidence_max)
    axes[1].grid(True, axis="y", alpha=0.3)
    axes[1].set_xticks(positions)
    axes[1].set_xticklabels(labels, rotation=45, ha="right")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def build_summary(
    reference_record: dict[str, object],
    results: list[dict[str, object]],
    reid_threshold: float,
) -> dict[str, object]:
    passing_results = [result for result in results if bool(result["pass_fail"])]
    failing_results = [result for result in results if not bool(result["pass_fail"])]
    lowest_passing = passing_results[-1] if passing_results else None
    first_failing = failing_results[0] if failing_results else None

    return {
        "reference_image_name": reference_record["image_name"],
        "reference_resolution": reference_record["resolution_label"],
        "reference_selection_reason": "highest_resolution_with_detection",
        "reid_threshold": reid_threshold,
        "total_images": len(results),
        "pass_count": len(passing_results),
        "lowest_passing_resolution": (
            lowest_passing["resolution_label"] if lowest_passing is not None else None
        ),
        "lowest_passing_image": (
            lowest_passing["image_name"] if lowest_passing is not None else None
        ),
        "first_failing_resolution": (
            first_failing["resolution_label"] if first_failing is not None else None
        ),
        "first_failing_image": (
            first_failing["image_name"] if first_failing is not None else None
        ),
        "ordered_results": [result["image_name"] for result in results],
    }


def run() -> None:
    args = parse_args()
    source_dir = Path(args.source_dir)
    source_paths = resolve_source_paths(source_dir)

    output_dir = Path(args.output_dir)
    annotated_dir = output_dir / "annotated"
    output_dir.mkdir(parents=True, exist_ok=True)
    annotated_dir.mkdir(parents=True, exist_ok=True)

    attr_extractor = PaddleAttributeExtractor(
        pdmodel_path=str(args.pdmodel),
        pdiparams_path=str(args.pdiparams),
    )
    model = YOLO(str(DEFAULT_MODEL))
    reid_extractor = ReIDEmbeddingExtractor(
        model_name=args.reid_model_name,
        model_path=args.reid_model_path,
        device=args.reid_device,
        verbose=args.reid_verbose,
    )

    image_entries: list[dict[str, object]] = []
    reference_index: int | None = None
    reference_detection: dict[str, object] | None = None

    for index, image_path in enumerate(source_paths):
        resolution = parse_resolution(image_path)
        analysis = analyze_image(
            source_path=image_path,
            model=model,
            attr_extractor=attr_extractor,
            reid_extractor=reid_extractor,
            det_conf=args.det_conf,
        )
        image_entries.append(
            {
                "image_path": image_path,
                "resolution": resolution,
                "analysis": analysis,
            }
        )
        if reference_index is None and analysis["detections"]:
            reference_index = index
            reference_detection = select_reference_detection(
                analysis["detections"],
                target_policy=args.target_policy,
            )

    if reference_index is None or reference_detection is None:
        raise ValueError(
            "no detected person found in any image under the source directory"
        )

    reference_entry = image_entries[reference_index]
    reference_path = reference_entry["image_path"]
    reference_resolution = reference_entry["resolution"]
    reference_analysis = reference_entry["analysis"]
    reference_annotated_path = annotated_dir / reference_path.name
    annotate_selected_detection(
        reference_analysis["image_bgr"],
        reference_detection,
        reference_annotated_path,
        label="reference",
    )
    reference_record = build_result_record(
        image_path=reference_path,
        resolution=reference_resolution,
        detection=reference_detection,
        similarity=1.0,
        reid_threshold=args.reid_threshold,
        annotated_image_path=reference_annotated_path,
    )

    results: list[dict[str, object]] = []
    reference_feature = reference_detection["feature"]

    for index, image_entry in enumerate(image_entries):
        image_path = image_entry["image_path"]
        resolution = image_entry["resolution"]
        analysis = image_entry["analysis"]
        if index == reference_index:
            results.append(reference_record)
            continue

        detection, similarity = choose_best_detection(
            analysis["detections"],
            reference_feature,
        )
        annotated_image_path = annotated_dir / image_path.name
        label = (
            f"sim:{similarity:.3f}"
            if detection is not None and similarity is not None
            else "no-match"
        )
        annotate_selected_detection(
            analysis["image_bgr"],
            detection,
            annotated_image_path,
            label=label,
        )
        results.append(
            build_result_record(
                image_path=image_path,
                resolution=resolution,
                detection=detection,
                similarity=similarity,
                reid_threshold=args.reid_threshold,
                annotated_image_path=annotated_image_path,
            )
        )

    summary = build_summary(
        reference_record=reference_record,
        results=results,
        reid_threshold=args.reid_threshold,
    )

    results_json_path = output_dir / "results.json"
    summary_json_path = output_dir / "summary.json"
    csv_path = output_dir / "results.csv"
    graph_path = output_dir / "quality_plot.png"

    results_json_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary_json_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_csv(results, csv_path)
    write_graph(results, graph_path)

    print(f"done: evaluated_images={len(results)}")
    print(f"results saved: {results_json_path}")
    print(f"summary saved: {summary_json_path}")
    print(f"csv saved: {csv_path}")
    print(f"graph saved: {graph_path}")
    print(f"annotated images saved: {annotated_dir}")


if __name__ == "__main__":
    run()
