from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

from paddle_attr import PaddleAttributeExtractor
from person_pipeline import analyze_image
from reid_utils import ReIDEmbeddingExtractor, cosine_similarity

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_ROOT_DIR = BASE_DIR / "datasets/resized_samples"
DEFAULT_MODEL = BASE_DIR / "yolo26n.pt"
DEFAULT_PDMODEL = BASE_DIR / "inference.pdmodel"
DEFAULT_PDIPARAMS = BASE_DIR / "inference.pdiparams"
DEFAULT_OUTPUT_DIR = BASE_DIR / "runs/detect/person_output/batch_quality_eval"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate resolution impact on person ReID across multiple images"
    )
    parser.add_argument(
        "--root-dir",
        default=str(DEFAULT_ROOT_DIR),
        help="Root directory containing resolution subdirectories (e.g. 1024, 512...)",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for batch quality evaluation outputs",
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
        "--limit",
        type=int,
        default=None,
        help="Limit the number of base images to process for testing",
    )
    parser.add_argument(
        "--reid-model-name",
        default="osnet_x1_0",
        help="ReID embedding model name",
    )
    parser.add_argument(
        "--reid-device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device for ReID extraction",
    )
    return parser.parse_args()


def collect_image_groups(root_dir: Path) -> dict[str, dict[int, Path]]:
    """
    Groups images by their filename across resolution directories.
    Returns: { 'base_name': { resolution_int: path } }
    """
    groups = defaultdict(dict)
    # Subdirectories are expected to be integers representing resolution (e.g. "1024")
    for res_dir in root_dir.iterdir():
        if not res_dir.is_dir() or not res_dir.name.isdigit():
            continue

        resolution = int(res_dir.name)
        for img_path in res_dir.iterdir():
            if img_path.suffix.lower() in IMAGE_SUFFIXES:
                groups[img_path.name][resolution] = img_path

    return groups


def select_reference_detection(detections: list[dict[str, object]]) -> dict[str, object]:
    """Selects the detection with the largest bbox area."""
    return max(detections, key=lambda d: float(d["bbox_area"]))


def find_best_match(
    detections: list[dict[str, object]], reference_feature: np.ndarray
) -> tuple[dict[str, object] | None, float | None]:
    """Finds the detection most similar to the reference feature."""
    best_det = None
    best_sim = -1.0
    for det in detections:
        sim = cosine_similarity(det["feature"], reference_feature)
        if sim > best_sim:
            best_sim = sim
            best_det = det
    return best_det, best_sim if best_det else None


def run_batch_eval() -> None:
    args = parse_args()
    root_dir = Path(args.root_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Collect groups
    image_groups = collect_image_groups(root_dir)
    if not image_groups:
        logger.error(f"No image groups found in {root_dir}")
        return

    base_names = sorted(image_groups.keys())
    if args.limit:
        base_names = base_names[: args.limit]

    logger.info(f"Processing {len(base_names)} image groups...")

    # 2. Initialize models
    attr_extractor = PaddleAttributeExtractor(args.pdmodel, args.pdiparams)
    model = YOLO(str(DEFAULT_MODEL))
    reid_extractor = ReIDEmbeddingExtractor(
        model_name=args.reid_model_name, device=args.reid_device
    )

    batch_results = []

    # 3. Process each group
    for base_name in tqdm(base_names, desc="Evaluating images"):
        resolutions = sorted(image_groups[base_name].keys(), reverse=True)
        if not resolutions:
            continue

        # Highest resolution is reference
        ref_res = resolutions[0]
        ref_path = image_groups[base_name][ref_res]

        try:
            ref_analysis = analyze_image(
                ref_path, model, attr_extractor, reid_extractor, args.det_conf
            )
        except Exception as e:
            logger.warning(f"Failed to analyze reference {ref_path}: {e}")
            continue

        if not ref_analysis["detections"]:
            # Skip if reference has no person
            continue

        ref_det = select_reference_detection(ref_analysis["detections"])
        ref_feature = ref_det["feature"]

        group_record = {
            "base_name": base_name,
            "reference_resolution": ref_res,
            "variants": [],
        }

        # Evaluate all resolutions (including ref itself for consistency)
        for res in sorted(resolutions, reverse=True):
            img_path = image_groups[base_name][res]
            try:
                analysis = analyze_image(
                    img_path, model, attr_extractor, reid_extractor, args.det_conf
                )
                best_det, similarity = find_best_match(
                    analysis["detections"], ref_feature
                )

                record = {
                    "resolution": res,
                    "detected": best_det is not None,
                    "confidence": (
                        float(best_det["yolo_confidence"]) if best_det else None
                    ),
                    "similarity": float(similarity) if similarity is not None else None,
                }
                group_record["variants"].append(record)
            except Exception as e:
                logger.warning(f"Failed to analyze {img_path}: {e}")

        batch_results.append(group_record)

    # 4. Aggregate metrics
    res_stats = defaultdict(lambda: {"conf": [], "sim": [], "detect_count": 0, "total": 0})
    for entry in batch_results:
        for var in entry["variants"]:
            res = var["resolution"]
            stats = res_stats[res]
            stats["total"] += 1
            if var["detected"]:
                stats["detect_count"] += 1
                if var["confidence"] is not None:
                    stats["conf"].append(var["confidence"])
                if var["similarity"] is not None:
                    stats["sim"].append(var["similarity"])

    summary_data = []
    sorted_res = sorted(res_stats.keys(), reverse=True)
    for res in sorted_res:
        s = res_stats[res]
        summary_data.append(
            {
                "resolution": res,
                "detection_rate": s["detect_count"] / s["total"] if s["total"] > 0 else 0,
                "mean_confidence": np.mean(s["conf"]) if s["conf"] else 0,
                "mean_similarity": np.mean(s["sim"]) if s["sim"] else 0,
                "sample_count": s["total"],
            }
        )

    # 5. Save results
    with (output_dir / "batch_results.json").open("w") as f:
        json.dump(batch_results, f, indent=2, ensure_ascii=False)

    with (output_dir / "batch_summary.json").open("w") as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)

    # 6. Plotting
    plot_path = output_dir / "batch_quality_plot.png"
    fig, ax1 = plt.subplots(figsize=(10, 6))

    x_labels = [str(r["resolution"]) for r in summary_data]
    x_pos = np.arange(len(x_labels))

    sims = [r["mean_similarity"] for r in summary_data]
    confs = [r["mean_confidence"] for r in summary_data]
    rates = [r["detection_rate"] for r in summary_data]

    ax1.set_xlabel("Resolution")
    ax1.set_ylabel("Score (Similarity / Confidence)", color="tab:blue")
    ax1.plot(x_pos, sims, marker="o", label="Mean ReID Similarity", color="tab:blue")
    ax1.plot(
        x_pos, confs, marker="s", label="Mean YOLO Confidence", color="tab:cyan"
    )
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.set_ylim(0, 1.05)

    ax2 = ax1.twinx()
    ax2.set_ylabel("Detection Rate", color="tab:red")
    ax2.plot(x_pos, rates, marker="x", label="Detection Rate", color="tab:red", linestyle="--")
    ax2.tick_params(axis="y", labelcolor="tab:red")
    ax2.set_ylim(0, 1.05)

    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(x_labels)
    ax1.grid(True, alpha=0.3)
    fig.legend(loc="upper right", bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)
    plt.title(f"Batch Resolution Impact Evaluation (n={len(batch_results)})")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=200)

    logger.info(f"Evaluation complete. Results saved to {output_dir}")
    logger.info(f"Summary: {output_dir / 'batch_summary.json'}")
    logger.info(f"Plot: {plot_path}")


if __name__ == "__main__":
    run_batch_eval()
