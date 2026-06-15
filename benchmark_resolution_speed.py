from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import pandas as pd
import torch
from tqdm import tqdm
from ultralytics import YOLO

from paddle_attr import PaddleAttributeExtractor
from person_pipeline import clamp_bbox
from reid_utils import ReIDEmbeddingExtractor

# Constants and Defaults
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_YOLO_MODEL = BASE_DIR / "yolo26n.pt"
DEFAULT_PDMODEL = BASE_DIR / "inference.pdmodel"
DEFAULT_PDIPARAMS = BASE_DIR / "inference.pdiparams"
DEFAULT_DATA_DIR = BASE_DIR / "datasets/resized_samples"
DEFAULT_OUTPUT_DIR = BASE_DIR / "runs/detect/person_output/benchmark_speed"
MAX_IMAGES_PER_RES = 1004
WARMUP_COUNT = 5


class PersonPipeline:
    def __init__(
        self,
        yolo_path: str,
        pdmodel_path: str,
        pdiparams_path: str,
        device: str = "auto",
    ) -> None:
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"Initializing models on {device}...")
        self.model = YOLO(yolo_path)
        self.model.to(device)
        
        self.attr_extractor = PaddleAttributeExtractor(pdmodel_path, pdiparams_path)
        # PaddleAttributeExtractor currently forces CPU in its __init__, 
        # but we follow its existing implementation.
        
        self.reid_extractor = ReIDEmbeddingExtractor(device=device)

    def analyze_with_timing(self, image_path: Path) -> dict[str, float | int] | None:
        """Analyze image and return timing metrics in seconds."""
        try:
            # 1. Image Loading
            t0 = time.perf_counter()
            image_bgr = cv2.imread(str(image_path))
            if image_bgr is None:
                return None
            t_load = time.perf_counter() - t0

            image_h, image_w = image_bgr.shape[:2]

            # 2. Detection (YOLO)
            t1 = time.perf_counter()
            # verbose=False to minimize overhead
            result = self.model.predict(
                source=image_bgr, conf=0.25, classes=[0], verbose=False
            )[0]
            t_det = time.perf_counter() - t1

            # 3. Feature Extraction (ReID + Attributes)
            t_feat = 0.0
            detections_count = 0
            
            if result.boxes is not None and len(result.boxes) > 0:
                person_boxes = result.boxes.xyxy.cpu().tolist()
                for raw_box in person_boxes:
                    person_box = [float(v) for v in raw_box]
                    clamped = clamp_bbox(person_box, image_w, image_h)
                    if clamped is None:
                        continue

                    x1, y1, x2, y2 = clamped
                    crop_bgr = image_bgr[y1:y2, x1:x2]
                    
                    # Measurement start for features
                    tf_start = time.perf_counter()
                    
                    # ReID
                    _ = self.reid_extractor.extract(crop_bgr)
                    
                    # Attributes (requires RGB)
                    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
                    _ = self.attr_extractor.predict_attributes(crop_rgb)
                    
                    t_feat += (time.perf_counter() - tf_start)
                    detections_count += 1

            return {
                "load_time": t_load,
                "det_time": t_det,
                "feat_time": t_feat,
                "total_time": t_load + t_det + t_feat,
                "detections": detections_count,
            }
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"\nOOM error encountered for {image_path.name}")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                return {"error": "OOM"}
            raise e
        except Exception as e:
            print(f"\nError processing {image_path.name}: {e}")
            return None


def run_benchmark():
    parser = argparse.ArgumentParser(description="Benchmark resolution impact on speed")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--max-images", type=int, default=MAX_IMAGES_PER_RES)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Initialize Pipeline
    pipeline = PersonPipeline(
        yolo_path=str(DEFAULT_YOLO_MODEL),
        pdmodel_path=str(DEFAULT_PDMODEL),
        pdiparams_path=str(DEFAULT_PDIPARAMS),
    )

    # 2. Identify and sort resolutions
    res_dirs = []
    for d in data_dir.iterdir():
        if d.is_dir() and d.name.isdigit():
            res_dirs.append(d)
    
    # Sort by resolution value
    res_dirs.sort(key=lambda x: int(x.name))

    results = []

    print(f"Starting benchmark for {len(res_dirs)} resolutions...")

    for res_dir in res_dirs:
        resolution = int(res_dir.name)
        image_paths = sorted(list(res_dir.glob("*.jpg")))
        if not image_paths:
            continue
        
        # Limit images
        test_images = image_paths[:args.max_images + WARMUP_COUNT]
        
        print(f"\nTesting Resolution: {resolution} ({len(test_images)} images)")
        
        res_metrics = []
        
        for i, img_path in enumerate(tqdm(test_images, desc=f"Res {resolution}")):
            metric = pipeline.analyze_with_timing(img_path)
            
            # Skip warmup
            if i < WARMUP_COUNT:
                continue
            
            if metric and "error" not in metric:
                res_metrics.append(metric)
            elif metric and metric.get("error") == "OOM":
                # If OOM happens, we stop this resolution but keep results
                break

        if not res_metrics:
            print(f"No metrics collected for resolution {resolution}")
            continue

        # Calculate statistics (ms)
        total_times = [m["total_time"] * 1000 for m in res_metrics]
        det_times = [m["det_time"] * 1000 for m in res_metrics]
        feat_times = [m["feat_time"] * 1000 for m in res_metrics]
        
        avg_total = statistics.mean(total_times)
        avg_det = statistics.mean(det_times)
        avg_feat = statistics.mean(feat_times)
        fps = 1000.0 / avg_total if avg_total > 0 else 0
        
        results.append({
            "resolution": resolution,
            "avg_ms": avg_total,
            "avg_det_ms": avg_det,
            "avg_feat_ms": avg_feat,
            "fps": fps,
            "std_ms": statistics.stdev(total_times) if len(total_times) > 1 else 0,
            "min_ms": min(total_times),
            "max_ms": max(total_times),
            "sample_count": len(res_metrics)
        })

    # 3. Save Results
    df = pd.DataFrame(results)
    csv_path = output_dir / "benchmark_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")

    # 4. Visualization
    plot_results(df, output_dir)


def plot_results(df: pd.DataFrame, output_dir: Path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot 1: Resolution vs FPS
    ax1.plot(df["resolution"], df["fps"], marker='o', linestyle='-', color='b')
    ax1.set_xscale('log', base=2)
    ax1.set_xlabel("Resolution (px)")
    ax1.set_ylabel("Throughput (FPS)")
    ax1.set_title("Resolution vs FPS")
    ax1.grid(True, which="both", ls="-", alpha=0.5)

    # Plot 2: Resolution vs Latency (Stacked)
    ax2.bar(df["resolution"].astype(str), df["avg_det_ms"], label="Detection")
    ax2.bar(df["resolution"].astype(str), df["avg_feat_ms"], bottom=df["avg_det_ms"], label="Feature Extraction")
    ax2.set_xlabel("Resolution (px)")
    ax2.set_ylabel("Average Latency (ms)")
    ax2.set_title("Latency Breakdown by Resolution")
    ax2.legend()
    ax2.grid(True, axis='y', ls="-", alpha=0.5)

    plt.tight_layout()
    plot_path = output_dir / "speed_analysis.png"
    plt.savefig(plot_path)
    print(f"Analysis plot saved to {plot_path}")


if __name__ == "__main__":
    run_benchmark()
