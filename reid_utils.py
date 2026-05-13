from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np
import torch
from torchreid.reid.utils.feature_extractor import FeatureExtractor


@dataclass
class TrackState:
    person_id: int
    feature: np.ndarray
    bbox: Tuple[float, float, float, float]
    last_frame_index: int


class ReIDEmbeddingExtractor:
    def __init__(
        self,
        model_name: str = "osnet_x1_0",
        model_path: str = "",
        device: str = "auto",
        verbose: bool = False,
    ) -> None:
        selected_device = device
        if selected_device == "auto":
            selected_device = "cuda" if torch.cuda.is_available() else "cpu"

        self.extractor = FeatureExtractor(
            model_name=model_name,
            model_path=model_path,
            device=selected_device,
            verbose=verbose,
        )

    def extract(self, image_bgr: np.ndarray) -> np.ndarray:
        if image_bgr.size == 0:
            return np.zeros((512,), dtype=np.float32)

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        with torch.no_grad():
            feature_tensor = self.extractor(image_rgb)
        feature = feature_tensor[0].detach().cpu().numpy().astype(np.float32)

        norm = float(np.linalg.norm(feature))
        if norm > 1e-10:
            feature = feature / norm
        return feature


def bbox_center(bbox_xyxy: Sequence[float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = bbox_xyxy
    return (float(x1 + x2) / 2.0, float(y1 + y2) / 2.0)


def calc_iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
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
    if denom <= 0.0:
        return 0.0
    return float(inter_area / denom)


def extract_hsv_hist_feature(
    image_bgr: np.ndarray, bins: Tuple[int, int, int] = (8, 8, 8)
) -> np.ndarray:
    if image_bgr.size == 0:
        return np.zeros((bins[0] * bins[1] * bins[2],), dtype=np.float32)

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist(
        [hsv],
        [0, 1, 2],
        None,
        [bins[0], bins[1], bins[2]],
        [0, 180, 0, 256, 0, 256],
    )
    hist = cv2.normalize(hist, hist).flatten().astype(np.float32)
    return hist


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    denom = np.linalg.norm(vec_a) * np.linalg.norm(vec_b)
    if denom <= 1e-10:
        return 0.0
    return float(np.dot(vec_a, vec_b) / denom)


def normalized_center_distance(
    center_a: Tuple[float, float],
    center_b: Tuple[float, float],
    frame_width: int,
    frame_height: int,
) -> float:
    dx = (center_a[0] - center_b[0]) / max(frame_width, 1)
    dy = (center_a[1] - center_b[1]) / max(frame_height, 1)
    return float(np.sqrt(dx * dx + dy * dy))


class SimpleReIDMatcher:
    def __init__(
        self,
        similarity_threshold: float = 0.70,
        max_inactive_frames: int = 15,
        use_center_distance: bool = False,
        center_distance_threshold: float = 0.25,
    ) -> None:
        self.similarity_threshold = similarity_threshold
        self.max_inactive_frames = max_inactive_frames
        self.use_center_distance = use_center_distance
        self.center_distance_threshold = center_distance_threshold
        self._next_person_id = 1
        self._tracks: Dict[int, TrackState] = {}

    def _drop_inactive(self, frame_index: int) -> None:
        to_delete: List[int] = []
        for person_id, state in self._tracks.items():
            if frame_index - state.last_frame_index > self.max_inactive_frames:
                to_delete.append(person_id)
        for person_id in to_delete:
            del self._tracks[person_id]

    def assign(
        self,
        detections: List[Dict[str, object]],
        frame_index: int,
        frame_width: int = 1,
        frame_height: int = 1,
    ) -> None:
        self._drop_inactive(frame_index)

        used_track_ids = set()
        for det in detections:
            det_feature = det["feature"]

            best_id = None
            best_score = -1.0

            for person_id, state in self._tracks.items():
                if person_id in used_track_ids:
                    continue

                if self.use_center_distance:
                    det_center = det["center"]
                    state_center = bbox_center(state.bbox)
                    dist = normalized_center_distance(
                        det_center,
                        state_center,
                        frame_width,
                        frame_height,
                    )
                    if dist > self.center_distance_threshold:
                        continue

                sim = cosine_similarity(det_feature, state.feature)
                if sim >= self.similarity_threshold and sim > best_score:
                    best_score = sim
                    best_id = person_id

            if best_id is None:
                best_id = self._next_person_id
                self._next_person_id += 1

            used_track_ids.add(best_id)
            det["person_id"] = best_id

            self._tracks[best_id] = TrackState(
                person_id=best_id,
                feature=det_feature,
                bbox=det["bbox"],
                last_frame_index=frame_index,
            )
