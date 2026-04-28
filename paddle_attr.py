from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from paddle.inference import Config, create_predictor

ATTRIBUTE_KEYS = (
    "hat",
    "glasses",
    "short_sleeve",
    "long_sleeve",
    "trousers",
    "handbag",
    "shoulderbag",
    "backpack",
    "hold_objects_in_front",
)

ATTRIBUTE_INDEX_MAP = {
    0: "hat",
    1: "glasses",
    2: "short_sleeve",
    3: "long_sleeve",
    11: "trousers",
    15: "handbag",
    16: "shoulderbag",
    17: "backpack",
    18: "hold_objects_in_front",
}


class PaddleAttributeExtractor:
    def __init__(
        self,
        pdmodel_path: str,
        pdiparams_path: str,
        threshold: float = 0.5,
    ) -> None:
        self.threshold = threshold

        model_path = Path(pdmodel_path)
        params_path = Path(pdiparams_path)
        if not model_path.is_file():
            raise FileNotFoundError(f"pdmodel not found: {model_path}")
        if not params_path.is_file():
            raise FileNotFoundError(f"pdiparams not found: {params_path}")

        config = Config(str(model_path), str(params_path))
        config.disable_gpu()
        config.disable_mkldnn()

        self.predictor = create_predictor(config)
        self.input_name = self.predictor.get_input_names()[0]
        self.output_name = self.predictor.get_output_names()[0]

    def predict_attributes(self, image_rgb: np.ndarray) -> dict[str, bool]:
        attrs = {key: False for key in ATTRIBUTE_KEYS}
        if image_rgb.size == 0:
            return attrs

        resized = cv2.resize(image_rgb, (192, 256), interpolation=cv2.INTER_LINEAR)
        data = resized.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        data = (data - mean) / std
        input_data = np.expand_dims(data.transpose((2, 0, 1)), axis=0).astype(
            np.float32
        )

        input_handle = self.predictor.get_input_handle(self.input_name)
        input_handle.copy_from_cpu(input_data)
        self.predictor.run()

        output_handle = self.predictor.get_output_handle(self.output_name)
        pred = output_handle.copy_to_cpu()[0]

        for idx, key in ATTRIBUTE_INDEX_MAP.items():
            attrs[key] = bool(float(pred[idx]) > self.threshold)
        return attrs
