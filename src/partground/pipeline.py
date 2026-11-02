"""Text-guided detection followed by box-prompted segmentation.

Model imports are deliberately lazy so the geometry and integration tests can
run without downloading weights. See README for model and data requirements.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


class GroundedSAM:
    """Hugging Face Grounding DINO + SAM inference wrapper."""

    def __init__(self, device: str = "auto") -> None:
        try:
            import torch
            from transformers import SamModel, SamProcessor, pipeline
        except ImportError as exc:
            raise RuntimeError("Install the model dependencies: pip install -e '.[models]'") from exc

        if device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be auto, cpu, or cuda")
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but PyTorch cannot see a CUDA GPU")
        self.torch = torch
        self.device = "cuda" if device != "cpu" and torch.cuda.is_available() else "cpu"
        self.detector = pipeline(
            "zero-shot-object-detection",
            model="IDEA-Research/grounding-dino-tiny",
            device=0 if self.device == "cuda" else -1,
        )
        self.processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
        self.segmenter = SamModel.from_pretrained("facebook/sam-vit-base").to(self.device).eval()

    def detect(self, image: Image.Image, prompt: str, threshold: float) -> list[dict[str, Any]]:
        return self.detector(image, candidate_labels=[prompt], threshold=threshold)

    def segment(self, image: Image.Image, boxes: list[list[int]]) -> list[np.ndarray]:
        if not boxes:
            return []
        inputs = self.processor(images=image, input_boxes=[boxes], return_tensors="pt").to(self.device)
        with self.torch.inference_mode():
            outputs = self.segmenter(**inputs, multimask_output=False)
        masks = self.processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu(),
        )[0]
        # One input image, one candidate mask per input box.
        return [masks[index, 0].numpy().astype(bool) for index in range(len(boxes))]


def run(image: Image.Image, prompt: str, threshold: float, model: Any) -> tuple[list[dict], list[np.ndarray]]:
    """Run a detector/segmenter and normalize the detections for serialization."""
    if not prompt.strip():
        raise ValueError("The text prompt must not be empty")
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")
    width, height = image.size
    if width < 1 or height < 1:
        raise ValueError("The image must not be empty")
    detections = []
    for result in model.detect(image, prompt.strip(), threshold):
        box = result["box"]
        x0 = max(0, min(width, int(box["xmin"])))
        y0 = max(0, min(height, int(box["ymin"])))
        x1 = max(0, min(width, int(box["xmax"])))
        y1 = max(0, min(height, int(box["ymax"])))
        if x0 >= x1 or y0 >= y1:
            continue
        detections.append({"label": str(result.get("label", prompt)), "score": float(result["score"]), "box": [x0, y0, x1, y1]})
    masks = model.segment(image, [d["box"] for d in detections]) if detections else []
    if len(masks) != len(detections):
        raise ValueError("The segmenter returned a different number of masks and boxes")
    masks = [np.asarray(mask, dtype=bool) for mask in masks]
    if any(mask.shape != (height, width) for mask in masks):
        raise ValueError("Every mask must match the image dimensions")
    return detections, masks


def save_results(image: Image.Image, detections: list[dict], masks: list[np.ndarray], output: Path) -> None:
    """Write a color overlay, one binary PNG per detection, and metadata."""
    if len(detections) != len(masks):
        raise ValueError("The number of detections and masks must match")
    output.mkdir(parents=True, exist_ok=True)
    base = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    palette = [(31, 193, 169), (245, 154, 61), (145, 109, 231), (235, 96, 118)]
    records = []
    for index, (detection, mask) in enumerate(zip(detections, masks)):
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != base.shape[:2]:
            raise ValueError("Every mask must match the image dimensions")
        color = np.asarray(palette[index % len(palette)], dtype=float)
        base[mask] = (0.55 * base[mask] + 0.45 * color).astype(np.uint8)
        filename = f"mask_{index:03d}.png"
        Image.fromarray(mask.astype(np.uint8) * 255).save(output / filename)
        records.append({**detection, "mask": filename})
    overlay = Image.fromarray(base)
    draw = ImageDraw.Draw(overlay)
    for index, detection in enumerate(detections):
        color = palette[index % len(palette)]
        draw.rectangle(detection["box"], outline=color, width=3)
        draw.text((detection["box"][0] + 3, detection["box"][1] + 3),
                  f"{detection['label']} {detection['score']:.2f}", fill=color)
    overlay.save(output / "overlay.png")
    (output / "detections.json").write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
