import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from partground.pipeline import run, save_results


class FakeModel:
    def detect(self, image, prompt, threshold):
        return [{"box": {"xmin": 1, "ymin": 1, "xmax": 4, "ymax": 3}, "label": prompt, "score": 0.8}]

    def segment(self, image, boxes):
        mask = np.zeros((image.height, image.width), dtype=bool)
        mask[1:3, 1:4] = True
        return [mask for _ in boxes]


class PipelineTests(unittest.TestCase):
    def test_writes_masks_overlay_and_metadata(self):
        image = Image.new("RGB", (5, 4), "white")
        detections, masks = run(image, "tool head", 0.3, FakeModel())
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            save_results(image, detections, masks, output)
            data = json.loads((output / "detections.json").read_text())
            self.assertEqual(data[0]["box"], [1, 1, 4, 3])
            self.assertEqual(data[0]["mask"], "mask_000.png")
            self.assertTrue((output / "overlay.png").exists())
            self.assertEqual(np.asarray(Image.open(output / "mask_000.png")).sum(), 6 * 255)

    def test_empty_detection_does_not_call_segmenter(self):
        class NoDetections:
            def detect(self, *args):
                return []
            def segment(self, *args):
                raise AssertionError("Segmenter should not be called")
        image = Image.new("RGB", (5, 4))
        detections, masks = run(image, "part", 0.3, NoDetections())
        self.assertEqual((detections, masks), ([], []))
        with tempfile.TemporaryDirectory() as directory:
            save_results(image, detections, masks, Path(directory))
            self.assertEqual(json.loads((Path(directory) / "detections.json").read_text()), [])

    def test_rejects_mask_shape_mismatch(self):
        class WrongMask(FakeModel):
            def segment(self, image, boxes):
                return [np.zeros((2, 2))]
        with self.assertRaises(ValueError):
            run(Image.new("RGB", (5, 4)), "part", 0.3, WrongMask())


if __name__ == "__main__":
    unittest.main()
