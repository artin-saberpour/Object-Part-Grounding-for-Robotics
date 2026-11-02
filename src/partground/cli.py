"""Command-line entry points for model inference and geometry."""

import argparse
import json
from pathlib import Path

from PIL import Image, ImageOps

from .geometry import ViewBox, intersect_views
from .pipeline import GroundedSAM, run, save_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Experimental object-part grounding")
    commands = parser.add_subparsers(dest="command", required=True)
    ground = commands.add_parser("ground", help="Run text-guided detection and segmentation")
    ground.add_argument("--image", type=Path, required=True)
    ground.add_argument("--text", required=True, help='Text such as "the head of a hammer"')
    ground.add_argument("--output", type=Path, default=Path("outputs/grounding"))
    ground.add_argument("--threshold", type=float, default=0.3)
    ground.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    geometry = commands.add_parser("intersect", help="Intersect orthographic view boxes")
    geometry.add_argument("input", type=Path, help="JSON file describing bounds and view boxes")
    args = parser.parse_args()
    if args.command == "intersect":
        data = json.loads(args.input.read_text(encoding="utf-8"))
        result = intersect_views([ViewBox(**view) for view in data["views"]], data["bounds_min"], data["bounds_max"])
        print(json.dumps(result, indent=2))
    else:
        with Image.open(args.image) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
        model = GroundedSAM(args.device)
        detections, masks = run(image, args.text, args.threshold, model)
        save_results(image, detections, masks, args.output)
        print(f"Saved {len(detections)} detection(s) to {args.output}")


if __name__ == "__main__":
    main()
