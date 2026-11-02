# Text-guided object-part grounding

An exploratory prototype for locating an object part from a text description. It connects a pretrained open-vocabulary detector (Grounding DINO) to a pretrained segmentation model (SAM), and includes a separate utility for approximating a 3D part extent from orthographic views.

This repository is an **AI-assisted reconstruction and refactoring of exploratory research scripts**, prepared from the author's archive. The experiments informed an investigation alongside a VLM research project; they were not presented as the final method of the associated paper. The models are pretrained third-party components, not models trained for this project.

## What is implemented

- Given an image and a phrase, detect candidate boxes with Grounding DINO and use them as SAM prompts. Save an overlay, binary masks and detection metadata.
- Given manually supplied boxes and calibrated axis conventions, intersect orthographic view intervals with known 3D bounds to estimate a part bounding box.
- Run the geometry example and logic tests without model weights or a GPU.

The inference and geometry tools are separate: the geometry command does **not** automatically pair detections across views. Earlier CLIP localization, YOLO segmentation and object-orientation trials are related explorations and are not included in this implementation.

**Validation status:** synthetic geometry and integration tests with a fake model pass. This cleaned implementation has not yet been run end to end with the real model weights or evaluated on a benchmark. Detection quality, part association and 3D accuracy are unmeasured; results will depend on the input and prompt.

## Setup

Python 3.10 or 3.11 is recommended. The lightweight geometry example needs NumPy and Pillow. Model inference additionally requires PyTorch, torchvision and Transformers. A CUDA-capable GPU is recommended for inference; CPU execution may be slow. First install the appropriate CUDA-enabled PyTorch and torchvision builds for your system using the [official PyTorch selector](https://pytorch.org/get-started/locally/).

In PowerShell, from the repository directory:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[models]"
```

On macOS or Linux, activate with `source .venv/bin/activate` instead. Use `python -m pip install -e .` if you only want the geometry example and tests.

The first inference run downloads model weights from Hugging Face, so it needs network access and disk space. If downloads fail, check network access and the model pages' terms. Do not commit downloaded weights or local images to the repository without considering their rights and sensitivity.

## Run part detection and segmentation

```powershell
partground ground --image path\to\image.jpg --text "hammer head" --output outputs\hammer --device cuda
```

Use `--device cpu` if CUDA is unavailable, or omit it for automatic selection. Tune `--threshold` between 0 and 1 for the detector; the default is 0.3. Inspect the masks rather than assuming that the detector has identified the intended part.

The output directory contains `overlay.png`, one `mask_000.png`-style binary image per detection, and `detections.json` containing labels, scores, pixel boxes and mask filenames. With no detections, the overlay is the original image and the JSON list is empty.

## Intersect orthographic views

```powershell
partground intersect examples\orthographic_boxes.json
```

The example is synthetic. It returns minimum and maximum coordinates and the center of the intersection inside the supplied 3D bounds. Each box is `[left, top, right, bottom]` in pixels, paired with `[width, height]`. Each view names the world axis represented by its horizontal and vertical image directions. `flip_horizontal` and `flip_vertical` reverse those directions when needed; both default to `false`.

The method assumes each entire image axis maps linearly across the corresponding supplied 3D bounds, with orthographic projection and consistent axis alignment. Cropping, padding, perspective, inconsistent views or incorrect bounds invalidate that mapping. Every 3D axis must be constrained; disjoint intervals return `null`. The output is an axis-aligned approximation from boxes, not a reconstructed 3D mask or a general camera-calibration solution.

## Test

```powershell
python -m unittest discover -s tests -v
```

The integration tests use a fake detector/segmenter. A real-model smoke test with a known image and manually inspected output is still needed.

## Model credits

- [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO), via [`IDEA-Research/grounding-dino-tiny`](https://huggingface.co/IDEA-Research/grounding-dino-tiny).
- [Segment Anything](https://github.com/facebookresearch/segment-anything), via [`facebook/sam-vit-base`](https://huggingface.co/facebook/sam-vit-base).
- The model interfaces use [Hugging Face Transformers](https://huggingface.co/docs/transformers/).

Review the upstream code, model and dataset licenses before reuse.
