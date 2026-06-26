# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

End-to-end DL framework for **ISRO Problem Statement 10**: super-resolve + colorize monochrome
Infrared (Landsat 8/9) satellite imagery into realistic RGB, with a semantic guardrail against
hallucination. The project is currently a **scaffold** — module entrypoints exist with documented
contracts but raise `NotImplementedError`; implementation fills them in. Read `README.md` for the
scientific blueprint, this file for how the code is wired.

## Commands

```bash
make setup                          # venv + install requirements + editable install
make data                           # ircolor.data.prepare: scenes -> co-registered tiles
make train STAGE=sr                 # train super-resolution stage
make train STAGE=color              # train colorization stage (STAGE in: sr | color | joint)
make infer CKPT=... INPUT=x.tif     # tiled georeferenced inference
make eval CKPT=...                  # PSNR/SSIM/FID + downstream mIoU + latency
make lint                           # ruff check + mypy
make fmt                            # ruff --fix + black
make test                           # pytest
pytest tests/test_losses.py::test_gradient_loss_positive_on_blur   # single test
```

Config overrides are OmegaConf dotlists appended to the command, **not** flags:
`python -m ircolor.training.train stage=color train.lr=1e-4 model.semantic.enabled=false`.

On Windows, install **GDAL via conda or OSGeo4W before** `pip install rasterio/fiona` — the pip
wheels need the native GDAL library present.

## Architecture

The system is a **two-stage cascade with a loss-only guardrail**, all assembled in
`src/ircolor/models/pipeline.py::IRColorPipeline`:

```
IR tile ──> sr (Real-ESRGAN) ──> colorizer (Pix2PixHD/CUT) ──> RGB
                                         │
                              semantic (frozen SegFormer) ──> loss only
```

- **`sr` runs before `colorizer`** — structure is recovered first, then color is painted on the
  high-res IR. Order matters; don't swap.
- **`semantic` is frozen and never in the inference `forward()` path.** It only feeds
  `losses/objectives.py::semantic_consistency_loss`. Keep it out of `predict.py`.
- Every sub-module is **swappable via config** (`model.sr.arch`, `model.color.arch`,
  `model.semantic.backbone`). Add new architectures behind those keys, not by editing the pipeline.

### Module map (`src/ircolor/`)
- `data/prepare.py` — pipeline stage 1: rasterio/GDAL co-registration + windowed tiling → `data/tiles/{id}_ir.tif`, `{id}_rgb.tif`.
- `data/dataset.py` — `LandsatIRDataset` reads those tile pairs.
- `models/pipeline.py` — the composite model (central abstraction).
- `losses/objectives.py` — `gradient_intensity_loss` (anti-blur), `semantic_consistency_loss` (anti-hallucination).
- `training/train.py` — Lightning entrypoint; `stage` selects what is optimized.
- `inference/predict.py` — tiled inference with seam blending + CRS passthrough.
- `eval/evaluate.py` — the three evaluation vectors.

## Non-negotiable invariants

These encode the hard-won domain constraints; violating them silently breaks results.

1. **Stay 16-bit float end-to-end.** Never round-trip IR/thermal through 8-bit PNG — it crushes the
   high dynamic range. The only 8-bit conversion is the final display export.
2. **Normalize per tile from scene statistics** (`per_tile_zscore`/`per_tile_minmax`), never with
   global fixed thresholds. Thermal temperature ranges vary scene to scene.
3. **Preserve CRS + geotransform** through tiling and inference so outputs stay georeferenced for GIS.
4. **For temporal/spatial mismatch between RGB and thermal passes, use unpaired translation**
   (CUT/CycleGAN via `model.color.arch`) instead of pixel-aligned Pix2Pix.
5. **Success is downstream mIoU**, not just PSNR/SSIM — eval must compare segmenter accuracy on
   output vs. raw IR. Don't declare success on reconstruction metrics alone.

## Config

`configs/default.yaml` is the single source of truth (data bands, normalization, model arch
selection, loss weights, eval targets). Landsat band indices live here (RGB = OLI B4/B3/B2,
IR = NIR B5 + TIRS B10) — change bands there, not in code.

## Stack
PyTorch + Lightning · Rasterio/GDAL/Fiona · OpenCV/Albumentations/kornia · Real-ESRGAN, Pix2PixHD,
CUT, SegFormer · torchmetrics + clean-fid · Weights & Biases · Hydra/OmegaConf config.
