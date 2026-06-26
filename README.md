# IR-Colorize: Infrared Image Colorization & Enhancement

**ISRO Problem Statement 10 — Infrared Image Colorization and Enhancement for Improved Object Interpretation**

An end-to-end deep learning framework that **simultaneously super-resolves** low-resolution Infrared (IR) satellite
imagery **and colorizes** it into realistic visible-spectrum (RGB) output, with semantic guardrails to prevent
hallucination. Built for Landsat 8/9 imagery.

```
[Raw IR Input] ──> [Super-Resolution Net] ──> [Colorization Engine] ──> [Enhanced RGB Output]
                                                      ▲
                                            [Semantic Constraints]
```

## Objectives
- **Super-Resolution:** Upscale low-resolution IR bands; recover faint edges and micro-textures.
- **Colorization:** Map monochrome IR/thermal signatures to true-to-life RGB (cold water → deep blue, dense vegetation → green).
- **Semantic Integrity:** A frozen segmentation network constrains output so land-cover meaning is preserved.
- **Downstream boosting:** Outputs measurably improve segmentation/detection accuracy (mIoU) vs. raw IR.

## Pipeline
1. **Geospatial ingest** — Landsat 8/9 L2 (OLI RGB + TIRS/NIR). Co-register, reproject, resample IR to RGB grid with Rasterio/GDAL. Keep 16-bit float; scale per-tile (never global 8-bit crush).
2. **Super-Resolution** — Real-ESRGAN / SRGAN with gradient-intensity loss to penalize blur.
3. **Colorization** — Pix2PixHD (paired) or CUT/CycleGAN (unpaired/temporal mismatch). LDM as advanced alternative.
4. **Semantic constraint** — Frozen SegFormer/U-Net adds a semantic loss when predicted RGB classifies differently from ground truth.

## Tech Stack
| Category | Tools |
|---|---|
| Core | PyTorch, PyTorch Lightning |
| Geospatial | Rasterio, GDAL, Fiona, pyproj |
| Vision | OpenCV, Albumentations, kornia |
| Models | Real-ESRGAN, Pix2PixHD, CUT, SegFormer |
| Tracking | Weights & Biases |
| Metrics | torchmetrics (PSNR/SSIM), clean-fid (FID) |

## Evaluation Targets
- PSNR > 28 dB · SSIM > 0.85 · FID lower-is-better
- Downstream mIoU must increase vs. raw IR
- Inference < 500 ms per 512×512 tile on T4/A10G

## Quick Start
```bash
make setup                 # create venv + install deps
make data                  # download/prepare Landsat tiles
make train STAGE=sr        # train super-resolution stage
make train STAGE=color     # train colorization stage
make infer CKPT=outputs/best.ckpt INPUT=data/tiles/sample.tif
make eval CKPT=outputs/best.ckpt
```

See `CLAUDE.md` for architecture details and developer workflow.
"# IR-Image-Colouring" 
