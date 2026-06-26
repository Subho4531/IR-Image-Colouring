"""Inference: IR .tif -> colorized RGB .tif, preserving CRS + geotransform.

Tiles the input with overlap, runs the pipeline per tile, and blends seams so the
georeferenced output mosaics cleanly. Output stays georeferenced for GIS use.
"""
from __future__ import annotations

import argparse


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--input", required=True, help="path to IR .tif")
    ap.add_argument("--output", default="outputs/pred_rgb.tif")
    args = ap.parse_args()
    raise NotImplementedError(
        f"Tiled inference {args.input} -> {args.output} with seam blending + CRS passthrough."
    )


if __name__ == "__main__":
    main()
