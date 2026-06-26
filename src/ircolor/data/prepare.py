"""Stage 1 of the pipeline: geospatial ingest -> co-registered tiles.

Responsibilities (see CLAUDE.md "Data pipeline"):
  1. Read Landsat 8/9 L2 scenes (OLI RGB bands + TIRS/NIR bands) with rasterio.
  2. Reproject + resample IR bands onto the RGB grid so pixels are co-registered.
  3. Window into overlapping tiles (tile_size, overlap) preserving CRS + geotransform.
  4. Write {id}_ir.tif / {id}_rgb.tif as 16-bit float -- NO 8-bit conversion.
"""
from __future__ import annotations

import argparse


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    raise NotImplementedError(
        f"Wire rasterio/GDAL co-registration + windowed tiling using {args.config}."
    )


if __name__ == "__main__":
    main()
