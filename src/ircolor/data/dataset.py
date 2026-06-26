"""Geospatial-aware dataset for IR->RGB super-resolution + colorization.

KEY INVARIANT: data stays 16-bit float until the very last step. We never round-trip
through 8-bit PNG, which would crush the high dynamic range of thermal bands. All
normalization is computed PER TILE from scene statistics, not from global thresholds.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class TilePair:
    """One co-registered training sample. Arrays are float32, CHW, 16-bit dynamic range preserved."""
    ir: np.ndarray          # (C_ir, H, W) low-res IR/thermal input
    rgb: np.ndarray         # (3, H, W) high-res RGB target (None for unpaired/CUT mode)
    crs: str                # source coordinate reference system (kept for georeferenced export)
    transform: tuple        # affine geotransform of the tile


class LandsatIRDataset(Dataset):
    """Reads pre-tiled .tif pairs produced by `ircolor.data.prepare`.

    Tiles live in `tiles_dir` as `{id}_ir.tif` and `{id}_rgb.tif`. See data/prepare.py
    for the co-registration + windowed-tiling step that creates them.
    """

    def __init__(self, tiles_dir: str | Path, normalize: str = "per_tile_zscore", paired: bool = True):
        self.tiles_dir = Path(tiles_dir)
        self.normalize = normalize
        self.paired = paired
        self.ids = sorted(p.stem.removesuffix("_ir") for p in self.tiles_dir.glob("*_ir.tif"))

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int) -> TilePair:  # pragma: no cover - IO heavy
        raise NotImplementedError("Implement rasterio windowed read + per-tile normalize here.")
