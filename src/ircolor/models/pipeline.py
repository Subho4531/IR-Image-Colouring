"""The composite model: SR -> Colorize, with an optional frozen semantic guardrail.

This is the central abstraction. Each sub-module is swappable via config (`model.*.arch`):
  - sr:        Real-ESRGAN / SRGAN          (recovers structure, runs first)
  - color:     Pix2PixHD / CUT / CycleGAN   (IR->RGB translation)
  - semantic:  frozen SegFormer/U-Net       (loss only, never trained, never in inference path)

Training can run a single stage (`stage=sr` or `stage=color`) or `stage=joint`.
The semantic network contributes ONLY to the loss -- it is excluded from `forward` at inference.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class IRColorPipeline(nn.Module):
    def __init__(self, sr: nn.Module, colorizer: nn.Module, semantic: nn.Module | None = None):
        super().__init__()
        self.sr = sr
        self.colorizer = colorizer
        self.semantic = semantic  # frozen; used by losses, not by forward()
        if self.semantic is not None:
            for p in self.semantic.parameters():
                p.requires_grad_(False)

    def forward(self, ir: torch.Tensor) -> torch.Tensor:
        """ir: (B, C_ir, H, W) 16-bit-range float -> rgb: (B, 3, H*scale, W*scale)."""
        hr_ir = self.sr(ir)
        rgb = self.colorizer(hr_ir)
        return rgb
