"""Three-vector evaluation (see README "Evaluation Targets"):

  A. Reconstruction quality -- PSNR (>28 dB), SSIM (>0.85), FID (lower better).
  B. Task-based -- downstream mIoU using a pre-trained segmenter must beat raw-IR baseline;
     inference latency < 500 ms / 512x512 tile on T4/A10G.
  C. Qualitative -- export side-by-side grids for human hallucination review.
"""
from __future__ import annotations

import argparse


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    raise NotImplementedError(
        f"Compute PSNR/SSIM/FID + downstream mIoU + latency for {args.ckpt}."
    )


if __name__ == "__main__":
    main()
