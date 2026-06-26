"""Training entrypoint. `stage` selects which sub-network is optimized.

  python -m ircolor.training.train --config configs/default.yaml stage=sr
  python -m ircolor.training.train stage=color train.lr=1e-4   # CLI overrides via OmegaConf

A LightningModule wraps IRColorPipeline; W&B logs loss curves + sample image grids.
"""
from __future__ import annotations

import argparse


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("overrides", nargs="*", help="OmegaConf dotlist, e.g. stage=color train.lr=1e-4")
    args = ap.parse_args()
    raise NotImplementedError(
        f"Build LightningModule + Trainer from {args.config} with overrides {args.overrides}."
    )


if __name__ == "__main__":
    main()
