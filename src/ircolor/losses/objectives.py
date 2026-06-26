"""Loss terms. Total = w_pix*L1 + w_adv*GAN + w_grad*Gradient + w_sem*Semantic.

Gradient-intensity loss penalizes blurry edges during super-resolution.
Semantic loss penalizes the colorizer when predicted RGB classifies (via the frozen
segmentation net) differently from the ground-truth RGB -- this is the anti-hallucination
guardrail described in the project blueprint.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def gradient_intensity_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Penalize edge/gradient mismatch (Sobel-like finite differences)."""
    def grad(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gx = x[..., :, 1:] - x[..., :, :-1]
        gy = x[..., 1:, :] - x[..., :-1, :]
        return gx, gy

    pgx, pgy = grad(pred)
    tgx, tgy = grad(target)
    return F.l1_loss(pgx, tgx) + F.l1_loss(pgy, tgy)


def semantic_consistency_loss(
    pred_rgb: torch.Tensor, target_rgb: torch.Tensor, frozen_seg: torch.nn.Module
) -> torch.Tensor:
    """KL/CE between land-cover logits of predicted vs. ground-truth RGB. `frozen_seg` is eval-only."""
    with torch.no_grad():
        target_logits = frozen_seg(target_rgb)
    pred_logits = frozen_seg(pred_rgb)
    return F.kl_div(
        F.log_softmax(pred_logits, dim=1), F.softmax(target_logits, dim=1), reduction="batchmean"
    )
