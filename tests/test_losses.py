import torch

from ircolor.losses.objectives import gradient_intensity_loss


def test_gradient_loss_zero_on_identical():
    x = torch.rand(2, 3, 16, 16)
    assert gradient_intensity_loss(x, x).item() == 0.0


def test_gradient_loss_positive_on_blur():
    x = torch.rand(2, 3, 16, 16)
    blurred = torch.nn.functional.avg_pool2d(x, 3, 1, 1)
    assert gradient_intensity_loss(blurred, x).item() > 0.0
