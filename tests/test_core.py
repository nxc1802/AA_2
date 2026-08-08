import torch
from aa.metrics import (
    compute_spatial_l0,
    exact_spatial_topk_mask,
    project_l0,
    compute_per_sample_psnr,
    compute_per_sample_ssim
)


def test_spatial_l0():
    delta = torch.zeros(2, 3, 32, 32)
    # Modify 3 spatial locations for sample 0
    delta[0, 0, 0, 0] = 0.5
    delta[0, 1, 0, 0] = 0.2
    delta[0, 2, 5, 5] = 0.1
    delta[0, 0, 10, 10] = 0.8

    # Modify 1 spatial location for sample 1
    delta[1, 1, 2, 2] = 0.9

    l0 = compute_spatial_l0(delta)
    assert l0[0].item() == 3
    assert l0[1].item() == 1


def test_exact_spatial_topk_mask():
    scores = torch.randn(2, 1, 32, 32)
    k = 8
    mask = exact_spatial_topk_mask(scores, k)
    assert mask.shape == (2, 1, 32, 32)
    assert mask[0].sum().item() == k
    assert mask[1].sum().item() == k


def test_project_l0():
    delta = torch.randn(2, 3, 32, 32)
    k = 5
    proj_delta = project_l0(delta, k)
    l0 = compute_spatial_l0(proj_delta)
    assert l0[0].item() <= k
    assert l0[1].item() <= k


def test_psnr_identical():
    x = torch.rand(2, 3, 32, 32)
    psnr = compute_per_sample_psnr(x, x)
    assert (psnr >= 99.0).all()


def test_ssim_identical():
    x = torch.rand(2, 3, 32, 32)
    ssim = compute_per_sample_ssim(x, x)
    assert (ssim > 0.99).all()
