import pytest
import torch
import torch.nn as nn
from aa.attacks.external.cornersearch import CornerSearch
from aa.attacks.registry import get_attack_spec, create_attack


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 10, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        h = self.conv(x)
        return self.pool(h).squeeze(-1).squeeze(-1)


def test_cornersearch_registry():
    spec = get_attack_spec("cornersearch")
    assert spec.mode == "progressive"
    assert spec.name == "CornerSearch"


def test_cornersearch_basic_invariants():
    model = TinyModel()
    model.eval()

    torch.manual_seed(42)
    x = torch.rand(2, 3, 32, 32)
    y = torch.tensor([0, 1])

    k_max = 4
    attack = CornerSearch(model=model, k=k_max, max_iter=20, n_max=10, seed=42)
    output = attack.attack(x, y)

    # 1. Shape check
    assert output.x_adv.shape == x.shape
    # 2. Finite check
    assert torch.isfinite(output.x_adv).all()
    # 3. [0, 1] bounds check
    assert (output.x_adv >= 0.0).all()
    assert (output.x_adv <= 1.0).all()

    # 4. Spatial L0 check (<= Kmax)
    diff = output.x_adv - x
    spatial_l0 = (diff.abs().max(dim=1).values > 1e-5).sum(dim=(1, 2)).cpu()
    assert (spatial_l0 <= k_max).all(), f"Spatial L0 {spatial_l0} exceeds max K {k_max}"

    # 5. Queries check
    assert output.queries > 0

    # 6. Metadata and upstream diagnostic cross-check
    assert output.metadata is not None
    assert "upstream_pixels_changed" in output.metadata
    assert "upstream_success" in output.metadata
    assert "aa2_l0" in output.metadata

    upstream_l0 = torch.tensor(output.metadata["upstream_pixels_changed"])
    aa2_l0 = torch.tensor(output.metadata["aa2_l0"])
    assert (aa2_l0 - upstream_l0).abs().max().item() <= 1


def test_cornersearch_reproducibility():
    model = TinyModel()
    model.eval()

    torch.manual_seed(42)
    x = torch.rand(2, 3, 32, 32)
    y = torch.tensor([0, 1])

    atk1 = CornerSearch(model=model, k=4, max_iter=20, n_max=10, seed=123)
    out1 = atk1.attack(x, y)

    atk2 = CornerSearch(model=model, k=4, max_iter=20, n_max=10, seed=123)
    out2 = atk2.attack(x, y)

    assert torch.equal(out1.x_adv, out2.x_adv)
    assert out1.queries == out2.queries
