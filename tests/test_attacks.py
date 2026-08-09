import torch
import torch.nn as nn
from aa.attacks import create_attack, ATTACK_REGISTRY
from aa.metrics import compute_spatial_l0


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # layer4 is needed for FeatureExtractorAdapter hook (feature_guidance=True default)
        self.layer4 = nn.Conv2d(3, 10, kernel_size=3, padding=1)

    def forward(self, x):
        return self.layer4(x).mean(dim=[2, 3])


def test_attacks_contract():
    model = DummyModel()
    x = torch.rand(2, 3, 32, 32)
    y = torch.tensor([0, 1])
    k = 4

    # Use strict=False: benchmark scripts pass shared kwargs and rely on filtering.
    # Strict validation is enforced at the config/paper-run level, not unit test level.
    for name in ["fgsm", "pgd", "ours"]:
        attack_kwargs = {"k": k} if name == "ours" else {}
        attack = create_attack(name, model=model, strict=False, **attack_kwargs)
        output = attack.attack(x, y)

        assert output.x_adv.shape == x.shape
        assert (output.x_adv >= 0.0).all() and (output.x_adv <= 1.0).all()

        spec = ATTACK_REGISTRY[name]
        if spec.mode == "budget":
            l0 = compute_spatial_l0(output.x_adv - x)
            assert (l0 <= k).all(), f"Attack {name} exceeded budget k={k}, got l0={l0}"


def test_strict_kwargs_validation():
    """create_attack() must raise ValueError for unknown kwargs when strict=True (default)."""
    import pytest
    model = DummyModel()
    with pytest.raises(ValueError, match="Unknown kwargs"):
        # 'k' is not a valid param for FGSM
        create_attack("fgsm", model=model, k=4)

    # strict=False should NOT raise
    attack = create_attack("fgsm", model=model, strict=False, k=4)
    assert attack is not None
