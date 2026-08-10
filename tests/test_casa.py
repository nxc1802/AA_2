import torch
import torch.nn as nn
from aa.attacks import create_attack, ATTACK_REGISTRY
from aa.attacks.casa import CoalitionSparseAttack
from aa.metrics import compute_spatial_l0


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 10, kernel_size=3, padding=1)

    def forward(self, x):
        return self.conv(x).mean(dim=[2, 3])


def test_casa_registry():
    assert "casa" in ATTACK_REGISTRY
    assert "ours_v2" in ATTACK_REGISTRY
    spec = ATTACK_REGISTRY["casa"]
    assert spec.mode == "budget"
    assert spec.factory == CoalitionSparseAttack


def test_casa_attack_contract():
    model = DummyModel()
    x = torch.rand(2, 3, 16, 16)
    y = torch.tensor([0, 1])
    k = 4

    attack = create_attack("casa", model=model, k=k, steps=5, inner_steps=3, repair_steps=2)
    output = attack.attack(x, y)

    assert output.x_adv.shape == x.shape
    assert (output.x_adv >= 0.0).all() and (output.x_adv <= 1.0).all()

    l0 = compute_spatial_l0(output.x_adv - x)
    assert (l0 <= k).all(), f"CASA attack exceeded budget k={k}, got l0={l0}"
    assert output.forward_evals > 0
    assert output.backward_evals > 0
