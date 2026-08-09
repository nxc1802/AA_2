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

    for name in ["fgsm", "pgd", "ours"]:
        attack = create_attack(name, model=model, k=k)
        output = attack.attack(x, y)

        assert output.x_adv.shape == x.shape
        assert (output.x_adv >= 0.0).all() and (output.x_adv <= 1.0).all()

        spec = ATTACK_REGISTRY[name]
        if spec.mode == "budget":
            l0 = compute_spatial_l0(output.x_adv - x)
            assert (l0 <= k).all(), f"Attack {name} exceeded budget k={k}, got l0={l0}"
