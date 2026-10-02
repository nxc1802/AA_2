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


def test_casa_gct_and_spatial_nms():
    model = DummyModel()
    x = torch.rand(2, 3, 16, 16)
    y = torch.tensor([0, 1])

    # Test K=1 (GCT active)
    atk_k1 = create_attack("casa", model=model, k=1, steps=3, inner_steps=2, repair_steps=2, enable_gct=True)
    out_k1 = atk_k1.attack(x, y)
    l0_k1 = compute_spatial_l0(out_k1.x_adv - x)
    assert (l0_k1 <= 1).all()

    # Test K=2 (Spatial NMS active)
    atk_k2 = create_attack("casa", model=model, k=2, steps=3, inner_steps=2, repair_steps=2, spatial_nms=True)
    out_k2 = atk_k2.attack(x, y)
    l0_k2 = compute_spatial_l0(out_k2.x_adv - x)
    assert (l0_k2 <= 2).all()


def test_casa_support_operations():
    from aa.attacks.casa.support import Support

    # Batch of 2, 4x4 image
    indices = torch.tensor([[1, 5], [2, 10]])
    supp = Support.from_indices(indices, B=2, H=4, W=4, max_k=2)

    assert (supp.size() == 2).all()
    assert supp.to_spatial_mask().shape == (2, 1, 4, 4)
    assert supp.to_flat_mask().shape == (2, 16)

    # Test get_indices
    idx_list = supp.get_indices()
    assert len(idx_list) == 2
    assert torch.equal(idx_list[0], torch.tensor([1, 5]))
    assert torch.equal(idx_list[1], torch.tensor([2, 10]))

    # Test contains
    assert (supp.contains(torch.tensor([1, 2])) == torch.tensor([True, True])).all()
    assert (supp.contains(torch.tensor([0, 0])) == torch.tensor([False, False])).all()

    # Test swap: remove index 1, add index 0 for sample 0; remove 10, add 15 for sample 1
    new_supp = supp.swap(torch.tensor([1, 10]), torch.tensor([0, 15]))
    assert (new_supp.size() == 2).all()
    assert (new_supp.contains(torch.tensor([0, 15])) == torch.tensor([True, True])).all()
    assert (new_supp.contains(torch.tensor([1, 10])) == torch.tensor([False, False])).all()

    # Test add & remove
    added = supp.add(torch.tensor([0, 15]))
    assert (added.size() == 3).all()
    removed = added.remove(torch.tensor([0, 15]))
    assert (removed.size() == 2).all()


