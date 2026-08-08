import unittest
import torch
import torch.nn as nn
from src.core import compute_spatial_l0, set_seed, get_best_device, prepare_model_for_eval
from src.attacks.adapters import (
    SparseRSOfficialAdapter,
    CornerSearchOfficialAdapter,
    PGD0OfficialAdapter,
    SparseFoolOfficialAdapter,
    SigmaZeroOfficialAdapter,
    SparsePGDOfficialAdapter,
    HomotopyOfficialAdapter,
    GSEOfficialAdapter
)

class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 10, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        return x.flatten(1)

class TestOfficialAdapters(unittest.TestCase):
    def setUp(self):
        set_seed(42)
        self.device = get_best_device()
        self.model = prepare_model_for_eval(DummyModel(), self.device)
        self.x = torch.rand(2, 3, 32, 32, device=self.device)
        self.y = torch.tensor([0, 1], device=self.device)

    def test_sparse_rs_adapter(self):
        adapter = SparseRSOfficialAdapter(self.model, n_pixels=5, n_queries=10)
        x_adv = adapter.attack(self.x, self.y)
        self.assertEqual(x_adv.shape, self.x.shape)
        l0 = compute_spatial_l0(x_adv - self.x)
        self.assertTrue((l0 <= 5).all().item())

    def test_pgd0_adapter(self):
        adapter = PGD0OfficialAdapter(self.model, k=8, steps=5)
        x_adv = adapter.attack(self.x, self.y)
        self.assertEqual(x_adv.shape, self.x.shape)
        l0 = compute_spatial_l0(x_adv - self.x)
        self.assertTrue((l0 <= 8).all().item())

    def test_sigma_zero_adapter(self):
        adapter = SigmaZeroOfficialAdapter(self.model, k=10, steps=5)
        x_adv = adapter.attack(self.x, self.y)
        self.assertEqual(x_adv.shape, self.x.shape)

    def test_spgd_adapter(self):
        adapter = SparsePGDOfficialAdapter(self.model, sparsity_budget=10, steps=5)
        x_adv = adapter.attack(self.x, self.y)
        self.assertEqual(x_adv.shape, self.x.shape)

    def test_gse_adapter(self):
        adapter = GSEOfficialAdapter(self.model, k=10, steps=5)
        x_adv = adapter.attack(self.x, self.y)
        self.assertEqual(x_adv.shape, self.x.shape)

    def test_corner_search_adapter(self):
        adapter = CornerSearchOfficialAdapter(self.model, k=2, max_iter=10)
        x_adv = adapter.attack(self.x, self.y)
        self.assertEqual(x_adv.shape, self.x.shape)
        self.assertTrue(hasattr(adapter, "last_queries"))
        self.assertEqual(len(adapter.last_queries), 2)
        # Sample 0 is correctly classified so CSattack runs candidate evaluations (> 100)
        self.assertGreater(adapter.last_queries[0], 100)

    def test_adapters_fail_hard_on_invalid_path(self):
        import src.attacks.adapters.sigma_zero_adapter as sz_mod
        old_path = sz_mod.THIRD_PARTY_SIGMA_ZERO
        sz_mod.THIRD_PARTY_SIGMA_ZERO = "/non_existent_third_party_path"
        try:
            adapter = sz_mod.SigmaZeroOfficialAdapter(self.model, k=10, steps=5)
            with self.assertRaises(RuntimeError):
                adapter.attack(self.x, self.y)
        finally:
            sz_mod.THIRD_PARTY_SIGMA_ZERO = old_path

if __name__ == "__main__":
    unittest.main()
