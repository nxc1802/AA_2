import unittest
import torch
import torch.nn as nn
from src.core import (
    compute_spatial_l0,
    exact_spatial_topk_mask,
    project_l0,
    compute_per_sample_psnr,
    compute_per_sample_ssim,
    compute_distortion_metrics,
    prepare_model_for_eval,
    set_seed
)
from src.attacks.optimization.pgd0 import PGD0Attack
from src.attacks.optimization.sparse_pgd import SparsePGDAttack
from src.attacks.classical.onepixel import OnePixelAttack
from src.attacks.classical.corner_search import CornerSearchAttack
from src.attacks.blackbox.sparse_rs import SparseRSAttack
from src.attacks.blackbox.brusle import BruSLeAttack
from src.attacks.blackbox.pixle import PixleAttack
from src.attacks.proposed.cpa import CooperativePixelsAttack
from src.attacks.proposed.fcsa import FunctionalCoalitionSparseAttack
from src.attacks.proposed.fmsa import FeatureToMinimalSupportAttack
from src.attacks.proposed.hsa import HypergraphSparseAttack

from src.defenses.preprocessing.gaussian_blur import GaussianBlurDefense
from src.defenses.preprocessing.median_filter import MedianFilterDefense
from src.defenses.preprocessing.jpeg_compression import JPEGCompressionDefense
from src.defenses.preprocessing.tvm import TotalVariationMinimizationDefense

class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 10, kernel_size=3, padding=1)
        self.layer4 = nn.Conv2d(10, 10, kernel_size=3, padding=1) # Conv layer for FMSA hook
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.conv(x)
        x = self.layer4(x)
        x = self.pool(x)
        return x.flatten(1)

class TestAttacks(unittest.TestCase):
    def test_exact_spatial_topk_mask(self):
        scores = torch.rand(4, 1, 32, 32)
        k = 8
        mask = exact_spatial_topk_mask(scores, k)
        self.assertEqual(mask.shape, (4, 1, 32, 32))
        self.assertEqual(mask.dtype, torch.bool)
        counts = mask.flatten(1).sum(dim=1)
        self.assertTrue((counts == k).all().item())

    def test_exact_spatial_topk_mask_k0(self):
        scores = torch.rand(2, 1, 16, 16)
        mask = exact_spatial_topk_mask(scores, 0)
        self.assertEqual(mask.shape, (2, 1, 16, 16))
        self.assertEqual(mask.sum().item(), 0)

    def test_project_l0(self):
        delta = torch.randn(4, 3, 32, 32)
        k = 15
        proj_delta = project_l0(delta, k)
        l0 = compute_spatial_l0(proj_delta)
        self.assertTrue((l0 <= k).all().item())

    def test_prepare_model_for_eval(self):
        model = DummyModel()
        model.train()
        model = prepare_model_for_eval(model)
        self.assertFalse(model.training)
        for p in model.parameters():
            self.assertFalse(p.requires_grad)

    def test_pgd0_attack_budget(self):
        set_seed(42)
        model = DummyModel()
        x = torch.rand(2, 3, 32, 32)
        y = torch.tensor([0, 1])
        k = 5
        attacker = PGD0Attack(model, k=k, steps=5)
        x_adv = attacker.attack(x, y)
        l0 = compute_spatial_l0(x_adv - x)
        self.assertTrue((l0 <= k).all().item())

    def test_spgd_attack_budget(self):
        set_seed(42)
        model = DummyModel()
        x = torch.rand(2, 3, 32, 32)
        y = torch.tensor([0, 1])
        k = 10
        attacker = SparsePGDAttack(model, sparsity_budget=k, steps=5)
        x_adv = attacker.attack(x, y)
        l0 = compute_spatial_l0(x_adv - x)
        self.assertTrue((l0 <= k).all().item())

    def test_sparse_rs_budget(self):
        set_seed(42)
        model = DummyModel()
        x = torch.rand(2, 3, 32, 32)
        y = torch.tensor([0, 1])
        k = 6
        attacker = SparseRSAttack(model, k=k, steps=5)
        x_adv = attacker.attack(x, y)
        l0 = compute_spatial_l0(x_adv - x)
        self.assertTrue((l0 <= k).all().item())

    def test_brusle_budget(self):
        set_seed(42)
        model = DummyModel()
        x = torch.rand(2, 3, 32, 32)
        y = torch.tensor([0, 1])
        k = 9
        attacker = BruSLeAttack(model, k=k, steps=5)
        x_adv = attacker.attack(x, y)
        l0 = compute_spatial_l0(x_adv - x)
        self.assertTrue((l0 <= k).all().item())

    def test_pixle_budget(self):
        set_seed(42)
        model = DummyModel()
        x = torch.rand(2, 3, 32, 32)
        y = torch.tensor([0, 1])
        k = 4
        attacker = PixleAttack(model, k=k, steps=5)
        x_adv = attacker.attack(x, y)
        l0 = compute_spatial_l0(x_adv - x)
        self.assertTrue((l0 <= 2 * k).all().item())

    def test_proposed_methods_budget(self):
        set_seed(42)
        model = DummyModel()
        x = torch.rand(2, 3, 32, 32)
        y = torch.tensor([0, 1])
        k = 8
        
        cpa = CooperativePixelsAttack(model, coalition_size=k, steps=5)
        fcsa = FunctionalCoalitionSparseAttack(model, max_coalition_size=k, steps=5)
        fmsa = FeatureToMinimalSupportAttack(model, support_budget=k, steps=5)
        hsa = HypergraphSparseAttack(model, budget=k, steps=5)

        for attacker in [cpa, fcsa, fmsa, hsa]:
            x_adv = attacker.attack(x, y)
            l0 = compute_spatial_l0(x_adv - x)
            self.assertTrue((l0 <= k).all().item())

    def test_metrics_psnr_ssim(self):
        orig = torch.rand(2, 3, 32, 32)
        adv = orig.clone()
        adv[:, :, :2, :2] += 0.1
        psnr = compute_per_sample_psnr(orig, adv)
        ssim = compute_per_sample_ssim(orig, adv)
        self.assertEqual(psnr.shape, torch.Size([2]))
        self.assertEqual(ssim.shape, torch.Size([2]))
        self.assertTrue(((ssim >= 0.0) & (ssim <= 1.0)).all().item())

    def test_distortion_metrics_success_conditioned(self):
        l0 = torch.tensor([10.0, 0.0, 5.0])
        l2 = torch.tensor([1.2, 0.0, 0.8])
        linf = torch.tensor([0.3, 0.0, 0.2])
        psnr = torch.tensor([30.0, 100.0, 35.0])
        ssim = torch.tensor([0.9, 1.0, 0.95])
        succ_mask = torch.tensor([True, False, True])
        
        res = compute_distortion_metrics(l0, l2, linf, psnr, ssim, None, succ_mask)
        self.assertEqual(res["succ_count"], 2)
        self.assertEqual(res["succ_l0_mean"], 7.5)
        self.assertEqual(res["succ_psnr_mean"], 32.5)

    def test_defenses_execution(self):
        x = torch.rand(2, 3, 32, 32)
        g_blur = GaussianBlurDefense()
        m_filt = MedianFilterDefense()
        jpeg = JPEGCompressionDefense()
        tvm = TotalVariationMinimizationDefense(steps=2)

        for d in [g_blur, m_filt, jpeg, tvm]:
            out = d.defend(x)
            self.assertEqual(out.shape, x.shape)
            self.assertTrue((out >= 0.0).all().item() and (out <= 1.0).all().item())

if __name__ == "__main__":
    unittest.main()
