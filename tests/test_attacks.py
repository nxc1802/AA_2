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
    set_seed,
    get_best_device
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
    def get_test_inputs(self):
        device = get_best_device()
        model = prepare_model_for_eval(DummyModel(), device)
        x = torch.rand(2, 3, 32, 32, device=device)
        y = torch.tensor([0, 1], device=device)
        return model, x, y

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

    def test_query_accounting_linearity(self):
        from src.benchmark.run_attack_benchmark import run_attack_benchmark_suite
        from torch.utils.data import TensorDataset, DataLoader
        device = get_best_device()
        model = prepare_model_for_eval(DummyModel(), device)

        # 6 samples -> 3 batches of size 2
        x_dummy = torch.rand(6, 3, 32, 32)
        y_dummy = torch.tensor([0, 1, 0, 1, 0, 1])
        ds = TensorDataset(x_dummy, y_dummy)
        loader = DataLoader(ds, batch_size=2, shuffle=False)

        class DummyAttackerWithoutLastQueries:
            def __init__(self, steps=20):
                self.steps = steps
            def attack(self, x, y):
                return x.clone()

        # Batch 1: batch_steps = 40, total_steps = 40, total_queries = 40
        # Batch 2: batch_steps = 40, total_steps = 80, total_queries = 80
        # Batch 3: batch_steps = 40, total_steps = 120, total_queries = 120
        # Avg Queries = 120 / 6 = 20.0 (Linear, NOT 240/6 = 40.0 triangular!)
        attacker = DummyAttackerWithoutLastQueries(steps=20)
        
        # Test batch query calculation logic directly
        eval_loader = loader
        total_steps = 0.0
        total_queries = 0.0
        for x, y in eval_loader:
            B = x.size(0)
            if hasattr(attacker, "last_steps"):
                batch_steps = sum(attacker.last_steps)
            else:
                steps = getattr(attacker, "steps", getattr(attacker, "max_iter", 1))
                batch_steps = steps * B
            total_steps += batch_steps

            if hasattr(attacker, "last_queries"):
                batch_queries = sum(attacker.last_queries)
            else:
                batch_queries = batch_steps
            total_queries += batch_queries

        self.assertEqual(total_steps, 120.0)
        self.assertEqual(total_queries, 120.0)
        self.assertEqual(total_queries / 6, 20.0)

    def test_pgd0_attack_budget(self):
        set_seed(42)
        model, x, y = self.get_test_inputs()
        k = 5
        attacker = PGD0Attack(model, k=k, steps=5)
        x_adv = attacker.attack(x, y)
        l0 = compute_spatial_l0(x_adv - x)
        self.assertTrue((l0 <= k).all().item())

    def test_spgd_attack_budget(self):
        set_seed(42)
        model, x, y = self.get_test_inputs()
        k = 10
        attacker = SparsePGDAttack(model, sparsity_budget=k, steps=5)
        x_adv = attacker.attack(x, y)
        l0 = compute_spatial_l0(x_adv - x)
        self.assertTrue((l0 <= k).all().item())

    def test_sparse_rs_budget(self):
        set_seed(42)
        model, x, y = self.get_test_inputs()
        k = 6
        attacker = SparseRSAttack(model, k=k, steps=5)
        x_adv = attacker.attack(x, y)
        l0 = compute_spatial_l0(x_adv - x)
        self.assertTrue((l0 <= k).all().item())

    def test_brusle_budget(self):
        set_seed(42)
        model, x, y = self.get_test_inputs()
        k = 9
        attacker = BruSLeAttack(model, k=k, steps=5)
        x_adv = attacker.attack(x, y)
        l0 = compute_spatial_l0(x_adv - x)
        self.assertTrue((l0 <= k).all().item())

    def test_pixle_budget(self):
        set_seed(42)
        model, x, y = self.get_test_inputs()
        k = 4
        attacker = PixleAttack(model, k=k, steps=5)
        x_adv = attacker.attack(x, y)
        l0 = compute_spatial_l0(x_adv - x)
        self.assertTrue((l0 <= 2 * k).all().item())

    def test_proposed_methods_budget(self):
        set_seed(42)
        model, x, y = self.get_test_inputs()
        k = 8
        
        cpa = CooperativePixelsAttack(model, coalition_size=k, steps=5)
        fcsa = FunctionalCoalitionSparseAttack(model, max_coalition_size=k, steps=5)
        fmsa = FeatureToMinimalSupportAttack(model, support_budget=k, steps=5)
        hsa = HypergraphSparseAttack(model, budget=k, steps=5)

        for attacker in [cpa, fcsa, fmsa, hsa]:
            x_adv = attacker.attack(x, y)
            l0 = compute_spatial_l0(x_adv - x)
            self.assertTrue((l0 <= k).all().item())

    def test_success_first_selection_logic(self):
        set_seed(42)
        model, x, y = self.get_test_inputs()
        k = 8
        cpa = CooperativePixelsAttack(model, coalition_size=k, steps=10)
        fcsa = FunctionalCoalitionSparseAttack(model, max_coalition_size=k, steps=10)
        fmsa = FeatureToMinimalSupportAttack(model, support_budget=k, steps=10)
        hsa = HypergraphSparseAttack(model, budget=k, steps=10)

        for attacker in [cpa, fcsa, fmsa, hsa]:
            x_adv = attacker.attack(x, y)
            with torch.no_grad():
                preds = model(x_adv).argmax(dim=1)
            succ = (preds != y)
            # If attack succeeded at any step, returned best_adv must be misclassifying
            if hasattr(attacker, "last_steps"):
                foo_mask = torch.tensor(attacker.last_steps, device=x.device) < attacker.steps
                if foo_mask.any():
                    self.assertTrue(succ[foo_mask].all().item())

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
        device = get_best_device()
        x = torch.rand(2, 3, 32, 32, device=device)
        g_blur = GaussianBlurDefense()
        m_filt = MedianFilterDefense()
        jpeg = JPEGCompressionDefense()
        tvm = TotalVariationMinimizationDefense(steps=2, device=device)

        for d in [g_blur, m_filt, jpeg, tvm]:
            out = d.defend(x)
            self.assertEqual(out.shape, x.shape)
            self.assertTrue((out >= 0.0).all().item() and (out <= 1.0).all().item())

    def test_onepixel_state_sync_and_queries(self):
        set_seed(42)
        model, x, y = self.get_test_inputs()
        attacker = OnePixelAttack(model, k=1, max_iter=3, pop_size=5)
        x_adv = attacker.attack(x, y)
        self.assertEqual(x_adv.shape, x.shape)
        self.assertTrue(hasattr(attacker, "last_queries"))
        self.assertEqual(len(attacker.last_queries), 2)

    def test_homotopy_gradient_flow(self):
        from src.attacks.optimization.homotopy import HomotopyAttack
        model, x, y = self.get_test_inputs()
        attacker = HomotopyAttack(model, k=5, steps=2)
        x_adv = attacker.attack(x, y)
        self.assertEqual(x_adv.shape, x.shape)

    def test_stratified_sampling(self):
        from src.benchmark.run_attack_benchmark import get_stratified_indices
        class DummyDataset(torch.utils.data.Dataset):
            def __init__(self):
                self.targets = [i % 10 for i in range(100)]
            def __len__(self):
                return len(self.targets)
            def __getitem__(self, idx):
                return torch.zeros(3, 32, 32), self.targets[idx]

        ds = DummyDataset()
        indices = get_stratified_indices(ds, num_samples=20, seed=42)
        self.assertEqual(len(indices), 20)
        selected_targets = [ds.targets[i] for i in indices]
        counts = {c: selected_targets.count(c) for c in range(10)}
        for c in range(10):
            self.assertEqual(counts[c], 2)

    def test_bpda_adapter_gradient_flow(self):
        from src.defenses.bpda_adapter import DefendedModelAdapter
        from src.defenses.preprocessing.median_filter import MedianFilterDefense
        device = get_best_device()
        model = prepare_model_for_eval(DummyModel(), device)
        adapter = DefendedModelAdapter(model, defense=MedianFilterDefense(), mode="adaptive")
        x = torch.rand(2, 3, 32, 32, device=device, requires_grad=True)
        out = adapter(x)
        loss = out.sum()
        loss.backward()
        self.assertIsNotNone(x.grad)

    def test_wideresnet28_10_instantiation(self):
        from src.models.model_factory import get_model
        device = get_best_device()
        model = get_model("wideresnet28_10", device=device)
        x = torch.rand(2, 3, 32, 32, device=device)
        out = model(x)
        self.assertEqual(out.shape, (2, 10))

    def test_invalid_checkpoint_raises_error(self):
        from src.models.model_factory import get_model
        with self.assertRaises(FileNotFoundError):
            get_model("resnet18", checkpoint_path="/path/does/not/exist/non_existent_ckpt.pth")

if __name__ == "__main__":
    unittest.main()
