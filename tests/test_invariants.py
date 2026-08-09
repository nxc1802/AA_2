"""
Invariant tests for AA_2 benchmark correctness (upgrade.md section 32).

Tested invariants:
    1. ASR + CRA == 100%  (conditional_robust_accuracy + asr == 100)
    2. CRA denominator == clean_correct_count  (not total_samples)
    3. ASR@K monotonic non-decreasing in K
    4. L0(x_adv - x) <= K for budgeted attacks
    5. x_adv in [0, 1]
    6. Feature attack works through DefendedModelAdapter
    7. FCSA synergy > 0 (no longer degenerate)
    8. BatchMetrics has adv_correct and lpips fields
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from aa.attacks import create_attack, ATTACK_REGISTRY
from aa.attacks.proposed import SparseFeatureAttack, FeatureExtractorAdapter
from aa.benchmark import evaluate_attack, derive_minimal_asr_curve
from aa.defenses import DefendedModelAdapter, GaussianBlurDefense
from aa.metrics import compute_spatial_l0, BatchMetrics
from aa.utils import get_best_device, prepare_model_for_eval


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

class SmallResNetLike(nn.Module):
    """Tiny model mimicking ResNet-like structure for feature hook testing."""
    def __init__(self):
        super().__init__()
        self.layer4 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(16, 10)

    def forward(self, x):
        x = self.layer4(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


def _make_loader(n=10, device="cpu"):
    model = SmallResNetLike().to(device).eval()
    x = torch.rand(n, 3, 32, 32, device=device)
    with torch.no_grad():
        preds = model(x).argmax(dim=1)
    y = preds.clone()
    y[: n // 2] = (y[: n // 2] + 1) % 10
    return model, x.cpu(), y.cpu()


# ---------------------------------------------------------------------------
# Invariant 1 & 2: ASR + CRA == 100%, CRA denominator
# ---------------------------------------------------------------------------

def test_asr_plus_cra_equals_100():
    """ASR + conditional_robust_accuracy must equal 100%."""
    device = get_best_device()
    model, x, y = _make_loader(n=10, device=device)
    model = prepare_model_for_eval(model, device=device)
    ds = TensorDataset(x, y)
    loader = DataLoader(ds, batch_size=5)
    attack = create_attack("fgsm", model=model, eps=0.1)
    res = evaluate_attack(model, attack, loader, device=device)
    asr = res["asr"]
    cra = res["conditional_robust_accuracy"]
    assert abs(asr + cra - 100.0) < 1e-4, (
        f"ASR ({asr:.4f}) + CRA ({cra:.4f}) = {asr + cra:.4f} != 100"
    )


def test_cra_denominator_is_clean_correct():
    """CRA must equal (clean_correct - success) / clean_correct, not / total."""
    device = get_best_device()
    model, x, y = _make_loader(n=10, device=device)
    model = prepare_model_for_eval(model, device=device)
    ds = TensorDataset(x, y)
    loader = DataLoader(ds, batch_size=10)
    attack = create_attack("fgsm", model=model, eps=0.3)
    res = evaluate_attack(model, attack, loader, device=device)
    cc = res["clean_correct_count"]
    sc = res["success_count"]
    if cc > 0:
        expected_cra = 100.0 * (cc - sc) / cc
        actual_cra = res["conditional_robust_accuracy"]
        assert abs(actual_cra - expected_cra) < 1e-4, (
            f"CRA={actual_cra:.4f} but expected {expected_cra:.4f}"
        )


# ---------------------------------------------------------------------------
# Invariant 3: ASR@K monotonic
# ---------------------------------------------------------------------------

def test_asr_at_k_monotonic():
    """ASR@K must be non-decreasing as K increases."""
    device = get_best_device()
    model, x, y = _make_loader(n=10, device=device)
    model = prepare_model_for_eval(model, device=device)
    ds = TensorDataset(x, y)
    loader = DataLoader(ds, batch_size=10)
    attack = create_attack("ours", model=model, k=64, steps=5)
    res = evaluate_attack(model, attack, loader, device=device)
    k_values = [1, 2, 4, 8, 16, 32, 64]
    curves = derive_minimal_asr_curve(res, k_values)
    prev_asr = -1.0
    for k in k_values:
        k_asr = curves[f"k_{k}"]["asr"]
        assert k_asr >= prev_asr - 1e-6, (
            f"ASR@K not monotonic: ASR@{k}={k_asr:.4f} < prev={prev_asr:.4f}"
        )
        prev_asr = k_asr


# ---------------------------------------------------------------------------
# Invariant 4 & 5: L0 budget + pixel range
# ---------------------------------------------------------------------------

def test_l0_budget_and_pixel_range():
    """Budgeted attacks must respect L0 <= K and keep x_adv in [0, 1]."""
    device = get_best_device()
    model, x, y = _make_loader(n=6, device=device)
    model = prepare_model_for_eval(model, device=device)
    x_dev = x.to(device)
    y_dev = y.to(device)
    for name in ["ours"]:
        k = 8
        attack = create_attack(name, model=model, k=k, steps=5)
        spec = ATTACK_REGISTRY[name]
        output = attack.attack(x_dev, y_dev)
        assert (output.x_adv >= 0.0).all() and (output.x_adv <= 1.0).all(), (
            f"Attack '{name}': x_adv out of [0,1] range"
        )
        if spec.mode == "budget":
            l0 = compute_spatial_l0(output.x_adv - x_dev)
            assert (l0 <= k).all(), (
                f"Attack '{name}': L0 {l0.tolist()} exceeded budget k={k}"
            )


# ---------------------------------------------------------------------------
# Invariant 6: Feature attack works through DefendedModelAdapter
# ---------------------------------------------------------------------------

def test_feature_attack_through_defended_wrapper():
    """SparseFeatureAttack with feature_guidance=True must work on DefendedModelAdapter."""
    device = get_best_device()
    base_model = SmallResNetLike().to(device).eval()
    defense = GaussianBlurDefense(kernel_size=3, sigma=1.0)
    defended = DefendedModelAdapter(base_model, defense=defense, mode="adaptive")
    x = torch.rand(2, 3, 32, 32, device=device)
    y = torch.zeros(2, dtype=torch.long, device=device)
    attack = SparseFeatureAttack(
        model=defended,
        k=4,
        steps=3,
        feature_guidance=True,
    )
    output = attack.attack(x, y)
    assert output.x_adv.shape == x.shape
    assert (output.x_adv >= 0.0).all() and (output.x_adv <= 1.0).all()


# ---------------------------------------------------------------------------
# Invariant 7: FCSA synergy is non-trivially > 0
# ---------------------------------------------------------------------------

def test_fcsa_synergy_nonzero():
    """FCSA interaction score must exceed plain indiv_contrib on non-uniform gradients."""
    device = get_best_device()
    model = SmallResNetLike().to(device).eval()
    attack = SparseFeatureAttack(
        model=model, k=8, steps=1,
        feature_guidance=False, interaction=True, interaction_mode="fcsa",
    )
    grad = torch.zeros(1, 3, 32, 32, device=device)
    grad[0, :, 0, 0] = 1.0    # strong pixel
    grad[0, :, 1, 1] = 0.01   # weak neighbour
    score = attack._compute_spatial_interaction(grad)
    grad_mag_00 = grad[0].abs().sum(0)[0, 0]
    grad_max_00 = grad[0].abs().max(0).values[0, 0]
    indiv_00 = (grad_mag_00 * grad_max_00).item()
    score_00 = score[0, 0, 0, 0].item()
    assert score_00 > indiv_00, (
        f"FCSA synergy appears zero: score={score_00:.6f} <= indiv={indiv_00:.6f}"
    )


# ---------------------------------------------------------------------------
# Invariant 8: BatchMetrics has adv_correct and lpips fields
# ---------------------------------------------------------------------------

def test_batch_metrics_has_all_fields():
    """BatchMetrics NamedTuple must contain adv_correct and lpips fields."""
    fields = BatchMetrics._fields
    assert "adv_correct" in fields, "BatchMetrics missing 'adv_correct' field"
    assert "lpips" in fields, "BatchMetrics missing 'lpips' field"
