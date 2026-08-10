import os
import shutil
import pytest
import torch
import torch.nn as nn
from aa.cache import AttackArtifactCache
from aa.scheduler import MultiGPUScheduler
from aa.attacks.casa import CoalitionSparseAttack
from aa.attacks.base import AttackOutput
from aa.data import get_sample_batch_indices


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 10, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        return self.pool(self.conv(x)).squeeze(-1).squeeze(-1)


def test_artifact_cache_roundtrip():
    test_cache_dir = "result/test_attack_cache"
    if os.path.exists(test_cache_dir):
        shutil.rmtree(test_cache_dir)

    cache = AttackArtifactCache(cache_dir=test_cache_dir)
    cache_key = cache.compute_cache_key(
        dataset_hash="hash123",
        model_identifier="resnet18_sha256",
        attack_name="pgd",
        attack_kwargs={"steps": 20, "eps": 0.03137},
        seed=42,
        k=16
    )

    assert not cache.has(cache_key)

    dummy_x = torch.rand(4, 3, 32, 32)
    output = AttackOutput(
        x_adv=dummy_x,
        queries=100,
        forward_evals=20,
        backward_evals=20,
        metadata={"k": 16}
    )

    cache.put(cache_key, output)
    assert cache.has(cache_key)

    loaded = cache.get(cache_key)
    assert loaded is not None
    assert torch.equal(loaded.x_adv, dummy_x)
    assert loaded.queries == 100
    assert loaded.forward_evals == 20
    assert loaded.backward_evals == 20
    assert loaded.metadata["k"] == 16

    if os.path.exists(test_cache_dir):
        shutil.rmtree(test_cache_dir)


def test_multi_gpu_scheduler_fallback():
    scheduler = MultiGPUScheduler()
    assert isinstance(scheduler.num_gpus, int)
    if scheduler.num_gpus <= 1:
        assert not scheduler.is_multi_gpu()


def test_batched_casa_drop_and_repair():
    model = DummyModel()
    model.eval()

    torch.manual_seed(42)
    x = torch.rand(4, 3, 32, 32)
    y = torch.tensor([0, 1, 2, 3])

    attack = CoalitionSparseAttack(
        model=model,
        k=16,
        steps=5,
        inner_steps=2,
        repair_steps=2,
        drop_and_repair=True
    )
    output = attack.attack(x, y)

    # 1. Bounds check [0, 1]
    assert (output.x_adv >= 0.0).all()
    assert (output.x_adv <= 1.0).all()

    # 2. L0 constraint check <= K
    diff = output.x_adv - x
    l0 = (diff.abs().max(dim=1).values > 1e-5).sum(dim=(1, 2))
    assert (l0 <= 16).all()


def test_data_loader_custom_batch_size():
    loader, indices, h = get_sample_batch_indices(
        dataset_name="cifar10",
        batch_size=512,
        num_samples=20,
        seed=42,
        num_workers=0
    )
    assert loader.batch_size == 512
    assert len(indices) == 20
