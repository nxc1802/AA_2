import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from aa.training.adversarial import PGDTrainAttack, SparsePGDTrainAttack, AdversarialTrainer


class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 3, padding=1)
        self.fc = nn.Linear(4 * 8 * 8, 2)

    def forward(self, x):
        h = torch.relu(self.conv(x))
        h = h.view(h.size(0), -1)
        return self.fc(h)


def test_pgd_train_attack_constraints():
    model = ConvNet()
    criterion = nn.CrossEntropyLoss()
    attack = PGDTrainAttack(eps=8.0/255.0, alpha=2.0/255.0, steps=5)

    x = torch.rand(4, 3, 8, 8)
    y = torch.tensor([0, 1, 0, 1])

    x_adv = attack.generate(model, x, y, criterion)

    assert x_adv.shape == x.shape
    assert x_adv.min() >= 0.0 and x_adv.max() <= 1.0
    diff = (x_adv - x).abs()
    assert diff.max() <= (8.0 / 255.0 + 1e-5)


def test_sparse_pgd_train_attack_constraints():
    model = ConvNet()
    criterion = nn.CrossEntropyLoss()
    k = 4
    attack = SparsePGDTrainAttack(k=k, step_size=0.1, steps=5)

    x = torch.rand(4, 3, 8, 8)
    y = torch.tensor([0, 1, 0, 1])

    x_adv = attack.generate(model, x, y, criterion)

    assert x_adv.shape == x.shape
    assert x_adv.min() >= 0.0 and x_adv.max() <= 1.0

    # Verify per-sample modified pixels count is <= k
    diff = (x_adv - x).abs().sum(dim=1)  # (batch, H, W)
    changed_pixels = (diff > 1e-4).sum(dim=(1, 2))
    assert (changed_pixels <= k).all()


def test_adversarial_trainer_step(tmp_path):
    model = ConvNet()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    attack = PGDTrainAttack(eps=8.0/255.0, alpha=2.0/255.0, steps=2)

    x = torch.rand(16, 3, 8, 8)
    y = torch.randint(0, 2, (16,))
    loader = DataLoader(TensorDataset(x, y), batch_size=8)

    trainer = AdversarialTrainer(
        model=model,
        optimizer=optimizer,
        attack_generator=attack,
        clean_weight=0.5,
        adversarial_weight=0.5,
        criterion=criterion,
        device=torch.device("cpu")
    )

    metrics = trainer.train_epoch(loader, epoch=1)

    assert "loss" in metrics
    assert "accuracy" in metrics
    assert "adv_accuracy" in metrics
    assert metrics["loss"] > 0.0
