import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from aa.training import Trainer, CheckpointManager, TrainingHistory, create_optimizer, create_scheduler


class TinyModel(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.fc = nn.Linear(10, num_classes)

    def forward(self, x):
        return self.fc(x)


@pytest.fixture
def synthetic_data():
    x = torch.randn(32, 10)
    y = torch.randint(0, 2, (32,))
    ds = TensorDataset(x, y)
    loader = DataLoader(ds, batch_size=8, shuffle=False)
    return loader


def test_trainer_fit(synthetic_data, tmp_path):
    model = TinyModel()
    initial_weight = model.fc.weight.clone()

    opt_config = {"name": "sgd", "lr": 0.1, "momentum": 0.0, "weight_decay": 0.0}
    sched_config = {"name": "cosine", "min_lr": 0.01}
    optimizer = create_optimizer(model, opt_config)
    scheduler = create_scheduler(optimizer, sched_config, total_epochs=2)

    ckpt_mgr = CheckpointManager(
        checkpoint_dir=str(tmp_path),
        experiment_name="tiny_exp",
        dataset_name="synthetic",
        architecture="tinymodel",
        seed=42
    )

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=torch.device("cpu"),
        checkpoint_manager=ckpt_mgr
    )

    history = trainer.fit(
        train_loader=synthetic_data,
        val_loader=synthetic_data,
        epochs=2,
        history=TrainingHistory()
    )

    assert len(history.records) == 2
    assert "train_loss" in history.records[0]
    assert "train_acc" in history.records[0]
    assert history.records[0]["train_loss"] >= 0.0

    # Verify weights changed
    assert not torch.equal(model.fc.weight, initial_weight)

    # Verify checkpoints created
    assert (tmp_path / "tiny_exp" / "last.pth").is_file()
    assert (tmp_path / "tiny_exp" / "best.pth").is_file()
