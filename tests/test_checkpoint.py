import pytest
import torch
import torch.nn as nn
from aa.training.checkpoint import CheckpointManager, get_rng_states, set_rng_states


class DummyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(5, 2)

    def forward(self, x):
        return self.fc(x)


def test_checkpoint_save_load_exact_logits(tmp_path):
    torch.manual_seed(42)
    model1 = DummyNet()
    optimizer1 = torch.optim.SGD(model1.parameters(), lr=0.01)

    x = torch.randn(4, 5)
    out1 = model1(x)

    mgr = CheckpointManager(
        checkpoint_dir=str(tmp_path),
        experiment_name="test_ckpt",
        dataset_name="synthetic",
        architecture="dummynet",
        seed=42
    )

    last_path, sha256 = mgr.save(
        model=model1,
        optimizer=optimizer1,
        scheduler=None,
        epoch=5,
        current_metric=85.0,
        is_best=True
    )

    assert (tmp_path / "test_ckpt" / "last.pth").is_file()
    assert (tmp_path / "test_ckpt" / "best.pth").is_file()
    assert isinstance(sha256, str) and len(sha256) == 64

    # Load into fresh model
    model2 = DummyNet()
    optimizer2 = torch.optim.SGD(model2.parameters(), lr=0.01)
    start_epoch = mgr.load_resume(last_path, model2, optimizer2)

    assert start_epoch == 5
    out2 = model2(x)

    assert torch.allclose(out1, out2, atol=1e-6)


def test_rng_state_capture_restore():
    rng1 = get_rng_states()
    val1 = torch.randn(5)

    set_rng_states(rng1)
    val2 = torch.randn(5)

    assert torch.allclose(val1, val2)
