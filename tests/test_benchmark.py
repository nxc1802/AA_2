import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from aa.utils import get_best_device, prepare_model_for_eval
from aa.attacks import create_attack
from aa.benchmark import evaluate_attack


class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 10, kernel_size=3, padding=1)

    def forward(self, x):
        return self.conv(x).mean(dim=[2, 3])


def test_evaluate_attack():
    device = get_best_device()
    model = prepare_model_for_eval(LinearModel(), device=device)
    x = torch.rand(4, 3, 32, 32, device=device)
    with torch.no_grad():
        preds = model(x).argmax(dim=1)

    y = preds.clone()
    y[2] = (y[2] + 1) % 10
    y[3] = (y[3] + 1) % 10

    ds = TensorDataset(x.cpu(), y.cpu())
    loader = DataLoader(ds, batch_size=2)

    attack = create_attack("fgsm", model=model, eps=0.01)
    res = evaluate_attack(model, attack, loader, device=device)

    assert res["total_samples"] == 4
    assert res["clean_correct_count"] == 2
    assert res["clean_accuracy"] == 50.0
