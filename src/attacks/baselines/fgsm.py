import torch
import torch.nn as nn
from src.core.types import AttackResult
from src.core.projections import compute_spatial_l0
from src.core.utils import prepare_model_for_eval

class FGSMAttack:
    def __init__(self, model: nn.Module, eps: float = 8/255.0, device: torch.device = None):
        self.model = prepare_model_for_eval(model, device)
        self.eps = eps
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.criterion = nn.CrossEntropyLoss(reduction='none')

    def attack(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        y = y.to(self.device)
        B = x.size(0)

        x_adv = x.clone().detach().requires_grad_(True)
        out = self.model(x_adv)
        loss = self.criterion(out, y).sum()
        
        self.model.zero_grad()
        loss.backward()
        
        grad_sign = x_adv.grad.sign()
        x_adv = torch.clamp(x + self.eps * grad_sign, 0.0, 1.0).detach()
        
        self.last_steps = [1] * B
        return x_adv
