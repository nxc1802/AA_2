import torch
import torch.nn as nn
from src.core.projections import exact_spatial_topk_mask
from src.core.utils import prepare_model_for_eval, get_best_device

class SigmaZeroAttack:
    """
    SigmaZero Attack (ICLR 2025).
    Adaptive L0 optimization with EMA gradient momentum and exact top-K spatial projection.
    """
    def __init__(self, model: nn.Module, k: int = 15, steps: int = 25, alpha: float = 4/255.0, device: torch.device = None):
        self.model = prepare_model_for_eval(model, device)
        self.k = k
        self.steps = steps
        self.alpha = alpha
        self.device = device if device is not None else get_best_device()
        self.criterion = nn.CrossEntropyLoss(reduction='none')

    def attack(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        y = y.to(self.device)
        B, C, H, W = x.shape
        x_adv = x.clone().detach()
        accum_grad = torch.zeros_like(x)

        steps_to_fool = torch.full((B,), self.steps, dtype=torch.float, device=self.device)
        
        with torch.no_grad():
            preds = self.model(x_adv).argmax(dim=1)
        fooled_mask = (preds != y)
        steps_to_fool[fooled_mask] = 0.0

        for step in range(self.steps):
            if fooled_mask.all():
                break

            x_adv.requires_grad = True
            out = self.model(x_adv)
            loss_vec = self.criterion(out, y)
            loss = loss_vec.sum()

            self.model.zero_grad()
            loss.backward()

            grad = x_adv.grad.data
            accum_grad = 0.9 * accum_grad + grad
            score = accum_grad.abs().sum(dim=1, keepdim=True) # (B, 1, H, W)

            mask = exact_spatial_topk_mask(score, self.k).float()
            active_mask = (~fooled_mask).view(B, 1, 1, 1).float()
            
            step_update = self.alpha * accum_grad.sign() * mask * active_mask
            x_adv = torch.clamp(x_adv + step_update, 0.0, 1.0).detach()

            with torch.no_grad():
                preds = self.model(x_adv).argmax(dim=1)
                current_fooled = (preds != y)
                newly_fooled = current_fooled & (~fooled_mask)
                steps_to_fool[newly_fooled] = step + 1
                fooled_mask = fooled_mask | newly_fooled

        self.last_steps = steps_to_fool.cpu().numpy().tolist()
        return x_adv
