import torch
import torch.nn as nn
from src.core.projections import exact_spatial_topk_mask
from src.core.utils import prepare_model_for_eval, get_best_device

class SparseFoolAttack:
    """SparseFool Attack with DeepFool-inspired boundary margin and exact top-K spatial mask."""
    def __init__(self, model: nn.Module, k: int = 15, steps: int = 20, max_iter: int = None, lambda_val: float = 3.0, device: torch.device = None):
        self.model = prepare_model_for_eval(model, device)
        self.k = k
        self.steps = max_iter if max_iter is not None else steps
        self.lambda_val = lambda_val
        self.device = device if device is not None else get_best_device()

    def attack(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        y = y.to(self.device)
        B, C, H, W = x.shape
        x_adv = x.clone().detach()

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
            
            top2 = out.argsort(dim=1, descending=True)[:, :2]
            target_cls = torch.where(top2[:, 0] == y, top2[:, 1], top2[:, 0])

            loss = (out[torch.arange(B), target_cls] - out[torch.arange(B), y]).sum()
            self.model.zero_grad()
            loss.backward()

            grad = x_adv.grad.data
            grad_mag = grad.abs().sum(dim=1, keepdim=True)
            
            sparse_mask = exact_spatial_topk_mask(grad_mag, self.k).float()
            active_mask = (~fooled_mask).view(B, 1, 1, 1).float()
            step_direction = grad.sign() * sparse_mask * active_mask

            with torch.no_grad():
                x_adv = torch.clamp(x_adv + (1.0 / 255.0) * self.lambda_val * step_direction, 0.0, 1.0).detach()

                preds = self.model(x_adv).argmax(dim=1)
                current_fooled = (preds != y)
                newly_fooled = current_fooled & (~fooled_mask)
                steps_to_fool[newly_fooled] = step + 1
                fooled_mask = fooled_mask | newly_fooled

        self.last_steps = steps_to_fool.cpu().numpy().tolist()
        return x_adv
