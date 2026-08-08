import torch
import torch.nn as nn
from src.core.projections import project_l0, exact_spatial_topk_mask
from src.core.utils import prepare_model_for_eval, get_best_device

class SAIFAttack:
    """Sparsity-Aware Iterative Fast Attack (SAIF) with exact spatial L0 projection."""
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

        delta = torch.zeros_like(x)
        cum_grad_mag = torch.zeros((B, 1, H, W), device=self.device)
        best_adv = x.clone()

        with torch.no_grad():
            out_init = self.model(x)
            best_loss = self.criterion(out_init, y)

        steps_to_fool = torch.full((B,), self.steps, dtype=torch.float, device=self.device)
        fooled_mask = (out_init.argmax(dim=1) != y)
        steps_to_fool[fooled_mask] = 0.0

        for step in range(self.steps):
            x_adv = (x + delta).clamp(0.0, 1.0).requires_grad_(True)
            out = self.model(x_adv)
            loss_vec = self.criterion(out, y)
            loss = loss_vec.sum()

            self.model.zero_grad()
            loss.backward()

            grad = x_adv.grad.data
            cum_grad_mag += grad.abs().sum(dim=1, keepdim=True)
            
            mask = exact_spatial_topk_mask(cum_grad_mag, self.k).float()
            candidate_delta = delta + self.alpha * grad.sign() * mask
            
            delta = project_l0(candidate_delta, self.k)
            adv_images_proj = torch.clamp(x + delta, 0.0, 1.0)

            with torch.no_grad():
                out_step = self.model(adv_images_proj)
                curr_loss = self.criterion(out_step, y)
                preds = out_step.argmax(dim=1)

                improved = curr_loss > best_loss
                best_loss[improved] = curr_loss[improved]
                best_adv[improved] = adv_images_proj[improved]

                current_fooled = (preds != y)
                newly_fooled = current_fooled & (~fooled_mask)
                steps_to_fool[newly_fooled] = step + 1
                fooled_mask = fooled_mask | newly_fooled

        self.last_steps = steps_to_fool.cpu().numpy().tolist()
        return best_adv
