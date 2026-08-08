import torch
import torch.nn as nn
from src.core.projections import exact_spatial_topk_mask
from src.core.utils import prepare_model_for_eval, get_best_device

class SparsePGDAttack:
    """
    Sparse-PGD (sPGD) Attack (Separated Mask and Magnitude Optimization).
    Ref: 'sPGD: Sparse Projected Gradient Descent for Sparse Adversarial Attacks'
    Perturbation is parameterized as delta = p * m, where p is magnitude and m is sparsity mask.
    """
    def __init__(self, model: nn.Module, sparsity_budget: int = 15, steps: int = 25, alpha: float = 4/255.0, device: torch.device = None):
        self.model = prepare_model_for_eval(model, device)
        self.k = sparsity_budget
        self.steps = steps
        self.alpha = alpha
        self.device = device if device is not None else get_best_device()
        self.criterion = nn.CrossEntropyLoss(reduction='none')

    def attack(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        y = y.to(self.device)
        B, C, H, W = x.shape

        p = torch.zeros_like(x)
        m_logits = torch.zeros((B, 1, H, W), device=self.device)

        best_adv = x.clone()
        with torch.no_grad():
            out_init = self.model(x)
            best_loss = self.criterion(out_init, y)

        steps_to_fool = torch.full((B,), self.steps, dtype=torch.float, device=self.device)
        fooled_mask = (out_init.argmax(dim=1) != y)
        steps_to_fool[fooled_mask] = 0.0

        for step in range(self.steps):
            # Compute loss on unmasked perturbation to update spatial mask logits across all pixels
            p = p.detach().requires_grad_(True)
            x_full = (x + p).clamp(0.0, 1.0)
            out_full = self.model(x_full)
            loss_full = self.criterion(out_full, y).sum()

            self.model.zero_grad()
            loss_full.backward()

            with torch.no_grad():
                grad_p = p.grad.data if p.grad is not None else torch.zeros_like(p)
                p = (p + self.alpha * grad_p.sign()).clamp(-1.0, 1.0)
                
                # Update mask logits based on gradient magnitude attribution across all pixels
                spatial_grad_mag = grad_p.abs().sum(dim=1, keepdim=True)
                m_logits = m_logits + spatial_grad_mag

            # Apply top-K mask for actual adversarial step evaluation
            mask = exact_spatial_topk_mask(m_logits, self.k).float()
            delta = p * mask
            x_adv = (x + delta).clamp(0.0, 1.0)

            with torch.no_grad():
                out_step = self.model(x_adv)
                curr_loss = self.criterion(out_step, y)
                preds = out_step.argmax(dim=1)

                improved = curr_loss > best_loss
                best_loss[improved] = curr_loss[improved]
                best_adv[improved] = x_adv[improved]

                current_fooled = (preds != y)
                newly_fooled = current_fooled & (~fooled_mask)
                steps_to_fool[newly_fooled] = step + 1
                fooled_mask = fooled_mask | newly_fooled

        self.last_steps = steps_to_fool.cpu().numpy().tolist()
        return best_adv
