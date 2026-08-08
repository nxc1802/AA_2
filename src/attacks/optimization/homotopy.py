import torch
import torch.nn as nn
from src.core.projections import project_l0, exact_spatial_topk_mask
from src.core.utils import prepare_model_for_eval

class HomotopyAttack:
    """Homotopy Continuation Sparse Attack with exact spatial L0 projection."""
    def __init__(self, model: nn.Module, k: int = 15, target_sparsity: int = None, steps: int = 25, alpha: float = 4/255.0, device: torch.device = None):
        self.model = prepare_model_for_eval(model, device)
        self.k = target_sparsity if target_sparsity is not None else k
        self.steps = steps
        self.alpha = alpha
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.criterion = nn.CrossEntropyLoss(reduction='none')

    def attack(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        y = y.to(self.device)
        B, C, H, W = x.shape

        delta = torch.zeros_like(x)
        best_adv = x.clone()

        with torch.no_grad():
            out_init = self.model(x)
            best_loss = self.criterion(out_init, y)

        steps_to_fool = torch.full((B,), self.steps, dtype=torch.float, device=self.device)
        fooled_mask = (out_init.argmax(dim=1) != y)
        steps_to_fool[fooled_mask] = 0.0

        for step in range(self.steps):
            gamma = (step + 1) / float(self.steps)
            x_adv = (x + delta).clamp(0.0, 1.0).requires_grad_(True)
            out = self.model(x_adv)
            loss_cls = self.criterion(out, y).sum()

            perturbation = x_adv - x
            loss_l0_proxy = torch.sum(perturbation.abs() / (perturbation.abs() + 1e-3))
            loss = loss_cls - gamma * 0.01 * loss_l0_proxy

            self.model.zero_grad()
            loss.backward()

            grad = x_adv.grad.data
            score = grad.abs().sum(dim=1, keepdim=True)
            mask = exact_spatial_topk_mask(score, self.k).float()
            
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
