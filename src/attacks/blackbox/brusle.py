import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from src.core.utils import prepare_model_for_eval

class BruSLeAttack:
    """
    Patch-based Random Search Sparse Attack (BruSLe).
    Guarantees strict spatial L0 <= K by maintaining patch perturbation directly from clean x.
    """
    def __init__(self, model: nn.Module, k: int = 15, block_size: int = None, steps: int = 25, n_queries: int = None, alpha: float = 4/255.0, device: torch.device = None):
        self.model = prepare_model_for_eval(model, device)
        self.k = (block_size * block_size) if block_size is not None else k
        self.steps = n_queries if n_queries is not None else steps
        self.alpha = alpha
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def attack(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        y = y.to(self.device)
        B, C, H, W = x.shape
        x_adv = x.clone().detach()
        patch_dim = max(1, int(math.sqrt(self.k)))

        steps_to_fool = torch.full((B,), self.steps, dtype=torch.float, device=self.device)
        
        with torch.no_grad():
            out_init = self.model(x)
            best_losses = F.cross_entropy(out_init, y, reduction='none')
            preds = out_init.argmax(dim=1)
        fooled_mask = (preds != y)
        steps_to_fool[fooled_mask] = 0.0

        for step in range(self.steps):
            if fooled_mask.all():
                break

            cand_delta = torch.zeros_like(x)
            top_y = torch.randint(0, H - patch_dim + 1, (B,), device=self.device)
            top_x = torch.randint(0, W - patch_dim + 1, (B,), device=self.device)
            noise = (torch.randint(0, 2, (B, C, patch_dim, patch_dim), device=self.device).float() * 2.0 - 1.0) * self.alpha

            active = (~fooled_mask)
            for b in range(B):
                if active[b]:
                    ty, tx = top_y[b].item(), top_x[b].item()
                    cand_delta[b, :, ty:ty+patch_dim, tx:tx+patch_dim] = noise[b]

            cand = (x + cand_delta).clamp(0.0, 1.0)

            with torch.no_grad():
                l_cand = F.cross_entropy(self.model(cand), y, reduction='none')
                improved = (l_cand > best_losses) & (~fooled_mask)
                x_adv = torch.where(improved.view(B, 1, 1, 1), cand, x_adv)
                best_losses = torch.where(improved, l_cand, best_losses)

                preds = self.model(x_adv).argmax(dim=1)
                current_fooled = (preds != y)
                newly_fooled = current_fooled & (~fooled_mask)
                steps_to_fool[newly_fooled] = step + 1
                fooled_mask = fooled_mask | newly_fooled

        self.last_steps = steps_to_fool.cpu().numpy().tolist()
        return x_adv
