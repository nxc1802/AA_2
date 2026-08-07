import torch
import torch.nn as nn
import torch.nn.functional as F
from src.core.utils import prepare_model_for_eval

class SparseRSAttack:
    """
    Sparse Random Search Attack (Sparse-RS).
    Guarantees strict spatial L0 <= K at every iteration by maintaining fixed-size spatial perturbation set.
    """
    def __init__(self, model: nn.Module, k: int = 15, n_pixels: int = None, steps: int = 25, n_queries: int = None, alpha: float = 4/255.0, device: torch.device = None):
        self.model = prepare_model_for_eval(model, device)
        self.k = n_pixels if n_pixels is not None else k
        self.steps = n_queries if n_queries is not None else steps
        self.alpha = alpha
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def attack(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        y = y.to(self.device)
        B, C, H, W = x.shape
        x_adv = x.clone().detach()

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

            # Always build candidate perturbation strictly from original clean image x
            cand_delta = torch.zeros_like(x)
            coords = torch.randint(0, H * W, (B, self.k), device=self.device)
            signs = (torch.randint(0, 2, (B, self.k, C), device=self.device).float() * 2.0 - 1.0) * self.alpha

            b_idx = torch.arange(B, device=self.device).unsqueeze(1).expand(-1, self.k)
            c_y = coords // W
            c_x = coords % W

            for c_ch in range(C):
                cand_delta[b_idx, c_ch, c_y, c_x] = signs[:, :, c_ch]

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
