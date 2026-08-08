import torch
import torch.nn as nn
import torch.nn.functional as F
from src.core.utils import prepare_model_for_eval, get_best_device

class PixleAttack:
    """
    Pixle Attack (Modas et al., 2021).
    Rearranges (swaps) pairs of pixel locations within images.
    """
    def __init__(self, model: nn.Module, k: int = 15, n_swaps: int = None, steps: int = 25, max_trials: int = None, alpha: float = 4/255.0, device: torch.device = None):
        self.model = prepare_model_for_eval(model, device)
        self.k = n_swaps if n_swaps is not None else k
        self.steps = max_trials if max_trials is not None else steps
        self.device = device if device is not None else get_best_device()

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

            cand = x.clone() # Swap relative to clean original image x
            # Sample source and target pixel coordinates for swapping
            src_coords = torch.randint(0, H * W, (B, self.k), device=self.device)
            dst_coords = torch.randint(0, H * W, (B, self.k), device=self.device)

            for b in range(B):
                if fooled_mask[b]:
                    continue
                for i in range(self.k):
                    y1, x1 = (src_coords[b, i] // W).item(), (src_coords[b, i] % W).item()
                    y2, x2 = (dst_coords[b, i] // W).item(), (dst_coords[b, i] % W).item()
                    
                    # Swap RGB pixel values between (y1, x1) and (y2, x2)
                    temp = cand[b, :, y1, x1].clone()
                    cand[b, :, y1, x1] = cand[b, :, y2, x2]
                    cand[b, :, y2, x2] = temp

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
