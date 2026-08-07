import torch
import torch.nn as nn
import torch.nn.functional as F
from src.core.utils import prepare_model_for_eval

class CornerSearchAttack:
    """
    CornerSearch Attack (Croce & Hein, ICCV 2019).
    Evaluates extreme pixel perturbations ('corners' at 0.0 or 1.0) while tracking modified pixels.
    """
    def __init__(self, model: nn.Module, k: int = 15, max_pixels: int = None, max_iter: int = 100, device: torch.device = None):
        self.model = prepare_model_for_eval(model, device)
        self.k = max_pixels if max_pixels is not None else k
        self.max_iter = max_iter
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def attack(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        y = y.to(self.device)
        B, C, H, W = x.shape
        x_adv = x.clone().detach()

        max_steps = min(self.k, H * W)
        steps_to_fool = torch.full((B,), max_steps, dtype=torch.float, device=self.device)
        
        with torch.no_grad():
            preds = self.model(x_adv).argmax(dim=1)
        fooled_mask = (preds != y)
        steps_to_fool[fooled_mask] = 0.0

        # Mask of selected pixels per sample to prevent duplicate picks
        selected_pixels = torch.zeros((B, H * W), dtype=torch.bool, device=self.device)

        for step in range(max_steps):
            if fooled_mask.all():
                break

            x_adv.requires_grad = True
            out = self.model(x_adv)
            loss_vec = F.cross_entropy(out, y, reduction='none')
            loss = loss_vec.sum()

            self.model.zero_grad()
            loss.backward()

            grad_mag = x_adv.grad.data.abs().sum(dim=1).view(B, -1) # (B, H*W)
            # Mask out already selected pixels
            grad_mag[selected_pixels] = -1.0

            top_idx = grad_mag.argmax(dim=1) # (B,)
            selected_pixels.scatter_(1, top_idx.unsqueeze(1), True)

            row = top_idx // W
            col = top_idx % W

            with torch.no_grad():
                cand0 = x_adv.clone()
                cand1 = x_adv.clone()
                active = (~fooled_mask)

                for b in range(B):
                    if active[b]:
                        r, c = row[b].item(), col[b].item()
                        cand0[b, :, r, c] = 0.0
                        cand1[b, :, r, c] = 1.0

                out0 = self.model(cand0)
                out1 = self.model(cand1)
                l0 = F.cross_entropy(out0, y, reduction='none')
                l1 = F.cross_entropy(out1, y, reduction='none')

                use_c1 = (l1 > l0).view(B, 1, 1, 1)
                new_adv = torch.where(use_c1, cand1, cand0)
                x_adv = torch.where(active.view(B, 1, 1, 1), new_adv, x_adv).detach()

                preds = self.model(x_adv).argmax(dim=1)
                current_fooled = (preds != y)
                newly_fooled = current_fooled & (~fooled_mask)
                steps_to_fool[newly_fooled] = step + 1
                fooled_mask = fooled_mask | newly_fooled

        self.last_steps = steps_to_fool.cpu().numpy().tolist()
        return x_adv
