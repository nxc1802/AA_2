import torch
import torch.nn as nn
import torch.nn.functional as F

class SparseRSAttack:
    """Vectorized Sparse Random Search Attack (Sparse-RS)."""
    def __init__(self, model, k=15, n_pixels=None, steps=25, n_queries=None, alpha=4/255.0, device=None):
        self.model = model
        self.k = n_pixels if n_pixels is not None else k
        self.steps = n_queries if n_queries is not None else steps
        self.alpha = alpha
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.criterion = nn.CrossEntropyLoss()

    def attack(self, x, y):
        x = x.to(self.device)
        y = y.to(self.device)
        B, C, H, W = x.shape
        x_adv = x.clone().detach()

        steps_to_fool = torch.full((B,), self.steps, dtype=torch.float, device=self.device)
        fooled_mask = torch.zeros(B, dtype=torch.bool, device=self.device)

        with torch.no_grad():
            best_losses = F.cross_entropy(self.model(x_adv), y, reduction='none')
            preds = self.model(x_adv).argmax(dim=1)
        fooled_mask = (preds != y)
        steps_to_fool[fooled_mask] = 0.0

        for step in range(self.steps):
            if fooled_mask.all():
                break

            cand = x_adv.clone()
            coords = torch.randint(0, H * W, (B, self.k), device=self.device)
            signs = (torch.randint(0, 2, (B, self.k, C), device=self.device).float() * 2.0 - 1.0) * self.alpha

            b_idx = torch.arange(B, device=self.device).unsqueeze(1).expand(-1, self.k)
            c_y = coords // W
            c_x = coords % W

            for c_ch in range(C):
                cand[b_idx, c_ch, c_y, c_x] = torch.clamp(cand[b_idx, c_ch, c_y, c_x] + signs[:, :, c_ch], 0.0, 1.0)

            active_mask = (~fooled_mask).view(B, 1, 1, 1)
            cand = torch.where(active_mask, cand, x_adv)

            with torch.no_grad():
                l_cand = F.cross_entropy(self.model(cand), y, reduction='none')
                improved = (l_cand > best_losses) & (~fooled_mask)
                x_adv = torch.where(improved.unsqueeze(1).unsqueeze(2).unsqueeze(3), cand, x_adv)
                best_losses = torch.where(improved, l_cand, best_losses)

            with torch.no_grad():
                preds = self.model(x_adv).argmax(dim=1)
            current_fooled = (preds != y)
            newly_fooled = current_fooled & (~fooled_mask)
            steps_to_fool[newly_fooled] = step + 1
            fooled_mask = fooled_mask | newly_fooled

        self.last_steps = steps_to_fool.cpu().numpy().tolist()
        return x_adv
