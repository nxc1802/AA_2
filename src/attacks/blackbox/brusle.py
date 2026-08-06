import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class BruSLeAttack:
    """Vectorized Patch-based Random Search Sparse Attack (BruSLe)."""
    def __init__(self, model, k=15, block_size=None, steps=25, n_queries=None, alpha=4/255.0, device=None):
        self.model = model
        self.k = (block_size * block_size) if block_size is not None else k
        self.steps = n_queries if n_queries is not None else steps
        self.alpha = alpha
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.criterion = nn.CrossEntropyLoss()

    def attack(self, x, y):
        x = x.to(self.device)
        y = y.to(self.device)
        B, C, H, W = x.shape
        x_adv = x.clone().detach()
        patch_dim = max(1, int(math.sqrt(self.k)))

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
            top_y = torch.randint(0, H - patch_dim + 1, (B,), device=self.device)
            top_x = torch.randint(0, W - patch_dim + 1, (B,), device=self.device)

            noise = (torch.randint(0, 2, (B, C, patch_dim, patch_dim), device=self.device).float() * 2.0 - 1.0) * self.alpha

            active = (~fooled_mask)
            for b in range(B):
                if active[b]:
                    ty, tx = top_y[b].item(), top_x[b].item()
                    cand[b, :, ty:ty+patch_dim, tx:tx+patch_dim] = torch.clamp(
                        cand[b, :, ty:ty+patch_dim, tx:tx+patch_dim] + noise[b], 0.0, 1.0
                    )

            cand = torch.where(active.view(B, 1, 1, 1), cand, x_adv)

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
