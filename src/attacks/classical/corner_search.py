import torch
import torch.nn as nn
import torch.nn.functional as F

class CornerSearchAttack:
    """Vectorized CornerSearch Attack."""
    def __init__(self, model, k=15, max_pixels=None, max_iter=20, device=None):
        self.model = model
        self.k = max_pixels if max_pixels is not None else k
        self.max_iter = max_iter
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def attack(self, x, y):
        x = x.to(self.device)
        y = y.to(self.device)
        B, C, H, W = x.shape
        x_adv = x.clone().detach()

        max_steps = min(self.k, self.max_iter)
        steps_to_fool = torch.full((B,), max_steps, dtype=torch.float, device=self.device)
        fooled_mask = torch.zeros(B, dtype=torch.bool, device=self.device)

        with torch.no_grad():
            preds = self.model(x_adv).argmax(dim=1)
        fooled_mask = (preds != y)
        steps_to_fool[fooled_mask] = 0.0

        for step in range(max_steps):
            if fooled_mask.all():
                break

            x_adv.requires_grad = True
            loss = nn.CrossEntropyLoss()(self.model(x_adv), y)
            self.model.zero_grad()
            loss.backward()

            grad = x_adv.grad.data.abs().sum(dim=1)  # (B, H, W)
            top_idx = grad.view(B, -1).argmax(dim=1)  # (B,)

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

                l0 = F.cross_entropy(self.model(cand0), y, reduction='none')
                l1 = F.cross_entropy(self.model(cand1), y, reduction='none')

                use_c1 = (l1 > l0).unsqueeze(1).unsqueeze(2).unsqueeze(3)
                new_adv = torch.where(use_c1, cand1, cand0)
                
                x_adv = torch.where(active.view(B, 1, 1, 1), new_adv, x_adv).detach()

            with torch.no_grad():
                preds = self.model(x_adv).argmax(dim=1)
            current_fooled = (preds != y)
            newly_fooled = current_fooled & (~fooled_mask)
            steps_to_fool[newly_fooled] = step + 1
            fooled_mask = fooled_mask | newly_fooled

        self.last_steps = steps_to_fool.cpu().numpy().tolist()
        return x_adv
