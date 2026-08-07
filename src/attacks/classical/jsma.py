import torch
import torch.nn as nn

class JSMAAttack:
    """Batched High-Performance Jacobian-based Saliency Map Attack (JSMA)."""
    def __init__(self, model, k=15, theta=1.0, max_iter=25, device=None):
        self.model = model
        self.k = k
        self.theta = theta
        self.max_iter = max_iter
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def attack(self, x, y):
        x = x.to(self.device)
        y = y.to(self.device)
        B, C, H, W = x.shape
        x_adv = x.clone().detach()

        with torch.no_grad():
            init_out = self.model(x)
            top2 = init_out.argsort(dim=1, descending=True)[:, :2]
            target_cls = torch.where(top2[:, 0] == y, top2[:, 1], top2[:, 0])

        perturbed_count = torch.zeros(B, dtype=torch.int32, device=self.device)
        mask_perturbed = torch.zeros((B, H, W), dtype=torch.bool, device=self.device)
        
        effective_max_iter = max(self.max_iter, self.k)
        steps_to_fool = torch.full((B,), effective_max_iter, dtype=torch.float, device=self.device)
        fooled_mask = torch.zeros(B, dtype=torch.bool, device=self.device)

        with torch.no_grad():
            preds = self.model(x_adv).argmax(dim=1)
        fooled_mask = (preds != y)
        steps_to_fool[fooled_mask] = 0.0

        for step in range(effective_max_iter):
            with torch.no_grad():
                preds = self.model(x_adv).argmax(dim=1)
            
            current_fooled = (preds != y)
            newly_fooled = current_fooled & (~fooled_mask)
            steps_to_fool[newly_fooled] = step
            fooled_mask = fooled_mask | newly_fooled

            active = (preds == y) & (perturbed_count < self.k)
            if not active.any():
                break

            x_adv.requires_grad = True
            out = self.model(x_adv)
            loss = (out[torch.arange(B), target_cls] - out[torch.arange(B), y]).sum()
            self.model.zero_grad()
            loss.backward()

            grad = x_adv.grad.data.abs().sum(dim=1)  # (B, H, W)
            grad[mask_perturbed] = -1.0
            grad[~active] = -1.0

            flat_grad = grad.view(B, -1)
            best_idx = flat_grad.argmax(dim=1)

            row = best_idx // W
            col = best_idx % W

            x_adv = x_adv.detach()
            for b in range(B):
                if active[b]:
                    r, c = row[b].item(), col[b].item()
                    mask_perturbed[b, r, c] = True
                    perturbed_count[b] += 1
                    x_adv[b, :, r, c] = torch.clamp(x_adv[b, :, r, c] + self.theta, 0.0, 1.0)

        self.last_steps = steps_to_fool.cpu().numpy().tolist()
        return x_adv
