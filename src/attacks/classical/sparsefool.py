import torch
import torch.nn as nn

class SparseFoolAttack:
    """Vectorized SparseFool Attack ($L_0$ boundary projection)."""
    def __init__(self, model, k=15, steps=20, max_iter=None, lambda_val=3.0, device=None):
        self.model = model
        self.k = k
        self.steps = max_iter if max_iter is not None else steps
        self.lambda_val = lambda_val
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def attack(self, x, y):
        x = x.to(self.device)
        y = y.to(self.device)
        B, C, H, W = x.shape
        x_adv = x.clone().detach()

        steps_to_fool = torch.full((B,), self.steps, dtype=torch.float, device=self.device)
        fooled_mask = torch.zeros(B, dtype=torch.bool, device=self.device)

        with torch.no_grad():
            preds = self.model(x_adv).argmax(dim=1)
        fooled_mask = (preds != y)
        steps_to_fool[fooled_mask] = 0.0

        for step in range(self.steps):
            if fooled_mask.all():
                break

            x_adv.requires_grad = True
            out = self.model(x_adv)
            
            top2 = out.argsort(dim=1, descending=True)[:, :2]
            target_cls = torch.where(top2[:, 0] == y, top2[:, 1], top2[:, 0])

            loss = (out[torch.arange(B), target_cls] - out[torch.arange(B), y]).sum()
            self.model.zero_grad()
            loss.backward()

            grad = x_adv.grad.data
            grad_mag = grad.abs().sum(dim=1)  # (B, H, W)
            flat_grad = grad_mag.view(B, -1)

            topk_vals, _ = torch.topk(flat_grad, k=min(self.k, H*W), dim=1)
            thresh = topk_vals[:, -1].view(B, 1, 1)

            sparse_mask = (grad_mag >= thresh).unsqueeze(1).float()
            
            active_mask = (~fooled_mask).view(B, 1, 1, 1).float()
            step_direction = grad.sign() * sparse_mask * active_mask

            with torch.no_grad():
                x_adv = torch.clamp(x_adv + (1.0 / 255.0) * self.lambda_val * step_direction, 0.0, 1.0).detach()

            with torch.no_grad():
                preds = self.model(x_adv).argmax(dim=1)
            current_fooled = (preds != y)
            newly_fooled = current_fooled & (~fooled_mask)
            steps_to_fool[newly_fooled] = step + 1
            fooled_mask = fooled_mask | newly_fooled

        self.last_steps = steps_to_fool.cpu().numpy().tolist()
        return x_adv
