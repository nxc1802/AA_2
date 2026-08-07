import torch
import torch.nn as nn

class GradientGuidanceAttack:
    """Authentic Gradient Guidance Sparse Attack."""
    def __init__(self, model, k=15, sparsity_budget=None, steps=25, alpha=4/255.0, device=None):
        self.model = model
        self.k = sparsity_budget if sparsity_budget is not None else k
        self.steps = steps
        self.alpha = alpha
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

            margin_loss = (out[torch.arange(B), target_cls] - out[torch.arange(B), y]).sum()
            self.model.zero_grad()
            margin_loss.backward()

            grad = x_adv.grad.data
            grad_score = grad.abs().sum(dim=1)

            flat_score = grad_score.view(B, -1)
            topk_vals, _ = torch.topk(flat_score, k=min(self.k, H*W), dim=1)
            thresh = topk_vals[:, -1].view(B, 1, 1)

            mask = (grad_score >= thresh).unsqueeze(1).float()
            
            active_mask = (~fooled_mask).view(B, 1, 1, 1).float()
            step_update = self.alpha * grad.sign() * mask * active_mask
            x_adv = torch.clamp(x_adv + step_update, 0.0, 1.0).detach()
            from src.core.projections import project_l0
            x_adv = (x + project_l0(x_adv - x, self.k)).clamp(0.0, 1.0).detach()

            with torch.no_grad():
                preds = self.model(x_adv).argmax(dim=1)
            current_fooled = (preds != y)
            newly_fooled = current_fooled & (~fooled_mask)
            steps_to_fool[newly_fooled] = step + 1
            fooled_mask = fooled_mask | newly_fooled

        self.last_steps = steps_to_fool.cpu().numpy().tolist()
        return x_adv
