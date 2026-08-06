import torch
import torch.nn as nn

class BIMAttack:
    def __init__(self, model, eps=8/255.0, alpha=2/255.0, steps=10, device=None):
        self.model = model
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.criterion = nn.CrossEntropyLoss()

    def attack(self, x, y):
        x = x.to(self.device)
        y = y.to(self.device)
        B = x.size(0)
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
            loss = self.criterion(out, y)
            self.model.zero_grad()
            loss.backward()

            active_mask = (~fooled_mask).view(B, 1, 1, 1).float()
            step_adv = x_adv + self.alpha * x_adv.grad.sign() * active_mask
            eta = torch.clamp(step_adv - x, -self.eps, self.eps)
            x_adv = torch.clamp(x + eta, 0.0, 1.0).detach()

            with torch.no_grad():
                preds = self.model(x_adv).argmax(dim=1)
            current_fooled = (preds != y)
            newly_fooled = current_fooled & (~fooled_mask)
            steps_to_fool[newly_fooled] = step + 1
            fooled_mask = fooled_mask | newly_fooled

        self.last_steps = steps_to_fool.cpu().numpy().tolist()
        return x_adv
