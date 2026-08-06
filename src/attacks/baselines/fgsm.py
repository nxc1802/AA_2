import torch
import torch.nn as nn

class FGSMAttack:
    def __init__(self, model, eps=8/255.0, device=None):
        self.model = model
        self.eps = eps
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.criterion = nn.CrossEntropyLoss()

    def attack(self, x, y):
        B = x.size(0)
        x_adv = x.clone().detach().to(self.device).requires_grad_(True)
        out = self.model(x_adv)
        loss = self.criterion(out, y.to(self.device))
        self.model.zero_grad()
        loss.backward()
        self.last_steps = [1] * B
        return torch.clamp(x.to(self.device) + self.eps * x_adv.grad.sign(), 0.0, 1.0).detach()
