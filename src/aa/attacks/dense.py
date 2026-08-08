import torch
import torch.nn as nn
from aa.attacks.base import Attack, AttackOutput


class FGSM(Attack):
    def __init__(self, model: nn.Module, eps: float = 8 / 255):
        self.model = model
        self.eps = eps
        self.criterion = nn.CrossEntropyLoss()

    def attack(self, x: torch.Tensor, y: torch.Tensor) -> AttackOutput:
        x_adv = x.clone().detach().requires_grad_(True)
        outputs = self.model(x_adv)
        loss = self.criterion(outputs, y)
        loss.backward()

        with torch.no_grad():
            grad_sign = x_adv.grad.sign()
            x_adv = torch.clamp(x + self.eps * grad_sign, 0.0, 1.0)

        return AttackOutput(x_adv=x_adv.detach(), forward_evals=1, backward_evals=1)


class BIM(Attack):
    def __init__(self, model: nn.Module, eps: float = 8 / 255, alpha: float = 2 / 255, steps: int = 10):
        self.model = model
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.criterion = nn.CrossEntropyLoss()

    def attack(self, x: torch.Tensor, y: torch.Tensor) -> AttackOutput:
        x_adv = x.clone().detach()

        for _ in range(self.steps):
            x_adv.requires_grad_(True)
            outputs = self.model(x_adv)
            loss = self.criterion(outputs, y)
            loss.backward()

            with torch.no_grad():
                grad_sign = x_adv.grad.sign()
                x_adv = x_adv + self.alpha * grad_sign
                eta = torch.clamp(x_adv - x, min=-self.eps, max=self.eps)
                x_adv = torch.clamp(x + eta, 0.0, 1.0)

        return AttackOutput(
            x_adv=x_adv.detach(),
            forward_evals=self.steps,
            backward_evals=self.steps
        )


class PGD(Attack):
    def __init__(
        self,
        model: nn.Module,
        eps: float = 8 / 255,
        alpha: float = 2 / 255,
        steps: int = 20,
        random_start: bool = True
    ):
        self.model = model
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start
        self.criterion = nn.CrossEntropyLoss()

    def attack(self, x: torch.Tensor, y: torch.Tensor) -> AttackOutput:
        x_adv = x.clone().detach()

        if self.random_start:
            x_adv = x_adv + torch.empty_like(x_adv).uniform_(-self.eps, self.eps)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)

        for _ in range(self.steps):
            x_adv.requires_grad_(True)
            outputs = self.model(x_adv)
            loss = self.criterion(outputs, y)
            loss.backward()

            with torch.no_grad():
                grad_sign = x_adv.grad.sign()
                x_adv = x_adv + self.alpha * grad_sign
                eta = torch.clamp(x_adv - x, min=-self.eps, max=self.eps)
                x_adv = torch.clamp(x + eta, 0.0, 1.0)

        return AttackOutput(
            x_adv=x_adv.detach(),
            forward_evals=self.steps,
            backward_evals=self.steps
        )
