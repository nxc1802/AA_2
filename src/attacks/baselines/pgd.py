import torch
import torch.nn as nn
from src.core.utils import prepare_model_for_eval

class PGDAttack:
    def __init__(self, model: nn.Module, eps: float = 8/255.0, alpha: float = 2/255.0, steps: int = 20, device: torch.device = None):
        self.model = prepare_model_for_eval(model, device)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.criterion = nn.CrossEntropyLoss(reduction='none')

    def attack(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        y = y.to(self.device)
        B = x.size(0)

        # Uniform random start within L_inf ball
        x_adv = x.clone().detach() + torch.FloatTensor(*x.shape).uniform_(-self.eps, self.eps).to(self.device)
        x_adv = torch.clamp(x_adv, 0.0, 1.0).detach()

        best_adv = x_adv.clone()
        with torch.no_grad():
            out_init = self.model(x_adv)
            best_loss = self.criterion(out_init, y)

        steps_to_fool = torch.full((B,), self.steps, dtype=torch.float, device=self.device)
        fooled_mask = (out_init.argmax(dim=1) != y)
        steps_to_fool[fooled_mask] = 0.0

        for step in range(self.steps):
            x_adv.requires_grad = True
            out = self.model(x_adv)
            loss_vec = self.criterion(out, y)
            loss = loss_vec.sum()

            self.model.zero_grad()
            loss.backward()

            step_adv = x_adv + self.alpha * x_adv.grad.sign()
            eta = torch.clamp(step_adv - x, -self.eps, self.eps)
            x_adv = torch.clamp(x + eta, 0.0, 1.0).detach()

            with torch.no_grad():
                out_step = self.model(x_adv)
                curr_loss = self.criterion(out_step, y)
                preds = out_step.argmax(dim=1)

                improved = curr_loss > best_loss
                best_loss[improved] = curr_loss[improved]
                best_adv[improved] = x_adv[improved]

                current_fooled = (preds != y)
                newly_fooled = current_fooled & (~fooled_mask)
                steps_to_fool[newly_fooled] = step + 1
                fooled_mask = fooled_mask | newly_fooled

        self.last_steps = steps_to_fool.cpu().numpy().tolist()
        return best_adv
